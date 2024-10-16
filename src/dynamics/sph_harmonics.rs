/*
    Nyx, blazing fast astrodynamics
    Copyright (C) 2018-onwards Christopher Rabotin <christopher.rabotin@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use anise::errors::OrientationSnafu;
use anise::prelude::Almanac;
use snafu::ResultExt;

use crate::cosmic::{AstroPhysicsSnafu, Frame, Orbit};
use crate::dynamics::{AccelModel, Pines};
use crate::io::gravity::HarmonicsMem;
use crate::linalg::{Matrix3, Vector3, Vector4, U7};
use hyperdual::linalg::norm;
use hyperdual::{hyperspace_from_vector, Float, OHyperdual};
use std::cmp::min;
use std::fmt;
use std::panic;
use std::sync::{Arc, Mutex};
use std::thread;

use super::{DynamicsAlmanacSnafu, DynamicsAstroSnafu, DynamicsError};

#[derive(Clone)]
pub struct Harmonics {
    compute_frame: Frame,
    stor: HarmonicsMem,
    pines: Arc<Pines>,
}

impl Harmonics {
    /// Create a new Harmonics dynamical model from the provided gravity potential storage instance.
    pub fn from_stor(compute_frame: Frame, stor: HarmonicsMem) -> Arc<Self> {
        let degree = stor.max_degree_n();
        Arc::new(Self {
            compute_frame,
            stor,
            pines: Pines::new(degree),
        })
    }
}

impl fmt::Display for Harmonics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} gravity field {}x{} (order x degree)",
            self.compute_frame,
            self.stor.max_order_m(),
            self.stor.max_degree_n(),
        )
    }
}

impl AccelModel for Harmonics {
    fn eom(&self, osc: &Orbit, almanac: Arc<Almanac>) -> Result<Vector3<f64>, DynamicsError> {
        // Get a reference to Pines
        let pines = Arc::clone(&self.pines);

        // Convert the osculating orbit to the correct frame (needed for multiple harmonic fields)
        let state = almanac
            .transform_to(*osc, self.compute_frame, None)
            .context(DynamicsAlmanacSnafu {
                action: "transforming into gravity field frame",
            })?;

        // Using the GMAT notation, with extra character for ease of highlight
        let r_ = state.rmag_km();
        let s_ = state.radius_km.x / r_;
        let t_ = state.radius_km.y / r_;
        let u_ = state.radius_km.z / r_;
        let max_degree = self.stor.max_degree_n(); // In GMAT, the degree is NN
        let max_order = self.stor.max_order_m(); // In GMAT, the order is MM

        // Create the associated Legendre polynomials. Note that we add three items as per GMAT (this may be useful for the STM)
        let mut a_nm = pines.a_nm.clone();

        // Initialize the diagonal elements (not a function of the input)
        a_nm[(1, 0)] = u_ * 3.0f64.sqrt();
        for n in 1..=max_degree + 1 {
            let nf64 = n as f64;
            // Off diagonal
            a_nm[(n + 1, n)] = (2.0 * nf64 + 3.0).sqrt() * u_ * a_nm[(n, n)];
        }

        for m in 0..=max_order + 1 {
            for n in (m + 2)..=max_degree + 1 {
                let hm_idx = (n, m);
                a_nm[hm_idx] = u_ * pines.b_nm[hm_idx] * a_nm[(n - 1, m)]
                    - pines.c_nm[hm_idx] * a_nm[(n - 2, m)];
            }
        }

        // Generate r_m and i_m
        let mut r_m = Vec::with_capacity(min(max_degree, max_order) + 1);
        let mut i_m = Vec::with_capacity(min(max_degree, max_order) + 1);

        r_m.push(1.0);
        i_m.push(0.0);

        for m in 1..=min(max_degree, max_order) {
            r_m.push(s_ * r_m[m - 1] - t_ * i_m[m - 1]);
            i_m.push(s_ * i_m[m - 1] + t_ * r_m[m - 1]);
        }

        let eq_radius_km = self
            .compute_frame
            .mean_equatorial_radius_km()
            .context(AstroPhysicsSnafu)
            .context(DynamicsAstroSnafu)?;

        let mu_km3_s2 = self
            .compute_frame
            .mu_km3_s2()
            .context(AstroPhysicsSnafu)
            .context(DynamicsAstroSnafu)?;

        let rho = eq_radius_km / r_;
        let mut rho_np1 = mu_km3_s2 / r_ * rho;
        let accel4 = Arc::new(Mutex::new(Vector4::zeros()));
        let thread_accel = Arc::clone(&accel4);
        let stor = self.stor.clone();

        let handle = thread::spawn(move || {
            for n in 1..max_degree {
                let mut sum: Vector4<f64> = Vector4::zeros();
                rho_np1 *= rho;

                for m in 0..=min(n, max_order) {
                    let (c_val, s_val) = stor.cs_nm(n, m);
                    let d_ = (c_val * r_m[m] + s_val * i_m[m]) * 2.0.sqrt();
                    let e_ = if m == 0 {
                        0.0
                    } else {
                        (c_val * r_m[m - 1] + s_val * i_m[m - 1]) * 2.0.sqrt()
                    };
                    let f_ = if m == 0 {
                        0.0
                    } else {
                        (s_val * r_m[m - 1] - c_val * i_m[m - 1]) * 2.0.sqrt()
                    };

                    sum.x += (m as f64) * a_nm[(n, m)] * e_;
                    sum.y += (m as f64) * a_nm[(n, m)] * f_;
                    sum.z += pines.vr01[(n, m)] * a_nm[(n, m + 1)] * d_;
                    sum.w -= pines.vr11[(n, m)] * a_nm[(n + 1, m + 1)] * d_;
                }
                let rr = rho_np1 / eq_radius_km;
                let mut lock = thread_accel.lock().unwrap();
                (*lock) += rr * sum;
            }
        });

        match handle.join() {
            Ok(_) => {}
            Err(e) => panic::resume_unwind(e),
        }

        let lock = accel4.lock().unwrap();

        let accel = Vector3::new(
            lock.x + lock.w * s_,
            lock.y + lock.w * t_,
            lock.z + lock.w * u_,
        );
        // Rotate this acceleration vector back into the integration frame (no center change needed, it's just a vector)
        // As discussed with Sai, if the Earth was spinning faster, would the acceleration due to the harmonics be any different?
        // No. Therefore, we do not need to account for the transport theorem here.
        let dcm = almanac
            .rotate(self.compute_frame, osc.frame, osc.epoch)
            .context(OrientationSnafu {
                action: "transform state dcm",
            })
            .context(DynamicsAlmanacSnafu {
                action: "transforming into gravity field frame",
            })?;

        Ok(dcm.rot_mat * accel)
    }

    fn dual_eom(
        &self,
        osc: &Orbit,
        almanac: Arc<Almanac>,
    ) -> Result<(Vector3<f64>, Matrix3<f64>), DynamicsError> {
        // Get a reference to Pines
        let pines = Arc::clone(&self.pines);

        // Convert the osculating orbit to the correct frame (needed for multiple harmonic fields)
        let state = almanac
            .transform_to(*osc, self.compute_frame, None)
            .context(DynamicsAlmanacSnafu {
                action: "transforming into gravity field frame",
            })?;

        let radius: Vector3<OHyperdual<f64, U7>> = hyperspace_from_vector(&state.radius_km);

        // Using the GMAT notation, with extra character for ease of highlight
        let r_ = norm(&radius);
        let s_ = radius[0] / r_;
        let t_ = radius[1] / r_;
        let u_ = radius[2] / r_;
        let max_degree = self.stor.max_degree_n(); // In GMAT, the order is NN
        let max_order = self.stor.max_order_m(); // In GMAT, the order is MM

        // Create the associated Legendre polynomials. Note that we add three items as per GMAT (this may be useful for the STM)
        let mut a_nm = pines.a_nm_h.clone();

        // Initialize the diagonal elements (not a function of the input)
        a_nm[(1, 0)] = u_ * 3.0f64.sqrt();
        for n in 1..=max_degree + 1 {
            let nf64 = n as f64;
            // Off diagonal
            a_nm[(n + 1, n)] = OHyperdual::from((2.0 * nf64 + 3.0).sqrt()) * u_ * a_nm[(n, n)];
        }

        for m in 0..=max_order + 1 {
            for n in (m + 2)..=max_degree + 1 {
                let hm_idx = (n, m);
                a_nm[hm_idx] = u_ * pines.b_nm_h[hm_idx] * a_nm[(n - 1, m)]
                    - pines.c_nm_h[hm_idx] * a_nm[(n - 2, m)];
            }
        }

        // Generate r_m and i_m
        let mut r_m = Vec::with_capacity(min(max_degree, max_order) + 1);
        let mut i_m = Vec::with_capacity(min(max_degree, max_order) + 1);

        r_m.push(OHyperdual::<f64, U7>::from(1.0));
        i_m.push(OHyperdual::<f64, U7>::from(0.0));

        for m in 1..=min(max_degree, max_order) {
            r_m.push(s_ * r_m[m - 1] - t_ * i_m[m - 1]);
            i_m.push(s_ * i_m[m - 1] + t_ * r_m[m - 1]);
        }

        let real_eq_radius_km = self
            .compute_frame
            .mean_equatorial_radius_km()
            .context(AstroPhysicsSnafu)
            .context(DynamicsAstroSnafu)?;

        let real_mu_km3_s2 = self
            .compute_frame
            .mu_km3_s2()
            .context(AstroPhysicsSnafu)
            .context(DynamicsAstroSnafu)?;

        let eq_radius = OHyperdual::<f64, U7>::from(real_eq_radius_km);
        let rho = eq_radius / r_;
        let mut rho_np1 = OHyperdual::<f64, U7>::from(real_mu_km3_s2) / r_ * rho;

        let mut a0 = OHyperdual::<f64, U7>::from(0.0);
        let mut a1 = OHyperdual::<f64, U7>::from(0.0);
        let mut a2 = OHyperdual::<f64, U7>::from(0.0);
        let mut a3 = OHyperdual::<f64, U7>::from(0.0);
        let sqrt2 = OHyperdual::<f64, U7>::from(2.0.sqrt());

        for n in 1..max_degree {
            let mut sum0 = OHyperdual::from(0.0);
            let mut sum1 = OHyperdual::from(0.0);
            let mut sum2 = OHyperdual::from(0.0);
            let mut sum3 = OHyperdual::from(0.0);
            rho_np1 *= rho;

            for m in 0..=min(n, max_order) {
                let (c_valf64, s_valf64) = self.stor.cs_nm(n, m);
                let c_val = OHyperdual::<f64, U7>::from(c_valf64);
                let s_val = OHyperdual::<f64, U7>::from(s_valf64);

                let d_ = (c_val * r_m[m] + s_val * i_m[m]) * sqrt2;
                let e_ = if m == 0 {
                    OHyperdual::from(0.0)
                } else {
                    (c_val * r_m[m - 1] + s_val * i_m[m - 1]) * sqrt2
                };
                let f_ = if m == 0 {
                    OHyperdual::from(0.0)
                } else {
                    (s_val * r_m[m - 1] - c_val * i_m[m - 1]) * sqrt2
                };

                sum0 += OHyperdual::from(m as f64) * a_nm[(n, m)] * e_;
                sum1 += OHyperdual::from(m as f64) * a_nm[(n, m)] * f_;
                sum2 += pines.vr01_h[(n, m)] * a_nm[(n, m + 1)] * d_;
                sum3 += pines.vr11_h[(n, m)] * a_nm[(n + 1, m + 1)] * d_;
            }
            let rr = rho_np1 / eq_radius;
            a0 += rr * sum0;
            a1 += rr * sum1;
            a2 += rr * sum2;
            a3 -= rr * sum3;
        }

        let dcm = almanac
            .rotate(self.compute_frame, osc.frame, osc.epoch)
            .context(OrientationSnafu {
                action: "transform state dcm",
            })
            .context(DynamicsAlmanacSnafu {
                action: "transforming into gravity field frame",
            })?
            .rot_mat;

        // Convert DCM to OHyperdual DCMs
        let mut dcm_d = Matrix3::<OHyperdual<f64, U7>>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                dcm_d[(i, j)] = OHyperdual::from_fn(|k| {
                    if k == 0 {
                        dcm[(i, j)]
                    } else if i + 1 == k {
                        1.0
                    } else {
                        0.0
                    }
                })
            }
        }

        let accel = dcm_d * Vector3::new(a0 + a3 * s_, a1 + a3 * t_, a2 + a3 * u_);
        // Extract data
        let mut dx = Vector3::zeros();
        let mut grad = Matrix3::zeros();
        for i in 0..3 {
            dx[i] += accel[i].real();
            // NOTE: Although the hyperdual state is of size 7, we're only setting the values up to 3 (Matrix3)
            for j in 1..4 {
                grad[(i, j - 1)] += accel[i][j];
            }
        }
        Ok((dx, grad))
    }
}
