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

use anise::astro::{Aberration, AzElRange, PhysicsResult};
use anise::constants::frames::EARTH_J2000;
use anise::errors::AlmanacResult;
use anise::prelude::{Almanac, Frame, Orbit};
use indexmap::IndexSet;

use super::msr::measurement::Measurement;
use super::msr::MeasurementType;
use super::noise::StochasticNoise;
use super::{ODAlmanacSnafu, ODError, ODTrajSnafu, TrackingDevice};
use crate::errors::EventError;
use crate::io::ConfigRepr;
use crate::md::prelude::{Interpolatable, Traj};
use crate::md::EventEvaluator;
use crate::time::Epoch;
use crate::Spacecraft;
use hifitime::{Duration, TimeUnits, Unit};
use nalgebra::{allocator::Allocator, DefaultAllocator};
use rand_pcg::Pcg64Mcg;
use serde_derive::{Deserialize, Serialize};
use snafu::ResultExt;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// GroundStation defines a two-way ranging and doppler station.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "python", pyo3(module = "nyx_space.orbit_determination"))]
pub struct GroundStation {
    pub name: String,
    /// in degrees
    pub elevation_mask_deg: f64,
    /// in degrees
    pub latitude_deg: f64,
    /// in degrees
    pub longitude_deg: f64,
    /// in km
    pub height_km: f64,
    pub frame: Frame,
    pub measurement_types: IndexSet<MeasurementType>,
    /// Duration needed to generate a measurement (if unset, it is assumed to be instantaneous)
    /// TODO: Fix the deserialization of the measurements such that they also deserialize the integration time.
    /// Without it, we're stuck having to rebuild them from scratch.
    /// https://github.com/nyx-space/nyx/issues/140
    pub integration_time: Option<Duration>,
    /// Whether to correct for light travel time
    pub light_time_correction: bool,
    /// Noise on the timestamp of the measurement
    pub timestamp_noise_s: Option<StochasticNoise>,
    /// Noise on the range data of the measurement
    pub range_noise_km: Option<StochasticNoise>,
    /// Noise on the Doppler data of the measurement
    pub doppler_noise_km_s: Option<StochasticNoise>,
}

impl GroundStation {
    /// Initializes a point on the surface of a celestial object.
    /// This is meant for analysis, not for spacecraft navigation.
    pub fn from_point(
        name: String,
        latitude_deg: f64,
        longitude_deg: f64,
        height_km: f64,
        frame: Frame,
    ) -> Self {
        Self {
            name,
            elevation_mask_deg: 0.0,
            latitude_deg,
            longitude_deg,
            height_km,
            frame,
            measurement_types: IndexSet::new(),
            integration_time: None,
            light_time_correction: false,
            timestamp_noise_s: None,
            range_noise_km: None,
            doppler_noise_km_s: None,
        }
    }

    pub fn dss65_madrid(
        elevation_mask: f64,
        range_noise_km: StochasticNoise,
        doppler_noise_km_s: StochasticNoise,
        iau_earth: Frame,
    ) -> Self {
        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);
        Self {
            name: "Madrid".to_string(),
            elevation_mask_deg: elevation_mask,
            latitude_deg: 40.427_222,
            longitude_deg: 4.250_556,
            height_km: 0.834_939,
            frame: iau_earth,
            measurement_types,
            integration_time: None,
            light_time_correction: false,
            timestamp_noise_s: None,
            range_noise_km: Some(range_noise_km),
            doppler_noise_km_s: Some(doppler_noise_km_s),
        }
    }

    pub fn dss34_canberra(
        elevation_mask: f64,
        range_noise_km: StochasticNoise,
        doppler_noise_km_s: StochasticNoise,
        iau_earth: Frame,
    ) -> Self {
        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);
        Self {
            name: "Canberra".to_string(),
            elevation_mask_deg: elevation_mask,
            latitude_deg: -35.398_333,
            longitude_deg: 148.981_944,
            height_km: 0.691_750,
            frame: iau_earth,
            measurement_types,
            integration_time: None,
            light_time_correction: false,
            timestamp_noise_s: None,
            range_noise_km: Some(range_noise_km),
            doppler_noise_km_s: Some(doppler_noise_km_s),
        }
    }

    pub fn dss13_goldstone(
        elevation_mask: f64,
        range_noise_km: StochasticNoise,
        doppler_noise_km_s: StochasticNoise,
        iau_earth: Frame,
    ) -> Self {
        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);
        Self {
            name: "Goldstone".to_string(),
            elevation_mask_deg: elevation_mask,
            latitude_deg: 35.247_164,
            longitude_deg: 243.205,
            height_km: 1.071_149_04,
            frame: iau_earth,
            measurement_types,
            integration_time: None,
            light_time_correction: false,
            timestamp_noise_s: None,
            range_noise_km: Some(range_noise_km),
            doppler_noise_km_s: Some(doppler_noise_km_s),
        }
    }

    /// Computes the azimuth and elevation of the provided object seen from this ground station, both in degrees.
    /// This is a shortcut to almanac.azimuth_elevation_range_sez.
    pub fn azimuth_elevation_of(
        &self,
        rx: Orbit,
        obstructing_body: Option<Frame>,
        almanac: &Almanac,
    ) -> AlmanacResult<AzElRange> {
        let ab_corr = if self.light_time_correction {
            Aberration::LT
        } else {
            Aberration::NONE
        };
        almanac.azimuth_elevation_range_sez(
            rx,
            self.to_orbit(rx.epoch, almanac).unwrap(),
            obstructing_body,
            ab_corr,
        )
    }

    /// Return this ground station as an orbit in its current frame
    pub fn to_orbit(&self, epoch: Epoch, almanac: &Almanac) -> PhysicsResult<Orbit> {
        use anise::constants::usual_planetary_constants::MEAN_EARTH_ANGULAR_VELOCITY_DEG_S;
        Orbit::try_latlongalt(
            self.latitude_deg,
            self.longitude_deg,
            self.height_km,
            MEAN_EARTH_ANGULAR_VELOCITY_DEG_S,
            epoch,
            almanac.frame_from_uid(self.frame).unwrap(),
        )
    }

    /// Returns the noises for all measurement types configured for this ground station at the provided epoch, timestamp noise is the first entry.
    fn noises(&mut self, epoch: Epoch, rng: Option<&mut Pcg64Mcg>) -> Result<Vec<f64>, ODError> {
        let mut noises = vec![0.0; self.measurement_types.len() + 1];

        if let Some(rng) = rng {
            // Add the timestamp noise first

            if let Some(mut timestamp_noise) = self.timestamp_noise_s {
                noises[0] = timestamp_noise.sample(epoch, rng);
            }

            for (ii, msr_type) in self.measurement_types.iter().enumerate() {
                noises[ii + 1] = match msr_type {
                    MeasurementType::Range => self
                        .range_noise_km
                        .ok_or(ODError::NoiseNotConfigured { kind: "Range" })?
                        .sample(epoch, rng),
                    MeasurementType::Doppler => self
                        .doppler_noise_km_s
                        .ok_or(ODError::NoiseNotConfigured { kind: "Doppler" })?
                        .sample(epoch, rng),
                };
            }
        }

        Ok(noises)
    }
}

impl Default for GroundStation {
    fn default() -> Self {
        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);
        Self {
            name: "UNDEFINED".to_string(),
            measurement_types,
            elevation_mask_deg: 0.0,
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            height_km: 0.0,
            frame: EARTH_J2000,
            integration_time: None,
            light_time_correction: false,
            timestamp_noise_s: None,
            range_noise_km: None,
            doppler_noise_km_s: None,
        }
    }
}

impl ConfigRepr for GroundStation {}

impl TrackingDevice<Spacecraft> for GroundStation {
    fn measurement_types(&self) -> &IndexSet<MeasurementType> {
        &self.measurement_types
    }

    /// Perform a measurement from the ground station to the receiver (rx).
    fn measure(
        &mut self,
        epoch: Epoch,
        traj: &Traj<Spacecraft>,
        rng: Option<&mut Pcg64Mcg>,
        almanac: Arc<Almanac>,
    ) -> Result<Option<super::prelude::measurement::Measurement>, ODError> {
        match self.integration_time {
            Some(integration_time) => {
                // If out of traj bounds, return None.
                let rx_0 = traj.at(epoch - integration_time).context(ODTrajSnafu)?;
                let rx_1 = traj.at(epoch).context(ODTrajSnafu)?;

                let obstructing_body = if !self.frame.ephem_origin_match(rx_0.frame()) {
                    Some(rx_0.frame())
                } else {
                    None
                };

                let aer_t0 = self
                    .azimuth_elevation_of(rx_0.orbit, obstructing_body, &almanac)
                    .context(ODAlmanacSnafu {
                        action: "computing AER",
                    })?;
                let aer_t1 = self
                    .azimuth_elevation_of(rx_1.orbit, obstructing_body, &almanac)
                    .context(ODAlmanacSnafu {
                        action: "computing AER",
                    })?;

                if aer_t0.elevation_deg < self.elevation_mask_deg
                    || aer_t1.elevation_deg < self.elevation_mask_deg
                {
                    debug!(
                        "{} (el. mask {:.3} deg) but object moves from {:.3} to {:.3} deg -- no measurement",
                        self.name, self.elevation_mask_deg, aer_t0.elevation_deg, aer_t1.elevation_deg
                    );
                    return Ok(None);
                }

                // Noises are computed at the midpoint of the integration time.
                let noises = self.noises(epoch - integration_time * 0.5, rng)?;

                let mut msr = Measurement::new(self.name.clone(), epoch + noises[0].seconds());

                for (ii, msr_type) in self.measurement_types.iter().enumerate() {
                    let msr_value = msr_type.compute_two_way(aer_t0, aer_t1, noises[ii + 1])?;
                    msr.push(*msr_type, msr_value);
                }

                Ok(Some(msr))
            }
            None => self.measure_instantaneous(traj.at(epoch).context(ODTrajSnafu)?, rng, almanac),
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn location(&self, epoch: Epoch, frame: Frame, almanac: Arc<Almanac>) -> AlmanacResult<Orbit> {
        almanac.transform_to(self.to_orbit(epoch, &almanac).unwrap(), frame, None)
    }

    fn measure_instantaneous(
        &mut self,
        rx: Spacecraft,
        rng: Option<&mut Pcg64Mcg>,
        almanac: Arc<Almanac>,
    ) -> Result<Option<super::prelude::measurement::Measurement>, ODError> {
        let obstructing_body = if !self.frame.ephem_origin_match(rx.frame()) {
            Some(rx.frame())
        } else {
            None
        };

        let aer = self
            .azimuth_elevation_of(rx.orbit, obstructing_body, &almanac)
            .context(ODAlmanacSnafu {
                action: "computing AER",
            })?;

        if aer.elevation_deg >= self.elevation_mask_deg {
            // Only update the noises if the measurement is valid.
            let noises = self.noises(rx.orbit.epoch, rng)?;

            let mut msr = Measurement::new(self.name.clone(), rx.orbit.epoch + noises[0].seconds());

            for (ii, msr_type) in self.measurement_types.iter().enumerate() {
                let msr_value = msr_type.compute_one_way(aer, noises[ii + 1])?;
                msr.push(*msr_type, msr_value);
            }

            Ok(Some(msr))
        } else {
            debug!(
                "{} {} (el. mask {:.3} deg), object at {:.3} deg -- no measurement",
                self.name, rx.orbit.epoch, self.elevation_mask_deg, aer.elevation_deg
            );
            Ok(None)
        }
    }

    /// Returns the measurement noise of this ground station.
    ///
    /// # Methodology
    /// Noises are modeled using a [StochasticNoise] process, defined by the sigma on the turn-on bias and on the steady state noise.
    /// The measurement noise is computed assuming that all measurements are independent variables, i.e. the measurement matrix is
    /// a diagonal matrix. The first item in the diagonal is the range noise (in km), set to the square of the steady state sigma. The
    /// second item is the Doppler noise (in km/s), set to the square of the steady state sigma of that Gauss Markov process.
    fn measurement_covar(
        &self,
        msr_type: super::prelude::MeasurementType,
        epoch: Epoch,
    ) -> Result<f64, ODError> {
        Ok(match msr_type {
            super::msr::MeasurementType::Range => self
                .range_noise_km
                .ok_or(ODError::NoiseNotConfigured { kind: "Range" })?
                .covariance(epoch),
            super::msr::MeasurementType::Doppler => self
                .doppler_noise_km_s
                .ok_or(ODError::NoiseNotConfigured { kind: "Doppler" })?
                .covariance(epoch),
        })
    }
}

impl fmt::Display for GroundStation {
    // Prints the Keplerian orbital elements with units
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} (lat.: {:.4} deg    long.: {:.4} deg    alt.: {:.3} m) [{}]",
            self.name,
            self.latitude_deg,
            self.longitude_deg,
            self.height_km * 1e3,
            self.frame,
        )
    }
}

impl<S: Interpolatable> EventEvaluator<S> for &GroundStation
where
    DefaultAllocator: Allocator<S::Size> + Allocator<S::Size, S::Size> + Allocator<S::VecLength>,
{
    /// Compute the elevation in the SEZ frame. This call will panic if the frame of the input state does not match that of the ground station.
    fn eval(&self, rx_gs_frame: &S, almanac: Arc<Almanac>) -> Result<f64, EventError> {
        let dt = rx_gs_frame.epoch();
        // Then, compute the rotation matrix from the body fixed frame of the ground station to its topocentric frame SEZ.
        let tx_gs_frame = self.to_orbit(dt, &almanac).unwrap();

        let from = tx_gs_frame.frame.orientation_id * 1_000 + 1;
        let dcm_topo2fixed = tx_gs_frame
            .dcm_from_topocentric_to_body_fixed(from)
            .unwrap()
            .transpose();

        // Now, rotate the spacecraft in the SEZ frame to compute its elevation as seen from the ground station.
        // We transpose the DCM so that it's the fixed to topocentric rotation.
        let rx_sez = (dcm_topo2fixed * rx_gs_frame.orbit()).unwrap();
        let tx_sez = (dcm_topo2fixed * tx_gs_frame).unwrap();
        // Now, let's compute the range ρ.
        let rho_sez = (rx_sez - tx_sez).unwrap();

        // Finally, compute the elevation (math is the same as declination)
        // Source: Vallado, section 4.4.3
        // Only the sine is needed as per Vallado, and the formula is the same as the declination
        // because we're in the SEZ frame.
        Ok(rho_sez.declination_deg() - self.elevation_mask_deg)
    }

    fn eval_string(&self, state: &S, almanac: Arc<Almanac>) -> Result<String, EventError> {
        Ok(format!(
            "Elevation from {} is {:.6} deg on {}",
            self.name,
            self.eval(state, almanac)? + self.elevation_mask_deg,
            state.epoch()
        ))
    }

    /// Epoch precision of the election evaluator is 1 ms
    fn epoch_precision(&self) -> Duration {
        1 * Unit::Second
    }

    /// Angle precision of the elevation evaluator is 1 millidegree.
    fn value_precision(&self) -> f64 {
        1e-3
    }
}

#[cfg(test)]
mod gs_ut {

    use anise::constants::frames::IAU_EARTH_FRAME;
    use indexmap::IndexSet;

    use crate::io::ConfigRepr;
    use crate::od::prelude::*;

    #[test]
    fn test_load_single() {
        use std::env;
        use std::path::PathBuf;

        use hifitime::TimeUnits;

        // Get the path to the root directory of the current Cargo project
        let test_data: PathBuf = [
            env::var("CARGO_MANIFEST_DIR").unwrap(),
            "data".to_string(),
            "tests".to_string(),
            "config".to_string(),
            "one_ground_station.yaml".to_string(),
        ]
        .iter()
        .collect();

        assert!(test_data.exists(), "Could not find the test data");

        let gs = GroundStation::load(test_data).unwrap();

        dbg!(&gs);

        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);

        let expected_gs = GroundStation {
            name: "Demo ground station".to_string(),
            frame: IAU_EARTH_FRAME,
            measurement_types,
            elevation_mask_deg: 5.0,
            range_noise_km: Some(StochasticNoise {
                bias: Some(GaussMarkov::new(1.days(), 5e-3).unwrap()),
                ..Default::default()
            }),
            doppler_noise_km_s: Some(StochasticNoise {
                bias: Some(GaussMarkov::new(1.days(), 5e-5).unwrap()),
                ..Default::default()
            }),
            latitude_deg: 2.3522,
            longitude_deg: 48.8566,
            height_km: 0.4,
            light_time_correction: false,
            timestamp_noise_s: None,
            integration_time: None,
        };

        assert_eq!(expected_gs, gs);
    }

    #[test]
    fn test_load_many() {
        use hifitime::TimeUnits;
        use std::env;
        use std::path::PathBuf;

        // Get the path to the root directory of the current Cargo project

        let test_file: PathBuf = [
            env::var("CARGO_MANIFEST_DIR").unwrap(),
            "data".to_string(),
            "tests".to_string(),
            "config".to_string(),
            "many_ground_stations.yaml".to_string(),
        ]
        .iter()
        .collect();

        let stations = GroundStation::load_many(test_file).unwrap();

        dbg!(&stations);

        let mut measurement_types = IndexSet::new();
        measurement_types.insert(MeasurementType::Range);
        measurement_types.insert(MeasurementType::Doppler);

        let expected = vec![
            GroundStation {
                name: "Demo ground station".to_string(),
                frame: IAU_EARTH_FRAME.with_mu_km3_s2(398600.435436096),
                measurement_types: measurement_types.clone(),
                elevation_mask_deg: 5.0,
                range_noise_km: Some(StochasticNoise {
                    bias: Some(GaussMarkov::new(1.days(), 5e-3).unwrap()),
                    ..Default::default()
                }),
                doppler_noise_km_s: Some(StochasticNoise {
                    bias: Some(GaussMarkov::new(1.days(), 5e-5).unwrap()),
                    ..Default::default()
                }),
                latitude_deg: 2.3522,
                longitude_deg: 48.8566,
                height_km: 0.4,
                light_time_correction: false,
                timestamp_noise_s: None,
                integration_time: None,
            },
            GroundStation {
                name: "Canberra".to_string(),
                frame: IAU_EARTH_FRAME.with_mu_km3_s2(398600.435436096),
                measurement_types,
                elevation_mask_deg: 5.0,
                range_noise_km: Some(StochasticNoise {
                    bias: Some(GaussMarkov::new(1.days(), 5e-3).unwrap()),
                    ..Default::default()
                }),
                doppler_noise_km_s: Some(StochasticNoise {
                    bias: Some(GaussMarkov::new(1.days(), 5e-5).unwrap()),
                    ..Default::default()
                }),
                latitude_deg: -35.398333,
                longitude_deg: 148.981944,
                height_km: 0.691750,
                light_time_correction: false,
                timestamp_noise_s: None,
                integration_time: None,
            },
        ];

        assert_eq!(expected, stations);

        // Serialize back
        let reser = serde_yaml::to_string(&expected).unwrap();
        dbg!(reser);
    }
}
