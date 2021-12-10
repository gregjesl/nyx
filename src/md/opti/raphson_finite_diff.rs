/*
    Nyx, blazing fast astrodynamics
    Copyright (C) 2021 Christopher Rabotin <christopher.rabotin@gmail.com>

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

use super::targeter::{Targeter, TargeterSolution};
use crate::dynamics::guidance::Mnvr;
use crate::errors::TargetingError;
use crate::linalg::{DMatrix, DVector, Vector6};
use crate::md::rayon::prelude::*;
use crate::md::ui::*;
use crate::md::StateParameter;
pub use crate::md::{Variable, Vary};
use crate::polyfit::CommonPolynomial;
use crate::propagators::error_ctrl::ErrorCtrl;
use crate::utils::pseudo_inverse;
use hifitime::TimeUnitHelper;
use std::time::Instant;

impl<'a, E: ErrorCtrl> Targeter<'a, E> {
    /// Differential correction using finite differencing
    #[allow(clippy::comparison_chain)]
    pub fn try_achieve_fd(
        &self,
        initial_state: Spacecraft,
        correction_epoch: Epoch,
        achievement_epoch: Epoch,
    ) -> Result<TargeterSolution, NyxError> {
        if self.objectives.is_empty() {
            return Err(NyxError::Targeter(TargetingError::UnderdeterminedProblem));
        }

        let mut is_bplane_tgt = false;
        for obj in &self.objectives {
            if obj.parameter.is_b_plane() {
                is_bplane_tgt = true;
                break;
            }
        }

        // Now we know that the problem is correctly defined, so let's propagate as is to the epoch
        // where the correction should be applied.
        let xi_start = self
            .prop
            .with(initial_state)
            .until_epoch(correction_epoch)?;

        debug!("initial_state = {}", initial_state);
        debug!("xi_start = {}", xi_start);

        let mut xi = xi_start;
        // We'll store the initial state correction here.
        let mut state_correction = Vector6::<f64>::zeros();

        // Store the total correction in Vector3
        let mut total_correction = DVector::from_element(self.variables.len(), 0.0);

        // Create a default maneuver that will only be used if a finite burn is being targeted
        let mut mnvr = Mnvr {
            start: correction_epoch,
            end: correction_epoch + 5.seconds(),
            thrust_lvl: 1.0,
            alpha_inplane_degrees: CommonPolynomial::Quadratic(0.0, 0.0, 0.0),
            beta_outofplane_degrees: CommonPolynomial::Quadratic(0.0, 0.0, 0.0),
            frame: Frame::RCN,
        };

        let mut finite_burn_target = false;

        // Apply the initial guess
        for (i, var) in self.variables.iter().enumerate() {
            // Check the validity (this function will report to log and raise an error)
            var.valid()?;
            // Check that there is no attempt to target a position in a local frame
            if self.correction_frame.is_some() && var.component.vec_index() < 3 {
                // Then this is a position correction, which is not allowed if a frame is provided!
                let msg = format!(
                    "Variable is in frame {} but that frame cannot be used for a {:?} correction",
                    self.correction_frame.unwrap(),
                    var.component
                );
                error!("{}", msg);
                return Err(NyxError::Targeter(TargetingError::FrameError(msg)));
            }

            // Check that a thruster is provided since we'll be changing that and the burn duration
            if var.component.is_finite_burn() {
                if xi_start.thruster.is_none() {
                    // Can't do any conversion to finite burns without a thruster
                    return Err(NyxError::CtrlExistsButNoThrusterAvail);
                }
                finite_burn_target = true;
                // Modify the default maneuver
                match var.component {
                    Vary::Duration => mnvr.end = mnvr.start + var.init_guess.seconds(),
                    Vary::EndEpoch => mnvr.end = mnvr.end + var.init_guess.seconds(),
                    Vary::StartEpoch => mnvr.start = mnvr.start + var.init_guess.seconds(),
                    Vary::MnvrAlpha | Vary::MnvrAlphaDot | Vary::MnvrAlphaDDot => {
                        mnvr.alpha_inplane_degrees = mnvr
                            .alpha_inplane_degrees
                            .add_val_in_order(var.init_guess, var.component.vec_index())
                            .unwrap();
                    }
                    Vary::MnvrBeta | Vary::MnvrBetaDot | Vary::MnvrBetaDDot => {
                        mnvr.beta_outofplane_degrees = mnvr
                            .beta_outofplane_degrees
                            .add_val_in_order(var.init_guess, var.component.vec_index())
                            .unwrap();
                    }
                    _ => unreachable!(),
                }
                info!("Initial maneuver guess: {}", mnvr);
            } else {
                state_correction[var.component.vec_index()] += var.init_guess;
                // Now, let's apply the correction to the initial state
                if let Some(frame) = self.correction_frame {
                    // The following will error if the frame is not local
                    let dcm_vnc2inertial = xi.orbit.dcm_from_traj_frame(frame)?;
                    let velocity_correction =
                        dcm_vnc2inertial * state_correction.fixed_rows::<3>(3);
                    xi.orbit.apply_dv(velocity_correction);
                } else {
                    xi.orbit.x += state_correction[0];
                    xi.orbit.y += state_correction[1];
                    xi.orbit.z += state_correction[2];
                    xi.orbit.vx += state_correction[3];
                    xi.orbit.vy += state_correction[4];
                    xi.orbit.vz += state_correction[5];
                }
            }

            total_correction[i] += var.init_guess;
        }

        let mut prev_err_norm = std::f64::INFINITY;

        // Determine padding in debugging info
        // For the width, we find the largest desired values and multiply it by the order of magnitude of its tolerance
        let max_obj_val = self
            .objectives
            .iter()
            .map(|obj| {
                (obj.desired_value.abs().ceil() as i32
                    * 10_i32.pow(obj.tolerance.abs().log10().ceil() as u32)) as i32
            })
            .max()
            .unwrap();

        let max_obj_tol = self
            .objectives
            .iter()
            .map(|obj| obj.tolerance.log10().abs().ceil() as usize)
            .max()
            .unwrap();

        let width = f64::from(max_obj_val).log10() as usize + 2 + max_obj_tol;

        let start_instant = Instant::now();

        for it in 0..=self.iterations {
            // Modify each variable by the desired perturbatino, propagate, compute the final parameter, and store how modifying that variable affects the final parameter
            let cur_xi = xi;

            // If we are targeting a finite burn, let's set propagate in several steps to make sure we don't miss the burn
            let xf = if finite_burn_target {
                info!("#{} {}", it, mnvr);
                let mut prop = self.prop.clone();
                let prop_opts = prop.opts;
                let pre_mnvr = prop.with(cur_xi).until_epoch(mnvr.start)?;
                prop.dynamics = prop.dynamics.with_ctrl(Arc::new(mnvr));
                prop.set_max_step(mnvr.end - mnvr.start);
                let post_mnvr = prop
                    .with(pre_mnvr.with_guidance_mode(GuidanceMode::Thrust))
                    .until_epoch(mnvr.end)?;
                // Reset the propagator options to their previous configuration
                prop.opts = prop_opts;
                // And propagate until the achievement epoch
                prop.with(post_mnvr).until_epoch(achievement_epoch)?.orbit
            } else {
                self.prop.with(cur_xi).until_epoch(achievement_epoch)?.orbit
            };

            let xf_dual_obj_frame = match &self.objective_frame {
                Some((frame, cosm)) => {
                    let orbit_obj_frame = cosm.frame_chg(&xf, *frame);
                    OrbitDual::from(orbit_obj_frame)
                }
                None => OrbitDual::from(xf),
            };

            // Build the error vector
            let mut param_errors = Vec::with_capacity(self.objectives.len());
            let mut converged = true;

            // Build the B-Plane once, if needed, and always in the objective frame
            let b_plane = if is_bplane_tgt {
                Some(BPlane::from_dual(xf_dual_obj_frame)?)
            } else {
                None
            };

            // Build debugging information
            let mut objmsg = Vec::with_capacity(self.objectives.len());

            // The Jacobian includes the sensitivity of each objective with respect to each variable for the whole trajectory.
            // As such, it includes the STM of that variable for the whole propagation arc.
            let mut jac = DMatrix::from_element(self.objectives.len(), self.variables.len(), 0.0);

            for (i, obj) in self.objectives.iter().enumerate() {
                let partial = if obj.parameter.is_b_plane() {
                    match obj.parameter {
                        StateParameter::BdotR => b_plane.unwrap().b_r,
                        StateParameter::BdotT => b_plane.unwrap().b_t,
                        StateParameter::BLTOF => b_plane.unwrap().ltof_s,
                        _ => unreachable!(),
                    }
                } else {
                    xf_dual_obj_frame.partial_for(&obj.parameter)?
                };

                let achieved = partial.real();

                let (ok, param_err) = obj.assess_raw(achieved);
                if !ok {
                    converged = false;
                }
                param_errors.push(param_err);

                objmsg.push(format!(
                    "\t{:?}: achieved = {:>width$.prec$}\t desired = {:>width$.prec$}\t scaled error = {:>width$.prec$}",
                    obj.parameter,
                    achieved,
                    obj.desired_value,
                    param_err, width=width, prec=max_obj_tol
                ));

                let mut pert_calc: Vec<_> = self
                    .variables
                    .iter()
                    .enumerate()
                    .map(|(j, var)| (j, var, 0.0_f64))
                    .collect();

                pert_calc.par_iter_mut().for_each(|(_, var, jac_val)| {
                    let mut this_xi = xi;

                    let mut this_prop = self.prop.clone();
                    let mut this_mnvr = mnvr;

                    if var.component.is_finite_burn() {
                        // Modify the burn itself
                        let pert = var.perturbation;
                        // Modify the maneuver, but do not change the epochs of the maneuver unless the change is greater than one millisecond
                        match var.component {
                            Vary::Duration => {
                                if pert.abs() > 1e-3 {
                                    this_mnvr.end = mnvr.start + pert.seconds()
                                }
                            }
                            Vary::EndEpoch => {
                                if pert.abs() > 1e-3 {
                                    this_mnvr.end = mnvr.end + pert.seconds()
                                }
                            }
                            Vary::StartEpoch => {
                                if pert.abs() > 1e-3 {
                                    this_mnvr.start = mnvr.start + pert.seconds()
                                }
                            }
                            Vary::MnvrAlpha | Vary::MnvrAlphaDot | Vary::MnvrAlphaDDot => {
                                this_mnvr.alpha_inplane_degrees = mnvr
                                    .alpha_inplane_degrees
                                    .add_val_in_order(pert, var.component.vec_index())
                                    .unwrap();
                            }
                            Vary::MnvrBeta | Vary::MnvrBetaDot | Vary::MnvrBetaDDot => {
                                this_mnvr.beta_outofplane_degrees = mnvr
                                    .beta_outofplane_degrees
                                    .add_val_in_order(pert, var.component.vec_index())
                                    .unwrap();
                            }
                            _ => unreachable!(),
                        }
                        this_prop.dynamics = this_prop.dynamics.with_ctrl(Arc::new(this_mnvr));
                    } else {
                        let mut state_correction = Vector6::<f64>::zeros();
                        state_correction[var.component.vec_index()] += var.perturbation;
                        // Now, let's apply the correction to the initial state
                        if let Some(frame) = self.correction_frame {
                            // The following will error if the frame is not local
                            let dcm_vnc2inertial =
                                this_xi.orbit.dcm_from_traj_frame(frame).unwrap();
                            let velocity_correction =
                                dcm_vnc2inertial * state_correction.fixed_rows::<3>(3);
                            this_xi.orbit.apply_dv(velocity_correction);
                        } else {
                            this_xi = xi + state_correction;
                        }
                    }

                    let this_xf = if finite_burn_target {
                        let prop_opts = this_prop.opts;
                        let pre_mnvr = this_prop.with(cur_xi).until_epoch(this_mnvr.start).unwrap();
                        this_prop.dynamics = this_prop.dynamics.with_ctrl(Arc::new(this_mnvr));
                        this_prop.set_max_step(this_mnvr.end - this_mnvr.start);
                        let post_mnvr = this_prop
                            .with(pre_mnvr.with_guidance_mode(GuidanceMode::Thrust))
                            .until_epoch(this_mnvr.end)
                            .unwrap();
                        // Reset the propagator options to their previous configuration
                        this_prop.opts = prop_opts;
                        // And propagate until the achievement epoch
                        this_prop
                            .with(post_mnvr)
                            .until_epoch(achievement_epoch)
                            .unwrap()
                            .orbit
                    } else {
                        this_prop
                            .with(this_xi)
                            .until_epoch(achievement_epoch)
                            .unwrap()
                            .orbit
                    };

                    let xf_dual_obj_frame = match &self.objective_frame {
                        Some((frame, cosm)) => {
                            let orbit_obj_frame = cosm.frame_chg(&this_xf, *frame);
                            OrbitDual::from(orbit_obj_frame)
                        }
                        None => OrbitDual::from(this_xf),
                    };

                    let b_plane = if is_bplane_tgt {
                        Some(BPlane::from_dual(xf_dual_obj_frame).unwrap())
                    } else {
                        None
                    };

                    let partial = if obj.parameter.is_b_plane() {
                        match obj.parameter {
                            StateParameter::BdotR => b_plane.unwrap().b_r,
                            StateParameter::BdotT => b_plane.unwrap().b_t,
                            StateParameter::BLTOF => b_plane.unwrap().ltof_s,
                            _ => unreachable!(),
                        }
                    } else {
                        xf_dual_obj_frame.partial_for(&obj.parameter).unwrap()
                    };

                    let this_achieved = partial.real();
                    *jac_val = (this_achieved - achieved) / var.perturbation;
                });

                for (j, _, jac_val) in &pert_calc {
                    jac[(i, *j)] = *jac_val;
                }
            }

            if converged {
                let conv_dur = Instant::now() - start_instant;
                let mut corrected_state = xi_start;

                let mut state_correction = Vector6::<f64>::zeros();
                for (i, var) in self.variables.iter().enumerate() {
                    state_correction[var.component.vec_index()] += total_correction[i];
                }
                // Now, let's apply the correction to the initial state
                if let Some(frame) = self.correction_frame {
                    let dcm_vnc2inertial = corrected_state
                        .orbit
                        .dcm_from_traj_frame(frame)
                        .unwrap()
                        .transpose();
                    let velocity_correction =
                        dcm_vnc2inertial * state_correction.fixed_rows::<3>(3);
                    corrected_state.orbit.apply_dv(velocity_correction);
                } else {
                    corrected_state.orbit = corrected_state.orbit + state_correction;
                }

                let sol = TargeterSolution {
                    corrected_state,
                    achieved_state: xi_start.with_orbit(xf),
                    correction: total_correction,
                    computation_dur: conv_dur,
                    variables: self.variables.clone(),
                    achieved_errors: param_errors,
                    achieved_objectives: self.objectives.clone(),
                    iterations: it,
                };
                // Log success as info
                if it == 1 {
                    info!("Targeter -- CONVERGED in 1 iteration");
                } else {
                    info!("Targeter -- CONVERGED in {} iterations", it);
                }
                for obj in &objmsg {
                    info!("{}", obj);
                }
                return Ok(sol);
            }

            // We haven't converged yet, so let's build the error vector
            let err_vector = DVector::from(param_errors);
            if (err_vector.norm() - prev_err_norm).abs() < 1e-10 {
                return Err(NyxError::CorrectionIneffective(
                    "No change in objective errors".to_string(),
                ));
            }
            prev_err_norm = err_vector.norm();

            debug!("Jacobian {}", jac);

            // Perform the pseudo-inverse if needed, else just inverse
            let jac_inv = pseudo_inverse(&jac, NyxError::SingularJacobian)?;

            debug!("Inverse Jacobian {}", jac_inv);

            let mut delta = jac_inv * &err_vector;

            debug!("Error vector: {}\nRaw correction: {}", err_vector, delta);

            // And finally apply it to the xi
            let mut state_correction = Vector6::<f64>::zeros();
            for (i, var) in self.variables.iter().enumerate() {
                // Choose the minimum step between the provided max step and the correction.
                if delta[i].abs() > var.max_step.abs() {
                    delta[i] = var.max_step.abs() * delta[i].signum();
                } else if delta[i] > var.max_value {
                    delta[i] = var.max_value;
                } else if delta[i] < var.min_value {
                    delta[i] = var.min_value;
                }

                debug!(
                    "Correction {:?}{} (element {}): {}",
                    var.component,
                    match self.correction_frame {
                        Some(f) => format!(" in {:?}", f),
                        None => format!(""),
                    },
                    i,
                    delta[i]
                );

                let corr = delta[i];

                if var.component.is_finite_burn() {
                    // Modify the maneuver, but do not change the epochs of the maneuver unless the change is greater than one millisecond
                    match var.component {
                        Vary::Duration => {
                            if corr.abs() > 1e-3 {
                                mnvr.end = mnvr.start + corr.abs().seconds();
                            }
                        }
                        Vary::EndEpoch => {
                            if corr.abs() > 1e-3 {
                                mnvr.end = mnvr.end + corr.seconds()
                            }
                        }
                        Vary::StartEpoch => {
                            if corr.abs() > 1e-3 {
                                mnvr.start = mnvr.start + corr.seconds()
                            }
                        }
                        Vary::MnvrAlpha | Vary::MnvrAlphaDot | Vary::MnvrAlphaDDot => {
                            mnvr.alpha_inplane_degrees = mnvr
                                .alpha_inplane_degrees
                                .add_val_in_order(corr, var.component.vec_index())
                                .unwrap();
                        }
                        Vary::MnvrBeta | Vary::MnvrBetaDot | Vary::MnvrBetaDDot => {
                            mnvr.beta_outofplane_degrees = mnvr
                                .beta_outofplane_degrees
                                .add_val_in_order(corr, var.component.vec_index())
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                } else {
                    state_correction[var.component.vec_index()] += corr;
                }
            }

            // Now, let's apply the correction to the initial state
            if let Some(frame) = self.correction_frame {
                let dcm_vnc2inertial = xi.orbit.dcm_from_traj_frame(frame)?;
                let velocity_correction = dcm_vnc2inertial * state_correction.fixed_rows::<3>(3);
                xi.orbit.apply_dv(velocity_correction);
            } else {
                xi = xi + state_correction;
            }
            total_correction += delta;
            debug!("Total correction: {:e}", total_correction);

            // Log progress to debug
            info!("Targeter -- Iteration #{} -- {}", it, achievement_epoch);
            for obj in &objmsg {
                info!("{}", obj);
            }
        }

        Err(NyxError::MaxIterReached(format!(
            "Failed after {} iterations:\nError: {}\n\n{}",
            self.iterations, prev_err_norm, self
        )))
    }
}
