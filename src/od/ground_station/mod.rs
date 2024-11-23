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

use super::msr::MeasurementType;
use super::noise::StochasticNoise;
use super::{ODAlmanacSnafu, ODError, ODTrajSnafu, TrackingDevice};
use crate::io::ConfigRepr;
use crate::time::Epoch;
use hifitime::Duration;
use rand_pcg::Pcg64Mcg;
use serde_derive::{Deserialize, Serialize};
use std::fmt;

pub mod builtin;
pub mod event;
pub mod trk_device;

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
            integration_time: Some(60 * Unit::Second),
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
