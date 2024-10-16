use crate::linalg::{DMatrix, U7};
use hyperdual::OHyperdual;
use std::sync::Arc;
pub struct Pines {
    pub a_nm: DMatrix<f64>,
    pub b_nm: DMatrix<f64>,
    pub c_nm: DMatrix<f64>,
    pub vr01: DMatrix<f64>,
    pub vr11: DMatrix<f64>,
    pub a_nm_h: DMatrix<OHyperdual<f64, U7>>,
    pub b_nm_h: DMatrix<OHyperdual<f64, U7>>,
    pub c_nm_h: DMatrix<OHyperdual<f64, U7>>,
    pub vr01_h: DMatrix<OHyperdual<f64, U7>>,
    pub vr11_h: DMatrix<OHyperdual<f64, U7>>,
}

impl Pines {
    pub fn new(degree: usize) -> Arc<Pines> {
        let degree_np2 = degree + 2;
        let mut a_nm = DMatrix::from_element(degree_np2 + 1, degree_np2 + 1, 0.0);
        let mut b_nm = DMatrix::from_element(degree_np2, degree_np2, 0.0);
        let mut c_nm = DMatrix::from_element(degree_np2, degree_np2, 0.0);
        let mut vr01 = DMatrix::from_element(degree_np2, degree_np2, 0.0);
        let mut vr11 = DMatrix::from_element(degree_np2, degree_np2, 0.0);

        // Initialize the diagonal elements (not a function of the input)
        a_nm[(0, 0)] = 1.0;
        for n in 1..=degree_np2 {
            let nf64 = n as f64;
            // Diagonal element
            a_nm[(n, n)] = (1.0 + 1.0 / (2.0 * nf64)).sqrt() * a_nm[(n - 1, n - 1)];
        }

        // Pre-compute the B_nm, C_nm, vr01 and vr11 storages
        for n in 0..degree_np2 {
            for m in 0..degree_np2 {
                let nf64 = n as f64;
                let mf64 = m as f64;
                // Compute c_nm, which is B_nm/B_(n-1,m) in Jones' dissertation
                c_nm[(n, m)] = (((2.0 * nf64 + 1.0) * (nf64 + mf64 - 1.0) * (nf64 - mf64 - 1.0))
                    / ((nf64 - mf64) * (nf64 + mf64) * (2.0 * nf64 - 3.0)))
                    .sqrt();

                b_nm[(n, m)] = (((2.0 * nf64 + 1.0) * (2.0 * nf64 - 1.0))
                    / ((nf64 + mf64) * (nf64 - mf64)))
                    .sqrt();

                vr01[(n, m)] = ((nf64 - mf64) * (nf64 + mf64 + 1.0)).sqrt();
                vr11[(n, m)] = (((2.0 * nf64 + 1.0) * (nf64 + mf64 + 2.0) * (nf64 + mf64 + 1.0))
                    / (2.0 * nf64 + 3.0))
                    .sqrt();

                if m == 0 {
                    vr01[(n, m)] /= 2.0_f64.sqrt();
                    vr11[(n, m)] /= 2.0_f64.sqrt();
                }
            }
        }

        // Repeat for the hyperdual part in case we need to super the partials
        let mut a_nm_h =
            DMatrix::from_element(degree_np2 + 1, degree_np2 + 1, OHyperdual::from(0.0));
        let mut b_nm_h = DMatrix::from_element(degree_np2, degree_np2, OHyperdual::from(0.0));
        let mut c_nm_h = DMatrix::from_element(degree_np2, degree_np2, OHyperdual::from(0.0));
        let mut vr01_h = DMatrix::from_element(degree_np2, degree_np2, OHyperdual::from(0.0));
        let mut vr11_h = DMatrix::from_element(degree_np2, degree_np2, OHyperdual::from(0.0));

        // initialize the diagonal elements (not a function of the input)
        a_nm_h[(0, 0)] = OHyperdual::from(1.0);
        for n in 1..=degree_np2 {
            // Diagonal element
            a_nm_h[(n, n)] = OHyperdual::from(a_nm[(n, n)]);
        }

        // Pre-compute the B_nm, C_nm, vr01 and vr11 storages
        for n in 0..degree_np2 {
            for m in 0..degree_np2 {
                vr01_h[(n, m)] = OHyperdual::from(vr01[(n, m)]);
                vr11_h[(n, m)] = OHyperdual::from(vr11[(n, m)]);
                b_nm_h[(n, m)] = OHyperdual::from(b_nm[(n, m)]);
                c_nm_h[(n, m)] = OHyperdual::from(c_nm[(n, m)]);
            }
        }

        Arc::new(Self {
            a_nm,
            b_nm,
            c_nm,
            vr01,
            vr11,
            a_nm_h,
            b_nm_h,
            c_nm_h,
            vr01_h,
            vr11_h,
        })
    }
}
