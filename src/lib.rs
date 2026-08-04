use numpy::ndarray::{Array2, ArrayView3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

fn cf_share(qs: &ArrayView3<f64>, idx_n: usize, idx_t: usize, j: usize, k: usize) -> f64 {
    let mut mu_total = 0.0;
    for m in 0..k {
        mu_total += -(1.0 - qs[[m, idx_n, idx_t]]).ln();
    }
    if mu_total == 0.0 {
        return 0.0;
    }
    -(1.0 - qs[[j, idx_n, idx_t]]).ln() / mu_total
}

fn udd_share(
    qs: &ArrayView3<f64>,
    idx_n: usize,
    idx_t: usize,
    j: usize,
    k: usize,
    s: &[f64],
    w: &[f64],
) -> f64 {
    let mut total = 0.0;
    for idx in 0..s.len() {
        let mut product = qs[[j, idx_n, idx_t]];
        for m in 0..k {
            if m != j {
                product *= 1.0 - s[idx] * qs[[m, idx_n, idx_t]];
            }
        }
        total += w[idx] * product;
    }
    total
}

#[pyfunction]
fn ping() -> String {
    "pong desde Rust".to_string()
}

#[pyfunction]
fn compute_inf<'py>(py: Python<'py>, qs: PyReadonlyArray3<'py, f64>) -> Bound<'py, PyArray2<f64>> {
    let qs = qs.as_array();
    let k = qs.shape()[0];
    let n = qs.shape()[1];
    let t = qs.shape()[2];

    let mut result = vec![0.0_f64; n * t];

    result
        .par_chunks_mut(t)
        .enumerate()
        .for_each(|(idx_n, row)| {
            let mut prev = 1.0;
            for idx_t in 0..t {
                let mut stay = 1.0;
                for idx_k in 0..k {
                    stay *= 1.0 - qs[[idx_k, idx_n, idx_t]];
                }
                let val = prev * stay;
                row[idx_t] = val;
                prev = val;
            }
        });
    Array2::from_shape_vec((n, t), result)
        .unwrap()
        .into_pyarray(py)
}

#[pyfunction]
fn compute_exit<'py>(
    py: Python<'py>,
    qs: PyReadonlyArray3<'py, f64>,
    inf: PyReadonlyArray2<'py, f64>,
    method_id: i64,
    s: PyReadonlyArray1<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
    j: usize,
) -> Bound<'py, PyArray2<f64>> {
    let qs = qs.as_array();
    let inf = inf.as_array();
    let s = s.as_slice().unwrap();
    let w = w.as_slice().unwrap();

    let k = qs.shape()[0];
    let n = qs.shape()[1];
    let t = qs.shape()[2];

    let mut result = vec![0.0_f64; n * t];

    result
        .par_chunks_mut(t)
        .enumerate()
        .for_each(|(idx_n, row)| {
            for idx_t in 0..t {
                let prev = if idx_t > 0 {
                    inf[[idx_n, idx_t - 1]]
                } else {
                    1.0
                };

                if method_id == 0 {
                    // udd: reparto por cuadratura
                    row[idx_t] = prev * udd_share(&qs, idx_n, idx_t, j, k, s, w);
                } else {
                    // cf: salidas totales del período, repartidas por fuerza constante
                    let mut stay = 1.0;
                    for m in 0..k {
                        stay *= 1.0 - qs[[m, idx_n, idx_t]];
                    }
                    let exits = 1.0 - stay;
                    row[idx_t] = prev * exits * cf_share(&qs, idx_n, idx_t, j, k);
                }
            }
        });
    Array2::from_shape_vec((n, t), result)
        .unwrap()
        .into_pyarray(py)
}

#[pymodule]
fn _probabilities_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(compute_inf, m)?)?;
    m.add_function(wrap_pyfunction!(compute_exit, m)?)?;
    Ok(())
}
