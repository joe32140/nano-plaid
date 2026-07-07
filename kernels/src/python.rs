//! Python bridge (feature = "python"): exposes the kernel ladder to numpy so
//! `eval.py --backend rust` scores the binary stage-2 with the same Rust code
//! `examples/bench.rs` benchmarks. Built with maturin (see repo pyproject.toml).

// The #[pyfunction] wrapper generates `.into()` on the PyResult return; clippy
// flags it as a useless conversion but the span is macro code, so the allow
// must sit at module scope, not on the fn.
#![allow(clippy::useless_conversion)]

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{maxsim, quantize_query_i8};

/// Per-document binary MaxSim for one query against packed candidate rows.
///
/// - `query`: `[nq, dim]` f32, C-contiguous
/// - `payload`: `[sum(lens), dim/8]` u8 packed sign bits, C-contiguous
/// - `lens`: `[n_docs]` i64, tokens per candidate doc
///
/// Returns `[n_docs]` f32 — one MaxSim score per doc, identical to
/// `nanoplaid.score_binary` followed by the per-doc max-reduce and sum, but
/// computed through the dispatched Rust kernel (NEON SDOT where available).
/// The query is int8-quantized here exactly as `nanoplaid.quantize_query_i8`
/// does, so scores match the numpy path up to f32 rounding.
#[pyfunction]
fn maxsim_docs<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    payload: PyReadonlyArray2<'py, u8>,
    lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = query.shape()[1];
    let pd = dim / 8;
    if dim == 0 || dim % 8 != 0 {
        return Err(PyValueError::new_err(
            "dim must be a positive multiple of 8",
        ));
    }
    if payload.shape()[1] != pd {
        return Err(PyValueError::new_err(
            "payload column count must equal dim/8",
        ));
    }
    let q = query
        .as_slice()
        .map_err(|_| PyValueError::new_err("query must be C-contiguous"))?;
    let payload = payload
        .as_slice()
        .map_err(|_| PyValueError::new_err("payload must be C-contiguous"))?;
    let lens = lens
        .as_slice()
        .map_err(|_| PyValueError::new_err("lens must be C-contiguous"))?;

    if lens.iter().any(|&n| n <= 0) {
        return Err(PyValueError::new_err("every doc length must be positive"));
    }
    let total: usize = lens.iter().map(|&n| n as usize).sum();
    if total * pd != payload.len() {
        return Err(PyValueError::new_err("payload rows must equal sum(lens)"));
    }

    // Compute off the GIL: the borrowed slices outlive the closure, and the
    // rescore over all candidates is exactly the work worth parallelizing.
    let out = py.allow_threads(|| {
        let q8 = quantize_query_i8(q, dim);
        let mut out = Vec::with_capacity(lens.len());
        let mut off = 0usize;
        for &n in lens {
            let n = n as usize;
            out.push(maxsim(&q8, &payload[off * pd..(off + n) * pd], dim));
            off += n;
        }
        out
    });
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pymodule]
fn nanoplaid_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim_docs, m)?)?;
    Ok(())
}
