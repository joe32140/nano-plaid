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

use crate::{maxsim, maxsim_r4, quantize_query_i8, LutI8};

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
/// does (same scale, same round-to-even), so the integer scores are identical;
/// the returned f32 can differ only in the last few ulps, from the order the
/// per-query-token maxes are summed.
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
    if query.shape()[0] == 0 {
        // Guard here, not in quantize_query_i8: its debug assert would abort
        // inside the kernel rather than raising a clean Python error.
        return Err(PyValueError::new_err("query must have at least one token"));
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

    // Score under the GIL. The work is microseconds and eval.py is
    // single-threaded, so releasing it with py.allow_threads would buy nothing
    // here while adding a real soundness caveat: PyReadonlyArray does not lock
    // the numpy buffer, so another Python thread could mutate it mid-read.
    let q8 = quantize_query_i8(q, dim);
    let mut out = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &n in lens {
        let n = n as usize;
        out.push(maxsim(&q8, &payload[off * pd..(off + n) * pd], dim));
        off += n;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Per-document fused residual-4 MaxSim for one query against packed rows.
///
/// - `query`: `[nq, dim]` f32, C-contiguous
/// - `codes`: `[sum(lens), dim/2]` u8 packed 4-bit bucket indices
/// - `cids`: `[sum(lens)]` u32, each token's centroid id
/// - `cdot_t`: `[K, nq]` f32 — the stage-1 `q @ centroids.T` matrix,
///   TRANSPOSED so a token's per-query-row lookups are contiguous
/// - `lens`: `[n_docs]` i64, tokens per candidate doc
/// - `lut_values` / `lut_scale`: `nanoplaid.quantize_lut`'s 16-entry int8
///   table (entries must stay in [-127, 127] — the numpy quantizer clips)
///
/// Returns `[n_docs]` f32, matching `nanoplaid.score_residual_lut` followed by
/// the per-doc max-reduce and sum: identical per-(row, token) f32 scores; the
/// final sum can differ only in the last ulps (numpy sums pairwise).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn maxsim_docs_r4<'py>(
    py: Python<'py>,
    query: PyReadonlyArray2<'py, f32>,
    codes: PyReadonlyArray2<'py, u8>,
    cids: PyReadonlyArray1<'py, u32>,
    cdot_t: PyReadonlyArray2<'py, f32>,
    lens: PyReadonlyArray1<'py, i64>,
    lut_values: PyReadonlyArray1<'py, i8>,
    lut_scale: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = query.shape()[1];
    let nq = query.shape()[0];
    let pd = dim / 2;
    if dim == 0 || dim % 8 != 0 {
        return Err(PyValueError::new_err(
            "dim must be a positive multiple of 8",
        ));
    }
    if nq == 0 {
        return Err(PyValueError::new_err("query must have at least one token"));
    }
    if codes.shape()[1] != pd {
        return Err(PyValueError::new_err(
            "codes column count must equal dim/2 (nbits = 4)",
        ));
    }
    if cdot_t.shape()[1] != nq {
        return Err(PyValueError::new_err(
            "cdot_t must be [n_centroids, nq] (the transposed centroid matrix)",
        ));
    }
    let n_centroids = cdot_t.shape()[0];
    if lut_values.shape()[0] != 16 {
        return Err(PyValueError::new_err("lut_values must have 16 entries"));
    }
    let q = query
        .as_slice()
        .map_err(|_| PyValueError::new_err("query must be C-contiguous"))?;
    let codes = codes
        .as_slice()
        .map_err(|_| PyValueError::new_err("codes must be C-contiguous"))?;
    let cids = cids
        .as_slice()
        .map_err(|_| PyValueError::new_err("cids must be C-contiguous"))?;
    let cdot_t = cdot_t
        .as_slice()
        .map_err(|_| PyValueError::new_err("cdot_t must be C-contiguous"))?;
    let lens = lens
        .as_slice()
        .map_err(|_| PyValueError::new_err("lens must be C-contiguous"))?;
    let lv = lut_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("lut_values must be C-contiguous"))?;

    if lens.iter().any(|&n| n <= 0) {
        return Err(PyValueError::new_err("every doc length must be positive"));
    }
    let total: usize = lens.iter().map(|&n| n as usize).sum();
    if total * pd != codes.len() || total != cids.len() {
        return Err(PyValueError::new_err(
            "codes/cids rows must equal sum(lens)",
        ));
    }
    // The kernels index cdot_t rows through raw pointers; an out-of-range
    // centroid id would be UB there, so it is a hard error here.
    if cids.iter().any(|&c| c as usize >= n_centroids) {
        return Err(PyValueError::new_err("centroid id out of range for cdot_t"));
    }
    if lv.iter().any(|&v| v == i8::MIN) {
        return Err(PyValueError::new_err(
            "lut values must be in [-127, 127] (quantize_lut clips)",
        ));
    }

    let mut values = [0i8; 16];
    values.copy_from_slice(lv);
    let lut = LutI8 {
        values,
        scale: lut_scale,
    };
    let q8 = quantize_query_i8(q, dim);
    let mut out = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &n in lens {
        let n = n as usize;
        out.push(maxsim_r4(
            &q8,
            &lut,
            &codes[off * pd..(off + n) * pd],
            &cids[off..off + n],
            cdot_t,
        ));
        off += n;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pymodule]
fn nanoplaid_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxsim_docs, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_docs_r4, m)?)?;
    Ok(())
}
