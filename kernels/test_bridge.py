"""Cross-language parity for the pyo3 bridge: the Rust extension must
reproduce nanoplaid's numpy specs on whatever kernel this CPU dispatches to
(NEON on arm64, AVX2 on x86_64, scalar elsewhere) — this is what CI runs on
every platform after building the wheel.

    python kernels/test_bridge.py

Deterministic, no downloads; needs numpy + the built extension only.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nanoplaid as npl  # noqa: E402
import nanoplaid_kernels as nk  # noqa: E402

rng = np.random.default_rng(3)
nq, dim, K, n_docs = 32, 128, 64, 40
lens = rng.integers(1, 30, n_docs)
total = int(lens.sum())
bounds = np.concatenate([[0], np.cumsum(lens)[:-1]])
q = rng.standard_normal((nq, dim)).astype(np.float32)
q8 = npl.quantize_query_i8(q)

# --- binary: 2P − T ---------------------------------------------------------
packed = npl.binarize(rng.standard_normal((total, dim)).astype(np.float32))
sim = npl.score_binary(q8, packed, dim)
ref = np.maximum.reduceat(sim, bounds, axis=1).sum(axis=0)
got = nk.maxsim_docs(q, packed, lens.astype(np.int64))
err = np.abs(ref - got).max()
assert err < 1e-3, f"binary bridge drifted from numpy spec: {err}"
print(f"binary  parity ok  (max |numpy - rust| = {err:.2e})")

# --- fused residual-4: the LUT identity --------------------------------------
codes = rng.integers(0, 256, (total, dim // 2)).astype(np.uint8)
cids = rng.integers(0, K, total).astype(np.uint32)
cdot = (rng.standard_normal((nq, K)) * 2).astype(np.float32)
weights = np.sort(rng.standard_normal(16).astype(np.float32) * 0.03)
lut = npl.quantize_lut(npl.ResidualCodec(4, np.zeros(15, np.float32), weights))

sim = npl.score_residual_lut(q8, lut, codes, cids, cdot, dim, 4)
ref = np.maximum.reduceat(sim, bounds, axis=1).sum(axis=0)
got = nk.maxsim_docs_r4(q, codes, cids, np.ascontiguousarray(cdot.T),
                        lens.astype(np.int64),
                        np.ascontiguousarray(lut.values, np.int8),
                        float(lut.scale))
err = np.abs(ref - got).max()
assert err < 1e-4, f"residual bridge drifted from numpy spec: {err}"
print(f"residual parity ok (max |numpy - rust| = {err:.2e})")
