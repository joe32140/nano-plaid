# Baselines: compress+fused vs decompress+GEMM (per residual route)

`compare_gemm.py` answers one question for each residual route (nbits 4/2/1):
the codes are stored the same — so how you *score* them is the whole game.

- **baseline GEMM** — decode codes → f32 tokens, then a per-doc BLAS `sgemm` +
  max-fold. This is **next-plaid's `maxsim_score`** (per-doc `query.dot(doc.T)`
  + SIMD max), run here through the same Accelerate BLAS its Rust path uses.
  Dependency-free — the default reference.
- **ours (fused)** — score the codes directly through `nanoplaid_kernels`, with
  **no decode**. Our optimization.
- **mixedbread** *(optional)* — if [`maxsim-cpu`](https://github.com/mixedbread-ai/maxsim-cpu)
  is installed, also score the decoded f32 through their batched `sgemm`+fold
  (upstream wheel, unmodified). Skipped cleanly if absent — it ships as a pyo3
  `cdylib`, so there is no Rust crate to link and it's Python-only.

The GEMM rows must expand every token to **512 B** of f32 to score; ours stays
at the stored 64/32/16 B and never decodes (its decode time is even handed to
the GEMM rows for free). Scoring is exhaustive (no ANN), so the number is the
kernel.

## Result (SciFact, 300 queries × 5,183 docs, 1.19M tokens, dim=128)

Apple M-series, **single-thread** on Accelerate. `store` = stored bytes/token;
`score` = bytes touched to score a token.

f32 ceiling (exact MaxSim on the *uncompressed* embeddings): **NDCG@10 = 0.7629**

| route | method | store | score | µs/doc | vs base | NDCG@10 | retention |
|---|---|--:|--:|--:|--:|--:|--:|
| **r4** | baseline GEMM (next-plaid) | 64 B | 512 B | 6.80 | 1.00× | 0.7599 | 99.6% |
| | mixedbread GEMM *(opt.)* | 64 B | 512 B | 14.66 | 0.46× | 0.7599 | 99.6% |
| | **ours: fused on codes** | 64 B | **64 B** | **3.93** | **1.73×** | 0.7569 | 99.2% |
| **r2** | baseline GEMM (next-plaid) | 32 B | 512 B | 6.87 | 1.00× | 0.7356 | 96.4% |
| | mixedbread GEMM *(opt.)* | 32 B | 512 B | 15.12 | 0.45× | 0.7356 | 96.4% |
| | **ours: fused on codes** | 32 B | **32 B** | **3.93** | **1.75×** | 0.7372 | 96.6% |
| **r1** | baseline GEMM (next-plaid) | 16 B | 512 B | 6.90 | 1.00× | 0.6138 | 80.5% |
| | mixedbread GEMM *(opt.)* | 16 B | 512 B | 15.26 | 0.45× | 0.6138 | 80.5% |
| | **ours: fused on codes** | 16 B | **16 B** | **4.20** | **1.64×** | 0.6137 | 80.4% |

**Our fused kernel beats the next-plaid GEMM baseline 1.6–1.75× per route**, at
8–32× less scoring memory, needing no decode, at ~identical NDCG (our int8
ranking tracks the f32 reconstruction to a rounding error — 0.7569 vs 0.7599 at
r4). r1's 80% retention is the 1-bit residual *codec*, not the kernel; r4 keeps
99%.

## Why the GEMM baseline can't be beaten by a better GEMM

Everyone doing f32 shares an irreducible floor — the `sgemm` itself:

| what's timed (per doc) | µs/doc |
|---|--:|
| Accelerate `sgemm` only (`qf @ tok.T`) | 5.1 |
| + numpy `.max(axis=1).sum()` fold | 6.4 |
| **ours: fused on codes** | **3.9** |

The max-fold is `1/dim ≈ 0.8%` of the GEMM's FLOPs, so optimizing it (what an
"optimized GEMM" does) moves you *toward* the 5.1 floor, never below it. Our
kernel lands at 3.9 — **below the floor** — because it never runs the f32 GEMM:
it scores an int8 view of the codes with a fused dot that's cheaper per MAC and
never materializes the reconstruction. You don't beat BLAS by out-BLAS-ing
BLAS; you beat it by removing the GEMM.

This is also why the optional **mixedbread** column, though a genuinely
optimized GEMM (batched sgemm + hand-vectorized fold), doesn't help here.
Single-threaded on variable-length docs its per-call array-marshalling overhead
makes it *slower* than the plain per-doc baseline; even at its best case
(uniform contiguous API, all 10 cores — its design point) it reaches 6.0 µs/doc,
right at the `sgemm` floor and still 1.5× behind our single-core 3.9. Its win is
multicore-batch throughput over PyTorch, not per-core latency, and ours
parallelizes across docs the same way.

## Run it

```bash
CARGO_BUILD_TARGET=aarch64-apple-darwin pip install numpy .   # our bridge, arm64
pip install maxsim-cpu                                        # optional column
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python bench/compare_gemm.py data/scifact
```

`CARGO_BUILD_TARGET` keeps the bridge native arm64 — under Rosetta the NEON
kernels compile out and our side would bench the autovec fallback. On x86 drop
it; our side dispatches AVX-512 VNNI / AVX2.
