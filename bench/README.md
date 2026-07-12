# Baselines: compress+fused vs decompress+GEMM (per residual route)

`compare_mixedbread.py` answers one question for each residual route (nbits
4/2/1): the codes are stored the same — so how you *score* them is the whole
game. Three ways, head to head:

1. **baseline GEMM** — decode codes → f32 tokens, then a plain per-doc BLAS
   `sgemm` + max-fold (numpy). The naive path every system starts with.
2. **mixedbread GEMM** — decode codes → f32 tokens, then
   [`mixedbread-ai/maxsim-cpu`](https://github.com/mixedbread-ai/maxsim-cpu):
   a batched `sgemm` + hand-vectorized fold. Their upstream PyPI wheel,
   unmodified — the *optimized* GEMM. (It ships as a pyo3 `cdylib`, so it is
   only reachable from Python; there is no Rust crate to link.)
3. **ours (fused)** — score the codes directly through `nanoplaid_kernels`,
   **no decode at all**. Our optimization.

(1) and (2) score the identical f32 reconstruction, so their NDCG matches and
only speed differs (that gap is the GEMM optimization). (3) scores an int8 view
of the same codes, trading a hair of NDCG for never materializing f32: (1)/(2)
must expand every token to **512 B** to score; (3) stays at the stored
64/32/16 B. Scoring is exhaustive (no ANN), so the number is the kernel.

## Result (SciFact, 300 queries × 5,183 docs, 1.19M tokens, dim=128)

Apple M-series, **single-thread** on Accelerate (apples-to-apples per core;
every method here parallelizes trivially across docs). `store` = stored
bytes/token; `score` = bytes touched to score a token. Decode time is given to
the GEMM rows **for free** (not timed) — they still lose.

f32 ceiling (mixedbread on the *uncompressed* embeddings): **NDCG@10 = 0.7629**

| route | method | store | score | µs/doc | vs base | NDCG@10 | retention |
|---|---|--:|--:|--:|--:|--:|--:|
| **r4** | baseline GEMM (per-doc) | 64 B | 512 B | 6.80 | 1.00× | 0.7599 | 99.6% |
| | mixedbread GEMM | 64 B | 512 B | 14.66 | 0.46× | 0.7599 | 99.6% |
| | **ours: fused on codes** | 64 B | **64 B** | **3.93** | **1.73×** | 0.7569 | 99.2% |
| **r2** | baseline GEMM (per-doc) | 32 B | 512 B | 6.87 | 1.00× | 0.7356 | 96.4% |
| | mixedbread GEMM | 32 B | 512 B | 15.12 | 0.45× | 0.7356 | 96.4% |
| | **ours: fused on codes** | 32 B | **32 B** | **3.93** | **1.75×** | 0.7372 | 96.6% |
| **r1** | baseline GEMM (per-doc) | 16 B | 512 B | 6.90 | 1.00× | 0.6138 | 80.5% |
| | mixedbread GEMM | 16 B | 512 B | 15.26 | 0.45× | 0.6138 | 80.5% |
| | **ours: fused on codes** | 16 B | **16 B** | **4.20** | **1.64×** | 0.6137 | 80.4% |

Two things fall out:

- **Our fused kernel beats the naive GEMM baseline 1.6–1.75× per route**, at
  8–32× less scoring memory, needing no decode, at ~identical NDCG (our int8
  ranking tracks the f32 reconstruction to within a rounding error — 0.7569 vs
  0.7599 at r4). r1 is the quality floor (80% retention): the 1-bit residual
  codec, not the kernel.
- **mixedbread's GEMM optimization does not help here** — single-threaded on
  variable-length docs it is *slower* than naive per-doc numpy, because its win
  is multicore throughput on uniform batches, not per-core latency.

### Giving mixedbread its best case (fairness)

mixedbread is built for multicore uniform batches, so the single-thread
variable-API row above is its worst case. Measured on the same corpus:

| mixedbread config | µs/doc |
|---|--:|
| variable API, 1 thread (table above) | 15.0 |
| variable API, 10 threads | 11.9 |
| **uniform API (padded), 10 threads** — their fast path | **6.0** |

Even at its genuine best — contiguous uniform layout, all 10 cores, their
optimized `sgemm`+fold — mixedbread lands at 6.0 µs/doc, still **1.5× slower
than our single-core 3.9**, and our kernel parallelizes across docs the same
way. The compressed path wins because it never pays the decode-to-f32 or the
32× memory traffic the GEMM is optimizing *around*.

## Run it

```bash
CARGO_BUILD_TARGET=aarch64-apple-darwin pip install maxsim-cpu numpy .
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python bench/compare_mixedbread.py data/scifact
```

`CARGO_BUILD_TARGET` keeps the bridge native arm64 — under Rosetta the NEON
kernels compile out and our side would bench the autovec fallback (the trap the
kernel class warns about). On x86 drop it; our side dispatches AVX-512 VNNI /
AVX2 and mixedbread uses libxsmm.
