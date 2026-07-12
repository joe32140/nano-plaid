# Baselines: compress+fused vs decompress+GEMM

`compare_mixedbread.py` pits this repo's compressed fused MaxSim kernels
against [`mixedbread-ai/maxsim-cpu`](https://github.com/mixedbread-ai/maxsim-cpu),
the two honest ends of the same line:

- **mixedbread — decompress + GEMM.** Full `f32` document embeddings, a batched
  BLAS `sgemm`, and a hand-vectorized max-fold. This is the upstream PyPI wheel
  (`pip install maxsim-cpu`), their Rust + Accelerate/libxsmm code, called
  unmodified — not a reimplementation. (It ships as a pyo3 `cdylib`, so it is
  only reachable from Python; there is no Rust crate to link.)
- **ours — compress + fused kernel.** The document is packed sign bits (binary,
  16 B/token) or residual codes (32/64 B/token), scored with **no
  decompression** through `nanoplaid_kernels`, the pyo3 bridge over the same
  kernels the [kernel class](../docs/class4.html) dissects.

Both score every query against every document **exhaustively** (no ANN, no
candidate cap), so the measured number is the scoring kernel and nothing else.
We report the three axes that actually differ: per-doc latency, bytes/token,
and NDCG@10 against the real qrels — mixedbread's exact `f32` score is the
quality *ceiling*; ours pays compression error for 8–32× less memory.

## Result (SciFact, 300 queries × 5,183 docs, 1.19M doc tokens, dim=128)

Apple M-series, both engines **single-threaded** on Accelerate (`RAYON_NUM_THREADS=1`),
so this isolates kernel efficiency — both are embarrassingly parallel across
docs, so the single-core ratio is the apples-to-apples number.

| engine | B/tok | vs f32 | µs/doc | vs GEMM | NDCG@10 | retention |
|---|--:|--:|--:|--:|--:|--:|
| mixedbread f32 GEMM | 512 | 1× | 15.30 | 1.00× | 0.7629 | 100% |
| **ours: binary 1-bit fused** | 16 | **32×** | **3.77** | **4.06×** | 0.7513 | 98.5% |
| ours: residual-4 fused | 64 | 8× | 4.22 | 3.62× | 0.7569 | 99.2% |
| ours: residual-2 fused | 32 | 16× | 4.19 | 3.65× | 0.7372 | 96.6% |

The pragmatic takeaway: against a well-optimized `f32` GEMM engine, scoring the
compressed document directly is **3.6–4.1× faster at 8–32× less memory**, giving
up 1–3 NDCG points. The GEMM has to touch 32× the bytes and materialize a score
matrix; the fused kernel expands one doc token in registers and never leaves
integer-land until the final scale. Memory is the point — at 16 B/token a corpus
that needs 32 GB of `f32` fits in 1 GB.

## Run it

```bash
# native arm64 bridge (avoid the Rosetta trap that would bench our fallback):
CARGO_BUILD_TARGET=aarch64-apple-darwin pip install maxsim-cpu numpy .
python bench/compare_mixedbread.py data/scifact              # full
python bench/compare_mixedbread.py data/scifact --docs 400 --queries 40  # quick
```

On x86 drop the `CARGO_BUILD_TARGET`; the dispatcher picks AVX-512 VNNI / AVX2
for our side and mixedbread uses libxsmm.
