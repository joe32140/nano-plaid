<p align="center">
  <img src="assets/banner.png" width="100%"
       alt="nano-plaid — late-interaction retrieval (ColBERT + PLAID) in one numpy file. 1-bit documents: 25x smaller, 97.8% of exhaustive NDCG, 3.3x faster with SIMD kernels.">
</p>

<!-- banner.png is exported from assets/banner.svg (the editable source):
     chrome --headless --force-device-scale-factor=2 --window-size=1200,400 \
       --screenshot=assets/banner.png file://$PWD/assets/banner.svg -->

<p align="center">
  <a href="https://github.com/joe32140/nano-plaid/actions/workflows/ci.yml"><img src="https://github.com/joe32140/nano-plaid/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="https://joe32140.github.io/nano-plaid/"><b>🎛️ SIMD school</b> — the interactive course (4 classes)</a>
</p>

# nano-plaid

The simplest complete implementation of **late-interaction retrieval**
(ColBERT scoring + a PLAID-style two-stage index), in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT): one readable numpy file, a
real benchmark, and every design decision measurable on a laptop.

In late-interaction retrieval a query and a document are each a *bag of token
vectors*, scored by **MaxSim** — every query token finds its best-matching
document token, and the matches are summed:

```
score(Q, D) = Σᵢ maxⱼ  Q[i] · D[j]
```

This is much more accurate than single-vector dense retrieval, and much more
expensive to serve naively: a corpus is *hundreds of vectors per document*.
Everything interesting in this repo is about closing that gap — approximating
exhaustive MaxSim without touching every document token in float32.

## the arc

[`nanoplaid.py`](nanoplaid.py) (~400 lines, numpy is the only dependency)
builds up the full system in stages, each one a measurable trade of quality
vs speed vs storage:

0. **Exhaustive MaxSim** — exact, slow, the reference for everything else
1. **k-means centroids** — every corpus token gets a nearest-centroid id
2. **residual compression** — token ≈ centroid + quantized residual
   (nbits ∈ {1, 2, 4}: 7.5×–25× smaller than float32)
3. **binary compression** — token ≈ its sign bits (25× smaller), scored by an
   int8 query through the `2P − T` identity, no decompression
4. **the index** — centroids + codes + inverted file + compressed payload
5. **two-stage search** — probe centroids for candidates, rank them with
   centroid scores alone, exactly rescore only the top `n_full`

## quickstart

```bash
pip install numpy             # that's the whole engine
python nanoplaid.py           # self-test on synthetic data
python eval.py data/toy       # committed multi-domain toy — no downloads, ~30 MB
```

`data/toy` is a small slice of four [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai)
domains (SciFact, NFCorpus, FiQA, Quora), encoded with `lightonai/LateOn-regularized`
and checked in as fp16, so the whole quantization-vs-quality story runs in a
minute with numpy alone. For a full dataset (needs torch) or the Rust backend:

```bash
pip install pylate            # encoder only (torch; GPU/MPS helps)
python encode.py --nano SciFact --out data/scifact   # or --download scifact
python eval.py data/scifact

maturin develop -m kernels/Cargo.toml --release --features python
python eval.py data/scifact --backend rust
```

## results

**The toy, reproducible right now** (`python eval.py data/toy`, NDCG@10 per
domain, `lightonai/LateOn-regularized`, dim 128):

| dataset | exact | residual-4 | residual-2 | residual-1 | binary |
|---------|------:|-----------:|-----------:|-----------:|-------:|
| fiqa | 0.5974 | 0.6111 | 0.6162 | 0.5934 | 0.5945 |
| nfcorpus | 0.3995 | 0.4023 | 0.4038 | 0.3853 | 0.3804 |
| quora | 0.9868 | 0.9842 | 0.9835 | 0.9389 | 0.9772 |
| scifact | 0.7602 | 0.7737 | 0.7034 | 0.6517 | 0.7643 |
| **average** | 0.6860 | 0.6928 | 0.6767 | 0.6423 | 0.6791 |

B/token: exact 512, residual-4 68, residual-2 36, residual-1 20, binary 20.
**Binary keeps 99% of exact NDCG at 1/25th the storage** — but look at the
spread across domains. *Whether a corpus binarizes is domain-dependent*, and
this table is one `--model` flag away from testing your own. Watch residual-1:
it *ties* binary on FiQA and NFCorpus, then loses 4 points on Quora and 11 on
SciFact — the small corpora can hide a codec's failure mode that scale exposes
(see the at-scale table). (Caveats: the subsampled corpora make absolute NDCG
unrepresentative — NFCorpus especially, now balanced across all 50 queries —
and with 50 queries residual can tie or edge out exact on noise. It's for
comparing *schemes*, not for headline numbers.)

**At scale** (full SciFact, 5,183 docs → 1.19M tokens, 300 queries, Apple M4 —
`encode.py --nano SciFact` then `eval.py`):

| scheme | build s | bytes/token | NDCG@10 | retention | p50 ms/query |
|--------|--------:|------------:|--------:|----------:|-------------:|
| exhaustive f32 | – | 512 | 0.7629 | 100% | 20.0 |
| residual nbits=4 | 17 | 68 | 0.7567 | 99.2% | 111 |
| residual nbits=2 | 6.5 | 36 | 0.7340 | 96.2% | 82 |
| residual nbits=1 | 5.5 | 20 | 0.6312 | 82.7% | 64 |
| binary (1-bit) | 3.6 | 20 | 0.7460 | 97.8% | 17.7 |

These are the **baseline** numbers — pure numpy, stage 2 scored the honest
textbook way. The Rust kernels start from this table and cut every scheme to
6–7 ms (next section). Two rows to read twice. residual-4 is the *quality*
headline: 99.2% retention at 85 MB against a 610 MB float corpus. And
residual-1 spends **exactly binary's 20 B/token** yet loses 11 NDCG points to
it — same budget, worse codec. How you spend bits matters more than how many
you spend; sign bits happen to be a very good 1-bit code for this checkpoint.

## profiling (`eval.py --profile`)

`--profile` adds resident index memory, build time, and a per-stage latency
breakdown (probe / rank / rescore). On full SciFact (one sitting, Apple M4;
absolute ms drift a little run to run — compare within the table):

| scheme | index MB | build s | p50 ms | probe/rank/rescore % |
|--------|---------:|--------:|-------:|---------------------:|
| exact | 610 (float corpus) | – | 20.0 | – |
| residual-4 | 85 | 17 | 111 | 1 / 0 / 99 |
| residual-2 | 47 | 6.5 | 82 | 1 / 1 / 99 |
| residual-1 | 28 | 5.5 | 64 | 1 / 1 / 98 |
| binary | 28 | 3.6 | 17.7 | 4 / 3 / 94 |
| residual-4 `--backend rust` | 85 | – | **6.9** | 9 / 6 / 85 |
| residual-2 `--backend rust` | 47 | – | **6.9** | 9 / 6 / 85 |
| residual-1 `--backend rust` | 28 | – | 7.2 | 9 / 6 / 86 |
| binary `--backend rust` | 28 | – | **6.0** | 10 / 7 / 83 |

Two things the breakdown makes obvious. **Memory:** the binary index is 28 MB
against a 610 MB float corpus — 22×. **Where the time goes:** thanks to
centroid pruning (stage 1.5), the candidate set is small, so *exact rescore
dominates* — 82–99% of the query — exactly the shape a real product profile
has. That also means the Rust kernels pay for themselves: they attack the
dominant cost. The lesson is the ordering: the SIMD kernel was worthless until
pruning made rescore the bottleneck; profile first, optimize the tall bar.

### baselines: where each speedup is measured from

Every scheme's baseline is the numpy engine itself — same pipeline, same
index, stage 2 scored the straightforward way. The Rust kernels change *how*
the same math executes, never *what* it computes:

| scheme | numpy baseline (stage 2) | p50 | fused kernel | p50 | speedup | NDCG: numpy → rust |
|--------|--------------------------|----:|--------------|----:|--------:|:-------------------|
| binary | unpack bits → f32 GEMM | 17.7 | `2P−T`, SDOT/AVX2-SAD | **6.0** | 3.0× | 0.7460 → 0.7460 |
| residual-4 | decode floats → f32 GEMM | 111 | LUT (`tbl`/`pshufb`) + SDOT, vec fold | **6.9** | 16× | 0.7567 → 0.7562 |
| residual-2 | decode floats → f32 GEMM | 82 | LUT + SDOT, vec fold | **6.9** | 12× | 0.7340 → 0.7349 |
| residual-1 | decode floats → f32 GEMM | 64 | affine `2P−T`, vec fold | 7.2 | 9× | 0.6312 → 0.6315 |

The NDCG column is the **non-regression check**: binary is bit-identical by
construction; the residual rungs differ only by the LUT's int8 rounding
(≤ ±0.0009, within query noise), and `kernels/test_bridge.py` pins every
kernel to the numpy spec on every CI platform. Note residual-1's low score is
already in the *baseline* column — the quality loss is a property of the
1-bit-residual **codec** at this scale, not of the kernels (on the toy sets it
ties binary on FiQA and NFCorpus).

Two more honest readings. **The 16× is partly a numpy tax:** residual-4's
111 ms baseline is ~96% decode machinery (unpack bits, index the table,
materialize floats) and only ~4 ms of actual GEMM; a compiled decode → GEMM
(what next-plaid does today) would sit near 15–25 ms, so the win a production
port should expect is 2–3×, like binary's. That prediction has since been
paid: the next-plaid port (branch `feat/asymmetric-lut-residual`) measures
the fused path at **2.2–6.3×** against its own compiled decompress → GEMM on
real corpus shapes across x86 AVX2 / Neoverse / Apple M1 — and its phase
profiler shows decompression is 65–84% of that float path, which is the
entire win (you don't out-multiply BLAS; you stop feeding it). Quality on
9 real model×dataset cells: |ΔNDCG@10| ≤ 0.0021, 95% paired-bootstrap CIs
essentially inside ±0.005, and the three cells where a CI excludes zero all
favor the int8 path. One subtlety this toy engine doesn't have: next-plaid's
decompress **renormalizes** each reconstructed token, so the asymmetric port
must score against cached per-token `1/‖·‖` norms — skip that and nbits=1
loses up to 0.17 NDCG@10 on long-query corpora. Speedups are also
baseline-relative in a way any port should disclose: the same binary kernel
that is 13–22× against decompress+GEMM is ~3–4× against raw-float + vendor
BLAS and ~1× against Apple's AMX on raw floats — name the baseline, always. **The history:** the first release
said `--backend rust` was binary-only, "because residual rescore is a
`decode → BLAS matmul` and BLAS is already the fast path a hand kernel can't
beat." True — and beside the point. You don't out-multiply BLAS; you stop
feeding it: the [fused residual kernels](kernels/README.md) score the packed
codes *directly* — an in-register table lookup (NEON `tbl` / AVX2 `pshufb`)
replaces decompression, and the centroid half of every dot product is a lookup
into the matrix stage 1 already computed. The binary `2P − T` trick is the
1-bit special case of this LUT identity. The fused family is nearly
*nbits-flat* in latency (6.9 / 6.9 / 7.2 ms): the integer dot-product core and
the float max — which every nbits shares — dominate, while the payload bytes
(what nbits changes) don't; [Class 04](https://joe32140.github.io/nano-plaid/class4.html)
walks the cost breakdown. That's also why residual-1 buys no speed with its
smaller codes, while its quality collapse makes it a measured negative result
worth reading, not running. (That shared float max was itself the residual
kernel's tall pole: vectorizing the fold — the transferable trick
[mixedbread-ai/maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu) applies
to a plain float GEMM — made every rung ~2.1× faster in isolation and is what
these shipped p50s already use; [kernels/README](kernels/README.md) has the
head-to-head.)

**The knob that matters is `n_full`** — how many candidates get exact-rescored.
Since rescore dominates, it's the recall/latency dial (binary, SciFact):

| `n_full` | 128 | 256 | 512 | 1024 | 2048 |
|----------|----:|----:|----:|-----:|-----:|
| NDCG@10 | 0.642 | 0.685 | 0.726 | 0.746 | 0.750 |
| p50 ms | 3.2 | 5.3 | 9.4 | 17.7 | 34.7 |

`n_probe` (centroids probed per query token), by contrast, barely moves either —
the candidate scoring is cheap regardless, so 2–4 is plenty.

Two honest observations, both of which are the point of the repo:

- **Quality:** sign bits keep 98.5% of NDCG at 1/25th the storage — *for this
  checkpoint*. Whether a model binarizes is a property of the checkpoint and
  its dimensionality, not of the codec: the same pipeline on a dim=48 edge
  model collapses to ~6% retention, and a regularized vs unregularized
  training run of the same model differs by half a point. This is the single
  most interesting research knob here, and it is one `encode.py --model` flag
  away from your own experiments.
- **Speed:** in pure numpy on a 5K-doc corpus, the "smart" two-stage index
  *loses* to one exhaustive BLAS matmul. Clever indexing has real constant
  costs (gathers, decode, python), and BLAS is very hard to beat from
  interpreted code — which is exactly why production PLAID engines are
  systems projects. The same binary pipeline with real SIMD kernels
  ([next-plaid](https://github.com/lightonai/next-plaid)) serves this corpus
  at ~5.6ms/query vs ~21ms for its residual path. See [`kernels/`](kernels/)
  for where that speed actually comes from.

## the kernel ladder (`kernels/`)

The one inner loop that matters — int8 query × packed 1-bit docs MaxSim —
rebuilt in Rust as a ladder of rungs, from scalar reference to a fused SIMD
kernel (NEON SDOT on Apple Silicon, AVX2 masked-SAD or AVX-512 VNNI on
x86/Linux), all bit-identical,
benchmarked on the way up (spoiler: the algebraic identity alone makes things
*slower*; the memory layout and loop order are the speedup — 38× by rung 4).
Rung 5 swaps SDOT for the denser SMMLA matrix instruction, which *should* be
2× and instead ties on the M4 — a measured lesson in why you don't trust a MAC
count without running it. The epilogue is better: CI's Neoverse arm64 runner
later proved SMMLA **1.40× faster** there, so the "failed" rung now wins the
dispatch on `i8mm` cores — keep your negative results. Rung 6 is the x86 server
upgrade: AVX-512 **VNNI** (`vpdpbusd`, SDOT's exact x86 twin) runs the binary
kernel **2.09×** over AVX2 where a CPU has it — but only ~1.1× for the residual
rungs, a clean lesson in when doubling the vector width actually pays (and, since
GitHub's runners only *sometimes* have AVX-512, how to verify it deterministically
under Intel SDE). Plus field notes on the three ways microbenchmarks lied to us
while building the production version. See [kernels/README.md](kernels/README.md).

There is a **second ladder** for the residual schemes: the same `2P − T` idea
generalized to a shared weight table (one in-register `tbl`/`pshufb` lookup
replaces decompression), covering nbits ∈ {1, 2, 4} — the 1-bit rung needs no
table, just the affine form `(w₁−w₀)·P + w₀·T`. It retires this repo's
original "you can't beat the BLAS path" claim — see
[the profiling section](#profiling-evalpy---profile) and
[kernels/README.md](kernels/README.md).

A thin [pyo3 bridge](kernels/src/python.rs) exposes the dispatched kernels to
numpy, so `eval.py --backend rust` scores stage-2 through them: binary via
SDOT/AVX2-SAD (identical NDCG@10, 0.7460) and the residual family via the
fused LUT kernels (−0.0005 NDCG on residual-4). Because centroid pruning makes
rescore the dominant cost, swapping that one stage cuts SciFact end-to-end p50
to 6–7 ms for every scheme — the kernels attack the tall bar instead of a
rounding error. `kernels/test_bridge.py` pins the bridge to the numpy spec on
every CI platform (x86 AVX2, Apple NEON, Neoverse NEON).

## relationship to next-plaid

[next-plaid](https://github.com/lightonai/next-plaid) is the production
implementation: runtime CPU dispatch (AVX-512 VNNI / AVX2 / NEON), on-disk
formats, incremental updates, filtering, a CLI. nano-plaid is the textbook:
if you want to *change* the algorithm — a new compression scheme, a new
candidate generator, a different scoring identity — start here, measure with
`eval.py`, and port to next-plaid when it wins. The binary quantization
scheme here mirrors the one contributed to next-plaid in
[PR #155](https://github.com/lightonai/next-plaid/pull/155); the fused
residual LUT kernels (111 → 6.9 ms here for nbits=4) are the next porting
candidate — next-plaid's residual rescore is still `decompress → GEMM`.

## files

```
nanoplaid.py   the entire index + search engine (numpy only)
encode.py      BEIR / NanoBEIR + ColBERT model -> token-embedding bundle (pylate)
make_toy.py    builds the committed data/toy slice (pylate; run once, not needed to use)
eval.py        NDCG@10 / latency / storage per scheme, one bundle or a directory of them
data/toy/      committed fp16 4-domain NanoBEIR slice — the zero-download demo
kernels/       the Rust SIMD ladder + optional pyo3 bridge
pyproject.toml maturin config for the optional Rust extension
```

MIT license.
