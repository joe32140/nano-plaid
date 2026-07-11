# the kernel ladder

`nanoplaid.py` scores binary documents through numpy: unpack the sign bits,
one BLAS matmul, done. That is the right way to *understand* the math and the
wrong way to *run* it — the whole point of 1-bit documents is that you never
have to decompress them. This crate reimplements that single inner loop
(int8 query × packed-1-bit docs MaxSim) as a ladder, each rung one idea
faster, all returning **bit-identical** scores:

| rung | idea | µs/doc (M4) | vs rung 1 |
|------|------|------------:|----------:|
| 1 `maxsim_f32` | scalar float reference | 72.6 | 1.0× |
| 2 `maxsim_scalar` | the `2P − T` identity, one branch per bit | 174.5 | 0.42× |
| 3 `maxsim_autovec` | branchless masks → LLVM autovectorizes | 82.3 | 0.88× |
| 4 `maxsim_neon128` | fused doc-token-outer NEON SDOT | 1.89 | 38.5× |
| 5 `maxsim_smmla128` | fused SMMLA — half the instructions | 1.90 | 38.2× |

(32-token query × 2000 docs × 80 tokens, dim=128, Apple M4,
`cargo run --release --example bench`.)

**On x86_64** rung 4 is `maxsim_avx2_sad128` (AVX2 masked-SAD) instead of NEON
SDOT — same fused doc-token-outer idea, same bit-identical scores, AVX2-only so
it runs on essentially any Linux/x86 machine. The bench prints whichever fused
kernel your CPU supports; the dispatcher (`maxsim`) picks it automatically, so
`eval.py --backend rust` accelerates binary scoring on x86 too.

Read the table before the code: **the algebraic identity is not the speedup.**
Rung 2 is *slower* than the float loop it replaces — branching per bit costs
more than multiplying floats. The identity's real contribution is making the
problem *expressible* as integer ops a vector unit is good at; rungs 3–4 are
where that gets cashed in, and almost all of it comes from rung 4's two ideas:
a hardware dot-product instruction, and inverting the loop so each doc token
is bit-expanded once in registers and amortized over all query tokens.

**Rung 5 is the counterexample that proves you have to measure.** SMMLA is a
matrix instruction: it does 32 int8 MACs where SDOT does 16, so on paper it
should halve the instruction count and run ~2× faster. It doesn't — on the M4
it *ties* rung 4, because Apple's cores issue SMMLA at about half SDOT's rate.
Half the instructions at half the throughput nets zero. Counting MACs on paper
predicted a 2× win; the only way to know it evaporates is to run it on the
actual microarchitecture.

**Epilogue, measured on CI:** the moment this repo's CI grew a second arm64
platform (ubuntu-24.04-arm, a Neoverse N2), the prediction in the previous
paragraph's original ending — *"it may still win on cores that issue SMMLA at
SDOT's rate"* — came true: **SMMLA 3.46 vs SDOT 4.85 µs/doc, a 1.40× win**
(shared-VM numbers, treat ratios as the signal). The dispatcher now prefers
SMMLA wherever `i8mm` exists — a measured ~1% giveback on the M4 bought a
~40% win on Neoverse/Graviton-class cores. Keeping the "failed" rung, and
benching it on more than one microarchitecture, is the whole lesson.

Indicative per-platform numbers from CI's shared runners (one run, ratios >
absolutes; M4 numbers above are from an idle local machine). µs/doc:

| platform | binary | r4 | r2 | r1 | SMMLA |
|----------|-------:|---:|---:|---:|------:|
| x86_64 AVX2 (ubuntu-latest) | 7.4 | 12.9 | 12.9 | 11.0 | – |
| Apple M1 (macos-latest) | 2.6 | 6.6 | 6.6 | 8.7 | no i8mm |
| Neoverse N2 (ubuntu-24.04-arm) | 4.9 | 8.5 | 8.5 | 9.6 | **3.5** |

The nbits-flat pattern replicates on every microarchitecture — r4 and r2 are
within noise of each other on all three. One platform-specific inversion: on
AVX2, r1 is the *fastest* residual rung (its affine form rides the masked-SAD
machinery, cheaper than the shufb + sign-transfer chain), while on both ARM
cores r1 is the slowest. Same source, three CPUs, three different orderings —
the recurring moral of this crate.

## the math

Quantize the query row to int8 (`q ≈ scale · v`). Doc values are signs
`s ∈ {−1,+1}` stored as bits. Split the dot product over set and unset bits:

    v · s = P − (T − P) = 2P − T,   P = Σ v over set bits,  T = Σ v

`T` is per-query-token, precomputed once. So scoring a doc token needs only a
bit-masked sum of int8s — no decompression. `P` via SDOT is `v · bits` with
`bits ∈ {0,1}`, which is why rung 4 expands bits to 0/1 bytes, not ±1.

## the second ladder: fused residual-4

The repo's original claim was that only the 1-bit scheme deserved a custom
kernel — residual-4's rescore is `decode → BLAS matmul`, and you don't beat
BLAS at float GEMM. True, and beside the point: you beat it by never
materializing the floats it needs. The identity generalizes. A residual token
decodes to `centroid[cid] + weights[codes]`, so

    q · token = q · centroid[cid]  +  Σ_d v_d · weights[code_d]

The centroid term is a lookup into the `[nq, K]` matrix stage 1 already
computed. And because `weights` is ONE 16-entry table shared by every dim,
int8-quantize it once (`nanoplaid.quantize_lut`) and the residual term is an
integer dot between the query row and *table-looked-up* bytes — `2P − T` is
exactly this with nbits=1 and weights `{−1,+1}`. One instruction does 16
lookups in-register: NEON `tbl` / AVX2 `pshufb`, the same instrument FAISS's
4-bit fast-scan and llama.cpp's Q4 kernels are built on. Everything else is
rung 4's fusion, reused: split nibbles → look up 128 weight bytes once per doc
token → SDOT (or `psignb`+`pmaddubsw` on AVX2) against every query row.

The family covers every nbits the codec supports — and the 1-bit member
doesn't even need the table: with weights (w₀, w₁) the residual term is
`(w₁−w₀)·P + w₀·T`, the *affine generalization* of 2P − T, computed by the
binary kernel's machinery verbatim.

| kernel | idea | µs/doc (M4) | vs f32 loop |
|--------|------|------------:|------------:|
| `maxsim_r4_scalar` | the LUT identity, one lookup per value | 93.4 | 0.76× |
| `maxsim_r4_neon128` | fused `tbl` + SDOT (16-entry table) | 4.55 | 15.6× |
| `maxsim_r2_neon128` | fused `tbl` + SDOT (4-entry table) | 4.54 | 15.7× |
| `maxsim_r1_neon128` | affine 2P−T + SDOT (no table) | 5.52 | 12.9× |

(Same benchmark shape as the ladder table; the `*_avx2_128` x86 rungs are
verified bit-identical on CI.) Three lessons in one table. The scalar rung is
again *slower* than the float loop — the identity is never the speedup. The
fused rungs cost ~2.4× the binary kernel — NOT because of bytes, but because
of the float max: the centroid term varies per token, so each (row, token)
score folds to f32 before comparing instead of staying integer to the end.
And the per-doc cost is **flat across nbits** — 64, 32, or 16 bytes of codes
all score in ~4.5–5.5 µs, because the 8-SDOT inner loop and the fold dominate
while the expansion (what nbits changes) is amortized. At bench scale the
bytes were never the kernel's bottleneck; they're the *index's* (85 vs 47 vs
28 MB resident).

End to end this retires the "rust is binary-only" asterisk (full SciFact,
one sitting): residual-4 drops 111 → **7.9 ms (14×)** at a measured −0.0005
NDCG, and residual-2 drops 82 → **8.0 ms (10×)** at +0.0009 (noise) — the LUT
adds int8 error only to the *residual*; the centroid term stays float.

**And one measured negative, the family's most interesting number:**
residual-1 spends *exactly binary's budget* (16 B codes + 4 B centroid id vs
16 B signs + 4 B code) and scores **0.6312 NDCG vs binary's 0.7460** — an
11-point loss at identical bytes. A ±one-quantile nudge off a centroid
reconstructs every token so close to its cluster center that within-cluster
ranking collapses; raw sign bits keep 128 independent directions. Same
budget, different information — the codec you pick matters more than the
bytes you spend. (It's also the *slowest* fused rung: plane expansion plus
the float fold, with none of binary's integer-max shortcut.)

## how to not fool yourself (field notes)

All three of these happened while writing these kernels' production twin:

- **Dead-code elimination.** Benchmark results are discarded, so LLVM deleted
  an inlined kernel's inner loop entirely and reported a 43× "speedup" that
  was a no-op. Every timing in `examples/bench.rs` goes through
  `std::hint::black_box`. If a number looks too good, read the disassembly.
- **Rosetta.** An Apple Silicon Mac with an x86_64 Rust toolchain will
  silently build x86_64 binaries and emulate them — ~7× slower, and the NEON
  rung never dispatches. Zero `sdot` instructions in `objdump` was the tell.
  Check `rustup show`; build with `--target aarch64-apple-darwin` if needed.
- **Compile-time features ≠ runtime features.** `sdot` on a core without the
  `dotprod` extension is a SIGILL, not a wrong answer. Dispatch checks
  `is_aarch64_feature_detected!` at runtime; this is why production kernels
  can't just build with `-C target-cpu=native` and ship the binary.

## using it from python

The top rung is exposed to numpy through a [pyo3 bridge](src/python.rs)
(feature-gated, so `cargo test` and the bench stay dependency-free):

```bash
maturin develop -m kernels/Cargo.toml --release --features python
python ../eval.py ../data/scifact --backend rust
```

`maxsim_docs(query, payload, lens)` scores a query's candidate documents and
returns one MaxSim per doc — the exact numbers `nanoplaid.score_binary`
produces, computed through this crate's dispatched kernel.
`maxsim_docs_lut(query, codes, cids, cdot_t, lens, lut_values, lut_scale,
nbits)` is the fused residual twin (nbits ∈ {1, 2, 4}), matching
`nanoplaid.score_residual_lut`; `kernels/test_bridge.py` asserts every parity
and runs on every CI platform.
On an Apple Silicon Mac, build with `--target aarch64-apple-darwin` so the
extension is native and the SDOT rung actually runs (see the Rosetta note
above).

## exercises

Each has a production answer in
[next-plaid](https://github.com/lightonai/next-plaid)'s `src/binary.rs`:

1. **Portable SIMD** — rewrite rung 3 with nightly `std::simd`. How close to
   rung 4 can a platform-independent kernel get?
2. **AVX-512 VNNI** — `vpdpbusd` is SDOT's exact x86 cousin, faster than the
   AVX2 rung where a CPU has it. Port `maxsim_vnni128` from next-plaid and add
   it above AVX2 in the dispatch. (Less universal, which is why AVX2 is the
   shipped x86 kernel and this is the exercise.)
3. **Other dims** — rung 4 hardcodes dim=128. Generalize to any multiple of
   32; measure what zero-padding dim=48 to 64 costs vs the fallback.
4. **Store the expansion?** Precompute each doc token's 128 expanded bytes at
   index time and skip `extract_planes_128`. Measure why this *loses* (hint:
   16 B vs 128 B of memory traffic per token).
5. **residual-2** *(answered in-tree — `maxsim_r2_*`/`maxsim_r1_*` now exist).*
   The original question was whether residual-2 could hold its 84 ms numpy
   latency advantage over residual-4 once both were fused. Measured answer:
   no — 8.0 vs 7.9 ms, the fused family is nbits-flat, and residual-2's only
   remaining edge is index size. Follow-up exercise: **close the fold gap.**
   The residual rungs cost ~2.4× the binary kernel almost entirely from the
   per-(row, token) scalar f32 fold. Block 4 doc tokens like `maxsim_neon128`
   does and vectorize the fold (4 accs → one f32x4 max against a gathered
   cdot vector); how much of the 2.4× can you reclaim?
