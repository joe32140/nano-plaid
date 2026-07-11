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
absolutes; M4 numbers above are from an idle local machine). µs/doc, showing
each residual rung's `tbl`+SDOT baseline → its shipped `_vfold`:

| platform | binary | r4 → vfold | r2 → vfold | r1 → vfold | SMMLA |
|----------|-------:|-----------:|-----------:|-----------:|------:|
| x86_64 AVX2 (ubuntu-latest) | 7.4 | 12.9 → **7.9** | 12.9 → **7.9** | 11.0 → **5.5** | – |
| Apple M1 (macos-latest) | 2.5 | 6.5 → **3.1** | 6.6 → **3.3** | 8.3 → **3.7** | no i8mm |
| Neoverse N2 (ubuntu-24.04-arm) | 4.8 | 8.5 → **5.6** | 8.5 → **5.7** | 9.5 → **6.3** | **3.5** |

The vectorized fold wins on every microarchitecture — 1.5× (Neoverse) to 2.2×
(M1) — and the family stays nbits-flat after it. Two platform-specific
inversions survive into the vfold column. On AVX2 the fold helps r1 *most*
(2.0×), so **r1 vfold (5.5) is the fastest residual rung and beats even the
binary kernel (7.4)** — its affine form rides the cheap masked-SAD path, which
on x86 undercuts both the shufb chain and binary's own SAD loop. On both ARM
cores r1 stays the slowest residual rung. Same source, three CPUs, different
orderings — the recurring moral of this crate.

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
| `maxsim_r4_neon128` | fused `tbl` + SDOT (16-entry table) | 4.54 | 15.6× |
| `maxsim_r4_neon128_vfold` | + the fold vectorized | **2.19** | **32.3×** |
| `maxsim_r2_neon128` | fused `tbl` + SDOT (4-entry table) | 4.54 | 15.7× |
| `maxsim_r2_neon128_vfold` | + the fold vectorized | **2.17** | **32.5×** |
| `maxsim_r1_neon128` | affine 2P−T + SDOT (no table) | 5.48 | 12.9× |
| `maxsim_r1_neon128_vfold` | + the fold vectorized | **2.35** | **29.9×** |

(Same benchmark shape as the ladder table; the `*_avx2_128` x86 rungs are
verified bit-identical on CI.) Three lessons in one table. The scalar rung is
again *slower* than the float loop — the identity is never the speedup. The
`tbl`+SDOT rungs cost ~2.4× the binary kernel — NOT because of bytes, but
because of the **float max**: the centroid term varies per token, so each
(row, token) score folds to f32 before comparing instead of staying integer
to the end. And the per-doc cost is **flat across nbits** — 64, 32, or 16
bytes of codes all score in ~4.5–5.5 µs, because the 8-SDOT inner loop and the
fold dominate while the expansion (what nbits changes) is amortized. At bench
scale the bytes were never the kernel's bottleneck.

**The `_vfold` rungs prove where that 2.4× lived.** The only change is that the
per-(row, token) f32 fold — quantize `acc` to float, `sqw·acc + crow`, compare
against the running max — moves out of the scalar per-row loop into
`fold_block`, which does four rows per `vmaxq_f32` (eight per `_mm256_max_ps`
on AVX2). Nothing about the `tbl`/SDOT compute changes, and the scores stay
bit-identical (the parity test pins it). It **~2.1× every rung** (4.54 → 2.19
µs on residual-4) and collapses the gap to the binary kernel from 2.4× to
~1.15×. So the fold *was* the cost, not the payload bytes or the expansion —
a hypothesis the crate could only settle by building the counterfactual and
timing it. This is the transferable half of MaxSim: the same vectorized
max-reduction [mixedbread-ai/maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu)
bolts onto a plain float GEMM. We borrowed it to shore up the residual
family's one weak spot; it carries past this repo unchanged (it touches
neither the LUT identity nor the query layout). The shipped dispatcher
(`maxsim_r{4,2,1}`) now routes here, with the scalar-fold rung kept exposed
for the head-to-head — unlike SMMLA, the win isn't tied to an instruction's
issue rate (a branch-free pass plus a wide `max`), so it carries across
microarchitectures rather than needing a per-core dispatch flip.

End to end this retires the "rust is binary-only" asterisk (full SciFact,
one sitting): with the vectorized fold, residual-4 drops 111 → **6.9 ms
(16×)** at a measured −0.0005 NDCG, residual-2 → **6.9 ms**, and residual-1
→ **7.2 ms** — all within ~15% of the binary path's 6.0 ms. The kernel's 2.1×
becomes ~1.15× at the query level: Amdahl, since rescore is 85% of the query
and the gather/probe/rank around it are shared and untouched. The point of
the fold work was never the end-to-end number — it was proving *which* part of
the kernel was the tall pole.

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
5. **The fold gap** *(answered in-tree — `maxsim_r{4,2,1}_*_vfold` now exist).*
   Two questions, both settled by measurement. First: could residual-2 hold
   its numpy latency edge over residual-4 once both were fused? No — the fused
   family is nbits-flat (6.9 vs 6.9 ms), and residual-2's only remaining edge
   is index size. Second: the `tbl`+SDOT rungs cost ~2.4× the binary kernel —
   was that the per-(row, token) scalar f32 fold, or the payload bytes? Moving
   the fold into a vectorized `fold_block` (four rows per `vmaxq_f32`) made it
   ~2.1× faster with bit-identical scores, collapsing the gap to ~1.15×: the
   fold was the cost. Third, the follow-up — *answered in-tree, and the answer
   is microarchitecture-dependent, which is the real lesson.* The `_vfold`
   kernels still run one `vaddvq_s32` horizontal reduce per row; the `_tr`
   kernels remove it, folding four rows with a `vpaddq_s32` transpose-reduce so
   the four dot products land in one register (no scalar round-trip; r1 also
   moves its affine `dw·P + w0·T` into per-lane integers). Bit-identical, and
   measured on three cores (µs/doc, vfold → tr):

   | rung | Apple M4 (local) | Apple M1 (CI) | Neoverse N2 (CI) |
   |------|:----------------:|:-------------:|:----------------:|
   | r4 | 2.19 → 2.11 | 3.02 → **2.67** | 5.63 → 5.62 |
   | r2 | 2.15 → 2.11 | 3.05 → **2.60** | 5.68 → 5.65 |
   | r1 | 2.33 → 2.21 | 3.63 → **2.69** | 6.34 → **6.09** |
   | | wash (~3%) | **12–26% faster** | wash, r1 ~4% |

   So whether the per-row reduce is a real cost depends on the core: the wide
   M4 and the Neoverse N2 hide the `vaddvq` under SDOT latency; the narrower M1
   does not, and removing it there is a genuine win — biggest on **r1 (26%)**,
   which carried the most per-row scalar work (a reduce *and* the affine), and
   which was the M1's slowest rung under vfold. tr not only speeds the family
   up there, it *restores the nbits-flat line* vfold had broken (M1 tr: all
   ~2.6). This flips the "bench on more than one microarchitecture" moral the
   OTHER way from SMMLA (negative on M4, positive on Neoverse; here a wash on
   M4/Neoverse, a win on the M1). Non-negative everywhere, so the whole family
   ships it: `maxsim_r{4,2,1}` dispatch tr → vfold → scalar on NEON (r2 shares
   r4's `tbl` compute; r1 applies its affine per lane before the shared
   `fold4`). x86 keeps vfold.
