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
actual microarchitecture. (It may still win on cores that issue SMMLA at
SDOT's rate — some ARM Neoverse parts — which is why it's kept, not deleted.)

## the math

Quantize the query row to int8 (`q ≈ scale · v`). Doc values are signs
`s ∈ {−1,+1}` stored as bits. Split the dot product over set and unset bits:

    v · s = P − (T − P) = 2P − T,   P = Σ v over set bits,  T = Σ v

`T` is per-query-token, precomputed once. So scoring a doc token needs only a
bit-masked sum of int8s — no decompression. `P` via SDOT is `v · bits` with
`bits ∈ {0,1}`, which is why rung 4 expands bits to 0/1 bytes, not ±1.

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
produces, computed through this crate's dispatched kernel. On an Apple Silicon
Mac, build with `--target aarch64-apple-darwin` so the extension is native and
the SDOT rung actually runs (see the Rosetta note above).

## exercises

Each has a production answer in
[next-plaid](https://github.com/lightonai/next-plaid)'s `src/binary.rs`:

1. **Portable SIMD** — rewrite rung 3 with nightly `std::simd`. How close to
   rung 4 can a platform-independent kernel get?
2. **AVX2 masked-SAD** — x86 without VNNI: bias the query to u8, then
   `psadbw(qb & mask, 0) = P + 128·popcount(mask)`. (`maxsim_avx2_sad128`)
3. **AVX-512 VNNI** — `vpdpbusd` is SDOT's x86 cousin. (`maxsim_vnni128`)
4. **Other dims** — rung 4 hardcodes dim=128. Generalize to any multiple of
   32; measure what zero-padding dim=48 to 64 costs vs the fallback.
5. **Store the expansion?** Precompute each doc token's 128 expanded bytes at
   index time and skip `extract_planes_128`. Measure why this *loses* (hint:
   16 B vs 128 B of memory traffic per token).
