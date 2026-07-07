//! The kernel ladder: how fast can int8-query × 1-bit-document MaxSim go?
//!
//! nanoplaid.py stores binary documents as packed sign bits and scores them
//! with an int8 query through numpy/BLAS. This crate reimplements that one
//! inner loop as a ladder of kernels, each rung one idea faster than the
//! last, all returning bit-identical scores:
//!
//!   rung 1  `maxsim_f32`      the float reference — what we are approximating
//!   rung 2  `maxsim_scalar`   the 2P − T identity, one branch per bit
//!   rung 3  `maxsim_autovec`  branchless masks, written so LLVM autovectorizes
//!   rung 4  `maxsim_neon128`  fused doc-token-outer NEON SDOT (aarch64, dim 128)
//!
//! `maxsim` dispatches to the best available rung at runtime. Rungs left as
//! exercises (see README.md): portable `std::simd`, AVX2 masked-SAD and
//! AVX-512 VNNI — production versions of all three live in next-plaid.
//!
//! Layouts, chosen for kernels rather than ergonomics: a query is `nq` rows of
//! `dim` values flattened into one slice; a binary document is `nd` rows of
//! `dim/8` bytes, sign bits packed MSB-first (dim `d` is bit `7 - d%8` of byte
//! `d/8`, matching `np.packbits`). `dim` must be a multiple of 8.

// ---------------------------------------------------------------------------
// Shared: quantization. The query becomes integer codes `v` with a per-row
// scale (`v * scale ≈ q`) plus two precomputed row constants the identity
// needs: `T = Σ v` and, for rung 4, a plane-major copy of the codes.

pub struct QueryI8 {
    pub dim: usize,
    pub values: Vec<i8>,  // [nq * dim] row-major int8 codes
    pub scales: Vec<f32>, // [nq] dequantization scale per row
    pub sums: Vec<i32>,   // [nq] T = Σ codes, hoisted out of every kernel
    /// Plane-major codes for the fused NEON kernel (`dim == 128` only):
    /// `planes[qi*128 + p*16 + k] = values[qi*128 + k*8 + p]` — the order in
    /// which `extract_planes_128` emits the document's bits.
    pub planes: Vec<i8>,
}

pub fn quantize_query_i8(q: &[f32], dim: usize) -> QueryI8 {
    assert!(dim.is_multiple_of(8) && !q.is_empty() && q.len().is_multiple_of(dim));
    let nq = q.len() / dim;
    let mut values = vec![0i8; nq * dim];
    let mut scales = vec![0.0f32; nq];
    for (i, row) in q.chunks_exact(dim).enumerate() {
        let max_abs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if max_abs <= 0.0 {
            continue;
        }
        let scale = max_abs / 127.0;
        scales[i] = scale;
        for (d, &x) in row.iter().enumerate() {
            values[i * dim + d] = (x / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let sums = values
        .chunks_exact(dim)
        .map(|r| r.iter().map(|&x| x as i32).sum())
        .collect();
    let planes = if dim == 128 {
        let mut p = vec![0i8; nq * 128];
        for qi in 0..nq {
            for pl in 0..8 {
                for k in 0..16 {
                    p[qi * 128 + pl * 16 + k] = values[qi * 128 + k * 8 + pl];
                }
            }
        }
        p
    } else {
        Vec::new()
    };
    QueryI8 {
        dim,
        values,
        scales,
        sums,
        planes,
    }
}

/// Pack each row's sign bits MSB-first: 32× smaller than f32.
pub fn binarize(x: &[f32], dim: usize) -> Vec<u8> {
    assert!(dim.is_multiple_of(8) && x.len().is_multiple_of(dim));
    x.chunks_exact(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |b, (j, &v)| b | (((v > 0.0) as u8) << (7 - j)))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rung 1: the float reference. `d` here is the DEQUANTIZED ±1 matrix — this
// rung exists to define what "correct" means and to show the cost of touching
// f32 per (query token, doc token, dim).

pub fn maxsim_f32(q: &[f32], d: &[f32], dim: usize) -> f32 {
    let mut total = 0.0f32;
    for qrow in q.chunks_exact(dim) {
        let mut best = f32::NEG_INFINITY;
        for drow in d.chunks_exact(dim) {
            let dot: f32 = qrow.iter().zip(drow).map(|(a, b)| a * b).sum();
            best = best.max(dot);
        }
        total += best;
    }
    total
}

/// Unpack sign bits back to ±1 floats (feeds rung 1).
pub fn signs_pm1(bits: &[u8], dim: usize) -> Vec<f32> {
    bits.iter()
        .flat_map(|&b| (0..8).map(move |j| if (b >> (7 - j)) & 1 == 1 { 1.0 } else { -1.0 }))
        .take(bits.len() / (dim / 8) * dim)
        .collect()
}

// ---------------------------------------------------------------------------
// Rung 2: the algorithmic insight. With doc values s ∈ {−1,+1}, split the dot
// product over set and unset bits: v·s = P − (T − P) = 2P − T, where
// P = Σ v over SET bits and T = Σ v (precomputed). Scoring a compressed
// document now needs only integer adds selected by a bitmask — no
// decompression, no multiplies. Every later rung computes exactly this.

pub fn maxsim_scalar(q: &QueryI8, bits: &[u8], dim: usize) -> f32 {
    let pd = dim / 8;
    let mut total = 0.0f32;
    for qi in 0..q.sums.len() {
        let qrow = &q.values[qi * dim..(qi + 1) * dim];
        let mut best = i32::MIN;
        for drow in bits.chunks_exact(pd) {
            let mut p = 0i32;
            for (k, &byte) in drow.iter().enumerate() {
                for j in 0..8 {
                    if (byte >> (7 - j)) & 1 == 1 {
                        p += qrow[k * 8 + j] as i32;
                    }
                }
            }
            best = best.max(2 * p - q.sums[qi]);
        }
        total += best as f32 * q.scales[qi];
    }
    total
}

// ---------------------------------------------------------------------------
// Rung 3: same math, written for the compiler instead of the reader. The
// per-bit branch becomes a 0/−1 mask and an AND, so the inner loop is
// straight-line integer ops LLVM can autovectorize. Compare the disassembly
// with rung 2's — this rung is "free SIMD" before writing any intrinsics.

pub fn maxsim_autovec(q: &QueryI8, bits: &[u8], dim: usize) -> f32 {
    let pd = dim / 8;
    let mut total = 0.0f32;
    for qi in 0..q.sums.len() {
        let qrow = &q.values[qi * dim..(qi + 1) * dim];
        let mut best = i32::MIN;
        for drow in bits.chunks_exact(pd) {
            let mut p = 0i32;
            for (k, &byte) in drow.iter().enumerate() {
                let base = k * 8;
                for j in 0..8 {
                    let mask = -(((byte >> (7 - j)) & 1) as i32); // 0 or −1
                    p += (qrow[base + j] as i32) & mask;
                }
            }
            best = best.max(2 * p - q.sums[qi]);
        }
        total += best as f32 * q.scales[qi];
    }
    total
}

// ---------------------------------------------------------------------------
// Rung 4 (aarch64, dim = 128): the fused doc-token-outer kernel. Two ideas on
// top of rung 3:
//
//   1. Hardware dot product. SDOT computes 16 i8×i8 products and accumulates
//      into 4 i32 lanes in ONE instruction. To use it, each doc token's 128
//      bits are expanded to 128 bytes of 0/1 — 8 NEON shift+mask ops, since
//      one shift extracts bit-plane p from all 16 packed bytes at once. That
//      plane-major order is why QueryI8 carries a matching `planes` copy.
//      (Expansion trick due to mixedbread's aarch64 kernel.)
//      SDOT gives P via q·bits (bits ∈ {0,1}); the identity does the rest.
//
//   2. Loop inversion ("fused doc-token-outer"). Expanding a doc token costs
//      more than one dot product — so expand it ONCE into registers and score
//      it against ALL query tokens before moving on. The naive loop order
//      would re-expand every doc token nq times (or worse, store the
//      expansion: 128 B/token of memory traffic instead of 16 B packed —
//      measured slower than re-expanding in registers).
//
// Blocks of 4 doc tokens give each query token a 4-lane running max, and two
// accumulator chains per token hide SDOT's latency.

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// `sdot vD.4s, vN.16b, vM.16b` via inline asm (`vdotq_s32` is nightly).
    ///
    /// # Safety
    /// Requires the `dotprod` target feature at runtime — the dispatcher
    /// checks; calling this on ARMv8.0 without it is a SIGILL, not a wrong
    /// answer.
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let out: int32x4_t;
        std::arch::asm!(
            "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) acc => out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        out
    }

    /// One packed doc token (16 bytes) -> 128 bytes of 0/1, plane-major:
    /// `m[p*16 + k]` = bit p (MSB-first) of byte k, i.e. dim `k*8 + p`.
    #[inline(always)]
    unsafe fn extract_planes_128(bits: &[u8], m: &mut [i8; 128]) {
        let v = vld1q_u8(bits.as_ptr());
        let one = vdupq_n_u8(1);
        let p = m.as_mut_ptr();
        macro_rules! plane {
            ($i:expr, $sh:expr) => {
                vst1q_s8(
                    p.add($i * 16),
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<$sh>(v), one)),
                )
            };
        }
        plane!(0, 7);
        plane!(1, 6);
        plane!(2, 5);
        plane!(3, 4);
        plane!(4, 3);
        plane!(5, 2);
        plane!(6, 1);
        vst1q_s8(p.add(112), vreinterpretq_s8_u8(vandq_u8(v, one)));
    }

    /// # Safety
    /// Requires `dotprod`; `q.planes` must be populated (dim == 128) and
    /// `bits.len() == n_d * 16`.
    pub unsafe fn maxsim_neon128(q: &super::QueryI8, bits: &[u8]) -> f32 {
        let n_q = q.sums.len();
        let n_d = bits.len() / 16;
        // Per query token: a 4-lane running max, one lane per doc slot of the
        // current block. Tail blocks repeat the last token — repeats cannot
        // change a max.
        let mut best = vec![i32::MIN; n_q * 4];
        let mut planes = [0i8; 4 * 128];
        let mut db = 0usize;
        while db < n_d {
            for t in 0..4 {
                let d = (db + t).min(n_d - 1);
                let m: &mut [i8; 128] = (&mut planes[t * 128..(t + 1) * 128]).try_into().unwrap();
                extract_planes_128(&bits[d * 16..d * 16 + 16], m);
            }
            let pp = planes.as_ptr();
            for (qi, &sum) in q.sums.iter().enumerate() {
                let qp = q.planes.as_ptr().add(qi * 128);
                let q0 = vld1q_s8(qp);
                let q1 = vld1q_s8(qp.add(16));
                let q2 = vld1q_s8(qp.add(32));
                let q3 = vld1q_s8(qp.add(48));
                let q4 = vld1q_s8(qp.add(64));
                let q5 = vld1q_s8(qp.add(80));
                let q6 = vld1q_s8(qp.add(96));
                let q7 = vld1q_s8(qp.add(112));
                // One doc token = 8 SDOTs over two chains (latency hiding).
                macro_rules! tok {
                    ($off:expr) => {{
                        let mut a = vdupq_n_s32(0);
                        let mut b = vdupq_n_s32(0);
                        a = sdot(a, q0, vld1q_s8(pp.add($off)));
                        b = sdot(b, q1, vld1q_s8(pp.add($off + 16)));
                        a = sdot(a, q2, vld1q_s8(pp.add($off + 32)));
                        b = sdot(b, q3, vld1q_s8(pp.add($off + 48)));
                        a = sdot(a, q4, vld1q_s8(pp.add($off + 64)));
                        b = sdot(b, q5, vld1q_s8(pp.add($off + 80)));
                        a = sdot(a, q6, vld1q_s8(pp.add($off + 96)));
                        b = sdot(b, q7, vld1q_s8(pp.add($off + 112)));
                        vaddq_s32(a, b)
                    }};
                }
                let t0 = tok!(0);
                let t1 = tok!(128);
                let t2 = tok!(256);
                let t3 = tok!(384);
                // Pairwise-add tree -> [P0, P1, P2, P3] for the block.
                let p4 = vpaddq_s32(vpaddq_s32(t0, t1), vpaddq_s32(t2, t3));
                let sc = vsubq_s32(vshlq_n_s32::<1>(p4), vdupq_n_s32(sum));
                let bp = best.as_mut_ptr().add(qi * 4);
                vst1q_s32(bp, vmaxq_s32(vld1q_s32(bp), sc));
            }
            db += 4;
        }
        let mut total = 0.0f32;
        for (qi, &scale) in q.scales.iter().enumerate() {
            total += vmaxvq_s32(vld1q_s32(best.as_ptr().add(qi * 4))) as f32 * scale;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Dispatch: pick the best rung this CPU can run. Feature detection happens at
// RUNTIME — compiling with the feature enabled and shipping the binary to a
// core without it is a SIGILL. This is why production code can't just build
// with `-C target-cpu=native`.

pub fn maxsim(q: &QueryI8, bits: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if dim == 128 && !q.planes.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return unsafe { neon::maxsim_neon128(q, bits) };
    }
    maxsim_autovec(q, bits, dim)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so tests need no dependencies.
    fn randf(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    fn setup(nq: usize, nd: usize, dim: usize, seed: u64) -> (QueryI8, Vec<u8>) {
        let mut s = seed;
        let q: Vec<f32> = (0..nq * dim).map(|_| randf(&mut s)).collect();
        let d: Vec<f32> = (0..nd * dim).map(|_| randf(&mut s)).collect();
        (quantize_query_i8(&q, dim), binarize(&d, dim))
    }

    #[test]
    fn all_rungs_agree_exactly() {
        // Integer-domain kernels must agree bit-for-bit, across shapes that
        // exercise the NEON block tail (nd % 4 != 0).
        for &(nq, nd, dim) in &[(32, 80, 128), (7, 3, 128), (1, 1, 128), (5, 9, 64)] {
            let (q, bits) = setup(nq, nd, dim, 42 + nd as u64);
            let a = maxsim_scalar(&q, &bits, dim);
            let b = maxsim_autovec(&q, &bits, dim);
            let c = maxsim(&q, &bits, dim);
            assert_eq!(a, b, "scalar vs autovec ({nq},{nd},{dim})");
            assert_eq!(a, c, "scalar vs dispatched ({nq},{nd},{dim})");
        }
    }

    #[test]
    fn identity_matches_float_reference() {
        // 2P − T over ±1 docs == the float dot against dequantized ±1 values,
        // up to f32 summation error in the reference.
        let (q, bits) = setup(16, 40, 128, 7);
        let d_pm1 = signs_pm1(&bits, 128);
        let qf: Vec<f32> = q
            .values
            .chunks_exact(128)
            .zip(&q.scales)
            .flat_map(|(row, &s)| row.iter().map(move |&v| v as f32 * s))
            .collect();
        let reference = maxsim_f32(&qf, &d_pm1, 128);
        let fast = maxsim(&q, &bits, 128);
        assert!((reference - fast).abs() < 1e-2, "{reference} vs {fast}");
    }
}
