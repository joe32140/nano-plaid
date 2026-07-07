//! The kernel ladder: how fast can int8-query × 1-bit-document MaxSim go?
//!
//! nanoplaid.py stores binary documents as packed sign bits and scores them
//! with an int8 query through numpy/BLAS. This crate reimplements that one
//! inner loop as a ladder of kernels, each rung one more idea, all returning
//! bit-identical scores:
//!
//!   rung 1  `maxsim_f32`      the float reference — what we are approximating
//!   rung 2  `maxsim_scalar`   the 2P − T identity, one branch per bit
//!   rung 3  `maxsim_autovec`  branchless masks, written so LLVM autovectorizes
//!   rung 4  `maxsim_neon128`  fused doc-token-outer NEON SDOT (aarch64, dim 128)
//!   rung 5  `maxsim_smmla128` fused SMMLA — half the instructions, but only
//!                             ties rung 4 on the M4 (see its comment)
//!
//! `maxsim` dispatches to the best available rung at runtime (rung 4 on Apple
//! Silicon — rung 5 ties, so it isn't preferred). Rungs left as exercises (see
//! README.md): portable `std::simd`, AVX2 masked-SAD and AVX-512 VNNI —
//! production versions of all three live in next-plaid.
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
    /// Query tokens re-packed in pairs for the SMMLA kernel (`dim == 128`
    /// only): for pair `pidx` (tokens `2·pidx`, `2·pidx+1`), 16 chunks of
    /// `[qa's 8 plane bytes | qb's 8 plane bytes]` — the 16-byte operand SMMLA
    /// reads as two 8-wide rows. Same plane order as `planes`, so it dots
    /// against the same `extract_planes_128` doc bytes.
    pub planes_smmla: Vec<i8>,
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
            // round_ties_even matches numpy's np.rint (banker's rounding); plain
            // f32::round rounds halves away from zero and would assign a
            // different code on exact .5 values, breaking parity with the
            // numpy backend.
            values[i * dim + d] = (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
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
    // Pair-interleave the plane bytes so SMMLA can read a 16-byte operand as
    // two 8-wide rows (qa then qb). Odd tail: the last pair repeats the final
    // token, which cannot change a max.
    let planes_smmla = if dim == 128 {
        let n_pairs = nq.div_ceil(2);
        let mut ps = vec![0i8; n_pairs * 256];
        for pidx in 0..n_pairs {
            let qa = 2 * pidx;
            let qb = (2 * pidx + 1).min(nq - 1);
            for i in 0..16 {
                for j in 0..8 {
                    ps[pidx * 256 + i * 16 + j] = planes[qa * 128 + i * 8 + j];
                    ps[pidx * 256 + i * 16 + 8 + j] = planes[qb * 128 + i * 8 + j];
                }
            }
        }
        ps
    } else {
        Vec::new()
    };
    QueryI8 {
        dim,
        values,
        scales,
        sums,
        planes,
        planes_smmla,
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

    // -----------------------------------------------------------------------
    // Rung 5 (aarch64 + i8mm, dim = 128): SMMLA. One idea on top of rung 4.
    //
    // SDOT does 16 MACs per instruction (four 4-wide dot products). SMMLA does
    // 32: it reads each 16-byte operand as a 2×8 int8 matrix and computes the
    // full 2×2 product of dot products — so one instruction scores TWO query
    // tokens against TWO doc tokens over 8 dims at once:
    //
    //     acc[0]+=qa·da  acc[1]+=qa·db   (Vn = [qa 8B | qb 8B],
    //     acc[2]+=qb·da  acc[3]+=qb·db    Vm = [da 8B | db 8B])
    //
    // Covering dim 128 takes 16 SMMLA (16×8 dims) per 2×2 tile vs rung 4's
    // 8 SDOT per (query, doc) pair — half the instructions for the same tile,
    // so the ceiling is ~2× IF the core issues SMMLA as fast as SDOT.
    //
    // It does not, on the Apple M4: measured, rung 5 only *ties* rung 4
    // (~1.90 vs ~1.88 µs/doc). Half the instructions at ~half the issue rate
    // nets zero — the classic reason you benchmark the instruction on the
    // actual microarchitecture instead of counting MACs on paper. SMMLA can
    // still win on cores that issue it at SDOT's rate (some Neoverse parts),
    // which is why it stays exposed rather than deleted.
    //
    // The query is pre-interleaved into pairs (QueryI8::planes_smmla); each doc
    // pair is expanded to plane-major and zipped into `[da chunk | db chunk]`
    // 16-byte operands once, then reused across every query pair.

    /// `smmla vD.4s, vN.16b, vM.16b` via inline asm (`vmmlaq_s32` is nightly).
    ///
    /// # Safety
    /// Requires the `i8mm` target feature at runtime — the dispatcher checks;
    /// without it this is a SIGILL, not a wrong answer.
    #[inline(always)]
    unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let out: int32x4_t;
        // Unlike sdot (in the target's baseline), smmla needs i8mm enabled for
        // the assembler; the directive scopes it to this fragment.
        std::arch::asm!(
            ".arch_extension i8mm",
            "smmla {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) acc => out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        out
    }

    /// # Safety
    /// Requires `i8mm`; `q.planes_smmla` must be populated (dim == 128) and
    /// `bits.len() == n_d * 16`.
    pub unsafe fn maxsim_smmla128(q: &super::QueryI8, bits: &[u8]) -> f32 {
        let n_q = q.sums.len();
        let n_d = bits.len() / 16;
        let n_pairs = n_q.div_ceil(2);
        let mut best = vec![i32::MIN; n_q];
        let mut plane_a = [0i8; 128];
        let mut plane_b = [0i8; 128];
        let mut docbuf = [0i8; 256];
        let qbuf = q.planes_smmla.as_ptr();

        let mut dj = 0usize;
        while dj < n_d {
            let da = dj;
            let db = (dj + 1).min(n_d - 1);
            extract_planes_128(&bits[da * 16..da * 16 + 16], &mut plane_a);
            extract_planes_128(&bits[db * 16..db * 16 + 16], &mut plane_b);
            // Zip the two plane buffers into `[da chunk i | db chunk i]` 16-byte
            // operands: interleave in 64-bit groups (low/high halves).
            for j in 0..8 {
                let va = vld1q_s8(plane_a.as_ptr().add(j * 16));
                let vb = vld1q_s8(plane_b.as_ptr().add(j * 16));
                vst1q_s8(
                    docbuf.as_mut_ptr().add(2 * j * 16),
                    vcombine_s8(vget_low_s8(va), vget_low_s8(vb)),
                );
                vst1q_s8(
                    docbuf.as_mut_ptr().add((2 * j + 1) * 16),
                    vcombine_s8(vget_high_s8(va), vget_high_s8(vb)),
                );
            }
            let dp = docbuf.as_ptr();
            for pidx in 0..n_pairs {
                let qa = 2 * pidx;
                let qb = (2 * pidx + 1).min(n_q - 1);
                let qp = qbuf.add(pidx * 256);
                // 16 SMMLA over four accumulator chains to hide SMMLA latency.
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                let mut c = vdupq_n_s32(0);
                let mut d = vdupq_n_s32(0);
                let mut i = 0;
                while i < 16 {
                    a = smmla(a, vld1q_s8(qp.add(i * 16)), vld1q_s8(dp.add(i * 16)));
                    b = smmla(
                        b,
                        vld1q_s8(qp.add((i + 1) * 16)),
                        vld1q_s8(dp.add((i + 1) * 16)),
                    );
                    c = smmla(
                        c,
                        vld1q_s8(qp.add((i + 2) * 16)),
                        vld1q_s8(dp.add((i + 2) * 16)),
                    );
                    d = smmla(
                        d,
                        vld1q_s8(qp.add((i + 3) * 16)),
                        vld1q_s8(dp.add((i + 3) * 16)),
                    );
                    i += 4;
                }
                // p = [P(qa,da), P(qa,db), P(qb,da), P(qb,db)].
                let p = vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d));
                let tvec = {
                    let t = [q.sums[qa], q.sums[qa], q.sums[qb], q.sums[qb]];
                    vld1q_s32(t.as_ptr())
                };
                let sc = vsubq_s32(vshlq_n_s32::<1>(p), tvec);
                // pairwise max -> lane0 = best of qa over {da,db}, lane1 = qb.
                let pm = vpmaxq_s32(sc, sc);
                let m_qa = vgetq_lane_s32::<0>(pm);
                let m_qb = vgetq_lane_s32::<1>(pm);
                best[qa] = best[qa].max(m_qa);
                // Odd query tail: qb == qa, so fold its score into qa too.
                if qb != qa {
                    best[qb] = best[qb].max(m_qb);
                } else {
                    best[qa] = best[qa].max(m_qb);
                }
            }
            dj += 2;
        }
        let mut total = 0.0f32;
        for (qi, &scale) in q.scales.iter().enumerate() {
            total += best[qi] as f32 * scale;
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
    // SDOT first: rung 5 (SMMLA) only ties it on the M4 (see its comment), so
    // there's no reason to prefer the more complex kernel. SMMLA stays exposed
    // for the bench and for cores that may issue it faster.
    #[cfg(target_arch = "aarch64")]
    if let Some(v) = maxsim_sdot(q, bits, dim) {
        return v;
    }
    maxsim_autovec(q, bits, dim)
}

/// Rung 4 (fused NEON SDOT) if this CPU has `dotprod` and `dim == 128`, else
/// `None`. Exposed so the bench can time it head-to-head with SMMLA.
#[allow(unused_variables)]
pub fn maxsim_sdot(q: &QueryI8, bits: &[u8], dim: usize) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if dim == 128 && !q.planes.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_neon128(q, bits) });
    }
    None
}

/// Rung 5 (fused SMMLA) if this CPU has `i8mm` and `dim == 128`, else `None`.
#[allow(unused_variables)]
pub fn maxsim_smmla(q: &QueryI8, bits: &[u8], dim: usize) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if dim == 128 && !q.planes_smmla.is_empty() && std::arch::is_aarch64_feature_detected!("i8mm") {
        return Some(unsafe { neon::maxsim_smmla128(q, bits) });
    }
    None
}

// ---------------------------------------------------------------------------
// Python bridge — the numpy extension `eval.py --backend rust` imports. Opt-in
// (feature = "python") so `cargo test` / the bench example stay dependency-free.

#[cfg(feature = "python")]
mod python;

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
        // exercise the SDOT block tail (nd % 4 != 0) and the SMMLA pair tails
        // (odd nq and odd nd).
        for &(nq, nd, dim) in &[(32, 80, 128), (7, 3, 128), (1, 1, 128), (5, 9, 64)] {
            let (q, bits) = setup(nq, nd, dim, 42 + nd as u64);
            let a = maxsim_scalar(&q, &bits, dim);
            assert_eq!(
                a,
                maxsim_autovec(&q, &bits, dim),
                "autovec ({nq},{nd},{dim})"
            );
            assert_eq!(a, maxsim(&q, &bits, dim), "dispatched ({nq},{nd},{dim})");
            // Fused kernels, when this CPU/dim supports them, must also match.
            if let Some(v) = maxsim_sdot(&q, &bits, dim) {
                assert_eq!(a, v, "sdot ({nq},{nd},{dim})");
            }
            if let Some(v) = maxsim_smmla(&q, &bits, dim) {
                assert_eq!(a, v, "smmla ({nq},{nd},{dim})");
            }
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
