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
//!   rung 4  fused doc-token-outer, the platform's SIMD:
//!             `maxsim_neon128`      NEON SDOT   (aarch64 + dotprod)
//!             `maxsim_avx2_sad128`  AVX2 SAD    (x86_64 + avx2)
//!   rung 5  `maxsim_smmla128` fused SMMLA — half the instructions, but only
//!                             ties rung 4 on the M4 (see its comment)
//!
//! `maxsim` dispatches to the best available rung at runtime: SDOT on Apple
//! Silicon, AVX2 on x86, autovec elsewhere. Rungs left as exercises (see
//! README.md): portable `std::simd` and AVX-512 VNNI (faster than AVX2 but far
//! less universal) — production versions of all live in next-plaid.
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
    /// Row-major biased codes (`x ^ 0x80`, i.e. `x + 128` as `u8`) — the query
    /// layout the AVX2 masked-SAD kernel consumes, where the sum of selected
    /// biased bytes is `P + 128·popcount(bits)`.
    pub biased: Vec<u8>,
    /// Even/odd-permuted codes for the fused residual-4 kernels (`dim == 128`
    /// only). A 4-bit codes byte holds dims `2k` (high nibble) and `2k+1`
    /// (low nibble), so an in-register nibble split yields the EVEN dims'
    /// table lookups, then the ODD dims'. Per 64-dim group `g`:
    /// `perm4[qi*128 + g*64 + k]      = values[qi*128 + g*64 + 2k]` (evens)
    /// `perm4[qi*128 + g*64 + 32 + k] = values[qi*128 + g*64 + 2k+1]` (odds)
    /// — the order the looked-up weight bytes come out in.
    pub perm4: Vec<i8>,
    /// `|perm4|` as u8 — the unsigned operand for AVX2's `pmaddubsw` int8 dot
    /// (`perm4` itself supplies the signs through `psignb`).
    pub absq4: Vec<u8>,
    /// Bit-slot-permuted codes for the fused residual-2 kernels (`dim == 128`
    /// only). A 2-bit codes byte holds dims `4k..4k+3` MSB-first, so shifting
    /// every codes byte right by 6/4/2/0 (and masking to 2 bits) yields the
    /// table lookups for slot `j` — dims `4k+j` in k-order, all 32 of them
    /// since 32 codes bytes cover the whole token:
    /// `perm2[qi*128 + j*32 + k] = values[qi*128 + 4k + j]`   (j = bit slot)
    /// — the order the looked-up weight bytes come out in. Slot-major over the
    /// FULL 128 dims (unlike perm4's per-64 groups) so one extraction pairs
    /// with one AVX2 32-byte load, or with two NEON 16-byte loads (k < 16 from
    /// codes bytes 0..16 = dims < 64, k ≥ 16 from bytes 16..32).
    pub perm2: Vec<i8>,
    /// `|perm2|` as u8 — the `pmaddubsw` operand, exactly as `absq4`.
    pub absq2: Vec<u8>,
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
    // Biased u8 codes for the AVX2 masked-SAD kernel (row-major, all dims).
    let biased = values.iter().map(|&x| (x as u8) ^ 0x80).collect();
    // Even/odd permutation for the fused residual kernels (see field doc).
    let perm4 = if dim == 128 {
        let mut p = vec![0i8; nq * 128];
        for qi in 0..nq {
            for g in 0..2 {
                for k in 0..32 {
                    p[qi * 128 + g * 64 + k] = values[qi * 128 + g * 64 + 2 * k];
                    p[qi * 128 + g * 64 + 32 + k] = values[qi * 128 + g * 64 + 2 * k + 1];
                }
            }
        }
        p
    } else {
        Vec::new()
    };
    let absq4 = perm4.iter().map(|&x| x.unsigned_abs()).collect();
    // Bit-slot permutation for the fused residual-2 kernels (see field doc).
    let perm2 = if dim == 128 {
        let mut p = vec![0i8; nq * 128];
        for qi in 0..nq {
            for j in 0..4 {
                for k in 0..32 {
                    p[qi * 128 + j * 32 + k] = values[qi * 128 + 4 * k + j];
                }
            }
        }
        p
    } else {
        Vec::new()
    };
    let absq2 = perm2.iter().map(|&x| x.unsigned_abs()).collect();
    QueryI8 {
        dim,
        values,
        scales,
        sums,
        planes,
        planes_smmla,
        biased,
        perm4,
        absq4,
        perm2,
        absq2,
    }
}

// ---------------------------------------------------------------------------
// Fused residual scoring (nbits ∈ {4, 2, 1}): the LUT identity.
//
// A residual token decodes to `centroid[cid] + weights[codes]`, so
//
//     q · token = q · centroid[cid]  +  Σ_d q_d · weights[code_d]
//
// The centroid term is already sitting in stage 1's [nq, K] matrix — a table
// lookup. The residual term never needs the float token: `weights` is ONE
// 2^nbits-entry table shared by every dim, so int8-quantize it (numpy's
// `quantize_lut`) and the term becomes an integer dot between the query row
// and table-looked-up bytes. The binary identity is the 1-bit special case
// (weights = {−1,+1} gives 2P − T). One in-register instruction does 16
// lookups at once: NEON `tbl` / AVX2 `pshufb` — the same instruction FAISS's
// 4-bit fast-scan and llama.cpp's Q4 kernels are built on.
//
// Unlike the binary kernels, the max must happen in FLOAT: the centroid term
// varies per token, so per-(query row, token) the score is
// `fl(fl(scaleq·scalew) · acc) + cdot` — every rung computes those f32 ops in
// this exact order, which is also `nanoplaid.score_residual_lut`'s order, so
// all rungs stay bit-identical.

/// The int8-quantized decode table (`values · scale ≈ weights`), sized for
/// nbits = 4; for nbits < 4 only the first `2^nbits` entries are live and the
/// rest are zero (no code can index them — a 2-bit code is at most 3).
/// Invariant: every entry is in `[-127, 127]` (numpy's `quantize_lut` clips)
/// so AVX2's sign-flip trick can never hit `-128`.
pub struct LutI8 {
    pub values: [i8; 16],
    pub scale: f32,
}

/// Scalar reference for fused residual-4 MaxSim over one doc.
///
/// `codes`: `nd * dim/2` bytes of packed 4-bit bucket indices (np.packbits
/// order: high nibble = even dim). `cids[d]`: the token's centroid id.
/// `cdot_t`: the stage-1 centroid matrix TRANSPOSED, `[K, nq]` — one
/// contiguous nq-row per centroid, so a token's lookups are one cache line.
pub fn maxsim_r4_scalar(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> f32 {
    let dim = q.dim;
    let pd = dim / 2;
    let nq = q.scales.len();
    debug_assert_eq!(codes.len(), cids.len() * pd);
    let mut total = 0.0f32;
    for qi in 0..nq {
        let sqw = q.scales[qi] * lut.scale;
        let qrow = &q.values[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for (d, &cid) in cids.iter().enumerate() {
            let tok = &codes[d * pd..(d + 1) * pd];
            let mut acc = 0i32;
            for (j, &b) in tok.iter().enumerate() {
                acc += qrow[2 * j] as i32 * lut.values[(b >> 4) as usize] as i32;
                acc += qrow[2 * j + 1] as i32 * lut.values[(b & 15) as usize] as i32;
            }
            let s = sqw * acc as f32 + cdot_t[cid as usize * nq + qi];
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
}

/// Scalar reference for fused residual-2 MaxSim over one doc.
///
/// `codes`: `nd * dim/4` bytes of packed 2-bit bucket indices (np.packbits
/// order, MSB-first: byte `k` holds dims `4k..4k+3`, dim `4k` in bits 7..6).
/// Only `lut.values[0..4]` are live.
pub fn maxsim_r2_scalar(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> f32 {
    let dim = q.dim;
    let pd = dim / 4;
    let nq = q.scales.len();
    debug_assert_eq!(codes.len(), cids.len() * pd);
    let mut total = 0.0f32;
    for qi in 0..nq {
        let sqw = q.scales[qi] * lut.scale;
        let qrow = &q.values[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for (d, &cid) in cids.iter().enumerate() {
            let tok = &codes[d * pd..(d + 1) * pd];
            let mut acc = 0i32;
            for (j, &b) in tok.iter().enumerate() {
                acc += qrow[4 * j] as i32 * lut.values[(b >> 6) as usize] as i32;
                acc += qrow[4 * j + 1] as i32 * lut.values[(b >> 4 & 3) as usize] as i32;
                acc += qrow[4 * j + 2] as i32 * lut.values[(b >> 2 & 3) as usize] as i32;
                acc += qrow[4 * j + 3] as i32 * lut.values[(b & 3) as usize] as i32;
            }
            let s = sqw * acc as f32 + cdot_t[cid as usize * nq + qi];
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
}

/// Scalar reference for fused residual-1 MaxSim over one doc.
///
/// `codes`: `nd * dim/8` bytes of packed 1-bit bucket indices, MSB-first —
/// the same packing as the binary payload, but scored through a TRAINED
/// 2-entry table (`lut.values[0..2]`) instead of the hardwired {−1,+1}.
pub fn maxsim_r1_scalar(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> f32 {
    let dim = q.dim;
    let pd = dim / 8;
    let nq = q.scales.len();
    debug_assert_eq!(codes.len(), cids.len() * pd);
    let mut total = 0.0f32;
    for qi in 0..nq {
        let sqw = q.scales[qi] * lut.scale;
        let qrow = &q.values[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for (d, &cid) in cids.iter().enumerate() {
            let tok = &codes[d * pd..(d + 1) * pd];
            let mut acc = 0i32;
            for (j, &b) in tok.iter().enumerate() {
                for i in 0..8 {
                    acc += qrow[8 * j + i] as i32 * lut.values[(b >> (7 - i) & 1) as usize] as i32;
                }
            }
            let s = sqw * acc as f32 + cdot_t[cid as usize * nq + qi];
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
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
        // `.arch_extension dotprod` tells the ASSEMBLER to accept the
        // instruction even when the compile target's baseline is plain
        // ARMv8.0 (aarch64-unknown-linux-gnu). Apple targets have dotprod in
        // their baseline, which is why this only failed once CI grew a Linux
        // ARM runner. Runtime dispatch still gates actually EXECUTING it.
        std::arch::asm!(
            ".arch_extension dotprod",
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

    /// The residual family's float fold, vectorized across query rows.
    ///
    /// Every residual rung ends the same way: for one doc token it has an i32
    /// `acc` per query row, and folds `best[i] = max(best[i], sqw[i]·acc[i] +
    /// crow[i])`. The scalar-fold kernels do that one row at a time; this does
    /// four rows per `vmaxq_f32`. It is the transferable half of MaxSim — the
    /// same max-reduction mixedbread-ai/maxsim-cpu vectorizes on top of a plain
    /// float GEMM — so it improves the whole family with one helper and carries
    /// past this repo unchanged.
    ///
    /// Bit-identical to the scalar tail on purpose: `vcvtq_f32_s32` is the same
    /// round-to-nearest as `acc as f32`; the multiply and add stay SEPARATE
    /// (`vmulq` then `vaddq`, never `vfma`) so the two roundings match the
    /// spec's `sqw*acc + crow`; and for the finite scores this loop produces,
    /// `vmaxq_f32(best, s)` equals the scalar `if s > best` select. The parity
    /// test pins this exactly.
    #[inline(always)]
    unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, best: &mut [f32]) {
        let nq = accs.len();
        let mut i = 0;
        while i + 4 <= nq {
            let a = vcvtq_f32_s32(vld1q_s32(accs.as_ptr().add(i)));
            let s = vaddq_f32(vmulq_f32(vld1q_f32(sqw.as_ptr().add(i)), a), vld1q_f32(crow.add(i)));
            let b = vld1q_f32(best.as_ptr().add(i));
            vst1q_f32(best.as_mut_ptr().add(i), vmaxq_f32(b, s));
            i += 4;
        }
        while i < nq {
            let s = sqw[i] * accs[i] as f32 + *crow.add(i);
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
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
    // Fused residual-4: same doc-token-outer shape as the binary kernel, with
    // `tbl` replacing the bit-plane expansion. Per doc token: split the 64
    // packed bytes into high/low nibbles (even/odd dims) and run each through
    // the 16-entry int8 weight table — one `tbl` instruction does 16 lookups.
    // The 128 looked-up weight bytes live in 8 registers, amortized over every
    // query row exactly like the binary expansion; the inner loop is the same
    // 8 SDOTs. The float tail differs: the centroid term varies per token, so
    // each (query row, token) folds to a scalar and maxes in f32 (matching
    // `maxsim_r4_scalar`'s operation order bit-for-bit).

    /// # Safety
    /// Requires `dotprod`; `q.perm4` must be populated (dim == 128);
    /// `codes.len() == cids.len() * 64`; every cid indexes a `cdot_t` row.
    pub unsafe fn maxsim_r4_neon128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt = vld1q_s8(lut.values.as_ptr());
        let low4 = vdupq_n_u8(0x0F);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 64);
            let v0 = vld1q_u8(cp);
            let v1 = vld1q_u8(cp.add(16));
            let v2 = vld1q_u8(cp.add(32));
            let v3 = vld1q_u8(cp.add(48));
            // 8 weight vectors in perm4 order: per 64-dim group, evens (high
            // nibbles) then odds (low nibbles).
            let w = [
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v0)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v1)),
                vqtbl1q_s8(wt, vandq_u8(v0, low4)),
                vqtbl1q_s8(wt, vandq_u8(v1, low4)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v2)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v3)),
                vqtbl1q_s8(wt, vandq_u8(v2, low4)),
                vqtbl1q_s8(wt, vandq_u8(v3, low4)),
            ];
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qp = q.perm4.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), w[0]);
                b = sdot(b, vld1q_s8(qp.add(16)), w[1]);
                a = sdot(a, vld1q_s8(qp.add(32)), w[2]);
                b = sdot(b, vld1q_s8(qp.add(48)), w[3]);
                a = sdot(a, vld1q_s8(qp.add(64)), w[4]);
                b = sdot(b, vld1q_s8(qp.add(80)), w[5]);
                a = sdot(a, vld1q_s8(qp.add(96)), w[6]);
                b = sdot(b, vld1q_s8(qp.add(112)), w[7]);
                let acc = vaddvq_s32(vaddq_s32(a, b));
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // Fused residual-2: the `tbl` gets CHEAPER. A 2-bit codes byte holds FOUR
    // dims (MSB-first: dim 4k in bits 7..6), so 32 packed bytes carry the
    // whole token — half residual-4's code loads — and the four bit-slot
    // extractions ((>>6, >>4, >>2, >>0) & 3) feed the same one-instruction
    // table lookup, now with only 4 live entries. The lookups come out
    // slot-major (all dims 4k, then 4k+1, ...), which is exactly
    // QueryI8::perm2's order; everything after — 8 SDOTs per query row, the
    // f32 fold — is residual-4's inner loop verbatim.

    /// # Safety
    /// Requires `dotprod`; `q.perm2` must be populated (dim == 128);
    /// `codes.len() == cids.len() * 32`; every cid indexes a `cdot_t` row.
    pub unsafe fn maxsim_r2_neon128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt = vld1q_s8(lut.values.as_ptr());
        let two = vdupq_n_u8(0x03);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 32);
            let v0 = vld1q_u8(cp); // codes bytes 0..16  = dims 0..64
            let v1 = vld1q_u8(cp.add(16)); // codes bytes 16..32 = dims 64..128
                                           // 8 weight vectors in perm2 order: per bit slot j, the 16 lookups
                                           // for dims < 64 (v0), then the 16 for dims ≥ 64 (v1). >>6 needs no
                                           // mask (a byte shift can't drag in neighbors); the others do.
            let w = [
                vqtbl1q_s8(wt, vshrq_n_u8::<6>(v0)),
                vqtbl1q_s8(wt, vshrq_n_u8::<6>(v1)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<4>(v0), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<4>(v1), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<2>(v0), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<2>(v1), two)),
                vqtbl1q_s8(wt, vandq_u8(v0, two)),
                vqtbl1q_s8(wt, vandq_u8(v1, two)),
            ];
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qp = q.perm2.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), w[0]);
                b = sdot(b, vld1q_s8(qp.add(16)), w[1]);
                a = sdot(a, vld1q_s8(qp.add(32)), w[2]);
                b = sdot(b, vld1q_s8(qp.add(48)), w[3]);
                a = sdot(a, vld1q_s8(qp.add(64)), w[4]);
                b = sdot(b, vld1q_s8(qp.add(80)), w[5]);
                a = sdot(a, vld1q_s8(qp.add(96)), w[6]);
                b = sdot(b, vld1q_s8(qp.add(112)), w[7]);
                let acc = vaddvq_s32(vaddq_s32(a, b));
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // Fused residual-1: the 2-entry table degenerates into ALGEBRA — no `tbl`
    // at all. With bits b ∈ {0,1} and weights (w0, w1),
    //
    //     Σ_d q_d · w[b_d] = (w1 − w0)·P + w0·T,   P = Σ q over set bits
    //
    // — the affine generalization of 2P − T (set (w0, w1) = (−1, +1) and it
    // IS the binary identity). P is exactly what the binary SDOT kernel
    // computes, so this kernel is `maxsim_neon128`'s inner loop verbatim
    // (plane expansion, `q.planes`, 8 SDOTs) with an affine integer tail and
    // the residual family's float fold. Chosen over a 2-entry `tbl` of the
    // expanded planes: both are integer-exact, but the identity reuses the
    // binary machinery and query layout with zero new QueryI8 fields, and its
    // two extra scalar MACs land in a fold that already goes scalar per
    // (row, token). i32 is safe: |Δw·P| ≤ 254·16256 plus |w0·T| ≤ 127·16256.

    /// # Safety
    /// Requires `dotprod`; `q.planes` must be populated (dim == 128);
    /// `codes.len() == cids.len() * 16`; every cid indexes a `cdot_t` row.
    pub unsafe fn maxsim_r1_neon128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let dw = lut.values[1] as i32 - lut.values[0] as i32;
        let w0 = lut.values[0] as i32;
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut planes = [0i8; 128];
        for (d, &cid) in cids.iter().enumerate() {
            extract_planes_128(&codes[d * 16..d * 16 + 16], &mut planes);
            let pp = planes.as_ptr();
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qp = q.planes.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), vld1q_s8(pp));
                b = sdot(b, vld1q_s8(qp.add(16)), vld1q_s8(pp.add(16)));
                a = sdot(a, vld1q_s8(qp.add(32)), vld1q_s8(pp.add(32)));
                b = sdot(b, vld1q_s8(qp.add(48)), vld1q_s8(pp.add(48)));
                a = sdot(a, vld1q_s8(qp.add(64)), vld1q_s8(pp.add(64)));
                b = sdot(b, vld1q_s8(qp.add(80)), vld1q_s8(pp.add(80)));
                a = sdot(a, vld1q_s8(qp.add(96)), vld1q_s8(pp.add(96)));
                b = sdot(b, vld1q_s8(qp.add(112)), vld1q_s8(pp.add(112)));
                let p = vaddvq_s32(vaddq_s32(a, b));
                let acc = dw * p + w0 * q.sums[qi];
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // The fold-vectorized twins of the three rungs above. The SDOT/tbl compute
    // is byte-for-byte identical; the only change is structural — each doc
    // token's per-row accs land in a scratch buffer, then `fold_block` folds
    // four rows per instruction instead of one. This is the one place the
    // residual kernel is slower than binary (binary keeps an integer lane-max;
    // we fold to f32 per row), so it is the fold worth vectorizing. Whether it
    // actually pays is a measured question — hence keeping both twins.

    /// # Safety
    /// As `maxsim_r4_neon128`.
    pub unsafe fn maxsim_r4_neon128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt = vld1q_s8(lut.values.as_ptr());
        let low4 = vdupq_n_u8(0x0F);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 64);
            let v0 = vld1q_u8(cp);
            let v1 = vld1q_u8(cp.add(16));
            let v2 = vld1q_u8(cp.add(32));
            let v3 = vld1q_u8(cp.add(48));
            let w = [
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v0)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v1)),
                vqtbl1q_s8(wt, vandq_u8(v0, low4)),
                vqtbl1q_s8(wt, vandq_u8(v1, low4)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v2)),
                vqtbl1q_s8(wt, vshrq_n_u8::<4>(v3)),
                vqtbl1q_s8(wt, vandq_u8(v2, low4)),
                vqtbl1q_s8(wt, vandq_u8(v3, low4)),
            ];
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = q.perm4.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), w[0]);
                b = sdot(b, vld1q_s8(qp.add(16)), w[1]);
                a = sdot(a, vld1q_s8(qp.add(32)), w[2]);
                b = sdot(b, vld1q_s8(qp.add(48)), w[3]);
                a = sdot(a, vld1q_s8(qp.add(64)), w[4]);
                b = sdot(b, vld1q_s8(qp.add(80)), w[5]);
                a = sdot(a, vld1q_s8(qp.add(96)), w[6]);
                b = sdot(b, vld1q_s8(qp.add(112)), w[7]);
                *acc_qi = vaddvq_s32(vaddq_s32(a, b));
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
    }

    /// # Safety
    /// As `maxsim_r2_neon128`.
    pub unsafe fn maxsim_r2_neon128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt = vld1q_s8(lut.values.as_ptr());
        let two = vdupq_n_u8(0x03);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 32);
            let v0 = vld1q_u8(cp);
            let v1 = vld1q_u8(cp.add(16));
            let w = [
                vqtbl1q_s8(wt, vshrq_n_u8::<6>(v0)),
                vqtbl1q_s8(wt, vshrq_n_u8::<6>(v1)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<4>(v0), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<4>(v1), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<2>(v0), two)),
                vqtbl1q_s8(wt, vandq_u8(vshrq_n_u8::<2>(v1), two)),
                vqtbl1q_s8(wt, vandq_u8(v0, two)),
                vqtbl1q_s8(wt, vandq_u8(v1, two)),
            ];
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = q.perm2.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), w[0]);
                b = sdot(b, vld1q_s8(qp.add(16)), w[1]);
                a = sdot(a, vld1q_s8(qp.add(32)), w[2]);
                b = sdot(b, vld1q_s8(qp.add(48)), w[3]);
                a = sdot(a, vld1q_s8(qp.add(64)), w[4]);
                b = sdot(b, vld1q_s8(qp.add(80)), w[5]);
                a = sdot(a, vld1q_s8(qp.add(96)), w[6]);
                b = sdot(b, vld1q_s8(qp.add(112)), w[7]);
                *acc_qi = vaddvq_s32(vaddq_s32(a, b));
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
    }

    /// # Safety
    /// As `maxsim_r1_neon128`.
    pub unsafe fn maxsim_r1_neon128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let dw = lut.values[1] as i32 - lut.values[0] as i32;
        let w0 = lut.values[0] as i32;
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        let mut planes = [0i8; 128];
        for (d, &cid) in cids.iter().enumerate() {
            extract_planes_128(&codes[d * 16..d * 16 + 16], &mut planes);
            let pp = planes.as_ptr();
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qp = q.planes.as_ptr().add(qi * 128);
                let mut a = vdupq_n_s32(0);
                let mut b = vdupq_n_s32(0);
                a = sdot(a, vld1q_s8(qp), vld1q_s8(pp));
                b = sdot(b, vld1q_s8(qp.add(16)), vld1q_s8(pp.add(16)));
                a = sdot(a, vld1q_s8(qp.add(32)), vld1q_s8(pp.add(32)));
                b = sdot(b, vld1q_s8(qp.add(48)), vld1q_s8(pp.add(48)));
                a = sdot(a, vld1q_s8(qp.add(64)), vld1q_s8(pp.add(64)));
                b = sdot(b, vld1q_s8(qp.add(80)), vld1q_s8(pp.add(80)));
                a = sdot(a, vld1q_s8(qp.add(96)), vld1q_s8(pp.add(96)));
                b = sdot(b, vld1q_s8(qp.add(112)), vld1q_s8(pp.add(112)));
                let p = vaddvq_s32(vaddq_s32(a, b));
                *acc_qi = dw * p + w0 * q.sums[qi];
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
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
// The x86 rung 4: AVX2 masked-SAD (dim = 128), the same doc-token-outer idea as
// NEON SDOT but with the trick that fits x86's instruction set. It needs only
// AVX2, so it runs on essentially every x86_64 machine built since ~2013 -- the
// fused kernel Linux/x86 developers get. (AVX-512 VNNI is faster still where
// available; it's left as the README exercise since it's far less universal.)
//
// Expand each doc token's 128 bits ONCE into four ymm of 0xFF/0x00 masks
// (broadcast + pshufb + pcmpeqb), amortized over all query tokens. Scoring uses
// the biased-SAD identity: with the query stored as u8 `qb = q + 128`,
// `SAD(qb & mask, 0) = P + 128·popcount(bits)`, so `P = SAD − 128·popcount` and
// `score = 2P − T`. Every scoring op (pand / psadbw / paddq) is a cheap 1-µop
// instruction -- no widening multiply chains.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    // Broadcast byte k of a 4-byte word to lanes 8k..8k+8, then AND with the
    // MSB-first bit selector and compare-equal -> a 0xFF/0x00 mask per dim.
    // Shared by the binary SAD kernel and the residual-1 kernel below.
    const IDX: [i8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    const SEL: [i8; 32] = [
        -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02,
        0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, -128, 0x40, 0x20, 0x10, 0x08, 0x04,
        0x02, 0x01,
    ];

    /// The residual family's float fold, vectorized across query rows — the
    /// AVX2 twin of `neon::fold_block` (eight rows per `_mm256_max_ps` here).
    /// See that helper for the bit-exactness contract; the same holds:
    /// `_mm256_cvtepi32_ps` matches `acc as f32`, the multiply and add stay
    /// separate (no FMA), and `_mm256_max_ps` matches the scalar select for
    /// finite scores.
    ///
    /// # Safety
    /// Requires `avx2`. `accs`, `sqw`, `best` share length; `crow` is valid for
    /// that length.
    #[target_feature(enable = "avx2")]
    unsafe fn fold_block(accs: &[i32], sqw: &[f32], crow: *const f32, best: &mut [f32]) {
        let nq = accs.len();
        let mut i = 0;
        while i + 8 <= nq {
            let a = _mm256_cvtepi32_ps(_mm256_loadu_si256(accs.as_ptr().add(i) as *const __m256i));
            let sw = _mm256_loadu_ps(sqw.as_ptr().add(i));
            let s = _mm256_add_ps(_mm256_mul_ps(sw, a), _mm256_loadu_ps(crow.add(i)));
            let b = _mm256_loadu_ps(best.as_ptr().add(i));
            _mm256_storeu_ps(best.as_mut_ptr().add(i), _mm256_max_ps(b, s));
            i += 8;
        }
        while i < nq {
            let s = sqw[i] * accs[i] as f32 + *crow.add(i);
            if s > best[i] {
                best[i] = s;
            }
            i += 1;
        }
    }

    /// # Safety
    /// Requires `avx2`; `q.biased` must be populated (dim == 128) and
    /// `bits.len() == n_d * 16`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_avx2_sad128(q: &super::QueryI8, bits: &[u8]) -> f32 {
        let n_q = q.sums.len();
        let n_d = bits.len() / 16;
        let idx = _mm256_loadu_si256(IDX.as_ptr() as *const __m256i);
        let sel = _mm256_loadu_si256(SEL.as_ptr() as *const __m256i);
        let zero = _mm256_setzero_si256();

        // 0xFF mask ymm for dims 32g..32g+32 of the token at `bp`.
        macro_rules! mask32 {
            ($bp:expr, $g:expr) => {{
                let w = ($bp.add($g * 4) as *const u32).read_unaligned();
                let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
                _mm256_cmpeq_epi8(_mm256_and_si256(bytes, sel), sel)
            }};
        }

        let mut best = vec![i32::MIN; n_q];
        let qp = q.biased.as_ptr();
        for d in 0..n_d {
            let bp = bits.as_ptr().add(d * 16);
            let m0 = mask32!(bp, 0);
            let m1 = mask32!(bp, 1);
            let m2 = mask32!(bp, 2);
            let m3 = mask32!(bp, 3);
            let cnt = ((bp as *const u64).read_unaligned().count_ones()
                + (bp.add(8) as *const u64).read_unaligned().count_ones())
                as i32;
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let q0 = qp.add(qi * 128);
                let s0 = _mm256_sad_epu8(
                    _mm256_and_si256(m0, _mm256_loadu_si256(q0 as *const __m256i)),
                    zero,
                );
                let s1 = _mm256_sad_epu8(
                    _mm256_and_si256(m1, _mm256_loadu_si256(q0.add(32) as *const __m256i)),
                    zero,
                );
                let s2 = _mm256_sad_epu8(
                    _mm256_and_si256(m2, _mm256_loadu_si256(q0.add(64) as *const __m256i)),
                    zero,
                );
                let s3 = _mm256_sad_epu8(
                    _mm256_and_si256(m3, _mm256_loadu_si256(q0.add(96) as *const __m256i)),
                    zero,
                );
                let s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
                let x = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                let sad = _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_unpackhi_epi64(x, x))) as i32;
                // SAD = P + 128·popcount, so P = SAD − 128·cnt; score = 2P − T.
                let score = 2 * (sad - 128 * cnt) - q.sums[qi];
                if score > *best_qi {
                    *best_qi = score;
                }
            }
        }
        let mut total = 0.0f32;
        for (qi, &scale) in q.scales.iter().enumerate() {
            total += best[qi] as f32 * scale;
        }
        total
    }

    // -----------------------------------------------------------------------
    // Fused residual-4 on AVX2: `pshufb` is the 16-entry table lookup (the
    // same instruction FAISS's 4-bit fast-scan is built on), and the int8 dot
    // uses the classic sign-transfer pair: `psignb` moves the query's signs
    // onto the looked-up weights, `pmaddubsw` multiplies |q| (unsigned) by
    // them. With both operands ≤ 127 in magnitude a pair sum is ≤ 32258, so
    // the i16 lanes cannot saturate — that's why LutI8 clips to [-127, 127].

    /// # Safety
    /// Requires `avx2`; `q.perm4`/`q.absq4` must be populated (dim == 128);
    /// `codes.len() == cids.len() * 64`; every cid indexes a `cdot_t` row.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r4_avx2_128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.values.as_ptr() as *const __m128i));
        let low4 = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 64);
            let v0 = _mm256_loadu_si256(cp as *const __m256i);
            let v1 = _mm256_loadu_si256(cp.add(32) as *const __m256i);
            // 4 weight vectors in perm4 order: per 64-dim group, evens (high
            // nibbles) then odds (low nibbles). pshufb reads only the low 4
            // index bits when bit 7 is clear, and both extractions mask to 15.
            let w = [
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v0, 4), low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v0, low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v1, 4), low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v1, low4)),
            ];
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qs = q.perm4.as_ptr().add(qi * 128);
                let qa = q.absq4.as_ptr().add(qi * 128);
                let mut acc = _mm256_setzero_si256();
                for (g, &wg) in w.iter().enumerate() {
                    let sign_src = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                    let mag = _mm256_loadu_si256(qa.add(g * 32) as *const __m256i);
                    let ws = _mm256_sign_epi8(wg, sign_src); // w·sign(q), 0 where q=0
                    let p16 = _mm256_maddubs_epi16(mag, ws); // |q|·ws, adjacent pairs
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
                }
                let x = _mm_add_epi32(
                    _mm256_castsi256_si128(acc),
                    _mm256_extracti128_si256(acc, 1),
                );
                let x = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
                let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01));
                let acc = _mm_cvtsi128_si32(x);
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // Fused residual-2 on AVX2: same shape as residual-4, cheaper unpack. One
    // 32-byte load is the WHOLE token (4 dims per codes byte), and the four
    // bit-slot extractions feed the same `pshufb`. Lane 0 holds codes bytes
    // 0..16 (dims < 64), lane 1 bytes 16..32 — which is exactly why perm2 is
    // slot-major over the full 128 dims: byte i of slot j's lookup is dim
    // `4i+j` for i < 16 and dim `64 + 4(i−16) + j` for i ≥ 16, matching the
    // 32 query bytes at perm2 offset j*32 in one load.

    /// # Safety
    /// Requires `avx2`; `q.perm2`/`q.absq2` must be populated (dim == 128);
    /// `codes.len() == cids.len() * 32`; every cid indexes a `cdot_t` row.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r2_avx2_128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.values.as_ptr() as *const __m128i));
        let two = _mm256_set1_epi8(0x03);
        let ones = _mm256_set1_epi16(1);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 32);
            let v = _mm256_loadu_si256(cp as *const __m256i);
            // 4 weight vectors in perm2 order. srli_epi16 drags bits in from
            // the neighboring byte; the & 3 keeps only each byte's own two
            // bits (r4 masks its nibbles the same way).
            let w = [
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 6), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 4), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 2), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v, two)),
            ];
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let qs = q.perm2.as_ptr().add(qi * 128);
                let qa = q.absq2.as_ptr().add(qi * 128);
                let mut acc = _mm256_setzero_si256();
                for (g, &wg) in w.iter().enumerate() {
                    let sign_src = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                    let mag = _mm256_loadu_si256(qa.add(g * 32) as *const __m256i);
                    let ws = _mm256_sign_epi8(wg, sign_src); // w·sign(q), 0 where q=0
                    let p16 = _mm256_maddubs_epi16(mag, ws); // |q|·ws, adjacent pairs
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
                }
                let x = _mm_add_epi32(
                    _mm256_castsi256_si128(acc),
                    _mm256_extracti128_si256(acc, 1),
                );
                let x = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
                let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01));
                let acc = _mm_cvtsi128_si32(x);
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // Fused residual-1 on AVX2: the affine identity (see the NEON residual-1
    // comment): acc = (w1 − w0)·P + w0·T. P comes from the same masked-SAD
    // trick as the binary kernel — SAD(qb & mask, 0) = P + 128·popcount — so
    // this is `maxsim_avx2_sad128` with an affine integer tail and the
    // residual family's float fold. Zero new query layouts, integer-exact.

    /// # Safety
    /// Requires `avx2`; `q.biased` must be populated; dim == 128;
    /// `codes.len() == cids.len() * 16`; every cid indexes a `cdot_t` row.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r1_avx2_128(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let dw = lut.values[1] as i32 - lut.values[0] as i32;
        let w0 = lut.values[0] as i32;
        let idx = _mm256_loadu_si256(IDX.as_ptr() as *const __m256i);
        let sel = _mm256_loadu_si256(SEL.as_ptr() as *const __m256i);
        let zero = _mm256_setzero_si256();

        // 0xFF mask ymm for dims 32g..32g+32 of the token at `bp`.
        macro_rules! mask32 {
            ($bp:expr, $g:expr) => {{
                let w = ($bp.add($g * 4) as *const u32).read_unaligned();
                let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
                _mm256_cmpeq_epi8(_mm256_and_si256(bytes, sel), sel)
            }};
        }

        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let qp = q.biased.as_ptr();
        for (d, &cid) in cids.iter().enumerate() {
            let bp = codes.as_ptr().add(d * 16);
            let m0 = mask32!(bp, 0);
            let m1 = mask32!(bp, 1);
            let m2 = mask32!(bp, 2);
            let m3 = mask32!(bp, 3);
            let cnt = ((bp as *const u64).read_unaligned().count_ones()
                + (bp.add(8) as *const u64).read_unaligned().count_ones())
                as i32;
            let crow = cdot_t.as_ptr().add(cid as usize * nq);
            for (qi, best_qi) in best.iter_mut().enumerate() {
                let q0 = qp.add(qi * 128);
                let s0 = _mm256_sad_epu8(
                    _mm256_and_si256(m0, _mm256_loadu_si256(q0 as *const __m256i)),
                    zero,
                );
                let s1 = _mm256_sad_epu8(
                    _mm256_and_si256(m1, _mm256_loadu_si256(q0.add(32) as *const __m256i)),
                    zero,
                );
                let s2 = _mm256_sad_epu8(
                    _mm256_and_si256(m2, _mm256_loadu_si256(q0.add(64) as *const __m256i)),
                    zero,
                );
                let s3 = _mm256_sad_epu8(
                    _mm256_and_si256(m3, _mm256_loadu_si256(q0.add(96) as *const __m256i)),
                    zero,
                );
                let s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
                let x = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                let sad = _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_unpackhi_epi64(x, x))) as i32;
                // SAD = P + 128·popcount, so P = SAD − 128·cnt.
                let acc = dw * (sad - 128 * cnt) + w0 * q.sums[qi];
                let s = sqw[qi] * acc as f32 + *crow.add(qi);
                if s > *best_qi {
                    *best_qi = s;
                }
            }
        }
        best.iter().sum()
    }

    // -----------------------------------------------------------------------
    // Fold-vectorized twins (see the NEON block's comment). Same shufb/SAD
    // compute; each token's per-row accs land in a scratch buffer, then
    // `fold_block` folds eight rows per `_mm256_max_ps`.

    /// # Safety
    /// As `maxsim_r4_avx2_128`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r4_avx2_128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.values.as_ptr() as *const __m128i));
        let low4 = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 64);
            let v0 = _mm256_loadu_si256(cp as *const __m256i);
            let v1 = _mm256_loadu_si256(cp.add(32) as *const __m256i);
            let w = [
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v0, 4), low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v0, low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v1, 4), low4)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v1, low4)),
            ];
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qs = q.perm4.as_ptr().add(qi * 128);
                let qa = q.absq4.as_ptr().add(qi * 128);
                let mut acc = _mm256_setzero_si256();
                for (g, &wg) in w.iter().enumerate() {
                    let sign_src = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                    let mag = _mm256_loadu_si256(qa.add(g * 32) as *const __m256i);
                    let ws = _mm256_sign_epi8(wg, sign_src);
                    let p16 = _mm256_maddubs_epi16(mag, ws);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
                }
                let x =
                    _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
                let x = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
                let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01));
                *acc_qi = _mm_cvtsi128_si32(x);
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
    }

    /// # Safety
    /// As `maxsim_r2_avx2_128`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r2_avx2_128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let wt =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(lut.values.as_ptr() as *const __m128i));
        let two = _mm256_set1_epi8(0x03);
        let ones = _mm256_set1_epi16(1);
        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        for (d, &cid) in cids.iter().enumerate() {
            let cp = codes.as_ptr().add(d * 32);
            let v = _mm256_loadu_si256(cp as *const __m256i);
            let w = [
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 6), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 4), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(_mm256_srli_epi16(v, 2), two)),
                _mm256_shuffle_epi8(wt, _mm256_and_si256(v, two)),
            ];
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let qs = q.perm2.as_ptr().add(qi * 128);
                let qa = q.absq2.as_ptr().add(qi * 128);
                let mut acc = _mm256_setzero_si256();
                for (g, &wg) in w.iter().enumerate() {
                    let sign_src = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
                    let mag = _mm256_loadu_si256(qa.add(g * 32) as *const __m256i);
                    let ws = _mm256_sign_epi8(wg, sign_src);
                    let p16 = _mm256_maddubs_epi16(mag, ws);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
                }
                let x =
                    _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
                let x = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
                let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01));
                *acc_qi = _mm_cvtsi128_si32(x);
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
    }

    /// # Safety
    /// As `maxsim_r1_avx2_128`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn maxsim_r1_avx2_128_vfold(
        q: &super::QueryI8,
        lut: &super::LutI8,
        codes: &[u8],
        cids: &[u32],
        cdot_t: &[f32],
    ) -> f32 {
        let nq = q.scales.len();
        let dw = lut.values[1] as i32 - lut.values[0] as i32;
        let w0 = lut.values[0] as i32;
        let idx = _mm256_loadu_si256(IDX.as_ptr() as *const __m256i);
        let sel = _mm256_loadu_si256(SEL.as_ptr() as *const __m256i);
        let zero = _mm256_setzero_si256();

        macro_rules! mask32 {
            ($bp:expr, $g:expr) => {{
                let w = ($bp.add($g * 4) as *const u32).read_unaligned();
                let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
                _mm256_cmpeq_epi8(_mm256_and_si256(bytes, sel), sel)
            }};
        }

        let sqw: Vec<f32> = q.scales.iter().map(|&s| s * lut.scale).collect();
        let mut best = vec![f32::NEG_INFINITY; nq];
        let mut accs = vec![0i32; nq];
        let qp = q.biased.as_ptr();
        for (d, &cid) in cids.iter().enumerate() {
            let bp = codes.as_ptr().add(d * 16);
            let m0 = mask32!(bp, 0);
            let m1 = mask32!(bp, 1);
            let m2 = mask32!(bp, 2);
            let m3 = mask32!(bp, 3);
            let cnt = ((bp as *const u64).read_unaligned().count_ones()
                + (bp.add(8) as *const u64).read_unaligned().count_ones())
                as i32;
            for (qi, acc_qi) in accs.iter_mut().enumerate() {
                let q0 = qp.add(qi * 128);
                let s0 =
                    _mm256_sad_epu8(_mm256_and_si256(m0, _mm256_loadu_si256(q0 as *const __m256i)), zero);
                let s1 = _mm256_sad_epu8(
                    _mm256_and_si256(m1, _mm256_loadu_si256(q0.add(32) as *const __m256i)),
                    zero,
                );
                let s2 = _mm256_sad_epu8(
                    _mm256_and_si256(m2, _mm256_loadu_si256(q0.add(64) as *const __m256i)),
                    zero,
                );
                let s3 = _mm256_sad_epu8(
                    _mm256_and_si256(m3, _mm256_loadu_si256(q0.add(96) as *const __m256i)),
                    zero,
                );
                let s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
                let x = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                let sad = _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_unpackhi_epi64(x, x))) as i32;
                *acc_qi = dw * (sad - 128 * cnt) + w0 * q.sums[qi];
            }
            fold_block(&accs, &sqw, cdot_t.as_ptr().add(cid as usize * nq), &mut best);
        }
        best.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Dispatch: pick the best rung this CPU can run. Feature detection happens at
// RUNTIME — compiling with the feature enabled and shipping the binary to a
// core without it is a SIGILL. This is why production code can't just build
// with `-C target-cpu=native`.

pub fn maxsim(q: &QueryI8, bits: &[u8], dim: usize) -> f32 {
    // SMMLA first where i8mm exists: this ordering is itself a MEASURED
    // decision, and it flipped once CI grew a second arm64 platform. On the
    // Apple M4, SMMLA only ties SDOT (0.99×, see its comment) — but on the
    // Neoverse N2 CI runner it wins 1.40× (3.46 vs 4.85 µs/doc), exactly the
    // issue-rate difference the rung-5 writeup predicted. Cost of preferring
    // it on Apple: ~1%. Benefit on Neoverse/Graviton-class cores: ~40%.
    #[cfg(target_arch = "aarch64")]
    if let Some(v) = maxsim_smmla(q, bits, dim) {
        return v;
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(v) = maxsim_sdot(q, bits, dim) {
        return v;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(v) = maxsim_avx2(q, bits, dim) {
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

/// The x86 fused kernel (AVX2 masked-SAD) if this CPU has `avx2` and
/// `dim == 128`, else `None`. The x86 analog of `maxsim_sdot`.
#[allow(unused_variables)]
pub fn maxsim_avx2(q: &QueryI8, bits: &[u8], dim: usize) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    if dim == 128 && !q.biased.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_avx2_sad128(q, bits) });
    }
    None
}

/// Fused residual-4 MaxSim over one doc: dispatched (NEON tbl+sdot / AVX2
/// pshufb+psignb where available, scalar reference otherwise).
///
/// Ships the vectorized-fold kernel: measured ~2.1× over the scalar-fold twin
/// on the Apple M4, bit-identical, and — unlike the SMMLA decision — the win
/// (a branch-free acc pass plus a 4/8-wide `max`) is not tied to one
/// instruction's issue rate, so it carries across microarchitectures. CI's
/// bench confirms the AVX2/Neoverse numbers; the scalar-fold rung stays
/// exposed as `maxsim_r4_fused` for the head-to-head.
pub fn maxsim_r4(q: &QueryI8, lut: &LutI8, codes: &[u8], cids: &[u32], cdot_t: &[f32]) -> f32 {
    if let Some(v) = maxsim_r4_vfold_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    if let Some(v) = maxsim_r4_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    maxsim_r4_scalar(q, lut, codes, cids, cdot_t)
}

/// The fused residual kernel this CPU supports, if `dim == 128`, else `None`.
/// Exposed so the bench and tests can time/verify it explicitly.
#[allow(unused_variables)]
pub fn maxsim_r4_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.perm4.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r4_neon128(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.perm4.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r4_avx2_128(q, lut, codes, cids, cdot_t) });
    }
    None
}

/// Fused residual-2 MaxSim over one doc: dispatched exactly like `maxsim_r4`
/// (vectorized fold first, then the scalar-fold rung, then scalar).
pub fn maxsim_r2(q: &QueryI8, lut: &LutI8, codes: &[u8], cids: &[u32], cdot_t: &[f32]) -> f32 {
    if let Some(v) = maxsim_r2_vfold_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    if let Some(v) = maxsim_r2_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    maxsim_r2_scalar(q, lut, codes, cids, cdot_t)
}

/// The fused residual-2 kernel this CPU supports, if `dim == 128`, else `None`.
#[allow(unused_variables)]
pub fn maxsim_r2_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.perm2.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r2_neon128(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.perm2.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r2_avx2_128(q, lut, codes, cids, cdot_t) });
    }
    None
}

/// Fused residual-1 MaxSim over one doc: dispatched exactly like `maxsim_r4`
/// (vectorized fold first, then the scalar-fold rung, then scalar).
pub fn maxsim_r1(q: &QueryI8, lut: &LutI8, codes: &[u8], cids: &[u32], cdot_t: &[f32]) -> f32 {
    if let Some(v) = maxsim_r1_vfold_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    if let Some(v) = maxsim_r1_fused(q, lut, codes, cids, cdot_t) {
        return v;
    }
    maxsim_r1_scalar(q, lut, codes, cids, cdot_t)
}

/// The fused residual-1 kernel this CPU supports, if `dim == 128`, else `None`.
/// Rides the BINARY kernels' query layouts (`planes` on NEON, `biased` on
/// AVX2) — the affine identity needs nothing residual-specific in the query.
#[allow(unused_variables)]
pub fn maxsim_r1_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.planes.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r1_neon128(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.biased.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r1_avx2_128(q, lut, codes, cids, cdot_t) });
    }
    None
}

// The vectorized-fold twins of the three `*_fused` accessors above. Same
// dispatch gates; they route to the `*_vfold` kernels so the bench can time
// them head-to-head and the parity test can pin them bit-identical. The
// shipping dispatchers (`maxsim_r{4,2,1}`) point at whichever the measurement
// on this crate's CI platforms favored — see kernels/README.

/// The fold-vectorized fused residual-4 kernel, if this CPU has it.
#[allow(unused_variables)]
pub fn maxsim_r4_vfold_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.perm4.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r4_neon128_vfold(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.perm4.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r4_avx2_128_vfold(q, lut, codes, cids, cdot_t) });
    }
    None
}

/// The fold-vectorized fused residual-2 kernel, if this CPU has it.
#[allow(unused_variables)]
pub fn maxsim_r2_vfold_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.perm2.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r2_neon128_vfold(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.perm2.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r2_avx2_128_vfold(q, lut, codes, cids, cdot_t) });
    }
    None
}

/// The fold-vectorized fused residual-1 kernel, if this CPU has it.
#[allow(unused_variables)]
pub fn maxsim_r1_vfold_fused(
    q: &QueryI8,
    lut: &LutI8,
    codes: &[u8],
    cids: &[u32],
    cdot_t: &[f32],
) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    if q.dim == 128 && !q.planes.is_empty() && std::arch::is_aarch64_feature_detected!("dotprod") {
        return Some(unsafe { neon::maxsim_r1_neon128_vfold(q, lut, codes, cids, cdot_t) });
    }
    #[cfg(target_arch = "x86_64")]
    if q.dim == 128 && !q.biased.is_empty() && std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::maxsim_r1_avx2_128_vfold(q, lut, codes, cids, cdot_t) });
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
            if let Some(v) = maxsim_avx2(&q, &bits, dim) {
                assert_eq!(a, v, "avx2 ({nq},{nd},{dim})");
            }
        }
    }

    fn setup_r(
        nq: usize,
        nd: usize,
        dim: usize,
        n_cent: usize,
        nbits: usize,
        seed: u64,
    ) -> (QueryI8, LutI8, Vec<u8>, Vec<u32>, Vec<f32>) {
        let mut s = seed;
        let q: Vec<f32> = (0..nq * dim).map(|_| randf(&mut s)).collect();
        // Only the first 2^nbits entries are live (the LutI8 invariant); a
        // random codes byte can index nothing past them for any nbits.
        let mut values = [0i8; 16];
        for v in values.iter_mut().take(1 << nbits) {
            *v = (randf(&mut s) * 254.0) as i8; // in [-127, 127]: the invariant
        }
        let lut = LutI8 {
            values,
            scale: 0.0031,
        };
        let codes: Vec<u8> = (0..nd * dim * nbits / 8)
            .map(|_| ((randf(&mut s) + 0.5) * 255.99) as u8)
            .collect();
        let cids: Vec<u32> = (0..nd)
            .map(|_| ((randf(&mut s) + 0.5) * (n_cent as f32 - 0.01)) as u32)
            .collect();
        let cdot_t: Vec<f32> = (0..n_cent * nq).map(|_| randf(&mut s) * 4.0).collect();
        (quantize_query_i8(&q, dim), lut, codes, cids, cdot_t)
    }

    type RKernel = fn(&QueryI8, &LutI8, &[u8], &[u32], &[f32]) -> f32;
    type RFused = fn(&QueryI8, &LutI8, &[u8], &[u32], &[f32]) -> Option<f32>;
    // (nbits, scalar reference, dispatcher, scalar-fold fused, vfold fused)
    const R_FAMILY: [(usize, RKernel, RKernel, RFused, RFused); 3] = [
        (4, maxsim_r4_scalar, maxsim_r4, maxsim_r4_fused, maxsim_r4_vfold_fused),
        (2, maxsim_r2_scalar, maxsim_r2, maxsim_r2_fused, maxsim_r2_vfold_fused),
        (1, maxsim_r1_scalar, maxsim_r1, maxsim_r1_fused, maxsim_r1_vfold_fused),
    ];

    #[test]
    fn residual_rungs_agree_exactly() {
        // Scalar reference, dispatcher, and BOTH fused folds (scalar-fold and
        // vectorized-fold) must return bit-identical floats — the f32 op order
        // per (query row, token) is part of the spec, and the vfold helper is
        // built to preserve it (separate mul+add, not FMA; vector max == the
        // scalar select for finite scores). Same contract for every nbits rung.
        for &(nbits, scalar, dispatched, fused, vfold) in &R_FAMILY {
            for &(nq, nd) in &[(32, 80), (7, 3), (1, 1)] {
                let (q, lut, codes, cids, cdot_t) = setup_r(nq, nd, 128, 16, nbits, 99 + nd as u64);
                let a = scalar(&q, &lut, &codes, &cids, &cdot_t);
                assert_eq!(
                    a,
                    dispatched(&q, &lut, &codes, &cids, &cdot_t),
                    "dispatched (nbits={nbits}, {nq},{nd})"
                );
                if let Some(v) = fused(&q, &lut, &codes, &cids, &cdot_t) {
                    assert_eq!(a, v, "fused (nbits={nbits}, {nq},{nd})");
                }
                if let Some(v) = vfold(&q, &lut, &codes, &cids, &cdot_t) {
                    assert_eq!(a, v, "vfold (nbits={nbits}, {nq},{nd})");
                }
            }
            // Non-128 dims must fall back to the scalar rung, not crash.
            let (q, lut, codes, cids, cdot_t) = setup_r(3, 5, 64, 8, nbits, 7);
            assert!(fused(&q, &lut, &codes, &cids, &cdot_t).is_none());
            assert!(vfold(&q, &lut, &codes, &cids, &cdot_t).is_none());
            assert_eq!(
                dispatched(&q, &lut, &codes, &cids, &cdot_t),
                scalar(&q, &lut, &codes, &cids, &cdot_t)
            );
        }
    }

    /// Bucket index of dim `d` in one token's packed codes (np.packbits order,
    /// MSB-first within each byte) — the transparent decode the tests trust.
    fn code_at(tok: &[u8], d: usize, nbits: usize) -> usize {
        let per = 8 / nbits;
        let shift = 8 - nbits * (d % per + 1);
        ((tok[d / per] >> shift) as usize) & ((1 << nbits) - 1)
    }

    #[test]
    fn residual_identity_matches_float_reference() {
        // Dequantize everything (query rows, LUT weights) and recompute each
        // rung's score with a transparent float loop; only f32 association
        // error may separate them.
        for &(nbits, _, dispatched, _, _) in &R_FAMILY {
            let (nq, nd, dim, k) = (8, 20, 128, 16);
            let (q8, lut, codes, cids, cdot_t) = setup_r(nq, nd, dim, k, nbits, 5);
            let fast = dispatched(&q8, &lut, &codes, &cids, &cdot_t);
            let pd = dim * nbits / 8;
            let mut reference = 0.0f64;
            for qi in 0..nq {
                let mut best = f64::NEG_INFINITY;
                for (d, &cid) in cids.iter().enumerate() {
                    let tok = &codes[d * pd..(d + 1) * pd];
                    let mut dot = 0.0f64;
                    for dd in 0..dim {
                        let qv = q8.values[qi * dim + dd] as f64 * q8.scales[qi] as f64;
                        dot += qv * (lut.values[code_at(tok, dd, nbits)] as f64 * lut.scale as f64);
                    }
                    let s = dot + cdot_t[cid as usize * nq + qi] as f64;
                    if s > best {
                        best = s;
                    }
                }
                reference += best;
            }
            assert!(
                (reference as f32 - fast).abs() < 1e-2,
                "nbits={nbits}: {reference} vs {fast}"
            );
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
