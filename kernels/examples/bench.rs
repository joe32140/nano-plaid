//! Microbenchmark for the kernel ladder. Run NATIVE:
//!
//!   cargo run --release --example bench
//!
//! Two ways this exact benchmark has produced wrong numbers, both worth
//! internalizing before trusting any number it prints:
//!
//! 1. Dead-code elimination. Each kernel's result is discarded, so without
//!    `black_box` LLVM deletes the inner loop of whatever it can fully
//!    inline — an early version reported a 43× "speedup" that was really a
//!    no-op. If a benchmark number looks too good, read the disassembly.
//!
//! 2. Rosetta. On an Apple Silicon Mac with an x86_64 Rust toolchain
//!    installed, `cargo run` silently builds an x86_64 binary and emulates
//!    it (~7× slower, and the NEON rung never runs — the autovec rung
//!    dispatches instead). `rustup show` tells you; so does a disassembly
//!    with zero `sdot` instructions. Fix: `rustup default stable-aarch64-apple-darwin`
//!    or pass `--target aarch64-apple-darwin`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use nanoplaid_kernels::*;

const DIM: usize = 128;
const N_DOCS: usize = 2000;
const DOC_TOKENS: usize = 80;
const QUERY_TOKENS: usize = 32;
const REPS: usize = 20;

fn randf(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

fn best_of<F: Fn() -> f32>(f: F) -> Duration {
    let mut acc = black_box(f()); // warmup
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..REPS {
        let t = Instant::now();
        acc += black_box(f());
        best = best.min(t.elapsed());
    }
    black_box(acc);
    best
}

fn main() {
    let mut s = 42u64;
    let qf: Vec<f32> = (0..QUERY_TOKENS * DIM).map(|_| randf(&mut s)).collect();
    let docs: Vec<Vec<u8>> = (0..N_DOCS)
        .map(|_| {
            let d: Vec<f32> = (0..DOC_TOKENS * DIM).map(|_| randf(&mut s)).collect();
            binarize(&d, DIM)
        })
        .collect();
    let q = quantize_query_i8(&qf, DIM);
    // Rung 1 reference scores dequantized ±1 docs in f32.
    let docs_f32: Vec<Vec<f32>> = docs.iter().map(|b| signs_pm1(b, DIM)).collect();
    let q_deq: Vec<f32> = q
        .values
        .chunks_exact(DIM)
        .zip(&q.scales)
        .flat_map(|(row, &sc)| row.iter().map(move |&v| v as f32 * sc))
        .collect();

    println!(
        "kernel ladder: {QUERY_TOKENS}-token query vs {N_DOCS} docs x {DOC_TOKENS} tokens, dim={DIM}"
    );
    #[cfg(target_arch = "aarch64")]
    println!(
        "aarch64: dotprod={} i8mm={}\n",
        std::arch::is_aarch64_feature_detected!("dotprod"),
        std::arch::is_aarch64_feature_detected!("i8mm")
    );

    let t1 = best_of(|| docs_f32.iter().map(|d| maxsim_f32(&q_deq, d, DIM)).sum());
    let t2 = best_of(|| docs.iter().map(|b| maxsim_scalar(&q, b, DIM)).sum());
    let t3 = best_of(|| docs.iter().map(|b| maxsim_autovec(&q, b, DIM)).sum());
    // Rung 4/5: the fused doc-token-outer kernels — whichever this CPU supports
    // (aarch64: SDOT, and SMMLA where i8mm exists; x86_64: AVX2 SAD). Probe once
    // so unsupported kernels cost nothing, then time the ones that ran.
    let t_sdot = maxsim_sdot(&q, &docs[0], DIM)
        .map(|_| best_of(|| docs.iter().map(|b| maxsim_sdot(&q, b, DIM).unwrap()).sum()));
    let t_smmla = maxsim_smmla(&q, &docs[0], DIM)
        .map(|_| best_of(|| docs.iter().map(|b| maxsim_smmla(&q, b, DIM).unwrap()).sum()));
    let t_avx2 = maxsim_avx2(&q, &docs[0], DIM)
        .map(|_| best_of(|| docs.iter().map(|b| maxsim_avx2(&q, b, DIM).unwrap()).sum()));

    let us = |d: Duration| d.as_secs_f64() * 1e6 / N_DOCS as f64;
    let rel = |d: Duration| t1.as_secs_f64() / d.as_secs_f64();
    let line = |name: &str, t: Duration| {
        println!("  {name}  {:>8.3} us/doc   {:>5.2}x", us(t), rel(t));
    };
    println!("per-doc latency (lower is better; speedup vs rung 1):");
    line("rung 1  f32 reference   ", t1);
    line("rung 2  2P-T scalar     ", t2);
    line("rung 3  autovectorized  ", t3);
    let mut any_fused = false;
    for (name, t) in [
        ("rung 4  fused NEON SDOT ", t_sdot),
        ("rung 5  fused NEON SMMLA", t_smmla),
        ("rung 4  fused AVX2 SAD  ", t_avx2),
    ] {
        if let Some(t) = t {
            line(name, t);
            any_fused = true;
        }
    }
    if !any_fused {
        println!("  (no fused kernel for this CPU — needs aarch64 dotprod or x86 avx2)");
    }
    if let (Some(a), Some(b)) = (t_sdot, t_smmla) {
        println!(
            "\n  SMMLA vs SDOT: {:.2}x",
            a.as_secs_f64() / b.as_secs_f64()
        );
    }

    // ── the fused residual family: the LUT identity ───────────────────────
    // Same doc set, now as packed residual codes + a centroid id per token
    // (4-bit, then 2-bit, then 1-bit — random bytes are valid codes for any
    // nbits). The float baseline for this scheme is decompress + BLAS GEMM,
    // which this bench can't reproduce dependency-free; rung 1's f32 loop
    // stands in as the common yardstick, so compare the residual rungs to
    // each other and to the binary fused kernel above.
    let mut s2 = 7u64;
    const K: usize = 4096;
    let r4_codes: Vec<Vec<u8>> = (0..N_DOCS)
        .map(|_| {
            (0..DOC_TOKENS * DIM / 2)
                .map(|_| ((randf(&mut s2) + 0.5) * 255.99) as u8)
                .collect()
        })
        .collect();
    let r2_codes: Vec<Vec<u8>> = (0..N_DOCS)
        .map(|_| {
            (0..DOC_TOKENS * DIM / 4)
                .map(|_| ((randf(&mut s2) + 0.5) * 255.99) as u8)
                .collect()
        })
        .collect();
    let r1_codes: Vec<Vec<u8>> = (0..N_DOCS)
        .map(|_| {
            (0..DOC_TOKENS * DIM / 8)
                .map(|_| ((randf(&mut s2) + 0.5) * 255.99) as u8)
                .collect()
        })
        .collect();
    let r4_cids: Vec<Vec<u32>> = (0..N_DOCS)
        .map(|_| {
            (0..DOC_TOKENS)
                .map(|_| ((randf(&mut s2) + 0.5) * (K as f32 - 0.01)) as u32)
                .collect()
        })
        .collect();
    let cdot_t: Vec<f32> = (0..K * QUERY_TOKENS).map(|_| randf(&mut s2)).collect();
    let mut lut_vals = [0i8; 16];
    for v in lut_vals.iter_mut() {
        *v = (randf(&mut s2) * 254.0) as i8;
    }
    let lut = LutI8 {
        values: lut_vals,
        scale: 0.0031,
    };

    // One timing triple per nbits rung: scalar reference, scalar-fold fused,
    // and vectorized-fold fused (both fused kernels if this CPU supports them).
    macro_rules! time_rung {
        ($codes:expr, $scalar:path, $fused:path, $vfold:path) => {{
            let ts = best_of(|| {
                $codes
                    .iter()
                    .zip(&r4_cids)
                    .map(|(c, ids)| $scalar(&q, &lut, c, ids, &cdot_t))
                    .sum()
            });
            let tf = $fused(&q, &lut, &$codes[0], &r4_cids[0], &cdot_t).map(|_| {
                best_of(|| {
                    $codes
                        .iter()
                        .zip(&r4_cids)
                        .map(|(c, ids)| $fused(&q, &lut, c, ids, &cdot_t).unwrap())
                        .sum()
                })
            });
            let tv = $vfold(&q, &lut, &$codes[0], &r4_cids[0], &cdot_t).map(|_| {
                best_of(|| {
                    $codes
                        .iter()
                        .zip(&r4_cids)
                        .map(|(c, ids)| $vfold(&q, &lut, c, ids, &cdot_t).unwrap())
                        .sum()
                })
            });
            (ts, tf, tv)
        }};
    }
    let tr4 = time_rung!(
        r4_codes,
        maxsim_r4_scalar,
        maxsim_r4_fused,
        maxsim_r4_vfold_fused
    );
    let tr2 = time_rung!(
        r2_codes,
        maxsim_r2_scalar,
        maxsim_r2_fused,
        maxsim_r2_vfold_fused
    );
    let tr1 = time_rung!(
        r1_codes,
        maxsim_r1_scalar,
        maxsim_r1_fused,
        maxsim_r1_vfold_fused
    );

    #[cfg(target_arch = "aarch64")]
    let fused_names = [
        ("r4      fused NEON tbl  ", "r4      + vec fold      "),
        ("r2      fused NEON tbl  ", "r2      + vec fold      "),
        ("r1      fused NEON sdot ", "r1      + vec fold      "),
    ];
    #[cfg(target_arch = "x86_64")]
    let fused_names = [
        ("r4      fused AVX2 shufb", "r4      + vec fold      "),
        ("r2      fused AVX2 shufb", "r2      + vec fold      "),
        ("r1      fused AVX2 SAD  ", "r1      + vec fold      "),
    ];
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let fused_names = [("", ""); 3];

    println!("\nfused residual family (LUT identity; packed codes + centroid ids):");
    println!("  each rung: scalar-fold fused, then the vectorized-fold twin.");
    for (i, (ts, tf, tv)) in [tr4, tr2, tr1].into_iter().enumerate() {
        let n = [4, 2, 1][i];
        line(&format!("r{n}      scalar reference"), ts);
        match tf {
            Some(t) => line(fused_names[i].0, t),
            None => println!("  (no fused residual-{n} kernel for this CPU)"),
        }
        if let Some(t) = tv {
            line(fused_names[i].1, t);
        }
    }

    // Experimental r4 rung: the transpose-reduce fold (NEON only). Timed
    // against r4's vfold above — does eliminating the per-row horizontal
    // reduce actually help, or was it already hidden under the SDOTs?
    if maxsim_r4_tr_fused(&q, &lut, &r4_codes[0], &r4_cids[0], &cdot_t).is_some() {
        let t = best_of(|| {
            r4_codes
                .iter()
                .zip(&r4_cids)
                .map(|(c, ids)| maxsim_r4_tr_fused(&q, &lut, c, ids, &cdot_t).unwrap())
                .sum()
        });
        line("r4      + transpose-reduce", t);
    }
}
