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
}
