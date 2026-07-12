//! The kernel ladder, all in one run.
//!
//!   cargo run --release --example ladder            # full, comparable to README
//!   cargo run --release --example ladder -- quick   # 4x smaller, for a smoke test
//!
//! `bench.rs` times individual rungs; this example lines the WHOLE ladder up on
//! ONE dataset so every number shares a baseline and the ablation reads top to
//! bottom. It runs only the Rust kernels (no numpy), probes each fused rung so
//! unavailable ones are marked rather than skipped silently, and checks that
//! every integer rung is bit-identical to its scalar reference before trusting
//! any speedup.
//!
//! Two traps `bench.rs` documents apply here verbatim: without `black_box` LLVM
//! deletes a discarded kernel's inner loop (a fake 43x), and on Apple Silicon a
//! stray x86_64 toolchain builds an x86 binary that Rosetta emulates (the NEON
//! rungs then read "n/a"). Use `--target aarch64-apple-darwin` if the fused
//! aarch64 rungs unexpectedly show as unavailable.

use std::hint::black_box;
use std::time::{Duration, Instant};

use nanoplaid_kernels::*;

const DIM: usize = 128;
const QUERY_TOKENS: usize = 32;

fn randf(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

fn best_of<F: Fn() -> f32>(reps: usize, f: F) -> Duration {
    let mut acc = black_box(f()); // warmup
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..reps {
        let t = Instant::now();
        acc += black_box(f());
        best = best.min(t.elapsed());
    }
    black_box(acc);
    best
}

/// One line of a ladder: a name, a timing (None = not available on this CPU),
/// its total score (for the bit-identity check), and whether it is the float
/// reference being APPROXIMATED (so it is exempt from the identity check).
struct Rung {
    name: String,
    time: Option<Duration>,
    total: Option<f32>,
    approx: bool,
}

/// Time a rung if `avail`, else record it as unavailable. `run` returns the
/// summed score over all docs; it is called once for the total and again inside
/// `best_of`, so it must be a pure closure.
macro_rules! rung {
    ($v:expr, $reps:expr, $name:expr, $avail:expr, $approx:expr, $run:expr) => {{
        if $avail {
            let total = ($run)();
            let time = best_of($reps, &$run);
            $v.push(Rung {
                name: $name.into(),
                time: Some(time),
                total: Some(total),
                approx: $approx,
            });
        } else {
            $v.push(Rung {
                name: $name.into(),
                time: None,
                total: None,
                approx: $approx,
            });
        }
    }};
}

/// Print one scheme's ladder. `t_ref` is the shared f32-reference time (the
/// common yardstick for the `xf32` column); `id_ref` is the scalar total every
/// non-`approx` rung must match exactly.
fn print_ladder(title: &str, bytes_per_tok: usize, rungs: &[Rung], t_ref: Duration, n_docs: usize) {
    println!("\n═══ {title}  ({bytes_per_tok} B/token) ══════════════════════");
    println!(
        "  {:<30}{:>10}{:>9}{:>9}",
        "rung", "us/doc", "xf32", "xprev"
    );
    let us = |d: Duration| d.as_secs_f64() * 1e6 / n_docs as f64;
    let tref = t_ref.as_secs_f64();
    let mut prev: Option<Duration> = None;
    for (i, r) in rungs.iter().enumerate() {
        match r.time {
            Some(t) => {
                let xf32 = format!("{:.1}x", tref / t.as_secs_f64());
                let xprev = match prev {
                    Some(p) => format!("{:.2}x", p.as_secs_f64() / t.as_secs_f64()),
                    None => "—".to_string(),
                };
                println!(
                    "  {:>2} {:<27}{:>10.2}{:>9}{:>9}",
                    i + 1,
                    r.name,
                    us(t),
                    xf32,
                    xprev
                );
                prev = Some(t);
            }
            None => {
                println!(
                    "  {:>2} {:<27}{:>10}{:>9}{:>9}",
                    i + 1,
                    r.name,
                    "n/a",
                    "—",
                    "—"
                );
            }
        }
    }
    // Bit-identity: every integer rung must equal the scalar reference exactly.
    let id_ref = rungs
        .iter()
        .find(|r| !r.approx && r.total.is_some())
        .and_then(|r| r.total);
    if let Some(rref) = id_ref {
        let mut checked = 0;
        let mut bad = Vec::new();
        for r in rungs.iter().filter(|r| !r.approx) {
            if let Some(t) = r.total {
                checked += 1;
                if t != rref {
                    bad.push(format!("{} ({t} vs {rref})", r.name));
                }
            }
        }
        if bad.is_empty() {
            println!("  bit-identity: OK — {checked}/{checked} integer rungs match the scalar ref");
        } else {
            println!("  bit-identity: MISMATCH — {}", bad.join(", "));
        }
    }
}

fn best_available(rungs: &[Rung]) -> Option<(&str, Duration)> {
    rungs
        .iter()
        .filter_map(|r| r.time.map(|t| (r.name.as_str(), t)))
        .min_by_key(|(_, t)| *t)
}

fn main() {
    let quick = std::env::args().any(|a| a == "quick" || a == "--quick");
    let (n_docs, doc_tokens, reps) = if quick { (500, 80, 5) } else { (2000, 80, 20) };

    // ── shared data: one query, one doc set, three residual codebooks ──────
    let mut s = 42u64;
    let qf: Vec<f32> = (0..QUERY_TOKENS * DIM).map(|_| randf(&mut s)).collect();
    let docs: Vec<Vec<u8>> = (0..n_docs)
        .map(|_| {
            let d: Vec<f32> = (0..doc_tokens * DIM).map(|_| randf(&mut s)).collect();
            binarize(&d, DIM)
        })
        .collect();
    let q = quantize_query_i8(&qf, DIM);
    let docs_f32: Vec<Vec<f32>> = docs.iter().map(|b| signs_pm1(b, DIM)).collect();
    let q_deq: Vec<f32> = q
        .values
        .chunks_exact(DIM)
        .zip(&q.scales)
        .flat_map(|(row, &sc)| row.iter().map(move |&v| v as f32 * sc))
        .collect();

    // Residual codes: random bytes are valid codes for any nbits. The centroid
    // matrix and LUT match bench.rs so the numbers are directly comparable.
    let mut s2 = 7u64;
    const K: usize = 4096;
    let codes_r = |bytes_per_tok: usize, s2: &mut u64| -> Vec<Vec<u8>> {
        (0..n_docs)
            .map(|_| {
                (0..doc_tokens * bytes_per_tok)
                    .map(|_| ((randf(s2) + 0.5) * 255.99) as u8)
                    .collect()
            })
            .collect()
    };
    let r4_codes = codes_r(DIM / 2, &mut s2);
    let r2_codes = codes_r(DIM / 4, &mut s2);
    let r1_codes = codes_r(DIM / 8, &mut s2);
    let cids: Vec<Vec<u32>> = (0..n_docs)
        .map(|_| {
            (0..doc_tokens)
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

    println!(
        "kernel ladder: {QUERY_TOKENS}-token query vs {n_docs} docs x {doc_tokens} tokens, dim={DIM}, {reps} reps"
    );
    #[cfg(target_arch = "aarch64")]
    println!(
        "aarch64: dotprod={} i8mm={}",
        std::arch::is_aarch64_feature_detected!("dotprod"),
        std::arch::is_aarch64_feature_detected!("i8mm")
    );
    #[cfg(target_arch = "x86_64")]
    println!(
        "x86_64: avx2={} avx512vnni={}",
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("avx512vnni")
    );
    println!("columns: xf32 = vs the rung-1 f32 reference; xprev = vs the previous live rung");

    // ── binary ladder ──────────────────────────────────────────────────────
    let mut bin = Vec::new();
    rung!(bin, reps, "f32 reference", true, true, || docs_f32
        .iter()
        .map(|d| maxsim_f32(&q_deq, d, DIM))
        .sum());
    let t_ref = bin[0].time.unwrap();
    rung!(bin, reps, "scalar 2P-T", true, false, || docs
        .iter()
        .map(|b| maxsim_scalar(&q, b, DIM))
        .sum());
    rung!(bin, reps, "autovec masks", true, false, || docs
        .iter()
        .map(|b| maxsim_autovec(&q, b, DIM))
        .sum());
    rung!(
        bin,
        reps,
        "fused NEON SDOT",
        maxsim_sdot(&q, &docs[0], DIM).is_some(),
        false,
        || docs.iter().map(|b| maxsim_sdot(&q, b, DIM).unwrap()).sum()
    );
    rung!(
        bin,
        reps,
        "fused NEON SMMLA",
        maxsim_smmla(&q, &docs[0], DIM).is_some(),
        false,
        || docs.iter().map(|b| maxsim_smmla(&q, b, DIM).unwrap()).sum()
    );
    rung!(
        bin,
        reps,
        "fused AVX2 SAD",
        maxsim_avx2(&q, &docs[0], DIM).is_some(),
        false,
        || docs.iter().map(|b| maxsim_avx2(&q, b, DIM).unwrap()).sum()
    );
    rung!(
        bin,
        reps,
        "fused AVX-512 VNNI",
        maxsim_avx512(&q, &docs[0], DIM).is_some(),
        false,
        || docs
            .iter()
            .map(|b| maxsim_avx512(&q, b, DIM).unwrap())
            .sum()
    );
    print_ladder("binary  int8 x 1-bit", 16, &bin, t_ref, n_docs);

    // ── residual ladders (nbits = 4, 2, 1) ─────────────────────────────────
    // Same five ablation steps per scheme: scalar reference, the platform's
    // fused kernel (scalar fold), the vectorized fold, the transpose-reduce
    // (NEON), and the AVX-512 VNNI kernel (x86). Each is bit-identical.
    macro_rules! resid_ladder {
        ($title:expr, $bpt:expr, $codes:expr, $scalar:path, $fused:path, $vfold:path, $tr:path, $avx512:path) => {{
            let mut v = Vec::new();
            let c0 = &$codes[0];
            let id0 = &cids[0];
            rung!(v, reps, "scalar reference", true, false, || $codes
                .iter()
                .zip(&cids)
                .map(|(c, ids)| $scalar(&q, &lut, c, ids, &cdot_t))
                .sum());
            rung!(
                v,
                reps,
                "fused (scalar fold)",
                $fused(&q, &lut, c0, id0, &cdot_t).is_some(),
                false,
                || $codes
                    .iter()
                    .zip(&cids)
                    .map(|(c, ids)| $fused(&q, &lut, c, ids, &cdot_t).unwrap())
                    .sum()
            );
            rung!(
                v,
                reps,
                "+ vectorized fold",
                $vfold(&q, &lut, c0, id0, &cdot_t).is_some(),
                false,
                || $codes
                    .iter()
                    .zip(&cids)
                    .map(|(c, ids)| $vfold(&q, &lut, c, ids, &cdot_t).unwrap())
                    .sum()
            );
            rung!(
                v,
                reps,
                "+ transpose-reduce",
                $tr(&q, &lut, c0, id0, &cdot_t).is_some(),
                false,
                || $codes
                    .iter()
                    .zip(&cids)
                    .map(|(c, ids)| $tr(&q, &lut, c, ids, &cdot_t).unwrap())
                    .sum()
            );
            rung!(
                v,
                reps,
                "AVX-512 VNNI",
                $avx512(&q, &lut, c0, id0, &cdot_t).is_some(),
                false,
                || $codes
                    .iter()
                    .zip(&cids)
                    .map(|(c, ids)| $avx512(&q, &lut, c, ids, &cdot_t).unwrap())
                    .sum()
            );
            print_ladder($title, $bpt, &v, t_ref, n_docs);
            v
        }};
    }

    let r4 = resid_ladder!(
        "residual-4  LUT16 pshufb",
        64,
        r4_codes,
        maxsim_r4_scalar,
        maxsim_r4_fused,
        maxsim_r4_vfold_fused,
        maxsim_r4_tr_fused,
        maxsim_r4_avx512_fused
    );
    let r2 = resid_ladder!(
        "residual-2  LUT4 pshufb",
        32,
        r2_codes,
        maxsim_r2_scalar,
        maxsim_r2_fused,
        maxsim_r2_vfold_fused,
        maxsim_r2_tr_fused,
        maxsim_r2_avx512_fused
    );
    let r1 = resid_ladder!(
        "residual-1  affine 2P-T",
        16,
        r1_codes,
        maxsim_r1_scalar,
        maxsim_r1_fused,
        maxsim_r1_vfold_fused,
        maxsim_r1_tr_fused,
        maxsim_r1_avx512_fused
    );

    // ── cross-scheme summary: best live rung of each scheme, one baseline ───
    println!("\n═══ best live rung per scheme (shared f32 baseline) ══════════");
    println!(
        "  {:<16}{:>7}{:>26}{:>10}{:>8}",
        "scheme", "B/tok", "rung", "us/doc", "xf32"
    );
    let tref = t_ref.as_secs_f64();
    for (name, bpt, rungs) in [
        ("binary", 16, &bin),
        ("residual-4", 64, &r4),
        ("residual-2", 32, &r2),
        ("residual-1", 16, &r1),
    ] {
        if let Some((rn, t)) = best_available(&rungs[1..]) {
            println!(
                "  {:<16}{:>7}{:>26}{:>10.2}{:>7.1}x",
                name,
                bpt,
                rn,
                t.as_secs_f64() * 1e6 / n_docs as f64,
                tref / t.as_secs_f64()
            );
        }
    }
    println!("\n(shared-VM numbers are noisy; run on an idle machine for stable absolutes.)");
}
