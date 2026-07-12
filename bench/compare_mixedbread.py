#!/usr/bin/env python3
"""Compress+fused vs decompress+GEMM, per residual route, on real SciFact.

The residual schemes (nbits 4/2/1) store the SAME compressed codes; the only
thing that changes is HOW you score them. Three ways, head to head:

  1. baseline GEMM  — decode codes -> f32 tokens, then a plain per-doc BLAS
                      `sgemm` + max-fold (numpy). The naive path every system
                      starts with.
  2. mixedbread     — decode codes -> f32 tokens, then mixedbread-ai/maxsim-cpu:
                      a batched `sgemm` + hand-vectorized fold (their upstream
                      PyPI wheel, unmodified). The OPTIMIZED GEMM.
  3. ours (fused)   — score the codes directly through `nanoplaid_kernels`, no
                      decode at all. Our optimization.

(1) and (2) score the identical f32 reconstruction, so their NDCG is the same
and only their speed differs — that gap is the GEMM optimization. (3) scores an
int8 view of the same codes, so it trades a little NDCG for never materializing
the f32: (1)/(2) must expand every token to 512 B to score, (3) stays at the
stored 64/32/16 B. Scoring is exhaustive (no ANN) so the number is the kernel.

    CARGO_BUILD_TARGET=aarch64-apple-darwin pip install maxsim-cpu numpy .
    python bench/compare_mixedbread.py [data/scifact] [--time-queries 30]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# nanoplaid.py is the repo-root numpy engine (encoders + specs), not installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maxsim_cpu
import nanoplaid as npl
import nanoplaid_kernels as nk


def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def ndcg_at_k(ranked_ids, rel, k=10):
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def best_of(fn, reps):
    fn()  # warmup
    return min((_timed(fn) for _ in range(reps)))


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def naive_gemm(qf, recon_list):
    """Baseline: one BLAS sgemm per doc, then max per query token, summed."""
    out = np.empty(len(recon_list), np.float32)
    for i, tok in enumerate(recon_list):
        out[i] = (qf @ tok.T).max(axis=1).sum()
    return out


def mean_ndcg(score_all, n_qry, q_off, queries, qrels, query_ids, corpus_ids):
    vals = []
    for qi in range(n_qry):
        rel = qrels.get(query_ids[qi], {})
        if not rel:
            continue
        qf = np.ascontiguousarray(queries[q_off[qi] : q_off[qi + 1]])
        scores = np.asarray(score_all(qf))
        order = np.argsort(-scores)[:10]
        vals.append(ndcg_at_k([corpus_ids[j] for j in order], rel, 10))
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data/scifact")
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--queries", type=int, default=0, help="cap #queries for NDCG")
    ap.add_argument("--time-queries", type=int, default=30, help="#queries for timing")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    d = args.data
    corpus = l2norm(np.ascontiguousarray(np.load(f"{d}/corpus.npy"), dtype=np.float32))
    clens = np.load(f"{d}/corpus_lens.npy").astype(np.int64)
    queries = l2norm(np.ascontiguousarray(np.load(f"{d}/queries.npy"), dtype=np.float32))
    qlens = np.load(f"{d}/query_lens.npy").astype(np.int64)
    qrels = json.load(open(f"{d}/qrels.json"))
    corpus_ids = json.load(open(f"{d}/corpus_ids.json"))
    query_ids = json.load(open(f"{d}/query_ids.json"))
    dim = corpus.shape[1]

    n_docs = len(clens) if args.docs <= 0 else min(args.docs, len(clens))
    n_qry = len(qlens) if args.queries <= 0 else min(args.queries, len(qlens))
    doc_off = np.concatenate([[0], np.cumsum(clens)])
    q_off = np.concatenate([[0], np.cumsum(qlens)])
    clens = clens[:n_docs]
    total = int(clens.sum())
    corpus = corpus[: doc_off[n_docs]]
    doc_off = doc_off[: n_docs + 1]
    lens_i64 = clens.astype(np.int64)
    tq = min(n_qry, args.time_queries)

    docs_orig = [np.ascontiguousarray(corpus[doc_off[i] : doc_off[i + 1]]) for i in range(n_docs)]

    print(f"building shared codebook over {total} tokens ...", flush=True)
    idx4 = npl.build(corpus, clens, scheme="residual", nbits=4, seed=0, verbose=False)
    cents = idx4.centroids
    idxs = {
        4: idx4,
        2: npl.build(corpus, clens, scheme="residual", nbits=2, centroids=cents, seed=0, verbose=False),
        1: npl.build(corpus, clens, scheme="residual", nbits=1, centroids=cents, seed=0, verbose=False),
    }

    # Exact f32 ceiling: mixedbread on the ORIGINAL (uncompressed) embeddings.
    print("scoring f32 ceiling (mixedbread on original embeddings) ...", flush=True)
    ceiling = mean_ndcg(
        lambda qf: maxsim_cpu.maxsim_scores_variable(qf, docs_orig),
        n_qry, q_off, queries, qrels, query_ids, corpus_ids,
    )

    rows = []
    for nbits in (4, 2, 1):
        idx = idxs[nbits]
        lut = npl.quantize_lut(idx.codec)
        stored = dim * nbits // 8
        print(f"nbits={nbits}: decoding {total} tokens to f32 ...", flush=True)
        recon = npl.decode_rows(idx, np.arange(total)).astype(np.float32)
        recon_list = [np.ascontiguousarray(recon[doc_off[i] : doc_off[i + 1]]) for i in range(n_docs)]

        def ours(qf, idx=idx, lut=lut, nbits=nbits):
            cdot_t = np.ascontiguousarray((qf @ cents.T).T)
            return nk.maxsim_docs_lut(
                qf, idx.payload, idx.codes.astype(np.uint32), cdot_t,
                lens_i64, lut.values, float(lut.scale), nbits,
            )

        methods = [
            ("baseline GEMM (per-doc)", stored, 512, lambda qf: naive_gemm(qf, recon_list), False),
            ("mixedbread GEMM (var API)", stored, 512, lambda qf: maxsim_cpu.maxsim_scores_variable(qf, recon_list), False),
            ("ours: fused on codes", stored, stored, ours, True),
        ]
        # NDCG: baseline==mixedbread (same reconstruction), so compute the GEMM
        # NDCG once (via mixedbread) and reuse it for the baseline row; ours
        # differs (int8), compute separately.
        ndcg_gemm = mean_ndcg(methods[1][3], n_qry, q_off, queries, qrels, query_ids, corpus_ids)
        ndcg_ours = mean_ndcg(methods[2][3], n_qry, q_off, queries, qrels, query_ids, corpus_ids)
        ndcgs = [ndcg_gemm, ndcg_gemm, ndcg_ours]

        for (name, sb, scoreb, fn, _mine), ndcg in zip(methods, ndcgs):
            def run(fn=fn):
                for qi in range(tq):
                    qf = np.ascontiguousarray(queries[q_off[qi] : q_off[qi + 1]])
                    np.asarray(fn(qf)).sum()
            t = best_of(run, args.reps)
            rows.append((nbits, name, sb, scoreb, t * 1e6 / (tq * n_docs), ndcg))
        del recon, recon_list

    # ── report ──────────────────────────────────────────────────────────────
    print(
        f"\nSciFact residual routes, exhaustive MaxSim: NDCG over {n_qry} queries, "
        f"timing over {tq} x {n_docs} docs ({total} tokens), dim={dim}\n"
        f"native arm64 + Accelerate, single-thread. score-B/tok = bytes touched "
        f"to SCORE a token (GEMM must expand codes to f32).\n"
        f"f32 ceiling (mixedbread on uncompressed): NDCG@10 = {ceiling:.4f}\n"
    )
    hdr = f"  {'route':<7}{'method':<26}{'store':>7}{'score':>7}{'us/doc':>9}{'vs base':>9}{'NDCG@10':>9}{'retain':>8}"
    print(hdr)
    per_route_base = {}
    for nbits, name, sb, scoreb, us, ndcg in rows:
        if "baseline" in name:
            per_route_base[nbits] = us
        base = per_route_base[nbits]
        print(
            f"  r{nbits:<6}{name:<26}{sb:>6}B{scoreb:>6}B{us:>9.3f}"
            f"{base / us:>8.2f}x{ndcg:>9.4f}{(ndcg / ceiling if ceiling else 0):>7.1%}"
        )
        if "ours" in name:
            print()


if __name__ == "__main__":
    main()
