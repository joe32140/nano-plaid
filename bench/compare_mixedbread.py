#!/usr/bin/env python3
"""Our compressed fused MaxSim vs mixedbread-ai/maxsim-cpu, on real SciFact.

Two sides of one line. mixedbread's `maxsim-cpu` is the *decompress + GEMM*
engine: full f32 document embeddings, a batched BLAS sgemm, and a hand-
vectorized max-fold (their Rust, their Accelerate/libxsmm build — the upstream
PyPI wheel, called here unmodified). Ours is *compress + fused kernel*: the
document is packed sign bits (binary) or residual codes, scored with no
decompression through `nanoplaid_kernels` (the pyo3 bridge over the same
kernels this repo's class dissects).

The comparison is deliberately apples-to-apples on the WORK and honest about
the TRADEOFF: both score every query against every document exhaustively (no
ANN, no candidate cap), so the number is the scoring kernel and nothing else.
We report the three axes that actually differ — per-doc latency, bytes/token,
and NDCG@10 against the real qrels (mixedbread's exact f32 score is the quality
ceiling; ours pays compression error for 8-32x less memory).

    pip install maxsim-cpu numpy .      # . builds nanoplaid_kernels (bridge)
    python bench/compare_mixedbread.py [data/scifact] [--docs N] [--queries M]

Note on Apple Silicon: build the bridge for arm64
(`CARGO_BUILD_TARGET=aarch64-apple-darwin pip install .`) or the NEON kernels
compile out under Rosetta and our side benches the autovec fallback — the same
trap the kernel class warns about, and it would understate our kernels here.
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
    """NDCG@k for one query. `rel` maps doc_id -> graded relevance."""
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def best_of(fn, reps=3):
    """Return (best wall time over `reps`, last result)."""
    out = fn()  # warmup
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data/scifact")
    ap.add_argument("--docs", type=int, default=0, help="cap #docs (0 = all)")
    ap.add_argument("--queries", type=int, default=0, help="cap #queries (0 = all)")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    d = args.data
    corpus = np.ascontiguousarray(np.load(f"{d}/corpus.npy"), dtype=np.float32)
    clens = np.load(f"{d}/corpus_lens.npy").astype(np.int64)
    queries = np.ascontiguousarray(np.load(f"{d}/queries.npy"), dtype=np.float32)
    qlens = np.load(f"{d}/query_lens.npy").astype(np.int64)
    qrels = json.load(open(f"{d}/qrels.json"))
    corpus_ids = json.load(open(f"{d}/corpus_ids.json"))
    query_ids = json.load(open(f"{d}/query_ids.json"))
    dim = corpus.shape[1]

    # Optional caps for a quick run.
    n_docs = len(clens) if args.docs <= 0 else min(args.docs, len(clens))
    n_qry = len(qlens) if args.queries <= 0 else min(args.queries, len(qlens))

    # Normalize once; every engine sees identical vectors (mixedbread expects
    # normalized input, and sign bits / int8 are scale-robust, so this is the
    # consistent common ground rather than a thumb on the scale).
    corpus = l2norm(corpus)
    queries = l2norm(queries)

    doc_off = np.concatenate([[0], np.cumsum(clens)])
    q_off = np.concatenate([[0], np.cumsum(qlens)])
    clens = clens[:n_docs]
    doc_tokens = int(clens.sum())
    corpus = corpus[: doc_off[n_docs]]
    doc_off = doc_off[: n_docs + 1]

    # mixedbread wants a list of [len, dim] f32 arrays (variable length).
    docs_list = [
        np.ascontiguousarray(corpus[doc_off[i] : doc_off[i + 1]]) for i in range(n_docs)
    ]

    # Our binary payload: packed sign bits, one [len, dim/8] block per doc,
    # concatenated, plus the per-doc lengths the bridge folds over.
    payload = npl.binarize(corpus)
    lens_i64 = clens.astype(np.int64)

    # Our residual indexes (share one k-means so stage 1 is identical; only the
    # payload codec differs). cdot per query is q @ centroids.T.
    print(f"building residual codebooks over {doc_tokens} tokens ...", flush=True)
    idx4 = npl.build(corpus, clens, scheme="residual", nbits=4, seed=0, verbose=False)
    idx2 = npl.build(
        corpus, clens, scheme="residual", nbits=2, centroids=idx4.centroids,
        seed=0, verbose=False,
    )
    lut4, lut2 = npl.quantize_lut(idx4.codec), npl.quantize_lut(idx2.codec)
    cents = idx4.centroids  # [K, dim]

    def score_query(qi, engine):
        qf = np.ascontiguousarray(queries[q_off[qi] : q_off[qi + 1]])
        if engine == "mixedbread":
            return maxsim_cpu.maxsim_scores_variable(qf, docs_list)
        if engine == "binary":
            return nk.maxsim_docs(qf, payload, lens_i64)
        # residual r4 / r2
        idx, lut, nbits = (idx4, lut4, 4) if engine == "r4" else (idx2, lut2, 2)
        cdot_t = np.ascontiguousarray((qf @ cents.T).T)  # [K, nq]
        return nk.maxsim_docs_lut(
            qf, idx.payload, idx.codes.astype(np.uint32), cdot_t,
            lens_i64, lut.values, float(lut.scale), nbits,
        )

    engines = ["mixedbread", "binary", "r4", "r2"]
    bpt = {  # resident bytes/token (payload only; centroid id is shared overhead)
        "mixedbread": dim * 4, "binary": dim // 8, "r4": dim * 4 // 8, "r2": dim * 2 // 8,
    }

    # Quality: NDCG@10 vs qrels, exhaustive ranking per query.
    ndcg = {e: [] for e in engines}
    for qi in range(n_qry):
        rel = qrels.get(query_ids[qi], {})
        if not rel:
            continue
        for e in engines:
            scores = np.asarray(score_query(qi, e))
            order = np.argsort(-scores)[:10]
            ranked = [corpus_ids[j] for j in order]
            ndcg[e].append(ndcg_at_k(ranked, rel, 10))

    # Speed: score all queries once, per engine, best-of reps -> µs/doc.
    latency = {}
    for e in engines:
        def run():
            s = 0.0
            for qi in range(n_qry):
                s += float(np.asarray(score_query(qi, e)).sum())
            return s
        t, _ = best_of(run, args.reps)
        latency[e] = t * 1e6 / (n_qry * n_docs)  # µs per (query, doc)

    ceiling = float(np.mean(ndcg["mixedbread"])) if ndcg["mixedbread"] else 0.0
    print(
        f"\nSciFact exhaustive MaxSim: {n_qry} queries x {n_docs} docs "
        f"({doc_tokens} doc tokens), dim={dim}\n"
        f"both engines native arm64 + Accelerate; per-doc = per (query, doc)\n"
    )
    print(f"  {'engine':<26}{'B/tok':>7}{'vs f32':>8}{'us/doc':>9}{'vs GEMM':>9}{'NDCG@10':>9}{'retention':>11}")
    tg = latency["mixedbread"]
    names = {
        "mixedbread": "mixedbread f32 GEMM",
        "binary": "ours: binary 1-bit fused",
        "r4": "ours: residual-4 fused",
        "r2": "ours: residual-2 fused",
    }
    for e in engines:
        m = float(np.mean(ndcg[e])) if ndcg[e] else 0.0
        print(
            f"  {names[e]:<26}{bpt[e]:>7}{dim * 4 // bpt[e]:>7}x"
            f"{latency[e]:>9.3f}{tg / latency[e]:>8.2f}x{m:>9.4f}"
            f"{(m / ceiling if ceiling else 0):>10.1%}"
        )
    print("\n(mixedbread = exact f32 ceiling; ours trades compression error for memory.)")


if __name__ == "__main__":
    main()
