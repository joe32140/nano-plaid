"""Evaluate nanoplaid on a BEIR bundle: NDCG@10, latency, and storage for each
quantization scheme, against the exhaustive float baseline.

    python encode.py --download scifact --out data/scifact   # once, needs GPU-ish
    python eval.py data/scifact

Bundle layout (produced by encode.py): corpus.npy [T, dim] f32 concatenated
token vectors, corpus_lens.npy, corpus_ids.json, queries.npy, query_lens.npy,
query_ids.json, qrels.json {qid: {docid: relevance}}.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import nanoplaid as npl


def load_bundle(path):
    p = Path(path)
    b = {
        "corpus": np.load(p / "corpus.npy"),
        "doc_lens": np.load(p / "corpus_lens.npy"),
        "queries": np.load(p / "queries.npy"),
        "query_lens": np.load(p / "query_lens.npy"),
        "corpus_ids": json.loads((p / "corpus_ids.json").read_text()),
        "query_ids": json.loads((p / "query_ids.json").read_text()),
        "qrels": json.loads((p / "qrels.json").read_text()),
    }
    b["doc_offsets"] = np.concatenate([[0], np.cumsum(b["doc_lens"])[:-1]])
    b["q_offsets"] = np.concatenate([[0], np.cumsum(b["query_lens"])[:-1]])
    return b


def query_slices(b):
    for qid, off, n in zip(b["query_ids"], b["q_offsets"], b["query_lens"]):
        yield qid, b["queries"][off : off + int(n)]


def ndcg_at_k(ranked_doc_ids, rels, k=10):
    """Exponential-gain NDCG (BEIR-style). rels: {doc_id: graded relevance}."""
    dcg = sum((2 ** rels.get(d, 0) - 1) / np.log2(i + 2)
              for i, d in enumerate(ranked_doc_ids[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def run(b, search_fn, k=10):
    """search_fn(q) -> ranked doc indices. Returns (mean NDCG@10, latencies)."""
    ndcgs, lat = [], []
    for qid, q in query_slices(b):
        rels = b["qrels"].get(qid)
        if not rels:
            continue
        t = time.perf_counter()
        top = search_fn(q)
        lat.append(time.perf_counter() - t)
        ndcgs.append(ndcg_at_k([b["corpus_ids"][i] for i in top], rels, k))
    return float(np.mean(ndcgs)), np.array(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--schemes", default="exact,residual4,residual2,binary")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--n-full", type=int, default=1024)
    ap.add_argument("--backend", choices=["numpy", "rust"], default="numpy",
                    help="binary stage-2 scorer; 'rust' needs the kernels extension "
                         "(maturin develop -m kernels/Cargo.toml --release --features python)")
    args = ap.parse_args()

    binary_scorer = make_rust_scorer() if args.backend == "rust" else None

    b = load_bundle(args.bundle)
    dim = b["corpus"].shape[1]
    print(f"{len(b['doc_lens'])} docs, {len(b['corpus'])} tokens, "
          f"{len(b['query_lens'])} queries, dim={dim}, backend={args.backend}\n")

    schemes = args.schemes.split(",")
    header = (f"| scheme     | build s | B/token | NDCG@{args.k} | mean ms | p50 ms | p95 ms |")
    print(header + "\n|" + "|".join("-" * (len(c) - 1) for c in header.split("|")[1:-1]) + "|")

    centroids = None  # trained once, shared by every two-stage scheme
    for scheme in schemes:
        if scheme == "exact":
            n, lat = run(b, lambda q: search_exhaustive_ids(b, q, args.k), args.k)
            row("exact", None, 4 * dim, n, lat, args.k)
            continue
        name, nbits = ("binary", 0) if scheme == "binary" else ("residual", int(scheme[-1]))
        t = time.perf_counter()
        idx = npl.build(b["corpus"], b["doc_lens"], scheme=name, nbits=nbits,
                        centroids=centroids, verbose=centroids is None)
        build_s = time.perf_counter() - t
        centroids = idx.centroids
        scorer = binary_scorer if name == "binary" else None
        n, lat = run(b, lambda q: npl.search(idx, q, args.k, args.n_probe,
                                             args.n_full, scorer)[0], args.k)
        row(scheme, build_s, idx.bytes_per_token(), n, lat, args.k)


def make_rust_scorer():
    """Wrap the Rust extension as a nanoplaid binary_scorer. Coerce dtype and
    contiguity at the boundary: the pyo3 signature is strict (f32/u8/i64,
    C-contiguous) and a mismatch would raise rather than convert."""
    import nanoplaid_kernels

    def scorer(q, payload, lens):
        return nanoplaid_kernels.maxsim_docs(
            np.ascontiguousarray(q, np.float32),
            np.ascontiguousarray(payload, np.uint8),
            np.ascontiguousarray(lens, np.int64))

    return scorer


def search_exhaustive_ids(b, q, k):
    return npl.search_exhaustive(q, b["corpus"], b["doc_offsets"], k)[0]


def row(scheme, build_s, bpt, ndcg, lat, k):
    ms = lat * 1e3
    b = f"{build_s:7.1f}" if build_s is not None else "      -"
    print(f"| {scheme:<10s} | {b} | {bpt:7d} | {ndcg:7.4f} | {ms.mean():7.1f} |"
          f" {np.percentile(ms, 50):6.1f} | {np.percentile(ms, 95):6.1f} |", flush=True)


if __name__ == "__main__":
    main()
