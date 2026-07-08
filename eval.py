"""Evaluate nanoplaid on a BEIR bundle: NDCG@10, latency, and storage for each
quantization scheme, against the exhaustive float baseline.

    python eval.py data/toy               # committed multi-domain toy, numpy only
    python encode.py --download scifact --out data/scifact   # a full dataset
    python eval.py data/scifact

Pass a single bundle dir, or a directory of bundles (like data/toy) to evaluate
each and print a per-domain table with an average row.

Bundle layout: corpus.npy [T, dim] concatenated token vectors (f16 or f32),
corpus_lens.npy, corpus_ids.json, queries.npy, query_lens.npy, query_ids.json,
qrels.json {qid: {docid: relevance}}.
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
        # f16-on-disk bundles (the toy) upcast to f32 in memory: the kernels and
        # BLAS want f32, and f16 only ever bought us a smaller file to commit.
        "corpus": np.load(p / "corpus.npy").astype(np.float32),
        "doc_lens": np.load(p / "corpus_lens.npy"),
        "queries": np.load(p / "queries.npy").astype(np.float32),
        "query_lens": np.load(p / "query_lens.npy"),
        "corpus_ids": json.loads((p / "corpus_ids.json").read_text()),
        "query_ids": json.loads((p / "query_ids.json").read_text()),
        "qrels": json.loads((p / "qrels.json").read_text()),
    }
    b["doc_offsets"] = np.concatenate([[0], np.cumsum(b["doc_lens"])[:-1]])
    b["q_offsets"] = np.concatenate([[0], np.cumsum(b["query_lens"])[:-1]])
    return b


def is_bundle(path):
    return (Path(path) / "corpus.npy").exists()


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


def eval_bundle(b, schemes, args, scorer, verbose=False, profile=False):
    """Evaluate every scheme on one bundle. Returns {scheme: {ndcg, lat, bpt,
    build_s, index_mb, stages}}. With profile=True, per-stage search times are
    collected too."""
    dim = b["corpus"].shape[1]
    out = {}
    centroids = None  # trained once per bundle, shared across its two-stage schemes
    for scheme in schemes:
        if scheme == "exact":
            n, lat = run(b, lambda q: search_exhaustive_ids(b, q, args.k), args.k)
            # "index" for exact is just the float corpus you must keep resident.
            out[scheme] = dict(ndcg=n, lat=lat, bpt=4 * dim, build_s=None,
                               index_mb=b["corpus"].nbytes / 1e6, stages={})
            continue
        name, nbits = ("binary", 0) if scheme == "binary" else ("residual", int(scheme[-1]))
        t = time.perf_counter()
        idx = npl.build(b["corpus"], b["doc_lens"], scheme=name, nbits=nbits,
                        centroids=centroids, verbose=verbose)
        build_s = time.perf_counter() - t
        centroids = idx.centroids
        s = scorer if name == "binary" else None
        stages = {} if profile else None
        n, lat = run(b, lambda q: npl.search(idx, q, args.k, args.n_probe,
                                             args.n_full, s, stages)[0], args.k)
        out[scheme] = dict(ndcg=n, lat=lat, bpt=idx.bytes_per_token(),
                           build_s=build_s, index_mb=idx.nbytes() / 1e6,
                           stages=stages or {})
    return out


def eval_single(path, schemes, args, scorer):
    b = load_bundle(path)
    dim = b["corpus"].shape[1]
    print(f"{len(b['doc_lens'])} docs, {len(b['corpus'])} tokens, "
          f"{len(b['query_lens'])} queries, dim={dim}, backend={args.backend}\n")
    results = eval_bundle(b, schemes, args, scorer, verbose=True, profile=args.profile)
    if args.profile:
        print_profile(results, schemes, args.k)
        return
    header = f"| scheme     | build s | B/token | NDCG@{args.k} | mean ms | p50 ms | p95 ms |"
    print(header + "\n|" + "|".join("-" * (len(c) - 1) for c in header.split("|")[1:-1]) + "|")
    for scheme in schemes:
        r = results[scheme]
        row(scheme, r["build_s"], r["bpt"], r["ndcg"], r["lat"], args.k)


def eval_multi(root, schemes, args, scorer):
    """A directory of bundles -> per-domain NDCG@k matrix + average row (or, with
    --profile, an aggregate speed+memory table summed/pooled across domains)."""
    names = sorted(d.name for d in Path(root).iterdir() if is_bundle(d))
    per_domain = {}
    for name in names:
        b = load_bundle(Path(root) / name)
        print(f"evaluating {name} ({len(b['doc_lens'])} docs, {len(b['query_lens'])} queries)...",
              flush=True)
        per_domain[name] = eval_bundle(b, schemes, args, scorer, profile=args.profile)

    if args.profile:
        print_profile(merge_domains(per_domain, schemes), schemes, args.k,
                      note="index MB and build s summed across domains; latency pooled")
        return

    ndcg = {n: {s: per_domain[n][s]["ndcg"] for s in schemes} for n in names}
    bpt = {s: per_domain[names[0]][s]["bpt"] for s in schemes}
    w = max(len(n) for n in names + ["average"])
    head = f"| {'dataset':<{w}} | " + " | ".join(f"{s:>8}" for s in schemes) + " |"
    print("\n" + head)
    print("|" + "|".join("-" * (len(c)) for c in head.split("|")[1:-1]) + "|")
    for name in names:
        cells = " | ".join(f"{ndcg[name][s]:8.4f}" for s in schemes)
        print(f"| {name:<{w}} | {cells} |")
    avg = {s: float(np.mean([ndcg[n][s] for n in names])) for s in schemes}
    print("|" + "|".join("-" * (len(c)) for c in head.split("|")[1:-1]) + "|")
    print(f"| {'average':<{w}} | " + " | ".join(f"{avg[s]:8.4f}" for s in schemes) + " |")
    print(f"\nNDCG@{args.k}; B/token: " + ", ".join(f"{s} {bpt[s]}" for s in schemes))
    if "exact" in schemes and "binary" in schemes and avg["exact"] > 0:
        print(f"binary keeps {100 * avg['binary'] / avg['exact']:.1f}% of exact NDCG on average")


def merge_domains(per_domain, schemes):
    """Pool per-domain profile results into one: sum memory/build, concat
    latencies, sum stage times, mean NDCG."""
    agg = {}
    for s in schemes:
        rs = [per_domain[n][s] for n in per_domain]
        stages = {}
        for r in rs:
            for key, v in r["stages"].items():
                stages[key] = stages.get(key, 0.0) + v
        builds = [r["build_s"] for r in rs if r["build_s"] is not None]
        agg[s] = dict(
            ndcg=float(np.mean([r["ndcg"] for r in rs])),
            lat=np.concatenate([r["lat"] for r in rs]),
            bpt=rs[0]["bpt"],
            build_s=sum(builds) if builds else None,
            index_mb=sum(r["index_mb"] for r in rs),
            stages=stages,
        )
    return agg


def print_profile(results, schemes, k, note=None):
    """Speed + memory table: index footprint, build time, query latency, and
    where the two-stage search spends its time."""
    head = (f"| {'scheme':<10} | NDCG@{k} | idx MB | B/tok | build s | mean ms |"
            f" p50 ms | p95 ms | probe/rank/rescore % |")
    print("\n" + head)
    print("|" + "|".join("-" * len(c) for c in head.split("|")[1:-1]) + "|")
    for s in schemes:
        r = results[s]
        ms = r["lat"] * 1e3
        build = f"{r['build_s']:6.1f}" if r["build_s"] is not None else "     -"
        st = r["stages"]
        tot = sum(st.values())
        frac = ("/".join(f"{100 * st.get(x, 0) / tot:.0f}" for x in ("probe", "rank", "rescore"))
                if tot > 0 else "-")
        print(f"| {s:<10} | {r['ndcg']:6.4f} | {r['index_mb']:6.1f} | {r['bpt']:5d} |"
              f" {build} | {ms.mean():7.1f} | {np.percentile(ms, 50):6.1f} |"
              f" {np.percentile(ms, 95):6.1f} | {frac:>20} |")
    if note:
        print(f"\n{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="a bundle dir, or a directory of bundles (e.g. data/toy)")
    ap.add_argument("--schemes", default="exact,residual4,residual2,binary")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--n-full", type=int, default=1024)
    ap.add_argument("--backend", choices=["numpy", "rust"], default="numpy",
                    help="binary stage-2 scorer; 'rust' needs the kernels extension "
                         "(maturin develop -m kernels/Cargo.toml --release --features python)")
    ap.add_argument("--profile", action="store_true",
                    help="report index memory, build time, and per-stage query latency")
    args = ap.parse_args()

    scorer = make_rust_scorer() if args.backend == "rust" else None
    schemes = args.schemes.split(",")
    if is_bundle(args.bundle):
        eval_single(args.bundle, schemes, args, scorer)
    else:
        eval_multi(args.bundle, schemes, args, scorer)


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
