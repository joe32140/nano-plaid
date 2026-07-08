"""Build the committed multi-domain toy bundle from NanoBEIR + LateOn-regularized.

For each of a few Nano* datasets: encode with a ColBERT model, keep every
query's gold documents plus mined hard-negative distractors up to a per-dataset
target, truncate long docs, downcast to fp16, and write one bundle per domain
under data/toy/. That ~30 MB tree is the artifact checked into the repo so
`python eval.py data/toy` runs with numpy only — no torch, no downloads.

    pip install pylate datasets
    python make_toy.py                # writes data/toy/{scifact,...}

This is the only step that needs torch; eval.py and nanoplaid.py stay numpy-only.
Hard negatives are mined with nanoplaid's own exhaustive MaxSim, so the toy is
self-consistent with what it evaluates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import nanoplaid as npl

# (HF repo, target corpus size). Gold docs are always kept first; the rest of
# the budget is filled with mined hard negatives, then random docs if needed.
DATASETS = {
    "scifact": ("zeta-alpha-ai/NanoSciFact", 400),
    "nfcorpus": ("zeta-alpha-ai/NanoNFCorpus", 400),
    "fiqa": ("zeta-alpha-ai/NanoFiQA2018", 400),
    "quora": ("zeta-alpha-ai/NanoQuoraRetrieval", 1200),
}
MAX_DOC_TOKENS = 96          # truncate long docs so the bundle stays git-sized
HARD_NEG_PER_QUERY = 100     # depth of the mining pool per query
SEED = 0


def load_nano(repo):
    from datasets import load_dataset
    corpus = load_dataset(repo, "corpus", split="train")
    queries = load_dataset(repo, "queries", split="train")
    qrels_rows = load_dataset(repo, "qrels", split="train")
    corpus_ids = [str(x) for x in corpus["_id"]]
    query_ids = [str(x) for x in queries["_id"]]
    qrels = {}
    for row in qrels_rows:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = 1
    return corpus_ids, corpus["text"], query_ids, queries["text"], qrels


def truncate(emb, n):
    """Keep the first min(len, n) token rows of each [n_tok, dim] doc."""
    return [e[: min(len(e), n)].astype(np.float32) for e in emb]


def build_dataset(name, repo, target, model, out_root):
    print(f"\n=== {name} ({repo}), target {target} docs ===")
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_nano(repo)

    docs = truncate(model.encode(list(corpus_texts), batch_size=32,
                                 is_query=False, show_progress_bar=True), MAX_DOC_TOKENS)
    queries = truncate(model.encode(list(query_texts), batch_size=32,
                                    is_query=True, show_progress_bar=True), MAX_DOC_TOKENS)

    id_to_row = {cid: i for i, cid in enumerate(corpus_ids)}
    corpus_cat = np.concatenate(docs).astype(np.float32)
    doc_lens = np.array([len(d) for d in docs], np.int64)
    doc_offsets = np.concatenate([[0], np.cumsum(doc_lens)[:-1]])

    # Query-balanced gold selection: round-robin one gold per query per pass, so
    # a dense dataset (NFCorpus, ~50 relevant/query) keeps every query
    # represented instead of spending the whole budget on the first few queries.
    gold_lists = [[id_to_row[d] for d in rels if d in id_to_row] for rels in qrels.values()]
    keep, gold_rows = [], set()
    progress = True
    while len(keep) < target and progress:
        progress = False
        for lst in gold_lists:
            if len(keep) >= target:
                break
            while lst:  # next gold for this query not already kept
                r = lst.pop()
                if r not in gold_rows:
                    keep.append(r)
                    gold_rows.add(r)
                    progress = True
                    break

    # Fill the rest with mined hard negatives: the top-scoring non-gold docs
    # across queries, by exhaustive MaxSim on the truncated docs.
    neg_rank = {}
    for q in queries:
        top, _ = npl.search_exhaustive(q, corpus_cat, doc_offsets, k=HARD_NEG_PER_QUERY)
        for rank, row in enumerate(int(r) for r in top):
            if row not in gold_rows:
                neg_rank[row] = min(neg_rank.get(row, rank), rank)

    for row, _ in sorted(neg_rank.items(), key=lambda kv: kv[1]):
        if len(keep) >= target:
            break
        if row not in gold_rows:
            keep.append(row)
    if len(keep) < target:  # pad with random docs if the corpus is thin
        rng = np.random.default_rng(SEED)
        pool = [r for r in range(len(docs)) if r not in set(keep)]
        keep += list(rng.choice(pool, min(target - len(keep), len(pool)), replace=False))
    keep = sorted(set(keep))

    kept_ids = [corpus_ids[r] for r in keep]
    kept_id_set = set(kept_ids)
    kept_docs = [docs[r] for r in keep]
    kept_qrels = {q: {d: 1 for d in rels if d in kept_id_set} for q, rels in qrels.items()}
    kept_qrels = {q: r for q, r in kept_qrels.items() if r}  # drop queries with no gold left

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    _save_bundle(out, kept_docs, kept_ids, queries, query_ids, kept_qrels)
    gold_kept = sum(1 for r in keep if r in gold_rows)
    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"  kept {len(keep)} docs ({gold_kept} gold), {len(kept_qrels)} queries with gold, "
          f"{size_mb:.1f} MB")


def _save_bundle(out, docs, doc_ids, queries, query_ids, qrels):
    corpus = np.concatenate(docs).astype(np.float16)
    doc_lens = np.array([len(d) for d in docs], np.int64)
    q_cat = np.concatenate(queries).astype(np.float16)
    q_lens = np.array([len(q) for q in queries], np.int64)
    np.save(out / "corpus.npy", corpus)
    np.save(out / "corpus_lens.npy", doc_lens)
    np.save(out / "queries.npy", q_cat)
    np.save(out / "query_lens.npy", q_lens)
    (out / "corpus_ids.json").write_text(json.dumps(doc_ids))
    (out / "query_ids.json").write_text(json.dumps(query_ids))
    (out / "qrels.json").write_text(json.dumps(qrels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--model", default="lightonai/LateOn-regularized")
    ap.add_argument("--only", help="comma-separated subset of dataset names")
    args = ap.parse_args()

    from pylate import models
    model = models.ColBERT(model_name_or_path=args.model)
    names = args.only.split(",") if args.only else list(DATASETS)
    out_root = Path(args.out)
    for name in names:
        repo, target = DATASETS[name]
        build_dataset(name, repo, target, model, out_root)
    print(f"\nwrote {len(names)} bundles to {out_root}")


if __name__ == "__main__":
    main()
