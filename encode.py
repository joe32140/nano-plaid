"""Encode a BEIR dataset with a ColBERT model into a numpy bundle for eval.py.

    pip install pylate            # (needs torch; GPU/MPS recommended)
    python encode.py --download scifact --out data/scifact
    python encode.py --data /path/to/beir/scifact --out data/scifact

Outputs (--out):
  corpus.npy      [total_doc_tokens, dim] f32   concatenated token vectors
  corpus_lens.npy [n_docs]  i64                 tokens per document
  corpus_ids.json [n_docs]                      BEIR ids, row order
  queries.npy / query_lens.npy / query_ids.json (same, for test-split queries)
  qrels.json      {query_id: {doc_id: relevance}}

Ragged per-item arrays are recovered by walking the *_lens files. This is the
only file that needs torch; eval.py and nanoplaid.py are numpy-only.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def download_beir(name, dest):
    dest = Path(dest)
    if (dest / name / "corpus.jsonl").exists():
        return dest / name
    print(f"downloading BEIR/{name} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BEIR_URL.format(name)) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    return dest / name


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def pack(embeddings):
    """List of [n_i, dim] -> (concat [sum n_i, dim] f32, lens [N] i64)."""
    lens = np.array([e.shape[0] for e in embeddings], dtype=np.int64)
    return np.concatenate(embeddings, axis=0).astype(np.float32), lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="BEIR dataset dir (corpus.jsonl, queries.jsonl, qrels/)")
    ap.add_argument("--download", help="BEIR dataset name to fetch (e.g. scifact)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="answerdotai/answerai-colbert-small-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from pylate import models  # deferred: everything above is torch-free

    data = Path(args.data) if args.data else download_beir(args.download, "data/beir")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(data / "qrels" / f"{args.split}.tsv")

    corpus_ids, corpus_texts = [], []
    for row in read_jsonl(data / "corpus.jsonl"):
        corpus_ids.append(str(row["_id"]))
        corpus_texts.append((row.get("title", "") + " " + row.get("text", "")).strip())

    query_ids, query_texts = [], []
    for row in read_jsonl(data / "queries.jsonl"):
        if str(row["_id"]) in qrels:  # only queries in the eval split
            query_ids.append(str(row["_id"]))
            query_texts.append(row["text"])

    print(f"{len(corpus_ids)} docs, {len(query_ids)} queries; encoding with {args.model}")
    model = models.ColBERT(model_name_or_path=args.model)

    docs = model.encode(corpus_texts, batch_size=args.batch_size,
                        is_query=False, show_progress_bar=True)
    corpus, corpus_lens = pack(docs)
    queries, query_lens = pack(model.encode(query_texts, batch_size=args.batch_size,
                                            is_query=True, show_progress_bar=True))

    np.save(out / "corpus.npy", corpus)
    np.save(out / "corpus_lens.npy", corpus_lens)
    np.save(out / "queries.npy", queries)
    np.save(out / "query_lens.npy", query_lens)
    (out / "corpus_ids.json").write_text(json.dumps(corpus_ids))
    (out / "query_ids.json").write_text(json.dumps(query_ids))
    (out / "qrels.json").write_text(json.dumps(qrels))
    print(f"wrote bundle to {out}  (corpus {corpus.shape}, queries {queries.shape})")


if __name__ == "__main__":
    main()
