"""nanoplaid: late-interaction retrieval (ColBERT / PLAID) in plain numpy.

The simplest complete implementation of a PLAID-style two-stage index for
multi-vector ("late interaction") retrieval. One file, one dependency (numpy),
every stage measurable on a laptop. In the spirit of nanoGPT: the code is the
documentation, and the point is that you can change any piece of it.

The arc, top to bottom:

  0. MaxSim + exhaustive search   -- the exact baseline everything is scored
                                     against. ~15 lines.
  1. k-means centroids            -- the codebook. Every corpus token gets a
                                     nearest-centroid id ("code").
  2. residual compression         -- store each token as (centroid id +
                                     quantized residual). The storage knob:
                                     nbits in {2, 4}.
  3. binary compression           -- store each token as its sign bits only
                                     (32x smaller than float32), score with an
                                     int8 query via the 2P - T identity.
  4. the index                    -- centroids + codes + an inverted file
                                     (centroid -> doc ids) + compressed tokens.
  5. two-stage search             -- probe centroids to get candidates, rank
                                     them with cheap centroid scores, then
                                     exactly rescore only the top n_full.

Every knob trades quality against speed against storage; eval.py measures all
three on a real BEIR dataset so you can see what each stage buys you.

Assumes embeddings are L2-normalized per token (ColBERT models do this), so
dot product == cosine similarity.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# 0. MaxSim and exhaustive search
#
# Late interaction: a query and a document are each a *bag of token vectors*.
# The relevance score is MaxSim -- for each query token, take its best-matching
# document token, and sum:
#
#     score(Q, D) = sum_i max_j  Q[i] . D[j]
#
# Everything below is just a way to compute an approximation of this over the
# whole corpus without touching every document token in float32.


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    """Exact MaxSim between one query [nq, dim] and one doc [nd, dim]."""
    return float((q @ d.T).max(axis=1).sum())


def search_exhaustive(q, corpus, doc_offsets, k=10):
    """Score EVERY document exactly. Slow, exact, and the reference for
    everything else in this file.

    corpus is all documents' token vectors concatenated: [total_tokens, dim].
    doc_offsets[i] is the row where document i starts. np.maximum.reduceat
    computes a per-document max over the big similarity matrix in one call.
    """
    sim = q @ corpus.T                                     # [nq, total_tokens]
    per_doc = np.maximum.reduceat(sim, doc_offsets, axis=1)  # [nq, n_docs]
    scores = per_doc.sum(axis=0)
    return _topk(scores, k)


def _topk(scores, k):
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


# ----------------------------------------------------------------------------
# 1. k-means
#
# The codebook. PLAID quantizes every corpus token to its nearest centroid;
# the centroid id is both the coarse value of the token (for cheap scoring)
# and its bucket in the inverted file (for candidate generation).


def kmeans(x, k, n_iter=10, sample=262_144, seed=0, chunk=65_536, verbose=False):
    """Spherical k-means (cosine similarity) on a sample of the corpus tokens."""
    rng = np.random.default_rng(seed)
    if len(x) > sample:
        x = x[rng.choice(len(x), sample, replace=False)]
    x = np.ascontiguousarray(x, dtype=np.float32)
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    for it in range(n_iter):
        a = assign(x, centroids, chunk)
        # Mean of assigned points, one bincount per dimension (fast scatter-add).
        counts = np.bincount(a, minlength=k)
        sums = np.stack(
            [np.bincount(a, weights=x[:, d], minlength=k) for d in range(x.shape[1])],
            axis=1,
        )
        alive = counts > 0
        centroids[alive] = sums[alive] / counts[alive, None]
        # Re-seed dead centroids from random points so k stays honest.
        if (~alive).any():
            centroids[~alive] = x[rng.choice(len(x), int((~alive).sum()))]
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        if verbose:
            print(f"  kmeans iter {it + 1}/{n_iter}: {int(alive.sum())}/{k} centroids alive")
    return centroids.astype(np.float32)


def assign(x, centroids, chunk=65_536):
    """Nearest centroid id for each row of x (argmax dot == argmin angle)."""
    out = np.empty(len(x), dtype=np.int32)
    for i in range(0, len(x), chunk):
        out[i : i + chunk] = np.argmax(x[i : i + chunk] @ centroids.T, axis=1)
    return out


# ----------------------------------------------------------------------------
# 2. Residual compression (the ColBERTv2 / PLAID codec)
#
# A token vector is stored as: centroid[code] + quantized(residual).
# The residual's per-value distribution is carved into 2^nbits buckets by
# quantiles; each value stores only its bucket index (nbits bits), and decodes
# to the bucket's midpoint. nbits=4 is near-lossless; nbits=2 is 2x smaller
# and visibly lossier. bytes/token = dim * nbits / 8 (+4 for the code).


@dataclass
class ResidualCodec:
    nbits: int
    cutoffs: np.ndarray   # [2^nbits - 1] bucket boundaries
    weights: np.ndarray   # [2^nbits]     decoded value per bucket


def train_codec(residual_sample, nbits):
    vals = residual_sample.ravel()
    n_buckets = 2**nbits
    cutoffs = np.quantile(vals, np.arange(1, n_buckets) / n_buckets)
    # Decode each bucket to its MEDIAN, not its edge midpoint: the outer
    # buckets stretch to the distribution's tails, and a midpoint there is
    # dragged far from where the mass actually sits.
    weights = np.quantile(vals, (np.arange(n_buckets) + 0.5) / n_buckets)
    return ResidualCodec(nbits, cutoffs.astype(np.float32), weights.astype(np.float32))


def encode_residuals(codec, residuals):
    """[n, dim] float -> [n, dim*nbits/8] packed uint8."""
    codes = np.searchsorted(codec.cutoffs, residuals).astype(np.uint8)
    bits = (codes[..., None] >> np.arange(codec.nbits - 1, -1, -1)) & 1
    return np.packbits(bits.reshape(len(residuals), -1), axis=1)


def decode_residuals(codec, packed, dim):
    """Inverse of encode_residuals (up to quantization error, of course)."""
    bits = np.unpackbits(packed, axis=1, count=dim * codec.nbits)
    codes = bits.reshape(len(packed), dim, codec.nbits) @ (
        1 << np.arange(codec.nbits - 1, -1, -1)
    )
    return codec.weights[codes]


# ----------------------------------------------------------------------------
# 3. Binary compression
#
# Keep only the SIGN of each dimension: 1 bit/value, 32x smaller than float32.
# Whether this survives depends on the checkpoint -- see the README; it is the
# single most interesting research knob in this repo.
#
# Scoring trick: quantize the query to int8 (q ~= scale * v, v integer). With
# doc values s in {-1,+1}, split the dot product over set bits (s=+1) and
# unset bits (s=-1):
#
#     v . s = P - (T - P) = 2P - T
#
# where P = sum of v over set bits and T = sum of all of v (precomputed once
# per query token). So scoring needs ONLY a masked sum over the doc's bits --
# no decompression to float. This identity is what the SIMD kernels in
# kernels/ (and next-plaid) implement with VNNI / SAD / SDOT instructions;
# here numpy expands the bits and uses BLAS, which computes the same integers
# exactly (they stay far below 2^24, f32's integer limit).


@dataclass
class QueryI8:
    values: np.ndarray   # [nq, dim] int8
    scales: np.ndarray   # [nq] float32: values * scale ~= original query
    sums: np.ndarray     # [nq] int32:   T in the identity


def binarize(x):
    """[n, dim] float -> [n, dim/8] packed sign bits."""
    return np.packbits(x > 0, axis=1)


def quantize_query_i8(q):
    scales = np.maximum(np.abs(q).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    v = np.clip(np.rint(q / scales[:, None]), -127, 127).astype(np.int8)
    return QueryI8(v, scales, v.sum(axis=1, dtype=np.int32))


def score_binary(q8, packed_bits, dim):
    """2P - T for every (query token, doc token) pair -> [nq, n] float32."""
    bits = np.unpackbits(packed_bits, axis=1, count=dim)      # {0,1} per dim
    p = q8.values.astype(np.float32) @ bits.T.astype(np.float32)
    return (2.0 * p - q8.sums[:, None]) * q8.scales[:, None]


# ----------------------------------------------------------------------------
# 4. The index
#
# centroids    [K, dim]        the codebook
# codes        [T] int32       nearest centroid per corpus token
# payload      [T, bytes] u8   packed residuals (residual scheme) or packed
#                              sign bits (binary scheme)
# ivf_*                        inverted file in CSR form: for centroid c,
#                              ivf_docs[ivf_offsets[c]:ivf_offsets[c+1]] are
#                              the (unique) doc ids with a token in c


@dataclass
class Index:
    dim: int
    scheme: str                    # "residual" | "binary"
    centroids: np.ndarray
    codes: np.ndarray
    doc_lens: np.ndarray
    doc_offsets: np.ndarray        # [n_docs] start row of each doc
    ivf_offsets: np.ndarray
    ivf_docs: np.ndarray
    payload: np.ndarray
    codec: Optional[ResidualCodec] = None

    @property
    def n_docs(self):
        return len(self.doc_lens)

    def bytes_per_token(self):
        return self.payload.shape[1] + 4   # payload + int32 centroid code


def build(corpus, doc_lens, scheme="residual", nbits=4,
          n_centroids=None, centroids=None, seed=0, verbose=True):
    """Build an index from concatenated token embeddings.

    Pass precomputed `centroids` to reuse one k-means run across schemes --
    stage 1 is identical for all of them; only the payload differs.
    """
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    total, dim = corpus.shape
    if centroids is None:
        # Heuristic: ~4*sqrt(T) centroids, rounded to a power of two.
        k = n_centroids or 2 ** int(round(np.log2(4 * np.sqrt(total))))
        if verbose:
            print(f"kmeans: {k} centroids over {total} tokens")
        centroids = kmeans(corpus, k, seed=seed, verbose=verbose)
    codes = assign(corpus, centroids)

    doc_lens = np.asarray(doc_lens, dtype=np.int64)
    doc_offsets = np.concatenate([[0], np.cumsum(doc_lens)[:-1]])

    # Inverted file: unique (centroid, doc) pairs, CSR over centroid.
    doc_of_token = np.repeat(np.arange(len(doc_lens)), doc_lens)
    pairs = np.unique(np.stack([codes.astype(np.int64), doc_of_token], axis=1), axis=0)
    ivf_docs = pairs[:, 1].astype(np.int32)
    ivf_offsets = np.searchsorted(pairs[:, 0], np.arange(len(centroids) + 1)).astype(np.int64)

    if scheme == "residual":
        rng = np.random.default_rng(seed)
        sample = corpus[rng.choice(total, min(total, 65_536), replace=False)]
        codec = train_codec(sample - centroids[assign(sample, centroids)], nbits)
        payload = np.empty((total, dim * nbits // 8), dtype=np.uint8)
        for i in range(0, total, 262_144):
            sl = slice(i, min(i + 262_144, total))
            payload[sl] = encode_residuals(codec, corpus[sl] - centroids[codes[sl]])
    elif scheme == "binary":
        codec = None
        payload = binarize(corpus)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    return Index(dim, scheme, centroids, codes, doc_lens, doc_offsets,
                 ivf_offsets, ivf_docs, payload, codec)


def decode_rows(index, rows):
    """Reconstruct token vectors for the given corpus rows (residual scheme)."""
    return (decode_residuals(index.codec, index.payload[rows], index.dim)
            + index.centroids[index.codes[rows]])


# ----------------------------------------------------------------------------
# 5. Two-stage search
#
# Stage 1 (candidates): score query tokens against all K centroids (one small
#   matmul), take each token's top n_probe centroids, and pull every doc that
#   has a token in any of them from the inverted file.
# Stage 1.5 (approx rank): score each candidate using ONLY its tokens'
#   centroid ids -- sim(query token, centroid[code]) is already in the [nq, K]
#   matrix, so this is a gather + segmented max. Keep the top n_full.
# Stage 2 (exact rescore): decompress just those n_full docs and compute true
#   MaxSim (float path for residual, 2P - T path for binary).
#
# n_probe buys recall in stage 1; n_full buys rank quality in stage 2. Both
# cost latency. This split is the whole reason PLAID is fast: the expensive
# exact scoring touches n_full docs instead of the corpus.


def _gather_rows(index, cand):
    """Corpus row indices of all of `cand`'s tokens, plus per-doc start marks.
    (`np.repeat` of each doc's start, plus a running offset within each doc.)
    """
    lens = index.doc_lens[cand]
    bounds = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rows = np.repeat(index.doc_offsets[cand] - bounds, lens) + np.arange(lens.sum())
    return rows, bounds


def search(index, q, k=10, n_probe=8, n_full=1024, binary_scorer=None):
    """Two-stage search. `binary_scorer`, if given, replaces the numpy binary
    stage-2: it takes (query [nq, dim], candidates' packed rows, per-doc token
    counts) and returns one score per candidate doc -- the hook eval.py uses to
    drop in the Rust kernel. Ignored for the residual scheme.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    cs = q @ index.centroids.T                                   # [nq, K]

    # Stage 1: candidate docs from the inverted file.
    probes = np.argpartition(-cs, n_probe - 1, axis=1)[:, :n_probe]
    cand = np.unique(np.concatenate(
        [index.ivf_docs[index.ivf_offsets[c] : index.ivf_offsets[c + 1]]
         for c in probes.ravel()] or [np.empty(0, np.int32)]))
    if len(cand) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)

    # Stage 1.5: approximate scores from centroid ids alone.
    rows, bounds = _gather_rows(index, cand)
    approx = np.maximum.reduceat(cs[:, index.codes[rows]], bounds, axis=1).sum(axis=0)
    if len(cand) > n_full:
        cand = cand[np.sort(np.argpartition(-approx, n_full - 1)[:n_full])]
        rows, bounds = _gather_rows(index, cand)

    # Stage 2: exact rescore of the survivors. The Rust kernel returns per-doc
    # scores directly; both numpy paths produce a [nq, n_tok] similarity that
    # shares the same max-reduce-and-sum tail.
    if index.scheme == "binary" and binary_scorer is not None:
        scores = binary_scorer(q, index.payload[rows], index.doc_lens[cand])
    else:
        if index.scheme == "residual":
            sim = q @ decode_rows(index, rows).T
        else:
            sim = score_binary(quantize_query_i8(q), index.payload[rows], index.dim)
        scores = np.maximum.reduceat(sim, bounds, axis=1).sum(axis=0)

    top, top_scores = _topk(scores, k)
    return cand[top], top_scores


# ----------------------------------------------------------------------------
# Self-test on synthetic data: exact top-1 should survive both codecs, and the
# 2P - T identity must match a literal per-bit loop. Run: python nanoplaid.py


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def normed(n, d):
        x = rng.standard_normal((n, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    dim, n_docs = 128, 500
    doc_lens = rng.integers(20, 60, n_docs)
    corpus = normed(int(doc_lens.sum()), dim)
    doc_offsets = np.concatenate([[0], np.cumsum(doc_lens)[:-1]])
    # Query = a noisy copy of doc 123's first tokens, so it has a clear answer.
    q = corpus[doc_offsets[123] : doc_offsets[123] + 16] + 0.1 * normed(16, dim)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    exact_top, _ = search_exhaustive(q, corpus, doc_offsets, k=5)
    assert exact_top[0] == 123
    for scheme, nbits in [("residual", 4), ("residual", 2), ("binary", 0)]:
        idx = build(corpus, doc_lens, scheme=scheme, nbits=nbits,
                    n_centroids=256, verbose=False)
        top, _ = search(idx, q, k=5, n_probe=4, n_full=100)
        label = scheme + (f"-{nbits}b" if scheme == "residual" else "")
        print(f"{label:12s} top-5 {top.tolist()}  (exact: {exact_top.tolist()})")
        assert top[0] == 123, f"{label} lost the planted document"

    # The identity, verified against the slowest possible implementation.
    q8 = quantize_query_i8(q)
    packed = binarize(corpus[:50])
    fast = score_binary(q8, packed, dim)
    bits = np.unpackbits(packed, axis=1, count=dim)
    for i in range(len(q8.values)):
        for j in range(50):
            p = int(q8.values[i][bits[j] == 1].sum())
            slow = (2 * p - int(q8.sums[i])) * q8.scales[i]
            assert abs(slow - fast[i, j]) < 1e-3
    print("2P - T identity matches the per-bit loop. All good.")
