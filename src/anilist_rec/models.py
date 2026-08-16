"""The two baselines (SPEC §4): item-item BM25 and MostPopular.

Both expose the same scoring interface the eval harness batches over:
a callable taking a users-by-items fold-in CSR and returning dense scores.
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from anilist_rec.config import Config

ScoreFn = Callable[[sp.csr_matrix], np.ndarray]


def fit_bm25(cfg: Config, x_train: sp.csr_matrix) -> sp.csr_matrix:
    """Item-item BM25 similarity via `implicit` (cached on disk, keyed by config)."""
    if cfg.similarity_path.exists():
        return sp.load_npz(cfg.similarity_path)

    from implicit.nearest_neighbours import BM25Recommender

    model = BM25Recommender(K=cfg.k_neighbors, K1=1.2, B=0.75)
    model.fit(x_train, show_progress=False)
    assert model.similarity is not None  # set by fit; implicit types it Optional
    similarity = model.similarity.tocsr().astype("float32")
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cfg.similarity_path, similarity)
    return similarity


def bm25_scorer(similarity: sp.csr_matrix) -> ScoreFn:
    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        return np.asarray((fold_csr @ similarity).todense())

    return score


def apply_dial(scores: np.ndarray, item_counts: np.ndarray, alpha: float) -> np.ndarray:
    """Popularity dial (SPEC §1), in rank space: per-user score percentile minus
    alpha x popularity percentile; 0 = off (raw scores pass through untouched).

    Rank space makes the 0-1 knob genuinely architecture-independent: the first
    formulation divided raw scores by (count+1)^alpha, and the #19 sweep showed
    its useful range collapsing with the model's score scale (a cliff below
    dial 0.1 on SASRec logits while EASE barely moved before 0.6). Percentiles
    are scale-free, so alpha is the exchange rate between one step of model
    preference and one step of popularity, whatever the architecture.

    Architectures are compared dial-off (SPEC §5); the validation sweep after a
    winner is chosen ranks through this same re-rank.
    """
    if alpha == 0.0:
        return scores
    n = scores.shape[-1]
    pop_pct = item_counts.argsort().argsort() / (n - 1)
    score_pct = scores.argsort(axis=-1).argsort(axis=-1) / (n - 1)
    # the epsilon breaks exact preference/popularity ties toward the niche
    # item — at any alpha > 0 the caller asked for novelty
    return (score_pct - alpha * (1.0 + 1e-6) * pop_pct).astype("float32")


def most_popular_scorer(item_counts: np.ndarray) -> ScoreFn:
    """Same ranking for everyone: training popularity."""
    counts = item_counts.astype("float32")

    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        return np.tile(counts, (fold_csr.shape[0], 1))

    return score
