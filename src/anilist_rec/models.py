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
    """Popularity dial (SPEC §1): demote popular items by (count+1)^alpha; 0 = off.

    Architectures are compared dial-off (SPEC §5); the validation sweep after a
    winner is chosen ranks through this same re-rank.
    """
    return scores / np.power(item_counts + 1.0, alpha)


def most_popular_scorer(item_counts: np.ndarray) -> ScoreFn:
    """Same ranking for everyone: training popularity."""
    counts = item_counts.astype("float32")

    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        return np.tile(counts, (fold_csr.shape[0], 1))

    return score
