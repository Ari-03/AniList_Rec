"""Training matrix construction + baseline scorers."""

import numpy as np
import polars as pl
import pytest

from anilist_rec.config import Config
from anilist_rec.matrix import build_training_matrix, item_index
from anilist_rec.models import bm25_scorer, fit_bm25, most_popular_scorer
from anilist_rec.signals import PLAN, STD


def signals_frame(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(
        [dict(zip(["user_id", "anime_id", "kind", "weight", "ts"], r, strict=True)) for r in rows],
        schema={
            "user_id": pl.String,
            "anime_id": pl.Int32,
            "kind": pl.UInt8,
            "weight": pl.Float32,
            "ts": pl.Datetime("ms"),
        },
    )


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=tmp_path)


def test_matrix_excludes_holdout_plan_and_weightless(cfg):
    signals = signals_frame(
        [
            ("a", 10, STD, 1.0, None),
            ("a", 20, STD, 2.0, None),
            ("b", 20, STD, 1.0, None),
            ("b", 30, PLAN, 0.0, None),  # weightless: PLAN never trains
            ("held", 10, STD, 1.0, None),  # holdout user excluded entirely
        ]
    )
    item_ids = item_index(signals)
    assert item_ids.tolist() == [10, 20, 30]

    x_train, item_counts = build_training_matrix(cfg, signals, item_ids, {"held"})
    assert x_train.shape == (2, 3)
    assert x_train.sum() == 4.0
    assert item_counts.tolist() == [1.0, 2.0, 0.0]


def test_matrix_cap_is_deterministic(tmp_path):
    cfg = Config(data_dir=tmp_path, train_user_cap=3)
    signals = signals_frame([(f"u{i}", 10 + (i % 4), STD, 1.0, None) for i in range(20)])
    item_ids = item_index(signals)
    first, _ = build_training_matrix(cfg, signals, item_ids, set())
    second, _ = build_training_matrix(cfg, signals, item_ids, set())
    assert first.shape[0] == 3
    assert (first != second).nnz == 0


def test_dial_demotes_popular_items():
    from anilist_rec.models import apply_dial

    scores = np.array([1.0, 1.0])
    counts = np.array([999.0, 0.0])
    assert (apply_dial(scores, counts, 0.0) == scores).all()  # dial off is a no-op
    dialed = apply_dial(scores, counts, 0.5)
    assert dialed[1] > dialed[0]


def test_most_popular_scores_every_user_identically():
    import scipy.sparse as sp

    counts = np.array([5.0, 1.0, 3.0])
    scores = most_popular_scorer(counts)(sp.csr_matrix(np.eye(2, 3)))
    assert scores.shape == (2, 3)
    assert (scores[0] == scores[1]).all()
    assert scores[0].argmax() == 0


def test_bm25_fit_scores_cowatched_items(cfg):
    import scipy.sparse as sp

    # users 0-1 co-watch items 0+1, users 2-5 co-watch items 2+3; no cross-watching
    # (implicit's BM25 IDF is per user — users must hold < the full catalogue)
    x = sp.csr_matrix(
        np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0]] + [[0, 0, 1, 1]] * 4,
            dtype="float32",
        )
    )
    similarity = fit_bm25(cfg, x)
    assert similarity.shape == (4, 4)
    assert cfg.similarity_path.exists()

    scores = bm25_scorer(similarity)(sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype="float32")))
    assert scores[0, 1] > 0
    assert scores[0, 2] == scores[0, 3] == 0

    # cache round-trip returns the same matrix
    assert (fit_bm25(cfg, x) != similarity).nnz == 0
