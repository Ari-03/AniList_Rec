"""EASE closed form, sparsification, and scorer."""

import numpy as np
import scipy.sparse as sp

from anilist_rec.ease import ease_scorer, fit_ease, fit_gram, sparsify_topk


def toy_matrix() -> sp.csr_matrix:
    # users 0-2 co-watch items 0+1; users 3-4 co-watch items 2+3; one dropped row
    rows = np.array([[1, 1, 0, 0]] * 3 + [[0, 0, 1, 1]] * 2 + [[-1, 0, 0, 1]], dtype="float32")
    return sp.csr_matrix(rows)


def test_fit_ease_scores_cowatched_items():
    b = fit_ease(fit_gram(toy_matrix()), l2=1.0)
    assert np.allclose(np.diag(b), 0.0)
    # item 0's strongest neighbour is its co-watched partner
    assert b[0].argmax() == 1
    assert b[2].argmax() == 3


def test_negative_signal_pushes_apart():
    # user 5 dropped item 0 while finishing item 3 -> B[0,3] below B[2,3]
    b = fit_ease(fit_gram(toy_matrix()), l2=1.0)
    assert b[0, 3] < b[2, 3]


def test_gram_norm_changes_scale_not_diag():
    b = fit_ease(fit_gram(toy_matrix()), l2=1.0, gram_norm=True)
    assert np.allclose(np.diag(b), 0.0)
    assert b[0].argmax() == 1


def test_sparsify_keeps_topk_per_column():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(20, 20)).astype("float32")
    np.fill_diagonal(b, 0.0)
    b_k = sparsify_topk(b, 5)
    kept = np.asarray((b_k != 0).sum(axis=0)).ravel()
    assert (kept <= 5).all()
    # the largest |value| per column survives
    col = np.abs(b[:, 7]).argmax()
    assert b_k[col, 7] == b[col, 7]


def test_scorer_matches_dense_and_sparse():
    b = fit_ease(fit_gram(toy_matrix()), l2=1.0)
    fold = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype="float32"))
    dense_scores = ease_scorer(b)(fold)
    sparse_scores = ease_scorer(sparsify_topk(b, 4))(fold)
    assert dense_scores.shape == (1, 4)
    np.testing.assert_allclose(dense_scores, sparse_scores, atol=1e-6)
