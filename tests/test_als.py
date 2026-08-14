"""ALS fold-in: the standalone HKV solve must match implicit's recalculate_user."""

import numpy as np
import scipy.sparse as sp

from anilist_rec.als import als_scorer, fit_als, foldin_user_factor


def toy_matrix() -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    dense = (rng.random((60, 12)) < 0.3).astype("float32")
    dense[rng.random((60, 12)) < 0.05] = -0.5  # sprinkle negatives (DROPPED)
    return sp.csr_matrix(dense)


def test_foldin_matches_implicit_recalculate_user():
    x = toy_matrix()
    reg, alpha = 0.01, 10.0
    model = fit_als(x, factors=8, regularization=reg, alpha=alpha, seed=1)

    fold = np.zeros(12, dtype="float32")
    fold[[2, 5]] = [1.0, 2.0]
    ours = foldin_user_factor(fold, model.item_factors, reg, alpha)
    theirs = model.recalculate_user(0, sp.csr_matrix(fold))
    np.testing.assert_allclose(ours, np.asarray(theirs).ravel(), rtol=1e-3, atol=1e-5)


def test_negative_foldin_is_high_confidence_zero_preference():
    x = toy_matrix()
    model = fit_als(x, factors=8, regularization=0.01, alpha=10.0, seed=1)
    y = model.item_factors

    positive_only = np.zeros(12, dtype="float32")
    positive_only[2] = 1.0
    with_negative = positive_only.copy()
    with_negative[7] = -1.0  # dropped item

    a = foldin_user_factor(positive_only, y, 0.01, 10.0)
    b = foldin_user_factor(with_negative, y, 0.01, 10.0)
    # zero-preference at high confidence shrinks the dropped item's predicted
    # score toward 0 (Sherman-Morrison: scaled by 1/(1 + c·y'A⁻¹y))
    assert abs(y[7] @ b) < abs(y[7] @ a)


def test_scorer_shape_and_batching():
    x = toy_matrix()
    model = fit_als(x, factors=8, regularization=0.01, alpha=10.0, seed=1)
    scorer = als_scorer(model.item_factors, 0.01, 10.0)
    fold = sp.csr_matrix(
        np.array([[0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0]] * 3, dtype="float32")
    )
    scores = scorer(fold)
    assert scores.shape == (3, 12)
    np.testing.assert_allclose(scores[0], scores[2], rtol=1e-5)
