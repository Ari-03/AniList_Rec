"""Compare harness pieces: score replay, dial inside evaluate, bar and
default-dial rules (issue #19)."""

import numpy as np
import scipy.sparse as sp

from anilist_rec.compare import (
    cis_overlap,
    clears_bar,
    pick_default_dial,
    precompute_scores,
    replay_scorer,
)
from anilist_rec.evaluate import EvalUser, evaluate
from anilist_rec.franchise import FranchiseIndex
from anilist_rec.sweepplot import render_sweep_svg


def summary(ndcg, lo, hi, coverage=0.10, pop_lift=0.0):
    return {
        "ndcg10": ndcg,
        "ndcg10_ci_lo": lo,
        "ndcg10_ci_hi": hi,
        "coverage10": coverage,
        "pop_lift": pop_lift,
    }


def singleton_franchise(n: int) -> FranchiseIndex:
    return FranchiseIndex(
        item_franchise=-np.arange(1, n + 1, dtype=np.int64),
        is_entry=np.ones(n, dtype=bool),
        entry_of_franchise={-(i + 1): i for i in range(n)},
    )


def test_replay_reproduces_scorer_through_evaluate():
    rng = np.random.default_rng(0)
    n_items = 6
    users = [
        EvalUser(
            fold_idx=[int(i % n_items)],
            fold_w=[1.0],
            watched=[int(i % n_items)],
            plan_idx=[],
            targets={int((i + 1) % n_items): 1.0},
            negatives=[],
        )
        for i in range(7)
    ]
    table = rng.normal(size=(n_items, n_items)).astype("float32")

    def score_fn(fold_csr: sp.csr_matrix) -> np.ndarray:
        return np.asarray(fold_csr @ table)

    franchise = singleton_franchise(n_items)
    counts = np.arange(1, n_items + 1, dtype=np.float64)

    direct = evaluate(users, score_fn, franchise, counts, chunk=3)
    scores = precompute_scores(score_fn, users, n_items, chunk=2)  # different chunking
    replayed = evaluate(users, replay_scorer(scores), franchise, counts, chunk=3)
    np.testing.assert_allclose(direct.ndcg10, replayed.ndcg10)

    # and the dial changes the ranking inside evaluate, matching a manual re-rank
    dialed = evaluate(users, replay_scorer(scores), franchise, counts, chunk=3, dial=1.0)
    manual = evaluate(
        users,
        replay_scorer(scores / np.power(counts + 1.0, 1.0).astype("float32")),
        franchise,
        counts,
        chunk=3,
    )
    np.testing.assert_allclose(dialed.ndcg10, manual.ndcg10)


def test_clears_bar_two_sided_with_coverage_guard():
    bars = {
        "item-item BM25": summary(0.13, 0.126, 0.134, coverage=0.063),
        "MostPopular": summary(0.198, 0.195, 0.202, coverage=0.009),
    }
    assert clears_bar(summary(0.22, 0.216, 0.224, coverage=0.02), bars)
    assert not clears_bar(summary(0.15, 0.146, 0.154, coverage=0.08), bars)  # under MostPopular
    assert not clears_bar(summary(0.22, 0.216, 0.224, coverage=0.010), bars)  # degenerate


def test_cis_overlap():
    assert cis_overlap(summary(0.20, 0.19, 0.21), summary(0.205, 0.20, 0.215))
    assert not cis_overlap(summary(0.20, 0.19, 0.21), summary(0.23, 0.22, 0.24))


def test_pick_default_dial_largest_within_noise():
    sweep = {
        0.0: summary(0.300, 0.296, 0.304),  # half-width 0.004
        0.1: summary(0.299, 0.295, 0.303),
        0.2: summary(0.297, 0.293, 0.301),  # 0.003 below: within tolerance
        0.4: summary(0.290, 0.286, 0.294),  # 0.010 below: out
        1.0: summary(0.250, 0.246, 0.254),
    }
    assert pick_default_dial(sweep) == 0.2


def test_sweep_svg_renders_all_series_and_points():
    dials = [0.0, 0.5, 1.0]
    curves = {
        "SASRec": {d: summary(0.37 - d * 0.05, 0.36, 0.38, pop_lift=3 - 4 * d) for d in dials},
        "EASE": {d: summary(0.22 - d * 0.05, 0.21, 0.23, pop_lift=7 - 5 * d) for d in dials},
    }
    svg = render_sweep_svg(curves)
    assert svg.count("<polyline") == 2
    assert svg.count("<circle") == 6 + 2  # 3 points per series + 2 legend swatches
    assert "SASRec" in svg and "EASE" in svg
