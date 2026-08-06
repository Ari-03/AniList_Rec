"""Eval harness: serving-path filters, graded NDCG, and the guardrail metrics."""

import numpy as np
import pytest

from anilist_rec.evaluate import (
    EvalUser,
    bootstrap_ci,
    candidate_mask,
    collapse_targets,
    evaluate,
    popularity_percentile,
    summarize,
    user_arrays,
)
from anilist_rec.franchise import FranchiseIndex
from anilist_rec.signals import NEG, PLAN, STD, STRONG


def singleton_franchise(n: int) -> FranchiseIndex:
    return FranchiseIndex(
        item_franchise=-np.arange(1, n + 1, dtype=np.int64),
        is_entry=np.ones(n, dtype=bool),
        entry_of_franchise={-(i + 1): i for i in range(n)},
    )


def fixed_scorer(scores: list[float]):
    row = np.array(scores, dtype="float32")

    def score(fold_csr):
        return np.tile(row, (fold_csr.shape[0], 1))

    return score


# franchise fixture: items 0,1 form cluster 100 with entry 0; items 2,3 singletons
@pytest.fixture
def franchise():
    return FranchiseIndex(
        item_franchise=np.array([100, 100, -3, -4], dtype=np.int64),
        is_entry=np.array([True, False, True, True]),
        entry_of_franchise={100: 0, -3: 2, -4: 3},
    )


def test_user_arrays_routes_each_kind():
    item_pos = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
    user = user_arrays(
        [
            (10, STD, 1.0, True),  # fold-in input
            (20, NEG, 0.0, True),  # watched but weightless
            (30, PLAN, 0.0, False),
            (40, STRONG, 2.0, False),  # window target, graded 2
            (50, NEG, 0.0, False),  # window negative -> regret
            (99, STD, 1.0, False),  # outside the corpus -> dropped
        ],
        item_pos,
    )
    assert user == EvalUser(
        fold_idx=[0], fold_w=[1.0], watched=[0, 1], plan_idx=[2], targets={3: 2.0}, negatives=[4]
    )


def test_candidate_mask_excludes_watched_franchise_and_plan(franchise):
    mask = candidate_mask(franchise, watched=[1], plan_idx=[3])
    # watching item 1 kills its whole franchise (incl. entry 0); PLAN kills 3
    assert mask.tolist() == [False, False, True, False]


def test_collapse_drops_continuations_and_maps_to_entry(franchise):
    # user watched item 1: a target inside that franchise is a continuation -> dropped
    assert collapse_targets(franchise, {0: 1.0, 2: 1.0}, watched=[1]) == {2: 1.0}
    # unwatched franchise: the sequel target credits the entry point
    assert collapse_targets(franchise, {1: 2.0}, watched=[2]) == {0: 2.0}


def test_ndcg_perfect_ranking():
    fr = singleton_franchise(5)
    users = [EvalUser([0], [1.0], [0], [1], {2: 1.0}, [4])]
    result = evaluate(users, fixed_scorer([9, 8, 7, 6, 5]), fr, np.arange(5, 0, -1.0))
    assert result.ndcg10[0] == 1.0
    assert result.rec10[0] == 1.0
    assert result.regret10[0] == pytest.approx(0.1)  # the negative sneaks into the top-10


def test_ndcg_discounts_late_hit():
    fr = singleton_franchise(5)
    users = [EvalUser([0], [1.0], [0], [1], {2: 1.0}, [])]
    # item 3 outranks the target -> hit at rank 2
    result = evaluate(users, fixed_scorer([0, 0, 7, 8, 0]), fr, np.arange(5, 0, -1.0))
    assert result.ndcg10[0] == pytest.approx(1 / np.log2(3))


def test_ndcg_graded_gains():
    fr = singleton_franchise(5)
    users = [EvalUser([0], [1.0], [0], [], {2: 2.0, 3: 1.0}, [])]
    # standard target ranked above the strong one
    result = evaluate(users, fixed_scorer([0, 0, 7, 8, 0]), fr, np.arange(5, 0, -1.0))
    ideal = 2 + 1 / np.log2(3)
    assert result.ndcg10[0] == pytest.approx((1 + 2 / np.log2(3)) / ideal)


def test_hit_via_franchise_entry_point(franchise):
    # target is the sequel (item 1) of an unwatched franchise; ranker surfaces entry 0
    users = [EvalUser([2], [1.0], [2], [], {1: 1.0}, [])]
    result = evaluate(users, fixed_scorer([9, 0, 0, 0]), franchise, np.ones(4))
    assert result.ndcg10[0] == 1.0


def test_users_without_targets_skip_metrics_but_count_coverage():
    fr = singleton_franchise(5)
    users = [
        EvalUser([0], [1.0], [0], [], {2: 1.0}, []),
        EvalUser([0], [1.0], [0], [], {}, []),  # no window targets
    ]
    result = evaluate(users, fixed_scorer([0, 0, 3, 2, 1]), fr, np.arange(5, 0, -1.0))
    assert len(result.ndcg10) == 1
    assert result.coverage10 == 1.0


def test_chunking_does_not_change_results():
    fr = singleton_franchise(6)
    rng = np.random.default_rng(0)
    users = [
        EvalUser([int(i)], [1.0], [int(i)], [], {int((i + 1) % 6): 1.0}, [])
        for i in rng.integers(0, 6, 5)
    ]
    scorer = fixed_scorer(rng.random(6).tolist())
    one = evaluate(users, scorer, fr, np.ones(6), chunk=1)
    all_at_once = evaluate(users, scorer, fr, np.ones(6), chunk=1000)
    np.testing.assert_array_equal(one.ndcg10, all_at_once.ndcg10)
    assert one.coverage10 == all_at_once.coverage10


def test_popularity_percentile_orders_by_count():
    pct = popularity_percentile(np.array([50.0, 10.0, 30.0]))
    assert pct.tolist() == [100.0, 0.0, 50.0]


def test_bootstrap_ci_of_constant_is_tight():
    lo, hi = bootstrap_ci(np.full(100, 0.5), seed=1)
    assert lo == hi == 0.5


def test_summarize_reports_niche_slice():
    result = evaluate(
        [EvalUser([0], [1.0], [0], [], {2: 1.0}, [])],
        fixed_scorer([0, 0, 3, 2, 1]),
        singleton_franchise(5),
        np.arange(5, 0, -1.0),
    )
    summary = summarize(result, seed=1)
    assert summary["n_users"] == 1
    assert summary["ndcg10"] == 1.0
    assert summary["pop_lift"] == pytest.approx(float(result.top_pct[0] - result.prof_pct[0]))
