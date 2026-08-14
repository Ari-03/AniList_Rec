"""Fold-in mapping (§1 over AniList statuses) and the serving path end to end."""

import numpy as np
import polars as pl
import pytest

from anilist_rec.anilist import ListEntry, UserAnimeList
from anilist_rec.foldin import build_foldin, entry_signal
from anilist_rec.franchise import FranchiseIndex
from anilist_rec.serve import Recommender
from anilist_rec.signals import NEG, PARTIAL, PLAN, STD, STRONG


def make_entry(media_id: int, mal_id: int | None, status: str, **kw) -> ListEntry:
    return ListEntry(
        entry_id=kw.get("entry_id", media_id),
        media_id=media_id,
        mal_id=mal_id,
        status=status,
        score100=kw.get("score100", 0.0),
        progress=kw.get("progress"),
        repeat=kw.get("repeat", 0),
        episodes=kw.get("episodes"),
    )


@pytest.mark.parametrize(
    ("entry", "favourited", "kind", "weight"),
    [
        (make_entry(1, 1, "PLANNING"), False, PLAN, 0.0),
        (make_entry(1, 1, "COMPLETED", score100=85), False, STRONG, 2.0),
        (make_entry(1, 1, "COMPLETED", score100=80), False, STRONG, 2.0),
        (make_entry(1, 1, "COMPLETED", score100=65), False, STD, 1.0),
        (make_entry(1, 1, "COMPLETED"), False, STD, 1.0),  # unscored completion
        (make_entry(1, 1, "COMPLETED", score100=40), False, NEG, 0.0),
        (make_entry(1, 1, "COMPLETED", score100=1), False, NEG, 0.0),
        (make_entry(1, 1, "DROPPED", score100=85), False, NEG, 0.0),  # high score doesn't save it
        (make_entry(1, 1, "DROPPED"), True, STRONG, 2.0),  # favourite outranks dropped
        (make_entry(1, 1, "REPEATING"), False, STRONG, 2.0),
        (make_entry(1, 1, "COMPLETED", repeat=2), False, STRONG, 2.0),
        (make_entry(1, 1, "PAUSED"), False, PARTIAL, 0.25),
        (make_entry(1, 1, "CURRENT", progress=12, episodes=12), False, PARTIAL, 1.0),
        (make_entry(1, 1, "CURRENT", progress=0, episodes=12), False, PARTIAL, 0.5),
        (make_entry(1, 1, "CURRENT"), False, PARTIAL, 0.75),  # unknown episodes: midpoint
        (make_entry(1, 1, ""), False, PLAN, 0.0),  # unknown status never folds in
    ],
)
def test_entry_signal_mapping(entry, favourited, kind, weight):
    assert entry_signal(entry, favourited) == (kind, weight)


def test_build_foldin_routes_and_drops():
    item_pos = {10: 0, 20: 1, 30: 2, 40: 3}
    user = UserAnimeList(
        entries=[
            make_entry(100, 10, "COMPLETED", score100=90),  # strong fold-in
            make_entry(200, 20, "DROPPED"),  # watched, weightless
            make_entry(300, 30, "PLANNING"),  # excluded from fold-in, blocks output
            make_entry(400, 99, "COMPLETED"),  # MAL id outside the corpus -> dropped
            make_entry(500, None, "COMPLETED"),  # no MAL id -> dropped
        ],
    )
    fold = build_foldin(user, item_pos)
    assert fold.fold_idx == [0]
    assert fold.fold_w == [2.0]
    assert fold.watched == [0, 1]
    assert fold.plan_idx == [2]
    assert fold.n_entries == 5
    assert fold.n_unmapped == 2

    csr = fold.to_csr(4)
    assert csr.shape == (1, 4)
    assert csr[0, 0] == 2.0
    assert csr.nnz == 1


def test_favourite_matches_on_anilist_media_id():
    item_pos = {10: 0}
    user = UserAnimeList(
        entries=[make_entry(100, 10, "COMPLETED", score100=50)],
        favourite_media_ids={100},  # favourites come back as AniList ids
    )
    fold = build_foldin(user, item_pos)
    assert fold.fold_w == [2.0]


# --- serving path ---------------------------------------------------------


def catalogue_frame(rows: list[tuple[int, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [r[0] for r in rows],
            "idMal": [r[1] for r in rows],
            "title_english": [r[2] for r in rows],
            "title_romaji": [r[2] for r in rows],
            "popularity": [100 - i for i in range(len(rows))],
        },
        schema_overrides={"id": pl.Int32, "idMal": pl.Int32},
    )


def fixed_scorer(row: list[float]):
    arr = np.array(row, dtype="float32")

    def score(fold_csr):
        return np.tile(arr, (fold_csr.shape[0], 1))

    return score


@pytest.fixture
def recommender():
    # items (MAL ids) 10,20,30,40; 10+20 one franchise with entry 10; 30,40 singletons
    item_ids = np.array([10, 20, 30, 40])
    franchise = FranchiseIndex(
        item_franchise=np.array([100, 100, -3, -4], dtype=np.int64),
        is_entry=np.array([True, False, True, True]),
        entry_of_franchise={100: 0, -3: 2, -4: 3},
    )
    catalogue = catalogue_frame(
        [(1000, 10, "A"), (2000, 20, "A2"), (3000, 30, "B"), (4000, 40, "C")]
    )
    return Recommender(
        fixed_scorer([4.0, 3.0, 2.0, 1.0]),
        item_ids,
        franchise,
        item_counts=np.array([100.0, 10.0, 10.0, 1.0]),
        catalogue=catalogue,
    )


def test_recommend_excludes_own_list_and_franchise(recommender):
    # watched MAL 20 (franchise of 10+20), planning MAL 30
    user = UserAnimeList(
        entries=[
            make_entry(2000, 20, "COMPLETED", score100=90),
            make_entry(3000, 30, "PLANNING"),
        ],
    )
    recs = recommender.recommend_foldin(recommender.fold_in(user))
    # 20 watched kills entry 10 too (same franchise); PLANNING kills 30
    assert [r.mal_id for r in recs] == [40]
    assert recs[0].anilist_id == 4000
    assert recs[0].title == "C"


def test_recommend_dial_reranks_popular_down(recommender):
    user = UserAnimeList(entries=[make_entry(9999, None, "COMPLETED", score100=90)])
    fold = recommender.fold_in(user)
    dial_off = recommender.recommend_foldin(fold, dial=0.0)
    assert dial_off[0].mal_id == 10  # highest raw score
    dialed = recommender.recommend_foldin(fold, dial=1.0)
    # item 10 is 100x more popular; full dial pushes it below the niche items
    assert dialed[0].mal_id != 10


def test_recommend_respects_limit(recommender):
    user = UserAnimeList(entries=[make_entry(9999, None, "COMPLETED")])
    recs = recommender.recommend_foldin(recommender.fold_in(user), limit=2)
    assert len(recs) == 2
