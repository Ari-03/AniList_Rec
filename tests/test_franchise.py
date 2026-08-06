"""Franchise clustering + entry-point collapse (SPEC §1)."""

import json

import numpy as np
import polars as pl

from anilist_rec.franchise import build_franchise_index, franchise_clusters


def make_catalogue(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "idMal": None,
        "format": "TV",
        "seasonYear": 2010,
        "popularity": 100,
        "relations": "[]",
    }
    return pl.DataFrame(
        [{**defaults, **r} for r in rows],
        schema={
            "id": pl.Int32,
            "idMal": pl.Int32,
            "format": pl.String,
            "seasonYear": pl.Int32,
            "popularity": pl.Int64,
            "relations": pl.String,
        },
    )


def rel(relation_type: str, target_id: int, node_type: str = "ANIME") -> dict:
    return {"relationType": relation_type, "node": {"id": target_id, "type": node_type}}


def test_sequels_cluster_transitively():
    cat = make_catalogue(
        [
            {"id": 1, "relations": json.dumps([rel("SEQUEL", 2)])},
            {"id": 2, "relations": json.dumps([rel("SEQUEL", 3)])},
            {"id": 3},
            {"id": 9},
        ]
    )
    roots = franchise_clusters(cat)
    assert roots[1] == roots[2] == roots[3]
    assert roots[9] != roots[1]


def test_non_franchise_relations_ignored():
    cat = make_catalogue(
        [
            {"id": 1, "relations": json.dumps([rel("CHARACTER", 2)])},
            {"id": 2, "relations": json.dumps([rel("ADAPTATION", 1, node_type="MANGA")])},
        ]
    )
    roots = franchise_clusters(cat)
    assert roots[1] != roots[2]


def franchise_index(cat: pl.DataFrame, mal_ids: list[int]):
    return build_franchise_index(cat, np.array(mal_ids, dtype=np.int32))


def test_entry_point_prefers_tv_then_year():
    cat = make_catalogue(
        [
            # movie predates the TV series but TV still wins the entry point
            {
                "id": 1,
                "idMal": 11,
                "format": "MOVIE",
                "seasonYear": 2001,
                "relations": json.dumps([rel("SEQUEL", 2)]),
            },
            {
                "id": 2,
                "idMal": 12,
                "format": "TV",
                "seasonYear": 2005,
                "relations": json.dumps([rel("SEQUEL", 3)]),
            },
            {"id": 3, "idMal": 13, "format": "TV", "seasonYear": 2003},
        ]
    )
    idx = franchise_index(cat, [11, 12, 13])
    assert idx.item_franchise[0] == idx.item_franchise[1] == idx.item_franchise[2]
    # entry = earliest TV season (id 3 / MAL 13, corpus index 2)
    assert idx.is_entry.tolist() == [False, False, True]
    assert idx.entry_of_franchise[int(idx.item_franchise[0])] == 2


def test_popularity_breaks_ties():
    cat = make_catalogue(
        [
            {"id": 1, "idMal": 11, "popularity": 50, "relations": json.dumps([rel("SEQUEL", 2)])},
            {"id": 2, "idMal": 12, "popularity": 500},
        ]
    )
    idx = franchise_index(cat, [11, 12])
    assert idx.is_entry.tolist() == [False, True]


def test_entry_point_restricted_to_corpus():
    # MAL 13 is the natural entry but absent from the corpus, so MAL 12 stands in
    cat = make_catalogue(
        [
            {"id": 2, "idMal": 12, "seasonYear": 2005, "relations": json.dumps([rel("SEQUEL", 3)])},
            {"id": 3, "idMal": 13, "seasonYear": 2003},
        ]
    )
    idx = franchise_index(cat, [12])
    assert idx.is_entry.tolist() == [True]


def test_unmapped_items_are_singleton_entries():
    cat = make_catalogue([{"id": 1, "idMal": 11}])
    idx = franchise_index(cat, [11, 999])
    assert idx.is_entry.tolist() == [True, True]
    assert idx.item_franchise[1] < 0  # synthetic singleton label
    assert idx.item_franchise[0] != idx.item_franchise[1]


def test_duplicate_mal_ids_deduped_by_popularity():
    # 22 MAL ids map to 2 AniList entries (SPEC §2) — keep the most popular
    cat = make_catalogue(
        [
            {"id": 1, "idMal": 11, "popularity": 10, "format": "MOVIE"},
            {"id": 2, "idMal": 11, "popularity": 999, "format": "TV"},
        ]
    )
    idx = franchise_index(cat, [11])
    assert idx.item_franchise[0] == franchise_clusters(cat)[2]
