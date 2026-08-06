"""End-to-end smoke test: synthetic corpus through the whole pipeline."""

import json
from datetime import datetime

import numpy as np
import polars as pl

from anilist_rec.baseline import run
from anilist_rec.config import Config

N_ITEMS = 60  # 30 per taste cluster, comfortably above the ≥20 lifetime filter


def write_corpus(data_dir):
    """60 users with taste clusters: evens like even items, odds like odd items."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        liked = [i for i in range(N_ITEMS) if i % 2 == u % 2]
        watched = rng.permutation(liked)[:24]
        for step, item in enumerate(watched):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "anime_id": int(item) + 1,
                    "favorite": 0,
                    "score": int(rng.integers(7, 11)),
                    "status": "completed",
                    "progress": None,
                    # one undated row per user exercises the null-ts path
                    "last_interaction_date": None if step == 0 else datetime(2020, 1, 1 + step),
                }
            )
    pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "anime_id": pl.Int32,
            "favorite": pl.Int8,
            "score": pl.Int8,
            "status": pl.String,
            "progress": pl.Int32,
            "last_interaction_date": pl.Datetime("ms"),
        },
    ).write_parquet(data_dir / "interactions.parquet")

    # crosswalk: MAL id i+1 <-> AniList id 1000+i; items 1,2 are one franchise
    relations = [
        json.dumps([{"relationType": "SEQUEL", "node": {"id": 1001, "type": "ANIME"}}])
        if i == 0
        else "[]"
        for i in range(N_ITEMS)
    ]
    pl.DataFrame(
        {
            "id": [1000 + i for i in range(N_ITEMS)],
            "idMal": [i + 1 for i in range(N_ITEMS)],
            "format": ["TV"] * N_ITEMS,
            "title_romaji": [f"anime {i}" for i in range(N_ITEMS)],
            "title_english": [None] * N_ITEMS,
            "seasonYear": [2000 + i for i in range(N_ITEMS)],
            "episodes": [12.0] * N_ITEMS,
            "popularity": [1000 - i for i in range(N_ITEMS)],
            "genres": ["[]"] * N_ITEMS,
            "coverImage_medium": [None] * N_ITEMS,
            "siteUrl": [None] * N_ITEMS,
            "relations": relations,
            "isAdult": [False] * N_ITEMS,
        },
        schema_overrides={"id": pl.Int32, "idMal": pl.Int32, "seasonYear": pl.Int32},
    ).write_parquet(data_dir / "crosswalk_anilist_mal.parquet")


def test_pipeline_end_to_end(tmp_path):
    write_corpus(tmp_path)
    cfg = Config(data_dir=tmp_path, n_test=5, n_val=3, holdout_candidates=20, k_neighbors=10)
    report_path = tmp_path / "reports" / "baseline_bar.md"

    summaries = run(cfg, report_path)

    assert set(summaries) == {"item-item BM25", "MostPopular"}
    for summary in summaries.values():
        assert summary["n_users"] > 0
        assert 0.0 <= summary["ndcg10"] <= 1.0
        assert summary["ndcg10_ci_lo"] <= summary["ndcg10"] <= summary["ndcg10_ci_hi"]
    # taste clusters are strong signal: BM25 must crush the popularity baseline here
    assert summaries["item-item BM25"]["ndcg10"] > summaries["MostPopular"]["ndcg10"]

    text = report_path.read_text()
    assert "item-item BM25" in text and "MostPopular" in text
    assert "uncapped" in text

    # cached artifacts make a re-run reproduce the same numbers
    assert run(cfg, report_path) == summaries
