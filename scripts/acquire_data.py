# /// script
# requires-python = ">=3.12"
# dependencies = ["polars", "kagglehub"]
# ///
"""Acquire the training data (Ari-03/AniList_Rec#12).

Downloads both Kaggle datasets (public — no credentials needed), converts the
svanoo user_anime shards to a single Parquet file, and builds the
AniList<->MAL crosswalk from the calebmwelsh catalogue. Idempotent: kagglehub
caches downloads, and the outputs are rewritten in place.

    uv run scripts/acquire_data.py

Outputs under data/ (gitignored — interactions.parquet is 977 MB):
  interactions.parquet         223.8M rows: user_id, anime_id, favorite,
                               score, status, progress, last_interaction_date
  crosswalk_anilist_mal.parquet  20.4k AniList anime with MAL ids + titles

Review columns and the friend-graph file are excluded per the dataset
decision (Ari-03/AniList_Rec#6).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "kagglehub"))

import kagglehub
import polars as pl

svanoo = Path(kagglehub.dataset_download("svanoo/myanimelist-dataset"))
catalogue = Path(kagglehub.dataset_download("calebmwelsh/anilist-anime-dataset"))

out = ROOT / "interactions.parquet"
(
    pl.scan_csv(
        str(svanoo / "user_anime*.csv"),
        separator="\t",
        # read everything as string first; dirty rows otherwise break inference
        infer_schema=False,
    )
    .select(
        pl.col("user_id"),
        pl.col("anime_id").cast(pl.Int32),
        pl.col("favorite").cast(pl.Int8),
        pl.col("score").cast(pl.Int8, strict=False),
        pl.col("status").cast(pl.Categorical),
        pl.col("progress").cast(pl.Int32, strict=False),
        pl.col("last_interaction_date").str.strptime(
            pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S", strict=False
        ),
    )
    .sink_parquet(str(out), compression="zstd", row_group_size=1_000_000)
)
print("wrote", out, f"{out.stat().st_size / 1e9:.2f} GB")

xw_out = ROOT / "crosswalk_anilist_mal.parquet"
(
    pl.scan_csv(
        str(catalogue / "anilist_anime_data_complete.csv"),
        infer_schema_length=10000,
    )
    .filter(pl.col("type") == "ANIME")
    .select(
        pl.col("id").cast(pl.Int32),
        pl.col("idMal").cast(pl.Int32),
        "format",
        "title_romaji",
        "episodes",
        "isAdult",
    )
    .collect()
    .write_parquet(str(xw_out))
)
print("wrote", xw_out, f"{xw_out.stat().st_size / 1024:.0f} KB")
