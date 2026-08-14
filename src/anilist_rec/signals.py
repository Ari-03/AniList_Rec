"""Interaction → training-signal mapping (SPEC §1, adapted to svanoo's MAL statuses).

svanoo has no REPEATING status, so strong positives are favourited or score ≥ 8.
Scores are already 1-10; score null means unrated, never zero.
"""

import polars as pl

from anilist_rec.config import Config

# Signal kinds. The weight column is signed preference confidence: positive
# for training positives, negative for NEG rows (candidates that model
# negatives — SPEC §4 — read the sign; positive-only consumers filter
# weight > 0 as before), and 0.0 for PLAN, which never trains.
NEG, STD, STRONG, PARTIAL, PLAN = 0, 1, 2, 3, 4

# Graded gains for NDCG (SPEC §5): strong positives count double.
GAIN = {STRONG: 2.0, STD: 1.0}


def kind_expr() -> pl.Expr:
    """Classify one interaction row; null for statuses outside the mapping."""
    status = pl.col("status").cast(pl.String)
    score = pl.col("score")
    favourited = pl.col("favorite") == 1
    return (
        pl.when(status == "plan_to_watch")
        .then(PLAN)
        .when(favourited)
        .then(STRONG)
        .when((status == "dropped") | ((status == "completed") & score.is_between(1, 4)))
        .then(NEG)
        .when((status == "completed") & (score >= 8))
        .then(STRONG)
        .when(status == "completed")
        .then(STD)
        .when(status.is_in(["watching", "on_hold"]))
        .then(PARTIAL)
    )


def weight_expr() -> pl.Expr:
    """Signed training weight given a `kind` column; needs `episodes` joined in."""
    status = pl.col("status").cast(pl.String)
    # confidence scaled by progress/episodes (SPEC §1 CURRENT); midpoint when unknown
    progress_ratio = (pl.col("progress") / pl.col("episodes")).clip(0.0, 1.0).fill_null(0.5)
    watch_w = 0.5 + 0.5 * progress_ratio
    # DROPPED confidence inverse to progress (SPEC §1: dropped at ep 2 ≫ ep 20)
    drop_w = -(0.25 + 0.75 * (1.0 - progress_ratio))
    return (
        pl.when(pl.col("kind") == STRONG)
        .then(2.0)
        .when(pl.col("kind") == STD)
        .then(1.0)
        .when((pl.col("kind") == PARTIAL) & (status == "watching"))
        .then(watch_w)
        .when(pl.col("kind") == PARTIAL)
        .then(0.25)
        .when((pl.col("kind") == NEG) & (status == "dropped"))
        .then(drop_w)
        .when(pl.col("kind") == NEG)
        .then(-0.25)  # mild negative: completed but scored ≤ 4/10
        .otherwise(0.0)
    )


def episodes_by_mal(crosswalk: pl.LazyFrame) -> pl.LazyFrame:
    """Episode counts keyed by MAL id (max across duplicate crosswalk entries)."""
    return crosswalk.drop_nulls("idMal").group_by("idMal").agg(pl.col("episodes").max())


def map_signals(interactions: pl.LazyFrame, episodes: pl.LazyFrame) -> pl.LazyFrame:
    """One signal row per mappable interaction: user_id, anime_id, kind, weight, ts."""
    return (
        interactions.join(episodes, left_on="anime_id", right_on="idMal", how="left")
        .with_columns(kind_expr().alias("kind"))
        .filter(pl.col("kind").is_not_null())
        .with_columns(weight_expr().cast(pl.Float32).alias("weight"))
        .select(
            "user_id",
            "anime_id",
            pl.col("kind").cast(pl.UInt8),
            "weight",
            pl.col("last_interaction_date").alias("ts"),
        )
    )


def build_signals(cfg: Config) -> pl.LazyFrame:
    """Materialize the signal table (streamed; cached on disk)."""
    if not cfg.signals_path.exists():
        cfg.derived_dir.mkdir(parents=True, exist_ok=True)
        eps = episodes_by_mal(pl.scan_parquet(cfg.crosswalk_path))
        map_signals(pl.scan_parquet(cfg.interactions_path), eps).sink_parquet(
            cfg.signals_path, compression="zstd"
        )
    return pl.scan_parquet(cfg.signals_path)
