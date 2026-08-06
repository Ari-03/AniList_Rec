"""Held-out users with a per-user temporal 80/20 split (SPEC §5).

Holdout users are absent from training entirely, mirroring the fold-in serving
path: the earliest 80% of each user's history is the fold-in input, the last
20% the target window.

Null `ts` rows (1.3% of the corpus have no last_interaction_date) can't be
placed on the user's timeline, so they sort before every dated row and land in
the fold-in input — an undatable interaction is still valid model input, but
never a provably-later target. Users so undated that nulls would spill into
their window fail qualification.
"""

import numpy as np
import polars as pl

from anilist_rec.config import Config
from anilist_rec.signals import PLAN, STD, STRONG


def split_users(rows: pl.DataFrame, fold_fraction: float) -> pl.DataFrame:
    """Add a `fold` column: True = fold-in input, False = target window / PLAN."""
    non_plan = pl.col("kind") != PLAN
    return (
        # nulls_last=False pins undated rows to the front (see module docstring)
        rows.sort(["user_id", "ts", "anime_id"], nulls_last=False)
        .with_columns(
            non_plan.cum_sum().over("user_id").alias("rank"),
            non_plan.sum().over("user_id").alias("n"),
        )
        .with_columns(
            (non_plan & (pl.col("rank") <= (fold_fraction * pl.col("n")).floor())).alias("fold")
        )
        .drop("rank", "n")
    )


def qualified_users(split_rows: pl.DataFrame) -> set[str]:
    """Users with ≥5 target-window positives, a scoreable fold-in, and a dated window (SPEC §5)."""
    window = (~pl.col("fold")) & (pl.col("kind") != PLAN)
    stats = split_rows.group_by("user_id").agg(
        ((~pl.col("fold")) & pl.col("kind").is_in([STD, STRONG])).sum().alias("window_positives"),
        (pl.col("fold") & (pl.col("weight") > 0)).sum().alias("foldin_positives"),
        (window & pl.col("ts").is_null()).sum().alias("undated_window"),
    )
    ok = stats.filter(
        (pl.col("window_positives") >= 5)
        & (pl.col("foldin_positives") >= 1)
        & (pl.col("undated_window") == 0)
    )
    return set(ok["user_id"])


def build_holdout(cfg: Config, signals: pl.LazyFrame) -> pl.DataFrame:
    """Select test/val users and split them temporally (cached on disk).

    Columns: user_id, role (test|val), anime_id, kind, weight, fold.
    """
    if cfg.holdout_path.exists():
        return pl.read_parquet(cfg.holdout_path)

    # candidates: ≥20 lifetime interactions (PLAN doesn't count), seeded shuffle
    counts = (
        signals.filter(pl.col("kind") != PLAN)
        .group_by("user_id")
        .len()
        .filter(pl.col("len") >= 20)
        .sort("user_id")  # group_by order is nondeterministic; sort before the seeded shuffle
        .collect(engine="streaming")
    )
    rng = np.random.default_rng(cfg.seed)
    candidates = counts["user_id"].to_numpy()
    candidates = candidates[rng.permutation(len(candidates))][: cfg.holdout_candidates]

    rows = split_users(
        signals.filter(pl.col("user_id").is_in(candidates.tolist())).collect(engine="streaming"),
        cfg.fold_fraction,
    )

    # first qualifying n_test users (in shuffle order) are test, the next n_val val
    ok = qualified_users(rows)
    ordered = [u for u in candidates.tolist() if u in ok]
    roles = pl.DataFrame(
        {
            "user_id": ordered[: cfg.n_test + cfg.n_val],
            "role": ["test"] * min(cfg.n_test, len(ordered))
            + ["val"] * max(0, min(cfg.n_val, len(ordered) - cfg.n_test)),
        },
        schema={"user_id": pl.String, "role": pl.String},
    )

    holdout = rows.join(roles, on="user_id", how="inner").select(
        "user_id", "role", "anime_id", "kind", "weight", "fold"
    )
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    holdout.write_parquet(cfg.holdout_path)
    return holdout
