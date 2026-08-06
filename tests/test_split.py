"""Held-out user selection + per-user temporal 80/20 split (SPEC §5)."""

from datetime import datetime

import polars as pl
import pytest

from anilist_rec.config import Config
from anilist_rec.signals import NEG, PLAN, STD
from anilist_rec.split import build_holdout, split_users

SCHEMA = {
    "user_id": pl.String,
    "anime_id": pl.Int32,
    "kind": pl.UInt8,
    "weight": pl.Float32,
    "ts": pl.Datetime("ms"),
}


def day(n: int) -> datetime:
    return datetime(2020, 1, 1 + n)


def frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        [dict(zip(SCHEMA, r, strict=True)) for r in rows],
        schema=SCHEMA,
    )


def test_fold_is_earliest_80_percent():
    rows = frame([("u", i, STD, 1.0, day(i)) for i in range(10)])
    out = split_users(rows, fold_fraction=0.8).sort("ts")
    assert out["fold"].to_list() == [True] * 8 + [False] * 2


def test_null_dates_always_land_in_fold_in():
    # two undated rows with the largest anime_ids: still fold-in, never targets
    rows = frame(
        [("u", i, STD, 1.0, day(i)) for i in range(8)]
        + [("u", 100, STD, 1.0, None), ("u", 101, STD, 1.0, None)]
    )
    out = split_users(rows, fold_fraction=0.8)
    undated = out.filter(pl.col("ts").is_null())
    assert undated["fold"].to_list() == [True, True]
    # the two most recent *dated* rows become the target window instead
    assert set(out.filter(~pl.col("fold"))["anime_id"]) == {6, 7}


def test_planning_rows_neither_fold_nor_target():
    rows = frame(
        [("u", i, STD, 1.0, day(i)) for i in range(10)]
        + [("u", 50, PLAN, 0.0, day(0)), ("u", 51, PLAN, 0.0, day(20))]
    )
    out = split_users(rows, fold_fraction=0.8)
    plan = out.filter(pl.col("kind") == PLAN)
    assert plan["fold"].to_list() == [False, False]
    # PLAN rows don't count toward the 80/20 denominator
    assert out.filter(pl.col("fold")).height == 8


@pytest.fixture
def cfg(tmp_path):
    return Config(
        data_dir=tmp_path,
        n_test=3,
        n_val=2,
        holdout_candidates=50,
        seed=7,
    )


def user_rows(uid: str, n: int = 25, window_positives: int = 5) -> list[tuple]:
    """A user whose last `window_positives` of `n` dated rows are positives."""
    rows = []
    for i in range(n):
        kind = STD if i >= n - window_positives or i < n // 2 else NEG
        weight = 1.0 if kind == STD else 0.0
        rows.append((uid, i, kind, weight, day(i)))
    return rows


def build(cfg: Config, rows: list[tuple]) -> pl.DataFrame:
    return build_holdout(cfg, frame(rows).lazy())


def test_holdout_selects_disjoint_test_and_val(cfg):
    rows = [r for u in range(30) for r in user_rows(f"u{u}")]
    out = build(cfg, rows)
    roles = out.group_by("user_id").agg(
        pl.col("role").n_unique().alias("nroles"), pl.col("role").first()
    )
    assert roles["nroles"].max() == 1
    by_role = out.select("user_id", "role").unique().group_by("role").len().sort("role")
    assert dict(zip(by_role["role"], by_role["len"], strict=True)) == {"test": 3, "val": 2}


def test_holdout_excludes_thin_users(cfg):
    rows = [r for u in range(10) for r in user_rows(f"ok{u}")]
    rows += [
        (f"thin{u}", i, STD, 1.0, day(i)) for u in range(10) for i in range(19)
    ]  # <20 lifetime
    rows += [
        r for u in range(10) for r in user_rows(f"few{u}", window_positives=2)
    ]  # <5 window pos
    out = build(cfg, rows)
    users = set(out["user_id"])
    assert users and all(u.startswith("ok") for u in users)


def test_holdout_requires_a_scoreable_fold_in(cfg):
    # all fold-in rows are negatives -> nothing to fold in, user excluded
    rows = [("neg", i, NEG if i < 20 else STD, 0.0 if i < 20 else 1.0, day(i)) for i in range(25)]
    rows += [r for u in range(8) for r in user_rows(f"ok{u}")]
    out = build(cfg, rows)
    assert "neg" not in set(out["user_id"])


def test_holdout_excludes_users_with_undated_windows(cfg):
    # a user so undated that nulls spill into the target window is not a
    # temporal split at all — qualification must reject them
    rows = [("undated", i, STD, 1.0, None) for i in range(25)]
    rows += [r for u in range(8) for r in user_rows(f"ok{u}")]
    out = build(cfg, rows)
    assert "undated" not in set(out["user_id"])


def test_holdout_is_deterministic(cfg, tmp_path):
    rows = [r for u in range(30) for r in user_rows(f"u{u}")]
    first = build(cfg, rows)
    cfg.holdout_path.unlink()
    again = build(cfg, rows)
    assert first.equals(again)
