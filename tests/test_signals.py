"""Signal mapping (SPEC §1, adapted to svanoo's MAL statuses — no REPEATING)."""

import polars as pl
import pytest

from anilist_rec.signals import NEG, PARTIAL, PLAN, STD, STRONG, map_signals


def run_mapping(rows: list[dict], episodes: dict[int, float] | None = None) -> pl.DataFrame:
    defaults = {
        "user_id": "u",
        "anime_id": 1,
        "favorite": 0,
        "score": None,
        "status": "completed",
        "progress": None,
        "last_interaction_date": None,
    }
    interactions = pl.LazyFrame(
        [{**defaults, **r} for r in rows],
        schema={
            "user_id": pl.String,
            "anime_id": pl.Int32,
            "favorite": pl.Int8,
            "score": pl.Int8,
            "status": pl.String,
            "progress": pl.Int32,
            "last_interaction_date": pl.Datetime("ms"),
        },
    )
    eps = pl.LazyFrame(
        {"idMal": list((episodes or {}).keys()), "episodes": list((episodes or {}).values())},
        schema={"idMal": pl.Int32, "episodes": pl.Float64},
    )
    return map_signals(interactions, eps).collect()


def kind_weight(row: dict, episodes: dict[int, float] | None = None) -> tuple[int, float]:
    out = run_mapping([row], episodes)
    assert len(out) == 1
    return out["kind"][0], out["weight"][0]


def test_completed_high_score_is_strong():
    assert kind_weight({"status": "completed", "score": 8}) == (STRONG, 2.0)
    assert kind_weight({"status": "completed", "score": 10}) == (STRONG, 2.0)


def test_favourite_is_strong_regardless_of_status():
    assert kind_weight({"status": "dropped", "favorite": 1}) == (STRONG, 2.0)
    assert kind_weight({"status": "completed", "score": 3, "favorite": 1}) == (STRONG, 2.0)


def test_completed_unscored_or_mid_is_standard():
    assert kind_weight({"status": "completed", "score": None}) == (STD, 1.0)
    assert kind_weight({"status": "completed", "score": 5}) == (STD, 1.0)
    assert kind_weight({"status": "completed", "score": 7}) == (STD, 1.0)


def test_completed_low_score_is_mild_negative():
    assert kind_weight({"status": "completed", "score": 4}) == (NEG, -0.25)
    assert kind_weight({"status": "completed", "score": 1}) == (NEG, -0.25)


def test_dropped_is_negative_with_confidence_inverse_to_progress():
    # unknown progress: midpoint confidence
    assert kind_weight({"status": "dropped", "score": 9}) == (NEG, -0.625)
    # dropped at ep 2 of 10 ≫ dropped at ep 8 of 10 (SPEC §1)
    early = kind_weight({"status": "dropped", "anime_id": 7, "progress": 2}, episodes={7: 10.0})
    late = kind_weight({"status": "dropped", "anime_id": 7, "progress": 8}, episodes={7: 10.0})
    assert early == (NEG, pytest.approx(-0.85))
    assert late == (NEG, pytest.approx(-0.4))
    assert early[1] < late[1] < 0


def test_watching_weight_scales_with_progress():
    row = {"status": "watching", "anime_id": 7, "progress": 5}
    assert kind_weight(row, episodes={7: 10.0}) == (PARTIAL, 0.75)
    # progress beyond the episode count clips to full confidence
    assert kind_weight({**row, "progress": 25}, episodes={7: 10.0}) == (PARTIAL, 1.0)


def test_watching_unknown_progress_or_episodes_gets_midpoint():
    assert kind_weight({"status": "watching", "anime_id": 7}, episodes={7: 10.0}) == (
        PARTIAL,
        0.75,
    )
    assert kind_weight({"status": "watching", "anime_id": 99, "progress": 5}) == (PARTIAL, 0.75)


def test_on_hold_is_near_neutral():
    assert kind_weight({"status": "on_hold", "progress": 5}) == (PARTIAL, 0.25)


def test_planning_kept_but_weightless():
    assert kind_weight({"status": "plan_to_watch", "score": 9}) == (PLAN, 0.0)


def test_unknown_status_rows_are_dropped():
    assert len(run_mapping([{"status": "rewatching"}])) == 0


def test_output_schema():
    out = run_mapping([{"status": "completed", "score": 8}])
    assert out.columns == ["user_id", "anime_id", "kind", "weight", "ts"]
    assert out["kind"].dtype == pl.UInt8
    assert out["weight"].dtype == pl.Float32
