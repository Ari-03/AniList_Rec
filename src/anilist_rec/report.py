"""Markdown eval report: the SPEC §4 bar table plus run provenance."""

from datetime import UTC, datetime
from pathlib import Path

from anilist_rec.config import Config

BAR_COLUMNS = [
    (
        "NDCG@10 [95% CI]",
        lambda s: f"{s['ndcg10']:.4f} [{s['ndcg10_ci_lo']:.4f}, {s['ndcg10_ci_hi']:.4f}]",
    ),
    ("recall@10", lambda s: f"{s['recall10']:.3f}"),
    ("recall@50", lambda s: f"{s['recall50']:.3f}"),
    ("regret@10", lambda s: f"{s['regret10']:.4f}"),
    ("pop lift", lambda s: f"{s['pop_lift']:+.1f}"),
    ("pop lift (niche)", lambda s: f"{s['pop_lift_niche']:+.1f}"),
    ("coverage@10", lambda s: f"{s['coverage10']:.1%}"),
    ("users", lambda s: f"{s['n_users']}"),
]


def bar_table(summaries: dict[str, dict[str, float]]) -> str:
    header = "| model | " + " | ".join(name for name, _fmt in BAR_COLUMNS) + " |"
    divider = "|---" * (len(BAR_COLUMNS) + 1) + "|"
    rows = [
        "| " + " | ".join([model, *[fmt(summary) for _name, fmt in BAR_COLUMNS]]) + " |"
        for model, summary in summaries.items()
    ]
    return "\n".join([header, divider, *rows])


def render_report(
    summaries: dict[str, dict[str, float]], cfg: Config, run_stats: dict[str, int]
) -> str:
    cap = cfg.train_user_cap if cfg.train_user_cap is not None else "uncapped"
    return f"""# Baseline bar — full-scale offline eval

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by `uv run baseline`
([Ari-03/AniList_Rec#14](https://github.com/Ari-03/AniList_Rec/issues/14)).
Protocol: SPEC §5 — held-out users, per-user temporal 80/20, full-catalogue
ranking through the serving pipeline (franchise filter on), dial off, test users.

## Test-set metrics (dial off)

{bar_table(summaries)}

A candidate architecture ships only if it beats **both** models on NDCG@10
without MostPopular's degenerate coverage (SPEC §4). Pop lift = mean popularity
percentile of the top-10 minus the median of the user's profile (guardrail, not
optimized); niche = bottom quartile of users by profile popularity. regret@10 =
fraction of the top-10 the user actually dropped or scored low.

## Run provenance

| | |
|---|---|
| training users | {run_stats["n_train_users"]:,} (cap: {cap}) |
| training interactions (weight > 0) | {run_stats["n_train_interactions"]:,} |
| item universe | {run_stats["n_items"]:,} |
| holdout | {run_stats["n_test_users"]:,} test + {run_stats["n_val_users"]:,} validation users |
| seed | {cfg.seed} |
| BM25 | K={cfg.k_neighbors}, K1=1.2, B=0.75 |
"""


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
