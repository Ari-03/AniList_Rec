"""One command: train both baselines and emit the eval report (issue #14).

    uv run baseline

Builds every derived artifact it's missing (signal table, holdout split,
similarity matrix — all cached under data/derived/), evaluates on the test
users with the dial off, and writes the bar table to reports/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from anilist_rec.config import Config
from anilist_rec.evaluate import eval_users, evaluate, summarize
from anilist_rec.franchise import build_franchise_index
from anilist_rec.matrix import build_training_matrix, item_index, item_positions
from anilist_rec.models import bm25_scorer, fit_bm25, most_popular_scorer
from anilist_rec.report import bar_table, render_report, write_report
from anilist_rec.signals import build_signals
from anilist_rec.split import build_holdout


def run(cfg: Config, report_path: Path) -> dict[str, dict[str, float]]:
    """The full offline pipeline; returns the per-model metric summaries."""
    t0 = time.perf_counter()

    def stage(name: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {name}", flush=True)

    stage("signal table")
    signals = build_signals(cfg)

    stage("holdout split")
    holdout = build_holdout(cfg, signals)
    holdout_users = set(holdout["user_id"].unique())

    stage("training matrix")
    item_ids = item_index(signals)
    x_train, item_counts = build_training_matrix(cfg, signals, item_ids, holdout_users)
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.item_counts_path, item_counts)  # serving path reloads without the matrix

    stage("franchise index")
    catalogue = pl.read_parquet(cfg.crosswalk_path)
    franchise = build_franchise_index(catalogue, item_ids)

    stage(f"BM25 fit ({x_train.shape[0]:,} users x {x_train.shape[1]:,} items)")
    similarity = fit_bm25(cfg, x_train)

    stage("evaluate test users")
    test_users = eval_users(holdout, "test", item_positions(item_ids))
    results = {
        "item-item BM25": evaluate(test_users, bm25_scorer(similarity), franchise, item_counts),
        "MostPopular": evaluate(
            test_users, most_popular_scorer(item_counts), franchise, item_counts
        ),
    }
    summaries = {name: summarize(result, cfg.seed) for name, result in results.items()}

    stage("report")
    roles = holdout.select("user_id", "role").unique()
    run_stats = {
        "n_train_users": x_train.shape[0],
        "n_train_interactions": x_train.nnz,
        "n_items": len(item_ids),
        "n_test_users": roles.filter(pl.col("role") == "test").height,
        "n_val_users": roles.filter(pl.col("role") == "val").height,
    }
    write_report(report_path, render_report(summaries, cfg, run_stats))
    print(f"\n{bar_table(summaries)}\n\nwrote {report_path}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_bar.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="cap training users (dev speed only; default uncapped)",
    )
    args = parser.parse_args()
    cfg = Config(data_dir=args.data_dir, seed=args.seed, train_user_cap=args.cap)
    run(cfg, args.report)


if __name__ == "__main__":
    main()
