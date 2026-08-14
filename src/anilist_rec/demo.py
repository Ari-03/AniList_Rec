"""Live fold-in demo (issues #15/#16/#17): pull a public AniList list and rank
through the real serving path.

    uv run demo Zackhacks                # baseline BM25
    uv run demo Zackhacks --model ease   # EASE artifact (uv run ease first)
    uv run demo Zackhacks --model als    # ALS artifact (uv run als first)

The baseline builds any missing offline artifacts first (cached under
data/derived/); the candidate models load their exported artifacts.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.config import Config
from anilist_rec.franchise import build_franchise_index
from anilist_rec.matrix import build_training_matrix, item_index
from anilist_rec.models import bm25_scorer, fit_bm25
from anilist_rec.serve import Recommender
from anilist_rec.signals import build_signals
from anilist_rec.split import build_holdout


def load_baseline_recommender(cfg: Config) -> Recommender:
    """The baseline BM25 artifacts behind the serving path, built if missing."""
    signals = build_signals(cfg)
    item_ids = item_index(signals)
    catalogue = pl.read_parquet(cfg.crosswalk_path)

    if cfg.similarity_path.exists() and cfg.item_counts_path.exists():
        similarity = sp.load_npz(cfg.similarity_path)
        item_counts = np.load(cfg.item_counts_path)
    else:
        holdout = build_holdout(cfg, signals)
        x_train, item_counts = build_training_matrix(
            cfg, signals, item_ids, set(holdout["user_id"].unique())
        )
        similarity = fit_bm25(cfg, x_train)
        np.save(cfg.item_counts_path, item_counts)

    franchise = build_franchise_index(catalogue, item_ids)
    return Recommender(bm25_scorer(similarity), item_ids, franchise, item_counts, catalogue)


def load_candidate_recommender(cfg: Config, model: str) -> Recommender:
    """A candidate's exported artifact behind the same serving path."""
    signals = build_signals(cfg)
    item_ids = item_index(signals)
    catalogue = pl.read_parquet(cfg.crosswalk_path)
    franchise = build_franchise_index(catalogue, item_ids)
    item_counts = np.load(cfg.item_counts_path)

    if model == "ease":
        from anilist_rec.ease import ease_artifact_path, ease_scorer

        score_fn = ease_scorer(sp.load_npz(ease_artifact_path(cfg)))
    else:
        from anilist_rec.als import als_artifact_path, als_scorer

        artifact = np.load(als_artifact_path(cfg))
        score_fn = als_scorer(
            artifact["item_factors"],
            float(artifact["regularization"]),
            float(artifact["alpha"]),
        )
    return Recommender(score_fn, item_ids, franchise, item_counts, catalogue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("--model", choices=["bm25", "ease", "als"], default="bm25")
    parser.add_argument("--dial", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = Config(data_dir=args.data_dir, seed=args.seed)
    if args.model == "bm25":
        rec = load_baseline_recommender(cfg)
    else:
        rec = load_candidate_recommender(cfg, args.model)
    print(f"{args.model} artifacts ready in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    recs, fold = rec.recommend(args.username, dial=args.dial, limit=args.limit)
    print(
        f"{args.username}: {fold.n_entries} entries, {len(fold.fold_idx)} folded in, "
        f"{len(fold.plan_idx)} planning, {fold.n_unmapped} dropped outside corpus "
        f"({fold.n_unmapped / max(fold.n_entries, 1):.1%}) — "
        f"{time.perf_counter() - t0:.1f}s live"
    )
    for i, r in enumerate(recs, 1):
        print(f"{i:3d}. {r.title or '?':60s} anilist:{r.anilist_id:<7d} score {r.score:.3f}")


if __name__ == "__main__":
    main()
