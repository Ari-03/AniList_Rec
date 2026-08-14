"""ALS candidate (SPEC §4 candidate 2, issue #17).

    uv run als

ALS via `implicit`, Hu/Koren/Volinsky confidence weighting. The §1 mapping
maps directly: signed signal weights become confidences, and implicit treats
negative values as high confidence on *zero* preference — exactly DROPPED's
"explicit negative" semantics. Serve-time fold-in is `recalculate_user`
(one per-user least-squares solve against the item factors), so users absent
from training serve natively. Config swept on validation; test eval per §5.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.config import Config
from anilist_rec.evaluate import eval_users, evaluate, summarize
from anilist_rec.franchise import build_franchise_index
from anilist_rec.matrix import build_training_matrix, item_index, item_positions
from anilist_rec.models import ScoreFn
from anilist_rec.report import bar_table, write_report
from anilist_rec.signals import build_signals
from anilist_rec.split import build_holdout

# (factors, regularization, alpha) — a small validation sweep; ALS fits are
# the expensive stage, so the grid stays deliberately narrow.
CONFIG_SWEEP = [(128, 0.01, 40.0), (128, 0.01, 10.0), (256, 0.01, 40.0)]
ITERATIONS = 15


def als_artifact_path(cfg: Config) -> Path:
    return cfg.derived_dir / f"als_item_factors_seed{cfg.seed}.npz"


def fit_als(
    x_signed: sp.csr_matrix, factors: int, regularization: float, alpha: float, seed: int
):
    from implicit.cpu.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=ITERATIONS,
        calculate_training_loss=False,
        random_state=seed,
        num_threads=0,
    )
    model.fit(x_signed, show_progress=False)
    return model


def foldin_user_factor(
    fold_row: np.ndarray,
    item_factors: np.ndarray,
    regularization: float,
    alpha: float,
    yty: np.ndarray | None = None,
) -> np.ndarray:
    """The HKV fold-in solve: Xu = (Y'CuY + λI)^-1 Y'CuPu for one raw list.

    Mirrors implicit's recalculate_user but stands alone so the export
    container can serve from the item-factor artifact without implicit.
    """
    if yty is None:
        yty = item_factors.T @ item_factors
    a = yty + regularization * np.eye(item_factors.shape[1], dtype=item_factors.dtype)
    b = np.zeros(item_factors.shape[1], dtype=item_factors.dtype)
    idx = np.nonzero(fold_row)[0]
    for i in idx:
        confidence = alpha * fold_row[i]
        y = item_factors[i]
        if confidence > 0:
            b += confidence * y
        else:
            confidence = -confidence
        a += (confidence - 1.0) * np.outer(y, y)
    return np.linalg.solve(a, b)


def als_scorer(item_factors: np.ndarray, regularization: float, alpha: float) -> ScoreFn:
    """Fold-in scorer: solve each user's factor from their raw list, then Y·x."""

    yty = item_factors.T @ item_factors

    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        out = np.zeros((fold_csr.shape[0], item_factors.shape[0]), dtype="float32")
        dense = fold_csr.toarray()
        for r in range(dense.shape[0]):
            xu = foldin_user_factor(dense[r], item_factors, regularization, alpha, yty=yty)
            out[r] = item_factors @ xu
        return out

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("reports/als.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help='override the sweep: "factors,reg,alpha;factors,reg,alpha;..."',
    )
    args = parser.parse_args()
    sweep_configs = CONFIG_SWEEP
    if args.configs:
        sweep_configs = [
            (int(f), float(r), float(a))
            for f, r, a in (c.split(",") for c in args.configs.split(";"))
        ]
    cfg = Config(data_dir=args.data_dir, seed=args.seed, train_user_cap=args.cap)

    t0 = time.perf_counter()

    def stage(name: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {name}", flush=True)

    stage("signal table + holdout")
    signals = build_signals(cfg)
    holdout = build_holdout(cfg, signals)
    item_ids = item_index(signals)
    item_pos = item_positions(item_ids)

    stage("signed training matrix")
    x_signed, item_counts = build_training_matrix(
        cfg, signals, item_ids, set(holdout["user_id"].unique()), signed=True
    )
    n_neg = int((x_signed.data < 0).sum())
    print(f"  {x_signed.shape[0]:,} users, {x_signed.nnz:,} signals ({n_neg:,} negative)")

    stage("franchise index")
    franchise = build_franchise_index(pl.read_parquet(cfg.crosswalk_path), item_ids)

    stage(f"config sweep on validation ({sweep_configs})")
    val_users = eval_users(holdout, "val", item_pos)
    sweep: dict[tuple, dict] = {}
    fit_times: dict[tuple, float] = {}
    best_model, best_key = None, None
    for key in sweep_configs:
        factors, reg, alpha = key
        fit_start = time.perf_counter()
        model = fit_als(x_signed, factors, reg, alpha, cfg.seed)
        fit_times[key] = time.perf_counter() - fit_start
        scorer = als_scorer(model.item_factors, reg, alpha)
        s = summarize(evaluate(val_users, scorer, franchise, item_counts), cfg.seed)
        sweep[key] = s
        print(
            f"  f={factors} reg={reg} alpha={alpha:g}: val NDCG@10 {s['ndcg10']:.4f}"
            f" (fit {fit_times[key]:.0f}s)"
        )
        if best_key is None or s["ndcg10"] > sweep[best_key]["ndcg10"]:
            best_model, best_key = model, key
    factors, reg, alpha = best_key

    stage(f"test eval (f={factors}, reg={reg}, alpha={alpha:g})")
    test_users = eval_users(holdout, "test", item_pos)
    scorer = als_scorer(best_model.item_factors, reg, alpha)
    test_summary = summarize(evaluate(test_users, scorer, franchise, item_counts), cfg.seed)

    stage("artifact + report")
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        als_artifact_path(cfg),
        item_factors=best_model.item_factors,
        regularization=np.float64(reg),
        alpha=np.float64(alpha),
    )
    artifact_mb = als_artifact_path(cfg).stat().st_size / 1e6

    write_report(args.report, render_als_report(
        cfg,
        run={
            "n_train_users": x_signed.shape[0],
            "nnz": x_signed.nnz,
            "n_neg": n_neg,
            "best": best_key,
            "artifact_mb": artifact_mb,
            "fit_times": fit_times,
        },
        sweep=sweep,
        test_summary=test_summary,
    ))
    label = f"ALS (f={factors}, reg={reg}, alpha={alpha:g})"
    print(f"\n{bar_table({label: test_summary})}\n\nwrote {args.report}")


def render_als_report(cfg, run, sweep, test_summary) -> str:
    from datetime import UTC, datetime

    factors, reg, alpha = run["best"]
    sweep_rows = "\n".join(
        f"| {f} | {r} | {a:g} | {s['ndcg10']:.4f} [{s['ndcg10_ci_lo']:.4f}, "
        f"{s['ndcg10_ci_hi']:.4f}] | {run['fit_times'][(f, r, a)]:.0f}s |"
        for (f, r, a), s in sweep.items()
    )
    return f"""# ALS candidate — offline eval

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by `uv run als`
([Ari-03/AniList_Rec#17](https://github.com/Ari-03/AniList_Rec/issues/17)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Signal encoding (SPEC §1 → Hu/Koren/Volinsky confidences)

Signed signal weights feed `implicit` directly as confidences (scaled by
alpha): strong positives 2.0, standard 1.0, CURRENT 0.5-1.0 by progress,
PAUSED 0.25. Negative weights (DROPPED -0.25..-1.0 with confidence inverse
to progress, low-scored completions -0.25) are implicit's documented
negative-feedback path: high confidence on **zero** preference — the HKV
treatment of DROPPED the shortlist called for. {run["n_train_users"]:,}
training users, {run["nnz"]:,} signals ({run["n_neg"]:,} negative);
{ITERATIONS} iterations per fit.

Serve-time fold-in is the per-user HKV solve (`recalculate_user`
semantics) against the exported item factors — no learned user embeddings.

## Config sweep (validation users)

| factors | reg | alpha | val NDCG@10 [95% CI] | fit walltime |
|---|---|---|---|---|
{sweep_rows}

## Test-set metrics (dial off, f={factors}, reg={reg}, alpha={alpha:g})

{bar_table({f"ALS (f={factors}, reg={reg}, alpha={alpha:g})": test_summary})}

Item-factor artifact: `{als_artifact_path(cfg).name}`, {run["artifact_mb"]:.1f} MB.
Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md).
"""


if __name__ == "__main__":
    main()
