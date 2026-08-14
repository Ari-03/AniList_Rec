"""EASE candidate (SPEC §4 candidate 1, issue #16).

    uv run ease

Closed-form item-item linear model: B = (X'X + λI)^-1, column-normalized.
X encodes the §1 mapping as signed confidences (positives 0.25-2.0, DROPPED
and low-scored completions negative — the signed signal weights). λ is swept
on validation users; the dense B is sparsified top-k per column with the
accuracy delta measured on validation before the test eval.
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

# λ must scale with the corpus: at 1M training users the Gram diagonal is
# O(10^6), so sub-1000 λ barely regularizes (measured: flat 0.1237 val NDCG
# across 50..1000, then rising through 250k). Anchor low, then climb x4 from
# 250k until validation NDCG turns down (max L2_MAX_STEPS climb steps).
L2_ANCHORS = [50_000.0]
L2_CLIMB_START = 250_000.0
L2_CLIMB_FACTOR = 4.0
L2_MAX_STEPS = 8
TOPK_SWEEP = [50, 100, 200, 400, 800]
TOPK_MAX_NDCG_LOSS = 0.0010  # among k within this of the best, ship the smallest


def ease_artifact_path(cfg: Config) -> Path:
    return cfg.derived_dir / f"ease_B_seed{cfg.seed}.npz"


def fit_gram(x_signed: sp.csr_matrix) -> np.ndarray:
    """Dense G = X'X in float64 (13k² ≈ 1.4 GB; the one big allocation)."""
    return (x_signed.T @ x_signed).toarray().astype(np.float64)


def fit_ease(gram: np.ndarray, l2: float, gram_norm: bool = False) -> np.ndarray:
    """B with zero diagonal, float32. gram_norm applies the documented
    popularity mitigation: normalize G by item norms before inverting."""
    g = gram
    if gram_norm:
        scale = 1.0 / np.sqrt(np.diag(gram) + 1e-6)
        g = gram * scale[:, None] * scale[None, :]
    g = g.copy()
    diag = np.arange(g.shape[0])
    g[diag, diag] += l2
    p = np.linalg.inv(g)
    b = (p / (-np.diag(p))).astype(np.float32)  # B_ij = -P_ij / P_jj
    b[diag, diag] = 0.0
    return b


def sparsify_topk(b: np.ndarray, k: int) -> sp.csr_matrix:
    """Keep the top-k entries by |value| per column (the shipped artifact)."""
    n = b.shape[0]
    if k >= n:
        out = sp.csr_matrix(b)
        out.eliminate_zeros()
        return out
    keep_rows = np.argpartition(-np.abs(b), k, axis=0)[:k]  # k x n row indices per column
    cols = np.repeat(np.arange(n)[None, :], k, axis=0)
    out = sp.csr_matrix(
        (b[keep_rows.ravel(), cols.ravel()], (keep_rows.ravel(), cols.ravel())), shape=b.shape
    )
    out.eliminate_zeros()
    return out


def ease_scorer(b: np.ndarray | sp.spmatrix) -> ScoreFn:
    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        scores = fold_csr @ b
        return scores.toarray() if sp.issparse(scores) else np.asarray(scores)

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("reports/ease.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--gram-norm", action="store_true", help="popularity mitigation")
    args = parser.parse_args()
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

    stage("Gram matrix")
    gram = fit_gram(x_signed)
    fit_t0 = time.perf_counter()

    stage("λ sweep on validation (climb until the curve turns)")
    val_users = eval_users(holdout, "val", item_pos)
    val_by_l2: dict[float, dict] = {}
    b = None

    def try_l2(l2: float) -> float:
        nonlocal b
        b_l2 = fit_ease(gram, l2, gram_norm=args.gram_norm)
        s = summarize(evaluate(val_users, ease_scorer(b_l2), franchise, item_counts), cfg.seed)
        val_by_l2[l2] = s
        print(f"  λ={l2:g}: val NDCG@10 {s['ndcg10']:.4f}", flush=True)
        if max(val_by_l2, key=lambda k: val_by_l2[k]["ndcg10"]) == l2:
            b = b_l2  # keep only the best dense B in memory
        return s["ndcg10"]

    for l2 in L2_ANCHORS:
        try_l2(l2)
    l2, prev = L2_CLIMB_START, -1.0
    for _ in range(L2_MAX_STEPS):
        ndcg = try_l2(l2)
        if ndcg < prev:
            break
        l2, prev = l2 * L2_CLIMB_FACTOR, ndcg
    best_l2 = max(val_by_l2, key=lambda k: val_by_l2[k]["ndcg10"])
    dense_val = val_by_l2[best_l2]

    stage(f"positive-only X ablation at λ={best_l2:g} (val)")
    x_pos = x_signed.copy()
    x_pos.data = np.maximum(x_pos.data, 0)
    x_pos.eliminate_zeros()
    gram = None  # free the signed Gram before building the positive-only one
    b_pos = fit_ease(fit_gram(x_pos), best_l2, gram_norm=args.gram_norm)
    del x_pos
    pos_val = summarize(evaluate(val_users, ease_scorer(b_pos), franchise, item_counts), cfg.seed)
    print(
        f"  positive-only: val NDCG@10 {pos_val['ndcg10']:.4f}"
        f" (signed {dense_val['ndcg10']:.4f})"
    )
    # ship whichever X encoding wins on validation beyond noise; §1's negatives
    # are kept unless they measurably cost accuracy
    encoding = "signed"
    if pos_val["ndcg10"] > dense_val["ndcg10"] + 0.002:
        encoding, b, dense_val = "positive-only", b_pos, pos_val
    del b_pos
    fit_walltime = time.perf_counter() - fit_t0

    stage(f"sparsify sweep at λ={best_l2:g}, {encoding} X ({TOPK_SWEEP})")
    val_by_k: dict[int, dict] = {}
    for k in TOPK_SWEEP:
        b_k = sparsify_topk(b, k)
        s = summarize(evaluate(val_users, ease_scorer(b_k), franchise, item_counts), cfg.seed)
        val_by_k[k] = s
        print(f"  k={k}: val NDCG@10 {s['ndcg10']:.4f} (dense {dense_val['ndcg10']:.4f})")
    # sparsification can *beat* dense (truncation denoises the tail — measured),
    # so pick by val NDCG: smallest k within tolerance of the best sparse/dense
    peak = max(dense_val["ndcg10"], *(s["ndcg10"] for s in val_by_k.values()))
    ok = [k for k in TOPK_SWEEP if peak - val_by_k[k]["ndcg10"] <= TOPK_MAX_NDCG_LOSS]
    best_k = min(ok) if ok else max(val_by_k, key=lambda k: val_by_k[k]["ndcg10"])
    b_sparse = sparsify_topk(b, best_k)

    stage(f"test eval (λ={best_l2:g}, top-{best_k})")
    test_users = eval_users(holdout, "test", item_pos)
    test_result = evaluate(test_users, ease_scorer(b_sparse), franchise, item_counts)
    test_sparse = summarize(test_result, cfg.seed)

    stage("artifact + report")
    cfg.derived_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(ease_artifact_path(cfg), b_sparse)
    artifact_mb = ease_artifact_path(cfg).stat().st_size / 1e6
    dense_mb = b.nbytes / 1e6

    write_report(args.report, render_ease_report(
        cfg,
        run={
            "n_train_users": x_signed.shape[0],
            "nnz": x_signed.nnz,
            "n_neg": n_neg,
            "gram_norm": args.gram_norm,
            "fit_walltime_s": fit_walltime,
            "best_l2": best_l2,
            "best_k": best_k,
            "encoding": encoding,
            "pos_val_ndcg": pos_val["ndcg10"],
            "signed_val_ndcg": val_by_l2[best_l2]["ndcg10"],
            "artifact_mb": artifact_mb,
            "dense_mb": dense_mb,
        },
        val_by_l2=val_by_l2,
        val_by_k=val_by_k,
        dense_val=dense_val,
        test_sparse=test_sparse,
    ))
    label = f"EASE (λ={best_l2:g}, top-{best_k})"
    print(f"\n{bar_table({label: test_sparse})}\n\nwrote {args.report}")


def render_ease_report(cfg, run, val_by_l2, val_by_k, dense_val, test_sparse) -> str:
    from datetime import UTC, datetime

    l2_rows = "\n".join(
        f"| {l2:g} | {s['ndcg10']:.4f} [{s['ndcg10_ci_lo']:.4f}, {s['ndcg10_ci_hi']:.4f}] |"
        for l2, s in val_by_l2.items()
    )
    k_rows = "\n".join(
        f"| {k} | {s['ndcg10']:.4f} | {dense_val['ndcg10'] - s['ndcg10']:+.4f} |"
        for k, s in val_by_k.items()
    )
    return f"""# EASE candidate — offline eval

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by `uv run ease`
([Ari-03/AniList_Rec#16](https://github.com/Ari-03/AniList_Rec/issues/16)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Signal encoding (SPEC §1 → X)

X is the signed confidence matrix: strong positives 2.0, standard 1.0,
CURRENT 0.5-1.0 by progress, PAUSED 0.25, low-scored completions -0.25,
DROPPED -0.25..-1.0 with confidence inverse to progress (dropped early beats
dropped late). PLANNING is excluded. {run["n_train_users"]:,} training users,
{run["nnz"]:,} signals of which {run["n_neg"]:,} negative.
Gram-normalization mitigation: {"ON" if run["gram_norm"] else "off"}.

Encoding ablation at λ={run["best_l2"]:g} (validation NDCG@10): signed X
{run["signed_val_ndcg"]:.4f} vs positive-only X {run["pos_val_ndcg"]:.4f}.
**Shipped encoding: {run["encoding"]}** (winner beyond noise, else signed —
§1's negatives are kept unless they measurably cost accuracy).

## λ sweep (validation users, dense B)

| λ | val NDCG@10 [95% CI] |
|---|---|
{l2_rows}

## Sparsification (validation users, λ={run["best_l2"]:g})

Top-k by |value| per column; dense B is {run["dense_mb"]:.0f} MB.

| k | val NDCG@10 | delta vs dense |
|---|---|---|
{k_rows}

Shipped: top-{run["best_k"]} (largest loss tolerated: {TOPK_MAX_NDCG_LOSS}).
Artifact: `{ease_artifact_path(cfg).name}`, {run["artifact_mb"]:.1f} MB.
Full fit + sweep walltime: {run["fit_walltime_s"]:.0f}s.

## Test-set metrics (dial off, sparse artifact)

{bar_table({f"EASE (λ={run["best_l2"]:g}, top-{run["best_k"]})": test_sparse})}

Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md). Niche popularity lift is EASE's known
risk; Gram-normalization is the documented cheap mitigation if it's bad.
"""


if __name__ == "__main__":
    main()
