"""Full-catalogue eval harness (SPEC §5).

Ranking runs through the serving pipeline: never-recommend filter (own lists
incl. PLANNING), franchise collapse to entry points, full catalogue — no
sampled negatives. Targets are window positives; DROPPED / low-scored
completions feed the regret@10 diagnostic instead.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.franchise import FranchiseIndex
from anilist_rec.models import ScoreFn
from anilist_rec.signals import GAIN, NEG, PLAN, STD, STRONG


@dataclass(frozen=True)
class EvalUser:
    """One held-out user in item-index space."""

    fold_idx: list[int]  # fold-in items with positive weight (the model input)
    fold_w: list[float]
    watched: list[int]  # every fold-in item, weightless ones included
    plan_idx: list[int]
    targets: dict[int, float]  # window positives → graded gain
    negatives: list[int]  # window NEG items (regret diagnostic)


def user_arrays(rows: list[tuple[int, int, float, bool]], item_pos: dict[int, int]) -> EvalUser:
    """rows: (anime_id, kind, weight, fold) for one user; ids outside the corpus drop."""
    fold_idx, fold_w, watched, plan_idx, negatives = [], [], [], [], []
    targets: dict[int, float] = {}
    for anime_id, kind, weight, fold in rows:
        idx = item_pos.get(anime_id)
        if idx is None:
            continue
        if kind == PLAN:
            plan_idx.append(idx)
        elif fold:
            watched.append(idx)
            if weight > 0:
                fold_idx.append(idx)
                fold_w.append(weight)
        elif kind in (STD, STRONG):
            targets[idx] = max(targets.get(idx, 0.0), GAIN[kind])
        elif kind == NEG:
            negatives.append(idx)
    return EvalUser(fold_idx, fold_w, watched, plan_idx, targets, negatives)


def eval_users(holdout: pl.DataFrame, role: str, item_pos: dict[int, int]) -> list[EvalUser]:
    subset = holdout.filter(pl.col("role") == role)
    return [
        user_arrays(
            list(
                zip(group["anime_id"], group["kind"], group["weight"], group["fold"], strict=True)
            ),
            item_pos,
        )
        for _key, group in subset.group_by("user_id", maintain_order=True)
    ]


def candidate_mask(
    franchise: FranchiseIndex, watched: list[int], plan_idx: list[int]
) -> np.ndarray:
    """Serving filter: entry points of untouched franchises, minus the user's own lists."""
    mask = franchise.is_entry.copy()
    if watched:
        for cluster in np.unique(franchise.item_franchise[watched]):
            mask[franchise.item_franchise == cluster] = False
    mask[watched] = False
    mask[plan_idx] = False
    return mask


def collapse_targets(
    franchise: FranchiseIndex, targets: dict[int, float], watched: list[int]
) -> dict[int, float]:
    """Drop continuation targets; map franchise targets to their entry point."""
    touched = set(franchise.item_franchise[watched].tolist()) if watched else set()
    out: dict[int, float] = {}
    for idx, gain in targets.items():
        cluster = int(franchise.item_franchise[idx])
        if cluster in touched:
            continue
        entry = franchise.entry_of_franchise.get(cluster, idx)
        out[entry] = max(out.get(entry, 0.0), gain)
    return out


def popularity_percentile(item_counts: np.ndarray) -> np.ndarray:
    return 100.0 * item_counts.argsort().argsort() / (len(item_counts) - 1)


@dataclass(frozen=True)
class EvalResult:
    """Per-user metric arrays (users without post-collapse targets are skipped)."""

    ndcg10: np.ndarray
    rec10: np.ndarray
    rec50: np.ndarray
    regret10: np.ndarray
    prof_pct: np.ndarray  # median popularity percentile of the user's fold-in
    top_pct: np.ndarray  # mean popularity percentile of the user's top-10
    coverage10: float  # fraction of the catalogue any user saw in a top-10


def evaluate(
    users: list[EvalUser],
    score_fn: ScoreFn,
    franchise: FranchiseIndex,
    item_counts: np.ndarray,
    chunk: int = 1000,
) -> EvalResult:
    """Rank the full catalogue for each user and score the top-k against targets."""
    n_items = len(item_counts)
    pop_pct = popularity_percentile(item_counts)
    ndcg10, rec10, rec50, regret10, prof_pct, top_pct = [], [], [], [], [], []
    seen10 = np.zeros(n_items, dtype=bool)

    for lo in range(0, len(users), chunk):
        batch = users[lo : lo + chunk]
        rows, cols, vals = [], [], []
        for r, user in enumerate(batch):
            rows += [r] * len(user.fold_idx)
            cols += user.fold_idx
            vals += user.fold_w
        fold_csr = sp.csr_matrix((vals, (rows, cols)), shape=(len(batch), n_items), dtype="float32")
        scores = score_fn(fold_csr)

        for r, user in enumerate(batch):
            s = scores[r].copy()
            s[~candidate_mask(franchise, user.watched, user.plan_idx)] = -np.inf
            top50 = np.argsort(-s)[:50]
            top10 = top50[:10]
            seen10[top10] = True

            targets = collapse_targets(franchise, user.targets, user.watched)
            if not targets:
                continue
            gains = np.array([targets.get(int(i), 0.0) for i in top10])
            dcg = (gains / np.log2(np.arange(2, 2 + len(gains)))).sum()
            ideal = np.sort(list(targets.values()))[::-1][:10]
            idcg = (ideal / np.log2(np.arange(2, 2 + len(ideal)))).sum()
            ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
            rec10.append(len(set(top10.tolist()) & targets.keys()) / len(targets))
            rec50.append(len(set(top50.tolist()) & targets.keys()) / len(targets))
            regret10.append(len(set(top10.tolist()) & set(user.negatives)) / 10.0)
            prof_pct.append(np.median(pop_pct[user.fold_idx]) if user.fold_idx else 50.0)
            top_pct.append(pop_pct[top10].mean())

    return EvalResult(
        ndcg10=np.array(ndcg10),
        rec10=np.array(rec10),
        rec50=np.array(rec50),
        regret10=np.array(regret10),
        prof_pct=np.array(prof_pct),
        top_pct=np.array(top_pct),
        coverage10=float(seen10.mean()),
    )


def bootstrap_ci(
    arr: np.ndarray, seed: int, n_boot: int = 1000, levels: tuple[float, float] = (2.5, 97.5)
) -> tuple[float, float]:
    """Bootstrap CI of the mean, resampling users."""
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, levels)
    return float(lo), float(hi)


def summarize(result: EvalResult, seed: int) -> dict[str, float]:
    """Scalar metrics for the report; niche = bottom-quartile profile popularity."""
    ci_lo, ci_hi = bootstrap_ci(result.ndcg10, seed)
    niche = result.prof_pct <= np.quantile(result.prof_pct, 0.25)
    return {
        "ndcg10": float(result.ndcg10.mean()),
        "ndcg10_ci_lo": ci_lo,
        "ndcg10_ci_hi": ci_hi,
        "recall10": float(result.rec10.mean()),
        "recall50": float(result.rec50.mean()),
        "pop_lift": float((result.top_pct - result.prof_pct).mean()),
        "pop_lift_niche": float((result.top_pct[niche] - result.prof_pct[niche]).mean()),
        "coverage10": result.coverage10,
        "regret10": float(result.regret10.mean()),
        "n_users": len(result.ndcg10),
    }
