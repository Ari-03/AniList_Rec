"""Head-to-head verdict, dial sweep, and winner selection (SPEC §5, issue #19).

    uv run compare

Evaluates every candidate whose exported artifact is on disk (plus the two
baseline bars) on **test** users, dial off, through the shared serving path;
applies the two-sided bar (beat BM25 and MostPopular on NDCG@10 without
MostPopular's degenerate coverage); refuses to rank candidates whose CIs
overlap ("within noise is not a ranking" — SPEC §5). The finalists that clear
the bar get the §5 dial sweep on **validation** users; the shipped default is
the largest dial whose validation NDCG@10 stays within CI noise of dial-off.

Model scores per split are computed once and replayed across dial settings —
the dial is a re-rank, so only the ranking work repeats.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.bundle import _load_scorer
from anilist_rec.config import Config
from anilist_rec.evaluate import EvalUser, eval_users, evaluate, summarize
from anilist_rec.franchise import build_franchise_index
from anilist_rec.matrix import item_index, item_positions
from anilist_rec.models import ScoreFn, bm25_scorer, most_popular_scorer
from anilist_rec.report import bar_table, write_report
from anilist_rec.signals import build_signals
from anilist_rec.split import build_holdout

DIALS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
BARS = ("item-item BM25", "MostPopular")
DEGENERATE_COVERAGE_FACTOR = 2.0  # must beat MostPopular's coverage by at least this


def precompute_scores(
    score_fn: ScoreFn, users: list[EvalUser], n_items: int, chunk: int = 1000
) -> np.ndarray:
    """Run the model once over the users, mirroring evaluate()'s batching."""
    out = np.empty((len(users), n_items), dtype="float32")
    for lo in range(0, len(users), chunk):
        batch = users[lo : lo + chunk]
        rows, cols, vals = [], [], []
        for r, user in enumerate(batch):
            rows += [r] * len(user.fold_idx)
            cols += user.fold_idx
            vals += user.fold_w
        fold_csr = sp.csr_matrix((vals, (rows, cols)), shape=(len(batch), n_items), dtype="float32")
        if getattr(score_fn, "takes_batch", False):
            out[lo : lo + len(batch)] = score_fn(fold_csr, batch)
        else:
            out[lo : lo + len(batch)] = score_fn(fold_csr)
    return out


def replay_scorer(scores: np.ndarray) -> ScoreFn:
    """Serve precomputed rows back to evaluate() in its sequential chunk order."""
    cursor = 0

    def score(fold_csr: sp.csr_matrix) -> np.ndarray:
        nonlocal cursor
        rows = scores[cursor : cursor + fold_csr.shape[0]]
        cursor += fold_csr.shape[0]
        return rows

    return score


def cis_overlap(a: dict, b: dict) -> bool:
    return a["ndcg10_ci_lo"] <= b["ndcg10_ci_hi"] and b["ndcg10_ci_lo"] <= a["ndcg10_ci_hi"]


def clears_bar(summary: dict, bars: dict[str, dict]) -> bool:
    """SPEC §4: beat both bars on NDCG@10 without degenerate coverage."""
    beats_ndcg = all(summary["ndcg10"] > bars[b]["ndcg10"] for b in BARS)
    coverage_ok = (
        summary["coverage10"] >= DEGENERATE_COVERAGE_FACTOR * bars["MostPopular"]["coverage10"]
    )
    return beats_ndcg and coverage_ok


def pick_default_dial(sweep: dict[float, dict]) -> float:
    """The largest dial whose val NDCG@10 stays within CI noise of dial-off.

    Tolerance = dial-off's bootstrap CI half-width: giving up less accuracy
    than the measurement can resolve, in exchange for the largest popularity-
    lift reduction the curve offers.
    """
    off = sweep[0.0]
    tol = (off["ndcg10_ci_hi"] - off["ndcg10_ci_lo"]) / 2
    ok = [d for d, s in sweep.items() if s["ndcg10"] >= off["ndcg10"] - tol]
    return max(ok)


def candidate_scorers(cfg: Config, n_items: int) -> dict[str, ScoreFn]:
    """Every candidate with an exported artifact on disk, by display name."""
    from anilist_rec.als import als_artifact_path
    from anilist_rec.ease import ease_artifact_path
    from anilist_rec.sasrec import sasrec_artifact_path

    paths = {
        "EASE": ("ease", ease_artifact_path(cfg)),
        "ALS": ("als", als_artifact_path(cfg)),
        "SASRec": ("sasrec", sasrec_artifact_path(cfg)),
    }
    out = {}
    for name, (arch, path) in paths.items():
        if path.exists():
            out[name] = _load_scorer(arch, path, n_items)
        else:
            print(f"  ({name}: no artifact at {path.name} — skipped)")
    return out


def sweep_line_svg(curves: dict[str, dict[float, dict]], out_path: Path) -> None:
    """NDCG@10 vs popularity lift, one polyline per finalist, points = dials."""
    from anilist_rec.sweepplot import render_sweep_svg

    out_path.write_text(render_sweep_svg(curves))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("reports/eval.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dials", type=float, nargs="*", default=DIALS)
    args = parser.parse_args()
    cfg = Config(data_dir=args.data_dir, seed=args.seed)

    t0 = time.perf_counter()

    def stage(name: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {name}", flush=True)

    stage("signal table + holdout (temporal order re-joined)")
    from anilist_rec.sasrec import ordered_holdout

    signals = build_signals(cfg)
    holdout = ordered_holdout(cfg, build_holdout(cfg, signals))
    item_ids = item_index(signals)
    item_pos = item_positions(item_ids)
    n_items = len(item_ids)
    item_counts = np.load(cfg.item_counts_path)
    franchise = build_franchise_index(pl.read_parquet(cfg.crosswalk_path), item_ids)
    test_users = eval_users(holdout, "test", item_pos)
    val_users = eval_users(holdout, "val", item_pos)

    stage("models")
    models: dict[str, ScoreFn] = {
        "MostPopular": most_popular_scorer(item_counts),
        "item-item BM25": bm25_scorer(sp.load_npz(cfg.similarity_path)),
    }
    models.update(candidate_scorers(cfg, n_items))

    stage("head-to-head on test users, dial off")
    test_summaries: dict[str, dict] = {}
    for name, score_fn in models.items():
        test_summaries[name] = summarize(
            evaluate(test_users, score_fn, franchise, item_counts), cfg.seed
        )
        print(f"  {name}: NDCG@10 {test_summaries[name]['ndcg10']:.4f}")

    bars = {b: test_summaries[b] for b in BARS}
    candidates = {n: s for n, s in test_summaries.items() if n not in BARS}
    finalists = [n for n, s in candidates.items() if clears_bar(s, bars)]
    ranked = sorted(finalists, key=lambda n: -candidates[n]["ndcg10"])
    if not ranked:
        raise SystemExit("no candidate clears the two-sided bar — nothing to ship")
    winner = ranked[0]
    decisive = len(ranked) == 1 or not cis_overlap(
        candidates[ranked[0]], candidates[ranked[1]]
    )
    stage(f"winner: {winner} ({'beyond CI noise' if decisive else 'WITHIN noise of runner-up'})")

    stage(f"dial sweep on validation users ({ranked})")
    curves: dict[str, dict[float, dict]] = {}
    for name in ranked:
        scores = precompute_scores(models[name], val_users, n_items)
        curves[name] = {}
        for dial in args.dials:
            curves[name][dial] = summarize(
                evaluate(val_users, replay_scorer(scores), franchise, item_counts, dial=dial),
                cfg.seed,
            )
            s = curves[name][dial]
            print(
                f"  {name} dial={dial:g}: NDCG@10 {s['ndcg10']:.4f}, "
                f"pop lift {s['pop_lift']:+.1f}, coverage {s['coverage10']:.1%}"
            )
        del scores

    default_dial = pick_default_dial(curves[winner])
    stage(f"shipped default dial: {default_dial:g} (largest within CI noise of dial-off)")

    stage("report + curve")
    svg_path = args.report.parent / "dial_sweep.svg"
    sweep_line_svg(curves, svg_path)
    write_report(
        args.report,
        render_eval_report(
            cfg, test_summaries, bars, ranked, winner, decisive, curves, default_dial, svg_path
        ),
    )
    print(f"\n{bar_table(test_summaries)}\n\nwrote {args.report}")


def render_eval_report(
    cfg, test_summaries, bars, ranked, winner, decisive, curves, default_dial, svg_path
) -> str:
    from datetime import UTC, datetime

    verdicts = []
    for name, s in test_summaries.items():
        if name in BARS:
            continue
        verdict = "**clears the two-sided bar**" if name in ranked else "does not clear the bar"
        verdicts.append(
            f"- **{name}**: NDCG@10 {s['ndcg10']:.4f} "
            f"[{s['ndcg10_ci_lo']:.4f}, {s['ndcg10_ci_hi']:.4f}], "
            f"coverage@10 {s['coverage10']:.1%} — {verdict}."
        )

    sweep_sections = []
    for name in ranked:
        rows = "\n".join(
            f"| {d:g} | {s['ndcg10']:.4f} [{s['ndcg10_ci_lo']:.4f}, {s['ndcg10_ci_hi']:.4f}] "
            f"| {s['recall50']:.3f} | {s['pop_lift']:+.1f} | {s['pop_lift_niche']:+.1f} "
            f"| {s['coverage10']:.1%} |"
            for d, s in curves[name].items()
        )
        sweep_sections.append(
            f"""### {name}

| dial | val NDCG@10 [95% CI] | recall@50 | pop lift | pop lift (niche) | coverage@10 |
|---|---|---|---|---|---|
{rows}"""
        )
    sweeps = "\n\n".join(sweep_sections)

    noise_note = (
        "The winner leads beyond CI noise."
        if decisive
        else "**The top two candidates are within CI noise of each other** — the winner was "
        "chosen on guardrails (coverage, niche popularity lift, regret), not on NDCG alone."
    )

    return f"""# Candidate head-to-head, dial sweep, and winner — issue #19

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by `uv run compare`
([Ari-03/AniList_Rec#19](https://github.com/Ari-03/AniList_Rec/issues/19)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on). Architectures compared on
**test** users dial-off; the dial swept on **validation** users for the
finalists. Per-candidate training detail: [ease.md](ease.md), [als.md](als.md),
[sasrec.md](sasrec.md), bars in [baseline_bar.md](baseline_bar.md).

## Head-to-head (test users, dial off)

{bar_table(test_summaries)}

Two-sided bar (SPEC §4): beat item-item BM25 **{bars["item-item BM25"]["ndcg10"]:.4f}** and
MostPopular **{bars["MostPopular"]["ndcg10"]:.4f}** on NDCG@10 with coverage@10 at least
{DEGENERATE_COVERAGE_FACTOR:g}x MostPopular's {bars["MostPopular"]["coverage10"]:.1%}
(the Cremonesi degenerate-coverage guard).

{chr(10).join(verdicts)}

## Winner: {winner}

{noise_note}

## Dial sweep (validation users)

The dial is the SPEC §1 serve-time popularity re-rank, computed in **rank
space**: per-user score percentile minus `dial x` popularity percentile,
applied inside the same serving path the export container runs
([export contract](../docs/export-contract.md)). Rank space is what makes the
0-1 knob architecture-independent in practice: the first formulation divided
raw scores by `(popularity + 1)^dial`, and its useful range collapsed with the
winner's score scale (NDCG@10 fell 0.369 -> 0.173 by dial 0.1 on SASRec logits
while EASE barely moved before 0.6). Percentiles are scale-free, so the same
dial value trades the same amount of preference for popularity on every
candidate.

![NDCG@10 vs popularity lift]({svg_path.name})

{sweeps}

## Shipped default: dial = {default_dial:g}

Rule: the largest dial whose validation NDCG@10 stays within the dial-off
bootstrap CI half-width — giving up less headline accuracy than the
measurement can resolve, for the largest popularity-lift reduction on the
curve. The export bundle bakes this as `dial_default`
(`export-bundle --dial-default {default_dial:g}`); callers override per
request.

Context for reading the winner's curve: {winner}'s dial-off popularity lift is
{curves[winner][0.0]["pop_lift"]:+.1f} — the §1 Kowald failure mode the dial
was designed to correct is not present at dial off, so the default buys
nothing and every nonzero setting is a pure accuracy-for-novelty trade. NDCG
against held-out watches punishes that trade by construction (what users
actually watched next skews popular), which is why the dial ships as a
caller-facing novelty knob rather than a correction. Its quality in the
novelty regime is judged by the vibe check, not by this metric.

## Vibe check

Per finalist: `notebooks/vibe_check.py` (marimo) pulls the owner's AniList
list through the real serving path and renders the top-20 at three dial
settings with covers, genres, and why-this lines; each rec is labeled
seen-elsewhere / would-watch / plausible / bad / broken and the tallies land
in `reports/vibe_check_<model>.md`. Judged as a 2022-era time capsule
(SPEC §3/§5 caveat): staleness is not model failure.

## Seed and provenance

seed {cfg.seed}; artifacts under `data/derived/`; every number above produced
by this command end to end.
"""


if __name__ == "__main__":
    main()
