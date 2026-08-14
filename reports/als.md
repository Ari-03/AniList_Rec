# ALS candidate — offline eval

Generated 2026-08-14 09:55 UTC by `uv run als`
([Ari-03/AniList_Rec#17](https://github.com/Ari-03/AniList_Rec/issues/17)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Signal encoding (SPEC §1 → Hu/Koren/Volinsky confidences)

Signed signal weights feed `implicit` directly as confidences (scaled by
alpha): strong positives 2.0, standard 1.0, CURRENT 0.5-1.0 by progress,
PAUSED 0.25. Negative weights (DROPPED -0.25..-1.0 with confidence inverse
to progress, low-scored completions -0.25) are implicit's documented
negative-feedback path: high confidence on **zero** preference — the HKV
treatment of DROPPED the shortlist called for. 1,024,388
training users, 163,272,694 signals (11,759,135 negative);
15 iterations per fit.

Serve-time fold-in is the per-user HKV solve (`recalculate_user`
semantics) against the exported item factors — no learned user embeddings.

## Config sweep (validation users)

Swept across two runs (alpha descending after the first run showed lower
alpha winning); all rows measured on the same validation users:

| factors | reg | alpha | val NDCG@10 [95% CI] | fit walltime |
|---|---|---|---|---|
| 128 | 0.01 | 1 | 0.0896 [0.0869, 0.0923] | 493s |
| 128 | 0.01 | 3 | 0.0839 [0.0814, 0.0866] | 499s |
| 128 | 0.01 | 10 | 0.0757 [0.0733, 0.0782] | 497s |
| 128 | 0.01 | 40 | 0.0658 [0.0636, 0.0681] | 507s |
| 256 | 0.01 | 40 | 0.0695 [0.0672, 0.0719] | 841s |

The curve rises monotonically as alpha falls and flattens near alpha=1 —
the §1 weights are already confidence-shaped; inflating them distorts the
solve. Extra factors (256) do not help.

## Test-set metrics (dial off, f=128, reg=0.01, alpha=1)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| ALS (f=128, reg=0.01, alpha=1) | 0.0906 [0.0878, 0.0932] | 0.080 | 0.246 | 0.0023 | +1.1 | +3.9 | 7.7% | 9971 |

Item-factor artifact: `als_item_factors_seed42.npz`, 6.0 MB.
Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md).

## Verdict

**ALS does not clear the bar** (0.0906 vs 0.1300/0.1983) and the alpha curve
has flattened, so this is the model class, not the config: with only 13,369
items, full-rank item-item models (EASE at 0.2247, BM25 at 0.1300) out-rank
a 128-factor bottleneck. ALS does post the best *guardrails* of every model
measured — coverage@10 7.7% (vs EASE 2.0%), regret@10 0.0023, niche pop lift
+3.9 — a diversity-heavy ranker that loses on precision. Verdict for the
head-to-head is #19's call; on NDCG@10 it is out of the race.
