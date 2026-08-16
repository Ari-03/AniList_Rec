# ALS candidate — offline eval

Generated 2026-08-15 03:34 UTC by `uv run als`
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

| factors | reg | alpha | val NDCG@10 [95% CI] | fit walltime |
|---|---|---|---|---|
| 128 | 0.01 | 40 | 0.0659 [0.0637, 0.0682] | 502s |
| 128 | 0.01 | 10 | 0.0758 [0.0733, 0.0782] | 509s |
| 256 | 0.01 | 40 | 0.0695 [0.0673, 0.0719] | 842s |

## Test-set metrics (dial off, f=128, reg=0.01, alpha=10)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| ALS (f=128, reg=0.01, alpha=10) | 0.0757 [0.0730, 0.0783] | 0.074 | 0.243 | 0.0018 | +0.5 | +2.0 | 12.1% | 9971 |

Item-factor artifact: `als_item_factors_seed42.npz`, 6.0 MB.
Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md).
