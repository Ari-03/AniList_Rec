# EASE candidate — offline eval

Generated 2026-08-14 10:25 UTC by `uv run ease`
([Ari-03/AniList_Rec#16](https://github.com/Ari-03/AniList_Rec/issues/16)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Signal encoding (SPEC §1 → X)

X is the signed confidence matrix: strong positives 2.0, standard 1.0,
CURRENT 0.5-1.0 by progress, PAUSED 0.25, low-scored completions -0.25,
DROPPED -0.25..-1.0 with confidence inverse to progress (dropped early beats
dropped late). PLANNING is excluded. 1,024,388 training users,
163,272,694 signals of which 11,759,135 negative.
Gram-normalization mitigation: ON.

Encoding ablation at λ=256 (validation NDCG@10): signed X
0.2011 vs positive-only X 0.1959.
**Shipped encoding: signed** (winner beyond noise, else signed —
§1's negatives are kept unless they measurably cost accuracy).

## λ sweep (validation users, dense B)

| λ | val NDCG@10 [95% CI] |
|---|---|
| 0.25 | 0.1221 [0.1192, 0.1249] |
| 1 | 0.1373 [0.1341, 0.1403] |
| 4 | 0.1580 [0.1546, 0.1613] |
| 16 | 0.1824 [0.1788, 0.1860] |
| 64 | 0.2008 [0.1971, 0.2049] |
| 256 | 0.2011 [0.1974, 0.2050] |
| 1024 | 0.1887 [0.1852, 0.1927] |

## Sparsification (validation users, λ=256)

Top-k by |value| per column; dense B is 715 MB.

| k | val NDCG@10 | delta vs dense |
|---|---|---|
| 50 | 0.1393 | +0.0618 |
| 100 | 0.1581 | +0.0429 |
| 200 | 0.1812 | +0.0199 |
| 400 | 0.1925 | +0.0086 |
| 800 | 0.2011 | -0.0001 |

Shipped: top-800 (largest loss tolerated: 0.001).
Artifact: `ease_B_seed42.npz`, 51.1 MB.
Full fit + sweep walltime: 1393s.

## Test-set metrics (dial off, sparse artifact)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| EASE (λ=256, top-800) | 0.2008 [0.1968, 0.2047] | 0.139 | 0.362 | 0.0040 | +2.2 | +5.0 | 7.7% | 9971 |

Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md). Niche popularity lift is EASE's known
risk; Gram-normalization is the documented cheap mitigation if it's bad.
