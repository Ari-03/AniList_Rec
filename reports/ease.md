# EASE candidate — offline eval

Generated 2026-08-15 02:58 UTC by `uv run ease`
([Ari-03/AniList_Rec#16](https://github.com/Ari-03/AniList_Rec/issues/16)).
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Signal encoding (SPEC §1 → X)

X is the signed confidence matrix: strong positives 2.0, standard 1.0,
CURRENT 0.5-1.0 by progress, PAUSED 0.25, low-scored completions -0.25,
DROPPED -0.25..-1.0 with confidence inverse to progress (dropped early beats
dropped late). PLANNING is excluded. 1,024,388 training users,
163,272,694 signals of which 11,759,135 negative.
Gram-normalization mitigation: off.

Encoding ablation at λ=6.4e+07 (validation NDCG@10): signed X
0.2204 vs positive-only X 0.2193.
**Shipped encoding: signed** (winner beyond noise, else signed —
§1's negatives are kept unless they measurably cost accuracy).

## λ sweep (validation users, dense B)

| λ | val NDCG@10 [95% CI] |
|---|---|
| 50000 | 0.1283 [0.1255, 0.1312] |
| 250000 | 0.1391 [0.1362, 0.1420] |
| 1e+06 | 0.1558 [0.1526, 0.1590] |
| 4e+06 | 0.1808 [0.1774, 0.1843] |
| 1.6e+07 | 0.2101 [0.2064, 0.2139] |
| 6.4e+07 | 0.2204 [0.2165, 0.2243] |
| 2.56e+08 | 0.2193 [0.2154, 0.2232] |

## Sparsification (validation users, λ=6.4e+07)

Top-k by |value| per column; dense B is 715 MB.

| k | val NDCG@10 | delta vs dense |
|---|---|---|
| 50 | 0.2121 | +0.0083 |
| 100 | 0.2199 | +0.0005 |
| 200 | 0.2244 | -0.0040 |
| 400 | 0.2239 | -0.0035 |
| 800 | 0.2226 | -0.0023 |

Shipped: top-200 (largest loss tolerated: 0.001).
Artifact: `ease_B_seed42.npz`, 13.8 MB.
Full fit + sweep walltime: 1393s.

## Test-set metrics (dial off, sparse artifact)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| EASE (λ=6.4e+07, top-200) | 0.2247 [0.2208, 0.2286] | 0.139 | 0.332 | 0.0057 | +2.9 | +7.2 | 2.0% | 9971 |

Two-sided bar (SPEC §4): beat item-item BM25 **0.1300** and MostPopular
**0.1983** on NDCG@10 without degenerate coverage — see
[baseline_bar.md](baseline_bar.md). Niche popularity lift is EASE's known
risk; Gram-normalization is the documented cheap mitigation if it's bad.
