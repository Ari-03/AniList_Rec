# Baseline bar — full-scale offline eval

Generated 2026-08-14 18:51 UTC by `uv run baseline`
([Ari-03/AniList_Rec#14](https://github.com/Ari-03/AniList_Rec/issues/14)).
Protocol: SPEC §5 — held-out users, per-user temporal 80/20, full-catalogue
ranking through the serving pipeline (franchise filter on), dial off, test users.

## Test-set metrics (dial off)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| item-item BM25 | 0.1300 [0.1263, 0.1335] | 0.112 | 0.323 | 0.0045 | +0.7 | +2.9 | 6.3% | 9971 |
| MostPopular | 0.1983 [0.1947, 0.2021] | 0.124 | 0.291 | 0.0064 | +3.0 | +7.5 | 0.9% | 9971 |

A candidate architecture ships only if it beats **both** models on NDCG@10
without MostPopular's degenerate coverage (SPEC §4). Pop lift = mean popularity
percentile of the top-10 minus the median of the user's profile (guardrail, not
optimized); niche = bottom quartile of users by profile popularity. regret@10 =
fraction of the top-10 the user actually dropped or scored low.

## Run provenance

| | |
|---|---|
| training users | 1,023,799 (cap: uncapped) |
| training interactions (weight > 0) | 151,513,559 |
| item universe | 13,369 |
| holdout | 10,000 test + 10,000 validation users |
| seed | 42 |
| BM25 | K=200, K1=1.2, B=0.75 |
