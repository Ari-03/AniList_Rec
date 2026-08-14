# SASRec candidate — offline eval

Generated 2026-08-14 18:26 UTC by `uv run sasrec`
([Ari-03/AniList_Rec#18](https://github.com/Ari-03/AniList_Rec/issues/18)).
Run executed on molab (RTX Pro 6000) via
[notebooks/sasrec_molab.py](../notebooks/sasrec_molab.py); the device line
below is corrected from the generated output, which predated the fix that
reports the actual training device.
Protocol: SPEC §5 — held-out users, temporal 80/20, full-catalogue ranking
through the serving pipeline (franchise filter on), dial off.

## Timestamp sanity check (acceptance criterion)

Measured over 924,267 users with ≥10 positives:
median 88% of a user's rows carry distinct timestamps
(p10 72%); only 1.0% of users
look bulk-imported (one edit date covering >50% of rows). Per-user Spearman between
edit order and anime premiere year (19,983 sampled users):
mean +0.91, median +0.96,
positive for 99.1%. **Verdict: list-edit order is a real
temporal signal — positional embeddings stay on** (the documented fallback,
`--set-encoder`, was not needed).

## Model + training

SASRec (sequence model): d=64, 2 blocks, 1 head(s),
dropout 0.2, maxlen 200, tied item embeddings. §1 mapping:
input embeddings scaled by the entry's positive confidence (0.25-2.0), loss
weighted by target confidence. Sampled softmax with 1024 shared
uniform negatives, Adam lr=0.001, batch 256 users.
1,023,799 training users, 105,426,193 events
(most recent 201 per user). Early stop on validation NDCG@10;
best epoch 3, total training walltime
158s on cuda (molab RTX Pro 6000).

## Epoch curve (validation users)

| epoch | train loss | val NDCG@10 [95% CI] | walltime |
|---|---|---|---|
| 1 | 1.5159 | 0.3354 [0.3313, 0.3397] | 32s |
| 2 | 1.0547 | 0.3622 [0.3581, 0.3665] | 31s |
| 3 | 1.0053 | 0.3707 [0.3663, 0.3752] | 31s |
| 4 | 0.9840 | 0.3629 [0.3586, 0.3673] | 31s |

## Order-shuffle ablation (1994 val users)

Same fold-in items, per-user shuffled order: NDCG@10
0.3754 ordered vs 0.2141 shuffled —
how much of the score sequence order actually carries.

## Test-set metrics (dial off, best epoch)

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | regret@10 | pop lift | pop lift (niche) | coverage@10 | users |
|---|---|---|---|---|---|---|---|---|
| SASRec (d=64, 2 blocks) | 0.3671 [0.3627, 0.3715] | 0.255 | 0.535 | 0.0229 | -0.1 | +2.7 | 12.8% | 9971 |

Artifact: `sasrec_seed42.npz` (full state dict, npz),
3.4 MB. Two-sided bar (SPEC §4): beat item-item BM25
**0.1300** and MostPopular **0.1983** on NDCG@10 without degenerate
coverage — see [baseline_bar.md](baseline_bar.md).
