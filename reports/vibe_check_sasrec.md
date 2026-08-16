# Vibe check — SASRec (owner's pass)

Recorded 2026-08-16 from the owner's judgment of `notebooks/vibe_check.py`
run against their own AniList account (`Zackhacks`: 583 entries, 255 folded
in, 21.8% outside the corpus — the SPEC §3 time-capsule cost), top-20 per
dial setting through the real serving path. Judged as a 2022-era time capsule
(SPEC §3/§5): staleness is not model failure.

The owner gave a per-setting verdict rather than per-rec label tallies:

| dial setting | owner's verdict |
|---|---|
| off (= shipped default 0) | **great** |
| moderate novelty (0.05) | **great** |
| high novelty (0.2) | poor — "way too much random crap" |

## Reading

- The shipped configuration (dial 0) passes the human check, consistent with
  its offline profile: SASRec at dial-off is popularity-neutral (pop lift
  −0.1) with the best coverage of any candidate.
- The moderate setting (0.05) also passes, so the novelty knob has a usable
  range — despite costing most of the offline NDCG@10 (0.369 → 0.052 on
  validation), the dialed lists still read as taste-aware. This is the
  expected gap between NDCG-against-popular-rewatch-behaviour and perceived
  quality ([reports/eval.md](eval.md), Shipped default section).
- High novelty (0.2) fails the human check: past the model's preference
  signal, the re-rank surfaces noise. Callers should treat the useful dial
  range as roughly 0–0.1; the 0–1 contract knob is unchanged, but the future
  web app should map its slider onto the low end rather than the full range.

Verdict feeds [#19](https://github.com/Ari-03/AniList_Rec/issues/19); the
winner declaration and shipped default in [eval.md](eval.md) stand.
