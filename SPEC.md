# Spec: Exportable AniList Anime Recommender (v1)

The buildable spec assembled from the [wayfinder map](https://github.com/Ari-03/AniList_Rec/issues/1).
Each section links the ticket that decided it; the ticket's resolution comment is the
authoritative detail, this document is the assembled whole.

**What gets built:** a recommendation model developed in Python + marimo, trained on public
MyAnimeList interaction data (local Mac + [molab](https://molab.marimo.io): 4 CPU cores, 32 GB
RAM, RTX Pro 6000 Blackwell), exported as a black-box Docker scoring service that takes any
public AniList username and returns ranked anime the user hasn't seen. The public web app that
consumes the export is a separate, later effort.

---

## 1. Product definition — what a "good recommendation" is

Decided in [Define a "good recommendation" (#5)](https://github.com/Ari-03/AniList_Rec/issues/5).

- **Novelty is a tunable blend.** Top recommendations mix well-matched popular titles the user
  missed with lower-profile gems, controlled by a **popularity dial** (serve-time popularity
  re-rank) exposed in the export. An undialed ranker converges on popular-heavy lists
  (Kowald EWAF'25: +0.74–0.87 popularity lift for niche-taste users), so the dial is mandatory.
- **Never recommend anything already on any of the user's lists** — including PLANNING.
- **Franchises collapse to entry points.** Using AniList's `relations` graph
  (SEQUEL/PREQUEL/SIDE_STORY/…), never recommend direct continuations of watched titles;
  recommend an unwatched franchise via its canonical entry point (S1, not the recap movie).

### Signal mapping (list entry → training signal)

| Signal | Weight |
|---|---|
| `COMPLETED` + normalized score ≥ 8/10, `REPEATING` / `repeat > 0`, favourited | strongest positive |
| `COMPLETED` (unscored or mid score) | standard positive |
| `CURRENT` | positive, confidence scaled by `progress/episodes` |
| `PAUSED` | near-neutral |
| `COMPLETED` + normalized score ≤ 4/10 | mild negative |
| `DROPPED` | explicit negative, confidence inverse to `progress` (dropped at ep 2 ≫ ep 20) |
| `PLANNING` | excluded from training **and** from recommendation output |

Scores are normalized per-user via AniList's server-side `score(format:)` conversion at serve
time; svanoo scores are already 1–10. `score == 0` / null means **unrated**, not zero.

---

## 2. Data

Decided in [Choose the training dataset (#6)](https://github.com/Ari-03/AniList_Rec/issues/6);
materialized in [Acquire the training data (#12)](https://github.com/Ari-03/AniList_Rec/issues/12).
Survey: [research/datasets.md](https://github.com/Ari-03/AniList_Rec/blob/research/datasets/research/datasets.md).

- **Interaction corpus: [svanoo/myanimelist-dataset](https://www.kaggle.com/datasets/svanoo/myanimelist-dataset)**
  (CC0). On disk: 223,812,614 rows → `data/interactions.parquet` (977 MB zstd), 1,049,512 users,
  13,369 anime. Columns: `user_id` (string username), `anime_id` (MAL id, int32), `favorite`,
  `score` (null = unrated; 129.2M scored), `status`, `progress`, `last_interaction_date`.
  In-scope signal families: **interactions + favourites** (3.5M favourites). Ruled out: review
  text/sub-scores, friend graph (serve-time users bring no graph edges).
- **Corpus cutoff: 2022-03-22.** The new-anime gap at serve time is ~4 years — see §3.
  svanoo has **no REPEATING status**; strong positives there are favourited or score ≥ 8.
  1.3% of rows have a null date — temporal-split code must handle them.
- **Item catalogue + AniList↔MAL crosswalk:
  [calebmwelsh/anilist-anime-dataset](https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset)**
  (CC0) → `data/crosswalk_anilist_mal.parquet` (20,407 AniList ids, 19,395 with MAL id; 22 MAL
  ids map to 2 AniList entries — dedupe at join time). Coverage against svanoo: 85.6% of titles,
  **98.8% interaction-weighted** — effectively lossless. Never assume `id == idMal`; it breaks on
  modern titles (Frieren: AniList 154587 / MAL 52991).
- **Acquisition is one command, no Kaggle credentials:**
  [`scripts/acquire_data.py`](scripts/acquire_data.py) (`uv run`, works on molab). Data files
  stay out of git.
- **Schema constraint:** `media_type` is a first-class column from day one (manga extensibility, §8).
- **Licensing guardrail:** raw user rows never leave the training environment; the export carries
  item representations, crosswalk, and metadata only.

---

## 3. Item universe and cold start

Decided in [Spec the new-title projection (#13)](https://github.com/Ari-03/AniList_Rec/issues/13):
**no new-title projection in v1 — the training corpus is the item universe.**

- **Output:** titles premiering after 2022-03-22 (3,136 titles, 15.4% of the catalogue) are never
  recommended.
- **Input:** serve-time list entries that don't map into the corpus are silently dropped before
  fold-in. Measured cost on the vibe-check account: 13.8% of non-PLANNING entries dropped; the
  CURRENT list is ~90% post-cutoff, so the model is blind to currently-airing viewing. Accepted.
- **Consequence:** the content-embedding sidecar proposed in the architecture shortlist is
  **cut from v1** — with no cold items and the manga bridge out of scope, it has no load-bearing
  job. It returns with the future effort that reintroduces post-cutoff titles (that effort must
  first patch the calebmwelsh scraper's hardcoded 2025 year ceiling).

---

## 4. Architecture candidates

Decided in [Choose the architecture shortlist (#8)](https://github.com/Ari-03/AniList_Rec/issues/8),
from the surveys on [#3](https://github.com/Ari-03/AniList_Rec/issues/3). All candidates must
serve **users absent from training** natively (fold-in from a raw list at serve time); learned
per-user embeddings are ruled out (NCF, ID-tower two-tower, LightFM).

Implement and compare, in this order of expected accuracy:

1. **EASE** — closed-form item-item linear model (`B = (X'X + λI)^-1`, measured 47 s for the
   20k×20k inverse on the dev Mac; ~10 GB peak fits molab). Open design questions for the build:
   encoding the §1 signal mapping into the X matrix, and sparsifying the dense B (1.6 GB → ~32 MB
   top-k per column) with a measured accuracy check. Known risk: most popularity-biased of the
   three (Gram-normalization is the cheap mitigation).
2. **ALS** via [`implicit`](https://github.com/benfred/implicit) — Hu/Koren/Volinsky confidence
   weighting is the natural home for the signal mapping (DROPPED = high confidence on zero
   preference); documented `recalculate_user` fold-in; ~10 MB item-factor export.
3. **SASRec** — the one non-linear candidate; answers whether model class matters on this data.
   Trains on molab's GPU using svanoo timestamps. Fallback if list-edit timestamps prove
   unordered: disable/shuffle positional embeddings, degrading to a set encoder.

**Baseline: item-item BM25** (`implicit`; 19 s fit on 300k users). The shipping bar, measured
live by the [prototype (#9)](https://github.com/Ari-03/AniList_Rec/issues/9) on 9,963 held-out
test users, dial off:

| model | NDCG@10 [95% CI] | recall@10 | recall@50 | niche pop lift | coverage@10 |
|---|---|---|---|---|---|
| item-item BM25 | 0.1287 [0.1251, 0.1323] | 0.113 | 0.325 | +3.1 | 6.1% |
| MostPopular | 0.1955 [0.1914, 0.1992] | 0.123 | 0.293 | +7.4 | 1.0% |

**A candidate ships only if it beats *both* numbers on NDCG@10** without MostPopular's degenerate
1% coverage (the Cremonesi top-N trap, confirmed live in this data). Shared engineering note:
cap per-user history — `Σ_u n_u²` from the heaviest users is the hidden training cost in every
candidate.

---

## 5. Evaluation protocol

Decided in [Define the evaluation protocol (#7)](https://github.com/Ari-03/AniList_Rec/issues/7);
proven end-to-end by the [prototype (#9)](https://github.com/Ari-03/AniList_Rec/issues/9)
([notebook](https://github.com/Ari-03/AniList_Rec/blob/prototype/baseline-bm25/notebooks/baseline_bm25_prototype.py)).

- **Split:** held-out users (absent from training entirely — mirrors the fold-in serving path);
  per-user temporal 80/20 by timestamp (earliest 80% = fold-in input, last 20% = target window).
- **Holdout:** 10k test + 10k validation users, disjoint, each with ≥20 lifetime interactions and
  ≥5 target-window positives. Validation tunes; test is touched only for final numbers.
- **Targets:** positives only — COMPLETED (unscored or normalized > 4/10) and REPEATING in the
  window; favourites always count. DROPPED / low-scored completions feed a separate **regret@10**
  diagnostic (fraction of top-k the user actually dropped or scored low), never hits.
- **Ranking scope:** full catalogue minus fold-in items and PLANNING — no sampled negatives
  (Krichene & Rendle 2020).
- **Franchise handling:** evaluate through the serving pipeline with the franchise filter ON;
  drop targets that are direct continuations of fold-in items; a target inside an unwatched
  franchise hits if the ranker surfaces that franchise's entry point.
- **Headline: NDCG@10** with graded gains (2 = strong positive per §1, 1 = standard), bootstrap
  95% CI over test users — candidates within noise are not ranked on noise. Also report
  recall@10 and recall@50 (@50 checks the dial's re-rank has the right items in range).
- **Guardrails (reported, not optimized):** niche-sliced popularity lift (bottom quartile of
  users by median profile-item popularity — watches the Kowald failure mode) and
  catalogue coverage@10 (catches everyone-gets-the-same-list rankers).
- **Dial:** compare architectures at dial-off; after a winner is chosen, sweep the dial on
  validation, plot NDCG@10 vs popularity lift, and pick the shipped default from the curve.
  The curve goes in the eval report.
- **Vibe check:** per finalist, a marimo notebook pulls the owner's AniList list by public
  username through the real serving path and renders the top-20 at three dial settings
  (off / default / high-novelty) with covers, genres, and a why-this line; each rec labeled
  seen-elsewhere / would-watch / plausible / bad / broken.
  **Caveat (accepted in [#13](https://github.com/Ari-03/AniList_Rec/issues/13)):** the vibe check
  is judged as a **2022-era time capsule** — "does it know my taste," not "is it current."
  Staleness is not model failure.

---

## 6. Export contract

Decided in [Define the export contract (#10)](https://github.com/Ari-03/AniList_Rec/issues/10).

**The export is a black-box scoring service in a Docker container.** The web app never speaks
Python and never learns which architecture won. Inside: the winning candidate's model data in
open formats (npz/Parquet), the AniList↔MAL crosswalk, and serve-time metadata, wrapped in a
small FastAPI service. The container is the contract; the files inside are documented but not
promised.

### Endpoints

- **Primary:** `recommend(username, dial?, limit?, type?)` → ranked `[{anilist_id, score}]`.
  The box fetches the user's public AniList list itself (owning the serve-time traps once, §7),
  applies the §1 mapping, folds in, collapses franchises to entry points, ranks. Bare ids only —
  display metadata is the caller's job (one batched AniList request per ~50 recs).
- **Secondary:** raw `{media_id, status, score}` entries in — the internal scoring layer the
  username endpoint wraps, for offline testing and account-less callers.

### Knobs

- `dial` (0–1; default = the validated setting from the §5 sweep) — per request, so the web app
  can ship a hidden-gems slider.
- `limit` (default 20).
- `type` (`ANIME` now; `MANGA` reserved, returns not-implemented until the manga effort).
- Internals (franchise filter, status weighting) are **not** exposed — swapping the winning
  architecture must not change what the public knobs mean.

### Promises

Typed errors passed through (unknown username, private list, AniList outage). No caching in v1 —
freshness and rate-limit budgeting (~2–4 AniList requests per call against 30/min) are the
caller's concern. Every response carries `model_version`; swapping the architecture or enabling
manga are non-breaking.

---

## 7. AniList integration (serve time)

From [Research: AniList API capabilities (#4)](https://github.com/Ari-03/AniList_Rec/issues/4)
([full findings](https://github.com/Ari-03/AniList_Rec/blob/research/anilist-api/research/anilist-api.md)),
confirmed live by the prototype. AniList is queried **only for the requesting user's own list at
serve time** — training data never comes from the API (ToS prohibits mass collection).

- `MediaListCollection(userName:, type:)` returns a public user's full list — no OAuth, no key.
- **Set a real `User-Agent`** — Cloudflare 403s default client agents.
- Statuses are `CURRENT` (not `WATCHING`) / `PLANNING` / `COMPLETED` / `DROPPED` / `PAUSED` /
  `REPEATING`.
- Use `score(format: POINT_100)` (or `POINT_10_DECIMAL`) so the server normalizes the user's
  score format; `score == 0` means unrated.
- Entries duplicate across custom lists — dedupe on `entries[].id`; don't skip custom lists
  (users hide entries from status lists).
- Rate limit is **30 req/min** (degraded state); 429 → `Retry-After: 60`; `X-RateLimit-Remaining`
  is unreliable and there's an undocumented burst limiter — pace steadily.
- Map ids via the baked-in crosswalk; for gaps, `Page(media(idMal_in: [...]))` in batches of 50
  (drops unknown ids silently). Always pass `type:` — anime and manga MAL id spaces overlap.
- Fetch favourites too (one extra request) — favourited is a strongest-positive signal in §1.
  (The prototype skipped this; the export must not.)

---

## 8. Manga extensibility

The architecture stays manga-extensible without speccing the manga model (out of scope):

- `media_type` is first-class in every data schema (§2).
- The export contract reserves `type: MANGA` — enabling it is a value change, not a breaking
  change (§6).
- AniList serves manga lists through the same `MediaListCollection(type: MANGA)` call, and
  `ADAPTATION`/`SOURCE` relations bridge anime↔manga when that effort starts.

---

## 9. Known simplifications to revisit during the build

Carried from the [prototype (#9)](https://github.com/Ari-03/AniList_Rec/issues/9):

- Training users capped at 300k of ~1M (config constant; uncap for real runs).
- "Direct continuation" is approximated as same franchise cluster (union-find over relations) —
  acceptable for eval, verify it doesn't over-suppress at serve time.
- BM25 baseline consumes positives only; candidates should use the full §1 confidence mapping.
- Serve-time favourites fetch not yet wired (see §7).

## 10. Out of scope for this effort

Per the [map](https://github.com/Ari-03/AniList_Rec/issues/1): the public web app; MAL/Kitsu
account connections; the manga recommender; post-cutoff (new-title) support and the
content-embedding sidecar.
