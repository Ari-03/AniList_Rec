# Public anime user–interaction datasets

Research for [#2](https://github.com/Ari-03/AniList_Rec/issues/2). Compiled 2026-07-30.

**Question.** Which public datasets of user–anime interactions could train this recommender, and how do they compare?

**Method.** Every Kaggle figure below comes from Kaggle's own dataset-metadata API
(`https://www.kaggle.com/api/v1/datasets/view/{owner}/{slug}`, which serves the dataset page's
description and byte counts as JSON — the rendered pages are client-side JS and can't be scraped
directly), plus each page's `application/ld+json` block for the download size. API facts come from the
official docs, and the AniList claims were additionally verified with live calls against
`https://graphql.anilist.co` on 2026-07-30. Anything I could not confirm from a primary source is
marked *(unverified)*.

A note on the two size numbers: Kaggle's `view` endpoint reports uncompressed total bytes, while the
`list` endpoint and the page's JSON-LD `distribution.contentSize` report the size of the zip you
actually download. Both are given below because the gap is large (svanoo is 1.9 GB down, 14.1 GB on disk).

---

## Comparison table

| | **svanoo** MyAnimeList Dataset | **hernan4444** Anime Rec. DB 2020 | **azathoth42** MyAnimeList | **dbdmobile** Anime Dataset 2023 | **AniList GraphQL** (scrape) | **Jikan** (scrape) | **MAL API v2** (scrape) |
|---|---|---|---|---|---|---|---|
| Users | 1,123,284 | 325,772 | 302,675 | not published *(unverified)* | you choose | — | you choose |
| Interactions | **223,812,614** | 109M (57M scored) | 80,076,112 (46.4M scored) | not published *(unverified)* | you choose | — | you choose |
| Anime | 13,379 | 17,562 | 14,478 | not published *(unverified)* | ~all | ~all | ~all |
| List statuses | ✅ 5 statuses | ✅ 5 statuses | ✅ 5 statuses (int codes) | ❌ **scores only** | ✅ 6 (incl. REPEATING) | n/a | ✅ 5 |
| Score | ✅ 1–10 + 6 review sub-scores | ✅ 1–10 (0 = unscored) | ✅ 1–10 | ✅ rating | ✅ any format, normalizable | n/a | ✅ 0–10 |
| Progress (eps) | ✅ `progress` | ✅ `watched_episodes` | ✅ | ❌ | ✅ `progress` | n/a | ✅ |
| **Timestamps** | ✅ `last_interaction_date` | ❌ **none** | ✅ `my_last_updated` | ❌ **none** | ✅ created/updated/started/completed | n/a | ✅ *(unverified)* |
| Item metadata | very rich (synopsis, studios, genres, full score histogram, clubs, pics) | rich (35 cols incl. score histogram) | rich | rich (24 cols + synopsis + image URL) | richest (tags w/ weights, relations, staff) | rich | rich |
| Side signals | ✅ friend graph (5.0M), anime–anime recs (214k), favourites, reviews | reviews via HTML dump (off-site) | user demographics (age/gender/location) | user demographics | follow graph, activity feed | — | — |
| Scrape recency | **2022-03-27** | Feb–Mar 2020 | 2018-06 | **2023-07-28** | **live** | live | live |
| License (declared) | CC0 | CC0 | ODbL 1.0 / DbCL 1.0 | ODbL 1.0 / DbCL 1.0 | ToS: mass collection prohibited | MIT code, MAL ToS on data | restrictive agreement |
| Download (zip) | 1.92 GB | **0.69 GB** | 2.06 GB | 1.93 GB | n/a | n/a | n/a |
| On disk | 14.14 GB | **2.87 GB** | 9.61 GB | 7.35 GB | n/a | n/a | n/a |
| ID scheme | MAL | MAL | MAL | MAL | **AniList + `idMal`** | MAL | MAL |
| Manga interactions | ❌ | ❌ (author lists it as future work) | ❌ | ❌ | ✅ | ❌ | ✅ |

---

## Per-dataset notes

### svanoo — "MyAnimeList Dataset" (the flagged candidate)

<https://www.kaggle.com/datasets/svanoo/myanimelist-dataset> · CC0 · v2, last updated **2022-03-27** ·
1.92 GB zipped / 14.14 GB uncompressed · 2,997 downloads.

Self-described as the "Largest Anime Dataset on Kaggle", and the interaction count backs that up. Its
own overview states: 13,379 anime, **1,123,284 users**, **223,812,614 user–anime interactions**,
5,048,994 user–user friendship edges, and 214,271 anime–anime edges (recommendations + relations).

The interaction table is split across **70 files**, `user_anime000000000000.csv` … `user_anime00000000069.csv`,
with columns: `user_id`, `anime_id`, `favorite` (0/1), `review_id`, `review_date`, `review_num_useful`,
`review_score`, `review_story_score`, `review_animation_score`, `review_sound_score`,
`review_character_score`, `review_enjoyment_score`, `score`, `status` (completed / watching /
plan_to_watch / dropped / on_hold), `progress` (episodes watched), and **`last_interaction_date`**.

This is the richest interaction schema of the four dumps — it is the only one carrying per-interaction
timestamps *and* favourites *and* review sub-scores in the same table. `anime.csv` is correspondingly
deep (synopsis, studios, genres, season, the full 1–10 score histogram, per-status member counts,
clubs, picture URLs), and `user.csv` adds per-user aggregates and `last_online_date`.

Two things to weigh against it. The **13,379 anime is the smallest catalogue of the four**, despite
being the largest by users — the crawl went wide on users, not on titles, so long-tail and niche
entries are thinner than hernan4444's 17,562. And 14.1 GB across 70 CSVs with free-text review
columns is the heaviest thing here to handle on a laptop.

The author credits the **Jikan API** as the source, published the crawler and ETL
([github.com/svan0/anime_recommendation_system](https://github.com/svan0/anime_recommendation_system/tree/main)),
and runs a live recommender on it.

### hernan4444 — "Anime Recommendation Database 2020"

<https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020> · CC0 · v7, updated
2021-07-13, **data scraped 26 Feb – 20 Mar 2020** · 0.69 GB zipped / 2.87 GB uncompressed · 27,815
downloads · usability 1.0.

17,562 anime, 325,772 users. Two interaction files:

- `animelist.csv` — **109M rows**: `user_id`, `anime_id` (MAL id), `score` (1–10, **0 = unscored**),
  `watching_status`, `watched_episodes`. Covers all five statuses; `watching_status.csv` decodes the codes.
- `rating_complete.csv` — **57M rows**, 16,872 anime, 310,059 users: the subset that was both completed
  and scored, as plain `user_id, anime_id, rating`.

`anime.csv` has 35 documented columns including the full per-score histogram (`Score-10` … `Score-1`)
and per-status member counts. Crawler is public
([github.com/Hernan4444/MyAnimeList-Database](https://github.com/Hernan4444/MyAnimeList-Database)); Jikan
is credited for the user preference data.

**The decisive weakness is that there are no timestamps anywhere in the interaction files** — no
`updated_at`, no watch date. Any temporal split, recency weighting, or sequential model is off the
table; you're limited to random holdout. Also note the author's explicit warning that hentai entries
are included, which you will want to filter. Includes an HTML dump (2 sample zips on Kaggle, the full
set on Google Drive) if you ever want reviews/synopses/staff beyond the CSVs.

The author lists "Build the same dataset with manga and novel" as future work — it was never done.

### azathoth42 — "MyAnimeList Dataset"

<https://www.kaggle.com/datasets/azathoth42/myanimelist> · ODbL 1.0 database / DbCL 1.0 contents · v9,
**last updated 2018-06-29** · 2.06 GB zipped / 9.61 GB uncompressed · 23,129 downloads.

302,675 users, **80,076,112 list records** (46,358,322 with ratings), 14,478 anime.
`UserAnimeList.csv` carries username, anime id, score, status, **and the last-updated timestamp** —
the same core schema as svanoo minus favourites and reviews. `my_status` is integer-coded and the page
documents it: `1` watching, `2` completed, `3` on hold, `4` dropped, `6` plan to watch ("other values
are not known").

Distinctive strengths: it is the only dump with **real user demographics** (gender, birth date,
location, join date), and it ships three tiers — raw, `*_filtered` (116,133 users with complete
demographics; 35.8M records), and `*_cleaned` (the filtered set with impossible episode counts,
absurd ages, and bogus `last_online` values repaired). The cleaning work is genuinely useful and
documented on the page.

Two caveats. It is **eight years stale** — nothing after mid-2018, which is a large fraction of the
modern catalogue missing. And the sampling is non-random in a way the author is refreshingly open
about: MAL has no iterable user ids, so usernames were harvested from *watching-challenge forum
threads (2015–2018) and ~40k of MAL's ~80k clubs*. That biases hard toward engaged, forum-active,
club-joining users. Analysis notebooks: [github.com/racinmat/mal-analysis](https://github.com/racinmat/mal-analysis).

### dbdmobile — "Anime Dataset 2023"

<https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset> · ODbL 1.0 / DbCL 1.0 · v5,
**last updated 2023-07-28** ("Added 2023 datasets") · 1.93 GB zipped / 7.35 GB uncompressed ·
19,491 downloads · usability 1.0.

The most recent of the four dumps, and the item metadata is good: `anime-dataset-2023.csv` has 24
columns including English/native titles, synopsis, genres, studios, producers, licensors, source,
`Rank`, `Popularity`, `Favorites`, `Scored By`, `Members`, and image URLs. `users-details-2023.csv`
gives per-user aggregates plus gender, birthday, location, join date, and `Rewatched`.

**But the interaction file is impoverished.** `users-score-2023.csv` has exactly five columns:
`user_id`, `Username`, `anime_id`, `Anime Title`, `rating`. That's it — **no list status, no episode
progress, no timestamp**, and it denormalizes the username and title into every row (which is most of
why the file is so large for what it carries). You cannot distinguish "dropped at episode 2" from
"completed and loved it" beyond the score, and plan-to-watch/watching entries appear to be absent
entirely since only rated entries have a rating.

The dataset page does not publish row, user, or anime counts for any file, so I can't state its
interaction volume from a primary source *(unverified)*.

Net: **best-in-class recency, worst-in-class interaction schema.** Its natural role here is as an
item-metadata refresh layered on a richer interaction source, not as the interaction source itself.

### Also found (metadata only — no interactions, but useful)

- **calebmwelsh — "Anilist Anime Dataset"**
  <https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset> · CC0 · **updated 2026-07-26**
  (four days ago) · 0.93 GB. Built directly off the AniList GraphQL API, one row per anime, and
  critically it carries **both `id` (AniList) and `idMal`** alongside tags, relations, characters,
  staff, studios, rankings, and `averageScore`/`meanScore`/`popularity`. This is a ready-made,
  CC0-licensed, current MAL↔AniList crosswalk *and* AniList-native item-feature table. Usability 0.82;
  contents not independently spot-checked *(unverified)*.
- **andreuvallhernndez — "MyAnimeList Anime and Manga Datasets"**
  <https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist> · CC0 · 2023-08-07 · 75 MB.
  24,165 anime and **67,273 manga**, cleaned, scraped via the official MAL API + Jikan. No user
  interactions, but it's the best cheap manga *item* table found, for when the project extends.

I found **no public dataset of user–manga interactions** at any comparable scale. That gap has to be
filled from the AniList or MAL APIs.

---

## Scraping the APIs directly

### AniList GraphQL — the only API that can both discover users and read their lists

Endpoint `https://graphql.anilist.co`, POST only ([docs](https://docs.anilist.co/guide/graphql/)).
**No auth is needed for public reads** — verified live today with an unauthenticated call.

- **Rate limit: 30 req/min, not 90.** The [rate-limiting page](https://docs.anilist.co/guide/rate-limiting)
  says "The API has a rate limit of 90 requests per minute" but carries a standing banner: *"The API is
  currently in a degraded state and is limited to 30 requests per minute."* The live response header
  confirms the banner — `x-ratelimit-limit: 30`. The same page states **"We are not currently accepting
  requests for increased rate limits."** There is also a separate burst limiter that trips on
  near-simultaneous requests even with quota remaining, so requests must be spaced, not just budgeted.
  Ceiling: ~43,200 requests/day.
- **One request = one user's entire list.** `MediaListCollection(userName:, type: ANIME)` returned 325
  entries across 6 lists for a test user in a single call. Fields on `MediaList` (confirmed by schema
  introspection): `status`, `score(format:)`, `progress`, `progressVolumes`, `repeat`, `priority`,
  `private`, `notes`, `customLists`, `advancedScores`, `startedAt`, `completedAt`, `createdAt`,
  `updatedAt`. `MediaListStatus` = `CURRENT, PLANNING, COMPLETED, DROPPED, PAUSED, REPEATING` — six
  statuses, one more than MAL, and `REPEATING` plus `repeat` is signal no MAL dump has.
  `score(format: POINT_100)` normalizes across users' differing score formats.
- **Two traps.** `MediaListCollection` is
  [documented](https://docs.anilist.co/guide/graphql/queries/media-list) as "limited to returning the
  11,000 most recently updated unique entries", and the same page warns not to skip custom lists —
  entries hidden from status lists come back with `status: null` and are only reachable via
  `customLists`. Use `forceSingleCompletedList: true` and dedupe on `mediaId`.
- **User enumeration works.** `Page { users(sort: ID) }` runs unauthenticated, but page depth is capped
  at 5,000 entries per sort ordering. `User(id: N)` probing works too; `sort: ID_DESC` puts the current
  max user id at ~8,178,153, and ids are sparse. Realistic bootstrap is the 5,000 from `Page.users`
  plus a follow-graph crawl (`Following`/`Follower`/`Activity`).
- **Manga is fully supported**: `MediaListCollection(type: MANGA)` returns `progress` (chapters),
  `progressVolumes`, and `Media.chapters`/`.volumes`.
- **ToS is the real constraint.** [Terms of Use](https://docs.anilist.co/guide/terms-of-use):
  *"Hoarding or mass collection of data from the AniList API is strictly prohibited"* and *"Using the
  AniList API as a backup or data storage service is strictly prohibited."* Competing list/tracker
  services are also prohibited without authorization. **There is an explicit carve-out:** *"Data
  collection for educational purposes — For purely educational projects, such as school assignments, we
  tend to be very lenient on the 3rd point"* (the 3rd point being mass collection). That leniency is
  discretionary, not a grant, and says nothing about *redistributing* what you collect. Commercial use
  is free below $150/month revenue. [Considerations](https://docs.anilist.co/guide/considerations) adds
  that AniList may block IPs issuing high request volumes.

### Jikan — off the table for interaction data

v4, `https://api.jikan.moe/v4`, 3 req/sec and 60 req/min, unlimited daily, 24-hour cache
([docs](https://docs.api.jikan.moe/)). It works by scraping MAL and requires no auth.

**The user-list endpoints are dead.** `/users/{username}/animelist` and `/mangalist` are still present
in the v4 OpenAPI spec — so a naive client will find them — but each is annotated *"User Anime lists
have been discontinued since May 1st, 2022"*, per
[Jikan's official statement](https://docs.google.com/document/d/1-6H-agSnqa8Mfmw802UYfGQrceIEnAaEh4uCXAPiX5A),
which directs users to the official MAL API instead. A live call today returns HTTP 504. In fact
**every** `/users/*` path probed today (profile, statistics, userupdates, search, userbyid, random)
returned 504, while `/anime/1` returns 200 normally; whether that is a transient outage or a permanent
MAL block is not documented anywhere I could find *(unverified)*.

Jikan remains a fine source for *catalogue* metadata — it's what both svanoo and hernan4444 used, back
when the list endpoints still existed.

### MyAnimeList official API v2 — technically capable, contractually the worst fit

[Docs](https://myanimelist.net/apiconfig/references/api/v2) (marked Beta). Requires at minimum an
`X-MAL-CLIENT-ID` header; unauthenticated calls return 403. `GET /users/{user_name}/animelist`
documents its path param as *"User name or `@me`"* and accepts bare client-ID auth, so **arbitrary
public users are readable** — unlike `GET /users/{user_name}`, which is documented as *"You can only
specify `@me`."* `limit` maxes at 1000 per page. A symmetric `/mangalist` exists with
`num_volumes_read` / `num_chapters_read`. **No rate limit is documented anywhere**; the status-code
table just lists 403 as "DoS detected etc.", so throttling is undisclosed and signalled by 403 rather
than 429 — plan adaptive backoff *(unverified: any numeric limit)*.

The [API License and Developer Agreement](https://myanimelist.net/static/apiagreement.html) (last
modified 2019-08-08) is the problem:

- **§3(c)** — you may *"not maintain, store or process any MyAnimeList Content that consists of
  personal information of MyAnimeList users or content generated by such users … on the server-side of
  Your Applications"*, client-side only and only as long as needed. A server-side training corpus of
  user lists is close to the literal definition of what this forbids.
- **§4(i)** — bans *"data scraping, web scraping, web harvesting, or other data extraction methods"*,
  which also makes Jikan a ToS problem independent of whether it works.
- **§4(f)** — no making user personal information available to third parties without written
  authorization; **§4(g)** — MAL may demand a copy of everything you collect; **§3(e)** — deletions must
  propagate within 24 hours.
- Research/free use *is* named as non-commercial (§1(i) covers "research applications"), but §3(c) and
  §4(f) are not scoped to commercial use and apply regardless.

---

## Mapping MAL ids → AniList ids

Every usable interaction dump is MAL-keyed; the product is AniList-facing. The crosswalk is the join,
and it turns out to be cheap. `Media.idMal` is populated and queryable in bulk — **`idMal_in` accepts a
batch and `perPage` goes to 50**, verified live today:

```
Page(page:1, perPage:50){ media(type:ANIME, idMal_in:[...]){ id idMal } }
```

returned all 12 probe titles correctly. So a ~25,000-title catalogue maps in **~500 requests ≈ 17
minutes** at the degraded 30 req/min — a one-time, cacheable cost, and modest enough that it doesn't
read as "mass collection".

**Do not assume identity mapping.** Early ids coincide because AniList seeded from MAL (`1→1`,
`21→21`, `5114→5114`, `11061→11061`), but they diverge for anything newer: *Kimi no Na wa.* is AniList
21519 / MAL 32281, *Kimetsu no Yaiba* is 101922 / 38000, *Sousou no Frieren* is 154587 / 52991.

Note also that `Page.media` has **no `id_greater` argument** (the schema suggests `episodes_greater`,
`chapters_greater`, etc. instead), and `pageInfo.total` always reports `5000` regardless of the real
match count — so full-catalogue enumeration must partition by something like `seasonYear` and paginate
until `hasNextPage` is false, rather than trusting `total`. The `idMal_in` route sidesteps this
entirely when you already hold a MAL id list, which you will.

Zero-request alternative: the CC0 **calebmwelsh** Kaggle set above already ships `id` + `idMal`
columns and was refreshed 2026-07-26.

---

## Implications for this project

**On the flagged candidate.** svanoo is a defensible pick and genuinely the largest — 224M interactions
with statuses, progress, timestamps, favourites, and a friend graph is more signal than anything else
public. The costs are its 14.1 GB footprint across 70 CSVs and its 13,379-title catalogue, the
narrowest of the four. If the goal is one dataset that maximizes signal per row, it wins; if the goal
is to get a first model training on a laptop this week, hernan4444 is 5× smaller and has 30% more titles.

**Training on a Mac / free Colab.** Free Colab gives ~12.7 GB RAM, so svanoo's 14.1 GB cannot be loaded
naively and needs lazy columnar reads (Polars `scan_csv` / Parquet) with column projection — you only
need `user_id, anime_id, score, status, progress, last_interaction_date`, which is a small slice of a
table whose bulk is review text. hernan4444's 109M rows fit comfortably: with int32/int8 downcasting
that's roughly 1.3 GB in memory (12 bytes/row), versus ~4.4 GB at pandas' default 64-bit types
(40 bytes/row). `rating_complete.csv` (57M
rows, three columns) is the easiest possible starting point for a first working model. Converting
whichever dump wins to Parquet once, up front, is worth doing regardless.

**Serving arbitrary AniList users not in training.** This rules out any model whose user
representation is a learned per-user embedding looked up by id. The user side has to be computed from
their list at inference — item-item collaborative filtering, or item embeddings (ALS/EASE/SLIM/two-tower)
with the user vector formed by pooling the items they've rated. All four dumps support this equally,
since only item-side parameters are learned and retained. The corollary is that **`user_id` in these
dumps is disposable** — hernan4444's ids are explicitly randomized and non-identifiable — which is
convenient both technically and for the privacy questions below.

**Timestamps are the sharpest discriminator.** Only svanoo and azathoth42 have them. Without
timestamps (hernan4444, dbdmobile) you cannot do a temporal train/test split, and evaluation has to
fall back on random holdout — which optimistically biases results for a recommender, since predicting
a title a user watched *before* the ones you trained on is easier than predicting the next one. If
honest offline evaluation matters, that argues for svanoo.

**Recency and the cold-catalogue problem.** Even the newest dump stops at mid-2023, so roughly three
years of anime — everything from late 2023 through 2026 — has **zero interaction data** in any of them.
For a live recommender this is a permanent structural gap, not a staleness annoyance: recent seasons
are exactly what users most want recommended. Practical shape: learn item embeddings from the dump's
interactions, then project new titles into that space from AniList content features (tags with weights,
genres, studios, staff, relations, `averageScore`) — which is a second argument for pulling AniList-native
item metadata rather than relying on the dumps' MAL metadata. dbdmobile's 2023 metadata or calebmwelsh's
2026 AniList metadata both help here; neither fixes the missing *interactions*.

**Licensing.** The Kaggle licences (CC0 for svanoo and hernan4444; ODbL/DbCL for azathoth42 and
dbdmobile) are uploader-declared and do not override MAL's own terms on the underlying data — MAL's
§4(f) forbids redistributing user personal information obtained via its API, and all four dumps are
ultimately MAL-derived. In practice: consuming these dumps for training is the well-trodden path
(tens of thousands of downloads each, hundreds of public notebooks), but **redistributing raw user
rows, or shipping them inside the exported model, is the thing to avoid.** Exporting learned *item*
embeddings rather than any user-level data sidesteps this cleanly — and happens to be the same design
the "arbitrary AniList user" requirement already forces.

**Scraping as a supplement, not a substitute.** AniList is the only API that can both discover users
and read full lists without auth, but at 30 req/min with no increases available, collecting a corpus
comparable to even hernan4444's 325k users would take weeks of continuous crawling and would run
straight into the "hoarding or mass collection is strictly prohibited" clause. The educational
carve-out is discretionary and worth an email to contact@anilist.co before relying on it. What AniList
*is* clearly right for, at trivial request cost: the id crosswalk (~500 requests), current item
metadata for post-2023 titles, and fetching the target user's own list at inference time — which is
ordinary intended use, not collection.

**Manga extension.** No public manga *interaction* dataset exists at usable scale. When the project
extends, manga interactions must come from AniList (`MediaListCollection(type: MANGA)`, with
`progress` = chapters and `progressVolumes`) or MAL's `/mangalist`; andreuvallhernndez's CC0 set covers
the 67k manga *items*. Worth designing the schema now so `media_type` is a first-class column rather
than an anime-only assumption baked into table names.

---

## Open questions

- Row/user/anime counts for dbdmobile's `users-score-2023.csv` are not published on its dataset page;
  they'd need a download to establish.
- "Newest anime covered" per dump is inferred from scrape/update dates, not verified title-by-title.
- MAL API v2's read-side `list_status` field names could not be confirmed — the docs render "No sample"
  and give no schema for the read path; only the write-side `PATCH` body is documented.
- Whether Jikan's blanket `/users/*` 504s are transient or permanent is undocumented.
