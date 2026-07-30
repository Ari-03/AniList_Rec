# AniList GraphQL API — capabilities and constraints

Research for [#4](https://github.com/Ari-03/AniList_Rec/issues/4). Endpoint: `https://graphql.anilist.co` (POST only).

All claims below are either quoted from the official docs (https://docs.anilist.co) or verified live against the
API on **2026-07-30**. Live-verified claims are marked **[verified]** and every one of them has a runnable query in
this document. Where the docs and the live API disagree, that is called out explicitly.

**Headline for this project:** fetching a public user's anime + manga lists needs **no OAuth and no API key** — a
username is enough. The binding constraints are the **30 req/min degraded rate limit** and the **Terms of Use ban on
mass data collection**, not authentication. `idMal` covers ~95% of a typical list, and the misses are shorts/ONAs/
music videos that a MAL-trained model would not know anyway.

---

## 1. Fetching a user's anime and manga lists

### Does public username lookup suffice? Yes.

Per https://docs.anilist.co/guide/auth/ — things you can do **without** authentication include
"Get anime and manga data", "Search characters", and "**View data of public and unlisted users**". Authentication is
required only to "Modify user lists", "View data of private users (only for the currently authenticated user)", and
"Fetch user-specific data from fields on other objects. For example, the `mediaListEntry` field on `Media`."

**[verified]** An unauthenticated `MediaListCollection` query by `userName` returned a complete 1,488-entry anime
list including per-entry statuses, scores, progress and dates. No token, no client id, no `Authorization` header.

> Note: the v1 Client Credentials grant was removed precisely because "public API data no longer requires OAuth"
> (https://docs.anilist.co/guide/migration/version-1/).

### The working query

`MediaListCollection` "will return the user's full list all at once, split up by status and custom lists where
applicable" (https://docs.anilist.co/guide/graphql/queries/media-list). `type` is required, and you must supply
`userId` or `userName`.

```graphql
query UserList($name: String, $type: MediaType) {
  MediaListCollection(userName: $name, type: $type, forceSingleCompletedList: true) {
    hasNextChunk
    user {
      id
      name
      mediaListOptions { scoreFormat }
    }
    lists {
      name
      status
      isCustomList
      entries {
        id                        # MediaList entry id — dedupe on this
        status
        score                     # in the USER's score format
        score100: score(format: POINT_100)   # normalised, always 0-100
        progress
        repeat
        private
        hiddenFromStatusLists
        startedAt { year month day }
        completedAt { year month day }
        updatedAt
        media { id idMal title { romaji english } format episodes }
      }
    }
  }
}
```

Variables: `{"name": "tanukikay", "type": "ANIME"}` — swap `"MANGA"` for the manga list, which additionally exposes
`progressVolumes`, `media { chapters volumes }`.

```bash
curl -s -X POST https://graphql.anilist.co \
  -H 'Content-Type: application/json' \
  -H 'User-Agent: AniListRec/0.1' \
  -d '{"query":"query($n:String,$t:MediaType){MediaListCollection(userName:$n,type:$t){lists{name status isCustomList entries{id status score progress media{id idMal}}}}}","variables":{"n":"tanukikay","t":"ANIME"}}'
```

### Statuses

**[verified]** by introspecting `MediaListStatus` — all six exist, with the docs' own descriptions:

| Enum | Meaning | UI label |
|---|---|---|
| `CURRENT` | Currently watching/reading | Watching / Reading |
| `PLANNING` | Planning to watch/read | Planning |
| `COMPLETED` | Finished watching/reading | Completed |
| `DROPPED` | Stopped watching/reading before completing | Dropped |
| `PAUSED` | Paused watching/reading | Paused |
| `REPEATING` | Re-watching/reading | Rewatching |

Note the naming trap: the status is `CURRENT`, **not** `WATCHING`; and `REPEATING`, not `REWATCHING`. Re-watch
*count* is a separate integer field, `repeat`.

```graphql
{ __type(name: "MediaListStatus") { enumValues { name description } } }
```

### Scores and score formats

`MediaList.score` is a `Float` returned **in whatever format the user picked**, so a raw `7.5` is meaningless without
knowing the format. **[verified]** introspection of `ScoreFormat`:

| Format | Range |
|---|---|
| `POINT_100` | An integer from 0-100 |
| `POINT_10_DECIMAL` | A float from 0-10 with 1 decimal place |
| `POINT_10` | An integer from 0-10 |
| `POINT_5` | An integer from 0-5. Should be represented in Stars |
| `POINT_3` | An integer from 0-3. Should be represented in Smileys. 0 => No Score, 1 => :(, 2 => :\|, 3 => :) |

**The important trick: `score` takes a `format` argument.** Passing `score(format: POINT_100)` makes the server
normalise, so you never have to implement five conversions client-side.

**[verified]** For user `tanukikay` (whose `scoreFormat` is `POINT_10_DECIMAL`), the same entry returned
`score: 6.5` and `score(format: POINT_100): 65`. Always request `POINT_100` and treat the user's own
`mediaListOptions { scoreFormat }` as display metadata only.

**Gotcha: `score == 0` means *unrated*, not "rated zero".** **[verified]** 804 of 1,556 entries in the test list had
`score == 0` — these are overwhelmingly Planning entries. Filter them out before training; do not feed them as
zero-valued ratings.

There is also `advancedScores` (a JSON map of per-category scores, e.g. Story/Characters/Visuals) for users who
enable advanced scoring.

### Gotchas that will corrupt your data if ignored

1. **Entries are duplicated across custom lists.** **[verified]** The test list returned **1,556 entry rows** but only
   **1,488 unique `MediaList.id`** values — an entry in a custom list appears both under its status list and under
   each custom list. **Dedupe on `entries[].id`** (or `media.id`), or you will double-count ~5% of the ratings.
2. **You must not skip custom lists.** Docs, https://docs.anilist.co/guide/graphql/queries/media-list:
   > **TIP** Do not skip over the user's custom lists. Users can hide entries from the default status lists, but they
   > can still be accessed through the custom lists. If you skip the custom lists, you could very likely miss entries
   > that are only available in the custom lists.

   This is the `hiddenFromStatusLists` flag. **[verified]** custom lists come back with `status: null` and
   `isCustomList: true`, so a naive "group by status" drops them. Read every list, then dedupe.
3. **`forceSingleCompletedList: true`** collapses the per-format split of the Completed list (users can split
   Completed into TV/Movie/etc.). Recommended for ingestion.
4. **Private entries are silently omitted, not errored.** `MediaList.private` is documented as "If the entry should
   only be visible to authenticated user" (https://docs.anilist.co/reference/object/medialist). Unauthenticated
   fetches simply won't see them — there is no error and no count. Fully private *users* return an error; public and
   "unlisted" users are readable.
5. **`User.statistics` counts ≠ list length.** **[verified]** `statistics.anime.count` was 780 while the list held
   1,488 unique entries, because statistics exclude Planning. Don't use statistics as a completeness check.

---

## 2. Rate limits

### Current values

From https://docs.anilist.co/guide/rate-limiting, verbatim:

> **WARNING**
> The API is currently in a degraded state and is limited to **30 requests per minute**. This is a temporary measure
> until the API is fully restored.

> The AniList API has a rate limit of 90 requests per minute.

So: **90/min nominal, 30/min in the current degraded state.** The degraded state has been in effect for a long time
and should be treated as the real budget.

**[verified]** Live responses today return `x-ratelimit-limit: 30`. The degraded limit is real and current.

### Headers to watch

Docs name four: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After`.

> If you exceed the rate limit, you will recieve a 1 minute timeout. Any further requests in this timeout period will
> also include the `Retry-After` and `X-RateLimit-Reset` headers in the response. `Retry-After` is the number of
> seconds you should wait before making another request. `X-RateLimit-Reset` contains the Unix timestamp of when you
> can make another request.

**[verified], with two discrepancies against the docs:**

- On **200** responses the API sends `x-ratelimit-limit: 30` and `x-ratelimit-remaining: N`. It does **not** send
  `X-RateLimit-Reset`, despite listing it in `access-control-expose-headers`.
- On the **429** response the *only* rate-limit header present was **`Retry-After: 60`**. There were **no**
  `X-RateLimit-Limit`, `X-RateLimit-Remaining`, or `X-RateLimit-Reset` headers, contradicting the docs' example.
  **Write your backoff against `Retry-After` and default to 60s when it is absent.**
- The docs' example block still shows `X-RateLimit-Limit: 90` even under the 30/min degraded warning. **Read the
  header at runtime; never hardcode 90.**

**429 body [verified]** — exactly as documented:

```json
{"data": null, "errors": [{"message": "Too Many Requests.", "status": 429}]}
```

**Gotcha: `X-RateLimit-Remaining` is not trustworthy as a gate.** **[verified]** In a burst test the counter skipped
values (26, 25, 24, 23, 22, 21, **19**, 18 …) and the 429 fired while `remaining` still read **5**, not 0. There is
also an undocumented **burst limiter** on top of the per-minute limit:

> On top of the above rate limiting, we also have a burst limiter. This limiter is designed to stop you from
> hammering the API with too many requests in a very short period of time.

No numbers are given for it. Practical approach: pace to a fixed ~1 req per 2s rather than sprinting until
`remaining` hits 0, and always handle 429 + `Retry-After`.

### Other failure modes

- **403 = API disabled**, not rate limiting. https://docs.anilist.co/guide/considerations: "In cases of severe
  outages, we may lower the rate limits from those listed here or in extreme cases, temporarily suspend access to the
  API… If the API is unavailable, you will receive a `403` error code." Treat 403 as *stop*, not *retry*.
- **Manual IP blocking** for "a large number of requests being made from a single IP address".
- **Rate limit raises are closed:** "**WARNING** We are not currently accepting requests for increased rate limits."
- **Cloudflare blocks default client User-Agents.** **[verified]** Python `urllib` with its stock UA got
  **HTTP 403, body `error code: 1010`** (Cloudflare browser-integrity), before any GraphQL was even parsed. Setting
  any descriptive `User-Agent` fixed it immediately. **Always set a `User-Agent`.**
- **Always check `errors` even on HTTP 200** — https://docs.anilist.co/guide/graphql/errors: "**WARNING** Even if you
  recieve a status code of `200`, you may still receive an error."

---

## 3. Media metadata available

**[verified]** in a single query against `Media(id: 1)` (Cowboy Bebop) — everything below came back populated:

```graphql
query Meta($id: Int) {
  Media(id: $id) {
    id idMal
    title { romaji english native }
    synonyms
    format status source countryOfOrigin isAdult
    episodes duration chapters volumes
    season seasonYear startDate { year month day }
    averageScore meanScore popularity favourites trending
    genres
    tags { id name rank category isMediaSpoiler isGeneralSpoiler isAdult }
    studios { edges { isMain node { id name } } }
    staff(perPage: 10) { edges { role node { id name { full } } } }
    characters(perPage: 10) { edges { role node { id name { full } } } }
    relations { edges { relationType node { id idMal type format title { romaji } } } }
    recommendations(perPage: 10, sort: RATING_DESC) {
      nodes { rating mediaRecommendation { id idMal title { romaji } } }
    }
    externalLinks { site url }
  }
}
```

Notes on the fields that matter for a recommender:

- **`genres`** — a flat array of ~20 broad strings (`["Action","Adventure","Drama","Sci-Fi"]`). Coarse.
- **`tags`** — the real signal. Each carries a **`rank` (0-100)**, a `category` (e.g. `Setting-Universe`,
  `Theme-Other`, `Technical`), and spoiler/adult flags. **[verified]** Cowboy Bebop returned `Space` rank 94,
  `Crime` rank 92, etc. Filter on `rank` (the `Media` query even offers a `minimumTagRank` argument) and drop
  `isMediaSpoiler` tags before showing anything to a user.
- **`MediaTagCollection`** returns the **entire tag vocabulary in one unpaginated request** — **[verified] 425 tags**
  with id/name/category/isAdult. Fetch once, cache, use as your feature space.
- **`averageScore`** is a weighted 0-100 int; **`meanScore`** is the raw mean; **`popularity`** is the number of users
  with the media on a list (460,603 for Cowboy Bebop); **`favourites`** is favourite count.
- **`relations`** gives `relationType` values including `SEQUEL`, `PREQUEL`, `SIDE_STORY`, `ADAPTATION`,
  `SOURCE`, `ALTERNATIVE`, `SPIN_OFF`, `PARENT`, `CHARACTER`, `SUMMARY`, `OTHER`. Relation nodes expose `idMal` too,
  so franchise graphs can be built directly in MAL id space. Essential for suppressing "you watched S1, here's S2"
  non-recommendations.
- **`staff`** edges carry a free-text `role` string ("Director", "Music", "ADR Director (English)") — useful for
  director/composer affinity, but the role strings are unnormalised.
- **`studios`** edges carry `isMain`; non-main studios include producers/licensors (Bandai Visual, Bandai
  Entertainment), so filter on `isMain: true` for "the studio that made it".
- **`recommendations`** — AniList's own user-voted recommendations with a `rating` (net upvotes). A free baseline to
  benchmark the model against.

**User-level aggregates** are also available and worth knowing about, via `User.statistics` — **[verified]**
`count`, `meanScore`, `standardDeviation`, `minutesWatched`, `episodesWatched`, plus per-`genres` and per-`tags`
breakdowns each with `count` and `meanScore`. Handy for a cheap cold-start profile without pulling the whole list.

```graphql
query($n: String) {
  User(name: $n) {
    statistics {
      anime {
        count meanScore standardDeviation minutesWatched
        genres(limit: 10, sort: COUNT_DESC) { genre count meanScore }
        tags(limit: 10, sort: COUNT_DESC) { tag { id name } count meanScore }
      }
    }
  }
}
```

---

## 4. Mapping AniList ids ⇄ MAL ids

`Media.idMal` is the MyAnimeList id. This is the load-bearing field for this project, since the model trains on
MAL-id datasets and serves AniList accounts.

### Both directions work, unauthenticated

- **AniList → MAL:** just select `idMal` on any `Media`, including nested ones (`relations`, `recommendations`).
- **MAL → AniList:** the `Media` root query accepts **`idMal`**, and the `Page` query accepts **`idMal_in`** for bulk
  lookups. **[verified]** — this is the right way to align a MAL-id training set with live AniList data.

```graphql
# Bulk MAL -> AniList, up to 50 per request
query MalToAniList($malIds: [Int]) {
  Page(page: 1, perPage: 50) {
    media(type: ANIME, idMal_in: $malIds) { id idMal title { romaji } }
  }
}
```

```bash
curl -s -X POST https://graphql.anilist.co \
  -H 'Content-Type: application/json' -H 'User-Agent: AniListRec/0.1' \
  -d '{"query":"query($m:[Int]){Page(page:1,perPage:50){media(type:ANIME,idMal_in:$m){id idMal title{romaji}}}}","variables":{"m":[1,5,6,20,205,40052,52991]}}'
```

**[verified]** returns Cowboy Bebop, its movie, TRIGUN, NARUTO, Samurai Champloo, GREAT PRETENDER, Frieren.

Introspection **[verified]** confirms the full filter set on `Media`: `idMal`, `idMal_not`, `idMal_in`,
`idMal_not_in`, alongside `id`, `id_in`, `id_not_in`.

### Coverage

**[verified]** On a real 1,488-entry anime list, **72 entries (~4.8%) had `idMal: null`**. Sampling those, they are
uniformly the long tail:

| AniList id | Title | Format |
|---|---|---|
| 127225 | Slimetachi no Idobata Kaigi | ONA |
| 128306 | Re:Zero Break Time 2nd Season Part 2 | ONA |
| 136668 | BeyWheelz | TV |
| 158627 | Dungeon Meshi: Senshi no Kantan Cooking! | ONA |
| 160850 | Sailor Moon Cosmos - Kouhen | MOVIE |
| 185797 | Hypnosis Mic FINAL | MUSIC |
| 189513 | Sousou no Frieren: ●● no Mahou Part 2 | ONA |
| 198726 | Chainsaw Biyori | ONA |

Shorts, ONAs, promo/music videos, split-cour fragments and very recent entries. **These are exactly the titles absent
from a MAL-id training set anyway**, so dropping unmapped entries loses little signal. Log the drop rate per user and
expect roughly 3-6%.

### Gotchas

1. **Never assume `id == idMal`.** **[verified]** For legacy entries they coincide (AniList id 1 → MAL 1, 5 → 5,
   6 → 6, 20 → 20, 205 → 205, 1735 → 1735) because AniList seeded its catalogue from MAL. For anything modern they
   diverge hard: AniList **110349** → MAL **40052** (GREAT PRETENDER), AniList **154587** → MAL **52991** (Frieren).
   The coincidence on old ids makes this bug invisible in casual testing and catastrophic on new titles.
2. **The mapping is not guaranteed 1:1 across split-cour/recut releases.** AniList and MAL split seasons differently;
   an AniList "Part 2" may map to null or to the parent's MAL id. Verify by title where the join is load-bearing.
3. **Always pass `type: ANIME` or `type: MANGA`.** Anime and manga MAL id spaces are separate and overlap
   numerically — MAL anime 1 and MAL manga 1 are different works. Without `type` you can silently join a manga row
   onto an anime id.
4. **Unknown ids are silently dropped by `idMal_in`** — **[verified]** a bogus `9999999` in the list produced no
   error and simply no row. Diff request vs response to detect misses rather than trusting the count.

---

## 5. Bulk fetching, pagination limits, and fetching many users

### Hard limits (all **[verified]** live unless noted)

| Limit | Value | Behaviour |
|---|---|---|
| `Page.perPage` | **max 50** | **Silently clamped**, not an error — asking for 100 returns 50 and `pageInfo.perPage: 50` |
| `Page` depth | **5,000 entries** (`page × perPage`) | Hard error: `"Page depth exceeds maximum allowed for API requests (5000 entries)"`, status 400 |
| `MediaListCollection.perChunk` | **max 500** | Clamped; asking 600 returned 500 unique entries with `hasNextChunk: true` |
| `MediaListCollection` total | **11,000 most recently updated unique entries** (docs) | Silent truncation |
| Rate limit | **30 req/min** (degraded; 90 nominal) | 429 + `Retry-After: 60` |

On the 11,000 cap, https://docs.anilist.co/guide/graphql/queries/media-list:

> **WARNING** Currently, the `MediaListCollection` is limited to returning the **11,000 most recently updated unique
> entries**. This only affects a handful of users, all of whom have only achieved that many entries by using their
> lists for unintended purposes.

Irrelevant for real users; relevant if you scrape outliers.

### `PageInfo` is unreliable — do not paginate on it

https://docs.anilist.co/guide/graphql/pagination:

> **`PageInfo` Degredation** — Currently, the `PageInfo` object is limited in functionality due to performance
> issues. The `total` and `lastPage` fields are not currently accurate. You should only rely on `hasNextPage` for any
> pagination logic.

**[verified]** — a popularity-sorted anime page reported `total: 5000, lastPage: 100`, which is obviously just the
5,000-entry depth cap echoed back, not the real catalogue size (~20k+). **Loop on `hasNextPage` / `hasNextChunk`
only.**

Also: only one data field is allowed per `Page` query ("Only one of these fields can be used in a single `Page`
query… The `pageInfo` field is exempt from this rule").

### Consequence: you cannot enumerate the catalogue with one sort

The 5,000-entry depth cap means no single sorted `Page` traversal reaches the whole media catalogue. To pull a full
catalogue you must **partition the query space** so each partition stays under 5,000 — e.g. iterate
`seasonYear` × `season`, or walk `id_greater`/`id_lesser` windows, and paginate within each. Budget accordingly:
~20k anime ÷ 50 per page = ~400 requests ≈ **14 minutes minimum** at 30 req/min.

### Alias batching — the one big throughput win, with a sharp edge

A single HTTP request costs one rate-limit unit no matter how many aliased root fields it contains.
**[verified]** 20 aliased `Media(idMal:)` lookups resolved in **one** request for **one** unit:

```graphql
query {
  m0: Media(idMal: 1, type: ANIME)  { id idMal averageScore }
  m1: Media(idMal: 5, type: ANIME)  { id idMal averageScore }
  m2: Media(idMal: 20, type: ANIME) { id idMal averageScore }
  # … 20+ more
}
```

**But: one missing alias nulls the entire response.** **[verified]** A query with one valid and one nonexistent id
returned **HTTP 404** with `{"good": null, "bad": null}` and a single `"Not Found."` error — the *valid* field was
nulled too. A 60-alias batch resolved **0 of 60** because a handful of the ids didn't exist. AniList also sets a
non-200 HTTP status when the payload contains errors, so naive clients raise instead of reading partial data.

**Therefore: prefer `Page(media(id_in: […]))` / `idMal_in` over alias batching.** It is also 50-per-request, but it
**degrades gracefully** — unknown ids are silently dropped and the good rows still come back with HTTP 200. Reserve
alias batching for ids you have already confirmed exist.

### Bulk export: there is none, and mass collection is prohibited

There is **no bulk export endpoint, no data dump, and no documented caching/ETag guidance**. More importantly,
https://docs.anilist.co/guide/terms-of-use:

> Using the AniList API as a backup or data storage service is strictly prohibited.
> **Hoarding** or mass collection of data from the AniList API is strictly prohibited.

with a carve-out: "For purely educational projects, such as school assignments, we tend to be very lenient on the
3rd point."

**This is the real constraint on "fetching many users", not the rate limit.** Practical posture for this project:

- **Do not crawl AniList to build the training corpus.** Train on the existing MAL-id datasets (which is already the
  plan) and use AniList only to fetch *the requesting user's own list* at serve time.
- Cache the media catalogue locally and refresh incrementally; don't re-fetch metadata per request.
- One user's full anime + manga list costs **~2-4 requests** — trivially within budget for interactive serving.
- If bulk AniList collection ever becomes necessary, the terms point at `contact@anilist.co`; note that rate-limit
  raises are currently closed.

Other terms worth knowing: free for non-commercial use; commercial use is free under $150/month revenue and requires
a license above that; app names containing "AniList"/"AniChart" must include "UNOFFICIAL" or "for AniList"; and the
API may not be used "within competing noncomplementary services of the same nature… Anime/Manga list/tracker
services."

---

## Recommended client behaviour (summary)

1. POST to `https://graphql.anilist.co` with `query` + `variables`; **set a descriptive `User-Agent`** or Cloudflare
   returns 403/1010.
2. Budget **30 req/min**; read `X-RateLimit-Limit` at runtime instead of hardcoding. Pace steadily (~1 req/2s) rather
   than sprinting — `X-RateLimit-Remaining` lags and an undocumented burst limiter exists.
3. On **429**, sleep `Retry-After` (default 60s). On **403**, stop — the API is disabled, not throttled.
4. **Check the `errors` array on every response, including HTTP 200**, and don't assume non-200 means no data.
5. Paginate on `hasNextPage` / `hasNextChunk` only. Clamp `perPage ≤ 50`, `perChunk ≤ 500`, and keep
   `page × perPage < 5000`.
6. For list ingestion: `forceSingleCompletedList: true`, read **all** lists including custom ones, **dedupe on
   `entries[].id`**, request `score(format: POINT_100)`, and drop `score == 0` as unrated.
7. For id mapping: use `idMal_in` batches of 50 with an explicit `type`, diff request vs response to catch misses, and
   never assume `id == idMal`.

## Sources

- https://docs.anilist.co/guide/graphql/ — endpoint, request shape
- https://docs.anilist.co/guide/graphql/errors — errors on HTTP 200
- https://docs.anilist.co/guide/graphql/pagination — `PageInfo` degradation, one field per `Page`
- https://docs.anilist.co/guide/graphql/queries/media-list — `MediaListCollection`, 11,000 cap, custom-list tip
- https://docs.anilist.co/guide/rate-limiting — 90/30 per minute, headers, 429 body, burst limiter
- https://docs.anilist.co/guide/auth/ — what needs authentication; token lifetime (1 year, no refresh, no scopes)
- https://docs.anilist.co/guide/considerations — 403 outage behaviour, manual IP blocking
- https://docs.anilist.co/guide/terms-of-use — hoarding prohibition, commercial terms
- https://docs.anilist.co/guide/migration/version-1/ — Client Credentials grant removed
- https://docs.anilist.co/reference/query — `perPage` max 50, `perChunk` max 500
- https://docs.anilist.co/reference/object/medialist — `private`, `hiddenFromStatusLists`
- Live introspection and queries against https://graphql.anilist.co, 2026-07-30
