# Export contract — the scoring service

The export (SPEC §6, decided in
[Define the export contract (#10)](https://github.com/Ari-03/AniList_Rec/issues/10))
is a **black-box scoring service in a Docker container**. The container is the
contract; the files inside it are documented but not promised. The web app
never speaks Python and never learns which architecture is inside.

## Build and run

```sh
# the v1 ship: the SASRec winner with the swept dial default (reports/eval.md)
uv run export-bundle --arch sasrec --dial-default 0 \
    --model-version "1.0.0+sasrec.seed42" --out bundle
docker build -t anirec-scoring .
docker run --rm -p 8000:8000 anirec-scoring
```

(Any trained architecture exports the same way — `--arch bm25` bakes the
baseline; the service dispatches on the bundle's manifest.)

`export-bundle` bakes: the model artifact (open formats — npz), the corpus item
universe, popularity counts for the dial, and the AniList↔MAL crosswalk /
catalogue. **No raw user rows ever enter the image** (SPEC §2 licensing
guardrail).

## Endpoints

### `GET /recommend` — primary

The box fetches the user's public AniList list itself (list + favourites,
≤ ~4 AniList requests against the 30/min limit), applies the SPEC §1 signal
mapping, folds in, collapses franchises to entry points, ranks the full
corpus catalogue.

| query param | type | default | meaning |
|---|---|---|---|
| `username` | string | required | public AniList username |
| `dial` | float 0–1 | baked validated default | popularity dial; higher = more hidden gems |
| `limit` | int 1–200 | 20 | number of recommendations |
| `type` | `ANIME` \| `MANGA` | `ANIME` | `MANGA` reserved → `501` until the manga effort |

```json
{"model_version": "…", "recommendations": [{"anilist_id": 1535, "score": 12.3}, …]}
```

Bare ids only — display metadata (titles, covers) is the caller's job: one
batched AniList `Page(media(id_in: …))` request per ~50 recs.

### `POST /recommend/raw` — secondary

The internal scoring layer the username endpoint wraps, for offline testing
and account-less callers. Same response shape and knobs (`dial`, `limit`,
`type` ride in the body).

```json
{
  "entries": [
    {"media_id": 21, "status": "COMPLETED", "score": 85,
     "progress": null, "repeat": 0, "episodes": null, "updated_at": null}
  ],
  "favourite_media_ids": [21],
  "dial": 0.3,
  "limit": 20
}
```

- `media_id` is the **AniList** id; entries that don't map into the training
  corpus are silently dropped before fold-in (SPEC §3).
- `score` is POINT_100; `0` means unrated, never "rated zero".
- `status` uses AniList values (`CURRENT`/`PLANNING`/`COMPLETED`/`DROPPED`/
  `PAUSED`/`REPEATING`). `PLANNING` entries never fold in but still block
  recommendation output. Unknown statuses are ignored the same way.
- `updated_at` (unix epoch) orders the sequence for order-aware models;
  omitting it is fine for bag-of-items models.

### `GET /health`

`{"status": "ok", "model_version": "…"}` — liveness + version probe.

## Errors (typed, passed through)

Every error body is `{"error": <code>, "message": "…", "model_version": "…"}`.

| HTTP | `error` | when |
|---|---|---|
| 404 | `unknown_username` | AniList doesn't know the user |
| 403 | `private_list` | the user's list is private |
| 503 | `rate_limited` | AniList 429; `Retry-After` header passed through |
| 502 | `anilist_outage` | AniList disabled/unreachable/5xx |
| 501 | `not_implemented` | `type=MANGA` (reserved) |
| 422 | — (FastAPI validation shape) | knob out of range (`dial` ∉ [0,1], `limit` < 1, unknown `type`) |

## Promises

- Every response — success or error — carries `model_version`.
- **No caching in v1**: freshness and AniList rate-limit budgeting are the
  caller's concern.
- Internals (franchise filter, status weighting, winning architecture) are not
  exposed; swapping the model or enabling `MANGA` is **non-breaking** — knob
  semantics never change with the architecture.
- The item universe ends at the corpus cutoff (2022-03-22, SPEC §3): titles
  premiering after it are neither recommended nor read from input lists.
