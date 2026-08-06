# AniList Recommender

An exportable anime recommendation model: connect an AniList account, get ranked recommendations for anime you haven't seen. Developed in Python + marimo; a public web app will later consume the exported model.

**The buildable spec is [SPEC.md](SPEC.md)** — assembled from the completed [wayfinder map](https://github.com/Ari-03/AniList_Rec/issues/1), whose closed sub-issues hold each decision's detail. Research findings live on `research/<name>` branches, linked from their tickets.

## The offline pipeline

The `anilist_rec` package (`src/`) is the shared offline pipeline every candidate
architecture runs on: the SPEC §1 signal mapping, franchise clusters with
entry-point collapse, the held-out temporal split, and the full-catalogue eval
harness (SPEC §5). The current baseline bar lives in
[reports/baseline_bar.md](reports/baseline_bar.md).

```sh
uv run scripts/acquire_data.py   # one-time: download + convert the datasets (no credentials)
uv run baseline                  # train both baselines, refresh reports/baseline_bar.md
uv run pytest                    # test the package
```
