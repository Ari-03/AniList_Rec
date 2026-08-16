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

Candidate training CLIs: `uv run ease`, `uv run als`, `uv run sasrec` (GPU via
[notebooks/sasrec_molab.py](notebooks/sasrec_molab.py)); `uv run compare`
regenerates the head-to-head, dial sweep, and winner verdict in
[reports/eval.md](reports/eval.md).

## The export (v1)

The shipped model is **SASRec** with `dial_default = 0`, chosen in
[reports/eval.md](reports/eval.md) and vibe-checked in
[reports/vibe_check_sasrec.md](reports/vibe_check_sasrec.md). The export is a
black-box scoring container (SPEC §6); its API is documented in
[docs/export-contract.md](docs/export-contract.md).

```sh
# assemble the winner's bundle from trained artifacts (uv run sasrec first)
uv run export-bundle --arch sasrec --dial-default 0 --model-version "1.0.0+sasrec.seed42" --out bundle

# bake it into the container and run
docker build -t anirec-scoring .
docker run --rm -p 8000:8000 anirec-scoring

curl "localhost:8000/recommend?username=<anilist-username>&limit=20"
```

`uv run pytest tests/test_ship_v1.py` runs the end-to-end acceptance suite
against the exported bundle (franchise + PLANNING exclusions, dial default and
direction, knob semantics); it skips itself unless a SASRec bundle sits at
`bundle/` (or `ANIREC_BUNDLE_DIR`). Two tests hit the live AniList API —
deselect with `-m "not network"` when offline.
