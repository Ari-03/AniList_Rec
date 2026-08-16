"""Ship-v1 acceptance (issue #21): the winner's bundle through the export
contract, end to end against SPEC.md.

Runs only where the shipped SASRec bundle has been exported:

    uv run export-bundle --arch sasrec --dial-default 0 \\
        --model-version "1.0.0+sasrec.seed42" --out bundle

(or point ANIREC_BUNDLE_DIR at it). A fresh checkout, CI, and a fixture/bm25
bundle at bundle/ all skip this module — the fixture-bundle tests in
test_service.py cover the contract there. Tests marked `network` hit the live
AniList API; deselect with -m "not network". The offline bar itself is proven
in reports/eval.md; these tests prove the shipped configuration serves it.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import completed, raw_body

from anilist_rec.bundle import load_bundle
from anilist_rec.service import build_app

BUNDLE = Path(
    os.environ.get("ANIREC_BUNDLE_DIR", Path(__file__).resolve().parent.parent / "bundle")
)
SHIPPED_VERSION = "1.0.0+sasrec.seed42"

# FMA: Brotherhood — old enough that AniList id == MAL id; its franchise entry
# point is the 2003 series. Pinned so the tested subject can't drift with a
# catalogue refresh.
FMAB_ANILIST = 5114


def _manifest_arch() -> str | None:
    try:
        return json.loads((BUNDLE / "manifest.json").read_text())["architecture"]
    except (OSError, KeyError, ValueError):
        return None


pytestmark = pytest.mark.skipif(
    _manifest_arch() != "sasrec",
    reason=f"shipped SASRec bundle not at {BUNDLE} (see module docstring; "
    "a fixture/bm25 bundle there is not the ship candidate)",
)


@pytest.fixture(scope="module")
def loaded():
    return load_bundle(BUNDLE)


@pytest.fixture(scope="module")
def client(loaded):
    # one bundle load for the whole module; the recommender keeps its live
    # AniList client for the network-marked tests
    return TestClient(build_app(BUNDLE, loaded=loaded), raise_server_exceptions=False)


def recommend_ids(client, entries, **knobs):
    resp = client.post("/recommend/raw", json=raw_body(entries=entries, **knobs))
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert recs, "shipped model returned nothing"
    return [r["anilist_id"] for r in recs]


def test_manifest_ships_the_winner(loaded):
    m = loaded.manifest
    assert m.architecture == "sasrec"  # the winner per reports/eval.md
    assert m.dial_default == 0.0  # the §5 validation-sweep default
    assert m.model_version == SHIPPED_VERSION  # bumped from the 0.1.0 baseline bundle


def test_version_in_responses(client, loaded):
    assert client.get("/health").json()["model_version"] == loaded.manifest.model_version
    resp = client.post("/recommend/raw", json=raw_body(entries=[completed(FMAB_ANILIST)]))
    assert resp.json()["model_version"] == loaded.manifest.model_version


def anilist_to_index(loaded):
    """AniList id -> corpus index, one guarded mapping (crosswalk gaps drop,
    never KeyError)."""
    rec = loaded.recommender
    return {
        rec.mal_to_anilist[int(mal)]: idx
        for idx, mal in enumerate(rec.item_ids)
        if int(mal) in rec.mal_to_anilist
    }


def test_franchise_collapse_through_the_shipped_container(client, loaded):
    # Self-calibrating: only franchise entry points are recommendable, so take
    # a top rec that belongs to a multi-member franchise (it provably ranks),
    # watch one of its *siblings*, and the whole franchise must vanish.
    cluster_of = loaded.recommender.franchise.item_franchise
    a2i = anilist_to_index(loaded)

    seed = [completed(FMAB_ANILIST)]
    ranked = recommend_ids(client, seed, limit=50)
    target, members = next(
        (
            (t, {a for a, i in a2i.items() if cluster_of[i] == cluster_of[a2i[t]]})
            for t in ranked
            if t in a2i and (cluster_of == cluster_of[a2i[t]]).sum() >= 2
        ),
        (None, None),
    )
    assert target is not None, "no multi-member franchise in the top 50 — index broken?"

    sibling = next(m for m in members if m != target)
    got = recommend_ids(client, [*seed, completed(sibling)], limit=200)
    assert target not in got
    assert not (set(got) & members)


def test_planning_blocks_output_through_the_shipped_container(client):
    # whatever the model ranks first for this list must vanish once PLANNING'd
    seed = [completed(FMAB_ANILIST)]
    top = recommend_ids(client, seed)[0]
    blocked = recommend_ids(
        client, [*seed, {"media_id": top, "status": "PLANNING"}], limit=200
    )
    assert top not in blocked


def test_dial_defaults_to_the_swept_value(client, loaded):
    # omitted dial == the manifest default; the novelty knob re-orders, and in
    # the popularity-demoting direction (compare ranked ids, not score scales)
    seed = [completed(FMAB_ANILIST), completed(20583)]
    default = recommend_ids(client, seed)
    explicit = recommend_ids(client, seed, dial=loaded.manifest.dial_default)
    assert default == explicit

    dialed = recommend_ids(client, seed, dial=0.05)
    assert dialed != default
    counts, a2i = loaded.recommender.item_counts, anilist_to_index(loaded)

    def mean_popularity(ids):
        return float(np.mean([counts[a2i[a]] for a in ids]))

    assert mean_popularity(dialed) < mean_popularity(default)


def test_knob_semantics_unchanged_by_the_swap(client):
    # the acceptance record that swapping bm25 -> sasrec changed no public
    # knob; exhaustive validation coverage lives in test_service.py
    assert client.post("/recommend/raw", json=raw_body(entries=[], type="MANGA")).status_code == 501
    data = client.post(
        "/recommend/raw", json=raw_body(entries=[completed(FMAB_ANILIST)], limit=3)
    ).json()
    assert len(data["recommendations"]) == 3
    assert set(data["recommendations"][0]) == {"anilist_id", "score"}


@pytest.mark.network
def test_typed_errors_live_against_anilist(client):
    # a username that cannot exist -> typed 404 from real AniList
    resp = client.get("/recommend", params={"username": "zz_no_such_user_x9341"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_username"


@pytest.mark.network
def test_live_list_ranks_through_the_shipped_model(client):
    # the vibe-check account through the primary endpoint
    resp = client.get("/recommend", params={"username": "Zackhacks", "limit": 5})
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert len(recs) == 5
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)  # ranked, not shuffled
