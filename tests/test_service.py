"""The export contract over HTTP (SPEC §6, issue #20): knobs, typed errors,
model_version everywhere, MANGA reserved."""

import pytest
from fixture_bundle import FIXTURE_MODEL_VERSION, make_fixture_bundle
from helpers import raw_body, service_client

from anilist_rec.anilist import AniListClient


@pytest.fixture
def client(bundle_dir):
    return service_client(bundle_dir)


def test_health_carries_model_version(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": FIXTURE_MODEL_VERSION}


def test_raw_ranks_bare_anilist_ids(client):
    resp = client.post("/recommend/raw", json=raw_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == FIXTURE_MODEL_VERSION
    # sim row for item 30 scores entry 1000 highest; watched 3000 never appears
    ids = [r["anilist_id"] for r in data["recommendations"]]
    assert ids[0] == 1000
    assert 3000 not in ids
    assert set(data["recommendations"][0]) == {"anilist_id", "score"}


def test_raw_excludes_watched_franchise_and_planning(client):
    body = {
        "entries": [
            {"media_id": 2000, "status": "COMPLETED", "score": 90},  # franchise of 1000
            {"media_id": 3000, "status": "PLANNING"},
        ]
    }
    resp = client.post("/recommend/raw", json=body)
    ids = [r["anilist_id"] for r in resp.json()["recommendations"]]
    assert ids == [4000]  # 1000 collapsed away with 2000; 3000 blocked by PLANNING


def test_dial_demotes_popular(client):
    dial_off = client.post("/recommend/raw", json=raw_body(dial=0.0)).json()
    dialed = client.post("/recommend/raw", json=raw_body(dial=1.0)).json()
    assert dial_off["recommendations"][0]["anilist_id"] == 1000  # popular, raw score 4.0
    assert dialed["recommendations"][0]["anilist_id"] == 4000  # niche, wins re-ranked


def test_limit_knob(client):
    resp = client.post("/recommend/raw", json=raw_body(limit=1))
    assert len(resp.json()["recommendations"]) == 1


def test_dial_default_comes_from_the_manifest(tmp_path):
    # a bundle baked with a non-zero default: omitted dial must equal that
    # default, not dial-off (guards the manifest -> service wiring in CI)
    client = service_client(make_fixture_bundle(tmp_path / "b", dial_default=0.25))
    default = client.post("/recommend/raw", json=raw_body()).json()["recommendations"]
    explicit = client.post("/recommend/raw", json=raw_body(dial=0.25)).json()["recommendations"]
    off = client.post("/recommend/raw", json=raw_body(dial=0.0)).json()["recommendations"]
    assert default == explicit
    assert default != off  # dial 0.25 rescores; dial 0 passes raw scores through


@pytest.mark.parametrize("bad", [{"dial": 1.5}, {"dial": -0.1}, {"limit": 0}, {"type": "OVA"}])
def test_knob_validation(client, bad):
    assert client.post("/recommend/raw", json=raw_body(**bad)).status_code == 422


def test_manga_reserved_not_implemented(client):
    resp = client.post("/recommend/raw", json=raw_body(type="MANGA"))
    assert resp.status_code == 501
    assert resp.json()["error"] == "not_implemented"
    assert resp.json()["model_version"] == FIXTURE_MODEL_VERSION

    resp = client.get("/recommend", params={"username": "x", "type": "MANGA"})
    assert resp.status_code == 501


def test_unknown_entries_fold_out_silently(client):
    # AniList ids outside the corpus drop before fold-in (SPEC §3), not an error
    body = {
        "entries": [
            {"media_id": 3000, "status": "COMPLETED", "score": 90},
            {"media_id": 999999, "status": "COMPLETED", "score": 90},
        ]
    }
    assert client.post("/recommend/raw", json=body).status_code == 200


# --- username endpoint through a fake AniList transport ----------------------


def anilist_ok_transport(payload: dict):
    if "MediaListCollection" in payload["query"]:
        body = {
            "data": {
                "MediaListCollection": {
                    "lists": [
                        {
                            "entries": [
                                {
                                    "id": 1,
                                    "status": "COMPLETED",
                                    "score": 90,
                                    "progress": 24,
                                    "repeat": 0,
                                    "updatedAt": 100,
                                    "media": {"id": 3000, "idMal": 30, "episodes": 24},
                                }
                            ]
                        }
                    ]
                }
            }
        }
    else:
        body = {
            "data": {
                "User": {
                    "favourites": {
                        "anime": {"pageInfo": {"hasNextPage": False}, "nodes": []}
                    }
                }
            }
        }
    return 200, {}, body


def app_with_transport(bundle_dir, transport):
    return service_client(bundle_dir, anilist_client=AniListClient(transport=transport))


def test_recommend_username_live_path(bundle_dir):
    client = app_with_transport(bundle_dir, anilist_ok_transport)
    resp = client.get("/recommend", params={"username": "someone"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == FIXTURE_MODEL_VERSION
    assert data["recommendations"][0]["anilist_id"] == 1000


@pytest.mark.parametrize(
    ("response", "status", "code"),
    [
        (
            (200, {}, {"errors": [{"message": "User not found", "status": 404}]}),
            404,
            "unknown_username",
        ),
        ((200, {}, {"errors": [{"message": "Private user"}]}), 403, "private_list"),
        ((500, {}, {}), 502, "anilist_outage"),
    ],
)
def test_typed_errors_pass_through(bundle_dir, response, status, code):
    client = app_with_transport(bundle_dir, lambda _p: response)
    resp = client.get("/recommend", params={"username": "someone"})
    assert resp.status_code == status
    assert resp.json()["error"] == code
    assert resp.json()["model_version"] == FIXTURE_MODEL_VERSION


def test_rate_limit_carries_retry_after(bundle_dir):
    client = app_with_transport(bundle_dir, lambda _p: (429, {"Retry-After": "17"}, {}))
    resp = client.get("/recommend", params={"username": "someone"})
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "17"
    assert resp.json()["error"] == "rate_limited"
