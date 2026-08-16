"""Shared builders for the service-contract test files."""

from fastapi.testclient import TestClient

from anilist_rec.anilist import AniListClient
from anilist_rec.service import build_app


def service_client(bundle_dir, anilist_client: AniListClient | None = None) -> TestClient:
    """The one TestClient construction. raise_server_exceptions=False is
    load-bearing: it lets typed-error tests observe the HTTP response instead
    of a raised exception."""
    return TestClient(build_app(bundle_dir, client=anilist_client), raise_server_exceptions=False)


def completed(media_id: int, score: float = 90) -> dict:
    return {"media_id": media_id, "status": "COMPLETED", "score": score}


def raw_body(**overrides) -> dict:
    """A /recommend/raw body; default entries suit the fixture bundle
    (completed item 3000, a singleton: candidates are 1000/2000/4000)."""
    body = {"entries": [completed(3000)]}
    body.update(overrides)
    return body
