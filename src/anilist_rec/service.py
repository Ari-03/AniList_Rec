"""The scoring service (SPEC §6, issue #20): the export contract over HTTP.

A small FastAPI app wrapping the shared serving path. The container is the
contract: ranked bare `[{anilist_id, score}]` out, typed AniList failures
passed through as typed HTTP errors, `model_version` in every response —
success or error. Internals (franchise filter, status weighting, which
architecture won) are never exposed; swapping the model is a bundle swap.

    uvicorn --factory anilist_rec.service:create_app

The factory reads ANIREC_BUNDLE_DIR (default /app/bundle — the Docker
layout); tests call `build_app` with a fixture bundle and a fake transport.
"""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from anilist_rec.anilist import (
    AniListClient,
    AniListOutageError,
    ListEntry,
    PrivateListError,
    RateLimitedError,
    UnknownUserError,
    UserAnimeList,
)
from anilist_rec.bundle import LoadedBundle, load_bundle

MAX_LIMIT = 200

MediaType = Literal["ANIME", "MANGA"]


class RecommendationOut(BaseModel):
    anilist_id: int
    score: float


class RecommendResponse(BaseModel):
    model_version: str
    recommendations: list[RecommendationOut]


class RawEntry(BaseModel):
    """One list entry in AniList terms (SPEC §6 secondary endpoint)."""

    media_id: int  # AniList media id
    status: str
    score: float = Field(default=0.0, ge=0, le=100, description="POINT_100; 0 = unrated")
    progress: int | None = None
    repeat: int = 0
    episodes: int | None = None
    updated_at: int | None = Field(default=None, description="unix epoch; orders the sequence")


class RawRequest(BaseModel):
    entries: list[RawEntry]
    favourite_media_ids: list[int] = Field(default_factory=list)
    dial: float | None = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)
    type: MediaType = "ANIME"


def build_app(
    bundle_dir: Path,
    client: AniListClient | None = None,
    loaded: LoadedBundle | None = None,
) -> FastAPI:
    """`loaded` lets a caller that already holds the bundle skip the second
    load (the ship acceptance suite); `client` is ignored when it is passed."""
    if loaded is None:
        loaded = load_bundle(bundle_dir, client=client)
    rec = loaded.recommender
    version = loaded.manifest.model_version
    dial_default = loaded.manifest.dial_default
    # AniList id → MAL id, for raw entries arriving without the MAL side.
    # The crosswalk dedupes the MAL side only (SPEC §2: some MAL ids share an
    # AniList entry), so invert first-wins: mal_to_anilist iterates in
    # popularity order, keeping the popular mapping deterministically.
    anilist_to_mal: dict[int, int] = {}
    for mal, anilist in rec.mal_to_anilist.items():
        anilist_to_mal.setdefault(anilist, mal)

    app = FastAPI(title="AniRec scoring service", version=version)

    def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={"error": code, "message": message, "model_version": version},
            headers=headers,
        )

    @app.exception_handler(UnknownUserError)
    async def unknown_user(_req: Request, exc: UnknownUserError) -> JSONResponse:
        return error(404, "unknown_username", str(exc))

    @app.exception_handler(PrivateListError)
    async def private_list(_req: Request, exc: PrivateListError) -> JSONResponse:
        return error(403, "private_list", str(exc))

    @app.exception_handler(RateLimitedError)
    async def rate_limited(_req: Request, exc: RateLimitedError) -> JSONResponse:
        return error(
            503, "rate_limited", str(exc), headers={"Retry-After": str(exc.retry_after)}
        )

    @app.exception_handler(AniListOutageError)
    async def outage(_req: Request, exc: AniListOutageError) -> JSONResponse:
        return error(502, "anilist_outage", str(exc))

    def to_response(recommendations) -> RecommendResponse:
        return RecommendResponse(
            model_version=version,
            recommendations=[
                RecommendationOut(anilist_id=r.anilist_id, score=r.score)
                for r in recommendations
            ],
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": version}

    @app.get("/recommend", response_model=RecommendResponse)
    def recommend(
        username: str,
        dial: float | None = Query(default=None, ge=0.0, le=1.0),
        limit: int = Query(default=20, ge=1, le=MAX_LIMIT),
        type: MediaType = "ANIME",
    ):
        if type == "MANGA":
            return error(501, "not_implemented", "MANGA is reserved for the manga effort")
        recs, _fold = rec.recommend(
            username, dial=dial if dial is not None else dial_default, limit=limit
        )
        return to_response(recs)

    @app.post("/recommend/raw", response_model=RecommendResponse)
    def recommend_raw(req: RawRequest):
        if req.type == "MANGA":
            return error(501, "not_implemented", "MANGA is reserved for the manga effort")
        entries = [
            ListEntry(
                entry_id=i,
                media_id=e.media_id,
                mal_id=anilist_to_mal.get(e.media_id),
                status=e.status,
                score100=e.score,
                progress=e.progress,
                repeat=e.repeat,
                episodes=e.episodes,
                updated_at=e.updated_at,
            )
            for i, e in enumerate(req.entries)
        ]
        fold = rec.fold_in(
            UserAnimeList(entries=entries, favourite_media_ids=set(req.favourite_media_ids))
        )
        recs = rec.recommend_foldin(
            fold, dial=req.dial if req.dial is not None else dial_default, limit=req.limit
        )
        return to_response(recs)

    return app


def create_app() -> FastAPI:
    """Uvicorn factory entry point; the bundle location comes from the environment."""
    return build_app(Path(os.environ.get("ANIREC_BUNDLE_DIR", "/app/bundle")))
