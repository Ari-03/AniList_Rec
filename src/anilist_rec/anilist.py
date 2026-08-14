"""Serve-time AniList client (SPEC §7, issue #15).

Fetches a public user's anime list and favourites unauthenticated, owning the
serve-time traps once: a real User-Agent (Cloudflare 403s stock agents),
dedupe across custom lists on entries[].id, server-normalized POINT_100
scores (0 = unrated), favourites fetched as the §1 strongest-positive signal,
and typed failures the export contract passes through (SPEC §6).

Request budget per call: 1 MediaListCollection + up to 3 favourites pages —
≤ 4 requests against the 30/min degraded limit.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

API_URL = "https://graphql.anilist.co"
USER_AGENT = "AniList-Rec/0.1 (https://github.com/Ari-03/AniList_Rec)"
MAX_FAVOURITE_PAGES = 3


class AniListError(Exception):
    """Base for the typed failures the export contract passes through (SPEC §6)."""


class UnknownUserError(AniListError):
    pass


class PrivateListError(AniListError):
    pass


class AniListOutageError(AniListError):
    """API disabled (403), server error, or unreachable — stop, don't retry."""


class RateLimitedError(AniListOutageError):
    """429; an outage from the caller's perspective (no caching/retry in v1)."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


# (status, headers, parsed JSON body) — injectable for tests
Transport = Callable[[dict], tuple[int, dict, dict]]


@dataclass(frozen=True)
class ListEntry:
    """One deduped anime list entry, scores already server-normalized to 0-100."""

    entry_id: int
    media_id: int  # AniList id
    mal_id: int | None
    status: str  # CURRENT | PLANNING | COMPLETED | DROPPED | PAUSED | REPEATING
    score100: float  # 0 = unrated, never "rated zero"
    progress: int | None
    repeat: int
    episodes: int | None
    updated_at: int | None = None  # unix epoch of the last list edit (sequence order)


@dataclass(frozen=True)
class UserAnimeList:
    entries: list[ListEntry]
    favourite_media_ids: set[int] = field(default_factory=set)


LIST_QUERY = """
query ($name: String) {
  MediaListCollection(userName: $name, type: ANIME, forceSingleCompletedList: true) {
    hasNextChunk
    lists {
      entries {
        id
        status
        score(format: POINT_100)
        progress
        repeat
        updatedAt
        media { id idMal episodes }
      }
    }
  }
}
"""

FAVOURITES_QUERY = """
query ($name: String, $page: Int) {
  User(name: $name) {
    favourites {
      anime(page: $page, perPage: 50) {
        pageInfo { hasNextPage }
        nodes { id }
      }
    }
  }
}
"""


def _http_transport(payload: dict) -> tuple[int, dict, dict]:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # AniList sets non-200 statuses on GraphQL errors; the body still parses
        try:
            body = json.loads(e.read())
        except (json.JSONDecodeError, OSError):
            body = {}
        return e.code, dict(e.headers or {}), body
    except (urllib.error.URLError, TimeoutError) as e:
        raise AniListOutageError(f"AniList unreachable: {e}") from e


class AniListClient:
    def __init__(self, transport: Transport = _http_transport):
        self._transport = transport

    def _query(self, query: str, variables: dict) -> dict:
        status, headers, body = self._transport({"query": query, "variables": variables})
        errors = body.get("errors") or []

        if status == 429:
            retry = headers.get("Retry-After") or headers.get("retry-after") or 60
            raise RateLimitedError("AniList rate limit hit", retry_after=int(retry))
        for err in errors:
            message = str(err.get("message", ""))
            if "private" in message.lower():
                raise PrivateListError(message)
            if err.get("status") == 404:
                raise UnknownUserError(message or "User not found")
        if status == 403:
            raise AniListOutageError("AniList API disabled (403)")
        if status >= 500:
            raise AniListOutageError(f"AniList server error ({status})")
        if errors:  # errors can ride on HTTP 200 — never ignore them
            raise AniListOutageError("; ".join(str(e.get("message")) for e in errors))
        return body.get("data") or {}

    def fetch_list(self, username: str) -> list[ListEntry]:
        """The user's full anime list, deduped across custom lists (1 request)."""
        data = self._query(LIST_QUERY, {"name": username})
        collection = data.get("MediaListCollection")
        if collection is None:
            raise UnknownUserError(f"User not found: {username}")

        by_entry_id: dict[int, ListEntry] = {}
        for lst in collection.get("lists") or []:
            for e in lst.get("entries") or []:
                media = e.get("media") or {}
                if media.get("id") is None:
                    continue
                by_entry_id.setdefault(
                    e["id"],
                    ListEntry(
                        entry_id=e["id"],
                        media_id=media["id"],
                        mal_id=media.get("idMal"),
                        status=e.get("status") or "",
                        score100=float(e.get("score") or 0),
                        progress=e.get("progress"),
                        repeat=int(e.get("repeat") or 0),
                        episodes=media.get("episodes"),
                        updated_at=e.get("updatedAt") or None,
                    ),
                )
        return list(by_entry_id.values())

    def fetch_favourites(self, username: str) -> set[int]:
        """Favourited AniList media ids (≤ MAX_FAVOURITE_PAGES requests)."""
        favourites: set[int] = set()
        for page in range(1, MAX_FAVOURITE_PAGES + 1):
            data = self._query(FAVOURITES_QUERY, {"name": username, "page": page})
            user = data.get("User")
            if user is None:
                raise UnknownUserError(f"User not found: {username}")
            anime = (user.get("favourites") or {}).get("anime") or {}
            favourites.update(n["id"] for n in anime.get("nodes") or [])
            if not (anime.get("pageInfo") or {}).get("hasNextPage"):
                break
        return favourites

    def fetch_user(self, username: str) -> UserAnimeList:
        """List + favourites; ≤ ~4 AniList requests total."""
        return UserAnimeList(
            entries=self.fetch_list(username),
            favourite_media_ids=self.fetch_favourites(username),
        )
