"""AniList client: dedupe, favourites pagination, and the typed failures."""

import pytest

from anilist_rec.anilist import (
    AniListClient,
    AniListOutageError,
    PrivateListError,
    RateLimitedError,
    UnknownUserError,
)


def entry(entry_id: int, media_id: int, mal_id: int | None = None, **kw) -> dict:
    return {
        "id": entry_id,
        "status": kw.get("status", "COMPLETED"),
        "score": kw.get("score", 0),
        "progress": kw.get("progress"),
        "repeat": kw.get("repeat", 0),
        "media": {"id": media_id, "idMal": mal_id, "episodes": kw.get("episodes")},
    }


def list_payload(lists: list[list[dict]]) -> dict:
    return {
        "data": {
            "MediaListCollection": {
                "hasNextChunk": None,
                "lists": [{"entries": entries} for entries in lists],
            }
        }
    }


def favourites_payload(ids: list[int], has_next: bool = False) -> dict:
    return {
        "data": {
            "User": {
                "favourites": {
                    "anime": {
                        "pageInfo": {"hasNextPage": has_next},
                        "nodes": [{"id": i} for i in ids],
                    }
                }
            }
        }
    }


def canned(responses: list[tuple[int, dict, dict]]):
    """Transport returning each response once, recording the payloads sent."""
    calls: list[dict] = []

    def transport(payload: dict) -> tuple[int, dict, dict]:
        calls.append(payload)
        return responses[len(calls) - 1]

    return transport, calls


def test_fetch_list_dedupes_across_custom_lists():
    # entry 1 appears on its status list and again on a custom list
    payload = list_payload([[entry(1, 100, 10), entry(2, 200, 20)], [entry(1, 100, 10)]])
    transport, calls = canned([(200, {}, payload)])
    entries = AniListClient(transport).fetch_list("someone")
    assert [e.entry_id for e in entries] == [1, 2]
    assert entries[0].mal_id == 10
    # the query asks the server to normalize scores and collapse completed lists
    assert "POINT_100" in calls[0]["query"]
    assert "forceSingleCompletedList: true" in calls[0]["query"]


def test_score_zero_is_unrated_not_zero():
    payload = list_payload([[entry(1, 100, 10, score=0)]])
    transport, _ = canned([(200, {}, payload)])
    (e,) = AniListClient(transport).fetch_list("someone")
    assert e.score100 == 0.0  # foldin treats 0 as unrated, never "rated zero"


def test_favourites_paginate_and_merge():
    transport, calls = canned(
        [
            (200, {}, favourites_payload([100, 101], has_next=True)),
            (200, {}, favourites_payload([102], has_next=False)),
        ]
    )
    favs = AniListClient(transport).fetch_favourites("someone")
    assert favs == {100, 101, 102}
    assert [c["variables"]["page"] for c in calls] == [1, 2]


def test_favourites_page_budget_is_capped():
    transport, calls = canned([(200, {}, favourites_payload([1], has_next=True))] * 10)
    AniListClient(transport).fetch_favourites("someone")
    assert len(calls) == 3  # ≤ ~4 requests per fetch_user incl. the list call


def test_unknown_user_is_typed():
    body = {"data": None, "errors": [{"message": "User not found", "status": 404}]}
    transport, _ = canned([(404, {}, body)])
    with pytest.raises(UnknownUserError):
        AniListClient(transport).fetch_list("nosuchuser")


def test_private_list_is_typed():
    body = {"data": None, "errors": [{"message": "Private User", "status": 404}]}
    transport, _ = canned([(404, {}, body)])
    with pytest.raises(PrivateListError):
        AniListClient(transport).fetch_list("privateuser")


def test_rate_limit_carries_retry_after():
    body = {"data": None, "errors": [{"message": "Too Many Requests.", "status": 429}]}
    transport, _ = canned([(429, {"Retry-After": "42"}, body)])
    with pytest.raises(RateLimitedError) as exc:
        AniListClient(transport).fetch_list("someone")
    assert exc.value.retry_after == 42


def test_server_error_is_outage():
    transport, _ = canned([(500, {}, {})])
    with pytest.raises(AniListOutageError):
        AniListClient(transport).fetch_list("someone")


def test_api_disabled_403_is_outage():
    transport, _ = canned([(403, {}, {})])
    with pytest.raises(AniListOutageError):
        AniListClient(transport).fetch_list("someone")


def test_errors_on_http_200_are_not_ignored():
    body = {"data": None, "errors": [{"message": "Internal Server Error", "status": 500}]}
    transport, _ = canned([(200, {}, body)])
    with pytest.raises(AniListOutageError):
        AniListClient(transport).fetch_list("someone")


def test_fetch_user_costs_at_most_four_requests():
    transport, calls = canned(
        [
            (200, {}, list_payload([[entry(1, 100, 10)]])),
            (200, {}, favourites_payload([100], has_next=False)),
        ]
    )
    user = AniListClient(transport).fetch_user("someone")
    assert len(calls) == 2
    assert user.favourite_media_ids == {100}
