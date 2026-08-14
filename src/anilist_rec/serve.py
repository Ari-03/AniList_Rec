"""The shared serving path (SPEC §6/§7): username → ranked AniList ids.

Fold-in vector → model scores → franchise entry-point filter → popularity
dial → top-k, mapped back to AniList ids via the crosswalk. Every candidate
architecture serves through this path by supplying a ScoreFn; the export
container (issue #20) wraps `Recommender` in FastAPI.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from anilist_rec.anilist import AniListClient, UserAnimeList
from anilist_rec.evaluate import candidate_mask
from anilist_rec.foldin import FoldinVector, build_foldin
from anilist_rec.franchise import FranchiseIndex
from anilist_rec.matrix import item_positions
from anilist_rec.models import ScoreFn, apply_dial


@dataclass(frozen=True)
class Recommendation:
    anilist_id: int
    mal_id: int
    score: float
    title: str | None  # display metadata for demos/vibe checks; contract emits bare ids


def crosswalk_maps(catalogue: pl.DataFrame) -> tuple[dict[int, int], dict[int, str]]:
    """MAL id → (AniList id, display title), deduped on popularity (SPEC §2)."""
    xw = (
        catalogue.drop_nulls("idMal")
        .sort("popularity", descending=True, nulls_last=True)
        .unique(subset="idMal", keep="first")
    )
    mal_to_anilist = dict(zip(xw["idMal"], xw["id"], strict=True))
    titles = {
        int(m): t or r
        for m, t, r in zip(xw["idMal"], xw["title_english"], xw["title_romaji"], strict=True)
    }
    return {int(k): int(v) for k, v in mal_to_anilist.items()}, titles


class Recommender:
    """One candidate's artifacts behind the serving path; ScoreFn-agnostic."""

    def __init__(
        self,
        score_fn: ScoreFn,
        item_ids: np.ndarray,
        franchise: FranchiseIndex,
        item_counts: np.ndarray,
        catalogue: pl.DataFrame,
        client: AniListClient | None = None,
    ):
        self.score_fn = score_fn
        self.item_ids = item_ids
        self.item_pos = item_positions(item_ids)
        self.franchise = franchise
        self.item_counts = item_counts
        self.mal_to_anilist, self.titles = crosswalk_maps(catalogue)
        self.client = client or AniListClient()

    def fold_in(self, user_list: UserAnimeList) -> FoldinVector:
        return build_foldin(user_list, self.item_pos)

    def recommend_foldin(
        self, fold: FoldinVector, dial: float = 0.0, limit: int = 20
    ) -> list[Recommendation]:
        """The internal scoring layer (SPEC §6 secondary endpoint)."""
        fold_csr = fold.to_csr(len(self.item_ids))
        if getattr(self.score_fn, "takes_batch", False):  # sequence models read fold_idx order
            scores = self.score_fn(fold_csr, [fold])[0]
        else:
            scores = self.score_fn(fold_csr)[0]
        scores = apply_dial(scores, self.item_counts, dial)
        scores[~candidate_mask(self.franchise, fold.watched, fold.plan_idx)] = -np.inf

        recs: list[Recommendation] = []
        for idx in np.argsort(-scores):
            if len(recs) >= limit or scores[idx] == -np.inf:
                break
            mal_id = int(self.item_ids[idx])
            anilist_id = self.mal_to_anilist.get(mal_id)
            if anilist_id is None:
                continue  # not in the crosswalk: can't be surfaced as an AniList id
            recs.append(
                Recommendation(
                    anilist_id=anilist_id,
                    mal_id=mal_id,
                    score=float(scores[idx]),
                    title=self.titles.get(mal_id),
                )
            )
        return recs

    def recommend(
        self, username: str, dial: float = 0.0, limit: int = 20
    ) -> tuple[list[Recommendation], FoldinVector]:
        """The primary endpoint path: fetch the list live, fold in, rank."""
        fold = self.fold_in(self.client.fetch_user(username))
        return self.recommend_foldin(fold, dial=dial, limit=limit), fold
