"""Franchise clusters and entry-point collapse (SPEC §1).

"Direct continuation" is approximated as same franchise cluster: union-find over
the AniList relations graph, then one canonical entry point per cluster
(prefer TV, then earliest season, then popularity), restricted to corpus items.
"""

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from anilist_rec.matrix import item_positions

# Relation types that tie two anime into one franchise (CHARACTER/ADAPTATION/etc. don't).
RELATION_TYPES = {
    "SEQUEL",
    "PREQUEL",
    "PARENT",
    "SIDE_STORY",
    "SUMMARY",
    "ALTERNATIVE",
    "SPIN_OFF",
    "FULL_STORY",
    "COMPILATION",
}


def franchise_clusters(catalogue: pl.DataFrame) -> dict[int, int]:
    """AniList id → franchise root, via union-find over anime-anime relation edges."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for aid, relations in zip(catalogue["id"], catalogue["relations"], strict=True):
        if not relations or relations == "[]":
            continue
        for edge in json.loads(relations):
            node = edge.get("node") or {}
            if edge.get("relationType") in RELATION_TYPES and node.get("type") == "ANIME":
                ra, rb = find(aid), find(node["id"])
                if ra != rb:
                    parent[ra] = rb

    return {int(aid): find(int(aid)) for aid in catalogue["id"]}


@dataclass(frozen=True)
class FranchiseIndex:
    """Franchise structure in corpus item-index space (parallel to the item_ids array)."""

    item_franchise: np.ndarray  # cluster label per item; negative = synthetic singleton
    is_entry: np.ndarray  # True where the item is its franchise's entry point
    entry_of_franchise: dict[int, int]  # cluster label → entry item index


def build_franchise_index(catalogue: pl.DataFrame, item_ids: np.ndarray) -> FranchiseIndex:
    """Project franchise clusters onto the corpus items (MAL ids in `item_ids`)."""
    roots = franchise_clusters(catalogue)
    n_items = len(item_ids)
    item_pos = item_positions(item_ids)

    # dedupe the MAL side of the crosswalk on popularity (SPEC §2)
    xw = (
        catalogue.drop_nulls("idMal")
        .sort("popularity", descending=True, nulls_last=True)
        .unique(subset="idMal", keep="first")
    )
    mal_to_row = {int(m): r for r, m in enumerate(xw["idMal"]) if int(m) in item_pos}

    # corpus items missing from the catalogue get unique negative singleton labels
    item_franchise = -np.arange(1, n_items + 1, dtype=np.int64)
    for mal, row in mal_to_row.items():
        item_franchise[item_pos[mal]] = roots.get(int(xw["id"][row]), -item_pos[mal] - 1)

    # entry point per franchise among corpus items: TV first, then earliest, then popular
    sortkey = {
        item_pos[mal]: (
            0 if xw["format"][row] == "TV" else 1,
            xw["seasonYear"][row] or 9999,
            -(xw["popularity"][row] or 0),
        )
        for mal, row in mal_to_row.items()
    }
    best: dict[int, tuple[tuple[int, int, int], int]] = {}
    for idx in range(n_items):
        cluster = int(item_franchise[idx])
        key = sortkey.get(idx, (2, 9999, 0))
        if cluster not in best or key < best[cluster][0]:
            best[cluster] = (key, idx)

    is_entry = np.zeros(n_items, dtype=bool)
    for _key, idx in best.values():
        is_entry[idx] = True
    entry_of_franchise = {cluster: idx for cluster, (_key, idx) in best.items()}

    return FranchiseIndex(item_franchise, is_entry, entry_of_franchise)
