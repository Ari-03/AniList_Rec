"""AniList list → corpus-mapped weighted fold-in vector (SPEC §1 + §3, issue #15).

The serve-time mirror of signals.py: same §1 mapping, but over AniList's
statuses (CURRENT/REPEATING exist here; svanoo had no REPEATING) with
POINT_100 scores. Entries that don't map into the training corpus are
dropped before fold-in (SPEC §3); PLANNING never folds in but still blocks
recommendation output.
"""

from dataclasses import dataclass

import scipy.sparse as sp

from anilist_rec.anilist import ListEntry, UserAnimeList
from anilist_rec.signals import NEG, PARTIAL, PLAN, STD, STRONG


def entry_signal(entry: ListEntry, favourited: bool) -> tuple[int, float]:
    """(kind, positive-preference weight) for one entry — signals.py order, §1 semantics."""
    score = entry.score100  # 0 = unrated
    if entry.status == "PLANNING":
        return PLAN, 0.0
    if favourited or entry.status == "REPEATING" or entry.repeat > 0:
        return STRONG, 2.0
    if entry.status == "DROPPED" or (entry.status == "COMPLETED" and 1 <= score <= 40):
        return NEG, 0.0
    if entry.status == "COMPLETED" and score >= 80:
        return STRONG, 2.0
    if entry.status == "COMPLETED":
        return STD, 1.0
    if entry.status == "CURRENT":
        # confidence scaled by progress/episodes; midpoint when unknown
        if entry.progress is not None and entry.episodes:
            ratio = min(max(entry.progress / entry.episodes, 0.0), 1.0)
        else:
            ratio = 0.5
        return PARTIAL, 0.5 + 0.5 * ratio
    if entry.status == "PAUSED":
        return PARTIAL, 0.25
    return PLAN, 0.0  # unknown status: never train on it, never recommend it


@dataclass(frozen=True)
class FoldinVector:
    """A user's list in corpus item-index space — same shape eval users take."""

    fold_idx: list[int]  # positive-weight fold-in input
    fold_w: list[float]
    watched: list[int]  # every mapped non-PLANNING entry (output exclusion)
    plan_idx: list[int]  # mapped PLANNING entries (also excluded from output)
    n_entries: int  # deduped list size
    n_unmapped: int  # entries dropped for not mapping into the corpus (SPEC §3)

    def to_csr(self, n_items: int) -> sp.csr_matrix:
        return sp.csr_matrix(
            (self.fold_w, ([0] * len(self.fold_idx), self.fold_idx)),
            shape=(1, n_items),
            dtype="float32",
        )


def build_foldin(user_list: UserAnimeList, item_pos: dict[int, int]) -> FoldinVector:
    """Map a fetched list onto the corpus; ids outside it drop (SPEC §3).

    Entries are walked in list-edit order (updatedAt ascending, undated first —
    the split.py convention), so fold_idx doubles as the sequence input for
    order-aware models; bag-of-items models see the same set either way.
    """
    fold_idx, fold_w, watched, plan_idx = [], [], [], []
    n_unmapped = 0
    ordered = sorted(
        user_list.entries, key=lambda e: (e.updated_at is not None, e.updated_at or 0)
    )
    for entry in ordered:
        idx = item_pos.get(entry.mal_id) if entry.mal_id is not None else None
        if idx is None:
            n_unmapped += 1
            continue
        kind, weight = entry_signal(entry, entry.media_id in user_list.favourite_media_ids)
        if kind == PLAN:
            plan_idx.append(idx)
            continue
        watched.append(idx)
        if weight > 0:
            fold_idx.append(idx)
            fold_w.append(weight)
    return FoldinVector(
        fold_idx=fold_idx,
        fold_w=fold_w,
        watched=watched,
        plan_idx=plan_idx,
        n_entries=len(user_list.entries),
        n_unmapped=n_unmapped,
    )
