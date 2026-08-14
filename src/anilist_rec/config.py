"""Pipeline configuration and on-disk layout.

Derived artifacts are cached on disk keyed by the values meant to vary between
runs (seed, training cap, K). The SPEC §5 protocol constants (holdout sizes,
fold fraction) are not in the cache keys — changing those means clearing
data/derived/.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Knobs for the offline pipeline (defaults = the SPEC §5 protocol, uncapped)."""

    data_dir: Path
    seed: int = 42
    n_test: int = 10_000
    n_val: int = 10_000
    holdout_candidates: int = 60_000
    train_user_cap: int | None = None  # None = every non-holdout user (SPEC §9 uncap)
    k_neighbors: int = 200
    fold_fraction: float = 0.8

    @property
    def interactions_path(self) -> Path:
        return self.data_dir / "interactions.parquet"

    @property
    def crosswalk_path(self) -> Path:
        return self.data_dir / "crosswalk_anilist_mal.parquet"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def signals_path(self) -> Path:
        return self.derived_dir / "signals.parquet"

    @property
    def holdout_path(self) -> Path:
        return self.derived_dir / f"holdout_seed{self.seed}.parquet"

    @property
    def similarity_path(self) -> Path:
        cap = self.train_user_cap if self.train_user_cap is not None else "none"
        return self.derived_dir / f"sim_bm25_K{self.k_neighbors}_cap{cap}_seed{self.seed}.npz"

    @property
    def item_counts_path(self) -> Path:
        cap = self.train_user_cap if self.train_user_cap is not None else "none"
        return self.derived_dir / f"item_counts_cap{cap}_seed{self.seed}.npy"
