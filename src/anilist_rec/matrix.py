"""Training matrix: positive-weight signals from non-holdout users, as user-by-item CSR."""

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.config import Config


def item_index(signals: pl.LazyFrame) -> np.ndarray:
    """The item universe: every MAL id with any signal (SPEC §3), sorted."""
    return (
        signals.select(pl.col("anime_id").unique().sort())
        .collect(engine="streaming")["anime_id"]
        .to_numpy()
    )


def item_positions(item_ids: np.ndarray) -> dict[int, int]:
    """MAL id → row index in every item-space array."""
    return {int(a): i for i, a in enumerate(item_ids)}


def build_training_matrix(
    cfg: Config,
    signals: pl.LazyFrame,
    item_ids: np.ndarray,
    holdout_users: set[str],
    signed: bool = False,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Returns (X_train user-by-item CSR, per-item positive training user counts).

    Training pool = every user outside the holdout with ≥1 positive-weight row;
    cfg.train_user_cap seed-samples that pool (dev speed only — None for real runs).
    `signed=True` keeps NEG rows as negative values for candidates that model
    negatives (SPEC §4); item counts stay positive-only either way (popularity
    for the dial and guardrails means positive interactions).
    """
    weight_filter = pl.col("weight") != 0 if signed else pl.col("weight") > 0
    triples = signals.filter(weight_filter).filter(
        ~pl.col("user_id").is_in(sorted(holdout_users))
    )
    if cfg.train_user_cap is not None:
        pool = (
            triples.select(pl.col("user_id").unique().sort())
            .collect(engine="streaming")["user_id"]
            .to_numpy()
        )
        rng = np.random.default_rng(cfg.seed)
        pool = pool[rng.permutation(len(pool))][: cfg.train_user_cap]
        triples = triples.filter(pl.col("user_id").is_in(pool.tolist()))

    df = triples.select("user_id", "anime_id", "weight").collect(engine="streaming")
    user_codes = df["user_id"].cast(pl.Categorical).to_physical().to_numpy()
    item_codes = np.searchsorted(item_ids, df["anime_id"].to_numpy()).astype(np.int32)

    n_items = len(item_ids)
    weights = df["weight"].to_numpy()
    x_train = sp.csr_matrix(
        (weights, (user_codes, item_codes)),
        shape=(int(user_codes.max()) + 1, n_items),
    )
    item_counts = np.bincount(
        item_codes[weights > 0] if signed else item_codes, minlength=n_items
    ).astype(np.float64)
    return x_train, item_counts
