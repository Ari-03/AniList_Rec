"""Bundle round-trip: write_bundle → load_bundle behind the serving path."""

import numpy as np
import pytest

from anilist_rec.bundle import Manifest, load_bundle, write_bundle
from anilist_rec.foldin import FoldinVector


def test_load_bundle_serves(bundle_dir):
    loaded = load_bundle(bundle_dir)
    assert loaded.manifest.architecture == "bm25"
    assert loaded.manifest.model_version == "0.0.0+fixture.bm25"

    # fold item index 2 (MAL 30): its similarity row makes AniList 1000 the top rec
    fold = FoldinVector(
        fold_idx=[2], fold_w=[1.0], watched=[2], plan_idx=[], n_entries=1, n_unmapped=0
    )
    recs = loaded.recommender.recommend_foldin(fold)
    assert recs[0].anilist_id == 1000


def test_franchise_collapse_survives_round_trip(bundle_dir):
    loaded = load_bundle(bundle_dir)
    # watched MAL 20 (index 1) kills its franchise partner 10; PLANNING kills 30
    fold = FoldinVector(
        fold_idx=[1], fold_w=[1.0], watched=[1], plan_idx=[2], n_entries=2, n_unmapped=0
    )
    assert [r.anilist_id for r in loaded.recommender.recommend_foldin(fold)] == [4000]


def test_unknown_architecture_rejected(tmp_path, bundle_dir):
    manifest = Manifest(
        model_version="x",
        architecture="word2vec",
        dial_default=0.0,
        seed=0,
        created_utc="2026-01-01T00:00:00Z",
    )
    with pytest.raises(ValueError, match="unknown architecture"):
        write_bundle(
            tmp_path / "out",
            manifest,
            item_ids=np.array([1]),
            item_counts=np.array([1.0]),
            catalogue=None,
            model_artifact=bundle_dir / "model" / "similarity.npz",
        )
