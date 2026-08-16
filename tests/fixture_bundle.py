"""A tiny synthetic bundle for service tests and the CI container smoke test.

Four-item catalogue: MAL 10/20 are one franchise (entry: AniList 1000), 30 and
40 are singletons. The BM25 similarity rows are hand-picked so dial behaviour
is observable: folding in item 30 scores the popular franchise entry highest
at dial 0, and the niche item 40 highest at dial 1.

    uv run python tests/fixture_bundle.py <out_dir>   # CI smoke bundle
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.bundle import Manifest, write_bundle

FIXTURE_MODEL_VERSION = "0.0.0+fixture.bm25"


def make_fixture_bundle(out_dir: Path, dial_default: float = 0.0) -> Path:
    relations = {
        1000: [{"relationType": "SEQUEL", "node": {"id": 2000, "type": "ANIME"}}],
        2000: [{"relationType": "PREQUEL", "node": {"id": 1000, "type": "ANIME"}}],
    }
    catalogue = pl.DataFrame(
        {
            "id": [1000, 2000, 3000, 4000],
            "idMal": [10, 20, 30, 40],
            "title_english": ["A", "A Season 2", "B", "C"],
            "title_romaji": ["A", "A2", "B", "C"],
            "popularity": [1000, 500, 100, 10],
            "relations": [json.dumps(relations.get(i, [])) for i in [1000, 2000, 3000, 4000]],
            "format": ["TV", "TV", "TV", "TV"],
            "seasonYear": [2010, 2012, 2015, 2018],
            "episodes": [12, 12, 24, 12],
        },
        schema_overrides={"id": pl.Int32, "idMal": pl.Int32},
    )
    # rows are the folded-in item; columns the scored catalogue
    similarity = np.array(
        [
            [0.0, 1.0, 0.5, 0.1],
            [1.0, 0.0, 0.5, 0.1],
            [4.0, 3.0, 0.0, 3.9],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    model_path = out_dir / "similarity_src.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(model_path, sp.csr_matrix(similarity))
    write_bundle(
        out_dir,
        Manifest(
            model_version=FIXTURE_MODEL_VERSION,
            architecture="bm25",
            dial_default=dial_default,
            seed=0,
            created_utc="2026-01-01T00:00:00Z",
        ),
        item_ids=np.array([10, 20, 30, 40]),
        item_counts=np.array([100.0, 10.0, 5.0, 1.0]),
        catalogue=catalogue,
        model_artifact=model_path,
    )
    model_path.unlink()  # only the copy under model/ belongs to the bundle
    return out_dir


if __name__ == "__main__":
    print(make_fixture_bundle(Path(sys.argv[1])))
