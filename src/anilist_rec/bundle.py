"""The export bundle (SPEC §6, issue #20): model data in open formats.

A bundle is a self-contained directory the scoring container bakes in —
everything the serving path needs with no training data and no raw user rows
(SPEC §2 licensing guardrail):

    bundle/
      manifest.json     model_version, architecture, dial_default, provenance
      item_ids.npy      the corpus item universe (sorted MAL ids)
      item_counts.npy   positive training user counts (dial + guardrails)
      catalogue.parquet AniList catalogue + MAL crosswalk (franchise relations,
                        entry-point metadata, display titles)
      model/            architecture-specific artifact files

`write_bundle` is the pure writer (tests and CI feed it synthetic artifacts);
`export-bundle` (main) assembles a real bundle from data/derived. The loader
dispatches on the manifest's architecture, so swapping the winner (#21) is a
bundle swap — no service code changes.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.anilist import AniListClient
from anilist_rec.config import Config
from anilist_rec.franchise import build_franchise_index
from anilist_rec.matrix import item_index
from anilist_rec.models import ScoreFn, bm25_scorer
from anilist_rec.serve import Recommender
from anilist_rec.signals import build_signals

MANIFEST_NAME = "manifest.json"
CORPUS_CUTOFF = "2022-03-22"  # SPEC §2: the item universe is a time capsule

# architecture → artifact file name under model/
ARTIFACT_FILES = {
    "bm25": "similarity.npz",
    "ease": "ease_B.npz",
    "als": "als_item_factors.npz",
    "sasrec": "sasrec_state.npz",
}


@dataclass(frozen=True)
class Manifest:
    model_version: str
    architecture: str
    dial_default: float
    seed: int
    created_utc: str
    corpus_cutoff: str = CORPUS_CUTOFF

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @staticmethod
    def read(bundle_dir: Path) -> "Manifest":
        return Manifest(**json.loads((bundle_dir / MANIFEST_NAME).read_text()))


def write_bundle(
    out_dir: Path,
    manifest: Manifest,
    item_ids: np.ndarray,
    item_counts: np.ndarray,
    catalogue: pl.DataFrame,
    model_artifact: Path,
) -> None:
    """Lay out a bundle directory; `model_artifact` is copied under model/."""
    if manifest.architecture not in ARTIFACT_FILES:
        raise ValueError(f"unknown architecture: {manifest.architecture}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    (out_dir / MANIFEST_NAME).write_text(manifest.to_json())
    np.save(out_dir / "item_ids.npy", item_ids)
    np.save(out_dir / "item_counts.npy", item_counts)
    catalogue.write_parquet(out_dir / "catalogue.parquet")
    target = out_dir / "model" / ARTIFACT_FILES[manifest.architecture]
    target.write_bytes(model_artifact.read_bytes())


def _load_scorer(architecture: str, artifact: Path, n_items: int) -> ScoreFn:
    if architecture == "bm25":
        return bm25_scorer(sp.load_npz(artifact))
    if architecture == "ease":
        from anilist_rec.ease import ease_scorer

        return ease_scorer(sp.load_npz(artifact))
    if architecture == "als":
        from anilist_rec.als import als_scorer

        data = np.load(artifact)
        return als_scorer(
            data["item_factors"], float(data["regularization"]), float(data["alpha"])
        )
    if architecture == "sasrec":
        import torch

        from anilist_rec.sasrec import make_model, sasrec_scorer

        data = np.load(artifact)
        d, blocks, heads, use_positions = (int(v) for v in data["__config__"])
        model = make_model(n_items, d, blocks, heads, dropout=0.0, seed=0)
        model.use_positions = bool(use_positions)
        state = {k: torch.from_numpy(v) for k, v in data.items() if k != "__config__"}
        model.load_state_dict(state)
        model.eval()
        return sasrec_scorer(model)
    raise ValueError(f"unknown architecture: {architecture}")


@dataclass(frozen=True)
class LoadedBundle:
    recommender: Recommender
    manifest: Manifest


def load_bundle(bundle_dir: Path, client: AniListClient | None = None) -> LoadedBundle:
    """Bundle directory → a ready Recommender behind the shared serving path."""
    manifest = Manifest.read(bundle_dir)
    item_ids = np.load(bundle_dir / "item_ids.npy")
    item_counts = np.load(bundle_dir / "item_counts.npy")
    catalogue = pl.read_parquet(bundle_dir / "catalogue.parquet")
    score_fn = _load_scorer(
        manifest.architecture,
        bundle_dir / "model" / ARTIFACT_FILES[manifest.architecture],
        len(item_ids),
    )
    recommender = Recommender(
        score_fn,
        item_ids,
        build_franchise_index(catalogue, item_ids),
        item_counts,
        catalogue,
        client=client,
    )
    return LoadedBundle(recommender=recommender, manifest=manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("bundle"))
    parser.add_argument("--arch", choices=sorted(ARTIFACT_FILES), default="bm25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dial-default", type=float, default=0.0)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()
    cfg = Config(data_dir=args.data_dir, seed=args.seed)

    from anilist_rec.als import als_artifact_path
    from anilist_rec.ease import ease_artifact_path
    from anilist_rec.sasrec import sasrec_artifact_path

    artifact_of = {
        "bm25": cfg.similarity_path,
        "ease": ease_artifact_path(cfg),
        "als": als_artifact_path(cfg),
        "sasrec": sasrec_artifact_path(cfg),
    }
    artifact = artifact_of[args.arch]
    if not artifact.exists() or not cfg.item_counts_path.exists():
        raise SystemExit(
            f"missing artifacts for {args.arch} (run the training CLI first): {artifact}"
        )

    manifest = Manifest(
        model_version=args.model_version or f"0.1.0+{args.arch}.seed{cfg.seed}",
        architecture=args.arch,
        dial_default=args.dial_default,
        seed=cfg.seed,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_bundle(
        args.out,
        manifest,
        item_ids=item_index(build_signals(cfg)),
        item_counts=np.load(cfg.item_counts_path),
        catalogue=pl.read_parquet(cfg.crosswalk_path),
        model_artifact=artifact,
    )
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {args.out}/ ({manifest.model_version}, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
