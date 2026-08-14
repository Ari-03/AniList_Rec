# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anilist-rec",
#     "torch>=2.6",
#     "kagglehub>=0.3",
# ]
#
# [tool.uv.sources]
# anilist-rec = { git = "https://github.com/Ari-03/AniList_Rec", branch = "main" }
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""SASRec training on molab (Ari-03/AniList_Rec#18).

Sync this file into molab ("Create a synced notebook" → paste the GitHub URL)
and run all cells. The sandbox header installs the anilist_rec package from
GitHub with CUDA torch (cu128 — the RTX Pro 6000 is Blackwell, sm_120);
the cells acquire the data (idempotent, ~15 GB disk), train on the GPU, and
render the eval report with download buttons for the artifacts.
"""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # SASRec candidate — molab training run

        [Issue #18](https://github.com/Ari-03/AniList_Rec/issues/18) ·
        [SPEC §4](https://github.com/Ari-03/AniList_Rec/blob/main/SPEC.md).
        Runs the same `uv run sasrec` pipeline the repo ships, on molab's GPU:
        data acquisition → timestamp sanity check → training with early stop on
        validation NDCG@10 → full test eval through the shared serving path.
        """
    )
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none — CPU fallback"
    mo.md(f"**Device:** `{device}` ({gpu})")
    return Path, device, sys


@app.cell
def _(Path, mo):
    import runpy
    import urllib.request

    raw = "https://raw.githubusercontent.com/Ari-03/AniList_Rec/main"
    data_dir = Path("data")
    if not (data_dir / "interactions.parquet").exists():
        # the acquire script writes to <its parent>/../data, so land it in ./scripts
        script = Path("scripts/acquire_data.py")
        script.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{raw}/scripts/acquire_data.py", script)
        runpy.run_path(str(script), run_name="__main__")
    mo.md(f"Data ready under `{data_dir.resolve()}` (kagglehub download is cached).")
    return (data_dir,)


@app.cell
def _(Path, data_dir, device, mo, sys):
    from anilist_rec.sasrec import main as sasrec_main

    report_path = Path("reports/sasrec_molab.md")
    sys.argv = [
        "sasrec",
        "--data-dir", str(data_dir),
        "--report", str(report_path),
        "--device", device,
    ]
    sasrec_main()  # stage lines + epoch curve stream to this cell's console
    mo.md(f"Training + eval finished — report at `{report_path}`.")
    return (report_path,)


@app.cell
def _(mo, report_path):
    mo.md(report_path.read_text())
    return


@app.cell
def _(data_dir, mo, report_path):
    artifact = data_dir / "derived" / "sasrec_seed42.npz"
    mo.hstack(
        [
            mo.download(report_path.read_bytes(), filename="sasrec_molab.md"),
            mo.download(artifact.read_bytes(), filename=artifact.name),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
