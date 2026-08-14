# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anilist-rec",
# ]
#
# [tool.uv.sources]
# anilist-rec = { path = "..", editable = true }
# ///
"""Vibe check (SPEC §5, issue #19): the owner's list through the real serving path.

    uvx marimo edit notebooks/vibe_check.py

Pulls a public AniList list, folds it in, and renders the top-20 at three dial
settings (off / shipped default / high-novelty) with covers, genres, and a
why-this line. Each rec gets a label — seen-elsewhere / would-watch / plausible
/ bad / broken — and the tallies export to reports/vibe_check_<model>.md.

Judged as a **2022-era time capsule** (SPEC §3): the corpus ends 2022-03-22,
so "does it know my taste", not "is it current" — staleness is not model
failure.
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
        # Vibe check — the export's eyes on your own list

        [Issue #19](https://github.com/Ari-03/AniList_Rec/issues/19) ·
        [SPEC §5](https://github.com/Ari-03/AniList_Rec/blob/main/SPEC.md).
        Ranked through the same serving path the container runs. Label every
        rec, then export the tallies. **Time-capsule caveat:** nothing after
        2022-03-22 exists here.
        """
    )
    return


@app.cell
def _():
    LABELS = ["seen-elsewhere", "would-watch", "plausible", "bad", "broken"]
    SHIPPED_DEFAULT_DIAL = 0.15  # from the §5 validation sweep (reports/eval.md)
    HIGH_NOVELTY_DIAL = 1.0
    TOP_K = 20
    return HIGH_NOVELTY_DIAL, LABELS, SHIPPED_DEFAULT_DIAL, TOP_K


@app.cell
def _(mo):
    from pathlib import Path

    import numpy as np
    import polars as pl

    from anilist_rec.als import als_artifact_path
    from anilist_rec.config import Config
    from anilist_rec.ease import ease_artifact_path
    from anilist_rec.sasrec import sasrec_artifact_path

    repo_root = Path(__file__).resolve().parent.parent
    cfg = Config(data_dir=repo_root / "data")
    catalogue = pl.read_parquet(cfg.crosswalk_path)

    available = {"bm25 (baseline)": "bm25"}
    for _label, _arch, _path in [
        ("EASE", "ease", ease_artifact_path(cfg)),
        ("ALS", "als", als_artifact_path(cfg)),
        ("SASRec", "sasrec", sasrec_artifact_path(cfg)),
    ]:
        if _path.exists():
            available[_label] = _arch
    artifact_paths = {
        "bm25": cfg.similarity_path,
        "ease": ease_artifact_path(cfg),
        "als": als_artifact_path(cfg),
        "sasrec": sasrec_artifact_path(cfg),
    }
    mo.md(f"Artifacts on disk: **{', '.join(available)}** (data dir `{cfg.data_dir}`)")
    return artifact_paths, available, catalogue, cfg, np, repo_root


@app.cell
def _(available, mo):
    username = mo.ui.text(value="Zackhacks", label="AniList username")
    model = mo.ui.dropdown(
        options=list(available), value=list(available)[-1], label="finalist"
    )
    run = mo.ui.run_button(label="Fetch list and rank")
    mo.hstack([username, model, run], justify="start")
    return model, run, username


@app.cell
def _(artifact_paths, available, catalogue, cfg, mo, model, np, run):
    mo.stop(not run.value, mo.md("*Press **Fetch list and rank** to start.*"))

    import scipy.sparse as sp

    from anilist_rec.bundle import _load_scorer
    from anilist_rec.franchise import build_franchise_index
    from anilist_rec.matrix import item_index
    from anilist_rec.models import bm25_scorer
    from anilist_rec.serve import Recommender
    from anilist_rec.signals import build_signals

    arch = available[model.value]
    item_ids = item_index(build_signals(cfg))
    artifact_path = artifact_paths[arch]
    score_fn = (
        bm25_scorer(sp.load_npz(artifact_path))
        if arch == "bm25"
        else _load_scorer(arch, artifact_path, len(item_ids))
    )
    recommender = Recommender(
        score_fn,
        item_ids,
        build_franchise_index(catalogue, item_ids),
        np.load(cfg.item_counts_path),
        catalogue,
    )
    return arch, artifact_path, item_ids, recommender, sp


@app.cell
def _(arch, artifact_path, np, sp):
    # Why-this: which of the fold-in items pull each rec up. Item-item models
    # expose exact contributions (w_i * S[i, j]); SASRec approximates with
    # embedding cosine — "closest items on your list", not attention.
    if arch in ("bm25", "ease"):
        _sim = sp.load_npz(artifact_path)
    elif arch == "sasrec":
        _emb = np.load(artifact_path)["item_emb.weight"][1:]  # drop the padding row
        _sim = _emb / (np.linalg.norm(_emb, axis=1, keepdims=True) + 1e-9)
    else:
        _sim = None
    _dense = _sim if arch == "sasrec" else None
    _sparse = _sim if arch in ("bm25", "ease") else None

    def contributors(fold, rec_item_idx, k=3):
        if (_dense is None and _sparse is None) or not fold.fold_idx:
            return []
        if _dense is not None:
            contrib = np.asarray(fold.fold_w) * (_dense[fold.fold_idx] @ _dense[rec_item_idx])
        else:
            col = np.asarray(_sparse[fold.fold_idx, rec_item_idx].todense()).ravel()
            contrib = np.asarray(fold.fold_w) * col
        order = np.argsort(-contrib)[:k]
        return [fold.fold_idx[i] for i in order if contrib[i] > 0]

    return (contributors,)


@app.cell
def _(mo, recommender, run, username):
    mo.stop(not run.value)
    user_list = recommender.client.fetch_user(username.value)
    fold = recommender.fold_in(user_list)
    mo.md(
        f"**{username.value}**: {fold.n_entries} entries, {len(fold.fold_idx)} folded in, "
        f"{len(fold.plan_idx)} planning, {fold.n_unmapped} outside the corpus "
        f"({fold.n_unmapped / max(fold.n_entries, 1):.1%} — the §3 time-capsule cost)"
    )
    return (fold,)


@app.cell
def _(HIGH_NOVELTY_DIAL, SHIPPED_DEFAULT_DIAL, TOP_K, fold, mo, recommender, run):
    mo.stop(not run.value)
    recs_by_dial = {
        "off": recommender.recommend_foldin(fold, dial=0.0, limit=TOP_K),
        f"default ({SHIPPED_DEFAULT_DIAL:g})": recommender.recommend_foldin(
            fold, dial=SHIPPED_DEFAULT_DIAL, limit=TOP_K
        ),
        f"high-novelty ({HIGH_NOVELTY_DIAL:g})": recommender.recommend_foldin(
            fold, dial=HIGH_NOVELTY_DIAL, limit=TOP_K
        ),
    }
    return (recs_by_dial,)


@app.cell
def _(LABELS, catalogue, contributors, fold, mo, recommender, recs_by_dial, run):
    mo.stop(not run.value)

    _meta = {int(r["idMal"]): r for r in catalogue.drop_nulls("idMal").iter_rows(named=True)}
    _pos_of = {mal: i for i, mal in enumerate(recommender.item_ids.tolist())}

    def _title(mal_id):
        m = _meta.get(mal_id)
        return (m and (m["title_english"] or m["title_romaji"])) or f"MAL {mal_id}"

    def _card(i, rec):
        m = _meta.get(rec.mal_id) or {}
        why = ", ".join(
            _title(int(recommender.item_ids[j]))
            for j in contributors(fold, _pos_of[rec.mal_id])
        )
        genres = ", ".join((m.get("genres") or "").split("|")[:4])
        url = m.get("siteUrl") or f"https://anilist.co/anime/{rec.anilist_id}"
        return (
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0">'
            f'<img src="{m.get("coverImage_medium") or ""}" width="46" '
            f'style="border-radius:4px"/>'
            f'<div><b>{i + 1}.</b> <a href="{url}" target="_blank">'
            f"<b>{_title(rec.mal_id)}</b></a> "
            f'<small>({m.get("seasonYear") or "?"})</small><br/>'
            f"<small>{genres}</small><br/>"
            f'<small><i>because of: {why or "—"}</i></small></div></div>'
        )

    labelers = {
        name: mo.ui.array(
            [mo.ui.dropdown(options=LABELS, label=f"#{i + 1}") for i in range(len(recs))]
        )
        for name, recs in recs_by_dial.items()
    }
    tabs = mo.ui.tabs(
        {
            name: mo.hstack(
                [
                    mo.Html("".join(_card(i, r) for i, r in enumerate(recs))),
                    labelers[name],
                ],
                justify="space-between",
            )
            for name, recs in recs_by_dial.items()
        }
    )
    tabs
    return (labelers,)


@app.cell
def _(LABELS, labelers, mo, run):
    mo.stop(not run.value)
    tallies = {
        name: {label: arr.value.count(label) for label in LABELS}
        for name, arr in labelers.items()
    }
    _rows = [
        "| dial | " + " | ".join(LABELS) + " | unlabeled |",
        "|---" * (len(LABELS) + 2) + "|",
    ]
    for _name, _t in tallies.items():
        _unlabeled = len(labelers[_name].value) - sum(_t.values())
        _rows.append(
            f"| {_name} | " + " | ".join(str(_t[la]) for la in LABELS) + f" | {_unlabeled} |"
        )
    tally_md = "\n".join(_rows)
    mo.md("## Tallies\n\n" + tally_md)
    return tallies, tally_md


@app.cell
def _(arch, mo, run):
    mo.stop(not run.value)
    export = mo.ui.run_button(label=f"Write reports/vibe_check_{arch}.md")
    export
    return (export,)


@app.cell
def _(arch, export, labelers, mo, recs_by_dial, repo_root, run, tallies, tally_md, username):
    mo.stop(not run.value)
    mo.stop(not export.value, mo.md("*label, then export*"))
    from datetime import UTC, datetime

    _ = tallies
    _out = repo_root / "reports" / f"vibe_check_{arch}.md"
    _lines = [
        f"# Vibe check — {arch}",
        "",
        f"Labeled by the owner (`{username.value}`) on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d')} via `notebooks/vibe_check.py` "
        "([#19](https://github.com/Ari-03/AniList_Rec/issues/19)). Judged as a "
        "2022-era time capsule (SPEC §3/§5).",
        "",
        tally_md,
        "",
    ]
    for _name, _recs in recs_by_dial.items():
        _lines += [f"## {_name}", ""]
        _lines += [
            f"- {_r.title or _r.anilist_id} — **{labelers[_name].value[_i] or 'unlabeled'}**"
            for _i, _r in enumerate(_recs)
        ]
        _lines.append("")
    _out.write_text("\n".join(_lines))
    mo.md(f"wrote `{_out}`")
    return


if __name__ == "__main__":
    app.run()
