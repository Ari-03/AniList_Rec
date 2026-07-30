# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "scipy",
#     "implicit",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # PROTOTYPE — baseline recommender thin slice

    **Throwaway code** answering [AniList_Rec#9](https://github.com/Ari-03/AniList_Rec/issues/9):
    does an end-to-end slice hold up? Item-item **BM25** (`implicit`) per the
    [architecture shortlist](https://github.com/Ari-03/AniList_Rec/issues/8), trained on the
    svanoo dump, evaluated per the
    [evaluation protocol](https://github.com/Ari-03/AniList_Rec/issues/7), with a live
    AniList fold-in at the bottom.

    Prototype simplifications (call out in reaction if any matter):
    training users capped (config below); BM25 takes positives only, so DROPPED /
    low-score negatives inform eval but not the model; svanoo has no REPEATING status,
    so strong positives = favourited or score ≥ 8; franchise "direct continuation" ≈
    same franchise cluster; favourites not fetched from AniList at serve time.
    """)
    return


@app.cell
def _():
    import json
    import urllib.request
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    import scipy.sparse as sp

    return Path, json, mo, np, pl, sp, urllib


@app.cell
def _(Path, mo):
    ROOT = Path(mo.notebook_dir()).resolve().parent
    DATA = ROOT / "data"
    DERIVED = DATA / "derived"

    SEED = 42
    N_TEST = 10_000
    N_VAL = 10_000
    HOLDOUT_CANDIDATES = 60_000
    TRAIN_USER_CAP = 300_000  # None = all ~1M non-holdout users (slower fit)
    K_NEIGHBORS = 200
    FOLD_FRACTION = 0.8

    # signal kinds (from "Define a good recommendation" #5, adapted to MAL statuses)
    NEG, STD, STRONG, PARTIAL, PLAN = 0, 1, 2, 3, 4
    GAIN = {STRONG: 2.0, STD: 1.0}

    DIAL_PRESETS = {"off": 0.0, "default": 0.25, "high novelty": 0.5}
    return (
        DATA,
        DERIVED,
        DIAL_PRESETS,
        FOLD_FRACTION,
        GAIN,
        HOLDOUT_CANDIDATES,
        K_NEIGHBORS,
        NEG,
        N_TEST,
        N_VAL,
        PARTIAL,
        PLAN,
        SEED,
        STD,
        STRONG,
        TRAIN_USER_CAP,
    )


@app.cell
def _(DATA, json, pl):
    _cat_csv = (
        DATA
        / "kagglehub/datasets/calebmwelsh/anilist-anime-dataset/versions/90/anilist_anime_data_complete.csv"
    )
    catalogue = (
        pl.scan_csv(_cat_csv, infer_schema_length=10000)
        .filter(pl.col("type") == "ANIME")
        .select(
            pl.col("id").cast(pl.Int32),
            pl.col("idMal").cast(pl.Int32),
            "title_romaji",
            "title_english",
            "format",
            pl.col("seasonYear").cast(pl.Int32, strict=False),
            pl.col("episodes").cast(pl.Float64, strict=False),
            pl.col("popularity").cast(pl.Int64, strict=False),
            "genres",
            "coverImage_medium",
            "siteUrl",
            "relations",
        )
        .collect()
    )

    # franchise clusters: union-find over anime-anime relation edges
    _REL = {"SEQUEL", "PREQUEL", "PARENT", "SIDE_STORY", "SUMMARY", "ALTERNATIVE", "SPIN_OFF", "FULL_STORY", "COMPILATION"}
    _parent = {}

    def _find(x):
        while _parent.setdefault(x, x) != x:
            _parent[x] = _parent[_parent[x]]
            x = _parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            _parent[ra] = rb

    for _aid, _rel in zip(catalogue["id"], catalogue["relations"]):
        if not _rel or _rel == "[]":
            continue
        for _e in json.loads(_rel):
            _node = _e.get("node") or {}
            if _e.get("relationType") in _REL and _node.get("type") == "ANIME":
                _union(_aid, _node["id"])

    franchise_of_anilist = {aid: _find(aid) for aid in catalogue["id"]}
    return catalogue, franchise_of_anilist


@app.cell
def _(DATA, DERIVED, NEG, PARTIAL, PLAN, STD, STRONG, catalogue, pl):
    # derived signal table: one row per interaction, mapped per decision #5
    signals_path = DERIVED / "signals.parquet"

    if not signals_path.exists():
        _eps = (
            catalogue.group_by("idMal")
            .agg(pl.col("episodes").max())
            .drop_nulls("idMal")
            .lazy()
        )
        _fav = pl.col("favorite") == 1
        _score = pl.col("score")
        _status = pl.col("status").cast(pl.String)
        _kind = (
            pl.when(_status == "plan_to_watch").then(PLAN)
            .when(_fav).then(STRONG)
            .when((_status == "dropped") | ((_status == "completed") & _score.is_between(1, 4))).then(NEG)
            .when((_status == "completed") & (_score >= 8)).then(STRONG)
            .when(_status == "completed").then(STD)
            .when(_status.is_in(["watching", "on_hold"])).then(PARTIAL)
        )
        _watch_w = 0.5 + 0.5 * (
            (pl.col("progress") / pl.col("episodes")).clip(0.0, 1.0).fill_null(0.5)
        )
        _weight = (
            pl.when(pl.col("kind") == STRONG).then(2.0)
            .when(pl.col("kind") == STD).then(1.0)
            .when((pl.col("kind") == PARTIAL) & (_status == "watching")).then(_watch_w)
            .when(pl.col("kind") == PARTIAL).then(0.25)
            .otherwise(0.0)
        )
        (
            pl.scan_parquet(DATA / "interactions.parquet")
            .join(_eps, left_on="anime_id", right_on="idMal", how="left")
            .with_columns(_kind.alias("kind"))
            .filter(pl.col("kind").is_not_null())
            .with_columns(_weight.cast(pl.Float32).alias("weight"))
            .select(
                "user_id",
                "anime_id",
                pl.col("kind").cast(pl.UInt8),
                "weight",
                pl.col("last_interaction_date").alias("ts"),
            )
            .sink_parquet(signals_path, compression="zstd")
        )

    signals = pl.scan_parquet(signals_path)
    return (signals,)


@app.cell
def _(
    DERIVED,
    FOLD_FRACTION,
    HOLDOUT_CANDIDATES,
    N_TEST,
    N_VAL,
    PLAN,
    SEED,
    STD,
    STRONG,
    np,
    pl,
    signals,
):
    # held-out users, temporal 80/20 within each (eval protocol #7)
    holdout_path = DERIVED / f"holdout_seed{SEED}.parquet"

    if not holdout_path.exists():
        _counts = (
            signals.filter(pl.col("kind") != PLAN)
            .group_by("user_id")
            .len()
            .filter(pl.col("len") >= 20)
            .collect(engine="streaming")
        )
        _rng = np.random.default_rng(SEED)
        _cand = _counts["user_id"].to_numpy()
        _cand = _cand[_rng.permutation(len(_cand))][:HOLDOUT_CANDIDATES]

        _rows = (
            signals.filter(pl.col("user_id").is_in(_cand.tolist()))
            .collect(engine="streaming")
            .sort(["user_id", "ts", "anime_id"])
            .with_columns(
                (pl.col("kind") != PLAN).cum_sum().over("user_id").alias("rank"),
                (pl.col("kind") != PLAN).sum().over("user_id").alias("n"),
            )
            .with_columns(
                (
                    (pl.col("kind") != PLAN)
                    & (pl.col("rank") <= (FOLD_FRACTION * pl.col("n")).floor())
                ).alias("fold")
            )
        )
        _stats = _rows.group_by("user_id").agg(
            ((~pl.col("fold")) & pl.col("kind").is_in([STD, STRONG])).sum().alias("wpos"),
            (pl.col("fold") & (pl.col("weight") > 0)).sum().alias("fpos"),
        )
        _ok = set(_stats.filter((pl.col("wpos") >= 5) & (pl.col("fpos") >= 1))["user_id"])
        _ordered = [u for u in _cand.tolist() if u in _ok]
        _role = pl.DataFrame(
            {
                "user_id": _ordered[: N_TEST + N_VAL],
                "role": ["test"] * min(N_TEST, len(_ordered))
                + ["val"] * max(0, min(N_VAL, len(_ordered) - N_TEST)),
            }
        )
        _rows.join(_role, on="user_id", how="inner").select(
            "user_id", "role", "anime_id", "kind", "weight", "fold"
        ).write_parquet(holdout_path)

    holdout = pl.read_parquet(holdout_path)
    holdout_users = set(holdout["user_id"].unique())
    return holdout, holdout_users


@app.cell
def _(PLAN, SEED, TRAIN_USER_CAP, holdout_users, np, pl, signals, sp):
    # training matrix: positives from non-holdout users (capped for prototype speed)
    item_ids = (
        signals.select(pl.col("anime_id").unique().sort())
        .collect(engine="streaming")["anime_id"]
        .to_numpy()
    )
    item_pos = {int(a): i for i, a in enumerate(item_ids)}
    n_items = len(item_ids)

    _users = (
        signals.filter(pl.col("kind") != PLAN)
        .select(pl.col("user_id").unique())
        .collect(engine="streaming")["user_id"]
        .to_numpy()
    )
    _users = np.array([u for u in _users if u not in holdout_users])
    _rng = np.random.default_rng(SEED)
    _users = _users[_rng.permutation(len(_users))]
    if TRAIN_USER_CAP:
        _users = _users[:TRAIN_USER_CAP]
    _uset = pl.Series("user_id", _users)

    _tri = (
        signals.filter(pl.col("weight") > 0)
        .filter(pl.col("user_id").is_in(_uset.implode()))
        .select("user_id", "anime_id", "weight")
        .collect(engine="streaming")
    )
    _ucodes = _tri["user_id"].cast(pl.Categorical).to_physical().to_numpy()
    _icodes = np.searchsorted(item_ids, _tri["anime_id"].to_numpy())
    X_train = sp.csr_matrix(
        (_tri["weight"].to_numpy(), (_ucodes, _icodes)),
        shape=(int(_ucodes.max()) + 1, n_items),
    )
    item_counts = np.asarray((X_train > 0).sum(axis=0)).ravel().astype(np.float64)
    return X_train, item_counts, item_ids, item_pos, n_items


@app.cell
def _(DERIVED, K_NEIGHBORS, SEED, TRAIN_USER_CAP, X_train, sp):
    # fit item-item BM25 (implicit); cache the similarity matrix
    _sim_path = DERIVED / f"sim_bm25_K{K_NEIGHBORS}_cap{TRAIN_USER_CAP}_seed{SEED}.npz"

    if _sim_path.exists():
        S = sp.load_npz(_sim_path)
    else:
        from implicit.nearest_neighbours import BM25Recommender

        _model = BM25Recommender(K=K_NEIGHBORS, K1=1.2, B=0.75)
        _model.fit(X_train)
        S = _model.similarity.tocsr().astype("float32")
        sp.save_npz(_sim_path, S)
    return (S,)


@app.cell
def _(catalogue, franchise_of_anilist, item_ids, item_pos, n_items, np, pl):
    # franchise arrays in item-index space + per-item metadata for display
    _xw = (
        catalogue.drop_nulls("idMal")
        .sort("popularity", descending=True, nulls_last=True)
        .unique(subset="idMal", keep="first")
    )
    _mal_to_row = {int(m): r for r, m in enumerate(_xw["idMal"]) if int(m) in item_pos}

    item_franchise = np.arange(n_items) * -1 - 1  # singleton by default (unique negatives)
    for _mal, _r in _mal_to_row.items():
        item_franchise[item_pos[_mal]] = franchise_of_anilist.get(
            int(_xw["id"][_r]), -item_pos[_mal] - 1
        )

    # entry point per franchise, restricted to items present in the corpus:
    # prefer TV, then earliest season year, then catalogue popularity
    _sortkey = {}
    for _mal, _r in _mal_to_row.items():
        _sortkey[item_pos[_mal]] = (
            0 if _xw["format"][_r] == "TV" else 1,
            _xw["seasonYear"][_r] or 9999,
            -(_xw["popularity"][_r] or 0),
        )
    _best = {}
    for _idx in range(n_items):
        _f = int(item_franchise[_idx])
        _key = _sortkey.get(_idx, (2, 9999, 0))
        if _f not in _best or _key < _best[_f][0]:
            _best[_f] = (_key, _idx)
    is_entry = np.zeros(n_items, dtype=bool)
    for _f in _best:
        is_entry[_best[_f][1]] = True
    entry_of_franchise = {f: idx for f, (_k, idx) in _best.items()}

    item_meta = (
        pl.DataFrame({"anime_id": item_ids.astype("int32")})
        .with_row_index("iidx")
        .join(
            _xw.select(
                pl.col("idMal").alias("anime_id"),
                "title_romaji",
                "title_english",
                "genres",
                "coverImage_medium",
                "siteUrl",
            ),
            on="anime_id",
            how="left",
        )
        .sort("iidx")
    )
    return entry_of_franchise, is_entry, item_franchise, item_meta


@app.cell
def _(
    GAIN,
    NEG,
    PLAN,
    STD,
    STRONG,
    entry_of_franchise,
    is_entry,
    item_counts,
    item_franchise,
    item_pos,
    n_items,
    np,
    sp,
):
    # evaluation: full-catalogue ranking through the serving pipeline
    pop_pct = 100.0 * item_counts.argsort().argsort() / (n_items - 1)

    def user_arrays(rows):
        """rows: (anime_id, kind, weight, fold) tuples for one user."""
        fold_idx, fold_w, watched, plan_idx, tgt, neg = [], [], [], [], {}, []
        for aid, kind, w, fold in rows:
            idx = item_pos.get(aid)
            if idx is None:
                continue
            if kind == PLAN:
                plan_idx.append(idx)
            elif fold:
                watched.append(idx)
                if w > 0:
                    fold_idx.append(idx)
                    fold_w.append(w)
            elif kind in (STD, STRONG):
                tgt[idx] = max(tgt.get(idx, 0), GAIN[kind])
            elif kind == NEG:
                neg.append(idx)
        return fold_idx, fold_w, watched, plan_idx, tgt, neg

    def candidate_mask(watched, plan_idx):
        """serving filter: entry points of untouched franchises, minus own lists."""
        mask = is_entry.copy()
        if watched:
            for f in np.unique(item_franchise[watched]):
                mask[item_franchise == f] = False
        mask[watched] = False
        mask[plan_idx] = False
        return mask

    def collapse_targets(tgt, watched):
        """drop continuation targets; map franchise targets to their entry point."""
        touched = set(item_franchise[watched].tolist()) if watched else set()
        out = {}
        for idx, gain in tgt.items():
            f = int(item_franchise[idx])
            if f in touched:
                continue
            e = entry_of_franchise.get(f, idx)
            out[e] = max(out.get(e, 0), gain)
        return out

    def evaluate(users, score_fn, chunk=1000):
        """users: list of user_arrays tuples. score_fn(fold_csr) -> dense scores."""
        ndcg10, rec10, rec50, regret, prof_pct, top_pct = [], [], [], [], [], []
        seen10 = np.zeros(n_items, dtype=bool)
        for lo in range(0, len(users), chunk):
            batch = users[lo : lo + chunk]
            rows, cols, vals = [], [], []
            for r, (fi, fw, _w, _p, _t, _n) in enumerate(batch):
                rows += [r] * len(fi)
                cols += fi
                vals += fw
            fold_csr = sp.csr_matrix(
                (vals, (rows, cols)), shape=(len(batch), n_items), dtype="float32"
            )
            scores = score_fn(fold_csr)
            for r, (fi, fw, watched, pi, tgt, neg) in enumerate(batch):
                s = scores[r].copy()
                s[~candidate_mask(watched, pi)] = -np.inf
                top50 = np.argsort(-s)[:50]
                top10 = top50[:10]
                seen10[top10] = True
                ctgt = collapse_targets(tgt, watched)
                if not ctgt:
                    continue
                gains = np.array([ctgt.get(int(i), 0.0) for i in top10])
                dcg = (gains / np.log2(np.arange(2, 12))).sum()
                ideal = np.sort(list(ctgt.values()))[::-1][:10]
                idcg = (ideal / np.log2(np.arange(2, 2 + len(ideal)))).sum()
                ndcg10.append(dcg / idcg if idcg > 0 else 0.0)
                rec10.append(len(set(top10.tolist()) & ctgt.keys()) / len(ctgt))
                rec50.append(len(set(top50.tolist()) & ctgt.keys()) / len(ctgt))
                regret.append(len(set(top10.tolist()) & set(neg)) / 10.0)
                prof_pct.append(np.median(pop_pct[fi]) if fi else 50.0)
                top_pct.append(pop_pct[top10].mean())
        return {
            "ndcg10": np.array(ndcg10),
            "rec10": np.array(rec10),
            "rec50": np.array(rec50),
            "regret10": np.array(regret),
            "prof_pct": np.array(prof_pct),
            "top_pct": np.array(top_pct),
            "coverage10": seen10.mean(),
        }

    return candidate_mask, evaluate, pop_pct, user_arrays


@app.cell
def _(S, evaluate, holdout, item_counts, np, pl, user_arrays):
    # run the eval on test users: BM25 vs MostPopular reference
    _test = holdout.filter(pl.col("role") == "test")
    test_users = [
        user_arrays(list(zip(g["anime_id"], g["kind"], g["weight"], g["fold"])))
        for _k, g in _test.group_by("user_id", maintain_order=True)
    ]

    def _bm25_scores(fold_csr):
        return np.asarray((fold_csr @ S).todense())

    def _pop_scores(fold_csr):
        return np.tile(item_counts.astype("float32"), (fold_csr.shape[0], 1))

    results = {
        "item-item BM25": evaluate(test_users, _bm25_scores),
        "MostPopular": evaluate(test_users, _pop_scores),
    }
    return (results,)


@app.cell
def _(SEED, mo, np, results):
    # headline metrics with bootstrap CI (protocol #6-7)
    def _ci(arr, n_boot=1000):
        rng = np.random.default_rng(SEED)
        means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
        return np.percentile(means, [2.5, 97.5])

    _rows = []
    for _name, _r in results.items():
        _lo, _hi = _ci(_r["ndcg10"])
        _niche = _r["prof_pct"] <= np.quantile(_r["prof_pct"], 0.25)
        _rows.append(
            f"| {_name} | {_r['ndcg10'].mean():.4f} [{_lo:.4f}, {_hi:.4f}] "
            f"| {_r['rec10'].mean():.4f} | {_r['rec50'].mean():.4f} "
            f"| {(_r['top_pct'] - _r['prof_pct']).mean():+.1f} "
            f"| {(_r['top_pct'][_niche] - _r['prof_pct'][_niche]).mean():+.1f} "
            f"| {_r['coverage10']:.3f} | {_r['regret10'].mean():.4f} | {len(_r['ndcg10'])} |"
        )
    metrics_table = mo.md(
        "### Test-set metrics (dial off)\n\n"
        "| model | NDCG@10 [95% CI] | recall@10 | recall@50 | pop lift | pop lift (niche) | coverage@10 | regret@10 | users |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nPop lift = mean popularity percentile of the top-10 minus the median of the "
        "user's profile (guardrail, not optimized). Sanity anchor from the protocol "
        "research: published anime-rec work reaches ~0.08-0.10 recall@20 at small scale."
    )
    if mo.app_meta().mode == "script":
        print(metrics_table.text)
    metrics_table
    return


@app.cell
def _(DIAL_PRESETS, mo):
    username_input = mo.ui.text(
        label="AniList username", placeholder="your AniList username", value=""
    )
    dial_radio = mo.ui.radio(
        options=list(DIAL_PRESETS), value="off", label="popularity dial"
    )
    mo.vstack([mo.md("### Vibe check — live AniList fold-in"), username_input, dial_radio])
    return dial_radio, username_input


@app.cell
def _(holdout, json, pl, urllib, user_arrays, username_input):
    # profile source: live AniList in interactive mode, a sampled val user otherwise
    def fetch_anilist(name):
        q = """query($name:String){MediaListCollection(userName:$name,type:ANIME){
            lists{entries{status score(format:POINT_10_DECIMAL) progress
            media{idMal episodes}}}}}"""
        req = urllib.request.Request(
            "https://graphql.anilist.co",
            data=json.dumps({"query": q, "variables": {"name": name}}).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AniList-Rec-prototype/0.1 (github.com/Ari-03/AniList_Rec)",
            },
        )
        payload = json.load(urllib.request.urlopen(req, timeout=30))
        rows = []
        for lst in payload["data"]["MediaListCollection"]["lists"]:
            for e in lst["entries"]:
                mal, score = e["media"]["idMal"], e["score"] or 0
                if mal is None:
                    continue
                st = e["status"]
                if st == "PLANNING":
                    kind, w = 4, 0.0
                elif st == "REPEATING":
                    kind, w = 2, 2.0
                elif st == "DROPPED":
                    kind, w = 0, 0.0
                elif st == "COMPLETED":
                    if 0 < score <= 4:
                        kind, w = 0, 0.0
                    elif score >= 8:
                        kind, w = 2, 2.0
                    else:
                        kind, w = 1, 1.0
                elif st == "CURRENT":
                    eps = e["media"]["episodes"] or 0
                    kind = 3
                    w = 0.5 + 0.5 * min((e["progress"] or 0) / eps, 1.0) if eps else 0.75
                elif st == "PAUSED":
                    kind, w = 3, 0.25
                else:
                    continue
                rows.append((mal, kind, w, True))  # whole list is fold-in at serve time
        return rows

    if username_input.value:
        profile_rows = fetch_anilist(username_input.value)
        profile_label = (
            f"AniList user **{username_input.value}** ({len(profile_rows)} list entries)"
        )
    else:
        _val = holdout.filter(pl.col("role") == "val")
        _uid = _val["user_id"][0]
        _g = _val.filter(pl.col("user_id") == _uid)
        profile_rows = [
            (a, k, w, True) for a, k, w in zip(_g["anime_id"], _g["kind"], _g["weight"])
        ]
        profile_label = (
            f"demo profile: held-out validation user `{_uid}` "
            "(type an AniList username above for the real thing)"
        )

    fold_idx_p, fold_w_p, watched_p, plan_p, _tgt, _neg = user_arrays(profile_rows)
    return fold_idx_p, fold_w_p, plan_p, profile_label, watched_p


@app.cell
def _(
    DIAL_PRESETS,
    S,
    candidate_mask,
    dial_radio,
    fold_idx_p,
    fold_w_p,
    item_counts,
    item_franchise,
    item_meta,
    mo,
    np,
    plan_p,
    pop_pct,
    profile_label,
    watched_p,
):
    # rank the full catalogue for the profile, apply the dial, render the top-20
    _w = np.zeros(S.shape[0], dtype="float32")
    _w[fold_idx_p] = fold_w_p
    _scores = np.asarray(_w @ S).ravel()

    _alpha = DIAL_PRESETS[dial_radio.value]
    _adj = _scores / np.power(item_counts + 1.0, _alpha)
    _adj[~candidate_mask(watched_p, plan_p)] = -np.inf
    _top = np.argsort(-_adj)[:20]

    def _why(j):
        contrib = np.asarray(S[:, j].todense()).ravel() * _w
        best = np.argsort(-contrib)[:2]
        names = [
            item_meta["title_english"][int(b)] or item_meta["title_romaji"][int(b)] or "?"
            for b in best
            if contrib[b] > 0
        ]
        return " + ".join(names) if names else "(no overlap with your list)"

    _cards = []
    for _rank, _j in enumerate(_top, 1):
        _m = item_meta.row(int(_j), named=True)
        _title = _m["title_english"] or _m["title_romaji"] or f"MAL id {_m['anime_id']}"
        _img = _m["coverImage_medium"] or ""
        _genres = (_m["genres"] or "[]").strip("[]").replace('"', "")
        _cards.append(
            '<div style="width:150px;font-family:sans-serif;font-size:12px">'
            f'<img src="{_img}" style="width:150px;height:210px;object-fit:cover;border-radius:6px"/>'
            f'<b>{_rank}. <a href="{_m["siteUrl"] or "#"}">{_title}</a></b><br/>'
            f'<span style="color:#666">{_genres}</span><br/>'
            f'<span style="color:#888">pop {pop_pct[_j]:.0f}th pct · franchise {int(item_franchise[_j])}</span><br/>'
            f"<i>because: {_why(int(_j))}</i></div>"
        )
    vibe = mo.vstack(
        [
            mo.md(f"{profile_label} — dial: **{dial_radio.value}**"),
            mo.Html(
                '<div style="display:flex;flex-wrap:wrap;gap:12px">' + "".join(_cards) + "</div>"
            ),
        ]
    )
    vibe
    return


if __name__ == "__main__":
    app.run()
