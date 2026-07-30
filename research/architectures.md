# Recommender architectures for an exportable AniList recommender

Research findings for [#3](https://github.com/Ari-03/AniList_Rec/issues/3). This is a **survey**, not a decision — the shortlist is [#8](https://github.com/Ari-03/AniList_Rec/issues/8).

## The constraints, restated as filters

| # | Constraint | What it actually rules out |
|---|---|---|
| C1 | Must recommend for users **not in training data** | Any model whose only user representation is a per-user-ID embedding, unless a fold-in exists |
| C2 | Implicit **and** explicit feedback (6 statuses, 0–10 scores) | Models that can only consume one signal type without contortion |
| C3 | Trains on a Mac or free Colab at ~10⁷–10⁸ interactions, ~20k items | Anything needing multi-GPU or >12 GB RAM in float32 |
| C4 | Exportable artifact for a later web app | Models whose "artifact" is a training framework you must re-host |
| C5 | Manga-extensibility a plus | Favours item-side generality |

**C1 is by far the sharpest filter**, and it is worth being precise about it because the families divide cleanly into three groups.

### The fold-in taxonomy

- **Native** — the user representation *is* a function of their item list, computed at scoring time. There is no user-specific parameter to be missing. Item-item CF, EASE/SLIM, SASRec, and content-embedding approaches are all in this group. A brand-new user costs one sparse matrix-vector product.
- **Closed-form fold-in** — the model stores per-user vectors during training, but the training objective admits an exact solve for one user's vector given the frozen item factors. ALS is the canonical case; `implicit` ships this as [`recalculate_user`](https://benfred.github.io/implicit/api/models/cpu/als.html).
- **None without redesign** — the user vector is learned by gradient descent against a user ID with no analytic inverse. Vanilla NCF/NeuMF and an ID-only two-tower are here. You would have to run inference-time gradient steps per user, which is not a serving story.

A subtlety that matters for LightFM specifically: it is often described as solving cold start, but that is only true *to the extent that you have user features*. See below.

## Comparison table

Sizes assume ~20k items. Arithmetic is shown per-family below; timings marked **[measured]** were run on the target machine (Apple M4 Pro, 25.8 GB, numpy 2.5.1 on Accelerate).

| Family | Unseen users | Train cost @10⁷–10⁸ × 20k | Export artifact | Popularity bias | Library maturity |
|---|---|---|---|---|---|
| **Item-item CF** (cosine/kNN) | **Native** | Minutes; one sparse Gramian | top-k lists, ~16 MB | Moderate; tunable via BM25/TF-IDF weighting | `implicit` — active (v0.7.3, May 2026) |
| **EASE^R** | **Native** (`r·B`) | **47 s** for the 20k³ inverse **[measured]** + Gramian | Dense B 1.6 GB, or ~32 MB sparsified | Inherits selection bias; correctable | ~15 lines of numpy; no library needed |
| **SLIM** | **Native** | Much worse than EASE: O(\|I\|·(\|I\|−1)^2.376) | Sparse B, ~tens of MB | Similar to EASE | `implicit` has SLIM-adjacent; mostly research code |
| **ALS / WMF** | **Closed-form fold-in** | Minutes–1 h, linear in nnz | Item factors ~10 MB | **Strong** — confidence scales with counts | `implicit` — active, mature, GPU optional |
| **LightFM hybrid** | **Only with real user features** | Hours, single-machine CPU | Feature embeddings ~6 MB | Moderate; WARP is rank-sensitive | **Dormant** — last commit Apr 2023 |
| **NCF / NeuMF** | **None** (ID embeddings) | GPU-hours | SavedModel, tens of MB | Moderate | Superseded; see Rendle 2020 |
| **Two-tower** | **Native iff the tower eats history, not IDs** | GPU-hours | SavedModel + ANN index | Sampling-dependent | TFRS active; ID-tower is the default tutorial |
| **SASRec** | **Native** (no user embedding at all) | GPU-hours; 1.7 s/epoch on ML-1M | Item emb + 2 blocks, ~6 MB | Sampling-dependent | RecBole (v1.2.1, Feb 2025) |
| **Content embeddings** | **Native** | Minutes to embed 20k synopses | 20k×384 = **30.7 MB** | **Lowest** — ignores popularity entirely | `sentence-transformers` — active |

## Per-family notes

### Item-item CF (cosine / kNN)

The oldest option ([Sarwar et al. 2001](https://dl.acm.org/doi/10.1145/371920.372071); [Deshpande & Karypis 2004](https://dl.acm.org/doi/10.1145/963770.963776)) and still the reference baseline. Score a user as the sum of similarities from the items they have already consumed, so an unseen user is handled with zero ceremony — this is the "native" case in its purest form.

`implicit` ships [`CosineRecommender`, `BM25Recommender`, `TFIDFRecommender`](https://benfred.github.io/implicit/api/models/cpu/nearest_neighbours.html). The weighting choice is the main popularity-bias lever: raw cosine on co-occurrence counts over-weights blockbusters, and BM25's saturation term is the standard mitigation.

Artifact: keep top-k neighbours per item. At k=100, 20,000 × 100 × (4 B index + 4 B value) = **16 MB**.

Its weakness in this domain is not accuracy but *triviality* — see the franchise problem below.

### EASE^R

[Steck, WWW 2019](https://arxiv.org/abs/1905.03375). A linear item-item autoencoder with a closed form. The objective is

> min_B ‖X − XB‖²_F + λ‖B‖²_F  s.t. diag(B) = 0

and the solution, with P̂ ≜ (XᵀX + λI)⁻¹, is

> **B̂ = I − P̂ · diagMat(1 ⊘ diag(P̂))**, i.e. B̂ᵢⱼ = 0 if i=j, else −P̂ᵢⱼ/P̂ⱼⱼ

Three properties make this unusually well-matched to the ticket's constraints:

1. **Cost is independent of the number of users and interactions.** The paper states the complexity is "determined by the matrix inversion… O(|I|³)", and that the cost is "independent of the number of users as well as the number of user-item-interactions". With ~20k items, the 10⁷–10⁸ axis of C3 simply does not bite on the expensive step.
2. **Scoring is `r_new · B`** — a pure fold-in, C1 satisfied by construction.
3. **It beats the deep models it was compared against.** On MSD, EASE^R improved on Mult-VAE by "25%, 17% and 23% regarding Recall@20, Recall@50, and NDCG@100". Reported R@20: EASE 0.391 / SLIM 0.370 / WMF 0.360 / Mult-VAE 0.395 on ML-20M; on MSD, EASE 0.333 vs Mult-VAE 0.266.

**I measured the one step that could have been a blocker.** Inverting a dense 20,000×20,000 matrix on this Mac:

| n | float32 | float64 | matrix size (f64) |
|---|---|---|---|
| 2,000 | 0.06 s | 0.05 s | 0.03 GB |
| 5,000 | 0.84 s | 0.74 s | 0.20 GB |
| 10,000 | 6.22 s | 5.81 s | 0.80 GB |
| **20,000** | **52.6 s** | **47.3 s** | **3.20 GB** |

So EASE's core operation at this item count is a **~47-second** job on the target hardware. The real constraint is memory, not time: float64 needs 3.2 GB for the Gram matrix plus a similar amount for the inverse, so ~7–10 GB peak — fine on a 25.8 GB Mac, **tight on free Colab's ~12.7 GB**, where float32 (1.6 GB per matrix) is the safer path.

Artifact: B is dense at 20,000² × 4 B = **1.6 GB**, which is too heavy to ship to a browser-facing app. Retaining the top 200 weights per column gives 20,000 × 200 × 8 B ≈ **32 MB**. Whether that sparsification is lossless enough is an open question worth measuring.

Caveat on bias: EASE is fit on raw co-occurrence and inherits the data's selection bias toward popular items; there is active work on [propensity-weighted linear autoencoders](https://arxiv.org/pdf/2512.20896) precisely to correct this.

### SLIM

[Ning & Karypis 2011](https://ieeexplore.ieee.org/document/6137254) is EASE's predecessor — same shape of model, but with non-negativity and L1 constraints that force a per-column coordinate-descent solve. EASE's paper puts SLIM at O(|I|·(|I|−1)^2.376), "several orders of magnitude" slower, for *lower* accuracy in their table (0.370 vs 0.391 R@20 on ML-20M). SLIM is mainly of interest if the induced sparsity is wanted for its own sake — which, given the 1.6 GB dense-B export problem above, is not a frivolous consideration.

### ALS / weighted matrix factorization

[Hu, Koren & Volinsky 2008](https://ieeexplore.ieee.org/document/4781121) — the standard implicit-feedback MF. Binary preference `p_ui` with confidence `c_ui = 1 + α·r_ui`, alternating ridge solves. This confidence formulation is the natural home for C2: the six AniList statuses become confidence weights rather than being flattened to binary.

Unseen users are handled by the closed-form solve for one user vector against frozen item factors, exposed in `implicit` as `recalculate_user(userid, user_items)`, with `recommend(..., recalculate_user=True)` documented as "don't rely on stored user embeddings and instead recalculate from the passed in user_items". That is a genuine, supported fold-in — not a workaround.

Cost per iteration is O(f²·N + f³·(|U|+|I|)) with N = nnz. At N = 10⁸ and f = 128, the dominant term is 128² × 10⁸ ≈ 1.6 × 10¹² FLOPs per iteration; 15 iterations is plausibly tens of minutes to about an hour on this hardware. Comfortably in budget.

Artifact: **only the item factors need to ship** — 20,000 × 128 × 4 B = **10.2 MB** — because user vectors are reconstructed at request time. This is the tidiest export story of any family here.

The known cost is **popularity bias**: confidence weights grow with interaction counts, so head items dominate the factor geometry. Given the skew measured below, this is a live concern rather than a theoretical one.

### LightFM

[Kula 2015](https://arxiv.org/abs/1507.08439). The [docs](https://making.lyst.com/lightfm/docs/home.html) state it "represents each user and item as the sum of the latent representations of their features", which generalises "to new items (via item features) and to new users (via user features)".

**The user-feature clause is the catch.** If you pass an identity matrix as user features — the common default — each user's representation is a single dedicated embedding, and the model degenerates to plain MF with **no fold-in path**. LightFM only satisfies C1 if there are real, generalising user features, and for an AniList recommender we essentially have none: no reliable demographics, and the user's *list* is interaction data rather than a feature vector. The item side (tags, genres, studios, format) is genuinely useful and would help item cold start and manga transfer, but that is a different problem from C1.

**Maintenance is a real risk.** Latest release is [1.17, 20 March 2023](https://pypi.org/project/lightfm/); the [GitHub repo](https://github.com/lyst/lightfm) shows last commit 2023-04-30 and 166 open issues. It is a Cython extension, which is exactly the kind of package that breaks on new Python versions when nobody is publishing wheels. Roughly three years dormant.

### Neural CF / two-tower

[He et al. 2017](https://arxiv.org/abs/1708.05031) proposed replacing the dot product with a learned MLP. [Rendle et al. 2020](https://arxiv.org/abs/2005.09683) revisited it and found that "with a proper hyperparameter selection, a simple dot product substantially outperforms the proposed learned similarities", adding that "MLPs are too costly to use for item recommendation in production environments while dot products allow to apply very efficient retrieval algorithms".

This lands inside a broader replication problem. [Dacrema, Cremonesi & Jannach 2019](https://arxiv.org/abs/1907.06902) examined 18 neural recommenders from top venues: only 7 were reproducible with reasonable effort, and 6 of those "can often be outperformed with comparably simple heuristic methods, e.g., based on nearest-neighbor or graph-based techniques". The one survivor did not consistently beat a well-tuned non-neural linear method — which is to say, something in EASE's family.

Vanilla NCF **fails C1**: the user is an ID embedding with no fold-in.

Two-tower retrieval ([Yi et al., RecSys 2019](https://dl.acm.org/doi/10.1145/3298689.3346996)) is the interesting variant, and its C1 status depends entirely on what the query tower consumes. The [TFRS retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) builds the user tower as a `StringLookup` over user IDs into an `Embedding` — which the docs themselves note "corresponds exactly to a classic matrix factorization approach", and which fails C1. But the docs also note you "can easily extend it to an arbitrarily complex model using standard Keras components, as long as we return an `embedding_dimension`-wide output". **A tower fed the user's watch history rather than their ID satisfies C1 natively** — at which point you have essentially built a learned, nonlinear version of item-item CF. The export story is good (`tf.saved_model.save`, with `BruteForce` or `ScaNN` top-k layers baked in), though it means shipping a TensorFlow runtime.

### Sequential models (SASRec, BERT4Rec)

[SASRec (Kang & McAuley 2018)](https://arxiv.org/abs/1808.09781) is the strongest C1 story of any learned model, and the paper is explicit about why: they "empirically find that adding an explicit user embedding doesn't improve performance (presumably because the model already considers all of the user's actions)". There are **no user-specific parameters at all** — the representation is a deterministic function of the item sequence, so an unseen user is not a special case. It is also cheap: 1.7 s/epoch on ML-1M, "over 11 times faster than Caser and 18 times faster than GRU4Rec++", converging in ~350 s.

**The domain problem is sequence order.** SASRec "use[s] timestamps to determine the sequence order of actions", and the ablation shows removing positional embeddings hurts on denser datasets. AniList list entries have `updatedAt`/`completedAt`, but these reflect *when someone edited their list*, not when they watched — a user importing a 300-title backlog in one session produces a meaningless order. Whether usable ordering exists is a dataset question ([#2](https://github.com/Ari-03/AniList_Rec/issues/2)) that gates this whole family.

[BERT4Rec](https://arxiv.org/abs/1904.06690) is the bidirectional successor, but [Petrov & Macdonald 2022](https://arxiv.org/abs/2207.07483) found its reported results frequently not reproducible from the published configurations — another reason to treat this family as high-variance.

### Content / text-embedding approaches

Embed each item's synopsis, tags and genres with a sentence encoder ([sentence-transformers](https://sbert.net/)), represent a user as an aggregate of their item vectors, rank by cosine. C1 is native. 20k synopses is a few minutes of encoding on an M-series Mac.

Artifact: 20,000 × 384 × 4 B = **30.7 MB** for MiniLM-dimensional vectors. Note that at 20k items, **ANN indexing is unnecessary** — a full user-vs-catalogue score is a 384 × 20,000 matvec, ~7.7M FLOPs, sub-millisecond in numpy. faiss/hnswlib are solving a problem this project does not have.

Content-only recommenders are the weakest on accuracy — they capture "similar description", not "similar taste", and cannot learn that two tonally unrelated shows share an audience. Their real value here is threefold: **lowest popularity bias** of any family (the encoder has never heard of watch counts), **item cold start** for anime newer than the dataset dump, and **the manga bridge** (C5) — synopsis embeddings transfer across `MediaType` with no retraining. [item2vec](https://arxiv.org/abs/1603.04259)-style embeddings learned from co-occurrence are a middle path, but they are collaborative signal wearing content clothing and inherit the popularity skew.

## What's specific to the anime/AniList domain

I queried the AniList GraphQL API directly for these rather than relying on secondary write-ups.

### Status and score semantics are first-class

`MediaListStatus` is exactly, per API introspection: `CURRENT` ("Currently watching/reading"), `PLANNING` ("Planning to watch/read"), `COMPLETED` ("Finished watching/reading"), `DROPPED` ("Stopped watching/reading before completing"), `PAUSED`, `REPEATING` ("Re-watching/reading").

Two of these are semantically unusual and worth designing around:

- **`DROPPED` is an explicit negative.** Most implicit-feedback datasets have no observed negatives at all — that is the entire reason for the confidence formulation. Here the user has told you they disliked something. Under Hu/Koren/Volinsky this maps naturally to high confidence on a *zero* preference, which is strictly more information than a missing entry.
- **`PLANNING` is intent, not consumption.** Treating it as a positive conflates "I want to watch this" with "I watched and liked this" — and since planning lists skew hard toward hyped, popular titles, folding it in as a positive is a direct popularity-bias injection.
- **`REPEATING` is arguably the strongest positive signal available** and has no analogue in most datasets.

This variety is a genuine argument for confidence-weighted models (ALS) or for careful weighting in the X matrix (EASE), and against families that want a flat binary matrix.

### Popularity skew is severe — measured

Sampling `Page(sort: POPULARITY_DESC)` from the live API:

| Popularity rank | Median list entries | meanScore (median) |
|---|---|---|
| 1–50 | 601,718 | 83 |
| 51–100 | 426,543 | 80 |
| 151–200 | 264,529 | 79 |
| 451–500 | 139,338 | 75 |
| 951–1,000 | 77,116 | 72 |
| 1,951–2,000 | 35,910 | 72 |
| 3,951–4,000 | 11,450 | 70 |

The most popular title has **1,035,515** list entries; by rank ~4,000 that is down to ~11,450, roughly a **90× decay**, and the catalogue extends well past 20k titles. Any model whose objective is dominated by interaction counts will spend its capacity on the top few hundred titles. This is the single strongest empirical argument in the document for taking popularity bias seriously rather than treating it as a refinement — cf. [Steck, RecSys 2011](https://dl.acm.org/doi/10.1145/2043932.2043957) on the accuracy/long-tail trade-off and [Abdollahpouri et al.](https://arxiv.org/abs/1901.07555) on re-ranking mitigations.

### Scores are compressed and skewed high

Over 349 sampled titles, `meanScore` percentiles were **p5=61, p25=70, p50=76, p75=81, p95=87** (min 47, max 91). Site-wide scores live almost entirely in a 60–90 band. Consequences: raw 0–10 scores carry less discriminative signal than their range suggests; per-user mean-centring is close to mandatory; and users demonstrably rate on different personal scales. This weakens the case for treating the problem as rating *prediction* and strengthens the case for ranking from implicit signal with scores as a modifier.

### The franchise problem

`MediaRelation` includes `SEQUEL`, `PREQUEL`, `SIDE_STORY`, `PARENT`, `ALTERNATIVE`, `SPIN_OFF`, `SUMMARY`, `CHARACTER`, `SAME_UNIVERSE`, and `COMPILATION` — verified by introspection. Because each season, OVA and recap film is a **separate media entry**, co-occurrence between "Season 1" and "Season 2" is near-total.

Any pure co-occurrence model — item-item CF most acutely, but EASE and ALS too — will therefore rank *Attack on Titan Season 2* as the top recommendation for someone who just finished Season 1. This is technically correct, perfectly useless, and will dominate offline metrics while destroying the vibe check that [#5](https://github.com/Ari-03/AniList_Rec/issues/5) and [#7](https://github.com/Ari-03/AniList_Rec/issues/7) care about. The relation graph is the fix — collapse franchises into a canonical entry, or filter/demote related entries at serving time — and it is **architecture-independent plumbing that every candidate family needs**. Worth flagging early because it is easy to mistake for a modelling failure.

### Manga extensibility is nearly free

`MediaType` is exactly `ANIME | MANGA` — one schema, one API, the same list statuses and score scale. Better still, `ADAPTATION` ("An adaption of this media into a different format") and `SOURCE` ("The source material the media was adapted from") give an explicit anime↔manga bridge, so an anime-trained item space can seed manga via adaptation links. C5 costs little as long as the chosen architecture keeps item representations general — which favours item-side models over anything with baked-in anime-specific structure.

### Library maintenance snapshot (checked via GitHub API, July 2026)

| Library | Latest release | Last commit | Verdict |
|---|---|---|---|
| [`implicit`](https://github.com/benfred/implicit) | v0.7.3 (2026-05-08) | 2026-05-08 | **Actively maintained** |
| [`lightfm`](https://github.com/lyst/lightfm) | 1.17 (2023-03-20) | 2023-04-30 | **Dormant ~3 yrs**, 166 open issues |
| [RecBole](https://github.com/RUCAIBox/RecBole) | v1.2.1 (2025-02-23) | 2025-02-24 | Quiet but usable |
| [Microsoft recommenders](https://github.com/recommenders-team/recommenders) | 1.2.1 (2024-12-24) | 2026-05-26 | Commits current, releases lag |

## Implications for this project

*(The choice belongs to [#8](https://github.com/Ari-03/AniList_Rec/issues/8); this is what the evidence points at.)*

**The constraints are much less binding than they first appear.** ~20k items is small. EASE's expensive step is 47 measured seconds; ALS's artifact is 10 MB; content embeddings are 31 MB. The C3 compute worry and the C4 export worry mostly evaporate at this scale — with one exception, EASE's 1.6 GB dense B, which needs sparsification to be shippable.

**Three families look strongest:**

1. **EASE^R** — best accuracy-per-unit-complexity in the literature, native fold-in, user-count-independent training, and an implementation that is a handful of numpy lines with no dependency risk. Open questions: sparsifying B for export, and whether the 6 statuses can be encoded into X well enough to exploit C2.
2. **ALS via `implicit`** — the only family with a *documented, supported* fold-in API, the most natural home for status-as-confidence (C2), the smallest artifact, and the healthiest library. Its popularity bias is the known cost and, given the 90× skew measured above, must be actively managed.
3. **Content embeddings** — not as the primary ranker, but as the answer to the problems the collaborative families cannot solve: item cold start for new anime, the manga bridge (C5), and long-tail reach. Cheap enough to be additive rather than a competing choice.

**Item-item CF should be built regardless** — as the baseline the others must beat. The replication literature (Dacrema; Rendle) is a standing warning that this baseline beats sophisticated models more often than anyone expects, and [#9](https://github.com/Ari-03/AniList_Rec/issues/9) needs a floor to measure against.

**Weakest fits:** vanilla **NCF** fails C1 outright and Rendle 2020 removed its accuracy rationale. **LightFM** is a maintenance liability whose headline cold-start benefit needs user features this project does not have — its item-feature side is real but reachable by other means. **SASRec** is architecturally an excellent C1 fit and shouldn't be dismissed, but it is gated on whether trustworthy temporal ordering exists in the data; that question belongs to [#2](https://github.com/Ari-03/AniList_Rec/issues/2) and should be answered before this family is costed further.

**Two cross-cutting items that are not architecture choices but will masquerade as them:** franchise collapsing via the relation graph, and popularity-bias handling. Both need a position in the spec no matter which family wins.

## Appendix: reproducing the measurements

The inversion benchmark generates a synthetic SPD matrix at each size and times `np.linalg.inv`, standing in for EASE's `(XᵀX + λI)⁻¹`. numpy 2.5.1 on Apple Accelerate; Apple M4 Pro, 25.8 GB. The AniList figures come from `POST https://graphql.anilist.co` — introspection for the enums, and `Page(sort: POPULARITY_DESC)` sampled at ranks 1–4000 for the skew table. Note the API caps `pageInfo.total` at 5000, so that field is a paging limit, not a catalogue count.

## Sources

- Steck, [Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375) (EASE^R), WWW 2019
- Ning & Karypis, [SLIM: Sparse Linear Methods for Top-N Recommender Systems](https://ieeexplore.ieee.org/document/6137254), ICDM 2011
- Hu, Koren & Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121), ICDM 2008
- Sarwar et al., [Item-based Collaborative Filtering Recommendation Algorithms](https://dl.acm.org/doi/10.1145/371920.372071), WWW 2001
- Deshpande & Karypis, [Item-based Top-N Recommendation Algorithms](https://dl.acm.org/doi/10.1145/963770.963776), TOIS 2004
- Kula, [Metadata Embeddings for User and Item Cold-start Recommendations](https://arxiv.org/abs/1507.08439) (LightFM), 2015
- He et al., [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031), WWW 2017
- Rendle et al., [Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://arxiv.org/abs/2005.09683), RecSys 2020
- Dacrema, Cremonesi & Jannach, [Are We Really Making Much Progress?](https://arxiv.org/abs/1907.06902), RecSys 2019
- Yi et al., [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996), RecSys 2019
- Kang & McAuley, [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (SASRec), ICDM 2018
- Sun et al., [BERT4Rec](https://arxiv.org/abs/1904.06690), CIKM 2019; Petrov & Macdonald, [replicability study](https://arxiv.org/abs/2207.07483), RecSys 2022
- Barkan & Koenigstein, [item2vec](https://arxiv.org/abs/1603.04259), 2016
- Steck, [Item Popularity and Recommendation Accuracy](https://dl.acm.org/doi/10.1145/2043932.2043957), RecSys 2011
- Abdollahpouri et al., [Managing Popularity Bias with Personalized Re-ranking](https://arxiv.org/abs/1901.07555), 2019
- [`implicit` docs](https://benfred.github.io/implicit/) — [ALS](https://benfred.github.io/implicit/api/models/cpu/als.html), [kNN models](https://benfred.github.io/implicit/api/models/cpu/nearest_neighbours.html)
- [LightFM docs](https://making.lyst.com/lightfm/docs/home.html) · [PyPI](https://pypi.org/project/lightfm/) · [GitHub](https://github.com/lyst/lightfm)
- [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
- [RecBole](https://recbole.io/) ([paper](https://arxiv.org/abs/2011.01731)) · [Microsoft recommenders](https://github.com/recommenders-team/recommenders)
- [sentence-transformers](https://sbert.net/)
- [AniList API docs](https://docs.anilist.co/) — enum values verified by GraphQL introspection against `https://graphql.anilist.co`
