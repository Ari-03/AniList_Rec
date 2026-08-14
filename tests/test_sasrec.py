"""SASRec: sequence plumbing, order sensitivity, and the takes_batch eval path."""

import numpy as np
import polars as pl
import scipy.sparse as sp

from anilist_rec.evaluate import EvalUser
from anilist_rec.sasrec import make_model, pad_batch, sasrec_scorer


def tiny_model(**kwargs):
    defaults = dict(n_items=12, d=16, blocks=1, heads=1, dropout=0.0, seed=0)
    defaults.update(kwargs)
    return make_model(**defaults)


def test_pad_batch_left_aligns_newest_last():
    seqs, conf = pad_batch(
        [np.array([3, 1, 7]), np.array([5])], [np.array([1.0, 2.0, 0.5]), np.array([1.0])], 5
    )
    assert seqs.tolist() == [[0, 0, 3, 1, 7], [0, 0, 0, 0, 5]]
    assert conf[0].tolist() == [0.0, 0.0, 1.0, 2.0, 0.5]
    # over-long input keeps the most recent items
    seqs, _ = pad_batch([np.arange(1, 9)], [np.ones(8)], 5)
    assert seqs.tolist() == [[4, 5, 6, 7, 8]]


def test_forward_shapes_and_no_nan_on_padding():
    import torch

    model = tiny_model()
    seqs = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 2, 3, 4]])  # row 0: fully padded
    conf = torch.ones_like(seqs, dtype=torch.float32)
    h = model(seqs, conf)
    assert h.shape == (2, 5, 16)
    assert torch.isfinite(h).all()


def test_order_changes_scores_positions_dont_when_disabled():
    import torch

    model = tiny_model()
    model.eval()
    conf = torch.ones((1, 4))
    with torch.no_grad():
        fwd = model(torch.tensor([[2, 5, 9, 3]]), conf)[:, -1]
        rev = model(torch.tensor([[3, 9, 5, 2]]), conf)[:, -1]
    assert not torch.allclose(fwd, rev)  # sequence model: order matters

    model.use_positions = False  # the documented set-encoder fallback
    with torch.no_grad():
        # same last item, permuted history → identical prefix attention pattern
        a = model(torch.tensor([[2, 5, 9, 3]]), conf)[:, -1]
        b = model(torch.tensor([[9, 2, 5, 3]]), conf)[:, -1]
    assert torch.allclose(a, b, atol=1e-5)


def test_scorer_reads_batch_order_not_csr():
    users = [
        EvalUser([2, 5, 9], [1.0, 1.0, 2.0], [2, 5, 9], [], {1: 1.0}, []),
        EvalUser([9, 5, 2], [2.0, 1.0, 1.0], [2, 5, 9], [], {1: 1.0}, []),
    ]
    scorer = sasrec_scorer(tiny_model())
    assert scorer.takes_batch
    csr = sp.csr_matrix((2, 12), dtype="float32")  # same (empty) bag either way
    scores = scorer(csr, users)
    assert scores.shape == (2, 12)
    assert np.isfinite(scores).all()
    assert not np.allclose(scores[0], scores[1])  # order flowed through


def test_evaluate_passes_batch_to_seq_scorers():
    from anilist_rec.evaluate import evaluate
    from anilist_rec.franchise import FranchiseIndex

    n = 12
    franchise = FranchiseIndex(
        item_franchise=-np.arange(1, n + 1),
        is_entry=np.ones(n, dtype=bool),
        entry_of_franchise={},
    )
    users = [EvalUser([2, 5], [1.0, 1.0], [2, 5], [], {7: 1.0}, [])]
    seen = {}

    def fake(fold_csr, batch=None):
        seen["batch"] = batch
        return np.zeros((fold_csr.shape[0], n), dtype="float32")

    fake.takes_batch = True
    evaluate(users, fake, franchise, np.ones(n))
    assert seen["batch"] == users


def test_build_sequences_orders_and_caps(tmp_path):
    from anilist_rec.config import Config
    from anilist_rec.sasrec import build_sequences

    rows = pl.DataFrame(
        {
            "user_id": ["a"] * 3 + ["b"] * 2 + ["h"],
            "anime_id": [30, 10, 20, 10, 30, 10],
            "kind": [1] * 6,
            "weight": [1.0, 2.0, 1.0, 1.0, 0.0, 1.0],  # b's 30 is weightless → dropped
            "ts": [3, 1, None, 2, 1, 1],
        },
        schema_overrides={"weight": pl.Float32},
    )
    cfg = Config(data_dir=tmp_path)
    item_ids = np.array([10, 20, 30])
    items, weights, offsets = build_sequences(cfg, rows.lazy(), item_ids, {"h"})
    by_user = {
        i: items[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)
    }
    # user a: null ts first (20), then ts order 10, 30 → codes 2, 1, 3
    assert sorted(by_user.values()) == [[1], [2, 1, 3]]
    assert len(items) == len(weights) == offsets[-1] == 4
