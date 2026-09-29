"""Tests for F1/F2 fairness aggregation and F3 routing."""
import sys
from pathlib import Path
import numpy as np
import torch

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
for p in (str(JOURNAL_ROOT), str(JOURNAL_ROOT.parent), str(JOURNAL_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.controllers.fairness import (
    complementarity, donor_weights, apply_fairness_aggregation,
    RISK_Z0, ETA_ABSORB,
)
from test_journal import tiny_client, MockClientHead, MockServerHead, probe_of_classes
from train.trainer import EdgeServer, uniform_average


def _two_cells():
    c0 = tiny_client(0, [0], seed=11)
    c1 = tiny_client(1, [1], seed=12)
    es = {0: EdgeServer(0), 1: EdgeServer(1)}
    es[0].attach(c0)
    es[1].attach(c1)
    es[0].scope, es[1].scope = {0, 1, 2}, {3, 4}
    es[0].aggregate()
    es[1].aggregate()
    return es


def test_complementarity():
    comp = complementarity({0: {0, 1, 2}, 1: {3, 4}}, num_classes=5)
    # cell 0 lacks {3,4}; donor 1 covers all of it
    assert comp[0][1] == 1.0
    # cell 1 lacks {0,1,2}; donor 0 covers all of it
    assert comp[1][0] == 1.0
    assert comp[0][0] == 0.0


def test_donor_weights_normalized_and_favor_complementary():
    comp = complementarity({0: {0, 1}, 1: {2, 3}, 2: {0, 1}}, num_classes=4)
    w = donor_weights([0, 1, 2], None, {0: 0.5, 1: 0.5, 2: 0.5}, comp)
    assert abs(sum(w[0].values()) - 1.0) < 1e-9
    # for cell 0 (lacks {2,3}), donor 1 (covers it) must outweigh donor 2 (doesn't)
    assert w[0][1] > w[0][2]


def test_low_risk_cell_keeps_own_model():
    es = _two_cells()
    own0 = {k: v.clone() for k, v in es[0].clients_avg_weights.items()}
    # risk below threshold everywhere -> Lambda_eff == Lambda; with Lambda=1
    # (all own), cells must remain exactly their own aggregate
    apply_fairness_aggregation(es, {0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0},
                               num_classes=5)
    assert all(torch.allclose(es[0].clients_avg_weights[k], own0[k], atol=1e-6)
               for k in own0), "low-risk cells must NOT be dragged toward the pool"


def test_high_risk_cell_absorbs_more():
    es = _two_cells()
    own0 = {k: v.clone() for k, v in es[0].clients_avg_weights.items()}
    donor = {k: v.clone() for k, v in es[1].clients_avg_weights.items()}
    apply_fairness_aggregation(es, {0: 0.8, 1: 0.8}, {0: RISK_Z0 + 2.0, 1: 0.0},
                               num_classes=5)
    got = es[0].clients_avg_weights
    # expected Lambda_eff = 0.8 - ETA_ABSORB*2
    lam = 0.8 - ETA_ABSORB * 2.0
    for k in own0:
        exp = lam * own0[k].float() + (1 - lam) * donor[k].float()
        assert torch.allclose(got[k], exp, atol=1e-5)


def test_fairness_logs_lam_eff():
    es = _two_cells()
    log = apply_fairness_aggregation(es, {0: 0.5, 1: 0.5},
                                     {0: RISK_Z0 + 1.0, 1: 0.0}, num_classes=5)
    assert log[0] < 0.5 and log[1] == 0.5


def test_routing_disagree_conf_beats_entropy_for_overconfident_client():
    """A converged overconfident client (low entropy on everything) never
    routes under the entropy rule; disagreement routing still reaches the
    competent server on OOP traffic."""
    from src.evaluation.routing import evaluate_routing_one_client

    class C:
        device = "cpu"
        client_model = MockClientHead()
        server_models = {0: MockServerHead()}
        edge_server_ids = [0]

    x = probe_of_classes([2, 3, 4, 5], n_per=4)  # OOP: client is wrong, server right
    y = x[:, 0, 0, 0].long()
    ds = torch.utils.data.TensorDataset(x, y)
    accs, n = evaluate_routing_one_client(C(), ds, list(range(len(ds))), eth=0.8)
    assert n == 16
    # MockClient outputs one-hot*10 logits -> entropy ~ 0 -> entropy rule keeps
    # the WRONG client answer; disagreement routing must recover
    assert accs["entropy"] == 0.0
    assert accs["disagree_conf"] == 1.0
    assert accs["oracle"] == 1.0
