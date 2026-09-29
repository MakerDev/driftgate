"""Correctness-audit tests (Phase 0, prompt section 2.2) + new-code tests.

Covers the 10 mandatory audit items:
 1  lambda -> 1 preserves the local model            (test_lambda_one_preserves_local)
 2  lambda -> 0 converges to the cluster average     (test_lambda_zero_gives_cluster_avg)
 3  Lambda -> 0 maximizes inter-cluster/global mix   (test_Lambda_zero_gives_global)
 4  main-only vs OOP-heavy probe disagreement gap    (test_disagreement_separates_oop)
 5  controller/signal APIs take no labels and no rho (test_apis_are_label_and_rho_free)
 6  same seed/config -> same split                   (test_same_seed_same_split)
 7  no mobility -> serving membership fixed          (test_no_mobility_membership_fixed)
 8  mobility -> membership changes                   (test_mobility_changes_membership)
 9  no-drift -> no controller oscillation            (test_no_drift_no_oscillation)
10  controller never uses future signal values       (test_normalizer_is_causal)
plus: training data contains only main-class labels; self-cal direction;
      warm-up neutrality; metric sanity.
"""
import sys
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.architectures import CIFARClient, CIFARServer
from train.trainer import (
    SplitOMCClient, EdgeServer, update_client_models, apply_network_aggregation,
    uniform_average, run_one_global_round,
)
from controller.adaptive import ClientDriftTracker, gather_signals_for_clients
from src.controllers.self_calibrating import SelfCalController
from src.controllers.normalizers import GuardedRobustNormalizer
from src.signals.library import compute_client_signals, RAW_SIGNAL_NAMES
from src.evaluation.signal_metrics import evaluate_signal, causal_normalize, detection_delay


# ---------------------------------------------------------------- helpers
class TinyFactory:
    def make_client(self):
        return CIFARClient(num_classes=10, base_channels=8)

    def make_server(self):
        return CIFARServer(num_classes=10, base_channels=8)


def tiny_client(cid, es_ids, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    return SplitOMCClient(cid, loader, TinyFactory(), es_ids, device="cpu",
                          lr=0.01, local_epochs=1)


class MockClientHead(nn.Module):
    """Knows only classes {0,1}; anything else it mislabels as class 0.
    The class id is embedded at x[:,0,0,0] (test-only construction)."""
    def forward(self, x):
        c = x[:, 0, 0, 0].long()
        B = x.shape[0]
        logits = torch.zeros(B, 10)
        pred = torch.where(c < 2, c, torch.zeros_like(c))
        logits[torch.arange(B), pred] = 10.0
        rep = torch.zeros(B, 32, 8, 8)
        rep[:, 0, 0, 0] = c.float()
        return logits, rep


class MockServerHead(nn.Module):
    """Knows every class (reads it back from the representation)."""
    def forward(self, rep):
        c = rep[:, 0, 0, 0].long()
        B = rep.shape[0]
        logits = torch.zeros(B, 10)
        logits[torch.arange(B), c] = 10.0
        return logits, None


def probe_of_classes(classes, n_per=8):
    xs = []
    for c in classes:
        x = torch.zeros(n_per, 3, 32, 32)
        x[:, 0, 0, 0] = float(c)
        xs.append(x)
    return torch.cat(xs)


def _sd_allclose(a, b, atol=1e-6):
    return all(torch.allclose(a[k].float(), b[k].float(), atol=atol) for k in a)


# ------------------------------------------------- audit 1-3: mixing semantics
def _one_es_two_clients():
    c0 = tiny_client(0, [0], seed=1)
    c1 = tiny_client(1, [0], seed=2)
    es = EdgeServer(0)
    es.attach(c0)
    es.attach(c1)
    es.aggregate()
    return c0, c1, es


def test_lambda_one_preserves_local():
    c0, c1, es = _one_es_two_clients()
    before = c0.get_client_state()
    update_client_models([c0, c1], {0: es}, {0: 1.0}, method="splitomc")
    after = c0.get_client_state()
    assert _sd_allclose(before, after), \
        "lambda=1 must keep the LOCAL client model (lambda is a model-mixing weight)"


def test_lambda_zero_gives_cluster_avg():
    c0, c1, es = _one_es_two_clients()
    avg = {k: v.clone() for k, v in es.clients_avg_weights.items()}
    update_client_models([c0, c1], {0: es}, {0: 0.0}, method="splitomc")
    assert _sd_allclose(c0.get_client_state(), avg)
    assert _sd_allclose(c1.get_client_state(), avg)


def test_Lambda_zero_gives_global():
    c0 = tiny_client(0, [0], seed=3)
    c1 = tiny_client(1, [1], seed=4)
    es0, es1 = EdgeServer(0), EdgeServer(1)
    es0.attach(c0)
    es1.attach(c1)
    es0.aggregate()
    es1.aggregate()
    glob = uniform_average([es0.clients_avg_weights, es1.clients_avg_weights])
    apply_network_aggregation({0: es0, 1: es1}, {0: 0.0, 1: 0.0})
    assert _sd_allclose(es0.clients_avg_weights, glob)
    assert _sd_allclose(es1.clients_avg_weights, glob), \
        "Lambda=0 must mean FULL global mixing (Lambda weights the own-cell model)"


# ------------------------------------------------- audit 4: signal separation
def test_disagreement_separates_oop():
    cm, sm = MockClientHead(), MockServerHead()
    tr = ClientDriftTracker(device="cpu")
    d_main, _ = tr.compute_disagreement(cm, [sm], probe_of_classes([0, 1]))
    d_oop, _ = tr.compute_disagreement(cm, [sm], probe_of_classes([2, 3, 4, 5]))
    assert d_main == 0.0
    assert d_oop == 1.0

    sig_main = compute_client_signals(cm, [sm], probe_of_classes([0, 1]), "cpu")
    sig_oop = compute_client_signals(cm, [sm], probe_of_classes([2, 3, 4, 5]), "cpu")
    assert sig_oop["delta_hard"] > sig_main["delta_hard"]
    assert sig_oop["kl_sym"] > sig_main["kl_sym"]
    assert sig_oop["js_div"] > sig_main["js_div"]
    assert sig_oop["tv_dist"] > sig_main["tv_dist"]


# ------------------------------------------------- audit 5: label/rho-free API
def test_apis_are_label_and_rho_free():
    import inspect
    forbidden = {"label", "labels", "y", "rho", "rho_z", "targets", "schedule"}
    for fn in (ClientDriftTracker.compute_disagreement,
               gather_signals_for_clients,
               compute_client_signals,
               SelfCalController.step):
        params = set(inspect.signature(fn).parameters)
        assert not (params & forbidden), f"{fn.__qualname__} exposes {params & forbidden}"


# ------------------------------------------------- audit 6: determinism
def test_same_seed_same_split():
    from data.partition import make_es_topology, nd1_partition
    labels = np.tile(np.arange(10), 500)
    out = []
    for _ in range(2):
        c2es, es2c = make_es_topology(20, 4, 50, seed=7)
        idx, mains, scope = nd1_partition(labels, 10, 20, c2es, es2c, seed=7)
        out.append((idx, mains, scope))
    assert out[0][0] == out[1][0]
    assert out[0][1] == out[1][1]
    assert out[0][2] == out[1][2]


def test_train_data_only_main_classes():
    from data.partition import make_es_topology, nd1_partition
    labels = np.tile(np.arange(10), 500)
    c2es, es2c = make_es_topology(20, 4, 50, seed=3)
    idx, mains, _ = nd1_partition(labels, 10, 20, c2es, es2c, seed=3)
    for cid in range(20):
        got = set(labels[idx[cid]])
        assert got <= mains[cid], \
            "training loss must never see OOP/OOR labels (audit item 3 of 2.2)"


# ------------------------------------------------- audit 7/8: mobility
def test_no_mobility_membership_fixed():
    c0 = tiny_client(0, [0], seed=5)
    c1 = tiny_client(1, [0, 1], seed=6)
    es = {0: EdgeServer(0), 1: EdgeServer(1)}
    for c in (c0, c1):
        for e in c.edge_server_ids:
            es[e].attach(c)
    before = [list(c.edge_server_ids) for c in (c0, c1)]
    run_one_global_round([c0, c1], es, {0: 0.4, 1: 0.4}, {0: 0.5, 1: 0.5},
                         method="splitomcplus")
    assert [list(c.edge_server_ids) for c in (c0, c1)] == before


def test_mobility_changes_membership():
    from network.mobility import GaussMarkovMobility, rewire_clients
    mob = GaussMarkovMobility(num_clients=10, num_es=4, map_size=200.0,
                              alpha=0.5, speed=30.0, dt=10.0, seed=1)
    first = mob.get_topk_es(k=2)
    changed = False
    for _ in range(30):
        mob.step()
        if mob.get_topk_es(k=2) != first:
            changed = True
            break
    assert changed, "mobility must actually change serving membership over time"

    c0 = tiny_client(0, [0], seed=8)
    es = {i: EdgeServer(i) for i in range(4)}
    es[0].attach(c0)
    rewire_clients([c0], es, {0: [1, 2]}, TinyFactory())
    assert c0.edge_server_ids == [1, 2]
    assert set(c0.server_models) == {1, 2}


# ------------------------------------------------- audit 9: no false oscillation
def test_no_drift_no_oscillation():
    rng = np.random.default_rng(0)
    ctrl = SelfCalController(neighbors={0: [], 1: []})
    lam_trace = []
    for t in range(80):
        sig = {0: 0.30 + rng.normal(0, 0.02), 1: 0.30 + rng.normal(0, 0.02)}
        lams, _ = ctrl.step(sig)
        if t >= 20:
            lam_trace.append(lams[0])
    lam_trace = np.array(lam_trace)
    assert lam_trace.std() < 0.03, f"lambda oscillates without drift: std={lam_trace.std():.4f}"
    assert lam_trace.min() > 0.15 + 0.01, "lambda must not hit the rail without drift"


# ------------------------------------------------- audit 10: causality
def test_normalizer_is_causal():
    rng = np.random.default_rng(1)
    base = list(0.3 + rng.normal(0, 0.02, 60))
    n1 = GuardedRobustNormalizer()
    out1 = [n1.update(v) for v in base[:40]]
    tail = list(0.9 + rng.normal(0, 0.02, 20))
    n2 = GuardedRobustNormalizer()
    out2 = [n2.update(v) for v in base[:40] + tail]
    assert np.allclose(out1, out2[:40]), \
        "outputs up to t must not depend on values after t"


def test_offline_matches_online():
    rng = np.random.default_rng(2)
    x = 0.3 + rng.normal(0, 0.05, 100)
    off = causal_normalize(x, kind="guarded")
    n = GuardedRobustNormalizer()
    on = np.array([n.update(v) for v in x])
    assert np.allclose(off, on)


# ------------------------------------------------- self-cal behavior
def test_selfcal_warmup_neutral_and_direction():
    ctrl = SelfCalController(neighbors={0: []}, warmup=15)
    lam_warm = None
    for t in range(15):
        lams, _ = ctrl.step({0: 0.30})
        lam_warm = lams[0]
    assert abs(lam_warm - 0.5 * (0.15 + 0.7)) < 1e-9, "warm-up must output neutral lambda"

    for _ in range(10):
        lams_no_drift, _ = ctrl.step({0: 0.30})
    for _ in range(10):
        lams_drift, _ = ctrl.step({0: 0.60})
    assert lams_drift[0] < lams_no_drift[0] - 0.15, \
        "sustained signal jump must push lambda toward generalization"

    for _ in range(40):
        lams_rec, _ = ctrl.step({0: 0.30})
    assert lams_rec[0] > lams_drift[0] + 0.1, "lambda must recover after drift ends"


def test_selfcal_retention_under_sustained_drift():
    """Guarded baseline: z must stay elevated through a LONG drift segment
    (the failure mode of unguarded rolling baselines)."""
    ctrl = SelfCalController(neighbors={0: []}, warmup=15)
    for _ in range(30):
        ctrl.step({0: 0.30})
    z_vals = []
    for _ in range(50):
        ctrl.step({0: 0.60})
        z_vals.append(ctrl.last_z[0])
    assert z_vals[-1] > 2.0, f"z decayed under sustained drift: {z_vals[-5:]}"


# ------------------------------------------------- metric sanity
def test_signal_metrics_sanity():
    rng = np.random.default_rng(3)
    T = 120
    rho = np.zeros(T)
    rho[60:] = 0.8
    good = 0.3 + 0.3 * rho + rng.normal(0, 0.02, T)
    bad = rng.normal(0.5, 0.05, T)
    mg = evaluate_signal(good, rho)
    mb = evaluate_signal(bad, rho)
    assert mg["auroc_norm"] > 0.9
    assert mg["spearman_rho"] > 0.7
    assert mb["auroc_norm"] < 0.75
    assert mg["detection_delay"] < 6


def test_detection_delay_synthetic():
    z = np.zeros(50)
    z[24:] = 3.0  # onset detected from round 25
    assert detection_delay(z, onset_round=21) == 4


def test_signal_names_match_library():
    cm, sm = MockClientHead(), MockServerHead()
    sig = compute_client_signals(cm, [sm], probe_of_classes([0, 1]), "cpu")
    assert set(sig) == set(RAW_SIGNAL_NAMES)


def test_v3c_asymmetric_guard_resists_stepped_ramp():
    """Boiling-frog check: a stepped ramp (0.30 -> 0.38 -> 0.46) must keep z
    elevated under the v3c asymmetric guard (z_guard=0.5), while the default
    guard (1.5) absorbs each step into the baseline."""
    from src.controllers.self_calibrating import SelfCalController

    def run(z_guard):
        # each step is ~1.2 sigma (sigma-floor = 0.1*0.30 = 0.03): individually
        # below the default 1.5 guard — the boiling-frog regime seen on the
        # Schedule-A dev traces
        ctrl = SelfCalController(neighbors={0: []}, warmup=15, burn_in=0,
                                 z_guard=z_guard)
        for _ in range(30):
            ctrl.step({0: 0.30})
        for _ in range(20):
            ctrl.step({0: 0.335})
        for _ in range(20):
            ctrl.step({0: 0.37})
        lam_mid = None
        for _ in range(20):
            lam_mid, _ = ctrl.step({0: 0.405})
        return lam_mid[0], ctrl.last_z[0]

    lam_v3c, z_v3c = run(0.5)
    lam_v3b, z_v3b = run(1.5)
    assert z_v3c > z_v3b + 1.0, f"asymmetric guard must retain elevation ({z_v3c} vs {z_v3b})"
    assert lam_v3c < 0.3, f"v3c must generalize under the sustained ramp (lam={lam_v3c})"
    # and it must still track a DECLINE (convergence direction)
    ctrl = SelfCalController(neighbors={0: []}, warmup=15, z_guard=0.5)
    for _ in range(30):
        ctrl.step({0: 0.30})
    for _ in range(40):
        lams, _ = ctrl.step({0: 0.15})
    assert lams[0] > 0.6, "declining signal must recover lambda (baseline tracks down)"


def test_spatial_norm_stratifies_static_heterogeneity():
    """A cell whose drift is STATIC (present from round 1) produces zero
    temporal z forever; the cross-cell spatial z must stratify it anyway."""
    from src.controllers.self_calibrating import SelfCalController, spatial_z

    sig = {0: 0.30, 1: 0.31, 2: 0.33, 3: 0.38, 4: 0.46}  # static spatial pattern
    zsp = spatial_z(sig)
    assert zsp[4] > 2.0 and zsp[0] < 0.5

    # temporal-only controller: blind (all lambdas ~equal, high)
    ct = SelfCalController(neighbors={i: [] for i in range(5)}, warmup=15)
    cs = SelfCalController(neighbors={i: [] for i in range(5)}, warmup=15,
                           spatial_norm=True)
    for _ in range(60):
        lam_t, _ = ct.step(dict(sig))
        lam_s, _ = cs.step(dict(sig))
    spread_t = max(lam_t.values()) - min(lam_t.values())
    spread_s = max(lam_s.values()) - min(lam_s.values())
    assert spread_t < 0.05, f"temporal-only should NOT stratify static pattern ({spread_t})"
    assert spread_s > 0.2, f"spatial norm must stratify ({spread_s})"
    assert lam_s[4] < lam_s[0] - 0.2, "high-rho cell must generalize (low lambda)"


def test_abs_cap_dual_view():
    """delta is a probability with intrinsic meaning: a STATIC high level
    (0.6 = most traffic outside client competence) must cap lambda even when
    the temporal z is zero; a low level must leave the z-view in charge."""
    from src.controllers.self_calibrating import SelfCalController

    def run(level):
        ctrl = SelfCalController(neighbors={0: []}, warmup=15, burn_in=0,
                                 z_guard=0.5, abs_cap=True)
        lams = None
        for _ in range(60):
            lams, _ = ctrl.step({0: level})
        return lams[0]

    lam_hi = run(0.60)   # cap: 0.7 - 0.55*0.6 = 0.37
    lam_lo = run(0.20)   # cap: 0.59 -> z-view (~0.63) capped mildly
    assert lam_hi < 0.40, f"high static delta must cap personalization ({lam_hi})"
    assert lam_lo > 0.55, f"low static delta must not over-cap ({lam_lo})"


def test_mobility_cid_indexing():
    """Bug O3: the mobility change-detection loop indexed the clients LIST by
    cid; with any empty (skipped) client the indices shift and change counts
    are computed against the WRONG client. The fixed code maps by cid.
    This test builds the misaligned situation directly."""
    # clients list where list-index != cid (cid 1 missing, as if it had no data)
    c0 = tiny_client(0, [0], seed=21)
    c2 = tiny_client(2, [1], seed=22)
    clients = [c0, c2]
    new_c2es = {0: [0], 1: [1], 2: [9]}  # only cid 2 actually changes
    by_cid = {c.cid: c for c in clients}
    changes = sum(1 for cid, c in by_cid.items()
                  if set(new_c2es[cid]) != set(c.edge_server_ids))
    assert changes == 1
    # the OLD buggy expression would have compared cid=1's target vs clients[1]
    # (which is cid 2) -> counts 1 change for the wrong client and misses cid 2
    buggy = sum(1 for cid in range(3)
                if cid < len(clients)
                and set(new_c2es[cid]) != set(clients[cid].edge_server_ids))
    assert buggy != changes, "regression guard: buggy form must disagree here"


def test_no_empty_clients_in_recorded_partitions():
    """Evidence that O3 was dormant: the ND1 partition used by every recorded
    run gives every client data for seeds 0-4 (so list index == cid held)."""
    from data.partition import make_es_topology, nd1_partition
    labels = np.tile(np.arange(10), 5000)
    for seed in range(5):
        c2es, es2c = make_es_topology(50, 5, 50, seed=seed)
        idx, _, _ = nd1_partition(labels, 10, 50, c2es, es2c, seed=seed)
        assert all(len(idx[cid]) > 0 for cid in range(50)), f"seed {seed}"


def test_role_ablation_same_role_generalizes_client():
    """R2 same_role: after the role-aware update, the client head equals the cell
    average (generalized), NOT a lambda-mix — i.e. role separation is removed."""
    import sys as _s; from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from src.role_ablation import role_aware_update
    from train.trainer import EdgeServer
    c0 = tiny_client(0, [0], seed=31); c1 = tiny_client(1, [0], seed=32)
    es = EdgeServer(0); es.attach(c0); es.attach(c1); es.aggregate()
    avg = {k: v.clone() for k, v in es.clients_avg_weights.items()}
    role_aware_update([c0, c1], {0: es}, {0: 0.7}, "same_role")  # lam ignored
    assert _sd_allclose(c0.get_client_state(), avg), \
        "same_role must set the client head to the cell average (generalized)"


def test_role_ablation_weak_server_mixes_server():
    """R4 weak_server: server head becomes a 50/50 mix of own + cell average
    (reduced generalization), while the client head stays lambda-personalized."""
    import sys as _s; from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from src.role_ablation import role_aware_update, WEAK_SERVER_MIX
    from train.trainer import EdgeServer, mix_state_dicts
    c0 = tiny_client(0, [0], seed=41); c1 = tiny_client(1, [0], seed=42)
    es = EdgeServer(0); es.attach(c0); es.attach(c1); es.aggregate()
    own = {k: v.clone() for k, v in c0.get_server_state(0).items()}
    cell = {k: v.clone() for k, v in es.server_avg_weights.items()}
    role_aware_update([c0, c1], {0: es}, {0: 0.4}, "weak_server")
    exp = mix_state_dicts(own, cell, WEAK_SERVER_MIX)
    assert _sd_allclose(c0.get_server_state(0), exp), \
        "weak_server must 50/50-mix the server head toward its own weights"


def test_gamma_convention_client_weight():
    """gamma weights the CLIENT loss: L = gamma*CE(client) + (1-gamma)*mean_z CE(server).
    gamma=1 -> only client; gamma=0 -> only server; gamma=0.5 -> equal."""
    from models.losses import multi_exit_loss, single_ce_loss
    torch.manual_seed(0)
    B, C = 8, 10
    cl = torch.randn(B, C); sl = [torch.randn(B, C)]
    y = torch.randint(0, C, (B,))
    lc = single_ce_loss(cl, y); ls = single_ce_loss(sl[0], y)
    assert torch.allclose(multi_exit_loss(cl, sl, y, gamma=1.0), lc, atol=1e-6), "gamma=1 -> client only"
    assert torch.allclose(multi_exit_loss(cl, sl, y, gamma=0.0), ls, atol=1e-6), "gamma=0 -> server only"
    assert torch.allclose(multi_exit_loss(cl, sl, y, gamma=0.5), 0.5*lc+0.5*ls, atol=1e-6)


def test_Lambda_one_keeps_cell_local():
    """Lambda=1 must keep each cell's own aggregate (no global mixing) — the
    complement of test_Lambda_zero_gives_global."""
    c0 = tiny_client(0, [0], seed=51); c1 = tiny_client(1, [1], seed=52)
    es0, es1 = EdgeServer(0), EdgeServer(1)
    es0.attach(c0); es1.attach(c1); es0.aggregate(); es1.aggregate()
    own0 = {k: v.clone() for k, v in es0.clients_avg_weights.items()}
    apply_network_aggregation({0: es0, 1: es1}, {0: 1.0, 1: 1.0})
    assert _sd_allclose(es0.clients_avg_weights, own0), "Lambda=1 keeps the cell aggregate"


def test_disjoint_pool_masks():
    """Task 1: controller/eval pools are stratified, disjoint, deterministic, and
    cover the class labels; masked labels return only their pool's indices."""
    from src.disjoint_pools import make_pool_masks
    labels = np.tile(np.arange(10), 1000)  # 10000 samples, 1000/class
    cl, el, man = make_pool_masks(labels, 10, controller_frac=0.2, seed=3)
    ctrl = set(np.where(cl >= 0)[0]); ev = set(np.where(el >= 0)[0])
    assert ctrl.isdisjoint(ev), "pools overlap"
    assert man["global_overlap"] == 0
    # stratified ~20/80 per class
    for c in range(10):
        assert abs(man["per_class"][c]["n_controller"] - 200) <= 1
        assert man["per_class"][c]["n_eval"] == 1000 - man["per_class"][c]["n_controller"]
    # np.where on masked labels returns only that pool's class-c indices
    for c in range(10):
        assert set(np.where(cl == c)[0]).issubset(ctrl)
        assert set(np.where(el == c)[0]).issubset(ev)
    # deterministic
    cl2, el2, _ = make_pool_masks(labels, 10, 0.2, seed=3)
    assert np.array_equal(cl, cl2) and np.array_equal(el, el2)
    # different seed -> different split
    cl3, _, _ = make_pool_masks(labels, 10, 0.2, seed=4)
    assert not np.array_equal(cl, cl3)


def test_server_nonmain_signals_range_and_direction():
    """Task 2: soft/hard server non-Main signals are in [0,1] and rise when the
    server puts mass/top-1 outside the client's Main classes."""
    from src.signals.library import server_nonmain_signals
    B = 8
    # server that predicts class = embedded id (MockServerHead-style logits)
    def logits_for(cls_list):
        z = torch.full((len(cls_list), 10), -10.0)
        for i, c in enumerate(cls_list):
            z[i, c] = 10.0
        return z
    main = {0, 1}
    # all in-Main -> soft~0, hard~0
    s_in = server_nonmain_signals(logits_for([0, 1, 0, 1, 0, 1, 0, 1]), main)
    assert 0.0 <= s_in["server_nonmain_soft"] <= 0.05
    assert s_in["server_nonmain_hard"] == 0.0
    # all out-of-Main -> soft~1, hard==1
    s_out = server_nonmain_signals(logits_for([2, 3, 4, 5, 6, 7, 8, 9]), main)
    assert s_out["server_nonmain_soft"] >= 0.95
    assert s_out["server_nonmain_hard"] == 1.0
    # half/half -> hard == 0.5
    s_mix = server_nonmain_signals(logits_for([0, 1, 2, 3, 0, 1, 4, 5]), main)
    assert abs(s_mix["server_nonmain_hard"] - 0.5) < 1e-6
    for d in (s_in, s_out, s_mix):
        for v in d.values():
            assert 0.0 <= v <= 1.0
