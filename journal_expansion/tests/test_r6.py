"""Round 6 tests: controller equivalence, environment invariants, request sets."""
import sys
from pathlib import Path

import numpy as np
import pytest

JR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))

from src.controllers.self_calibrating import SelfCalController  # noqa: E402
from src.r6_controller import EdgeDriftGate  # noqa: E402
from src import r6_env  # noqa: E402
from src.r6_requests import RequestBuilder  # noqa: E402


def _relonly(neighbors):
    return SelfCalController(neighbors=neighbors, warmup=15, burn_in=10, z_guard=0.5,
                             spatial_norm=True, abs_cap=False, abs_only=False,
                             consensus_steps=1, normalizer="guarded")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("graph", ["line", "star", "complete"])
def test_edge_controller_matches_selfcal(seed, graph):
    L = 5
    nb = {"line": {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]},
          "star": {0: [4], 1: [4], 2: [4], 3: [4], 4: [0, 1, 2, 3]},
          "complete": {z: [w for w in range(L) if w != z] for z in range(L)}}[graph]
    rng = np.random.default_rng(seed)
    ref, new = _relonly(nb), EdgeDriftGate(L, nb)
    for t in range(150):
        base = 0.3 + 0.2 * (60 <= t < 110)
        sig = {z: float(base + 0.05 * z + 0.03 * rng.standard_normal()) for z in range(L)}
        a_lam, a_Lam = ref.step(sig)
        b_lam, b_Lam = new.step(sig)
        assert a_lam == b_lam and a_Lam == b_Lam, t
        assert ref.last_z == {z: new.q[z] for z in range(L)}


def test_edge_controller_missing_edges_match_on_present():
    """An edge with no signal keeps its state; present edges match the original
    controller fed with the same subset (the original simply omits the edge)."""
    L = 5
    nb = {0: [4], 1: [4], 2: [4], 3: [4], 4: [0, 1, 2, 3]}
    rng = np.random.default_rng(7)
    ref, new = _relonly(nb), EdgeDriftGate(L, nb)
    for t in range(120):
        sig = {z: float(0.3 + 0.03 * rng.standard_normal()) for z in range(L)}
        if 40 <= t < 50:
            sig.pop(2)
        prev_lam2 = new.lam[2]
        a_lam, _ = ref.step(sig)
        b_lam, _ = new.step(sig)
        for z in sig:
            assert a_lam[z] == b_lam[z]
        if 2 not in sig:
            assert b_lam[2] == prev_lam2


def test_edge_controller_warmup_values():
    new = EdgeDriftGate(5, {z: [] for z in range(5)})
    for t in range(25):
        lam, Lam = new.step({z: 0.3 for z in range(5)})
        assert all(v == pytest.approx(0.425) for v in lam.values())
        assert all(v == pytest.approx(0.55) for v in Lam.values())


def test_env_streams_are_shared():
    s1, _, _, _ = r6_env.build_synthetic("S1", 1)
    fast, _, _, _ = r6_env.build_synthetic("S1fast", 1)
    low, _, _, _ = r6_env.build_synthetic("S4_lowreq", 1)
    p7, _, _, _ = r6_env.build_synthetic("S4_part07", 1)
    p5, _, _, _ = r6_env.build_synthetic("S4_part05", 1)
    l1, _, _, _ = r6_env.build_synthetic("S4_loss01", 1)
    l3, _, _, _ = r6_env.build_synthetic("S4_loss03", 1)
    assert np.array_equal(s1["home_pos"], fast["home_pos"])              # same plans
    assert np.array_equal(s1["plan_type"], fast["plan_type"])
    assert np.array_equal(s1["pos"], low["pos"]) and np.array_equal(s1["pos"], p7["pos"])
    assert not np.array_equal(s1["n_req"], low["n_req"])
    assert np.all(p5["avail"] <= p7["avail"])                            # nested participation
    assert np.all(l1["lost_c2e"] <= l3["lost_c2e"]) and np.all(l1["lost_nbr"] <= l3["lost_nbr"])
    assert s1["avail"].all() and not s1["lost_c2e"].any()


def test_env_membership_rules():
    env, meta, _, _ = r6_env.build_synthetic("S1", 0)
    cell_xy = env["cell_xy"].astype(float)
    pos = env["pos"].astype(float)
    m = env["member"]
    d = np.linalg.norm(pos[:, :, None, :] - cell_xy[None, None], axis=3)
    for k in range(pos.shape[0]):
        for t in range(pos.shape[1]):
            inside = np.flatnonzero(d[k, t] <= r6_env.R_KM + 1e-6)
            got = [int(z) for z in m[k, t] if z >= 0]
            if len(inside) == 0:
                assert got == [int(np.argmin(d[k, t]))]
            else:
                want = [int(z) for z in inside[np.argsort(d[k, t][inside], kind="stable")][:2]]
                assert got == want
    at_home = (m == env["home_cell"][:, None, None]).any(axis=2)
    assert np.array_equal(at_home, env["at_home"])
    assert np.allclose(env["rho"], np.where(at_home, 0.1, 0.8))
    # everyone is home at 05:00 and at 19:54 (no trip before 06:30, all back by ~20:00 on foot is not
    # guaranteed, so only the first round is asserted)
    assert env["at_home"][:, 0].all()
    # neighbour rule: hub <-> residential only
    nb = env["neighbors"]
    assert nb[4, :4].all() and not nb[:4, :4].any()


def test_request_builder_rule_and_fallbacks():
    labels = np.repeat(np.arange(10), 100)
    groups = {0: {0, 1, 2, 3}, 1: {4, 5}, 2: {0, 1}}
    rb = RequestBuilder(labels, 10, groups, set(range(10)))
    idx, oop, oor, flag = rb.build({0, 1}, [0], 0.8)
    assert oop == {2, 3} and oor == set(range(4, 10)) and flag == ""
    lab = labels[idx]
    assert (lab < 2).sum() == 200
    assert (np.isin(lab, [2, 3])).sum() == 2 * int(200 * 0.8 / 2)
    assert (lab >= 4).sum() == 6 * int(200 * 0.24 / 6)
    # OOP empty -> its share is drawn from OOR
    idx, oop, oor, flag = rb.build({0, 1}, [2], 0.8)
    assert flag == "oop_empty" and not oop
    assert (labels[idx] >= 2).sum() == 8 * int(200 * 1.04 / 8)
    # OOR empty -> its share is drawn from OOP
    rb2 = RequestBuilder(labels, 10, {0: set(range(10))}, set(range(10)))
    idx, oop, oor, flag = rb2.build({0, 1}, [0], 0.1)
    assert flag == "oor_empty" and (labels[idx] >= 2).sum() == 8 * int(200 * 0.13 / 8)


def _tiny_setup(seed=0, n_clients=6, n_es=3, per_client=48):
    import torch
    from data.partition import get_cifar10
    from models.architectures import ModelFactory
    from train.trainer import build_clients_and_es, synchronize_initial_models
    from src.runner_r6 import set_seed
    train_ds, test_ds = get_cifar10("./data_cache")
    labels = np.array(train_ds.targets)
    idx = {k: np.flatnonzero(labels == (k % 10))[:per_client].tolist() for k in range(n_clients)}
    c2es = {k: ([k % n_es] if k % 2 == 0 else sorted({k % n_es, (k + 1) % n_es})) for k in range(n_clients)}
    set_seed(seed)
    mf = ModelFactory(dataset="cifar10", num_classes=10)
    clients, ES = build_clients_and_es(n_clients, n_es, mf, train_ds, idx, c2es, device="cpu",
                                       batch_size=16, local_epochs=1)
    synchronize_initial_models(clients, ES)
    return clients, ES, test_ds


def test_training_round_matches_original():
    import torch
    from train.trainer import run_one_global_round
    from src.runner_r6 import r6_training_round, set_seed
    lam = {0: 0.3, 1: 0.5, 2: 0.6}
    Lam = {0: 0.45, 1: 0.55, 2: 0.65}
    cA, esA, _ = _tiny_setup()
    set_seed(5)
    run_one_global_round(cA, esA, lam, Lam, method="splitomcplus", gamma=0.5)
    cB, esB, _ = _tiny_setup()
    set_seed(5)
    r6_training_round(cB, np.ones(len(cB), bool), esB, lam, Lam, "fixed", 0.5)
    for a, b in zip(cA, cB):
        for k, v in a.client_model.state_dict().items():
            assert torch.equal(v, b.client_model.state_dict()[k])
        for z in a.edge_server_ids:
            for k, v in a.server_models[z].state_dict().items():
                assert torch.equal(v, b.server_models[z].state_dict()[k])


def test_training_round_skips_empty_and_inactive():
    import torch
    from src.runner_r6 import r6_training_round, set_seed
    clients, ES, _ = _tiny_setup()
    active = np.zeros(len(clients), bool)
    active[[0, 2]] = True            # clients 0, 2, both only in cell 0 / cell 2
    before = {c.cid: {k: v.clone() for k, v in c.client_model.state_dict().items()} for c in clients}
    es1_before = ES[1].clients_avg_weights
    set_seed(1)
    _, cells = r6_training_round(clients, active, ES, {z: 0.4 for z in ES}, {z: 0.5 for z in ES}, "fixed", 0.5)
    assert sorted(cells) == [0, 2]
    assert ES[1].clients_avg_weights is es1_before           # cell without active members untouched
    for c in clients:
        same = all(torch.equal(v, before[c.cid][k]) for k, v in c.client_model.state_dict().items())
        assert same == (not active[c.cid])


def test_eval_client_matches_evaluator():
    import torch
    from eval.evaluator import evaluate_one_client
    from src.runner_r6 import eval_client
    clients, ES, test_ds = _tiny_setup()
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(2000)])
    Y = torch.as_tensor(labels[:2000])
    idx = np.concatenate([np.flatnonzero(labels[:2000] == c)[:n] for c, n in ((0, 150), (1, 150), (2, 40), (7, 20))])
    for c in clients[:3]:
        ref = evaluate_one_client(c, test_ds, idx.tolist(), {0, 1}, {2}, {7}, eth=0.8)
        got = eval_client(c, X, Y, idx, {0, 1}, {2}, {7}, eth=0.8)
        assert got["n_total"] == ref["n_total"]
        assert got["correct"] / got["n_total"] == pytest.approx(ref["acc_total"], abs=1e-12)
        assert got["c_main"] / got["n_main"] == pytest.approx(ref["acc_main"], abs=1e-12)
        assert got["c_oop"] / got["n_oop"] == pytest.approx(ref["acc_oop"], abs=1e-12)
        assert got["c_oor"] / got["n_oor"] == pytest.approx(ref["acc_oor"], abs=1e-12)
        assert got["n_main"] == ref["n_main"] and got["n_oop"] == ref["n_oop"]


def test_cached_loader_training_is_bit_identical():
    """Round with the original DataLoader(Subset) vs CachedLoader: same weights, same RNG state."""
    import torch
    from src.runner_r6 import r6_training_round, set_seed, use_cached_loaders
    lam = {0: 0.3, 1: 0.5, 2: 0.6}
    Lam = {0: 0.45, 1: 0.55, 2: 0.65}
    cA, esA, _ = _tiny_setup()
    set_seed(9)
    for _ in range(2):
        r6_training_round(cA, np.ones(len(cA), bool), esA, lam, Lam, "fixed", 0.5)
    rngA = torch.get_rng_state()
    cB, esB, _ = _tiny_setup()
    from data.partition import get_cifar10
    train_ds, _ = get_cifar10("./data_cache")
    labels = np.array(train_ds.targets)
    idx = {k: np.flatnonzero(labels == (k % 10))[:48].tolist() for k in range(len(cB))}
    use_cached_loaders(cB, idx, train_ds, "cpu", 16)
    set_seed(9)
    for _ in range(2):
        r6_training_round(cB, np.ones(len(cB), bool), esB, lam, Lam, "fixed", 0.5)
    assert torch.equal(rngA, torch.get_rng_state())
    for a, b in zip(cA, cB):
        for k, v in a.client_model.state_dict().items():
            assert torch.equal(v, b.client_model.state_dict()[k]), k


@pytest.mark.parametrize("graph", ["line", "star", "complete"])
def test_no_neighbor_avg_both_controllers_agree_and_use_own_score(graph):
    """--no_neighbor_avg: q = own smoothed max score; SelfCalController and EdgeDriftGate bit-identical,
    and the neighbour structure no longer matters."""
    L = 5
    nb = {"line": {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]},
          "star": {0: [4], 1: [4], 2: [4], 3: [4], 4: [0, 1, 2, 3]},
          "complete": {z: [w for w in range(L) if w != z] for z in range(L)}}[graph]
    rng = np.random.default_rng(3)
    a = SelfCalController(neighbors=nb, warmup=15, burn_in=10, z_guard=0.5, spatial_norm=True,
                          consensus_steps=1, neighbor_avg=False)
    iso = SelfCalController(neighbors={z: [] for z in range(L)}, warmup=15, burn_in=10, z_guard=0.5,
                            spatial_norm=True, consensus_steps=1)
    b = EdgeDriftGate(L, nb, neighbor_avg=False)
    for t in range(150):
        sig = {z: float(0.3 + 0.2 * (z == 4 and 60 <= t < 110) + 0.03 * rng.standard_normal()) for z in range(L)}
        la, La = a.step(sig)
        li, Li = iso.step(sig)
        lb, Lb = b.step(sig)
        assert la == lb == li and La == Lb == Li, t
        assert a.last_z == {z: b.q[z] for z in range(L)}


def test_neighbor_avg_default_unchanged():
    """default (neighbor_avg=True) still averages: with a star graph the hub's q differs from its own score."""
    nb = {0: [4], 1: [4], 2: [4], 3: [4], 4: [0, 1, 2, 3]}
    a = SelfCalController(neighbors=nb, warmup=15, burn_in=10, z_guard=0.5, spatial_norm=True, consensus_steps=1)
    b = SelfCalController(neighbors=nb, warmup=15, burn_in=10, z_guard=0.5, spatial_norm=True, consensus_steps=1,
                          neighbor_avg=False)
    for t in range(60):
        sig = {z: 0.3 + (0.3 if (z == 4 and t >= 40) else 0.0) for z in range(5)}
        a.step(sig)
        b.step(sig)
    assert a.last_z[4] < b.last_z[4] and a.last_z[0] > b.last_z[0]


def test_absonly_no_neighbor_avg_uses_own_signal():
    nb = {0: [1], 1: [0, 2], 2: [1]}
    c = SelfCalController(neighbors=nb, warmup=15, burn_in=10, z_guard=0.5, spatial_norm=True,
                          abs_cap=True, abs_only=True, neighbor_avg=False)
    ema = {z: None for z in range(3)}
    for t in range(40):
        sig = {0: 0.2, 1: 0.5, 2: 0.8}
        lam, _ = c.step(sig)
        for z, v in sig.items():
            ema[z] = v if ema[z] is None else 0.7 * ema[z] + 0.3 * v
    for z in range(3):
        assert lam[z] == pytest.approx(0.7 - 0.55 * ema[z])


# ----------------------------------------------------------------------------- Round 7 gate
def test_device_controller_warmup_hold_and_spatial():
    from src.r6_controller import DeviceDriftGate
    from src.controllers.self_calibrating import spatial_z
    dc = DeviceDriftGate(4)
    ref = {k: [0, 1, 2, 3] for k in range(4)}
    rng = np.random.default_rng(0)
    for t in range(25):
        lam = dc.step({k: 0.3 + 0.01 * rng.standard_normal() for k in range(4)}, ref)
        assert all(v == pytest.approx(0.425) for v in lam.values())
    # client 3 sends nothing for a round -> its whole state is kept
    before = (dc.lam[3], dc.q[3], dc._n_obs[3], dc._z_smooth[3], dc._zsp_smooth[3])
    dc.step({k: 0.3 for k in range(3)}, {k: [0, 1, 2] for k in range(3)})
    assert (dc.lam[3], dc.q[3], dc._n_obs[3], dc._z_smooth[3], dc._zsp_smooth[3]) == before
    # spatial score = spatial_z over the reference set, EMA 0.3
    x = {0: 0.30, 1: 0.31, 2: 0.29, 3: 0.60}
    prev = dc._zsp_smooth[3]
    dc.step(x, ref)
    assert dc._zsp_smooth[3] == pytest.approx(0.7 * prev + 0.3 * spatial_z(x)[3])
    # fewer than 3 reference values -> temporal score only (spatial state untouched)
    prev = dict(dc._zsp_smooth)
    dc.step({0: 0.3, 1: 0.3}, {0: [0, 1], 1: [0, 1]})
    assert dc._zsp_smooth[0] == prev[0] and dc._zsp_smooth[1] == prev[1]


def test_per_client_lambda_matches_cell_lambda():
    """client_lams equal to the mean of the client's cell lambdas reproduce update_client_models exactly."""
    import torch
    from src.runner_r6 import r6_training_round, set_seed
    lam = {0: 0.3, 1: 0.5, 2: 0.6}
    Lam = {0: 0.5, 1: 0.5, 2: 0.5}
    cA, esA, _ = _tiny_setup()
    set_seed(3)
    r6_training_round(cA, np.ones(len(cA), bool), esA, lam, Lam, "fixed", 0.5)
    cB, esB, _ = _tiny_setup()
    per = {c.cid: (lam[c.edge_server_ids[0]] if len(c.edge_server_ids) == 1
                   else float(np.mean([lam[z] for z in c.edge_server_ids]))) for c in cB}
    set_seed(3)
    r6_training_round(cB, np.ones(len(cB), bool), esB, lam, Lam, "device", 0.5, client_lams=per)
    for a, b in zip(cA, cB):
        for k, v in a.client_model.state_dict().items():
            assert torch.equal(v, b.client_model.state_dict()[k]), k
        for z in a.edge_server_ids:
            for k, v in a.server_models[z].state_dict().items():
                assert torch.equal(v, b.server_models[z].state_dict()[k]), k


# ----------------------------------------------------------------------------- Round 8 check
def test_capture_keeps_training_and_reproduces_installed_model():
    """capture does not change training; mixing the captured (theta_local, cell average) with the client's
    training lambda gives the installed client-side state bit for bit."""
    import torch
    from src.runner_r6 import r6_training_round, set_seed
    from train.trainer import mix_state_dicts
    lam = {0: 0.4, 1: 0.4, 2: 0.4}
    Lam = {0: 0.5, 1: 0.5, 2: 0.5}
    cA, esA, _ = _tiny_setup()
    set_seed(4)
    r6_training_round(cA, np.ones(len(cA), bool), esA, lam, Lam, "fixed", 0.5)
    cB, esB, _ = _tiny_setup()
    cap = {}
    set_seed(4)
    r6_training_round(cB, np.ones(len(cB), bool), esB, lam, Lam, "fixed", 0.5, capture=cap)
    assert sorted(cap) == [c.cid for c in cB]
    for a, b in zip(cA, cB):
        sa, sb = a.client_model.state_dict(), b.client_model.state_dict()
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k
        local, avg = cap[b.cid]
        lt = float(np.mean([lam[z] for z in b.edge_server_ids])) if len(b.edge_server_ids) > 1 else lam[b.edge_server_ids[0]]
        mixed = mix_state_dicts(local, avg, lt)
        for k in sb:
            assert torch.equal(mixed[k].to(sb[k].dtype), sb[k]), k


def test_inference_mix_eval_restores_installed_model_and_mainaware_keeps_counts():
    import torch
    from src.runner_r6 import eval_client, r6_training_round, set_seed
    from train.trainer import mix_state_dicts
    clients, ES, test_ds = _tiny_setup()
    cap = {}
    set_seed(2)
    r6_training_round(clients, np.ones(len(clients), bool), ES, {0: 0.4, 1: 0.4, 2: 0.4},
                      {0: 0.5, 1: 0.5, 2: 0.5}, "fixed", 0.5, capture=cap)
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(1500)])
    Y = torch.as_tensor(labels[:1500])
    idx = np.concatenate([np.flatnonzero(labels[:1500] == c)[:n] for c, n in ((0, 120), (1, 120), (2, 40), (7, 30))])
    main = {0, 1}
    for c in clients[:4]:
        before = {k: v.clone() for k, v in c.client_model.state_dict().items()}
        base = eval_client(c, X, Y, idx, main, {2}, {7})
        ma = eval_client(c, X, Y, idx, main, {2}, {7}, mainaware=True)
        assert {k: ma[k] for k in base} == base
        # manual Main-aware rule
        with torch.no_grad():
            cl, rep = c.client_model(X[idx])
            sl = sum(sm(rep)[0] for sm in c.server_models.values()) / len(c.server_models)
        p = torch.softmax(cl, 1)
        ent = -(p * torch.log(p + 1e-12)).sum(1)
        sp, cp = sl.argmax(1), cl.argmax(1)
        route = (ent > 0.8) | ~torch.isin(sp, torch.tensor(sorted(main)))
        assert ma["ma_n_off"] == int(route.sum())
        assert abs(ma["ma_correct"] - int((torch.where(route, sp, cp) == Y[idx]).sum())) <= 2   # entropy ties only
        local, avg = cap[c.cid]
        installed = c.get_client_state()
        got = {}
        for li in (0.15, 0.4, 0.7):
            c.set_client_state(mix_state_dicts(local, avg, li))
            got[li] = eval_client(c, X, Y, idx, main, {2}, {7})
        c.set_client_state(installed)
        assert got[0.4] == base
        for k, v in c.client_model.state_dict().items():
            assert torch.equal(v, before[k]), k


# ----------------------------------------------------------------------------- Round 8 v2
def test_eval_client_record_reproduces_counts():
    import torch
    from src.runner_r6 import eval_client
    clients, ES, test_ds = _tiny_setup()
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(1500)])
    Y = torch.as_tensor(labels[:1500])
    idx = np.concatenate([np.flatnonzero(labels[:1500] == c)[:n] for c, n in ((0, 150), (1, 130), (2, 40), (7, 30))])
    for c in clients[:4]:
        base = eval_client(c, X, Y, idx, {0, 1}, {2}, {7})
        rec = {}
        got = eval_client(c, X, Y, idx, {0, 1}, {2}, {7}, record=rec)
        assert got == base
        y = labels[idx]
        assert rec["ent"].dtype == np.float32 and len(rec["cp"]) == len(idx)
        route = rec["ent"] > np.float32(0.8)
        final = np.where(route, rec["sp"], rec["cp"])
        assert int((final == y).sum()) == base["correct"] and int(route.sum()) == base["n_off"]
        main = np.isin(y, [0, 1])
        assert int(((final == y) & main).sum()) == base["c_main"]


def test_compute_client_signals_per_request():
    import torch
    from src.signals.library import compute_client_signals
    clients, ES, test_ds = _tiny_setup()
    X = torch.stack([test_ds[i][0] for i in range(64)])
    for c in clients[:3]:
        sms = list(c.server_models.values())
        a = compute_client_signals(c.client_model, sms, X, "cpu", None, main_classes={0, 1})
        b, per = compute_client_signals(c.client_model, sms, X, "cpu", None, main_classes={0, 1}, per_request=True)
        assert a == b
        assert len(per["tv"]) == 64 and per["sr"].dtype == bool
        assert abs(float(per["tv"].mean()) - a["tv_dist"]) < 1e-6
        assert float(per["sr"].mean()) == a["server_nonmain_hard"]


# ----------------------------------------------------------------------------- Round 10
def test_client_label_hist_matches_cached_loader_epochs():
    """the label counts recorded per round equal the labels a client's CachedLoader yields over local_epochs epochs."""
    import torch
    from src.runner_r6 import CachedLoader, client_label_hist, set_seed
    from data.partition import get_cifar10
    train_ds, _ = get_cifar10("./data_cache")
    labels = np.array(train_ds.targets)
    idx = np.concatenate([np.flatnonzero(labels == c)[:n] for c, n in ((0, 37), (3, 21), (8, 5))]).tolist()
    X = torch.zeros((len(labels), 1))
    Y = torch.as_tensor(labels)
    loader = CachedLoader(idx, X, Y, batch_size=16)
    set_seed(1)
    seen = np.zeros(10, np.int64)
    for _ in range(3):
        for _, yb in loader:
            seen += np.bincount(yb.numpy(), minlength=10)
    assert np.array_equal(seen, client_label_hist(idx, labels, 10, 3))


def test_eval_client_record_probs_keeps_counts_and_matches_predictions():
    import torch
    from src.runner_r6 import eval_client
    clients, ES, test_ds = _tiny_setup()
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(1500)])
    Y = torch.as_tensor(labels[:1500])
    idx = np.concatenate([np.flatnonzero(labels[:1500] == c)[:n] for c, n in ((0, 150), (1, 130), (2, 40), (7, 30))])
    for c in clients[:4]:
        base = eval_client(c, X, Y, idx, {0, 1}, {2}, {7})
        rec = {}
        got = eval_client(c, X, Y, idx, {0, 1}, {2}, {7}, record=rec, record_probs=True)
        assert got == base
        pc, ps = rec["pc"].astype(np.float64), rec["ps"].astype(np.float64)
        assert rec["pc"].dtype == np.float16 and pc.shape == (len(idx), 10)
        assert np.allclose(pc.sum(1), 1, atol=1e-2) and np.allclose(ps.sum(1), 1, atol=1e-2)
        assert (pc.argmax(1) == rec["cp"]).mean() > 0.99 and (ps.argmax(1) == rec["sp"]).mean() > 0.99
        ent = -(pc * np.log(np.clip(pc, 1e-12, 1))).sum(1)
        assert np.abs(ent - rec["ent"]).max() < 5e-2


# ----------------------------------------------------------------------------- Round 12
def test_r5_eval_record_uses_no_rng_and_matches_evaluator():
    """the Round 12 recording pass of the Round-5 runner leaves torch / numpy / random RNG states untouched and its
    per-client accuracy equals eval.evaluator.evaluate_one_client."""
    import random
    import torch
    from eval.evaluator import evaluate_one_client
    from src.runner import record_eval_requests_r5
    clients, ES, test_ds = _tiny_setup()
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    Y = torch.as_tensor(labels)
    sel = {c.cid: np.concatenate([np.flatnonzero(labels == q)[:n] for q, n in ((c.cid % 10, 60), ((c.cid + 1) % 10, 20), ((c.cid + 5) % 10, 10))]).tolist()
           for c in clients}
    mains = {c.cid: {c.cid % 10} for c in clients}
    oop = {c.cid: {(c.cid + 1) % 10} for c in clients}
    oor = {c.cid: {(c.cid + 5) % 10} for c in clients}
    c2es = {c.cid: list(c.edge_server_ids) for c in clients}
    st = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    recs = record_eval_requests_r5(clients, X, Y, sel, mains, oop, oor, 0.8, 3, 5, 0, c2es)
    assert torch.equal(st[0], torch.get_rng_state()) and np.array_equal(st[1], np.random.get_state()[1]) and st[2] == random.getstate()
    for m, c in zip(recs, clients):
        ref = evaluate_one_client(c, test_ds, sel[c.cid], mains[c.cid], oop[c.cid], oor[c.cid], eth=0.8)
        assert m["correct"] / m["n"] == pytest.approx(ref["acc_total"], abs=1e-12)
        assert sorted(m["arrival"]) == list(range(m["n"])) and m["pc"].shape == (m["n"], 10)
        assert set(np.unique(m["kind"])) <= {0, 1, 2}


# ----------------------------------------------------------------------------- Round 13b
@pytest.mark.parametrize("family,split,rep_shape", [("resnet20", "shallow", (16, 32, 32)), ("resnet20", "middle", (32, 16, 16)),
                                                    ("vgg11", "shallow", (128, 8, 8)), ("vgg11", "middle", (256, 4, 4))])
def test_r13b_split_models(family, split, rep_shape):
    """client exit / server exit shapes at the cut, total parameters of the standard models (ResNet-20 about 0.27 M with
    the projection shortcut, VGG-11-BN about 9.2 M), and arch_stats keeps the torch RNG state."""
    import torch
    from src.models_ext import FlexModelFactory, arch_stats
    f = FlexModelFactory(family, 100, split=split)
    c, s = f.make_client(), f.make_server()
    x = torch.randn(3, 3, 32, 32)
    lc, rep = c(x)
    ls, _ = s(rep)
    assert lc.shape == (3, 100) and ls.shape == (3, 100) and tuple(rep.shape[1:]) == rep_shape
    f10 = FlexModelFactory(family, 10, split=split)
    total = sum(p.numel() for m in (f10.make_client(), f10.make_server()) for p in m.parameters())
    assert (270_000 < total < 275_000) if family == "resnet20" else (9_200_000 < total < 9_260_000)
    st = torch.get_rng_state().clone()
    a = arch_stats(f10)
    assert torch.equal(st, torch.get_rng_state())
    assert a["smashed_bytes_float32"] == 4 * int(np.prod(rep_shape)) and a["client_block_flops"] > 0


# ----------------------------------------------------------------------------- Round 15
def test_r15_eval_logprobs_match_probs_and_use_no_rng():
    """record_logprobs adds float32 log-softmax of both exits that match the float16 probabilities, without touching
    the RNG states or the other record arrays."""
    import random
    import torch
    from src.runner_r6 import eval_client
    clients, ES, test_ds = _tiny_setup()
    labels = np.array(test_ds.targets)
    X = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    Y = torch.as_tensor(labels)
    c = clients[0]
    idx = np.concatenate([np.flatnonzero(labels == q)[:30] for q in (0, 1, 5)]).tolist()
    st = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    r0, r1 = {}, {}
    o0 = eval_client(c, X, Y, idx, {0}, {1}, {5}, record=r0, record_probs=True)
    o1 = eval_client(c, X, Y, idx, {0}, {1}, {5}, record=r1, record_probs=True, record_logprobs=True)
    assert torch.equal(st[0], torch.get_rng_state()) and np.array_equal(st[1], np.random.get_state()[1]) and st[2] == random.getstate()
    assert o0 == o1 and all(np.array_equal(r0[k], r1[k]) for k in r0)
    assert r1["lpc"].dtype == np.float32 and r1["lpc"].shape == r1["pc"].shape
    assert np.abs(np.exp(r1["lpc"]) - r1["pc"].astype(np.float32)).max() < 1e-3
    assert np.abs(np.exp(r1["lps"]) - r1["ps"].astype(np.float32)).max() < 1e-3
