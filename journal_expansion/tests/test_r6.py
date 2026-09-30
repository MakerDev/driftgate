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
