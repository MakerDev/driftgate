"""Unit tests for v3 implementation."""
import sys
import os
import pytest
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# --------- Partition tests ---------

def test_topology_basic():
    from data.partition import make_es_topology
    c2es, es2c = make_es_topology(50, 5, overlap_percentage=50, seed=0)
    # All clients should have at least 1 ES
    assert all(len(c2es[cid]) >= 1 for cid in range(50))
    # Some clients should have 2 ES (overlap)
    overlap = sum(1 for cid in range(50) if len(c2es[cid]) >= 2)
    assert overlap > 0
    assert overlap < 50  # not all
    # Each ES should have clients
    assert all(len(es2c[i]) > 0 for i in range(5))


def test_nd1_partition_cifar10():
    from data.partition import make_es_topology, nd1_partition
    np.random.seed(0)
    labels = np.tile(np.arange(10), 5000)
    c2es, es2c = make_es_topology(50, 5, overlap_percentage=50, seed=0)
    indices, main_classes, scope = nd1_partition(labels, 10, 50, c2es, es2c, seed=0)

    # 2 main classes per client (20% × 10)
    for cid in range(50):
        assert len(main_classes[cid]) == 2

    # ES scope: 4-7 classes
    for es_id in es2c:
        assert 4 <= len(scope[es_id]) <= 7

    # All clients have some data
    for cid in range(50):
        assert len(indices[cid]) > 0


def test_test_set_construction():
    from data.partition import build_per_client_test_set
    test_labels = np.tile(np.arange(10), 1000)
    main = {0, 1}
    scope = {0: {0, 1, 2, 3, 4}, 1: {3, 4, 5, 6}}
    all_used = {0, 1, 2, 3, 4, 5, 6}
    idxs, oop, oor = build_per_client_test_set(
        test_labels, main, scope, client_primary_es=0,
        all_used_classes=all_used, oop_ratio=0.4)
    # OOP = primary scope (0) ∩ remaining = {2, 3, 4}
    assert oop == {2, 3, 4}
    # OOR = remaining \ primary scope = {5, 6}
    assert oor == {5, 6}
    # Disjoint from main
    assert main.isdisjoint(oop)
    assert main.isdisjoint(oor)
    assert oop.isdisjoint(oor)


# --------- Loss tests ---------

def test_multi_exit_loss_basic():
    from models.losses import multi_exit_loss
    B, C = 8, 10
    cl = torch.randn(B, C, requires_grad=True)
    sl = [torch.randn(B, C, requires_grad=True) for _ in range(2)]
    y = torch.randint(0, C, (B,))
    loss = multi_exit_loss(cl, sl, y, gamma=0.5)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert cl.grad is not None
    assert sl[0].grad is not None


def test_multi_exit_loss_no_server():
    from models.losses import multi_exit_loss
    import torch.nn as nn
    cl = torch.randn(8, 10, requires_grad=True)
    y = torch.randint(0, 10, (8,))
    loss = multi_exit_loss(cl, [], y, gamma=0.5)
    ref = nn.CrossEntropyLoss()(cl, y)
    assert torch.allclose(loss, ref)


# --------- Model tests ---------

def test_model_factory():
    from models.architectures import ModelFactory, count_params
    f = ModelFactory()
    c = f.make_client()
    s = f.make_server()
    assert count_params(c) > 0
    assert count_params(s) > 0
    # Client should be smaller
    assert count_params(c) < count_params(s)

    x = torch.randn(4, 3, 32, 32)
    cl, rep = c(x)
    assert cl.shape == (4, 10)
    sl, _ = s(rep)
    assert sl.shape == (4, 10)


# --------- Aggregation tests ---------

def test_uniform_average():
    from train.trainer import uniform_average
    sd1 = {'w': torch.tensor([1.0, 2.0])}
    sd2 = {'w': torch.tensor([3.0, 4.0])}
    avg = uniform_average([sd1, sd2])
    assert torch.allclose(avg['w'], torch.tensor([2.0, 3.0]))


def test_mix_state_dicts_lambda_extremes():
    from train.trainer import mix_state_dicts
    sd_a = {'w': torch.tensor([1.0, 2.0])}
    sd_b = {'w': torch.tensor([3.0, 4.0])}
    # lambda=1 means all sd_a
    m1 = mix_state_dicts(sd_a, sd_b, 1.0)
    assert torch.allclose(m1['w'], sd_a['w'])
    # lambda=0 means all sd_b
    m0 = mix_state_dicts(sd_a, sd_b, 0.0)
    assert torch.allclose(m0['w'], sd_b['w'])


# --------- Controller tests ---------

def test_sigmoid_monotone():
    from controller.adaptive import map_H_to_lambda
    # High H -> low lambda (generalization)
    lam_low_H = map_H_to_lambda(0.5, mu_H=1.5, tau_H=0.4)
    lam_high_H = map_H_to_lambda(3.0, mu_H=1.5, tau_H=0.4)
    assert lam_low_H > lam_high_H
    # Bounded
    assert 0.1 <= lam_low_H <= 0.7
    assert 0.1 <= lam_high_H <= 0.7


def test_consensus_step():
    from controller.adaptive import consensus_step
    signals = {0: 1.0, 1: 2.0, 2: 3.0}
    neighbors = {0: [1], 1: [0, 2], 2: [1]}
    out = consensus_step(signals, neighbors, steps=1)
    # Node 0: average(1, 2) = 1.5
    assert abs(out[0] - 1.5) < 1e-6
    # Node 1: average(2, 1, 3) = 2.0
    assert abs(out[1] - 2.0) < 1e-6
    # Node 2: average(3, 2) = 2.5
    assert abs(out[2] - 2.5) < 1e-6


def test_adaptive_controller_step():
    from controller.adaptive import AdaptiveController
    # Isolated nodes (no neighbors): direct mapping
    ctrl = AdaptiveController(
        neighbors={0: [], 1: [], 2: []}, mu_H=1.5, tau_H=0.4,
        consensus_steps=1)
    H_per_es = {0: 2.5, 1: 1.0, 2: 0.3}
    lams, big_lams = ctrl.step(H_per_es)
    assert 0 in lams and 1 in lams and 2 in lams
    # Higher H -> lower lambda (favor generalization)
    assert lams[0] < lams[1] < lams[2]
    # All in valid range
    for es_id in (0, 1, 2):
        assert 0.1 <= lams[es_id] <= 0.7
        assert 0.2 <= big_lams[es_id] <= 0.8


# --------- Integration: tiny training loop ---------

def test_tiny_training_loop_runs():
    """Run 2 rounds on synthetic data to ensure pipeline works end-to-end."""
    import torch
    from data.partition import make_es_topology, nd1_partition, make_client_dataloader
    from models.architectures import ModelFactory
    from train.trainer import build_clients_and_es, synchronize_initial_models, run_one_global_round
    from torch.utils.data import Dataset

    class SyntheticDataset(Dataset):
        def __init__(self, n=200, num_classes=10):
            torch.manual_seed(0)
            self.images = torch.randn(n, 3, 32, 32)
            self.targets = [i % num_classes for i in range(n)]
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, i):
            return self.images[i], self.targets[i]

    ds = SyntheticDataset(n=200, num_classes=10)
    labels = np.array(ds.targets)
    c2es, es2c = make_es_topology(num_clients=10, num_edge_servers=2,
                                  overlap_percentage=20, seed=0)
    indices, main_classes, scope = nd1_partition(
        labels, 10, 10, c2es, es2c, seed=0)

    mf = ModelFactory()
    clients, es_dict = build_clients_and_es(
        num_clients=10, num_edge_servers=2, model_factory=mf,
        train_dataset=ds, client_indices=indices, client_to_es=c2es,
        device='cpu', batch_size=8, local_epochs=1)
    synchronize_initial_models(clients, es_dict)
    for es_id, es in es_dict.items():
        es.scope = scope[es_id]

    lamdas = {es_id: 0.2 for es_id in range(2)}
    big_lamdas = {es_id: 0.5 for es_id in range(2)}

    for r in range(2):
        loss = run_one_global_round(
            clients=clients, edge_servers_dict=es_dict,
            lamdas=lamdas, big_lambdas=big_lamdas,
            method='splitomc', gamma=0.5)
        assert isinstance(loss, float)
        assert not np.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
