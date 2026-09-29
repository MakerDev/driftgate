"""Tests for Phase-3 extensions: datasets, partitions, model split points."""
import sys
from pathlib import Path
import numpy as np
import torch

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))

from src.datasets_ext import dirichlet_partition, scope_from_mains
from src.models_ext import FlexModelFactory, split_stats


def test_dirichlet_partition_properties():
    labels = np.tile(np.arange(10), 500)
    idx, mains = dirichlet_partition(labels, 10, 20, alpha=0.3, seed=0)
    # data is partitioned without duplication
    all_idx = [i for v in idx.values() for i in v]
    assert len(all_idx) == len(set(all_idx))
    # main classes cover >= 90% of each client's samples
    for cid in range(20):
        if not idx[cid]:
            continue
        labs = labels[idx[cid]]
        frac = np.isin(labs, list(mains[cid])).mean()
        assert frac >= 0.9
    # lower alpha -> fewer main classes (more concentrated)
    _, mains_conc = dirichlet_partition(labels, 10, 20, alpha=0.1, seed=0)
    _, mains_unif = dirichlet_partition(labels, 10, 20, alpha=1.0, seed=0)
    assert (np.mean([len(m) for m in mains_conc.values()])
            < np.mean([len(m) for m in mains_unif.values()]))


def test_dirichlet_deterministic():
    labels = np.tile(np.arange(10), 300)
    a1 = dirichlet_partition(labels, 10, 10, alpha=0.3, seed=5)
    a2 = dirichlet_partition(labels, 10, 10, alpha=0.3, seed=5)
    assert a1[0] == a2[0] and a1[1] == a2[1]


def test_scope_from_mains():
    mains = {0: {1, 2}, 1: {3}}
    c2es = {0: [0], 1: [0, 1]}
    scope = scope_from_mains(mains, c2es, 2)
    assert scope[0] == {1, 2, 3}
    assert scope[1] == {3}


def test_all_split_points_forward():
    rows = {}
    for fam in ("cnn", "resnet18", "mobilenetv2"):
        for split in (("middle",) if fam == "cnn" else ("early", "middle", "late")):
            f = FlexModelFactory(family=fam, num_classes=10, split=split)
            st = split_stats(f, input_size=32)
            rows[(fam, split)] = st
            assert st["out_classes"] == 10
            assert st["client_params"] > 0 and st["server_params"] > 0
    # split point moves parameters from server to client
    assert (rows[("resnet18", "early")]["client_params"]
            < rows[("resnet18", "middle")]["client_params"]
            < rows[("resnet18", "late")]["client_params"])
    # and shrinks the activation at the cut
    assert (rows[("resnet18", "early")]["activation_floats_per_sample"]
            > rows[("resnet18", "late")]["activation_floats_per_sample"])


def test_flexcnn_matches_v4_contract():
    f = FlexModelFactory(family="cnn", num_classes=100, in_spatial=8)
    c, s = f.make_client(), f.make_server()
    x = torch.randn(2, 3, 32, 32)
    logits, rep = c(x)
    out, _ = s(rep)
    assert logits.shape == (2, 100) and out.shape == (2, 100)


def test_flexcnn_64px():
    f = FlexModelFactory(family="cnn", num_classes=200, in_spatial=16)
    c, s = f.make_client(), f.make_server()
    x = torch.randn(2, 3, 64, 64)
    logits, rep = c(x)
    assert rep.shape[-1] == 16
    out, _ = s(rep)
    assert out.shape == (2, 200)
