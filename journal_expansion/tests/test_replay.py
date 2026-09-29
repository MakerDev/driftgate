"""Tests for trajectory-replay transforms (Phase B decomposition)."""
import sys, json
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))
from src.replay import load_transformed_trajectory, trajectory_means


def _mk(tmp_path):
    T, cells = 40, [0, 1, 2]
    lam = [{str(z): 0.2 + 0.01 * t + 0.05 * z for z in cells} for t in range(T)]
    Lam = [{str(z): 0.5 for z in cells} for t in range(T)]
    p = tmp_path / "src.json"
    p.write_text(json.dumps({"lamdas": lam, "big_lamdas": Lam}))
    return p, T, cells


def test_value_preservation_and_transforms(tmp_path):
    p, T, cells = _mk(tmp_path)
    ident, _ = load_transformed_trajectory(p, "identity")
    assert len(ident) == T and abs(ident[5][1] - (0.2 + 0.05 + 0.05)) < 1e-9
    shuf, _ = load_transformed_trajectory(p, "shuffle", seed=0)
    vals_i = sorted(d[0] for d in ident)
    vals_s = sorted(d[0] for d in shuf)
    assert np.allclose(vals_i, vals_s), "shuffle must preserve the value multiset"
    assert [d[0] for d in shuf] != [d[0] for d in ident]
    rev, _ = load_transformed_trajectory(p, "reverse")
    assert rev[0][0] == ident[-1][0]
    sh10, _ = load_transformed_trajectory(p, "shift10")
    assert abs(sh10[10][0] - ident[0][0]) < 1e-9
    cs, _ = load_transformed_trajectory(p, "cluster_shuffle", seed=1)
    assert sorted(cs[7].values()) == sorted(ident[7].values())
    assert cs[7] != ident[7], "cells must swap trajectories"
    lp, _ = load_transformed_trajectory(p, "lowpass")
    a = np.array([d[0] for d in ident]); b = np.array([d[0] for d in lp])
    assert abs(a.mean() - b.mean()) < 1e-3
    assert np.abs(np.diff(b)).max() <= np.abs(np.diff(a)).max() + 1e-9


def test_trajectory_means(tmp_path):
    p, T, cells = _mk(tmp_path)
    m = trajectory_means(p)
    expect_c0 = 0.2 + 0.01 * (T - 1) / 2
    assert abs(m["lam_per_cell"][0] - expect_c0) < 1e-9
    assert abs(m["lam_global"] - (expect_c0 + 0.05)) < 1e-9
    assert abs(m["Lam_global"] - 0.5) < 1e-9
