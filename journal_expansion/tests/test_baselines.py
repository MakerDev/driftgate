"""Tests for the Phase-4 causal online baselines."""
import sys
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))

from src.baselines.online import (
    PeriodicProxyGrid, UCBBandit, EXP3, GreedyLabeledOracle,
    proxy_reward_from_signals, LAMBDA_GRID,
)


def _env_reward(lam, best=0.30):
    """Synthetic environment: reward peaks at lam=best."""
    return 1.0 - (lam - best) ** 2 * 4


def test_ucb_finds_best_arm():
    rng = np.random.default_rng(0)
    b = UCBBandit([0])
    picks = []
    for t in range(400):
        lam = b.select()[0]
        picks.append(lam)
        b.observe({0: _env_reward(lam) + rng.normal(0, 0.02)})
    late = picks[-100:]
    assert np.mean([p == 0.30 for p in late]) > 0.7, \
        f"UCB failed to converge to the best arm: {np.unique(late, return_counts=True)}"


def test_exp3_prefers_best_arm():
    rng = np.random.default_rng(1)
    b = EXP3([0], eta=0.2, gamma=0.1, seed=1)
    for t in range(600):
        lam = b.select()[0]
        b.observe({0: _env_reward(lam) + rng.normal(0, 0.02)})
    p = b._probs(0)
    assert p[LAMBDA_GRID.index(0.30)] == p.max()


def test_proxy_grid_sweeps_then_exploits():
    g = PeriodicProxyGrid([0], hold=2, exploit=5)
    lams = []
    for t in range(20):
        lam = g.select()[0]
        lams.append(lam)
        g.observe({0: _env_reward(lam)})
    # sweep covers all arms
    assert set(lams[:8]) == set(LAMBDA_GRID)
    # exploit phase picks the best arm
    assert lams[9] == 0.30


def test_greedy_oracle_climbs():
    o = GreedyLabeledOracle([0], lam0=0.6, step=0.05)
    for t in range(60):
        lam = o.select()[0]
        o.observe({0: _env_reward(lam)})
    assert abs(o.select()[0] - 0.30) <= 0.1, f"oracle stuck at {o.select()[0]}"


def test_oracle_is_causal_api():
    import inspect
    params = set(inspect.signature(GreedyLabeledOracle.observe).parameters)
    assert "future" not in params
    # observe only takes the CURRENT measured accuracy — nothing schedule-shaped
    assert params == {"self", "acc_per_es"}


def test_proxy_reward_direction():
    sig = {"conf_server": {0: 0.9}, "conf_client": {0: 0.9}, "delta_hard": {0: 0.0}}
    high = proxy_reward_from_signals(sig)[0]
    sig2 = {"conf_server": {0: 0.5}, "conf_client": {0: 0.5}, "delta_hard": {0: 0.8}}
    low = proxy_reward_from_signals(sig2)[0]
    assert high > low
