"""Tests for Phase-6 network abstractions."""
import sys
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))

from src.network.topology import make_topology, graph_stats
from src.network.impairments import SignalChannel, ParticipationSampler


def test_topologies_valid():
    for name in ("line", "ring", "grid", "star", "rgg", "dynamic"):
        nbrs = make_topology(name, 6, seed=1)
        assert set(nbrs) == set(range(6))
        # symmetric
        for i, ns in nbrs.items():
            for j in ns:
                assert i in nbrs[j]
        st = graph_stats(nbrs)
        assert 0.0 <= st["spectral_gap"] <= 1.0
        assert st["min_degree"] >= 1


def test_star_vs_line_spectral_gap():
    # star mixes faster than line
    gs = graph_stats(make_topology("star", 8))
    gl = graph_stats(make_topology("line", 8))
    assert gs["spectral_gap"] > gl["spectral_gap"]


def test_dynamic_topology_changes():
    a = make_topology("dynamic", 6, seed=1, round_idx=0)
    b = make_topology("dynamic", 6, seed=1, round_idx=50)
    assert a == make_topology("dynamic", 6, seed=1, round_idx=5)  # stable within window
    assert a != b or make_topology("dynamic", 6, seed=1, round_idx=20) != a


def test_signal_channel_delay():
    ch = SignalChannel(delay=2, loss=0.0, seed=0)
    got = []
    for t in range(6):
        out, ages = ch.send_and_receive({0: float(t)})
        got.append(out.get(0))
    # value t arrives at t+2
    assert got[:2] == [None, None] or 0 not in got[:2] or got[0] is None
    assert got[2] == 0.0 and got[3] == 1.0 and got[5] == 3.0


def test_signal_channel_loss_holds_last():
    ch = SignalChannel(delay=0, loss=1.0, seed=0)
    ch2 = SignalChannel(delay=0, loss=0.0, seed=0)
    out2, _ = ch2.send_and_receive({0: 5.0})
    assert out2[0] == 5.0
    # full loss: nothing ever delivered -> receiver has no value
    out, ages = ch.send_and_receive({0: 5.0})
    assert 0 not in out
    # partial loss: after a delivery, later losses HOLD the last value
    ch3 = SignalChannel(delay=0, loss=0.0, seed=0)
    ch3.send_and_receive({0: 1.0})
    ch3.loss = 1.0
    out3, ages3 = ch3.send_and_receive({0: 99.0})
    assert out3[0] == 1.0 and ages3[0] == 1


def test_participation_sampler():
    ps = ParticipationSampler(0.4, seed=0)
    clients = list(range(50))
    s1 = ps.sample(clients)
    s2 = ps.sample(clients)
    assert len(s1) == 20
    assert s1 != s2  # resampled per round (with overwhelming probability)
