"""Tests for Phase-G corruption machinery."""
import sys
from pathlib import Path
import torch
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT))
from src.corruptions import (CORRUPTIONS, severity_schedule, SeverityState,
                             CorruptedView)


def test_severity_zero_is_identity():
    x = torch.randn(3, 32, 32)
    for name, fn in CORRUPTIONS.items():
        y = fn(x.clone(), 0.0, torch.Generator().manual_seed(0))
        assert torch.allclose(x, y, atol=1e-6), name


def test_severity_monotone_distortion():
    x = torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(1))
    for name, fn in CORRUPTIONS.items():
        d_lo = (fn(x.clone(), 0.3, torch.Generator().manual_seed(2)) - x).abs().mean()
        d_hi = (fn(x.clone(), 0.9, torch.Generator().manual_seed(2)) - x).abs().mean()
        assert d_hi >= d_lo - 1e-6, f"{name}: distortion must not shrink with severity"


def test_corrupted_view_deterministic_and_label_preserving():
    xs = torch.randn(10, 3, 32, 32)
    ys = list(range(10))
    class DS(torch.utils.data.Dataset):
        targets = ys
        def __len__(self): return 10
        def __getitem__(self, i): return xs[i], ys[i]
    st = SeverityState(); st.severity = 0.8
    v = CorruptedView(DS(), "gaussian_noise", st, seed=3)
    a1, y1 = v[4]; a2, y2 = v[4]
    assert torch.allclose(a1, a2), "same index+severity must be deterministic"
    assert y1 == 4 == y2, "labels untouched (covariate drift only)"
    st.severity = 0.0
    b, _ = v[4]
    assert torch.allclose(b, xs[4])


def test_severity_schedules():
    assert severity_schedule("abrupt_sev", 10, 100) == 0.0
    assert severity_schedule("abrupt_sev", 60, 100) == 0.8
    g30 = severity_schedule("gradual_sev", 31, 100)
    g60 = severity_schedule("gradual_sev", 61, 100)
    assert 0 <= g30 < g60 <= 0.8
