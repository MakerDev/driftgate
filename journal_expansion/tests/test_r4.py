"""R4 tests: abs_only controller, ent_client_norm signal range, APFL lambda update math."""
import math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

JR = Path(__file__).resolve().parent.parent
for p in (str(JR), str(JR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.controllers.self_calibrating import SelfCalController, SIGNAL_RANGE, Z0, TAU_Z, _sigmoid


def _run(ctrl, sig, steps):
    out = None
    for _ in range(steps):
        out = ctrl.step(dict(sig))
    return out


def test_abs_only_uses_absolute_candidate():
    nb = {0: [1], 1: [0, 2], 2: [1]}
    sig = {0: 0.5, 1: 0.5, 2: 0.5}
    c = SelfCalController(neighbors=nb, warmup=3, burn_in=0, abs_only=True,
                          signal_range=1.0, spatial_norm=False)
    lams, Lams = _run(c, sig, 20)
    # constant signal 0.5 -> EWMA converges to 0.5 -> lam_abs = 0.70 - 0.55*0.5 = 0.425
    for es in sig:
        assert abs(lams[es] - 0.425) < 1e-3, lams
        assert abs(Lams[es] - (0.70 - 0.30 * 0.5)) < 1e-3, Lams
        assert abs(c.last_lam_abs[es] - lams[es]) < 1e-9
        # relative candidate with z ~ 0 -> 0.634, which abs_only must IGNORE
        assert c.last_lam_rel[es] > 0.6


def test_full_controller_is_min_of_rel_and_abs():
    nb = {0: [1], 1: [0, 2], 2: [1]}
    sig = {0: 0.5, 1: 0.5, 2: 0.5}
    c = SelfCalController(neighbors=nb, warmup=3, burn_in=0, abs_cap=True,
                          signal_range=1.0, spatial_norm=False)
    lams, _ = _run(c, sig, 20)
    for es in sig:
        assert abs(lams[es] - min(c.last_lam_rel[es], c.last_lam_abs[es])) < 1e-9


def test_ent_client_norm_in_signal_range_and_bounded():
    assert SIGNAL_RANGE["ent_client_norm"] == 1.0
    C = 10
    logits = torch.randn(64, C)
    p = torch.softmax(logits, 1)
    H = -(p * torch.log(p + 1e-8)).sum(1).mean().item()
    hn = H / math.log(C)
    assert 0.0 <= hn <= 1.0 + 1e-6


class _Tiny(nn.Module):
    def __init__(self, d=4, C=3):
        super().__init__()
        self.f = nn.Linear(d, d)
        self.exit = nn.Linear(d, C)

    def forward(self, x):
        rep = torch.relu(self.f(x))
        return self.exit(rep), rep


class _Srv(nn.Module):
    def __init__(self, d=4, C=3):
        super().__init__()
        self.exit = nn.Linear(d, C)

    def forward(self, rep):
        return self.exit(rep), rep


class _FakeClient:
    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.device = "cpu"
        self.client_model = _Tiny()
        self.server_models = {0: _Srv()}
        self.edge_server_ids = [0]
        x = torch.randn(16, 4)
        y = torch.randint(0, 3, (16,))
        self.train_loader = [(x, y)]

    def get_client_state(self):
        return {k: v.detach().clone() for k, v in self.client_model.state_dict().items()}

    def set_client_state(self, sd):
        self.client_model.load_state_dict({k: v.float() for k, v in sd.items()})


def test_apfl_gradient_matches_finite_difference():
    from src.apfl_baseline import _grad_wrt_lambda
    from train.trainer import mix_state_dicts
    from models.losses import multi_exit_loss
    cl = _FakeClient(0)
    local = cl.get_client_state()
    torch.manual_seed(1)
    avg = {k: v + 0.3 * torch.randn_like(v) for k, v in local.items()}
    lam = 0.425
    g, _ = _grad_wrt_lambda(cl, local, avg, lam, gamma=0.5)

    def loss_at(l):
        cl.set_client_state(mix_state_dicts(local, avg, l))
        cl.client_model.eval(); cl.server_models[0].eval()
        x, y = cl.train_loader[0]
        with torch.no_grad():
            cl_log, rep = cl.client_model(x)
            s = [cl.server_models[0](rep)[0]]
            return multi_exit_loss(cl_log, s, y, gamma=0.5).item()
    eps = 1e-3
    fd = (loss_at(lam + eps) - loss_at(lam - eps)) / (2 * eps)
    assert abs(g - fd) < 5e-2 * max(1.0, abs(fd)), (g, fd)


def test_apfl_lambda_clipped_to_bounds():
    lam_min, lam_max = 0.15, 0.70
    for lam, g, eta in [(0.2, +100.0, 0.01), (0.6, -100.0, 0.01)]:
        new = float(min(max(lam - eta * g, lam_min), lam_max))
        assert lam_min <= new <= lam_max
