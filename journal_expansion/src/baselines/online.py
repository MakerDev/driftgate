"""Causal online baselines for lambda control (Phase 4).

All baselines are strictly causal (only past/current observations). They are
label-free unless explicitly named 'labeled'/'oracle'. They control lambda per
edge server; Lambda stays at the fixed default (joint control is a Phase-5
mechanism, not a baseline).

B5  PeriodicProxyGrid  — hold each candidate for a window, keep the best proxy
B6  UCBBandit / EXP3   — bandit over a lambda grid, proxy or labeled reward
B7  EXP3 with soft mixture weights doubles as a mirror-descent-style controller
B10 update-norm signal — provided by the runner as signal 'update_norm', consumed
                         by the standard SelfCalController (no new class needed)
B11 GreedyLabeledOracle — hill-climbs lambda on per-round LABELED accuracy
                         (causal upper bound; NOT the hindsight oracle B1)

Proxy reward (pre-registered, label-free): mean routed confidence on the probe
  = mean_i max_prob( server if client-entropy>eth else client ) — mirrors the
deployed routing rule, so the bandit optimizes what the system actually serves.
"""
import numpy as np

LAMBDA_GRID = [0.15, 0.30, 0.45, 0.60]


class PeriodicProxyGrid:
    """B5: cycle through the grid, holding each arm `hold` rounds; after each
    full sweep, exploit the best-proxy arm for `exploit` rounds, then re-sweep."""

    def __init__(self, es_ids, grid=None, hold=5, exploit=20):
        self.grid = list(grid or LAMBDA_GRID)
        self.hold = hold
        self.exploit = exploit
        self.state = {es: {"phase": "sweep", "arm": 0, "t": 0,
                           "scores": [[] for _ in self.grid]} for es in es_ids}

    def select(self):
        out = {}
        for es, st in self.state.items():
            if st["phase"] == "sweep":
                out[es] = self.grid[st["arm"]]
            else:
                out[es] = self.grid[st["best"]]
        return out

    def observe(self, rewards):
        for es, r in rewards.items():
            st = self.state[es]
            st["t"] += 1
            if st["phase"] == "sweep":
                st["scores"][st["arm"]].append(r)
                if st["t"] >= self.hold:
                    st["t"] = 0
                    st["arm"] += 1
                    if st["arm"] >= len(self.grid):
                        means = [np.mean(s) if s else -1e9 for s in st["scores"]]
                        st["best"] = int(np.argmax(means))
                        st["phase"] = "exploit"
            else:
                if st["t"] >= self.exploit:
                    st.update(phase="sweep", arm=0, t=0,
                              scores=[[] for _ in self.grid])


class UCBBandit:
    """B6: UCB1 per edge server over the lambda grid."""

    def __init__(self, es_ids, grid=None, c=0.5):
        self.grid = list(grid or LAMBDA_GRID)
        self.c = c
        self.n = {es: np.zeros(len(self.grid)) for es in es_ids}
        self.s = {es: np.zeros(len(self.grid)) for es in es_ids}
        self._last = {}

    def select(self):
        out = {}
        for es in self.n:
            n, s = self.n[es], self.s[es]
            t = n.sum() + 1
            if (n == 0).any():
                a = int(np.argmin(n))
            else:
                ucb = s / n + self.c * np.sqrt(np.log(t) / n)
                a = int(np.argmax(ucb))
            self._last[es] = a
            out[es] = self.grid[a]
        return out

    def observe(self, rewards):
        for es, r in rewards.items():
            a = self._last.get(es)
            if a is None:
                continue
            self.n[es][a] += 1
            self.s[es][a] += r


class EXP3:
    """B6b/B7: EXP3 adversarial bandit; its softmax weights over the grid also
    serve as a mirror-descent-style continuous controller via the weighted
    mean lambda (mode='mean') or sampled arm (mode='sample')."""

    def __init__(self, es_ids, grid=None, eta=0.1, gamma=0.1, mode="sample", seed=0):
        self.grid = list(grid or LAMBDA_GRID)
        self.eta = eta
        self.gamma = gamma
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.w = {es: np.zeros(len(self.grid)) for es in es_ids}  # log-weights
        self._rbar = {es: None for es in es_ids}  # running reward baseline
        self._last = {}

    def _probs(self, es):
        w = self.w[es] - self.w[es].max()
        p = np.exp(w)
        p = p / p.sum()
        return (1 - self.gamma) * p + self.gamma / len(self.grid)

    def select(self):
        out = {}
        for es in self.w:
            p = self._probs(es)
            a = int(self.rng.choice(len(self.grid), p=p))
            self._last[es] = (a, p[a])
            if self.mode == "mean":
                out[es] = float(np.dot(p, self.grid))
            else:
                out[es] = self.grid[a]
        return out

    def observe(self, rewards):
        for es, r in rewards.items():
            if es not in self._last:
                continue
            a, pa = self._last[es]
            # advantage vs a running baseline: without it, near-1 rewards for
            # ALL arms make the importance-weighted update rich-get-richer and
            # EXP3 locks onto whichever arm it happened to play early
            if self._rbar[es] is None:
                self._rbar[es] = r
            adv = r - self._rbar[es]
            self._rbar[es] = 0.95 * self._rbar[es] + 0.05 * r
            self.w[es][a] += self.eta * (adv / max(pa, 1e-3))
            self.w[es] -= self.w[es].max()  # keep log-weights bounded


class GreedyLabeledOracle:
    """B11: causal dynamic oracle. Each round it sees the LABELED accuracy of
    the previous round and hill-climbs lambda (+/- step, keep on improvement,
    flip direction on regression). Strong causal upper bound, not deployable."""

    def __init__(self, es_ids, lam0=0.4, step=0.05, lo=0.15, hi=0.7):
        self.lam = {es: lam0 for es in es_ids}
        self.dir = {es: -1.0 for es in es_ids}
        self.step = step
        self.lo, self.hi = lo, hi
        self.prev_acc = {es: None for es in es_ids}

    def select(self):
        return dict(self.lam)

    def observe(self, acc_per_es):
        for es, acc in acc_per_es.items():
            prev = self.prev_acc.get(es)
            if prev is not None:
                if acc < prev - 1e-4:
                    self.dir[es] *= -1.0
                self.lam[es] = float(np.clip(self.lam[es] + self.dir[es] * self.step,
                                             self.lo, self.hi))
            self.prev_acc[es] = acc


def proxy_reward_from_signals(sig_per_es, eth_conf=None):
    """Pre-registered label-free proxy: routed confidence.
    Approximated from recorded per-ES aggregates: conf = max(conf_client,
    conf_server) weighted by agreement — implemented as
      reward = conf_server * (1 - delta_hard) + conf_client * delta_hard * 0.5
    (when heads disagree, at most one is right; discount)."""
    out = {}
    for es in sig_per_es["conf_server"]:
        cs = sig_per_es["conf_server"][es]
        cc = sig_per_es["conf_client"][es]
        d = sig_per_es["delta_hard"][es]
        out[es] = cs * (1 - d) + 0.5 * cc * d
    return out
