"""Round 6: DriftGate (relonly) computed edge by edge with explicit messages.

Same constants and building blocks as SelfCalController (spatial_norm=True,
abs_cap=False, abs_only=False, consensus_steps=1): GuardedRobustNormalizer
(warm-up 15, burn-in 10, beta 0.05, guard 0.5, clip [-2, 6]), EWMA 0.3 on the
temporal and on the spatial score, max of the two, one neighbour average,
lambda/Lambda from sigmoid((q - 1.5)/0.75). With every message delivered and
every edge present it returns bit-identical (lambda, Lambda) to
SelfCalController (tested in journal_expansion/tests/test_r6.py).

What is new (R6 directive §3.7, §S4):
  - an edge that received no client TV this round keeps lambda, Lambda, q and its
    normalizer state (no update at all);
  - lost_dbar[w, z] = True: edge z did not receive edge w's d-bar -> z's spatial
    score uses only the d-bars it has (its own + received), median/MAD over those;
    fewer than 3 values -> temporal score only (same rule as the original);
  - lost_nbr[w, z] = True: edge z did not receive neighbour w's score -> z averages
    its own score with the received ones only.
Per-edge compute time (ns) is measured for §5.1.
"""
import time

from src.controllers.normalizers import NORMALIZERS
from src.controllers.self_calibrating import EWMA_ALPHA, TAU_Z, Z0, _sigmoid, spatial_z


class EdgeDriftGate:
    def __init__(self, L, neighbors, lam_min=0.15, lam_max=0.70, Lam_min=0.40, Lam_max=0.70,
                 warmup=15, burn_in=10, z_guard=0.5, spatial_norm=True, normalizer="guarded"):
        self.L = L
        self.neighbors = {z: list(neighbors.get(z, [])) for z in range(L)}
        self.lam_min, self.lam_max, self.Lam_min, self.Lam_max = lam_min, lam_max, Lam_min, Lam_max
        self.warmup_total = warmup + burn_in
        self.spatial_norm = spatial_norm
        kw = {"warmup": warmup, "burn_in": burn_in}
        if z_guard is not None:
            kw["z_guard_hi"] = z_guard
        self._norms = {z: NORMALIZERS[normalizer](**kw) for z in range(L)}
        self._z_smooth = {z: 0.0 for z in range(L)}
        self._zsp_smooth = {z: 0.0 for z in range(L)}
        self._n_obs = {z: 0 for z in range(L)}
        self.lam = {z: 0.5 * (lam_min + lam_max) for z in range(L)}
        self.Lam = {z: 0.5 * (Lam_min + Lam_max) for z in range(L)}
        self.q = {z: 0.0 for z in range(L)}
        self.last = {}

    def step(self, dbar, lost_dbar=None, lost_nbr=None):
        """dbar: {edge: d-bar} for edges that received >= 1 client TV this round.
        lost_dbar / lost_nbr: [L, L] bool (sender, receiver) or None (no loss)."""
        s_temp, s_spat, zhat, t_ns = {}, {}, {}, {z: 0 for z in range(self.L)}
        present = [z for z in range(self.L) if z in dbar]
        # pass 1: temporal + spatial score per edge
        for z in present:
            t0 = time.perf_counter_ns()
            z_raw = self._norms[z].update(dbar[z])
            self._n_obs[z] += 1
            zt = (1 - EWMA_ALPHA) * self._z_smooth[z] + EWMA_ALPHA * z_raw
            self._z_smooth[z] = zt
            s_temp[z] = z_raw
            zh = zt
            if self.spatial_norm:
                known = {w: dbar[w] for w in present
                         if w == z or lost_dbar is None or not lost_dbar[w, z]}
                if len(known) >= 3:
                    sp = spatial_z(known)[z]
                    s_spat[z] = sp
                    sm = (1 - EWMA_ALPHA) * self._zsp_smooth[z] + EWMA_ALPHA * sp
                    self._zsp_smooth[z] = sm
                    zh = max(zt, sm)
            zhat[z] = zh
            t_ns[z] += time.perf_counter_ns() - t0
        # pass 2: one neighbour average, then lambda / Lambda
        for z in present:
            t0 = time.perf_counter_ns()
            nb = [zhat[w] for w in self.neighbors[z]
                  if w in zhat and (lost_nbr is None or not lost_nbr[w, z])]
            q = (zhat[z] + sum(nb)) / (1 + len(nb))
            self.q[z] = q
            if self._n_obs[z] <= self.warmup_total:
                lam, Lam = 0.5 * (self.lam_min + self.lam_max), 0.5 * (self.Lam_min + self.Lam_max)
            else:
                g = _sigmoid((q - Z0) / TAU_Z)
                lam = self.lam_max - (self.lam_max - self.lam_min) * g
                Lam = self.Lam_max - (self.Lam_max - self.Lam_min) * g
            self.lam[z], self.Lam[z] = lam, Lam
            t_ns[z] += time.perf_counter_ns() - t0
        self.last = dict(s_temp=s_temp, s_spat=s_spat,
                         m={z: self._norms[z].mu for z in range(self.L)},
                         present=present, t_ns=t_ns)
        return dict(self.lam), dict(self.Lam)
