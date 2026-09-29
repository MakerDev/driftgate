"""Self-calibrating online (lambda, Lambda) controller.

Replaces the absolute-threshold sigmoid (mu_drift/tau_drift — whose history
showed rho-dependent re-centering, see reports/phase1_calibration.md) with a
mapping in normalized z-space. All constants are dimensionless and
PRE-REGISTERED here; they are never tuned per dataset/schedule:

  warm-up W=15 rounds       -> lambda = (lam_min+lam_max)/2 (neutral), collect baseline
  z_t   = GuardedRobustNormalizer(signal)          per ES
  z~_t  = EWMA(z_t, alpha=0.3)                     smoothing
  z^_t  = 1-step neighbor consensus on z~          (same graph as legacy)
  lambda_z  = lam_max - (lam_max-lam_min) * sigmoid((z^ - z0)/tau_z)
  Lambda_z  = Lam_max - (Lam_max-Lam_min) * sigmoid((z^ - z0)/tau_z)
  z0 = 1.5, tau_z = 0.75

so z<=0 (no drift)  -> lambda ~= lam_max  (personalize)
   z>=3  (drift)    -> lambda ~= lam_min  (generalize)

The controller receives ONLY a per-ES scalar signal. It has no access to
labels, rho, or the schedule (enforced by the API surface; tested).

Optional hysteresis (switching cost): move at most max_step per round.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.controllers.normalizers import GuardedRobustNormalizer, NORMALIZERS

Z0 = 1.5
TAU_Z = 0.75
EWMA_ALPHA = 0.3

# Mathematical ranges for the absolute view (Gate-2 signal family). These are
# properties of each divergence, NOT dataset calibration: TV and hard delta are
# probabilities in [0,1]; JS is bounded by ln 2 (nats); logit-cosine distance by
# 2. Symmetric KL is unbounded -> its absolute view is ill-defined (documented;
# expected to be a weakness of that signal, not patched per-dataset).
SIGNAL_RANGE = {"delta_hard": 1.0, "tv_dist": 1.0, "js_div": 0.6931472,
                "cos_logit_dist": 2.0, "kl_sym": None,
                "server_nonmain_soft": 1.0, "server_nonmain_hard": 1.0,
                # R4/B2: client-exit entropy normalized by ln C -> [0,1]
                "ent_client_norm": 1.0}


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def spatial_z(signal_per_es, clip=(-2.0, 6.0)):
    """Cross-cell robust z of the CURRENT round's per-cell signal values."""
    import numpy as np
    vals = np.array(list(signal_per_es.values()), dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    floor = max(1e-3, 0.10 * max(abs(med), 0.05))
    sigma = max(1.4826 * mad, floor)
    return {es: float(np.clip((v - med) / sigma, *clip))
            for es, v in signal_per_es.items()}


def _consensus(vals, neighbors, steps=1):
    cur = dict(vals)
    for _ in range(steps):
        new = {}
        for es, v in cur.items():
            nb = [cur[n] for n in neighbors.get(es, []) if n in cur]
            new[es] = (v + sum(nb)) / (1 + len(nb))
        cur = new
    return cur


class SelfCalController:
    """Self-calibrating controller in z-space (main method candidate)."""

    def __init__(self, neighbors, lam_min=0.15, lam_max=0.7,
                 Lam_min=0.4, Lam_max=0.7, warmup=15,
                 normalizer="guarded", consensus_steps=1,
                 max_step=None, z0=Z0, tau_z=TAU_Z, burn_in=0, z_guard=None,
                 spatial_norm=False, abs_cap=False, signal_range=1.0,
                 abs_only=False):
        self.neighbors = neighbors
        # R4/B1: abs_only -> lambda = lambda_abs, Lambda = Lambda_abs (relative
        # views ignored). Implies the absolute-signal smoothing path.
        self.abs_only = abs_only
        self.last_lam_rel = {}   # R4/A2: per-es relative candidate (this step)
        self.last_lam_abs = {}   # R4/A2: per-es absolute candidate (this step)
        self.lam_min, self.lam_max = lam_min, lam_max
        self.Lam_min, self.Lam_max = Lam_min, Lam_max
        self.warmup = warmup + burn_in  # neutral lambda through burn-in AND warm-up
        self.burn_in = burn_in
        self.consensus_steps = consensus_steps
        self.max_step = max_step
        self.z0, self.tau_z = z0, tau_z
        # v3c: asymmetric guard. The default guard (freeze baseline only at
        # z>1.5) lets a STEPPED ramp be absorbed stage by stage (each step
        # yields z~1-2 briefly, the baseline adapts, the next step starts from
        # the drifted baseline — the boiling-frog failure seen on Schedule A
        # dev runs). z_guard=0.5 adapts the baseline only in stationary or
        # declining regimes: convergence moves the signal DOWN, drift moves it
        # UP, so tracking is kept exactly where it is safe.
        self.z_guard = z_guard
        # Spatio-temporal variant: temporal z answers "did MY traffic change?"
        # which is identically 0 for a cell whose drift is STATIC (present from
        # round 1, e.g. spatial heterogeneity) — nothing ever changes. The
        # spatial z answers "is my cell an outlier vs its peers RIGHT NOW":
        # z_sp = (s_z - median_k s_k) / (1.4826*MAD_k + floor), current round
        # only (causal, label-free, dimensionless). Effective z = max(both).
        self.spatial_norm = spatial_norm
        self._zsp_smooth = {}
        # Dual-view cap (v3d): valid ONLY when the signal is the hard
        # disagreement rate delta (a probability with intrinsic meaning:
        # the fraction of traffic outside the client head's competence).
        # Self-normalization deliberately discards the absolute level, which
        # made it blind to datasets whose STATIC delta is high (D4: CIFAR-100
        # delta~0.6 -> normalized away -> over-personalization, -3.8 pp).
        # lam_abs = lam_max - (lam_max-lam_min) * delta_hat has NO tunable
        # constants (delta is already dimensionless); final lam = min(both
        # views): personalize only if there was no recent drift AND the
        # traffic is mostly within competence.
        self.abs_cap = abs_cap
        self.signal_range = signal_range  # None => abs view disabled (unbounded signal)
        self._sig_smooth = {}
        self._norm_cls = NORMALIZERS[normalizer]
        self._norms = {}     # es_id -> normalizer
        self._z_smooth = {}  # es_id -> EWMA state
        self._n_obs = {}     # es_id -> observation count (warm-up bookkeeping)
        self._prev_lam = {}
        self.last_z = {}

    def _lam_warm(self):
        return 0.5 * (self.lam_min + self.lam_max)

    def _Lam_warm(self):
        return 0.5 * (self.Lam_min + self.Lam_max)

    def step(self, signal_per_es):
        """signal_per_es: {es_id: raw scalar}. Returns (lamdas, big_lamdas)."""
        z = {}
        for es, s in signal_per_es.items():
            if es not in self._norms:
                kw = {"warmup": self.warmup - self.burn_in, "burn_in": self.burn_in}
                if self.z_guard is not None:
                    kw["z_guard_hi"] = self.z_guard
                try:
                    self._norms[es] = self._norm_cls(**kw)
                except TypeError:  # normalizer variant without burn_in/guard support
                    self._norms[es] = self._norm_cls(warmup=self.warmup)
            z_raw = self._norms[es].update(s)
            self._n_obs[es] = self._n_obs.get(es, 0) + 1
            prev = self._z_smooth.get(es, 0.0)
            z[es] = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * z_raw
            self._z_smooth[es] = z[es]

        if self.spatial_norm and len(signal_per_es) >= 3:
            zsp = spatial_z(signal_per_es)
            for es in z:
                prev = self._zsp_smooth.get(es, 0.0)
                sm = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * zsp.get(es, 0.0)
                self._zsp_smooth[es] = sm
                z[es] = max(z[es], sm)

        zc = _consensus(z, self.neighbors, self.consensus_steps)
        self.last_z = dict(zc)

        sig_c = {}
        use_abs = self.abs_cap or self.abs_only
        if use_abs:
            sig_c = _consensus(dict(signal_per_es), self.neighbors,
                               self.consensus_steps)
            for es, v in sig_c.items():
                prev = self._sig_smooth.get(es, v)
                sig_c[es] = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * v
                self._sig_smooth[es] = sig_c[es]

        lamdas, big_lamdas = {}, {}
        self.last_lam_rel, self.last_lam_abs = {}, {}
        for es, zv in zc.items():
            in_warmup = self._n_obs.get(es, 0) <= self.warmup
            if in_warmup:
                lam, Lam = self._lam_warm(), self._Lam_warm()
            else:
                g = _sigmoid((zv - self.z0) / self.tau_z)
                lam_rel = self.lam_max - (self.lam_max - self.lam_min) * g
                Lam_rel = self.Lam_max - (self.Lam_max - self.Lam_min) * g
                lam, Lam = lam_rel, Lam_rel
                self.last_lam_rel[es] = lam_rel
                if use_abs and es in sig_c and self.signal_range:
                    d_hat = min(max(sig_c[es] / self.signal_range, 0.0), 1.0)
                    lam_abs = self.lam_max - (self.lam_max - self.lam_min) * d_hat
                    Lam_abs = self.Lam_max - (self.Lam_max - self.Lam_min) * d_hat
                    self.last_lam_abs[es] = lam_abs
                    if self.abs_only:          # R4/B1: absolute view only
                        lam, Lam = lam_abs, Lam_abs
                    else:                      # full DriftGate: min of both
                        lam = min(lam, lam_abs)
                        Lam = min(Lam, Lam_abs)
            if self.max_step is not None and es in self._prev_lam:
                p = self._prev_lam[es]
                lam = min(max(lam, p - self.max_step), p + self.max_step)
            self._prev_lam[es] = lam
            lamdas[es] = lam
            big_lamdas[es] = Lam
        return lamdas, big_lamdas


class SourceCalibratedController:
    """Absolute sigmoid controller whose (mu, tau) were fitted ONCE on a
    source calibration environment and then applied unchanged to unseen
    datasets/schedules. Identical mapping to the legacy controller; exists to
    separate 'hand-calibrated (leaky)' from 'source-calibrated (transferred)'."""

    def __init__(self, neighbors, mu, tau, lam_min=0.15, lam_max=0.7,
                 Lam_min=0.4, Lam_max=0.7, consensus_steps=1):
        self.neighbors = neighbors
        self.mu, self.tau = mu, tau
        self.lam_min, self.lam_max = lam_min, lam_max
        self.Lam_min, self.Lam_max = Lam_min, Lam_max
        self.consensus_steps = consensus_steps
        self.last_z = {}

    def step(self, signal_per_es):
        sc = _consensus(dict(signal_per_es), self.neighbors, self.consensus_steps)
        lamdas, big_lamdas = {}, {}
        for es, s in sc.items():
            zneg = -(s - self.mu) / max(self.tau, 1e-6)
            g = _sigmoid(zneg)
            lamdas[es] = self.lam_min + (self.lam_max - self.lam_min) * g
            big_lamdas[es] = self.Lam_min + (self.Lam_max - self.Lam_min) * g
            self.last_z[es] = -zneg
        return lamdas, big_lamdas
