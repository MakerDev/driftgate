"""S4 — causal temporal normalizers for drift signals.

All normalizers are strictly causal: the value at round t uses only
observations from rounds <= t. The online classes process one value at a
time; offline replicas in src/evaluation/signal_metrics.py must produce
bit-identical series (tested in tests/test_journal.py).

Design (pre-registered, dimensionless — the anti-leakage core):
  GuardedRobustNormalizer
    - warm-up: first W observations -> baseline mu = median, scale
      sigma = 1.4826 * MAD (floored).
    - after warm-up: z_t = (x_t - mu) / sigma.
      Baseline adapts by EWMA (beta) ONLY while z_t < z_guard_hi, i.e. it
      tracks slow convergence-driven decline but FREEZES under upward drift,
      preserving sustained-drift retention. No dataset-specific constants.
"""
import numpy as np
from collections import deque

EPS = 1e-8
SIGMA_FLOOR = 1e-3
# Relative noise floor: sigma >= REL_FLOOR * |baseline|. A short warm-up MAD
# can badly underestimate probe-sampling noise; without this floor the
# controller amplifies probe noise into lambda oscillation (audit item 9).
# Dimensionless and pre-registered BEFORE any benchmark run.
REL_FLOOR = 0.10


def _sigma_min(mu):
    return max(SIGMA_FLOOR, REL_FLOOR * max(abs(mu), 0.05))


def _mad_sigma(vals):
    med = float(np.median(vals))
    mad = float(np.median(np.abs(np.asarray(vals) - med)))
    return med, max(1.4826 * mad, _sigma_min(med))


class GuardedRobustNormalizer:
    """Primary self-calibrating normalizer (used by the online controller).

    burn_in: observations discarded entirely before the warm-up window starts.
    Gate-B pilot traces showed the first ~10 rounds carry the untrained-model
    transient (delta ~1.0 -> ~0.45); including them in warm-up inflates the MAD
    scale so much that a +73% drift jump maps to z~1.5 (controller v1 = burn_in 0,
    contaminated; controller v3b = burn_in 10 — see phase2 report §normalizers).
    """

    def __init__(self, warmup=15, beta=0.05, z_guard_hi=1.5, z_clip=(-2.0, 6.0),
                 burn_in=0):
        self.warmup = warmup
        self.beta = beta
        self.z_guard_hi = z_guard_hi
        self.z_clip = z_clip
        self.burn_in = burn_in
        self._warm_vals = []
        self.mu = None
        self.sigma = None
        self.n = 0

    @property
    def ready(self):
        return self.mu is not None

    def update(self, x):
        """Feed one observation; returns z (0.0 during burn-in + warm-up)."""
        self.n += 1
        if self.n <= self.burn_in:
            return 0.0
        if self.mu is None:
            self._warm_vals.append(float(x))
            if len(self._warm_vals) >= self.warmup:
                self.mu, self.sigma = _mad_sigma(self._warm_vals)
                self._warm_vals = None
            return 0.0
        z = (float(x) - self.mu) / max(self.sigma, _sigma_min(self.mu))
        if z < self.z_guard_hi:  # adapt only when not drift-elevated
            self.mu = (1 - self.beta) * self.mu + self.beta * float(x)
            dev = abs(float(x) - self.mu)
            self.sigma = max((1 - self.beta) * self.sigma + self.beta * 1.4826 * dev,
                             _sigma_min(self.mu))
        return float(np.clip(z, *self.z_clip))


class AnchoredRobustNormalizer:
    """Baseline frozen at warm-up stats (no adaptation). Comparison variant."""

    def __init__(self, warmup=15, z_clip=(-2.0, 6.0)):
        self.warmup = warmup
        self.z_clip = z_clip
        self._warm_vals = []
        self.mu = None
        self.sigma = None

    def update(self, x):
        if self.mu is None:
            self._warm_vals.append(float(x))
            if len(self._warm_vals) >= self.warmup:
                self.mu, self.sigma = _mad_sigma(self._warm_vals)
            return 0.0
        return float(np.clip((float(x) - self.mu) / max(self.sigma, _sigma_min(self.mu)),
                             *self.z_clip))


class EWMANormalizer:
    """z against an unguarded EWMA baseline (comparison variant)."""

    def __init__(self, warmup=15, beta=0.05, z_clip=(-2.0, 6.0)):
        self.warmup = warmup
        self.beta = beta
        self.z_clip = z_clip
        self._warm_vals = []
        self.mu = None
        self.sigma = None

    def update(self, x):
        x = float(x)
        if self.mu is None:
            self._warm_vals.append(x)
            if len(self._warm_vals) >= self.warmup:
                self.mu, self.sigma = _mad_sigma(self._warm_vals)
            return 0.0
        z = (x - self.mu) / max(self.sigma, _sigma_min(self.mu))
        self.mu = (1 - self.beta) * self.mu + self.beta * x
        dev = abs(x - self.mu)
        self.sigma = max((1 - self.beta) * self.sigma + self.beta * 1.4826 * dev,
                         _sigma_min(self.mu))
        return float(np.clip(z, *self.z_clip))


class RollingQuantileNormalizer:
    """Exceedance over the rolling q-quantile of a trailing window."""

    def __init__(self, warmup=15, window=40, q=0.75):
        self.warmup = warmup
        self.window = window
        self.q = q
        self.buf = deque(maxlen=window)

    def update(self, x):
        x = float(x)
        if len(self.buf) < self.warmup:
            self.buf.append(x)
            return 0.0
        ref = float(np.quantile(list(self.buf), self.q))
        scale = float(np.quantile(list(self.buf), 0.9) -
                      np.quantile(list(self.buf), 0.1)) + EPS
        out = (x - ref) / scale
        self.buf.append(x)
        return out


class CUSUMNormalizer:
    """One-sided CUSUM of guarded-z increments (accumulated change)."""

    def __init__(self, warmup=15, k=0.5, decay=0.9, inner=None):
        self.inner = inner or GuardedRobustNormalizer(warmup=warmup)
        self.k = k
        self.decay = decay
        self.s = 0.0

    def update(self, x):
        z = self.inner.update(x)
        self.s = max(0.0, self.decay * self.s + z - self.k)
        return self.s


NORMALIZERS = {
    "guarded": GuardedRobustNormalizer,
    "anchored": AnchoredRobustNormalizer,
    "ewma": EWMANormalizer,
    "rollq": RollingQuantileNormalizer,
    "cusum": CUSUMNormalizer,
}
