"""Drift-schedule registry (traffic-composition drift over rho).

All schedules are fraction-based: rho(r) depends on r/total_rounds, so the same
schedule shape works at any training horizon (pilot 100R, full 150R).

A schedule returns either
  - a scalar rho  (same for every cell), or
  - a dict {es_id: rho_z}  (per-cell, for staggered/spatial-temporal drift).

Terminology note: these schedules change the Main/OOP/OOR *traffic composition*
(distribution drift), not p(y|x); we never call this "concept drift".
"""
import numpy as np

RHO_HI = 0.8


def _piecewise(frac, values):
    n = len(values)
    seg = min(int(frac * n), n - 1)
    return values[seg]


def get_rho(name, round_idx, total_rounds, seed=0, num_cells=5):
    """rho for round_idx (1-indexed) under schedule `name`."""
    frac = (round_idx - 1) / max(total_rounds, 1)

    if name in (None, "static", "constant"):
        return None  # caller uses its config default

    if name == "A":  # ICTC main: 0 -> 0.4 -> 0.8 -> 0.4 -> 0 in equal fifths
        return _piecewise(frac, [0.0, 0.4, RHO_HI, 0.4, 0.0])

    if name == "A150_legacy":  # exact ICTC form: absolute 30-round segments
        seg = (round_idx - 1) // 30
        return [0.0, 0.4, RHO_HI, 0.4, 0.0][min(seg, 4)]

    if name == "B_legacy":  # exact ICTC schedule B
        rng = np.random.default_rng(round_idx // 30 + 1000)
        return float(rng.uniform(0, RHO_HI))

    if name == "abrupt":  # sustained step at half-way
        return 0.0 if frac < 0.5 else RHO_HI

    if name == "recurring":  # alternating no-drift / drift, 5 segments
        return _piecewise(frac, [0.0, RHO_HI, 0.0, RHO_HI, 0.0])

    if name == "burst":  # short burst: only [0.4, 0.5) of the horizon
        return RHO_HI if 0.4 <= frac < 0.5 else 0.0

    if name == "gradual_linear":  # linear ramp between 0.3 and 0.7, then hold
        if frac < 0.3:
            return 0.0
        if frac < 0.7:
            return RHO_HI * (frac - 0.3) / 0.4
        return RHO_HI

    if name == "gradual_sigmoid":
        return float(RHO_HI / (1.0 + np.exp(-(frac - 0.5) / 0.08)))

    if name == "asym_return":  # abrupt onset at 0.3, hold, slow linear return
        if frac < 0.3:
            return 0.0
        if frac < 0.5:
            return RHO_HI
        return float(max(0.0, RHO_HI * (1.0 - (frac - 0.5) / 0.5)))

    if name == "piecewise_random":  # random constant per 10% segment
        seg = min(int(frac * 10), 9)
        rng = np.random.default_rng(seed * 1000 + seg + 7)
        return float(rng.uniform(0, RHO_HI))

    if name == "staggered":  # per-cell onset at frac 0.25 + 0.1*z
        out = {}
        for z in range(num_cells):
            onset = 0.25 + 0.10 * z
            out[z] = RHO_HI if frac >= onset else 0.0
        return out

    raise ValueError(f"Unknown schedule: {name}")


def rho_series(name, total_rounds, seed=0, num_cells=5):
    """Full per-round rho trace. Returns (rounds, list-of-scalar-or-dict)."""
    rounds = list(range(1, total_rounds + 1))
    return rounds, [get_rho(name, r, total_rounds, seed, num_cells) for r in rounds]


# Schedules with a well-defined onset (for detection-delay metrics), as
# fractions of the horizon: (onset_start, onset_end) of the FIRST drift segment.
ONSET_FRACTIONS = {
    "abrupt": (0.5, 1.0),
    "recurring": (0.2, 0.4),
    "burst": (0.4, 0.5),
    "asym_return": (0.3, 0.5),
}

PILOT_SCHEDULES = ["A", "abrupt", "recurring", "burst"]
