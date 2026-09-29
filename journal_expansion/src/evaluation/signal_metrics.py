"""Signal-quality metrics: evaluate drift signals AS DETECTORS, independent of
downstream accuracy.

Inputs are per-round series: signal x_t (raw or normalized) and true rho_t
(environment ground truth — used ONLY here, offline, for evaluation; never
available to any controller).

Offline causal normalizers here replay the exact online classes from
src/controllers/normalizers.py one value at a time, so 'normalized signal'
metrics reflect precisely what an online controller would have seen.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.controllers.normalizers import NORMALIZERS

EPS = 1e-12


def causal_normalize(x, kind="guarded", warmup=15, **kw):
    norm = NORMALIZERS[kind](warmup=warmup, **kw)
    return np.array([norm.update(v) for v in x])


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    return avg[inv]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < EPS or y.std() < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(_rankdata(x), _rankdata(y))


def auroc(scores, labels):
    """labels: 1 = drift round, 0 = no-drift round."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def auprc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    rec = tp / lab.sum()
    return float(np.trapezoid(prec, rec))


def detection_delay(z, onset_round, z_thresh=2.0, sustain=3, horizon=None):
    """Rounds after onset until z > z_thresh for `sustain` consecutive rounds.
    z is 0-indexed per round (round r -> z[r-1]). Returns np.inf if never."""
    T = len(z) if horizon is None else min(horizon, len(z))
    run = 0
    for r in range(onset_round - 1, T):
        run = run + 1 if z[r] > z_thresh else 0
        if run >= sustain:
            return (r + 1) - onset_round - (sustain - 1)
    return float("inf")


def false_alarm_rate(z, no_drift_mask, z_thresh=2.0, warmup=15):
    m = np.asarray(no_drift_mask, bool).copy()
    m[:warmup] = False
    if m.sum() == 0:
        return float("nan")
    return float((np.asarray(z)[m] > z_thresh).mean())


def sustained_retention(x, drift_mask, min_len=12):
    """Late-vs-early signal level inside the longest sustained drift segment.
    ~1.0 = retained; << 1 = decayed. Uses the RAW signal."""
    drift_mask = np.asarray(drift_mask, bool)
    # longest contiguous run
    best = (0, 0)
    i = 0
    while i < len(drift_mask):
        if drift_mask[i]:
            j = i
            while j < len(drift_mask) and drift_mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    s, e = best
    if e - s < min_len:
        return float("nan")
    seg = np.asarray(x, float)[s:e]
    third = max(len(seg) // 3, 1)
    early, late = seg[:third].mean(), seg[-third:].mean()
    return float(late / (abs(early) + EPS))


def recovery_delay(z, offset_round, z_thresh=1.0, sustain=3):
    """Rounds after drift END until z < z_thresh sustained."""
    run = 0
    for r in range(offset_round - 1, len(z)):
        run = run + 1 if z[r] < z_thresh else 0
        if run >= sustain:
            return (r + 1) - offset_round - (sustain - 1)
    return float("inf")


def seed_stability(series_list):
    """Mean pairwise Pearson correlation of the same signal across seeds."""
    cors = []
    for i in range(len(series_list)):
        for j in range(i + 1, len(series_list)):
            n = min(len(series_list[i]), len(series_list[j]))
            cors.append(pearson(series_list[i][:n], series_list[j][:n]))
    return float(np.mean(cors)) if cors else float("nan")


def cluster_ranking_corr(sig_per_cell, rho_per_cell):
    """Spearman between per-cell signal and per-cell true rho (spatial runs)."""
    cells = sorted(set(sig_per_cell) & set(rho_per_cell))
    if len(cells) < 3:
        return float("nan")
    return spearman([sig_per_cell[c] for c in cells],
                    [rho_per_cell[c] for c in cells])


def evaluate_signal(x, rho, warmup=15, drift_level=0.5, normalizer="guarded"):
    """Full metric bundle for one signal series against a rho series.

    Correlation metrics are computed post-warmup on the RAW signal;
    detection metrics on the causally-normalized signal.
    """
    x = np.asarray(x, float)
    rho = np.asarray(rho, float)
    T = min(len(x), len(rho))
    x, rho = x[:T], rho[:T]
    drift_mask = rho >= drift_level
    z = causal_normalize(x, kind=normalizer, warmup=warmup)

    out = {
        "spearman_rho": spearman(x[warmup:], rho[warmup:]),
        "pearson_rho": pearson(x[warmup:], rho[warmup:]),
        "auroc_raw": auroc(x[warmup:], drift_mask[warmup:]),
        "auroc_norm": auroc(z[warmup:], drift_mask[warmup:]),
        "auprc_norm": auprc(z[warmup:], drift_mask[warmup:]),
        "false_alarm_rate": false_alarm_rate(z, ~drift_mask, warmup=warmup),
        "retention_raw": sustained_retention(x, drift_mask),
        "signal_std_nodrift": float(x[warmup:][~drift_mask[warmup:]].std())
        if (~drift_mask[warmup:]).sum() > 1 else float("nan"),
    }
    # onset/offset of first sustained drift segment
    idx = np.where(drift_mask)[0]
    if len(idx) > 0:
        onset = int(idx[0]) + 1
        seg_end = onset
        while seg_end - 1 < T and drift_mask[seg_end - 1]:
            seg_end += 1
        out["detection_delay"] = detection_delay(z, onset)
        if seg_end <= T:
            out["recovery_delay"] = recovery_delay(z, seg_end)
    return out
