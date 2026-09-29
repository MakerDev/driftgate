"""Trajectory-replay transforms for the adaptation-value decomposition (Phase B).

Loads the (lambda, Lambda) per-round per-cell trajectories recorded by a DV run
and applies value-preserving interventions that destroy specific structure:

  identity        — open-loop replay (sanity control)
  shuffle         — random time order (destroys ALL temporal alignment)
  shift10/shift20 — circular delay by k rounds (mis-aligns timing)
  reverse         — reversed time order
  cluster_shuffle — cells receive each other's trajectories (destroys spatial
                    assignment, keeps each trajectory's shape)
  lowpass         — centered moving average, window 15 (removes fast changes;
                    uses FUTURE values — analysis-only, never called deployable)

All transforms preserve the marginal distribution of lambda values (except
lowpass, which preserves the mean); differences vs the closed-loop DV run
therefore isolate the value of temporal/spatial ALIGNMENT.
"""
import json
import numpy as np


def load_transformed_trajectory(path, transform, seed=0, num_es=5):
    h = json.load(open(path))
    lam = [{int(k): float(v) for k, v in d.items()} for d in h["lamdas"] if d]
    Lam = [{int(k): float(v) for k, v in d.items()} for d in h["big_lamdas"] if d]
    T = len(lam)
    rng = np.random.default_rng(1234 + seed)

    if transform == "identity":
        return lam, Lam
    if transform == "shuffle":
        perm = rng.permutation(T)
        return [lam[i] for i in perm], [Lam[i] for i in perm]
    if transform.startswith("shift"):
        k = int(transform[5:])
        idx = [(i - k) % T for i in range(T)]
        return [lam[i] for i in idx], [Lam[i] for i in idx]
    if transform == "reverse":
        return lam[::-1], Lam[::-1]
    if transform == "cluster_shuffle":
        cells = sorted(lam[0].keys())
        perm = list(rng.permutation(cells))
        while any(a == b for a, b in zip(perm, cells)) and len(cells) > 1:
            perm = list(rng.permutation(cells))
        mapping = dict(zip(cells, perm))
        lam2 = [{z: d[mapping[z]] for z in cells} for d in lam]
        Lam2 = [{z: d[mapping[z]] for z in cells} for d in Lam]
        return lam2, Lam2
    if transform == "lowpass":
        w = 15
        cells = sorted(lam[0].keys())
        lam_arr = np.array([[d[z] for z in cells] for d in lam])
        Lam_arr = np.array([[d[z] for z in cells] for d in Lam])
        k = np.ones(w) / w
        def smooth(a):
            return np.stack([np.convolve(a[:, j], k, mode="same") /
                             np.convolve(np.ones(len(a)), k, mode="same")
                             for j in range(a.shape[1])], axis=1)
        ls, Ls = smooth(lam_arr), smooth(Lam_arr)
        return ([{z: float(ls[i, j]) for j, z in enumerate(cells)} for i in range(len(ls))],
                [{z: float(Ls[i, j]) for j, z in enumerate(cells)} for i in range(len(Ls))])
    raise ValueError(f"unknown transform {transform}")


def trajectory_means(path, warmup_included=True):
    """Global and per-cell mean (lambda, Lambda) of a recorded run."""
    h = json.load(open(path))
    lam = [{int(k): float(v) for k, v in d.items()} for d in h["lamdas"] if d]
    Lam = [{int(k): float(v) for k, v in d.items()} for d in h["big_lamdas"] if d]
    cells = sorted(lam[0].keys())
    lam_arr = np.array([[d[z] for z in cells] for d in lam])
    Lam_arr = np.array([[d[z] for z in cells] for d in Lam])
    return {
        "lam_global": float(lam_arr.mean()),
        "Lam_global": float(Lam_arr.mean()),
        "lam_per_cell": {z: float(lam_arr[:, j].mean()) for j, z in enumerate(cells)},
        "Lam_per_cell": {z: float(Lam_arr[:, j].mean()) for j, z in enumerate(cells)},
    }
