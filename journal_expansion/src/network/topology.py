"""Phase 6 — edge-server topologies (abstract, reproducible network model).

These are ABSTRACTIONS of a multi-edge deployment (documented as such — no
PHY-layer claims). Each topology returns an undirected neighbor map used by
the consensus step; dynamic topologies vary per round.
"""
import numpy as np


def _symmetrize(nbrs, n):
    out = {i: set() for i in range(n)}
    for i, ns in nbrs.items():
        for j in ns:
            if j != i:
                out[i].add(j)
                out[j].add(i)
    return {i: sorted(v) for i, v in out.items()}


def make_topology(name, num_es, seed=0, round_idx=0, radius=0.55):
    if name == "line":
        return _symmetrize({i: [i + 1] for i in range(num_es - 1)}, num_es)
    if name == "ring":
        return _symmetrize({i: [(i + 1) % num_es] for i in range(num_es)}, num_es)
    if name == "star":
        return _symmetrize({0: list(range(1, num_es))}, num_es)
    if name == "grid":
        side = int(np.ceil(np.sqrt(num_es)))
        nbrs = {}
        for i in range(num_es):
            r, c = divmod(i, side)
            ns = []
            if c + 1 < side and i + 1 < num_es:
                ns.append(i + 1)
            if r + 1 < side and i + side < num_es:
                ns.append(i + side)
            nbrs[i] = ns
        return _symmetrize(nbrs, num_es)
    if name in ("rgg", "dynamic"):
        # random geometric graph; 'dynamic' resamples positions every 10 rounds
        eff_seed = seed if name == "rgg" else seed * 10007 + (round_idx // 10)
        rng = np.random.default_rng(eff_seed)
        pos = rng.uniform(0, 1, (num_es, 2))
        nbrs = {i: [j for j in range(num_es) if j != i
                    and np.linalg.norm(pos[i] - pos[j]) < radius]
                for i in range(num_es)}
        out = _symmetrize(nbrs, num_es)
        # keep connected-ish: attach isolated nodes to their nearest neighbor
        for i in range(num_es):
            if not out[i]:
                d = np.linalg.norm(pos - pos[i], axis=1)
                d[i] = np.inf
                j = int(np.argmin(d))
                out[i] = [j]
                out[j] = sorted(set(out[j]) | {i})
        return out
    raise ValueError(f"unknown topology {name}")


def graph_stats(nbrs):
    """degree stats + spectral gap of the consensus (averaging) matrix."""
    n = len(nbrs)
    W = np.zeros((n, n))
    for i, ns in nbrs.items():
        deg = len(ns) + 1
        W[i, i] = 1.0 / deg
        for j in ns:
            W[i, j] = 1.0 / deg
    ev = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    return {
        "mean_degree": float(np.mean([len(v) for v in nbrs.values()])),
        "min_degree": int(min(len(v) for v in nbrs.values())),
        "spectral_gap": float(1.0 - ev[1]) if n > 1 else 1.0,
    }
