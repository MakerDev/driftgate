"""Round 6: partitions, per-cell class groups, and per-client request sets (§1.2, §3.3, §3.6).

Class kinds for client k at round t, relative to its CURRENT cells Z_k^t:
  scope(Z)  = union of the class groups of the cells in Z
              (class group of a cell = union of the Main classes of its residents)
  Main      = the client's own training classes
  OOP       = scope(Z) - Main
  OOR       = all_used - Main - scope(Z)       (all_used = union of every client's Main)

Request composition follows the existing rule (data/partition.py
build_per_client_test_set): all Main samples of the pool, OOP = rho * |Main| split
evenly over the OOP classes, OOR = 0.3 * rho * |Main| split evenly over the OOR
classes; per class the first max(1, int(share / n_classes)) pool samples are used.
§3.6 fallbacks: if OOR is empty its share is drawn from OOP; if OOP is empty its
share is drawn from OOR (flags returned so the frequency can be logged).
"""
import numpy as np

from data.partition import make_es_topology, nd1_partition

OOR_FACTOR = 0.3


def district_partition_seed(seed, d):
    """§S3: district 0 reuses S1's partition (seed); district d>=1 uses seed*1000 + d."""
    return int(seed) if d == 0 else int(seed) * 1000 + int(d)


def build_partition(train_labels, env, seed, num_classes=10, cfg=None):
    """Per-client train indices and Main classes for a Round-6 environment.

    Synthetic layouts (S1/S3/S4): district d runs nd1_partition on the static
    50-client topology with district_partition_seed(seed, d); client with local id l
    gets that district's client-l split (home cell = 5d + l//10).
    Trace layouts (S2): one nd1_partition over all K clients with
    client_to_es[k] = [client_group[k]] (the group rank of the home cell), seed = seed.
    Returns indices {k: [...]}, mains {k: set}, cell_groups {cell: set}, all_used set.
    """
    cfg = cfg or {}
    fr = (cfg.get("classes_per_es_min_frac", 0.4), cfg.get("classes_per_es_max_frac", 0.7))
    fc = cfg.get("classes_per_client_frac", 0.2)
    K = len(env["home_cell"])
    indices, mains = {}, {}
    if "trace_layout" in env and bool(env["trace_layout"]):
        c2es = {k: [int(env["client_group"][k])] for k in range(K)}
        es2c = {}
        for k, g in c2es.items():
            es2c.setdefault(g[0], []).append(k)
        es2c = {g: es2c.get(g, []) for g in range(int(env["client_group"].max()) + 1)}
        idx, mc, _ = nd1_partition(train_labels, num_classes, K, c2es, es2c, fr, fc, seed=int(seed))
        indices, mains = idx, mc
    else:
        cd, cl = env["client_district"], env["client_local"]
        for d in sorted(set(int(x) for x in cd)):
            c2es, es2c = make_es_topology(50, 5, 50, seed=0)
            idx, mc, _ = nd1_partition(train_labels, num_classes, 50, c2es, es2c, fr, fc,
                                       seed=district_partition_seed(seed, d))
            for k in np.flatnonzero(cd == d):
                indices[int(k)] = list(idx[int(cl[k])])
                mains[int(k)] = set(mc[int(cl[k])])
    L = len(env["cell_xy"])
    cell_groups = {z: set() for z in range(L)}
    for k in range(K):
        cell_groups[int(env["home_cell"][k])] |= mains[k]
    all_used = set().union(*mains.values())
    return indices, mains, cell_groups, all_used


class RequestBuilder:
    """Builds per-client request index lists from a (masked) label array."""

    def __init__(self, pool_labels, num_classes, cell_groups, all_used):
        pool_labels = np.asarray(pool_labels)
        self.by_class = {c: np.flatnonzero(pool_labels == c) for c in range(num_classes)}
        self.cell_groups = cell_groups
        self.all_used = set(all_used)

    def scope(self, cells):
        s = set()
        for z in cells:
            if z >= 0:
                s |= self.cell_groups[int(z)]
        return s

    def classes(self, main, cells):
        main = set(main)
        remaining = self.all_used - main
        sc = self.scope(cells)
        return main, sc & remaining, remaining - sc

    def build(self, main, cells, rho):
        """Returns (indices, oop_set, oor_set, flags) with flags in
        {'', 'oor_empty', 'oop_empty', 'both_empty'}."""
        main, oop, oor = self.classes(main, cells)
        idx = [self.by_class[c] for c in sorted(main)]
        main_total = int(sum(len(a) for a in idx))
        share_oop = main_total * rho
        share_oor = main_total * OOR_FACTOR * rho
        flag = ""
        if not oop and not oor:
            flag = "both_empty"
        elif not oor:
            flag = "oor_empty"
            share_oop, share_oor = share_oop + share_oor, 0.0
        elif not oop:
            flag = "oop_empty"
            share_oop, share_oor = 0.0, share_oop + share_oor
        if rho > 0 and oop and share_oop > 0:
            per = max(1, int(share_oop / len(oop)))
            idx += [self.by_class[c][:per] for c in sorted(oop)]
        if rho > 0 and oor and share_oor > 0:
            per = max(1, int(share_oor / len(oor)))
            idx += [self.by_class[c][:per] for c in sorted(oor)]
        out = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
        return out, oop, oor, flag
