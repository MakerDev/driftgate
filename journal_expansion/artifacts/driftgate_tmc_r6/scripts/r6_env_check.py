"""Round 6 §3.3 and §3.9 environment check (no training).

For S1, S1fast, S2, S3_K200, S3_K500 (all env seeds) computes, averaged over seeds:
  cell occupancy by clock time, cell mean rho, the realised non-Main share of the
  request pool (built exactly as the runner builds probe requests), share of clients
  at home, membership changes per round, share of clients in two cells, the request
  count n = min(N, 64) distribution, and the class groups with OOP/OOR emptiness.
Outputs (artifacts/driftgate_tmc_r6/env_check/):
  env_check_<scenario>.png / .csv, env_check_summary.csv, class_groups.csv,
  s1_partition_clients.csv (§3.3: members, Main classes and sample counts per client)
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parent.parent
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_requests import RequestBuilder, build_partition  # noqa: E402
from src.disjoint_pools import make_pool_masks  # noqa: E402

OUT = HERE / "env_check"
ENV = JR / "runs" / "phaseT6_env"
SCEN = {"S1": [0, 1, 2, 3, 4], "S1fast": [0, 1, 2], "S2": [0, 1, 2], "S3_K200": [0, 1, 2], "S3_K500": [0, 1, 2]}
LABEL = {"S1": "commute mobility", "S1fast": "commute mobility (vehicle speed)", "S2": "GeoLife trace",
         "S3_K200": "commute mobility, K=200", "S3_K500": "commute mobility, K=500"}


def labels():
    from torchvision import datasets
    tr = datasets.CIFAR10(str(JR.parent / "data_cache"), train=True, download=False)
    te = datasets.CIFAR10(str(JR.parent / "data_cache"), train=False, download=False)
    return np.array(tr.targets), np.array(te.targets)


def cell_series(env):
    """returns list of (name, cell index array) series to plot/aggregate."""
    L = len(env["cell_xy"])
    if L <= 5:
        names = r6_env.CELL_NAMES if not ("trace_layout" in env) else [f"cell {z}" for z in range(L)]
        return [(names[z] if "trace_layout" not in env else f"cell {z}", np.array([z])) for z in range(L)]
    hub = np.flatnonzero(env["cell_is_hub"])
    res = np.flatnonzero(~env["cell_is_hub"])
    return [("hub cells (mean)", hub), ("residential cells (mean)", res)]


def analyse(scen, seeds, train_labels, test_labels):
    per_seed = []
    groups_rows = []
    for s in seeds:
        env, meta = r6_env.load_env(ENV / f"{scen}_seed{s}.npz")
        K, T = env["member"].shape[:2]
        L = len(env["cell_xy"])
        idx, mains, cgroups, all_used = build_partition(train_labels, env, s)
        probe_labels, _, _ = make_pool_masks(test_labels, 10, 0.2, seed=s)
        rb = RequestBuilder(probe_labels, 10, cgroups, all_used)
        m = env["member"]
        occ = np.zeros((T, L))
        rho_c = np.full((T, L), np.nan)
        nm_c = np.full((T, L), np.nan)
        nm_client = np.zeros((K, T))
        flags = np.empty((K, T), dtype=object)
        for t in range(T):
            for k in range(K):
                cells = [int(z) for z in m[k, t] if z >= 0]
                pool, _, _, flag = rb.build(mains[k], cells, float(env["rho"][k, t]))
                main_n = int(np.isin(probe_labels[pool], sorted(mains[k])).sum())
                nm_client[k, t] = 1 - main_n / len(pool)
                flags[k, t] = flag
            for z in range(L):
                mem = (m[:, t, :] == z).any(axis=1)
                occ[t, z] = mem.sum()
                if mem.any():
                    rho_c[t, z] = env["rho"][mem, t].mean()
                    nm_c[t, z] = nm_client[mem, t].mean()
        changes = np.array([0] + [int((np.sort(m[:, t], axis=1) != np.sort(m[:, t - 1], axis=1)).any(axis=1).sum())
                                  for t in range(1, T)])
        n = np.minimum(env["n_req"], 64)
        home = env["at_home"]
        oor_empty_home = np.mean([flags[k, t] in ("oor_empty", "both_empty")
                                  for k in range(K) for t in range(T) if home[k, t]])
        per_seed.append(dict(occ=occ, rho=rho_c, nm=nm_c, home=home.mean(axis=0),
                             two=(m[:, :, 1] >= 0).mean(axis=0), changes=changes, n=n,
                             flags=flags, oor_empty_home=oor_empty_home, env=env, meta=meta))
        for z in range(L):
            groups_rows.append([scen, s, z, int(env["cell_district"][z]), int(env["cell_local"][z]),
                                " ".join(map(str, sorted(cgroups[z]))), len(cgroups[z]),
                                int((env["home_cell"] == z).sum())])
    return per_seed, groups_rows


def clock_axis(ax):
    ticks = [r6_env.round_start_min(r) for r in (1, 31, 61, 91, 121)] + [20 * 60]
    ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
    ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
    ax.set_xlim(1, 151)


def plot(scen, per_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env = per_seed[0]["env"]
    series = cell_series(env)
    T = per_seed[0]["occ"].shape[0]
    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]

    def agg(key, cells):
        return np.nanmean([np.nanmean(p[key][:, cells], axis=1) for p in per_seed], axis=0)
    for i, (name, cells) in enumerate(series):
        axes[0, 0].plot(x, agg("occ", cells), color=colors[i % 6], label=name, lw=1.5)
        axes[0, 1].plot(x, agg("rho", cells), color=colors[i % 6], lw=1.5)
        axes[0, 2].plot(x, agg("nm", cells), color=colors[i % 6], lw=1.5)
    axes[0, 0].set_title("(a) clients in cell")
    axes[0, 1].set_title("(b) cell mean rho")
    axes[0, 2].set_title("(c) non-Main share of requests")
    axes[0, 2].axhline(1.3 * 0.1 / 1.13, color="gray", ls=":", lw=1)
    axes[0, 2].axhline(1.3 * 0.8 / 2.04, color="gray", ls=":", lw=1)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].plot(x, np.mean([p["home"] for p in per_seed], axis=0), color="#333333", lw=1.5, label="at home")
    axes[1, 0].plot(x, np.mean([p["two"] for p in per_seed], axis=0), color="#999999", lw=1.5, label="in two cells")
    axes[1, 0].set_title("(d) share of clients")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].bar(x, np.mean([p["changes"] for p in per_seed], axis=0), color="#555555", width=1.0)
    axes[1, 1].set_title("(e) clients whose cells changed")
    allN = np.concatenate([p["n"].ravel() for p in per_seed])
    axes[1, 2].hist(allN, bins=np.arange(0, 66, 2), color="#555555")
    axes[1, 2].set_title(f"(f) probe requests n (n=0: {np.mean(allN == 0):.4f})")
    for ax in axes.ravel()[:5]:
        clock_axis(ax)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Environment check: {LABEL[scen]} ({len(per_seed)} seeds)", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"env_check_{scen}.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trl, tel = labels()
    summary, groups = [], []
    for scen, seeds in SCEN.items():
        if not (ENV / f"{scen}_seed{seeds[0]}.npz").exists():
            print(f"skip {scen}: no environment file")
            continue
        per_seed, grows = analyse(scen, seeds, trl, tel)
        groups += grows
        plot(scen, per_seed)
        env = per_seed[0]["env"]
        series = cell_series(env)
        T = per_seed[0]["occ"].shape[0]
        with open(OUT / f"env_check_{scen}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "clock", "series", "clients_in_cell", "cell_mean_rho", "non_main_share",
                        "share_at_home", "share_two_cells", "membership_changes"])
            for t in range(T):
                for name, cells in series:
                    w.writerow([t + 1, r6_env.hhmm(r6_env.round_start_min(t + 1)), name,
                                f"{np.mean([np.nanmean(p['occ'][t, cells]) for p in per_seed]):.3f}",
                                f"{np.nanmean([np.nanmean(p['rho'][t, cells]) for p in per_seed]):.4f}",
                                f"{np.nanmean([np.nanmean(p['nm'][t, cells]) for p in per_seed]):.4f}",
                                f"{np.mean([p['home'][t] for p in per_seed]):.4f}",
                                f"{np.mean([p['two'][t] for p in per_seed]):.4f}",
                                f"{np.mean([p['changes'][t] for p in per_seed]):.2f}"])
        allN = np.concatenate([p["n"].ravel() for p in per_seed])
        fl = np.concatenate([p["flags"].ravel() for p in per_seed])
        summary.append(dict(scenario=scen, seeds=" ".join(map(str, SCEN[scen])), K=int(env["member"].shape[0]),
                            L=len(env["cell_xy"]),
                            share_at_home=f"{np.mean([p['home'].mean() for p in per_seed]):.4f}",
                            share_two_cells=f"{np.mean([p['two'].mean() for p in per_seed]):.4f}",
                            membership_changes_per_round=f"{np.mean([p['changes'][1:].mean() for p in per_seed]):.3f}",
                            n_mean=f"{allN.mean():.2f}", n_p5=int(np.percentile(allN, 5)),
                            n_p50=int(np.percentile(allN, 50)), n_zero_share=f"{np.mean(allN == 0):.5f}",
                            n_below64_share=f"{np.mean(allN < 64):.4f}",
                            oor_empty_share=f"{np.mean(fl == 'oor_empty'):.5f}",
                            oop_empty_share=f"{np.mean(fl == 'oop_empty'):.5f}",
                            both_empty_share=f"{np.mean(fl == 'both_empty'):.5f}",
                            oor_empty_share_at_home=f"{np.mean([p['oor_empty_home'] for p in per_seed]):.5f}"))
        print(scen, summary[-1])
    with open(OUT / "env_check_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(OUT / "class_groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "cell", "district", "cell_local", "class_group", "n_classes", "residents"])
        w.writerows(groups)
    # §3.3: the static-topology partition behind S1 (seeds 0-4)
    from data.partition import make_es_topology, nd1_partition
    with open(OUT / "s1_partition_clients.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "client", "static_cells", "basic_member_of_cluster", "home_cell", "main_classes", "n_train"])
        for s in SCEN["S1"]:
            c2es, es2c = make_es_topology(50, 5, 50, seed=0)
            idx, mc, _ = nd1_partition(trl, 10, 50, c2es, es2c, (0.4, 0.7), 0.2, seed=s)
            for k in range(50):
                w.writerow([s, k, " ".join(map(str, c2es[k])), c2es[k][0], r6_env.CELL_NAMES[c2es[k][0]],
                            " ".join(map(str, sorted(mc[k]))), len(idx[k])])


if __name__ == "__main__":
    main()
