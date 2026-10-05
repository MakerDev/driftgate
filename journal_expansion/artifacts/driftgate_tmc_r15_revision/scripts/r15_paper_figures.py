"""Round 15: figures for the revised manuscript (IEEE widths 3.5 / 7.16 in, PDF and PNG), from the cached per-run
summaries of r15_analysis.py (no new computation) and the Round 7 table for the oracle figure.

Wording follows the manuscript: "devices" (not clients); rule names as displayed in the paper. The offloading curves
use the Round 15 partial-offloading DriftGate (both mean entropies over offloaded requests, manuscript Section 5.3).
Outputs: figures/paper/*.pdf|png and figures/paper/captions.md.
"""
import csv
import importlib.util
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
os.environ.setdefault("R13B_OUT", str(HERE))
_spec = importlib.util.spec_from_file_location("r15", HERE / "scripts" / "r15_analysis.py")
m15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m15)
OUT = HERE / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
COL = dict(m15.m14.COL)
MK = {"B0": "s", "B1": "v", "B2": "^", "B3": "D", "R-PoE": "P", "LEW": "X", "DriftGate": "o"}
SHOW = m15.SHOW


def load(settings):
    lr, lr_corr = m15.m14.read_lr(), m15.fit_lr_corr()
    R = {}
    for st in settings:
        rd, pat, seeds = m15.SETTINGS[st]
        tasks = [(str(rd), pat.format(s), lr, lr_corr) for s in seeds if (rd / f"{pat.format(s)}_evalprobs.npz").exists()]
        R[st] = m15.run_many(tasks)
    return R


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "axes.titlesize": 8, "axes.edgecolor": "#b5b5b0", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.3,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    save = lambda fig, n: (fig.savefig(OUT / f"{n}.pdf"), fig.savefig(OUT / f"{n}.png", dpi=300), plt.close(fig))
    R = load(["S1", "S2", "S1 day 1 (retrained)", "S1 frozen replay", "S2 day 1 (retrained)", "S2 frozen replay"])
    v = lambda st, rl, key="full": np.array([a["res"][rl][key] for a in R[st]]) * 100
    # motivation
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7), layout="constrained", gridspec_kw=dict(width_ratios=[2, 3]))
    for ax, cats, labs, title in ((axs[0], ["full_home", "full_away"], ["at home", "away"], "(a) devices at home / away"),
                                  (axs[1], ["full_Main", "full_OOP", "full_OOR"], ["Main", "OOP", "OOR"], "(b) request kind")):
        x = np.arange(len(cats))
        for i, rl in enumerate(("B1", "B2")):
            ax.bar(x + (i - 0.5) * 0.38, [v("S1", rl, c).mean() for c in cats], 0.36, yerr=[v("S1", rl, c).std(ddof=1) for c in cats],
                   color=COL[rl], label=SHOW[rl], error_kw=dict(lw=0.6, capsize=1.5), zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_title(title, fontsize=7)
    axs[0].set_ylabel("accuracy (%)")
    axs[1].legend(loc="upper right")
    save(fig, "motivation_exits")
    # device-level oracle (Round 7)
    t2 = [r for r in csv.DictReader(open(m15.ART / "driftgate_tmc_r7_gate" / "tables" / "T2_paired_differences.csv"))
          if r["comparison"].startswith("O (oracle)")]
    fig, ax = plt.subplots(figsize=(2.3, 1.7), layout="constrained")
    labs = {"integrated": "all\ndevices", "at home": "devices\nat home", "away": "devices\naway"}
    for i, r in enumerate(t2):
        ps_ = np.array([float(x) for x in r["per_seed_pp(5,6,7)"].split()])
        ax.bar(i, ps_.mean(), 0.6, yerr=ps_.std(ddof=1), color="#7a7a7a", error_kw=dict(lw=0.6, capsize=2), zorder=2)
        ax.plot(i + np.array([-0.2, 0, 0.2]), ps_, ls="none", marker="o", ms=2.5, color="#222222", zorder=3)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(t2)))
    ax.set_xticklabels([labs[r["accuracy"]] for r in t2], fontsize=7)
    ax.set_ylabel("oracle - fixed ratio (pp)")
    save(fig, "device_oracle")
    # home / away
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), layout="constrained")
    for ax, st in zip(axs, ("S1", "S2")):
        for rl in ("B0", "B1", "B2", "B3", "R-PoE", "DriftGate"):
            ax.plot(v(st, rl, "full_home").mean(), v(st, rl, "full_away").mean(), ls="none", marker=MK[rl], color=COL[rl],
                    ms=8 if rl == "DriftGate" else 6, label=SHOW[rl], zorder=3 if rl == "DriftGate" else 2)
        ax.set_title(st, fontsize=7)
        ax.set_xlabel("accuracy of devices at home (%)")
        ax.set_ylabel("accuracy of devices away (%)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "home_away")
    # offloading curves
    grid = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), layout="constrained")
    for ax, st in zip(axs, ("S1", "S2")):
        for cn, c_, ls_, lab in (("SplitGP", COL["B0"], "-", "Confidence-based offloading"),
                                 ("Geometric ensemble", COL["R-ZTW"], "-.", "Geometric ensemble with early exit"),
                                 ("DriftGate-P", COL["DriftGate"], "-", "DriftGate")):
            vals = [m15.m13.interp(a["curves"]["full"][cn]["srv"], a["curves"]["full"][cn]["acc"], grid) for a in R[st]]
            ax.plot(grid, np.nanmean(vals, axis=0) * 100, color=c_, ls=ls_, lw=1.8 if cn == "DriftGate-P" else 1.2, label=lab)
        ax.axvline(float(np.mean([a["srv"]["full"]["B0"] for a in R[st]])), color="#888888", lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(st, fontsize=7)
        ax.set_xlabel("offloading ratio")
        ax.set_ylabel("accuracy (%)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "offload_curves")
    # time of day (S1)
    a0 = R["S1"][0]
    hrs = a0["start_min"] / 60.0
    fig, ax = plt.subplots(figsize=(7.16, 2.0), layout="constrained")
    ax2 = ax.twinx()
    ax2.fill_between(hrs, np.mean([a["home_share"] for a in R["S1"]], axis=0) * 100, color="#d9d6cc", alpha=0.45, lw=0, step="mid")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("devices at home (%)", color="#8a877c")
    ax2.tick_params(axis="y", colors="#8a877c")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for rl in ("B0", "B1", "B2", "DriftGate"):
        ax.plot(hrs, np.mean([a["res"][rl]["per_round"] for a in R["S1"]], axis=0) * 100, color=COL[rl],
                lw=1.8 if rl == "DriftGate" else 1.1, ls="--" if rl == "B2" else "-", marker=MK[rl], ms=2.2, label=SHOW[rl])
    for _, lo, _ in m15.m14.BINS[1:]:
        ax.axvline(lo / 60.0, color="#bbbbbb", lw=0.5, ls="--")
    ax.set_xlim(5, 20)
    ax.set_xticks(range(5, 21))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower left", ncol=4)
    save(fig, "time_of_day")
    # replay vs first day (S1, S2)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), layout="constrained")
    for ax, (d1, rp) in zip(axs, (("S1 day 1 (retrained)", "S1 frozen replay"), ("S2 day 1 (retrained)", "S2 frozen replay"))):
        hrs = R[d1][0]["start_min"] / 60
        for rl in ("B0", "B1", "B2", "DriftGate"):
            for st, ls in ((d1, ":"), (rp, "-")):
                ax.plot(hrs, np.mean([a["res"][rl]["per_round"] for a in R[st]], axis=0) * 100, color=COL[rl], ls=ls,
                        lw=1.8 if rl == "DriftGate" else 1.1, label=f"{SHOW[rl]}, {'first day' if st == d1 else 'replay'}")
        ax.set_title(d1.split()[0], fontsize=8)
        ax.set_xlim(5, 20)
        ax.set_xlabel("time of day (h)")
        ax.set_ylabel("accuracy (%)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "replay_time_of_day")
    print("paper figures written to", OUT, flush=True)


if __name__ == "__main__":
    main()
