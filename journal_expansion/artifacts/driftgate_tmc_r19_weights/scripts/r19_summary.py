"""Round 19: summary table for the report and the paired-difference figure (written after the stage-0 tables).

Reads tables/R19_weight_curves.csv, R19_weight_curves_per_seed.csv, R19_table1_deployable.csv, R19_table2_oracles.csv,
R19_multiday_curves.csv and R19_state.json. Writes tables/R19_summary.csv and figures/R19_fig1b_paired_difference.{pdf,png}
(fixed w minus DriftGate per seed: mean and range over seeds; full window).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
T, F = HERE / "tables", HERE / "figures"
W = [round(0.05 * i, 2) for i in range(21)]


def first(x):
    return x.split()[0] if x else ""


def main():
    st = json.loads((T / "R19_state.json").read_text())
    w_dev = st["development_w"]
    cur = {(r["setting"], r["beta"], r["window"]): r for r in csv.DictReader(open(T / "R19_weight_curves.csv"))}
    t1 = {(r["setting"], r["beta"], r["window"]): r for r in csv.DictReader(open(T / "R19_table1_deployable.csv"))}
    t2 = {(r["setting"], r["beta"], r["window"]): r for r in csv.DictReader(open(T / "R19_table2_oracles.csv"))}
    rows = []
    for (s, b, wn), r in cur.items():
        a = t1[(s, b, wn)]
        o = t2[(s, b, wn)]
        dg = float(r["DriftGate accuracy"])
        rows.append([s, r["model"], b, wn, f"{dg:.2f}", r["DriftGate w mean"], r["best grid w"], r["w within 0.1 pp of best (min-max)"],
                     f"{dg - float(r['best grid accuracy']):+.3f}", first(a["DriftGate minus development w (S1 dev seeds) pp"]),
                     first(a["DriftGate minus initial fixed (label-free) pp"]), first(a["DriftGate minus corrected edge only (w = 0) pp"]), first(o["device constant (mean DriftGate w) minus DriftGate pp"]),
                     first(o["device oracle minus common oracle pp"]), first(o["time-bin oracle minus device oracle pp"])])
    with open(T / "R19_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "model", "beta", "window", "DriftGate", "DriftGate w mean (SD)", "best fixed w (seed mean)", "w within 0.1 pp",
                    "DriftGate minus best fixed pp", f"DriftGate minus development w {w_dev:.2f} pp", "DriftGate minus initial fixed pp",
                    "DriftGate minus corrected edge only pp", "device constant minus DriftGate pp (post hoc)",
                    "device oracle minus common oracle pp", "time-bin oracle minus device oracle pp"])
        w.writerows(rows)
    print(f"R19_summary.csv {len(rows)} rows")
    # figure: paired difference fixed w - DriftGate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = defaultdict(list)
    for r in csv.DictReader(open(T / "R19_weight_curves_per_seed.csv")):
        if r["window"] == "full":
            dg = float(r["DriftGate"])
            per[(r["setting"], r["beta"])].append(([float(r[f"w={w:.2f}"]) - dg for w in W], float(r["initial fixed"]) - dg))
    md = {(r["scenario"], r["beta"]): r for r in csv.DictReader(open(T / "R19_multiday_curves.csv"))}
    sets = [["S1", "S2", "ResNet-18", "K=500", "CIFAR-100"], ["ResNet-20 shallow", "ResNet-20 middle", "VGG-11 shallow", "VGG-11 middle", "S1 replay"]]
    fig, axs = plt.subplots(3, 5, figsize=(15, 8.4), sharex=True)
    for r_, row in enumerate(sets):
        for c, s in enumerate(row):
            ax = axs[r_, c]
            for b, col in (("1.0", "#2a6fbb"), ("0.5", "#16a085")):
                d = np.array([x[0] for x in per[(s, b)]])
                ax.plot(W, d.mean(0), color=col, lw=1.5, label=f"beta {float(b):g}")
                ax.fill_between(W, d.min(0), d.max(0), color=col, alpha=0.15, lw=0)
                ax.axhline(np.mean([x[1] for x in per[(s, b)]]), color=col, lw=0.9, ls="--")
            ax.axhline(0, color="#c0392b", lw=1.0)
            ax.axvline(w_dev, color="#555555", lw=0.7, ls=":")
            ax.set_title(s, fontsize=9)
            ax.tick_params(labelsize=7)
    for c, (sc, lab) in enumerate((("S1", "S1 three-day replay"), ("S2", "S2 three-day replay"), (None, "S2 replay"))):
        ax = axs[2, c]
        if sc is None:
            for b, col in (("1.0", "#2a6fbb"), ("0.5", "#16a085")):
                d = np.array([x[0] for x in per[("S2 replay", b)]])
                ax.plot(W, d.mean(0), color=col, lw=1.5)
                ax.fill_between(W, d.min(0), d.max(0), color=col, alpha=0.15, lw=0)
                ax.axhline(np.mean([x[1] for x in per[("S2 replay", b)]]), color=col, lw=0.9, ls="--")
        else:
            for b, col in (("1.0", "#2a6fbb"), ("0.5", "#16a085")):
                r = md[(sc, b)]
                ax.plot(W, [float(r[f"w={w:.2f}"]) - float(r["DriftGate"]) for w in W], color=col, lw=1.5)
        ax.axhline(0, color="#c0392b", lw=1.0)
        ax.axvline(w_dev, color="#555555", lw=0.7, ls=":")
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("device weight w", fontsize=8)
    for c in (3, 4):
        axs[2, c].axis("off")
    for r_ in range(3):
        axs[r_, 0].set_ylabel("fixed w minus DriftGate (pp)", fontsize=8)
    axs[0, 0].legend(fontsize=7, loc="lower center")
    for ax in axs.flat:
        if ax.axison:
            lo, hi = ax.get_ylim()
            ax.set_ylim(max(lo, -2.0), hi)
    fig.tight_layout()
    fig.text(0.62, 0.22, "solid: fixed w minus DriftGate (seed mean; band: range over seeds)\n"
             "dashed: label-free weight fixed after 128 offloaded requests minus DriftGate\n"
             f"red line: DriftGate; dotted: development w {w_dev:.2f}\n"
             "three-day replay panels: seed means only\ny axis cut at -2 pp", fontsize=9, va="top")
    for ext in ("pdf", "png"):
        fig.savefig(F / f"R19_fig1b_paired_difference.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
