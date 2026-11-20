"""Round 18 (own-class detector): accuracy vs offloading-ratio figures from the curve CSVs (directive 6.3 item 3).

  python .../r18oc_figures.py curves tables/R18oc_stage2_curves.csv figures/R18oc_stage2 [model config ...]
  python .../r18oc_figures.py curves tables/R18oc_stage3_curves.csv figures/R18oc_stage3 <main cfg> cnn-scratch

One figure per (setting, model configuration), three panels (first-day, second-day, frozen-replay). Lines: the new method
(detector-offload with edge-corrected-by-detector), the comparator envelope B (seed-mean maximum over the 25 comparators at
each ratio, as in the cell tables), the v32 method (device-entropy + average-corrected-even) and the request-kind oracle
(edge-raw). Everything is read from the CSV; no run file is opened.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SCORES = ["device-entropy", "device-max-prob", "device-energy", "own-mahalanobis", "own-knn"]
ANSWERS = ["edge-raw", "edge-corrected-even", "average-corrected-even", "probability-average", "logit-sum"]
CONDS = ["first-day", "second-day", "frozen-replay"]
NEW = "edge-corrected-by-detector"
# reference categorical palette (dataviz skill, light mode), fixed order: slot 1 new, 2 envelope, 3 v32, oracle in grey
COL = {"new": "#2a78d6", "env": "#eb6834", "v32": "#1baf7a", "oracle": "#52514e"}


def load(path):
    rows = list(csv.reader(open(path)))
    head, body = rows[0], rows[1:]
    i0 = head.index("answer rule") + 1
    grid = np.array([float(h.split()[1]) for h in head[i0:]])
    D = defaultdict(dict)
    for r in body:
        D[(r[0], r[1], r[2])][(r[3], r[4])] = np.array([float(x) for x in r[i0:]])
    return grid, D


def main_variant(curves):
    v = [k[0] for k in curves if k[1] == NEW and "@" not in k[0]]
    with_avg = [k[0] for k in curves if k[1].startswith(NEW + " +")]
    return with_avg[0] if with_avg else (v[0] if len(v) == 1 else None)


def curves(csv_path, out_prefix, cfgs):
    grid, D = load(HERE / csv_path)
    out = HERE / out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted({(st, cfg) for st, cfg, _ in D if not cfgs or cfg in cfgs})
    made = []
    for st, cfg in groups:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, cond in zip(axes, CONDS):
            C = D.get((st, cfg, cond))
            ax.set_title(cond, fontsize=10)
            if not C:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center")
                continue
            mv = main_variant(C)
            env = np.max(np.stack([C[(s, a)] for s in SCORES for a in ANSWERS]), axis=0)
            lines = [("comparator envelope B", env, COL["env"], "-"),
                     ("v32 method", C[("device-entropy", "average-corrected-even")], COL["v32"], "-"),
                     ("request-kind oracle", C[("request-kind-oracle", "edge-raw")], COL["oracle"], "--")]
            if mv:
                lines.insert(0, (f"new method ({mv})", C[(mv, NEW)], COL["new"], "-"))
            for lab, y, c, ls in lines:
                ax.plot(grid, y, ls, color=c, lw=2, marker="o", ms=3.5, label=lab)
            ax.grid(True, color="#e4e3df", lw=0.6)
            ax.set_xlabel("offloading ratio")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("accuracy (%)")
        h, l = axes[-1].get_legend_handles_labels() if axes[-1].lines else axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{st}, {cfg}", fontsize=11)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        f = Path(f"{out}_{st.replace(' ', '_').replace('(', '').replace(')', '')}_{cfg}.png")
        fig.savefig(f, dpi=150)
        plt.close(fig)
        made.append(f.name)
    print("\n".join(made))
    return made


if __name__ == "__main__":
    if sys.argv[1] == "curves":
        curves(sys.argv[2], sys.argv[3], sys.argv[4:])
