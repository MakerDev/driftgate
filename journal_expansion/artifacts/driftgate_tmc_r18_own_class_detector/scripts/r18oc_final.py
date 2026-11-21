"""Round 18 (own-class detector), stage 4: tables and the summary figure for report_final.md, merged from the CSVs in
tables/ (no cache is read here; every number comes from the stage-2 and attempt tables).

  python .../r18oc_final.py cells    # tables/R18oc_final_cells.csv: stage 2 and attempts 2, 3, one row per cell
  python .../r18oc_final.py figure   # figures/R18oc_final_judged_cells.{pdf,png}: G(0.5) of the S1/S2 second-day and
                                     # frozen-replay cells of the pretrained configurations, stage 2 and attempts 2, 3
  python .../r18oc_final.py md       # (stdout) markdown blocks used in report_final.md

Judged cell (directive 7.3, decisions D17 and D19): pretrained configuration, S1 or S2, second-day or frozen-replay, and room
(the request-kind oracle passes, 7.2) in the run that is judged; for attempt 2 the room comes from the attempt's own runs,
for attempt 3 (same runs as stage 2, only the answer rule changes) it is the stage-2 room.
"new below the strongest comparator" = G(0.5) < 0, i.e. the new method at offloading ratio 0.5 is below B(0.5), the best of
the 25 comparators at that ratio.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TABLES = HERE / "tables"
FIGS = HERE / "figures"
TAGS = {"cnn-scratch": "cnn", "pretrained-finetuned-block": "ft", "pretrained-frozen-block": "fz"}
PRE = ("pretrained-finetuned-block", "pretrained-frozen-block")
OUT = "R18oc_final_cells.csv"
HEAD = ["source", "setting", "model config", "condition", "judged cell", "room", "S", "G(0.5) pp mean (SD)", "G(0.5) pp mean",
        "new accuracy at 0.5", "B(0.5)", "B(0.5) rule", "judgement", "new below B(0.5)"]


def read(name):
    p = TABLES / name
    if not p.exists():
        return None
    return list(csv.DictReader(open(p)))


def num(x):
    return float(x.split(" (")[0])


def judged(setting, cfg, cond, room):
    return cfg in PRE and setting in ("S1", "S2") and cond in ("second-day", "frozen-replay") and room == "yes"


def cells():
    st2 = read("R18oc_stage2_cells.csv")
    key = lambda d: (d["setting"], d["model config"], d["condition"])
    S2 = {key(d): d for d in st2}
    rows = []
    for d in st2:
        rows.append(["stage 2", *key(d), "yes" if judged(*key(d), d["room"]) else "no", d["room"], d["S"], d["G(0.5) pp mean (SD)"],
                     f"{num(d['G(0.5) pp mean (SD)']):.2f}", d["new accuracy at 0.5"], d["B(0.5)"], d["B(0.5) rule"], d["judgement"],
                     "yes" if num(d["G(0.5) pp mean (SD)"]) < 0 else "no"])
    missing = []
    for att in (2, 3):
        for cfg in PRE:
            name = f"R18oc_attempt{att}_{TAGS[cfg]}_cells.csv"
            A = read(name)
            if A is None:
                missing.append(name)
                continue
            for d in A:
                k = key(d)
                if att == 2:
                    new, B, rule, jd = d["new accuracy at 0.5"], d["B(0.5)"], d["B(0.5) rule"], d["judgement"]
                else:   # attempt 3: same runs as stage 2, so the comparators and B(0.5) are the stage-2 ones
                    new, B, rule, jd = d["new accuracy at 0.5 (attempt 3)"], S2[k]["B(0.5)"], S2[k]["B(0.5) rule"], d["judgement (attempt 3)"]
                room = d["room"]
                rows.append([f"attempt {att}", *k, "yes" if judged(*k, room) else "no", room, d["S"], d["G(0.5) pp mean (SD)"],
                             f"{num(d['G(0.5) pp mean (SD)']):.2f}", new, B, rule, jd, "yes" if num(d["G(0.5) pp mean (SD)"]) < 0 else "no"])
    with open(TABLES / OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        w.writerows(rows)
    print(f"{OUT}: {len(rows)} rows" + (f"; missing {', '.join(missing)}" if missing else ""))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    R = read(OUT)
    src = [("stage 2", "#2a78d6", "o"), ("attempt 2", "#eb6834", "s"), ("attempt 3", "#1baf7a", "D")]
    order = [(cfg, s, c) for cfg in PRE for s in ("S1", "S2") for c in ("second-day", "frozen-replay")]
    fig, ax = plt.subplots(figsize=(7.16, 3.6))
    for x in (1.0, 2.0):
        ax.axvline(x, color="#8a8a85", lw=0.8, ls="--", zorder=0)
    ax.axvline(0.0, color="#3d3d3a", lw=0.9, zorder=0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.text(1.0, len(order) - 0.5, " partial pass\n G(0.5) ≥ 1.0", fontsize=6, color="#5f5f5a", va="top")
    ax.text(2.0, len(order) - 0.5, " pass\n G(0.5) ≥ 2.0", fontsize=6, color="#5f5f5a", va="top")
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        for j, (s, col, mk) in enumerate(src):
            d = next((r for r in R if r["source"] == s and (r["model config"], r["setting"], r["condition"]) == k), None)
            if d is None:
                continue
            g = num(d["G(0.5) pp mean (SD)"])
            sd = float(d["G(0.5) pp mean (SD)"].split("(")[1].rstrip(")")) if "(" in d["G(0.5) pp mean (SD)"] else 0.0
            yy = y + (1 - j) * 0.22
            ax.plot([g - sd, g + sd], [yy, yy], color=col, lw=1.0, alpha=0.6, zorder=2)
            filled = d["judged cell"] == "yes"
            ax.plot(g, yy, mk, ms=6, color=col, mfc=col if filled else "white", mew=1.4, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{TAGS[c]}  {s}  {cond}" for c, s, cond in reversed(order)], fontsize=7)
    ax.set_xlim(right=2.6)
    ax.set_xlabel("G(0.5): new method minus the best of 25 comparators at offloading ratio 0.5 (pp; mean ± SD over seeds 0-2)",
                  fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    hd = [Line2D([], [], marker=mk, color=col, ls="", ms=6, label=s) for s, col, mk in src]
    hd += [Line2D([], [], marker="o", color="#5f5f5a", mfc="#5f5f5a", ls="", ms=6, label="judged cell (room in the judged run)"),
           Line2D([], [], marker="o", color="#5f5f5a", mfc="white", ls="", ms=6, label="no room (not judged)")]
    fig.legend(handles=hd, fontsize=6.5, loc="lower center", ncol=5, frameon=False)
    ax.set_title("ft = pretrained-finetuned-block, fz = pretrained-frozen-block; evaluation seeds 0-2", fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"R18oc_final_judged_cells.{ext}", dpi=300)
    print("figures/R18oc_final_judged_cells.{pdf,png}")


def md(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |" for r in body]
    return "\n".join(out)


def markdown():
    R = read(OUT)
    st2 = [r for r in R if r["source"] == "stage 2"]
    print("### stage 2: S / G(0.5) pp (SD) / judgement, evaluation seeds 0-2\n")
    cfgs = list(dict.fromkeys(r["model config"] for r in st2))
    keys = list(dict.fromkeys((r["setting"], r["condition"]) for r in st2))
    body = []
    for s, c in keys:
        line = [s, c]
        for cfg in cfgs:
            d = next((r for r in st2 if (r["setting"], r["condition"], r["model config"]) == (s, c, cfg)), None)
            line.append("not run" if d is None else f"{d['S']} / {d['G(0.5) pp mean (SD)']} / {d['judgement']}"
                        + ("" if d["room"] == "yes" else " (no room)"))
        body.append(line)
    print(md(["setting", "condition"] + [TAGS[c] for c in cfgs], body))
    print("\n### judged cells: stage 2 and attempts 2, 3\n")
    J = [r for r in R if r["model config"] in PRE and r["setting"] in ("S1", "S2") and r["condition"] in ("second-day", "frozen-replay")]
    print(md(["source", "model config", "setting", "condition", "room", "judged cell", "S", "G(0.5) pp mean (SD)", "new at 0.5", "B(0.5)",
              "B(0.5) rule", "judgement"],
             [[r["source"], TAGS[r["model config"]], r["setting"], r["condition"], r["room"], r["judged cell"], r["S"],
               r["G(0.5) pp mean (SD)"], r["new accuracy at 0.5"], r["B(0.5)"], r["B(0.5) rule"], r["judgement"]] for r in J]))
    below = [r for r in R if r["new below B(0.5)"] == "yes"]
    print(f"\n### new method below the strongest comparator at ratio 0.5: {len(below)} of {len(R)} cells\n")
    for s in dict.fromkeys(r["source"] for r in R):
        n, b = sum(r["source"] == s for r in R), sum(r["source"] == s for r in below)
        print(f"- {s}: {b} of {n}")
    print("\nrules giving B(0.5) in those cells:\n")
    rc = {}
    for r in below:
        rc[r["B(0.5) rule"]] = rc.get(r["B(0.5) rule"], 0) + 1
    print(md(["B(0.5) rule", "cells"], sorted(rc.items(), key=lambda x: -x[1])))
    print("\ncells with G(0.5) >= 0:\n")
    print(md(["source", "setting", "model config", "condition", "room", "G(0.5) pp mean (SD)"],
             [[r["source"], r["setting"], r["model config"], r["condition"], r["room"], r["G(0.5) pp mean (SD)"]] for r in R
              if r["new below B(0.5)"] == "no"]))
    print("\n### stage 2 first-day cells with G(0.5) < -0.5 pp (directive 7.3, extra condition)\n")
    print(md(["setting", "model config", "G(0.5) pp mean (SD)"],
             [[r["setting"], r["model config"], r["G(0.5) pp mean (SD)"]] for r in st2
              if r["condition"] == "first-day" and float(r["G(0.5) pp mean"]) < -0.5]))
    print("\n### attempt gates (S1 development seeds 8-10, frozen-replay, ratio 0.5; gate +0.5 pp)\n")
    body = []
    for att in (1, 2, 3):
        for cfg in PRE:
            G = read(f"R18oc_attempt{att}_{TAGS[cfg]}_dev.csv")
            if G is None:
                continue
            m = G[-1]
            per = ", ".join(r["difference pp"] for r in G[:-1])
            au = ""
            X = read(f"R18oc_attempt{att}_{TAGS[cfg]}_dev_context.csv")   # seed means of the unrounded AUROC
            if X is not None:
                a = {r["run"]: r["AUROC all (chosen variant)"] for r in X if r["seed"] == "mean (SD)"}
                au = f"{a['without the change']} → {a[f'attempt {att}']}"
            body.append([att, TAGS[cfg], m["without the change: accuracy at 0.5 (%)"], m["attempt: accuracy at 0.5 (%)"],
                         m["difference pp"], per, au, "yes" if num(m["difference pp"]) >= 0.5 else "no"])
    print(md(["attempt", "config", "without the change (%)", "attempt (%)", "difference pp mean (SD)", "per seed (8, 9, 10)",
              "AUROC mean (without → attempt)", "passes the gate"], body))
    print("\n### cost of attempt 2 (deeper-split), one 64 x 64 image\n")
    C = read("R18oc_attempt2_cost.csv")
    print(md(list(C[0].keys()), [list(r.values()) for r in C]))


if __name__ == "__main__":
    {"cells": cells, "figure": figure, "md": markdown}[sys.argv[1]]()
