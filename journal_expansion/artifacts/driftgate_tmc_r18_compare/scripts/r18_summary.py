"""Round 18: summary of the same-budget table (tables/R18_online_same_budget_per_seed.csv) by comparison group.
uncorrected: Raw edge only, Probability average, Logit sum, No correction + adaptive w;
corrected: Correction + fixed w 0.5, Correction + development w 0.2, Corrected edge only, Correction + product,
Correction + learned weight. Per row the best rule of a group is chosen by seed mean; differences are paired by seed.
Output: tables/R18_online_same_budget_summary.csv"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parent.parent / "tables"
UNC = ["Raw edge only", "Probability average", "Logit sum", "No correction + adaptive w"]
COR = ["Correction + fixed w 0.5", "Correction + development w 0.2", "Corrected edge only", "Correction + product", "Correction + learned weight"]
V = defaultdict(dict)
for r in csv.DictReader(open(T / "R18_online_same_budget_per_seed.csv")):
    V[(r["setting"], r["beta"])].setdefault(r["rule"], []).append(float(r["accuracy pct"]))
meta = {(r["setting"], r["beta"]): r for r in csv.DictReader(open(T / "R18_online_same_budget.csv")) if r["rule"] == "DriftGate-P"}
rows = []
for (st, b), d in V.items():
    dg = np.array(d["DriftGate-P"])
    out = [st, b, f"{dg.mean():.2f}"]
    for grp in (UNC, COR):
        best = max(grp, key=lambda r: np.mean(d[r]))
        x = dg - np.array(d[best])
        out += [best, f"{np.mean(d[best]):.2f}", f"{x.mean():+.3f} ({x.std(ddof=1):.3f})"]
    out += [meta[(st, b)]["realized offloading"], meta[(st, b)]["edge calls per request"]]
    rows.append(out)
with open(T / "R18_online_same_budget_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting", "beta", "DriftGate-P", "best uncorrected rule", "its accuracy", "DriftGate-P minus it pp mean (SD)",
                "best corrected rule", "its accuracy", "DriftGate-P minus it pp mean (SD)", "realized offloading", "edge calls per request"])
    w.writerows(rows)
for r in rows:
    if r[1] in ("0.5", "1.0"):
        print(" | ".join(r))
