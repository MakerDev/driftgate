"""Round 18 A1: audit of the Round 17 comparison (strongest reference over the 17 rules, seed sets).

Reads the Round 16 phase A metadata (full-window accuracy of the 17 reference rules per run, Round 15 definitions) and the
Round 17 per-run caches (gate rules). Writes tables/R18_R17_corrected_margins.csv and tables/R18_R17_seed_sets.csv.
"""
import csv
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ART = HERE.parent
C16, C17 = ART / "driftgate_tmc_r16_selector" / "cache", ART / "driftgate_tmc_r17_featgate" / "cache"
SHOW = {"B0": "Confidence-based offloading", "B1": "Device only", "B2": "Edge only", "B3": "Probability average", "R-PoE": "Logit sum",
        "R-THE": "Lower-entropy exit", "LEW": "Logit-entropy weighting", "R-ZTW": "Geometric ensemble with early exit",
        "R-EM": "Label-shift EM", "R-LR": "Learned weight", "DriftGate": "DriftGate"}
R17_SET = ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "LEW", "R-ZTW", "R-EM", "R-LR", "corrected edge only"]


def meta(st, s):
    return json.loads((C16 / f"phaseT15_replay__r15_{st}_replay_s{s}__refs-v1.json").read_text())["full"]


def main():
    rows, seeds_rows = [], []
    for st, seeds, all_seeds in (("s1", [1, 2, 3, 4], [0, 1, 2, 3, 4]), ("s2", [0, 1, 2], [0, 1, 2])):
        M = [meta(st, s) for s in seeds]
        R17 = [pickle.load(open(C17 / f"r17_{st}_replay_s{s}.pkl", "rb"))["res"] for s in seeds]
        means = {k: np.mean([m[k] for m in M]) * 100 for k in M[0]}
        s17 = max(R17_SET, key=lambda k: means[k])
        s18 = max([k for k in means if k != "DriftGate"], key=lambda k: means[k])
        for rb in ("R-b D1@0.1", "R-b D2@0.1", "R-b D3", "R-b D4", "R-c D1@0.1 inf", "R-c D2@0.1 inf", "R-c D3 inf", "R-s D1@0.1"):
            v = np.array([r[rb]["all"] for r in R17]) * 100
            d17 = v - np.array([m[s17] for m in M]) * 100
            d18 = v - np.array([m[s18] for m in M]) * 100
            rows.append([st.upper() + " replay", " ".join(map(str, seeds)), rb, f"{v.mean():.3f}", SHOW.get(s17, s17), f"{d17.mean():+.3f} ({d17.std(ddof=1):.3f})",
                         SHOW.get(s18, s18), f"{means[s18]:.3f}", f"{d18.mean():+.3f} ({d18.std(ddof=1):.3f})"])
        dg_eval = np.mean([m["DriftGate"] for m in M]) * 100
        dg_all = np.mean([meta(st, s)["DriftGate"] for s in all_seeds]) * 100
        seeds_rows.append([st.upper() + " replay", " ".join(map(str, all_seeds)), f"{dg_all:.4f}", " ".join(map(str, seeds)), f"{dg_eval:.4f}",
                           " ".join(f"{meta(st, s)['DriftGate'] * 100:.4f}" for s in all_seeds)])
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    with open(out / "R18_R17_corrected_margins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "evaluation seeds", "Round 17 rule", "accuracy pct", "Round 17 strongest other (11 rules)", "minus it pp mean (SD)",
                    "strongest of the 17 reference rules", "its accuracy", "minus it pp mean (SD)"])
        w.writerows(rows)
    with open(out / "R18_R17_seed_sets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "Round 15 seeds", "DriftGate mean (Round 15 table A2)", "Round 17 evaluation seeds", "DriftGate mean (Round 17 table)",
                    "per seed"])
        w.writerows(seeds_rows)
    for r in rows + seeds_rows:
        print(r)


if __name__ == "__main__":
    main()
