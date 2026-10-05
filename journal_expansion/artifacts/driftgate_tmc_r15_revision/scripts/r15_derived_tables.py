"""Round 15: tables derived from the cached per-run summaries of r15_analysis.py (no new computation of answers).

R15_E_scale_reach.csv: offloading ratio at which the partial-offloading DriftGate curve (DriftGate-P, both mean
entropies over offloaded requests) first reaches the whole-day accuracy of confidence-based offloading, per run
(Round 13b first_reach on the run's own curve), mean (SD) over seeds; for S1, K=200 and K=500.
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


def main():
    lr, lr_corr = m15.m14.read_lr(), m15.fit_lr_corr()
    rows = []
    for st in ("S1", "K=200", "K=500"):
        rd, pat, seeds = m15.SETTINGS[st]
        A = m15.run_many([(str(rd), pat.format(s), lr, lr_corr) for s in seeds])
        reach = [m15.m13.first_reach(a["curves"]["full"]["DriftGate-P"]["srv"], a["curves"]["full"]["DriftGate-P"]["acc"],
                                     a["res"]["B0"]["full"]) for a in A]
        b0s = [a["srv"]["full"]["B0"] for a in A]
        rows.append([st, A[0]["K"], m15.fmt(np.array(b0s), 3), m15.fmt(np.array(reach), 3), " ".join(f"{x:.3f}" for x in reach)])
    with open(HERE / "tables" / "R15_E_scale_reach.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "devices", "B0 offloading mean (SD)", "DriftGate-P first reaches B0 accuracy mean (SD)", "per seed"])
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
