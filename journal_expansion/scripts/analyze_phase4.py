"""Phase-4 analysis: causal online baselines vs selfcal v3c / legacy / fixed.

Pulls the phase4_baselines runs plus the matched downstream_pilot runs (same
seeds/schedules) and reports: integrated acc, worst-cell, p10, lambda switching
frequency, and empirical regret vs the greedy labeled causal oracle (B11).
Output: tables/phase4_baselines.csv + printed summary.
"""
import json
import csv
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
P4 = JOURNAL_ROOT / "runs/phase4_baselines"
DP = JOURNAL_ROOT / "runs/downstream_pilot"
TABLES = JOURNAL_ROOT / "tables"

import sys
sys.path.insert(0, str(JOURNAL_ROOT / "scripts"))
from analyze_downstream import run_metrics


def collect():
    rows = []
    for f in sorted(P4.glob("*.json")):
        tag = f.stem
        ctrl, sched, seed = tag.rsplit("_", 2)[0], tag.rsplit("_", 2)[1], int(tag.rsplit("_", 2)[2][1:])
        m = run_metrics(json.load(open(f)))
        m.update(controller=ctrl, schedule=sched, seed=seed)
        rows.append(m)
    for f in sorted(DP.glob("*.json")):
        if f.name == "manifest.json":
            continue
        tag = f.stem
        parts = tag.rsplit("_", 2)
        ctrl, sched, seed = parts[0], parts[1], int(parts[2][1:])
        if ctrl not in ("sc_delta_g05", "legacy_delta") or sched not in ("A", "abrupt"):
            continue
        m = run_metrics(json.load(open(f)))
        m.update(controller=ctrl, schedule=sched, seed=seed)
        rows.append(m)
    # fixed baseline from passive recordings
    for f in sorted((JOURNAL_ROOT / "runs/signal_benchmark").glob("rec_*.json")):
        parts = f.stem.split("_")
        seed = int(parts[-1][1:])
        sched = "_".join(parts[1:-1])
        if sched not in ("A", "abrupt"):
            continue
        m = run_metrics(json.load(open(f)))
        m.update(controller="fixed_lam0.4", schedule=sched, seed=seed)
        rows.append(m)
    return rows


def main():
    rows = collect()
    TABLES.mkdir(exist_ok=True)
    cols = ["controller", "schedule", "seed", "integrated", "worst_cell_mean",
            "p10_client", "lam_mean", "lam_switch_per_round", "final_acc"]
    with open(TABLES / "phase4_baselines.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for sched in ("A", "abrupt"):
        oracle = {r["seed"]: r["integrated"] for r in rows
                  if r["schedule"] == sched and r["controller"] == "oracle_greedy"}
        print(f"\n=== {sched} ===")
        print(f"{'controller':<16}{'integ':>8}{'±':>7}{'worst':>8}{'switch':>8}{'regret_vs_B11':>14}")
        for ctrl in sorted({r["controller"] for r in rows if r["schedule"] == sched}):
            rs = [r for r in rows if r["schedule"] == sched and r["controller"] == ctrl]
            ints = [r["integrated"] for r in rs]
            reg = [oracle[r["seed"]] - r["integrated"] for r in rs if r["seed"] in oracle]
            sw = [r.get("lam_switch_per_round") for r in rs if r.get("lam_switch_per_round") is not None]
            print(f"{ctrl:<16}{np.mean(ints):>8.4f}{np.std(ints):>7.4f}"
                  f"{np.mean([r['worst_cell_mean'] for r in rs]):>8.4f}"
                  f"{np.mean(sw) if sw else float('nan'):>8.4f}"
                  f"{np.mean(reg) if reg else float('nan'):>+14.4f}")


if __name__ == "__main__":
    main()
