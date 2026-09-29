"""Gate-B downstream analysis: controller-in-the-loop comparison.

Reads runs/downstream_pilot/*.json (+ the passive fixed-lambda runs as the
fixed baseline) and reports per (schedule): integrated accuracy (mean over
eval rounds), per-rho segment accuracy, worst-cell, p10-client accuracy,
lambda switching stats, and routing-mode comparison — per seed and mean+-std,
with paired per-seed differences vs the fixed baseline.
"""
import json
import sys
import csv
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
DOWN = JOURNAL_ROOT / "runs/downstream_pilot"
PASSIVE = JOURNAL_ROOT / "runs/signal_benchmark"
TABLES = JOURNAL_ROOT / "tables"


def run_metrics(h):
    evals = h["eval"]
    out = {
        "integrated": float(np.mean([e["acc_total"] for e in evals])),
        "worst_cell_mean": float(np.mean([e.get("worst_cell_acc", np.nan) for e in evals])),
        "final_acc": evals[-1]["acc_total"],
    }
    by_rho = {}
    for e in evals:
        r = e["rho"]
        if isinstance(r, dict):
            r = float(np.mean(list(r.values())))
        by_rho.setdefault(round(float(r), 2), []).append(e["acc_total"])
    for rho, v in sorted(by_rho.items()):
        out[f"rho{rho}"] = float(np.mean(v))
    # p10 client accuracy (mean over eval rounds)
    p10s = []
    for e in evals:
        pca = e.get("per_client_acc")
        if pca:
            p10s.append(float(np.percentile(list(pca.values()), 10)))
    out["p10_client"] = float(np.mean(p10s)) if p10s else float("nan")
    # lambda switching (keys may vary per round under packet loss — nan-pad)
    lams = [d for d in h.get("lamdas", []) if d]
    if lams:
        keys = sorted({k for d in lams for k in d})
        arr = np.array([[d.get(k, np.nan) for k in keys] for d in lams], dtype=float)
        out["lam_mean"] = float(np.nanmean(arr))
        out["lam_switch_per_round"] = float(np.nanmean(np.abs(np.diff(arr, axis=0))))
    # routing comparison (if recorded)
    r_ent, r_dis = [], []
    for e in evals:
        rt = e.get("routing")
        if rt:
            r_ent.append(rt["entropy"])
            r_dis.append(rt["disagree_conf"])
    if r_ent:
        out["routing_entropy"] = float(np.mean(r_ent))
        out["routing_disagree"] = float(np.mean(r_dis))
    return out


def main():
    TABLES.mkdir(exist_ok=True)
    rows = []
    for f in sorted(DOWN.glob("*.json")):
        if f.name == "manifest.json":
            continue
        tag = f.stem  # <controller>_<schedule>_s<seed>
        parts = tag.rsplit("_", 2)
        ctrl, sched, seed = parts[0], parts[1], int(parts[2][1:])
        m = run_metrics(json.load(open(f)))
        m.update(controller=ctrl, schedule=sched, seed=seed)
        rows.append(m)
    for f in sorted(PASSIVE.glob("rec_*.json")):
        parts = f.stem.split("_")
        seed = int(parts[-1][1:])
        sched = "_".join(parts[1:-1])
        m = run_metrics(json.load(open(f)))
        m.update(controller="fixed_lam0.4", schedule=sched, seed=seed)
        rows.append(m)

    if not rows:
        print("nothing to analyze yet")
        return
    cols = sorted({k for r in rows for k in r}, key=str)
    lead = ["controller", "schedule", "seed", "integrated", "worst_cell_mean",
            "p10_client", "final_acc"]
    cols = lead + [c for c in cols if c not in lead]
    with open(TABLES / "downstream_pilot.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # paired summary vs fixed baseline
    print(f"{'schedule':<10} {'controller':<16} {'integrated':>10} {'±':>6} "
          f"{'Δvs fixed':>9} {'worst':>7} {'p10':>7}")
    scheds = sorted({r["schedule"] for r in rows})
    for sched in scheds:
        base = {r["seed"]: r["integrated"] for r in rows
                if r["schedule"] == sched and r["controller"] == "fixed_lam0.4"}
        for ctrl in sorted({r["controller"] for r in rows if r["schedule"] == sched}):
            rs = [r for r in rows if r["schedule"] == sched and r["controller"] == ctrl]
            ints = [r["integrated"] for r in rs]
            diffs = [r["integrated"] - base[r["seed"]] for r in rs if r["seed"] in base]
            print(f"{sched:<10} {ctrl:<16} {np.mean(ints):>10.4f} {np.std(ints):>6.4f} "
                  f"{np.mean(diffs) if diffs else float('nan'):>+9.4f} "
                  f"{np.mean([r['worst_cell_mean'] for r in rs]):>7.4f} "
                  f"{np.mean([r['p10_client'] for r in rs]):>7.4f}")


if __name__ == "__main__":
    main()
