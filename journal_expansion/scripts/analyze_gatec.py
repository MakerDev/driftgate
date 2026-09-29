"""Gate-C analysis: fairness mechanisms on spatial equal_spread.

Compares (per seed, then paired): sc_delta_b10 (no fairness) vs fair_deploy /
fair_oracle / fair_random vs fixed lam 0.2/0.4. Key metrics: worst-cell,
avg acc, cell gap, per-cell accuracy vector, plus F3 routing comparison.
Pre-registered pass: deploy - sc_delta worst-cell >= +2 pp, avg-acc loss
<= 0.5 pp (gates, not tuning targets).
"""
import json
import csv
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
RUNS = JOURNAL_ROOT / "runs/gatec_fairness"
TABLES = JOURNAL_ROOT / "tables"


def metrics(h):
    evals = h["eval"]
    late = [e for e in evals if e["round"] >= 50] or evals
    out = {
        "acc": float(np.mean([e["acc_total"] for e in late])),
        "worst": float(np.mean([e.get("worst_cell_acc", np.nan) for e in late])),
        "gap": float(np.mean([e.get("cell_gap", np.nan) for e in late])),
        "acc_final": evals[-1]["acc_total"],
        "worst_final": evals[-1].get("worst_cell_acc"),
        "cells_final": evals[-1].get("cell_means"),
    }
    rt = [e.get("routing") for e in late if e.get("routing")]
    if rt:
        out["rout_ent"] = float(np.mean([r["entropy"] for r in rt]))
        out["rout_dis"] = float(np.mean([r["disagree_conf"] for r in rt]))
        out["rout_ent_worst"] = float(np.mean([r["entropy_worst_cell"] for r in rt]))
        out["rout_dis_worst"] = float(np.mean([r["disagree_conf_worst_cell"] for r in rt]))
    return out


def main():
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        if f.name == "manifest.json":
            continue
        tag = f.stem
        seed = int(tag.rsplit("_s", 1)[1])
        method = tag.rsplit("_s", 1)[0]
        m = metrics(json.load(open(f)))
        m.update(method=method, seed=seed)
        rows.append(m)
    if not rows:
        print("no Gate-C runs yet")
        return
    TABLES.mkdir(exist_ok=True)
    cols = ["method", "seed", "acc", "worst", "gap", "acc_final", "worst_final",
            "rout_ent", "rout_dis", "rout_ent_worst", "rout_dis_worst", "cells_final"]
    with open(TABLES / "fairness_results.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    base = {r["seed"]: r for r in rows if r["method"] == "sc_delta_b10_sp"}
    print(f"{'method':<20}{'acc':>8}{'worst':>8}{'gap':>8}{'Δworst vs base':>15}{'Δacc':>8}")
    for method in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == method]
        dw = [r["worst"] - base[r["seed"]]["worst"] for r in rs if r["seed"] in base]
        da = [r["acc"] - base[r["seed"]]["acc"] for r in rs if r["seed"] in base]
        print(f"{method:<20}{np.mean([r['acc'] for r in rs]):>8.4f}"
              f"{np.mean([r['worst'] for r in rs]):>8.4f}"
              f"{np.mean([r['gap'] for r in rs]):>8.4f}"
              f"{np.mean(dw) if dw else float('nan'):>+15.4f}"
              f"{np.mean(da) if da else float('nan'):>+8.4f}")
    # routing gain (F3), all methods pooled
    re_, rd = [r.get("rout_ent") for r in rows], [r.get("rout_dis") for r in rows]
    re_ = [x for x in re_ if x is not None]
    rd = [x for x in rd if x is not None]
    if re_:
        print(f"\nF3 routing (pooled): entropy {np.mean(re_):.4f} vs "
              f"disagree_conf {np.mean(rd):.4f} (Δ {np.mean(rd)-np.mean(re_):+.4f})")


if __name__ == "__main__":
    main()
