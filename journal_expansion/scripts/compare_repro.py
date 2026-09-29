"""Phase-0 reproduction comparison: new E2 Schedule-A adaptive run vs the
canonical v4 result. Prints the table for phase0_reproduction_and_audit.md."""
import json
import sys
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent

ORIG = PROJECT_ROOT / "results/e2_temporal/schedule_A/adaptive_splitomc.json"
REPRO = JOURNAL_ROOT / "runs/reproduction/e2_temporal/schedule_A/adaptive_splitomc.json"


def seg_stats(hist):
    evals = hist["eval"]
    out = {"integrated": float(np.mean([e["acc_total"] for e in evals]))}
    by_rho = {}
    for e in evals:
        by_rho.setdefault(float(e["rho"]), []).append(e["acc_total"])
    for rho, v in sorted(by_rho.items()):
        out[f"rho={rho}"] = float(np.mean(v))
    out["final_acc"] = evals[-1]["acc_total"]
    out["worst_cell_final"] = evals[-1].get("worst_cell_acc")
    out["n_evals"] = len(evals)
    return out


def main():
    for p in (ORIG, REPRO):
        if not p.exists():
            print(f"MISSING: {p}")
            sys.exit(1)
    a = seg_stats(json.load(open(ORIG)))
    b = seg_stats(json.load(open(REPRO)))
    keys = sorted(set(a) | set(b))
    print(f"{'metric':<20} {'ICTC(v4)':>10} {'repro':>10} {'abs diff':>10} {'rel %':>8}")
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None or not isinstance(va, float):
            print(f"{k:<20} {va} {vb}")
            continue
        d = vb - va
        rel = 100 * d / va if va else float("nan")
        print(f"{k:<20} {va:>10.4f} {vb:>10.4f} {d:>+10.4f} {rel:>+7.2f}%")


if __name__ == "__main__":
    main()
