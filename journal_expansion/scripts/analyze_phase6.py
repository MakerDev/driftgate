"""Phase-6 analysis: network impairments vs the clean v3b baseline.

All phase6 runs are selfcal v3b (burn-in 10, default guard) on `abrupt`; the
matched clean baseline is sc_delta_b10_abrupt_s{0,1,2} from downstream_pilot.
Reports paired deltas per impairment. Output: tables/network_robustness.csv.
"""
import json
import csv
from pathlib import Path
import numpy as np
import sys

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
P6 = JOURNAL_ROOT / "runs/phase6_network"
DP = JOURNAL_ROOT / "runs/downstream_pilot"
TABLES = JOURNAL_ROOT / "tables"

sys.path.insert(0, str(JOURNAL_ROOT / "scripts"))
from analyze_downstream import run_metrics


def main():
    base = {}
    for s in (0, 1, 2):
        f = DP / f"sc_delta_b10_abrupt_s{s}.json"
        if f.exists():
            base[s] = run_metrics(json.load(open(f)))
    rows = []
    for f in sorted(P6.glob("*.json")):
        tag = f.stem  # e.g. delay5_s1, loss10_s0, topo_ring_s0, part50_s2
        seed = int(tag.rsplit("_s", 1)[1])
        cond = tag.rsplit("_s", 1)[0]
        m = run_metrics(json.load(open(f)))
        m.update(condition=cond, seed=seed)
        if seed in base:
            m["d_integrated"] = m["integrated"] - base[seed]["integrated"]
            m["d_worst"] = m["worst_cell_mean"] - base[seed]["worst_cell_mean"]
        rows.append(m)

    TABLES.mkdir(exist_ok=True)
    cols = ["condition", "seed", "integrated", "worst_cell_mean", "p10_client",
            "lam_switch_per_round", "d_integrated", "d_worst"]
    with open(TABLES / "network_robustness.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"clean v3b baseline (abrupt, 3 seeds): "
          f"integ {np.mean([b['integrated'] for b in base.values()]):.4f} "
          f"worst {np.mean([b['worst_cell_mean'] for b in base.values()]):.4f}")
    print(f"{'condition':<14}{'n':>3}{'integ':>8}{'Δinteg':>9}{'Δworst':>9}{'switch':>8}")
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        di = [r["d_integrated"] for r in rs if "d_integrated" in r]
        dw = [r["d_worst"] for r in rs if "d_worst" in r]
        sw = [r.get("lam_switch_per_round") for r in rs if r.get("lam_switch_per_round") is not None]
        print(f"{cond:<14}{len(rs):>3}{np.mean([r['integrated'] for r in rs]):>8.4f}"
              f"{np.mean(di) if di else float('nan'):>+9.4f}"
              f"{np.mean(dw) if dw else float('nan'):>+9.4f}"
              f"{np.mean(sw) if sw else float('nan'):>8.4f}")


if __name__ == "__main__":
    main()
