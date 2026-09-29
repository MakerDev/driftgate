"""Gate-D block analyzer: per-block summary with paired stats vs a reference arm.

Usage: python analyze_gated.py d1_unseen [ref_arm=main]
Parses runs/gated/<block>/<block-prefix>_<arm>_<sched>_s<seed>.json, groups by
(schedule, arm), prints mean±std, paired diffs vs the best FIXED arm and vs main,
sign counts, and paired t statistic. Writes tables/gated_<block>.csv.
"""
import json
import csv
import sys
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOURNAL_ROOT / "scripts"))
from analyze_downstream import run_metrics


def paired_t(diffs):
    d = np.asarray(diffs, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    block = sys.argv[1] if len(sys.argv) > 1 else "d1_unseen"
    runs_dir = JOURNAL_ROOT / "runs/gated" / block
    rows = []
    for f in sorted(runs_dir.glob("*.json")):
        tag = f.stem  # d1_<arm>_<sched>_s<n>  (arm may contain no underscores)
        parts = tag.split("_")
        seed = int(parts[-1][1:])
        arm = parts[1]
        sched = "_".join(parts[2:-1])
        m = run_metrics(json.load(open(f)))
        m.update(arm=arm, schedule=sched, seed=seed)
        rows.append(m)

    tab = JOURNAL_ROOT / "tables" / f"gated_{block}.csv"
    cols = ["arm", "schedule", "seed", "integrated", "worst_cell_mean", "p10_client",
            "lam_switch_per_round", "final_acc"]
    with open(tab, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    scheds = sorted({r["schedule"] for r in rows})
    for sched in scheds:
        sr = [r for r in rows if r["schedule"] == sched]
        by_arm = {}
        for r in sr:
            by_arm.setdefault(r["arm"], {})[r["seed"]] = r
        # hindsight best fixed per schedule
        fixed_arms = [a for a in by_arm if a.startswith("fixed")]
        best_fixed = max(fixed_arms, key=lambda a: np.mean(
            [v["integrated"] for v in by_arm[a].values()])) if fixed_arms else None
        print(f"\n=== {sched} (best fixed = {best_fixed}) ===")
        print(f"{'arm':<10}{'integ':>8}{'±':>7}{'worst':>8}{'p10':>8}"
              f"{'Δvs bestfix':>12}{'t':>6}{'sign':>6}")
        for arm in sorted(by_arm):
            vals = by_arm[arm]
            ints = [v["integrated"] for v in vals.values()]
            line = (f"{arm:<10}{np.mean(ints):>8.4f}{np.std(ints):>7.4f}"
                    f"{np.mean([v['worst_cell_mean'] for v in vals.values()]):>8.4f}"
                    f"{np.mean([v['p10_client'] for v in vals.values()]):>8.4f}")
            if best_fixed and arm != best_fixed:
                bf = by_arm[best_fixed]
                common = sorted(set(vals) & set(bf))
                d = [vals[s]["integrated"] - bf[s]["integrated"] for s in common]
                pos = sum(1 for x in d if x > 0)
                line += f"{np.mean(d):>+12.4f}{paired_t(d):>6.1f}{pos:>4}/{len(d)}"
            print(line)


if __name__ == "__main__":
    main()
