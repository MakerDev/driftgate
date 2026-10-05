"""Round 16 stage 1 (calibration_route.md): facts about the labeled pools, no accuracy.

For CIFAR-10 and CIFAR-100 and the run seeds, rebuild the probe (20 %) / evaluation (80 %) split of
src/disjoint_pools.make_pool_masks exactly as the runner does, and report per-class counts and overlaps. Training uses the
CIFAR training set only; evaluation requests are built from the evaluation pool only (runner_r6: eval_rb = RequestBuilder(eval_labels, ...)).
Output: tables/R16_C_pools.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
sys.path.insert(0, str(JR))
sys.path.insert(0, str(JR.parent))
from src.disjoint_pools import make_pool_masks          # noqa: E402


def labels(name):
    from torchvision import datasets
    if name == "CIFAR-10":
        return np.array(datasets.CIFAR10(str(JR.parent / "data_cache"), train=False, download=False).targets)
    from src.datasets_ext import DATA_ROOT
    return np.array(datasets.CIFAR100(str(DATA_ROOT / "cifar100"), train=False, download=False).targets)


def main():
    rows = []
    for ds, C, seeds in (("CIFAR-10", 10, range(8)), ("CIFAR-100", 100, range(3))):
        y = labels(ds)
        for s in seeds:
            probe, ev, man = make_pool_masks(y, C, 0.2, seed=s)
            pc = np.bincount(probe[probe >= 0], minlength=C)
            ec = np.bincount(ev[ev >= 0], minlength=C)
            overlap = int(((probe >= 0) & (ev >= 0)).sum())
            unused = int(((probe < 0) & (ev < 0)).sum())
            rows.append([ds, s, len(y), int((probe >= 0).sum()), int((ev >= 0).sum()), overlap, unused,
                         int((pc > 0).sum()), int(pc.min()), int(pc.max()), int(ec.min()), int(ec.max())])
    out = HERE / "tables" / "R16_C_pools.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "test_images", "probe_pool", "evaluation_pool", "overlap", "in_neither",
                    "classes_in_probe_pool", "probe_per_class_min", "probe_per_class_max", "eval_per_class_min", "eval_per_class_max"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
