"""Round 18 (own-class detector): context of an improvement attempt's gate (directive 7.4) on the S1 development seeds 8-10,
frozen-replay: the reference rows and the comparator envelope of the run without the change and of the attempt's run, per seed.
Descriptive only; the gate itself is r18oc_tables.attempt_dev (decision_rule.md 2.4, decisions.md D12).

  python .../r18oc_attempt_context.py <attempt n> <model config>   # tables/R18oc_attempt<n>_<tag>_dev_context.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tb", HERE / "scripts" / "r18oc_tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
an = T.an
SCORES = ["device-entropy", "device-max-prob", "device-energy", "own-mahalanobis", "own-knn"]
ANSWERS = ["edge-raw", "edge-corrected-even", "average-corrected-even", "probability-average", "logit-sum"]


def row(o, v, j):
    comp = {(s, a): o["curves"][(s, a)][j] for s in SCORES for a in ANSWERS}
    best = max(comp, key=comp.get)
    return [100 * o["reference"]["Device only"], 100 * o["reference"]["Edge only"], 100 * comp[best], "/".join(best),
            100 * o["curves"][(v, an.NEW)][j], o["auroc"][v]["all"]]


def context(n, cfg):
    v = T.chosen_variant(cfg)
    j = list(an.GRID).index(0.5)
    rows, M = [], {"without the change": [], f"attempt {n}": []}
    for s in T.DEV:
        for lab, att in (("without the change", None), (f"attempt {n}", n)):
            o = T.cache(cfg, "s1", s, "frozen-replay", att)
            if o is None:
                print(f"missing cache: seed {s} {lab}")
                return None
            r = row(o, v, j)
            M[lab].append([x for i, x in enumerate(r) if i != 3])
            rows.append([cfg, n, lab, s] + [f"{x:.2f}" for x in r[:3]] + [r[3], f"{r[4]:.2f}", f"{r[5]:.3f}"])
    for lab, X in M.items():
        X = np.array(X)
        rows.append([cfg, n, lab, "mean (SD)"] + [T.fmt(float(X[:, i].mean()), T.sd(X[:, i])) for i in range(3)] + [""]
                    + [T.fmt(float(X[:, 3].mean()), T.sd(X[:, 3])), f"{X[:, 4].mean():.3f}"])
    T.write(f"R18oc_attempt{n}_{T.TAG[cfg]}_dev_context.csv",
            ["model config", "attempt", "run", "seed", "Device only (%)", "Edge only (%)", "B(0.5) of the run (%)",
             "B(0.5) rule", "new method at 0.5 (%)", "AUROC all (chosen variant)"], rows)
    for r in rows:
        print(",".join(map(str, r)))


if __name__ == "__main__":
    context(int(sys.argv[1]), sys.argv[2])
