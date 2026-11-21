"""Round 18 (own-class detector): diagnosis of the stage-2 failure (directive 7.4 item 3: "diagnose the cause from the
stage-2 results"). Reads the stage-2 evaluation caches only (and the per-kind request counts, cached by r18oc_stage3.n_kind).

  python .../r18oc_diagnose.py stage2

Tables (tables/R18oc_stage2_diagnosis_*.csv), pretrained configurations, S1/S2/CIFAR-100, every condition:
  rules   seed-mean accuracy at ratio 0.25/0.5/0.75 of every (ranking, answer rule) pair recorded in the caches: the chosen
          detector variant with edge-corrected-by-detector, edge-raw, edge-corrected-even and the device-0.2 average;
          device-max-prob and device-entropy with the five comparator answer rules; the request-kind oracle; envelope B.
          Splits the gap to B into a ranking part (same answer rule, different ranking) and an answer part (same ranking).
  kind    accuracy by request kind (Main, OOP, OOR) at 0.25/0.5/0.75 for the same pairs (S1 and S2 only).
  r       the mean r of edge-corrected-by-detector on the offloaded requests at ratio 0.5 and the s-hat error, from the
          extras of the caches.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("s3", HERE / "scripts" / "r18oc_stage3.py")
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)
T, an = S3.T, S3.an
GRID, SCORES, ANSWERS, NEW, NEW_AVG, ORACLE = an.GRID, an.SCORES, an.ANSWERS, an.NEW, an.NEW_AVG, an.ORACLE
RHOS = (0.25, 0.5, 0.75)
JR = [list(GRID).index(r) for r in RHOS]
CONDS = ("first-day", "second-day", "frozen-replay")
NAME = {"s1": "S1", "s2": "S2", "c100": "CIFAR-100"}


def pairs(main):
    P = [("detector", main, a) for a in (NEW, "edge-raw", "edge-corrected-even", NEW_AVG)]
    P += [("device-max-prob", "device-max-prob", a) for a in ANSWERS]
    P += [("device-entropy", "device-entropy", a) for a in ANSWERS]
    P += [("oracle", ORACLE, a) for a in ("edge-raw", "edge-corrected-even")]
    return P


def stage2():
    rows_rules, rows_kind, rows_r = [], [], []
    for cfg in T.PRETRAINED:
        for st in ("s1", "s2", "c100"):
            for cond in CONDS:
                O = [T.cache(cfg, st, s, cond) for s in T.SEEDS[st]]
                main = O[0]["main"]
                comp = [(s, a) for s in SCORES for a in ANSWERS]
                per_env = np.max(np.stack([np.mean([o["curves"][k] for o in O], axis=0) for k in comp]), axis=0)
                env = 100 * per_env
                rows_rules.append([NAME[st], cfg, cond, "envelope B", "max over the 25 comparators"]
                                  + [f"{env[j]:.2f}" for j in JR] + [""] * 3)
                for lab, rk, ans in pairs(main):
                    per = 100 * np.array([o["curves"][(rk, ans)] for o in O])
                    m = per.mean(0)
                    rows_rules.append([NAME[st], cfg, cond, lab, ans] + [f"{m[j]:.2f} ({T.sd(per[:, j]):.2f})" for j in JR]
                                      + [f"{m[j] - env[j]:+.2f}" for j in JR])
                if st == "c100":
                    continue
                names = [f"r18oc_{T.TAG[cfg]}_{st}_s{s}" for s in T.SEEDS[st]]
                nk = [S3.n_kind(nm, cond) for nm in names]
                for o, nks in zip(O, nk):
                    tot = nks.sum(axis=(1, 2))
                    o["_share"] = tot / tot.sum()
                share = np.mean([o["_share"] for o in O], axis=0)
                for lab, rk, ans in pairs(main):
                    vals = {}
                    for o, nks in zip(O, nk):
                        for rho in RHOS:
                            for kk, kn in enumerate(("Main", "OOP", "OOR")):
                                cnt = o["cnt_kind"][(rk, ans)][rho][kk].astype(float)
                                v = S3.macro_mask(cnt, nks[kk], o["present"], np.ones_like(o["present"], bool))
                                vals.setdefault((rho, kn), []).append(100 * v)
                    rows_kind.append([NAME[st], cfg, cond, lab, ans]
                                     + [f"{np.nanmean(vals[(rho, kn)]):.2f}" for rho in RHOS for kn in ("Main", "OOP", "OOR")])
                rows_r.append([NAME[st], cfg, cond, main] + [f"{100 * x:.1f}" for x in share]
                              + [f"{np.mean([o['extra'][main]['curve 0.5']['r_mean'] for o in O]):.3f}",
                                 f"{np.mean([o['extra'][main]['curve 0.5']['s_err']['all'] for o in O]):.3f}",
                                 f"{np.mean([o['extra'][main]['fixed 0.5']['ratio'] for o in O]):.3f}"])
    S3.write("R18oc_stage2_diagnosis_rules.csv",
             ["setting", "model config", "condition", "ranking", "answer rule"] + [f"accuracy at {r} mean (SD)" for r in RHOS]
             + [f"minus B({r}) pp" for r in RHOS], rows_rules)
    S3.write("R18oc_stage2_diagnosis_kind.csv",
             ["setting", "model config", "condition", "ranking", "answer rule"]
             + [f"{kn} accuracy at {r}" for r in RHOS for kn in ("Main", "OOP", "OOR")], rows_kind)
    S3.write("R18oc_stage2_diagnosis_r.csv",
             ["setting", "model config", "condition", "detector variant", "Main share (%)", "OOP share (%)", "OOR share (%)",
              "mean r on offloaded requests at 0.5", "s-hat error at 0.5", "offloading ratio at fixed threshold 0.5"], rows_r)


if __name__ == "__main__":
    {"stage2": stage2}[sys.argv[1]]()
