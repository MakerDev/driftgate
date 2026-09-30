"""R0 tables: the Round-5 table set recomputed with the new DriftGate definition (no neighbour score average).

1. Runs the Round-5 table code (artifacts/driftgate_tmc_final/scripts/r5_tables.py, read only) with the
   controller-run patterns swapped to the R0 runs (runs/phaseT6_R0/r0_<original name>) and the output
   folder set to artifacts/driftgate_tmc_r6/r0/. Fixed-lambda and APFL patterns are unchanged (those runs
   use no controller). Nothing in the Round-5 package is modified.
2. Writes r0/tables/R0_neighbor_avg_record.csv: for every R0 run group, the paired difference
   (without neighbour average - with neighbour average) over matched seeds, where "with" is the original
   Round-4/5 run. The original runs ran on ubuntu20 (RTX 3090 Ti), the R0 runs on honeynaps (RTX 4090),
   so the difference also contains the hardware (floating-point order) effect.
   Also the Round-6 commute-mobility runs that were made with the neighbour average before the decision
   (seeds with both versions).
"""
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parent.parent
R5 = JR / "artifacts" / "driftgate_tmc_final" / "scripts" / "r5_tables.py"
OUT = HERE / "r0"
RUNS = JR / "runs"

SWAP = [
    ("phaseT4_B1_views/b1_relonly_", "phaseT6_R0/r0_b1_relonly_"),
    ("phaseT4_B1_views/b1_absonly_", "phaseT6_R0/r0_b1_absonly_"),
    ("phaseT1_disjoint/t1_A_ent_", "phaseT6_R0/r0_t1_A_ent_"),
    ("phaseT1_disjoint/t1_mob_ent_", "phaseT6_R0/r0_t1_mob_ent_"),
    ("phaseT2_signal/t2_svhn_ent_", "phaseT6_R0/r0_t2_svhn_ent_"),
    ("phaseT5_E1_relonly_transfer/e1_relonly_", "phaseT6_R0/r0_e1_relonly_"),
    ("phaseT5_E2_resnet/e2_res_relonly_", "phaseT6_R0/r0_e2_res_relonly_"),
    ("phaseT5_E3_const/e3_", "phaseT6_R0/r0_e3_"),
]
GROUPS = [  # (label, original pattern; the R0 pattern is swap(original))
    ("stepwise composition change / DriftGate", "phaseT4_B1_views/b1_relonly_A_s*.json"),
    ("client mobility / DriftGate", "phaseT4_B1_views/b1_relonly_mob_s*.json"),
    ("CIFAR-100 gradual / DriftGate", "phaseT4_B1_views/b1_relonly_c100gsig_s*.json"),
    ("CIFAR-10 gradual / DriftGate", "phaseT5_E1_relonly_transfer/e1_relonly_c10gsig_s*.json"),
    ("Tiny-ImageNet / DriftGate", "phaseT5_E1_relonly_transfer/e1_relonly_tiny_s*.json"),
    ("SVHN temporal / DriftGate", "phaseT5_E1_relonly_transfer/e1_relonly_svhn_s*.json"),
    ("ResNet-18 middle split / DriftGate", "phaseT5_E2_resnet/e2_res_relonly_A_s*.json"),
    ("stepwise / guard 0.25", "phaseT5_E3_const/e3_guard025_A_s*.json"),
    ("stepwise / guard 1.0", "phaseT5_E3_const/e3_guard100_A_s*.json"),
    ("stepwise / lambda_max 0.60", "phaseT5_E3_const/e3_lmax060_A_s*.json"),
    ("stepwise / lambda_max 0.80", "phaseT5_E3_const/e3_lmax080_A_s*.json"),
    ("client mobility / lambda_max 0.60", "phaseT5_E3_const/e3_lmax060_mob_s*.json"),
    ("client mobility / lambda_max 0.80", "phaseT5_E3_const/e3_lmax080_mob_s*.json"),
    ("stepwise / entropy controller", "phaseT1_disjoint/t1_A_ent_s*.json"),
    ("client mobility / entropy controller", "phaseT1_disjoint/t1_mob_ent_s*.json"),
    ("SVHN temporal / entropy controller", "phaseT2_signal/t2_svhn_ent_s*.json"),
    ("stepwise / absonly", "phaseT4_B1_views/b1_absonly_A_s*.json"),
    ("client mobility / absonly", "phaseT4_B1_views/b1_absonly_mob_s*.json"),
    ("CIFAR-100 gradual / absonly", "phaseT4_B1_views/b1_absonly_c100gsig_s*.json"),
]


def swap(p):
    for a, b in SWAP:
        p = p.replace(a, b)
    return p


def run_r5_code():
    src = R5.read_text()
    src = src.replace('HERE = Path(__file__).resolve().parent.parent', f'HERE = Path(r"{OUT}")')
    n = 0
    for a, b in SWAP:
        n += src.count(a)
        src = src.replace(a, b)
    assert n >= 19, n
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "GENERATED_BY.txt").write_text(
        "Tables in this folder = Round-5 r5_tables.py (unchanged logic) with controller runs swapped to runs/phaseT6_R0 "
        "(no neighbour score average). Swaps:\n" + "\n".join(f"  {a} -> {b}" for a, b in SWAP) + "\n")
    exec(compile(src, str(R5) + " [R0 patterns]", "exec"), {"__name__": "__r0__", "__file__": str(R5)})


def integ(f):
    return float(np.mean([e["acc_total"] for e in json.load(open(f))["eval"]]))


def fam(pat):
    out = {}
    for f in sorted(glob.glob(str(RUNS / pat))):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            out[int(m.group(1))] = f
    return out


def record():
    rows = []

    def add(label, a_files, b_files, note):
        seeds = sorted(set(a_files) & set(b_files))
        if not seeds:
            rows.append([label, "", "", "", "", "", 0, note, "", ""])
            return
        a = np.array([integ(a_files[s]) for s in seeds]) * 100
        b = np.array([integ(b_files[s]) for s in seeds]) * 100
        d = b - a
        if len(d) > 1:
            h = stats.sem(d) * stats.t.ppf(0.975, len(d) - 1)
            ci = f"[{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]"
        else:
            ci = ""
        rows.append([label, f"{a.mean():.4f}", f"{b.mean():.4f}", f"{d.mean():+.4f}", ci, int((d > 0).sum()), len(d),
                     note, "{" + ",".join(map(str, seeds)) + "}", " ".join(f"{x:+.2f}" for x in d)])
    for label, pat in GROUPS:
        add(label, fam(pat), fam(swap(pat)), "with = Round 4/5 run (ubuntu20, RTX 3090 Ti); without = R0 (honeynaps, RTX 4090)")
    for arm, lab in (("driftgate", "DriftGate"), ("entropy", "entropy controller")):
        add(f"Round 6 commute mobility / {lab}", fam(f"phaseT6_S1/s1_{arm}_s*.json"), fam(f"phaseT6_S1/s1_{arm}_own_s*.json"),
            "both on honeynaps; the with-average runs were made before the 2026-10-01 decision")
    with open(OUT / "tables" / "R0_neighbor_avg_record.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting / arm", "with_neighbor_avg_pct", "without_neighbor_avg_pct", "without_minus_with_pp", "ci95",
                    "n_pos(without higher)", "n", "note", "seeds", "per_seed_pp"])
        w.writerows(rows)
    print(f"  [R0_neighbor_avg_record.csv] {len(rows)} rows")


if __name__ == "__main__":
    run_r5_code()
    record()
