"""Task 0 — recompute every headline number from RAW run JSON (no markdown/CSV inputs).
Resolves the flagged inconsistencies and writes canonical files + an audit log.
"""
import json, glob, sys, csv
from pathlib import Path
import numpy as np

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]

def integrated(h):
    """Canonical metric: mean of acc_total over ALL eval rounds."""
    return float(np.mean([e["acc_total"] for e in h["eval"]]))

def eval_rounds(h):
    return tuple(e["round"] for e in h["eval"])

def high_drift(h, rho=0.8):
    v = [e["acc_total"] for e in h["eval"]
         if abs((e["rho"] if not isinstance(e["rho"], dict) else 0) - rho) < 1e-6]
    return float(np.mean(v)) if v else None

def worst(h):
    v = [e.get("worst_cell_acc") for e in h["eval"] if e.get("worst_cell_acc") is not None]
    return float(np.mean(v)) if v else None

def load_family(patterns):
    """patterns: list of globs (relative to JR). Returns {seed: (integ, rounds, high, worst, path)}."""
    out = {}
    for pat in patterns:
        for f in glob.glob(str(JR / pat)):
            try:
                s = int(f.split("_s")[-1].split(".")[0]); h = json.load(open(f))
            except Exception:
                continue
            out[s] = dict(integ=integrated(h), rounds=eval_rounds(h),
                          high=high_drift(h), worst=worst(h), path=f.split("journal_expansion/")[-1])
    return out

def seed_paired(a, b):
    c = sorted(set(a) & set(b))
    d = [a[s]["integ"] - b[s]["integ"] for s in c]
    return c, d

def dump_seed_csv(fam, name, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "seed", "integrated", "high_drift", "worst_cell", "n_eval_rounds", "path"])
        for method, d in fam.items():
            for s in sorted(d):
                r = d[s]
                w.writerow([method, s, round(r["integ"], 6), round(r["high"], 6) if r["high"] else "",
                            round(r["worst"], 6) if r["worst"] else "", len(r["rounds"]), r["path"]])

log = []
def L(x): log.append(x); print(x)

# ---------------- Schedule A ----------------
adaptA = load_family(["runs/phaseC_signals/dvsig_tv_A_s*.json", "runs/phaseS_supp/b1_adaptive_A_s*.json"])
# full grid at Lambda=0.5, all lambda
grid = {}
for lamk in ["fx00","fx10","fx20","fx30","fx40","fx50","fx60","fx70","fx80"]:
    fam = load_family([f"runs/phaseB_grid/{lamk}_A_s*.json", f"runs/phaseS_supp/b1_{lamk}_A_s*.json"])
    if fam: grid[lamk] = fam

L("# Task 0 — Headline Recomputation Audit (raw run JSON)\n")
L("## Schedule A")
L(f"adaptive DriftGate per-seed: " + ", ".join(f"s{s}={adaptA[s]['integ']:.4f}" for s in sorted(adaptA)))
L(f"  adaptive mean (n={len(adaptA)}) = {np.mean([adaptA[s]['integ'] for s in adaptA]):.6f}")
L(f"  adaptive eval-rounds set(s): {set(adaptA[s]['rounds'] for s in adaptA)}")
L("fixed grid (Λ=0.5) per-λ mean and seeds:")
grid_means = {}
for lamk in sorted(grid):
    fam = grid[lamk]; seeds = sorted(fam); m = np.mean([fam[s]['integ'] for s in seeds])
    grid_means[lamk] = (m, seeds)
    L(f"  {lamk}: mean {m:.6f} seeds {seeds} " + " ".join(f"s{s}={fam[s]['integ']:.4f}" for s in seeds))
# best fixed on the seed set shared with adaptive (5 seeds if available)
A5 = set(adaptA)
# recompute best fixed restricted to seeds present in adaptive, per lambda, and pick argmax on the COMMON seed set
best_lamk, best_mean, best_seeds = None, -1, None
for lamk, fam in grid.items():
    common = sorted(A5 & set(fam))
    if len(common) >= 2:
        m = np.mean([fam[s]['integ'] for s in common])
        if m > best_mean: best_mean, best_lamk, best_seeds = m, lamk, common
L(f"\nBest fixed on the adaptive-shared seed set: {best_lamk} mean {best_mean:.6f} seeds {best_seeds}")
common = sorted(A5 & set(grid[best_lamk]))
d = [adaptA[s]['integ'] - grid[best_lamk][s]['integ'] for s in common]
L(f"adaptive − {best_lamk} paired (seeds {common}, n={len(common)}): "
  f"mean {np.mean(d)*100:+.3f}pp, per-seed {[round(x,4) for x in d]}")
# 2-seed subset (s1,s2 originally reported)
sub = [s for s in common if s in (1, 2)]
if len(sub) == 2:
    d2 = [adaptA[s]['integ'] - grid[best_lamk][s]['integ'] for s in sub]
    L(f"  [2-seed subset s1,s2]: {best_lamk} mean {np.mean([grid[best_lamk][s]['integ'] for s in sub]):.4f}, adaptive−fixed {np.mean(d2)*100:+.3f}pp")

dump_seed_csv({"adaptive_DriftGate": adaptA}, "A", JR / "tables/canonical_scheduleA_per_seed.csv")
with open(JR / "tables/canonical_scheduleA_fixed_grid.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lambda_key","seed","integrated","n_eval_rounds","path"])
    for lamk in sorted(grid):
        for s in sorted(grid[lamk]):
            w.writerow([lamk, s, round(grid[lamk][s]['integ'],6), len(grid[lamk][s]['rounds']), grid[lamk][s]['path']])

# ---------------- Task C recompute (Schedule A cells) ----------------
L("\n## Task C Schedule-A cell recomputation")
dg_taskC = load_family(["runs/phaseC_signals/dvsig_tv_A_s*.json"])  # 3-seed as Task C used
robust = load_family(["runs/phaseB_grid/fx20_A_s*.json"])
L(f"  Task-C DriftGate (3-seed dvsig only) = {np.mean([dg_taskC[s]['integ'] for s in dg_taskC]):.6f} (seeds {sorted(dg_taskC)})")
L(f"  robust fixed λ0.2 (grid) = {np.mean([robust[s]['integ'] for s in robust]):.6f} (seeds {sorted(robust)})")
L(f"  hindsight best fixed (Λ0.5) = {best_lamk} but on 3-seed dvsig subset: {np.mean([grid[best_lamk][s]['integ'] for s in sorted(set(dg_taskC)&set(grid[best_lamk]))]):.6f}")

# ---------------- asym-return ----------------
L("\n## asym-return")
dg_as = load_family(["runs/phaseD_final/dv2_asym_return_s*.json"])
fx04_as = load_family(["runs/gated/d1_unseen/d1_fixed04_asym_return_s*.json"])
fx02_as = load_family(["runs/gated/d1_unseen/d1_fixed02_asym_return_s*.json"])
for nm, fam in [("DriftGate", dg_as), ("fixed λ0.4", fx04_as), ("fixed λ0.2(robust)", fx02_as)]:
    L(f"  {nm}: mean {np.mean([fam[s]['integ'] for s in fam]):.6f} seeds {sorted(fam)}")
c, d = seed_paired(dg_as, fx04_as); L(f"  DriftGate − fixed0.4 (best): {np.mean(d)*100:+.3f}pp seeds {c}")
c, d = seed_paired(dg_as, fx02_as); L(f"  DriftGate − fixed0.2 (robust/TaskC): {np.mean(d)*100:+.3f}pp seeds {c}")

# ---------------- CIFAR-100 spatial ----------------
L("\n## CIFAR-100 spatial fixed grid")
c100sp = {}
for lamk in ["fx00","fx10","fx20","fx30","fx40","fx50","fx60","fx70","fx80"]:
    fam = load_family([f"runs/phaseB_grid/{lamk}_c100sp_s*.json"])
    if fam: c100sp[lamk] = np.mean([fam[s]['integ'] for s in fam]), sorted(fam)
for lamk in sorted(c100sp):
    L(f"  {lamk}: {c100sp[lamk][0]:.6f} seeds {c100sp[lamk][1]}")
bestk = max(c100sp, key=lambda k: c100sp[k][0])
L(f"  >>> grid MAX = {bestk} {c100sp[bestk][0]:.6f}; fx20 = {c100sp.get('fx20',['?'])[0]}")

# canonical headline json
headline = {
 "scheduleA_adaptive_mean_5seed": float(np.mean([adaptA[s]['integ'] for s in adaptA])),
 "scheduleA_adaptive_per_seed": {s: adaptA[s]['integ'] for s in sorted(adaptA)},
 "scheduleA_best_fixed_key": best_lamk, "scheduleA_best_fixed_mean_5seed": best_mean,
 "scheduleA_adaptive_minus_bestfixed_5seed_pp": float(np.mean(d)*100) if False else float(np.mean([adaptA[s]['integ']-grid[best_lamk][s]['integ'] for s in common])*100),
 "taskC_dg_3seed": float(np.mean([dg_taskC[s]['integ'] for s in dg_taskC])),
 "robust_fixed_lam02_A": float(np.mean([robust[s]['integ'] for s in robust])),
 "c100sp_grid_max_key": bestk, "c100sp_grid_max": c100sp[bestk][0], "c100sp_fx20": c100sp.get('fx20',[None])[0],
}
json.dump(headline, open(JR / "tables/canonical_headline_metrics.json", "w"), indent=1)
open(JR / "reports/headline_metric_audit_log.md", "w").write("\n".join(log) + "\n")
print("\nwrote canonical_* + headline_metric_audit_log.md")
