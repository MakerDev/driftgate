"""Round 5 pre-checks.
1) entropy runs (A, mobility, SVHN) and B1 relonly runs follow the same controller path:
   no absolute cap, same constants. Behavioural test: for every post-warm-up cluster-round,
   recorded lambda == 0.70 - 0.55*sigmoid((z-1.5)/0.75) and Lambda == 0.70 - 0.30*sigmoid(...),
   where z is the logged controller_z (max of temporal/spatial EMA score after 1-step consensus).
   Warm-up (n_obs <= 25): lambda = 0.425, Lambda = 0.55.
2) B1 relonly: Lambda is set by the relative score (same test), and the absolute candidate
   log (history["lam_abs"]) is empty in every round.
"""
import json, glob, re, math, csv, sys
from pathlib import Path
JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT = Path(__file__).resolve().parent
def sig(x): return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))
GROUPS = {
  "entropy A":        "runs/phaseT1_disjoint/t1_A_ent_s*.json",
  "entropy mobility": "runs/phaseT1_disjoint/t1_mob_ent_s*.json",
  "entropy SVHN":     "runs/phaseT2_signal/t2_svhn_ent_s*.json",
  "relonly A":        "runs/phaseT4_B1_views/b1_relonly_A_s*.json",
  "relonly mobility": "runs/phaseT4_B1_views/b1_relonly_mob_s*.json",
  "relonly C100 gradual": "runs/phaseT4_B1_views/b1_relonly_c100gsig_s*.json",
}
rows = []
for grp, pat in GROUPS.items():
    for f in sorted(glob.glob(str(JR / pat))):
        h = json.load(open(f)); s = re.search(r"_s(\d+)\.json$", f).group(1)
        cz, lam, Lam = h["controller_z"], h["lamdas"], h["big_lamdas"]
        n = 0; e_lam = 0.0; e_Lam = 0.0; warm_bad = 0
        for r in range(len(lam)):
            for es in lam[r]:
                if r + 1 <= 25:   # burn-in 10 + warm-up 15 -> midpoints
                    warm_bad += (abs(lam[r][es] - 0.425) > 1e-9) + (abs(Lam[r][es] - 0.55) > 1e-9)
                    continue
                g = sig((float(cz[r][es]) - 1.5) / 0.75)
                e_lam = max(e_lam, abs(lam[r][es] - (0.70 - 0.55 * g)))
                e_Lam = max(e_Lam, abs(Lam[r][es] - (0.70 - 0.30 * g)))
                n += 1
        la = h.get("lam_abs")
        abs_log = "not logged (pre-R4 run)" if la is None else ("empty in all rounds" if all(len(d) == 0 for d in la) else "NON-EMPTY")
        rows.append([grp, s, n, f"{e_lam:.2e}", f"{e_Lam:.2e}", warm_bad, abs_log])
with open(OUT / "precheck_controller_path.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["group", "seed", "post_warmup_cluster_rounds", "max_abs_err_lambda", "max_abs_err_Lambda", "warmup_mismatches", "lam_abs_log"])
    w.writerows(rows)
for r in rows: print(r)
ok = all(float(r[3]) < 1e-9 and float(r[4]) < 1e-9 and r[5] == 0 for r in rows)
print("ALL MATCH relative-only mapping (no absolute cap, same constants):", ok)
