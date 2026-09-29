"""Task-1 (complete) + Task-2 (partial) analysis — recompute from RAW run JSON.
Canonical metric: integrated = mean(acc_total over that run's eval rounds).
NEVER mixes seed counts in a subtraction; pairs strictly by seed with matched
eval-round vectors. Reads only raw JSON, never markdown/CSV summaries.
"""
import json, glob, re, sys
import numpy as np
from pathlib import Path
from scipy import stats

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]

def load(path):
    h = json.load(open(path))
    return h

def integrated(h):
    return float(np.mean([e["acc_total"] for e in h["eval"]]))

def eval_rounds(h):
    return tuple(e["round"] for e in h["eval"])

def fam(patterns):
    """{seed: dict(integ, rounds, overlap, path)} from a list of globs."""
    out = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(JR / pat))):
            m = re.search(r"_s(\d+)\.json$", f)
            if not m: continue
            s = int(m.group(1))
            if s in out: continue  # first pattern wins
            h = load(f)
            out[s] = dict(integ=integrated(h), rounds=eval_rounds(h),
                          overlap=h.get("probe_eval_overlap"), path=f,
                          lam=h.get("lamdas"), Lam=h.get("big_lamdas"))
    return out

def paired(a, b):
    """seed-aligned diffs a-b; asserts matched eval-round vectors."""
    seeds = sorted(set(a) & set(b))
    d = []
    for s in seeds:
        assert a[s]["rounds"] == b[s]["rounds"], f"eval-round mismatch seed {s}"
        d.append(a[s]["integ"] - b[s]["integ"])
    return seeds, np.array(d)

def ci95(d):
    n = len(d)
    if n < 2: return (float("nan"), float("nan"))
    se = stats.sem(d); h = se * stats.t.ppf(0.975, n-1)
    return (float(d.mean()-h), float(d.mean()+h))

L = []
def p(s=""): L.append(s); print(s)

# ============================================================ TASK 1
p("="*70); p("TASK 1 — DISJOINT probe/eval pools (recomputed from raw JSON)"); p("="*70)

D = "runs/phaseT1_disjoint"
A = {
 "DriftGate(TV)": fam([f"{D}/t1_A_dg_s*.json"]),
 "entropy":       fam([f"{D}/t1_A_ent_s*.json"]),
 "a1(adaptλ,Λ=.5)":fam([f"{D}/t1_A_a1_s*.json"]),
 "fixed λ0.3":    fam([f"{D}/t1_A_fx30_s*.json"]),
 "fixed λ0.4":    fam([f"{D}/t1_A_fx40_s*.json"]),
 "fixed λ0.5":    fam([f"{D}/t1_A_fx50_s*.json"]),
}
p("\n[Schedule A] per-seed integrated accuracy (disjoint pools)")
p(f"{'arm':18s} " + " ".join(f"s{s}" for s in range(5)) + "   mean")
for k,v in A.items():
    row = " ".join(f"{v[s]['integ']*100:5.2f}" if s in v else "  -- " for s in range(5))
    mean = np.mean([v[s]['integ'] for s in v])*100
    p(f"{k:18s} {row}   {mean:5.2f}  (n={len(v)})")

# overlap assertion
ov = [ (k,s,v[s]['overlap']) for k,v in A.items() for s in v ]
bad = [x for x in ov if x[2] != 0]
p(f"\nprobe∩eval overlap: {'ALL ZERO ✓' if not bad else 'NONZERO! '+str(bad)}  (checked {len(ov)} A-runs)")

# best fixed = argmax mean over fx grid on the disjoint seed set (seed-aligned)
fixed_keys = ["fixed λ0.3","fixed λ0.4","fixed λ0.5"]
fmean = {k: np.mean([A[k][s]['integ'] for s in A[k]]) for k in fixed_keys}
best_fixed = max(fmean, key=fmean.get)
p(f"\nbest fixed (by mean over 5 seeds) = {best_fixed}  ({fmean[best_fixed]*100:.3f}%)")

p("\n[Schedule A] PAIRED differences (seed-aligned, disjoint)")
for name, other in [("DriftGate − "+best_fixed, A[best_fixed]),
                    ("DriftGate − entropy", A["entropy"]),
                    ("DriftGate − a1(adaptλ,Λ=.5)", A["a1(adaptλ,Λ=.5)"])]:
    seeds, d = paired(A["DriftGate(TV)"], other)
    lo,hi = ci95(d); npos = int((d>0).sum())
    try: _,pw = stats.wilcoxon(d)
    except Exception: pw = float('nan')
    p(f"  {name:30s} {d.mean()*100:+.3f}pp  CI95[{lo*100:+.2f},{hi*100:+.2f}]  "
      f"{npos}/{len(d)} seeds+  per-seed={[round(x*100,2) for x in d]}")

# ---- mobility-med
Mb = {
 "DriftGate(TV)": fam([f"{D}/t1_mob_dg_s*.json"]),
 "entropy":       fam([f"{D}/t1_mob_ent_s*.json"]),
 "a1(adaptλ,Λ=.5)":fam([f"{D}/t1_mob_a1_s*.json"]),
 "fixed λ0.3":    fam([f"{D}/t1_mob_fx30_s*.json"]),
 "fixed λ0.4":    fam([f"{D}/t1_mob_fx40_s*.json"]),
 "fixed λ0.5":    fam([f"{D}/t1_mob_fx50_s*.json"]),
}
p("\n[mobility-med] per-seed integrated accuracy (disjoint pools)")
p(f"{'arm':18s} " + " ".join(f"s{s}" for s in range(3)) + "   mean")
for k,v in Mb.items():
    row = " ".join(f"{v[s]['integ']*100:5.2f}" if s in v else "  -- " for s in range(3))
    mean = np.mean([v[s]['integ'] for s in v])*100 if v else float('nan')
    p(f"{k:18s} {row}   {mean:5.2f}  (n={len(v)})")
ovm=[(k,s,v[s]['overlap']) for k,v in Mb.items() for s in v]; badm=[x for x in ovm if x[2]!=0]
p(f"probe∩eval overlap: {'ALL ZERO ✓' if not badm else 'NONZERO! '+str(badm)}  (checked {len(ovm)} mob-runs)")
fmeanm={k:np.mean([Mb[k][s]['integ'] for s in Mb[k]]) for k in fixed_keys if Mb[k]}
best_fixed_m=max(fmeanm,key=fmeanm.get)
p(f"best fixed(mob) = {best_fixed_m} ({fmeanm[best_fixed_m]*100:.3f}%)")
for name, other in [("DriftGate − "+best_fixed_m, Mb[best_fixed_m]),
                    ("DriftGate − entropy", Mb["entropy"])]:
    seeds,d=paired(Mb["DriftGate(TV)"], other); lo,hi=ci95(d)
    p(f"  {name:30s} {d.mean()*100:+.3f}pp  CI95[{lo*100:+.2f},{hi*100:+.2f}]  "
      f"{int((d>0).sum())}/{len(d)} seeds+  per-seed={[round(x*100,2) for x in d]}")

# ---- same-pool vs disjoint absolute shift (secondary sanity)
# same-pool pilot runs may use a different eval-round grid; compare on the
# INTERSECTION of eval rounds so the abs-shift is measured on matched rounds.
def _acc_by_round(path):
    h = load(path); return {e["round"]: e["acc_total"] for e in h["eval"]}
def integ_common(pa, pb):
    a = _acc_by_round(pa); b = _acc_by_round(pb)
    common = sorted(set(a) & set(b))
    return np.mean([a[r] for r in common]), np.mean([b[r] for r in common]), len(common)
p("\n[sanity] disjoint − same-pool absolute integrated (matched eval rounds, paired by seed)")
SP = {
 "DriftGate(TV)": fam(["runs/phaseC_signals/dvsig_tv_A_s*.json","runs/phaseS_supp/b1_adaptive_A_s*.json"]),
 "entropy":       fam(["runs/downstream_pilot/sc_ent_A_s*.json"]),
 "a1(adaptλ,Λ=.5)":fam(["runs/phaseS_supp/a1_A_s*.json"]),
 "fixed λ0.4":    fam(["runs/phaseB_grid/fx40_A_s*.json","runs/phaseS_supp/b1_fx40_A_s*.json"]),
 "fixed λ0.5":    fam(["runs/phaseB_grid/fx50_A_s*.json","runs/phaseS_supp/b1_fx50_A_s*.json"]),
}
for k in ["DriftGate(TV)","entropy","a1(adaptλ,Λ=.5)","fixed λ0.4","fixed λ0.5"]:
    if k not in A or not SP.get(k): continue
    seeds = sorted(set(A[k]) & set(SP[k])); diffs=[]; ncr=None
    for s in seeds:
        di, sp, ncr = integ_common(A[k][s]["path"], SP[k][s]["path"]); diffs.append(di-sp)
    d = np.array(diffs)
    p(f"  {k:18s} disjoint−samepool = {d.mean()*100:+.3f}pp (|max|={np.abs(d).max()*100:.2f}pp, "
      f"n={len(d)} seeds {seeds}, {ncr} matched rounds)")

# ---- Task 1 verdict
seeds,dbf = paired(A["DriftGate(TV)"], A[best_fixed])
_,dent = paired(A["DriftGate(TV)"], A["entropy"])
cond1 = dbf.mean() > 0
cond2 = (dbf > 0).sum() >= 4
cond3 = dent.mean() > 0
p("\n[TASK 1 VERDICT §4.7]")
p(f"  (1) DriftGate−best_fixed mean > 0 : {cond1}  ({dbf.mean()*100:+.3f}pp)")
p(f"  (2) ≥4/5 seeds positive           : {cond2}  ({int((dbf>0).sum())}/5)")
p(f"  (3) TV−entropy mean > 0           : {cond3}  ({dent.mean()*100:+.3f}pp)")
p(f"  >>> CORE {'HOLDS under disjoint pools' if (cond1 and cond2 and cond3) else 'DOES NOT fully hold — report as-is'}")

# ============================================================ TASK 2 (partial)
p("\n"+"="*70); p("TASK 2 — direct server non-Main signals (PARTIAL: done runs only)"); p("="*70)
T2="runs/phaseT2_signal"
def merge_disjoint(name_t1, name_t2):
    # DriftGate/entropy references live in phaseT1_disjoint; soft/hard in phaseT2_signal
    return fam([name_t1]), fam([name_t2])

p("\n[Schedule A] downstream integrated accuracy — disjoint pools")
armsA = {
 "TV (DriftGate)": fam([f"{D}/t1_A_dg_s*.json"]),
 "soft(non-Main)": fam([f"{T2}/t2_A_soft_s*.json"]),
 "hard(non-Main)": fam([f"{T2}/t2_A_hard_s*.json"]),
 "entropy":        fam([f"{D}/t1_A_ent_s*.json"]),
 "fixed λ0.4":     fam([f"{D}/t1_A_fx40_s*.json"]),
}
p(f"{'signal':18s} " + " ".join(f"s{s}" for s in range(5)) + "   mean  n")
for k,v in armsA.items():
    row=" ".join(f"{v[s]['integ']*100:5.2f}" if s in v else "  -- " for s in range(5))
    mean=np.mean([v[s]['integ'] for s in v])*100 if v else float('nan')
    p(f"{k:18s} {row}   {mean:5.2f}  {len(v)}")
for direct in ["soft(non-Main)","hard(non-Main)"]:
    if not armsA[direct]: continue
    seeds,d = paired(armsA["TV (DriftGate)"], armsA[direct])
    lo,hi=ci95(d)
    p(f"  TV − {direct:16s} {d.mean()*100:+.3f}pp CI95[{lo*100:+.2f},{hi*100:+.2f}] "
      f"{int((d>0).sum())}/{len(d)} seeds(TV higher)  per-seed={[round(x*100,2) for x in d]}")

p("\n[mobility-med] downstream integrated accuracy — disjoint pools")
armsM = {
 "TV (DriftGate)": fam([f"{D}/t1_mob_dg_s*.json"]),
 "soft(non-Main)": fam([f"{T2}/t2_mob_soft_s*.json"]),
 "hard(non-Main)": fam([f"{T2}/t2_mob_hard_s*.json"]),
 "entropy":        fam([f"{D}/t1_mob_ent_s*.json"]),
}
p(f"{'signal':18s} " + " ".join(f"s{s}" for s in range(3)) + "   mean  n")
for k,v in armsM.items():
    row=" ".join(f"{v[s]['integ']*100:5.2f}" if s in v else "  -- " for s in range(3))
    mean=np.mean([v[s]['integ'] for s in v])*100 if v else float('nan')
    p(f"{k:18s} {row}   {mean:5.2f}  {len(v)}")
for direct in ["soft(non-Main)","hard(non-Main)"]:
    if not armsM[direct]: continue
    seeds,d=paired(armsM["TV (DriftGate)"], armsM[direct])
    p(f"  TV − {direct:16s} {d.mean()*100:+.3f}pp {int((d>0).sum())}/{len(d)} seeds(TV higher) per-seed={[round(x*100,2) for x in d]}")

# SVHN status
sv = {a: len(glob.glob(str(JR/f"{T2}/t2_svhn_{a}_s*.json"))) for a in ["tv","soft","hard","ent","fx20"]}
p(f"\n[SVHN temporal] status: {sv}  (target 5 each; signal-quality AUROC/Spearman + downstream deferred until SVHN completes)")

Path(JR/"tables").mkdir(exist_ok=True)
open(JR/"tables/task12_partial_analysis.txt","w").write("\n".join(L))
print("\n[written] tables/task12_partial_analysis.txt")
