"""Task 3 — FINAL tables + consistency tests, from ONE script, RAW JSON only.
Writes to tables/final_closure/ (never touches existing table files).
Canonical metric: integrated = mean(acc_total over that run's eval rounds).
Runs 7 automated consistency asserts; exits non-zero if any fails.
"""
import json, glob, re, csv, sys
import numpy as np
from pathlib import Path
from scipy import stats

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT = JR/"tables/final_closure"; OUT.mkdir(parents=True, exist_ok=True)
D="runs/phaseT1_disjoint"; T="runs/phaseT2_signal"

def load(p): return json.load(open(p))
def integrated(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def erounds(h): return tuple(e["round"] for e in h["eval"])
def fam(pat):
    out={}
    for f in sorted(glob.glob(str(JR/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: out[int(m.group(1))]=f
    return out
def integ_fam(pat): return {s:integrated(load(f)) for s,f in fam(pat).items()}
def rounds_fam(pat): return {s:erounds(load(f)) for s,f in fam(pat).items()}
def overlap_fam(pat): return {s:load(f).get("probe_eval_overlap") for s,f in fam(pat).items()}
def paired(a,b):
    seeds=sorted(set(a)&set(b)); return seeds, np.array([a[s]-b[s] for s in seeds])
def wcsv(name, header, rows):
    with open(OUT/name,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# ---------------- gather raw ----------------
A = {k:integ_fam(v) for k,v in {
    "DriftGate_TV":f"{D}/t1_A_dg_s*.json","entropy":f"{D}/t1_A_ent_s*.json",
    "a1_adaptLam0.5":f"{D}/t1_A_a1_s*.json","fixed_l0.3":f"{D}/t1_A_fx30_s*.json",
    "fixed_l0.4":f"{D}/t1_A_fx40_s*.json","fixed_l0.5":f"{D}/t1_A_fx50_s*.json"}.items()}
M = {k:integ_fam(v) for k,v in {
    "DriftGate_TV":f"{D}/t1_mob_dg_s*.json","entropy":f"{D}/t1_mob_ent_s*.json",
    "a1_adaptLam0.5":f"{D}/t1_mob_a1_s*.json","fixed_l0.3":f"{D}/t1_mob_fx30_s*.json",
    "fixed_l0.4":f"{D}/t1_mob_fx40_s*.json","fixed_l0.5":f"{D}/t1_mob_fx50_s*.json"}.items()}
SIG = {  # per-dataset downstream, disjoint
 "A":{k:integ_fam(v) for k,v in {"tv":f"{D}/t1_A_dg_s*.json","soft":f"{T}/t2_A_soft_s*.json",
      "hard":f"{T}/t2_A_hard_s*.json","entropy":f"{D}/t1_A_ent_s*.json","fixed_l0.4":f"{D}/t1_A_fx40_s*.json"}.items()},
 "mob":{k:integ_fam(v) for k,v in {"tv":f"{D}/t1_mob_dg_s*.json","soft":f"{T}/t2_mob_soft_s*.json",
      "hard":f"{T}/t2_mob_hard_s*.json","entropy":f"{D}/t1_mob_ent_s*.json","fixed_l0.4":f"{D}/t1_mob_fx40_s*.json"}.items()},
 "svhn":{k:integ_fam(v) for k,v in {"tv":f"{T}/t2_svhn_tv_s*.json","soft":f"{T}/t2_svhn_soft_s*.json",
      "hard":f"{T}/t2_svhn_hard_s*.json","entropy":f"{T}/t2_svhn_ent_s*.json","fixed_l0.2":f"{T}/t2_svhn_fx20_s*.json"}.items()},
}
best_fixed_A = max(["fixed_l0.3","fixed_l0.4","fixed_l0.5"], key=lambda k: np.mean(list(A[k].values())))
best_fixed_M = max(["fixed_l0.3","fixed_l0.4","fixed_l0.5"], key=lambda k: np.mean(list(M[k].values())))

# ---------------- signal quality on SVHN (neutral entropy backbone) ----------------
def roc_auc(label, score):
    label=np.asarray(label); score=np.asarray(score)
    order=np.argsort(score,kind="mergesort"); ranks=np.empty(len(score),float)
    sc=score[order]; i=0; r=1
    while i<len(sc):
        j=i
        while j+1<len(sc) and sc[j+1]==sc[i]: j+=1
        ranks[order[i:j+1]]=(r+r+(j-i))/2.0; r+=(j-i+1); i=j+1
    npos=int(label.sum()); nneg=len(label)-npos
    return float("nan") if npos==0 or nneg==0 else (ranks[label==1].sum()-npos*(npos+1)/2)/(npos*nneg)
NM={"tv":"tv_dist","soft":"server_nonmain_soft","hard":"server_nonmain_hard","entropy":"ent_client"}
spear={k:[] for k in NM}; auroc={k:[] for k in NM}
for s in range(5):
    f=JR/f"{T}/t2_svhn_ent_s{s}.json"
    if not f.exists(): continue
    h=load(f); spe=h["signals_per_es"]; rho=np.array(h["rho_trace"]); lab=(rho>=0.4).astype(int)
    for k,nm in NM.items():
        ser=np.array([np.mean([float(v) for v in d.values()]) for d in spe[nm]])
        spear[k].append(stats.spearmanr(ser,rho).statistic)
        if lab.min()!=lab.max(): auroc[k].append(roc_auc(lab,ser))

# ---------------- write CSVs ----------------
# 1 scheduleA (disjoint) per-seed
rows=[[k]+[f"{A[k].get(s,'')}" for s in range(5)]+[np.mean(list(A[k].values()))] for k in A]
wcsv("final_scheduleA_results.csv",["arm","s0","s1","s2","s3","s4","mean"],rows)
# 2 mobility per-seed
rows=[[k]+[f"{M[k].get(s,'')}" for s in range(3)]+[np.mean(list(M[k].values()))] for k in M]
wcsv("final_mobility_results.csv",["arm","s0","s1","s2","mean"],rows)
# 3 fixed grid
rows=[["A",k,np.mean(list(A[k].values())),len(A[k])] for k in ["fixed_l0.3","fixed_l0.4","fixed_l0.5"]]+\
     [["mob",k,np.mean(list(M[k].values())),len(M[k])] for k in ["fixed_l0.3","fixed_l0.4","fixed_l0.5"]]
wcsv("final_fixed_grid_results.csv",["block","arm","mean_integrated","n_seeds"],rows)
# 4 disjoint protocol (paired diffs + overlap)
def diffrow(block, a, b, lbl):
    seeds,d=paired(a,b); se=stats.sem(d) if len(d)>1 else 0; h=se*stats.t.ppf(0.975,len(d)-1) if len(d)>1 else 0
    return [block,lbl,f"{d.mean()*100:.4f}",f"{(d.mean()-h)*100:.4f}",f"{(d.mean()+h)*100:.4f}",int((d>0).sum()),len(d)]
rows=[diffrow("A_disjoint",A["DriftGate_TV"],A[best_fixed_A],f"DriftGate-{best_fixed_A}"),
      diffrow("A_disjoint",A["DriftGate_TV"],A["entropy"],"DriftGate-entropy"),
      diffrow("A_disjoint",A["DriftGate_TV"],A["a1_adaptLam0.5"],"DriftGate-a1(Lam=.5)"),
      diffrow("mob_disjoint",M["DriftGate_TV"],M[best_fixed_M],f"DriftGate-{best_fixed_M}"),
      diffrow("mob_disjoint",M["DriftGate_TV"],M["entropy"],"DriftGate-entropy")]
wcsv("final_disjoint_protocol_results.csv",["block","comparison","mean_pp","ci_lo_pp","ci_hi_pp","n_pos","n_seeds"],rows)
# 5 signal comparison (downstream, all datasets) — NEW name to not clobber existing table
rows=[]
for blk in ["A","mob","svhn"]:
    for k,v in SIG[blk].items():
        rows.append([blk,k,f"{np.mean(list(v.values()))*100:.4f}",len(v)])
    for direct in ["soft","hard"]:
        seeds,d=paired(SIG[blk]["tv"],SIG[blk][direct]); _,pt=stats.ttest_rel(
            [SIG[blk]["tv"][s] for s in seeds],[SIG[blk][direct][s] for s in seeds])
        rows.append([blk,f"TV-{direct}",f"{d.mean()*100:.4f}",f"tpair_p={pt:.4f};TVhigher={int((d>0).sum())}/{len(d)}"])
wcsv("final_signal_comparison.csv",["block","arm_or_diff","value","note"],rows)
# 6 svhn signal quality
rows=[[k,f"{np.mean(spear[k]):.4f}",f"{np.mean(auroc[k]):.4f}",len(spear[k])] for k in ["tv","soft","hard","entropy"]]
wcsv("final_svhn_signal_results.csv",["signal","spearman_vs_rho","auroc_drift_active","n_seeds"],rows)

# ---------------- headline json ----------------
canon = load(JR/"tables/canonical_headline_metrics.json")
headline = {
 "samepool_scheduleA_adaptive_mean_5seed": canon["scheduleA_adaptive_mean_5seed"],
 "samepool_best_fixed_key": canon["scheduleA_best_fixed_key"],
 "samepool_adaptive_minus_bestfixed_pp": canon["scheduleA_adaptive_minus_bestfixed_5seed_pp"],
 "disjoint_A_DriftGate_mean": float(np.mean(list(A["DriftGate_TV"].values()))),
 "disjoint_A_DriftGate_per_seed": {s:A["DriftGate_TV"][s] for s in sorted(A["DriftGate_TV"])},
 "disjoint_A_best_fixed_key": best_fixed_A,
 "disjoint_A_DriftGate_minus_bestfixed_pp": float(paired(A["DriftGate_TV"],A[best_fixed_A])[1].mean()*100),
 "disjoint_A_DriftGate_minus_entropy_pp": float(paired(A["DriftGate_TV"],A["entropy"])[1].mean()*100),
 "disjoint_A_DriftGate_minus_a1_pp": float(paired(A["DriftGate_TV"],A["a1_adaptLam0.5"])[1].mean()*100),
 "disjoint_mob_DriftGate_minus_bestfixed_pp": float(paired(M["DriftGate_TV"],M[best_fixed_M])[1].mean()*100),
 "svhn_TV_minus_soft_pp": float(paired(SIG["svhn"]["tv"],SIG["svhn"]["soft"])[1].mean()*100),
 "svhn_TV_minus_hard_pp": float(paired(SIG["svhn"]["tv"],SIG["svhn"]["hard"])[1].mean()*100),
 "svhn_signal_spearman": {k:float(np.mean(spear[k])) for k in NM},
 "svhn_signal_auroc": {k:float(np.mean(auroc[k])) for k in NM},
}
json.dump(headline, open(OUT/"final_headline_numbers.json","w"), indent=1)

# ================= CONSISTENCY TESTS =================
fails=[]
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond: fails.append(name)

# T1: same-pool adaptive per-seed recompute == task0 canonical
sp_adapt = integ_fam("runs/phaseC_signals/dvsig_tv_A_s*.json")
sp_adapt.update({s:v for s,v in integ_fam("runs/phaseS_supp/b1_adaptive_A_s*.json").items() if s not in sp_adapt})
maxd=max(abs(sp_adapt[int(s)]-canon["scheduleA_adaptive_per_seed"][s]) for s in canon["scheduleA_adaptive_per_seed"])
check("T1 same-pool adaptive recompute == task0 canonical", maxd<1e-9, f"max|Δ|={maxd:.2e}")

# T2: best_fixed key is the argmax-mean arm and in the grid
check("T2 disjoint best_fixed is argmax-mean fixed arm", best_fixed_A in ("fixed_l0.3","fixed_l0.4","fixed_l0.5"), best_fixed_A)

# T3: Task-1 A verdict — DriftGate>best_fixed, 5/5, and >entropy
_,dbf=paired(A["DriftGate_TV"],A[best_fixed_A]); _,den=paired(A["DriftGate_TV"],A["entropy"])
check("T3 DriftGate-best_fixed>0 & 5/5 seeds (A,disjoint)", dbf.mean()>0 and (dbf>0).sum()==5,
      f"{dbf.mean()*100:+.3f}pp {int((dbf>0).sum())}/5")
check("T4 TV-entropy>0 & 5/5 seeds (A,disjoint)", den.mean()>0 and (den>0).sum()==5,
      f"{den.mean()*100:+.3f}pp {int((den>0).sum())}/5")

# T5: ALL disjoint + signal runs have overlap 0
allpats=[f"{D}/t1_A_*_s*.json",f"{D}/t1_mob_*_s*.json",f"{T}/t2_A_*_s*.json",f"{T}/t2_mob_*_s*.json",
         f"{T}/t2_svhn_tv_s*.json",f"{T}/t2_svhn_soft_s*.json",f"{T}/t2_svhn_hard_s*.json",f"{T}/t2_svhn_ent_s*.json"]
ovs=[load(f).get("probe_eval_overlap") for pat in allpats for f in glob.glob(str(JR/pat))]
check("T5 probe∩eval overlap==0 for all disjoint/signal runs", all(o==0 for o in ovs), f"n={len(ovs)}")

# T6: canonical eval-round grid per schedule; no seed-count mixing in paired diffs
gridA=set(tuple(v) for v in rounds_fam(f"{D}/t1_A_dg_s*.json").values())
gridM=set(tuple(v) for v in rounds_fam(f"{D}/t1_mob_dg_s*.json").values())
gridS=set(tuple(v) for v in rounds_fam(f"{T}/t2_svhn_tv_s*.json").values())
ok_grid = (len(gridA)==1 and len(list(gridA)[0])==16 and len(gridM)==1 and len(list(gridM)[0])==13
           and len(gridS)==1 and len(list(gridS)[0])==16)
# equal seed sets for every headline paired diff
eq_seeds = (set(A["DriftGate_TV"])==set(A[best_fixed_A])==set(A["entropy"])==set(A["a1_adaptLam0.5"])
            and set(SIG["svhn"]["tv"])==set(SIG["svhn"]["soft"])==set(SIG["svhn"]["hard"]))
check("T6 canonical eval grids (A/SVHN=16, mob=13) & no seed-count mixing", ok_grid and eq_seeds,
      f"grids ok={ok_grid} seeds_aligned={eq_seeds}")

# T7: disjoint DriftGate mean within 0.5pp of same-pool canonical (protocol validity) +
#     headline internal consistency (mean == mean(per_seed))
dj_mean=np.mean(list(A["DriftGate_TV"].values()))
prot_ok=abs(dj_mean-canon["scheduleA_adaptive_mean_5seed"])*100 < 0.5
internal_ok=abs(headline["disjoint_A_DriftGate_mean"]-np.mean(list(headline["disjoint_A_DriftGate_per_seed"].values())))<1e-12
check("T7 disjoint≈same-pool (<0.5pp) & headline internally consistent", prot_ok and internal_ok,
      f"Δprotocol={abs(dj_mean-canon['scheduleA_adaptive_mean_5seed'])*100:.3f}pp")

print(f"\n{'ALL 7 CONSISTENCY TESTS PASSED' if not fails else 'FAILURES: '+', '.join(fails)}")
print(f"tables written to {OUT}")
sys.exit(1 if fails else 0)
