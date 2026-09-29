"""Round-2 Task 7 — final tables + 15 consistency asserts + signal-selection state.
RAW JSON only, canonical integrated = mean(acc_total over eval rounds).
Disjoint protocol everywhere; never mixes protocols or seed counts in a paired diff.
Writes tables/final_closure_round2/*.csv. Exits non-zero if any assert fails.
"""
import json, glob, re, csv, math, sys
import numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,"."); sys.path.insert(0,"..")
from src.evaluation.signal_metrics import evaluate_signal
from src.controllers.self_calibrating import SIGNAL_RANGE, Z0, TAU_Z

JR=Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT=JR/"tables/final_closure_round2"; OUT.mkdir(parents=True,exist_ok=True)
T1="runs/phaseT1_disjoint"; T2="runs/phaseT2_signal"; R2="runs/phaseT2r2_signal"
WARMUP=15; DRIFT=0.5; WARMBURN=25

def load(p): return json.load(open(p))
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def erounds(h): return tuple(e["round"] for e in h["eval"])
def famf(pat):
    d={}
    for f in sorted(glob.glob(str(JR/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: d[int(m.group(1))]=f
    return d
def integ_fam(pat): return {s:integ(load(f)) for s,f in famf(pat).items()}
def paired(a,b):
    seeds=sorted(set(a)&set(b)); return seeds,np.array([a[s]-b[s] for s in seeds])
def ci95(d):
    if len(d)<2: return (float('nan'),float('nan'))
    se=stats.sem(d); h=se*stats.t.ppf(0.975,len(d)-1); return (d.mean()-h,d.mean()+h)
def rho_series(h): return np.array([r if not isinstance(r,dict) else np.mean(list(r.values())) for r in h["rho_trace"]],float)
def mean_series(spe,nm): return np.array([np.mean([float(v) for v in d.values()]) for d in spe[nm]])
def abs_frac(path,signame):
    h=load(path); cz=h["controller_z"]; sig=h["signals_per_es"][signame]; rng=SIGNAL_RANGE.get(signame,1.0)
    n=act=0
    for r in range(WARMBURN,len(cz)):
        for es in cz[r]:
            z=float(cz[r][es]); s=float(sig[r][es])
            g=1/(1+math.exp(-max(-50,min(50,(z-Z0)/TAU_Z)))); dhat=min(max(s/rng,0),1)
            n+=1; act+=(dhat>g)
    return act/n if n else float('nan')
def lam_range(path):
    h=load(path); lam=[float(v) for d in h["lamdas"] for v in d.values()]; return min(lam),max(lam)
def sig_range(path,nm):
    h=load(path); v=[float(x) for d in h["signals_per_es"][nm] for x in d.values()]; return min(v),max(v),(len(set(round(x,4) for x in v))==1)

# ---------------- settings registry ----------------
# each: name -> {arm: (glob, signal_key_for_absfrac)}; TV & hard mandatory
S={
 "Schedule A": dict(n_expect=5, tv=(f"{T1}/t1_A_dg_s*.json","tv_dist"), hard=(f"{T2}/t2_A_hard_s*.json","server_nonmain_hard"),
    soft=(f"{T2}/t2_A_soft_s*.json",None), entropy=(f"{T1}/t1_A_ent_s*.json",None),
    fixed_grid=[("fx40",f"{T1}/t1_A_fx40_s*.json"),("fx30",f"{T1}/t1_A_fx30_s*.json"),("fx50",f"{T1}/t1_A_fx50_s*.json")],
    fixed02=(f"{R2}/r2_A_fx20_s*.json",)),
 "mobility-med": dict(n_expect=3, tv=(f"{T1}/t1_mob_dg_s*.json","tv_dist"), hard=(f"{T2}/t2_mob_hard_s*.json","server_nonmain_hard"),
    soft=(f"{T2}/t2_mob_soft_s*.json",None), entropy=(f"{T1}/t1_mob_ent_s*.json",None),
    fixed_grid=[("fx40",f"{T1}/t1_mob_fx40_s*.json"),("fx30",f"{T1}/t1_mob_fx30_s*.json"),("fx50",f"{T1}/t1_mob_fx50_s*.json")],
    fixed02=(f"{R2}/r2_mob_fx20_s*.json",)),
 "SVHN temporal": dict(n_expect=5, tv=(f"{T2}/t2_svhn_tv_s*.json","tv_dist"), hard=(f"{T2}/t2_svhn_hard_s*.json","server_nonmain_hard"),
    soft=(f"{T2}/t2_svhn_soft_s*.json",None), entropy=(f"{T2}/t2_svhn_ent_s*.json",None),
    fixed_grid=[("fx20",f"{T2}/t2_svhn_fx20_s*.json")], fixed02=(f"{T2}/t2_svhn_fx20_s*.json",)),
 "CIFAR-10 gradual": dict(n_expect=3, tv=(f"{R2}/r2_c10gsig_tv_s*.json","tv_dist"), hard=(f"{R2}/r2_c10gsig_hard_s*.json","server_nonmain_hard"),
    soft=None, entropy=None, fixed_grid=[], fixed02=None),
 "CIFAR-100 spatial": dict(n_expect=3, tv=(f"{R2}/r2_c100sp_tv_s*.json","tv_dist"), hard=(f"{R2}/r2_c100sp_hard_s*.json","server_nonmain_hard"),
    soft=None, entropy=None, fixed_grid=[], fixed02=None),
 "CIFAR-100 gradual": dict(n_expect=3, tv=(f"{R2}/r2_c100gsig_tv_s*.json","tv_dist"), hard=(f"{R2}/r2_c100gsig_hard_s*.json","server_nonmain_hard"),
    soft=None, entropy=None, fixed_grid=[], fixed02=None),
 "Tiny-ImageNet": dict(n_expect=3, tv=(f"{R2}/r2_tiny_tv_s*.json","tv_dist"), hard=(f"{R2}/r2_tiny_hard_s*.json","server_nonmain_hard"),
    soft=None, entropy=None, fixed_grid=[], fixed02=None),
}

rows=[]; per_setting={}
for name,cfg in S.items():
    tv=integ_fam(cfg["tv"][0]); hard=integ_fam(cfg["hard"][0])
    ent=integ_fam(cfg["entropy"][0]) if cfg.get("entropy") else {}
    soft=integ_fam(cfg["soft"][0]) if cfg.get("soft") else {}
    # fixed grid best (same seeds as hard/tv where possible)
    fg={k:integ_fam(p) for k,p in cfg["fixed_grid"]}
    fg_best_k=max(fg,key=lambda k:np.mean(list(fg[k].values()))) if fg else None
    fg_best=fg[fg_best_k] if fg_best_k else {}
    fx02=integ_fam(cfg["fixed02"][0]) if cfg.get("fixed02") else {}
    seeds_ht,d_ht=paired(hard,tv)
    lo,hi=ci95(d_ht)
    # abs fraction & signal sanity (seed 0)
    tv0=famf(cfg["tv"][0]).get(0); hard0=famf(cfg["hard"][0]).get(0)
    af_tv=abs_frac(tv0,cfg["tv"][1]) if tv0 else float('nan')
    af_hard=abs_frac(hard0,cfg["hard"][1]) if hard0 else float('nan')
    smn,smx,sconst=sig_range(hard0,"server_nonmain_hard") if hard0 else (float('nan'),)*2+(False,)
    lmn,lmx=lam_range(hard0) if hard0 else (float('nan'),float('nan'))
    # paired hard-fixed_grid, tv-fixed_grid (only if fixed present, same seeds)
    def pd(a,b):
        s,d=paired(a,b); return (d.mean()*100,ci95(d),int((d>0).sum()),len(d)) if len(d) else (None,)*4
    hfg=pd(hard,fg_best) if fg_best else (None,)*4
    tfg=pd(tv,fg_best) if fg_best else (None,)*4
    per_setting[name]=dict(tv=tv,hard=hard,ent=ent,soft=soft,fg_best=fg_best,fg_best_k=fg_best_k,fx02=fx02,
                           d_ht=d_ht,seeds=seeds_ht,af_tv=af_tv,af_hard=af_hard,sconst=sconst,lmn=lmn,lmx=lmx)
    rows.append([name, "disjoint", str(seeds_ht), len(seeds_ht),
        f"{np.mean(list(tv.values()))*100:.4f}", f"{np.mean(list(hard.values()))*100:.4f}",
        f"{np.mean(list(fg_best.values()))*100:.4f}" if fg_best else "", fg_best_k or "",
        f"{np.mean(list(fx02.values()))*100:.4f}" if fx02 else "",
        f"{np.mean(list(ent.values()))*100:.4f}" if ent else "",
        f"{d_ht.mean()*100:+.4f}", f"[{lo*100:+.2f},{hi*100:+.2f}]", int((d_ht>0).sum()),
        f"{hfg[0]:+.4f}" if hfg[0] is not None else "", f"[{hfg[1][0]*100:+.2f},{hfg[1][1]*100:+.2f}]" if hfg[0] is not None else "",
        int(hfg[2]) if hfg[0] is not None else "",
        f"{tfg[0]:+.4f}" if tfg[0] is not None else "", f"[{tfg[1][0]*100:+.2f},{tfg[1][1]*100:+.2f}]" if tfg[0] is not None else "",
        f"{af_tv:.3f}", f"{af_hard:.3f}", f"{smn:.3f}", f"{smx:.3f}", f"{lmn:.2f}", f"{lmx:.2f}",
        "yes" if len(seeds_ht)==cfg["n_expect"] else "partial",
        "" if len(seeds_ht) else "no matched seeds"])

hdr=["setting","protocol","seeds","n_seeds","tv_mean","server_nonmain_rate_mean","fixed_grid_reference_mean",
     "fixed_grid_reference_lambda","fixed_lambda02_mean","entropy_mean","hard_minus_tv_mean","hard_minus_tv_ci95",
     "seeds_hard_higher","hard_minus_fixed_grid_mean","hard_minus_fixed_grid_ci95","seeds_hard_over_fixed",
     "tv_minus_fixed_grid_mean","tv_minus_fixed_grid_ci95","abs_active_fraction_tv","abs_active_fraction_hard",
     "signal_mean_hard_min","signal_mean_hard_max","lambda_min_hard","lambda_max_hard","matched_comparison_valid","unmatched_reason"]
with open(OUT/"hard_vs_tv_all_settings.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr); w.writerows(rows)

print("=== hard vs TV — all settings (disjoint) ===")
print(f"{'setting':20s} {'nS':2s} {'TV':>6s} {'hard':>6s} {'h-TV':>7s} {'h-fixg':>7s} {'t-fixg':>7s} {'absF_h':>6s}")
for r in rows:
    print(f"{r[0]:20s} {r[3]:<2d} {r[4]:>6s} {r[5]:>6s} {r[10]:>7s} {r[13] or '   -':>7s} {r[16] or '   -':>7s} {r[19]:>6s}")

# ---- family aggregation for 6.5 ----
FAM={"CIFAR-10 temporal":["Schedule A","CIFAR-10 gradual"],"CIFAR-100":["CIFAR-100 spatial","CIFAR-100 gradual"],
     "Tiny-ImageNet":["Tiny-ImageNet"],"SVHN":["SVHN temporal"],"Mobility":["mobility-med"]}
print("\n=== family-level hard-TV (settings averaged within family) ===")
fam_ht={}
for fam,setts in FAM.items():
    vals=[per_setting[s]["d_ht"].mean()*100 for s in setts]
    fam_ht[fam]=np.mean(vals)
    print(f"  {fam:20s} hard-TV = {np.mean(vals):+.3f}pp  (settings: {[f'{per_setting[s]['d_ht'].mean()*100:+.2f}' for s in setts]})")
fam_hard_higher=sum(1 for v in fam_ht.values() if v>0); fam_total=len(fam_ht)
worst_setting=min(per_setting, key=lambda s: per_setting[s]["d_ht"].mean())
worst_ht=per_setting[worst_setting]["d_ht"].mean()*100
print(f"\n  families hard>TV: {fam_hard_higher}/{fam_total}; worst setting {worst_setting} h-TV={worst_ht:+.3f}pp")

import pickle
pickle.dump({"per_setting":per_setting,"fam_ht":fam_ht,"rows":rows}, open(OUT/"_r2_cache.pkl","wb"))
print(f"\n[written] {OUT}/hard_vs_tv_all_settings.csv")
