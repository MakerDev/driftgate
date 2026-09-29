"""DriftGate TMC export — tidy data extraction from RAW run JSON.
Produces every CSV the figures and the Korean numbers doc read.
Canonical integrated = mean(acc_total over that run's eval rounds).
Never mixes protocols/seed-counts in a paired diff; labels same-pool vs disjoint.
Run: python build_data.py   (idempotent; writes ../data/*.csv)
"""
import json, glob, re, csv, math, sys
import numpy as np
from pathlib import Path
from scipy import stats

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
DATA = Path(__file__).resolve().parent.parent/"data"; DATA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(JR)); sys.path.insert(0,str(JR.parent))

def load(p): return json.load(open(p))
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def famf(pat):
    d={}
    for f in sorted(glob.glob(str(JR/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: d[int(m.group(1))]=f
    return d
def rho_series(h): return np.array([r if not isinstance(r,dict) else np.mean(list(r.values())) for r in h["rho_trace"]],float)
def sig_series(h,nm): return np.array([np.mean([float(v) for v in d.values()]) for d in h["signals_per_es"][nm]])
def lam_series(h): return np.array([np.mean([float(v) for v in d.values()]) for d in h["lamdas"]])
def paired(a,b):
    s=sorted(set(a)&set(b)); return s,np.array([a[x]-b[x] for x in s])
def ci95(d):
    if len(d)<2: return (float('nan'),float('nan'))
    se=stats.sem(d); h=se*stats.t.ppf(0.975,len(d)-1); return (d.mean()-h,d.mean()+h)
def wcsv(nm,hdr,rows):
    with open(DATA/nm,"w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"  [{nm}] {len(rows)} rows")

LNC=math.log(10)  # CIFAR-10/SVHN class count for entropy normalization in F1 (both 10-class)

# ================= F1: signal dynamics (passive fixed-weight rec_* backbone) =================
print("F1 signal dynamics (rec_* passive fixed-lambda backbone, 3 seeds)")
rows=[]
for sched,tag in [("A","Schedule A"),("abrupt","post-convergence abrupt")]:
    fs=famf(f"runs/signal_benchmark/rec_{sched}_s*.json")
    if not fs: continue
    T=None; tv=[]; en=[]; rh=[]
    for s,f in fs.items():
        h=load(f); T=len(h["rho_trace"])
        tv.append(sig_series(h,"tv_dist")); en.append(sig_series(h,"ent_client")/LNC); rh.append(rho_series(h))
    tv=np.array(tv); en=np.array(en); rh=np.array(rh)
    for r in range(T):
        rows.append([tag,r+1,f"{rh[:,r].mean():.4f}",
                     f"{tv[:,r].mean():.5f}",f"{tv[:,r].std(ddof=1):.5f}",
                     f"{en[:,r].mean():.5f}",f"{en[:,r].std(ddof=1):.5f}",len(fs)])
wcsv("f1_signal_dynamics.csv",["schedule","round","rho","tv_mean","tv_sd","ent_norm_mean","ent_norm_sd","n_seeds"],rows)

# ================= F3: main trajectories (disjoint) =================
print("F3 trajectories (disjoint t1_* runs)")
T3={"Schedule A":dict(TV="runs/phaseT1_disjoint/t1_A_dg_s*.json",entropy="runs/phaseT1_disjoint/t1_A_ent_s*.json",
        **{"fixed l0.4":"runs/phaseT1_disjoint/t1_A_fx40_s*.json"}),
    "mobility-med":dict(TV="runs/phaseT1_disjoint/t1_mob_dg_s*.json",entropy="runs/phaseT1_disjoint/t1_mob_ent_s*.json",
        **{"fixed l0.4":"runs/phaseT1_disjoint/t1_mob_fx40_s*.json"})}
lam_rows=[]; acc_rows=[]; rho_rows=[]
for setting,arms in T3.items():
    # rho from TV run (deterministic; same across arms)
    any_h=load(famf(arms["TV"])[0]); rho=rho_series(any_h)
    for r in range(len(rho)): rho_rows.append([setting,r+1,f"{rho[r]:.4f}"])
    for method,pat in arms.items():
        fs=famf(pat)
        # lambda per round (seed-avg)
        lam=np.array([lam_series(load(f)) for f in fs.values()])
        for r in range(lam.shape[1]):
            lam_rows.append([setting,method,r+1,f"{lam[:,r].mean():.5f}",f"{lam[:,r].std(ddof=1):.5f}",len(fs)])
        # acc per eval round (seed-avg + sd)
        hs=[load(f) for f in fs.values()]
        ev_rounds=[e["round"] for e in hs[0]["eval"]]
        for i,rr in enumerate(ev_rounds):
            accs=[h["eval"][i]["acc_total"] for h in hs]
            acc_rows.append([setting,method,rr,f"{np.mean(accs)*100:.4f}",f"{np.std(accs,ddof=1)*100:.4f}",len(fs)])
wcsv("f3_traj_lambda.csv",["setting","method","round","lambda_mean","lambda_sd","n_seeds"],lam_rows)
wcsv("f3_traj_acc.csv",["setting","method","eval_round","acc_mean_pct","acc_sd_pct","n_seeds"],acc_rows)
wcsv("f3_rho.csv",["setting","round","rho"],rho_rows)

# ================= F4a: role ablation TV-rho correlation (same-pool, 3 seeds) =================
print("F4a role ablation (TV-rho Spearman; same-pool)")
role_map={"standard (separated)":"runs/phaseC_signals/dvsig_tv_A_s*.json",
          "same-role exits":"runs/phaseR_role/role_same_role_s*.json",
          "same-role independent init":"runs/phaseR_role/role_same_role_indep_s*.json",
          "weakened server":"runs/phaseR_role/role_weak_server_s*.json"}
rows=[]
for cond,pat in role_map.items():
    fs=famf(pat); corrs=[]
    for f in fs.values():
        h=load(f); tv=sig_series(h,"tv_dist"); rho=rho_series(h)
        corrs.append(stats.spearmanr(tv[15:],rho[15:]).statistic)
    if corrs:
        rows.append([cond,f"{np.mean(corrs):+.4f}",f"{np.std(corrs,ddof=1):.4f}" if len(corrs)>1 else "",len(corrs),"same-pool"])
wcsv("f4_role.csv",["condition","tv_rho_spearman_mean","sd","n_seeds","protocol"],rows)

# ================= F4d: absolute-view contribution (development-version: dual vs main) =================
print("F4d absolute-view contribution (development-version dual[tv+abs] vs main[ST,delta])")
rows=[]
for setting,dualp,mainp in [("CIFAR-100 gradual","runs/gated/d4_cifar100/d4_dual_gsig_s*.json","runs/gated/d4_cifar100/d4_main_gsig_s*.json"),
                            ("CIFAR-100 spatial","runs/gated/d4_cifar100/d4_dual_sp_s*.json","runs/gated/d4_cifar100/d4_main_sp_s*.json"),
                            ("Tiny-ImageNet","runs/gated/d5_tinyimagenet/d5_dual_gsig_s*.json","runs/gated/d5_tinyimagenet/d5_main_gsig_s*.json")]:
    du={s:integ(load(f)) for s,f in famf(dualp).items()}; mn={s:integ(load(f)) for s,f in famf(mainp).items()}
    s,d=paired(du,mn); lo,hi=ci95(d)
    rows.append([setting,f"{np.mean(list(du.values()))*100:.4f}",f"{np.mean(list(mn.values()))*100:.4f}",
                 f"{d.mean()*100:+.4f}",f"[{lo*100:+.2f},{hi*100:+.2f}]",int((d>0).sum()),len(d),
                 "development-version: dual=tv+abs, main=ST relative-only (delta_hard); same-pool"])
wcsv("f4_absolute_contribution.csv",["setting","full_dual_acc","relative_main_acc","full_minus_relative_pp","ci95","n_pos","n_seeds","note"],rows)

# ================= F5a: timing controls (same-pool replay) =================
print("F5a timing controls (adaptive vs global-mean/per-cell-mean/shuffled; same-pool replay)")
rows=[]
env_map={"Schedule A":"A","gradual":"gsig","static spatial":"sp"}
# adaptive source = the exact trajectory the replay controls were derived from (enqueue_phaseB ENVS)
adap_src={"A":"runs/gated/d2_horizon/d2_dual_A_s*.json",
          "gsig":"runs/gated/d1_unseen/d1_dual_gradual_sigmoid_s*.json",
          "sp":"runs/gated/d3_spatial/d3_dual_sp_s*.json"}
for setting,env in env_map.items():
    adap={s:integ(load(f)) for s,f in famf(adap_src[env]).items()}
    for ctrl,pat in [("global mean λ",f"runs/phaseB_decomp/gmm_{env}_s*.json"),
                     ("per-cluster mean λ",f"runs/phaseB_decomp/pcm_{env}_s*.json"),
                     ("shuffled λ",f"runs/phaseB_decomp/shuf_{env}_s*.json")]:
        cc={s:integ(load(f)) for s,f in famf(pat).items()}
        if not cc or not adap: rows.append([setting,ctrl,"","","","0","unmatched"]); continue
        s,d=paired(adap,cc); lo,hi=ci95(d)
        rows.append([setting,ctrl,f"{d.mean()*100:+.4f}",f"[{lo*100:+.2f},{hi*100:+.2f}]",int((d>0).sum()),len(d),"same-pool replay"])
wcsv("f5_timing.csv",["setting","control","adaptive_minus_control_pp","ci95","n_pos","n_seeds","protocol"],rows)

# ================= F5b: Lambda contribution (disjoint) =================
print("F5b Lambda contribution (full TV vs adaptive-lambda Lambda=0.5; disjoint)")
rows=[]
for setting,dgp,a1p in [("Schedule A","runs/phaseT1_disjoint/t1_A_dg_s*.json","runs/phaseT1_disjoint/t1_A_a1_s*.json"),
                        ("mobility-med","runs/phaseT1_disjoint/t1_mob_dg_s*.json","runs/phaseT1_disjoint/t1_mob_a1_s*.json")]:
    dg={s:integ(load(f)) for s,f in famf(dgp).items()}; a1={s:integ(load(f)) for s,f in famf(a1p).items()}
    s,d=paired(dg,a1); lo,hi=ci95(d)
    rows.append([setting,f"{d.mean()*100:+.4f}",f"[{lo*100:+.2f},{hi*100:+.2f}]",int((d>0).sum()),len(d),"disjoint"])
wcsv("f5_lambda.csv",["setting","full_minus_fixedLambda_pp","ci95","n_full_higher","n_seeds","protocol"],rows)

# ================= F6: server non-Main rate vs TV, 7 settings (disjoint) =================
print("F6 server non-Main rate - TV, 7 settings (disjoint) [from round2 recompute]")
src=JR/"tables/final_closure_round2/hard_vs_tv_all_settings.csv"
order=["SVHN temporal","CIFAR-10 Schedule A","Schedule A","CIFAR-10 gradual","mobility-med",
       "Tiny-ImageNet","CIFAR-100 gradual","CIFAR-100 spatial"]
rows=[]
if src.exists():
    d={r["setting"]:r for r in csv.DictReader(open(src))}
    for name in order:
        if name not in d: continue
        r=d[name]
        rows.append([name,r["n_seeds"],r["tv_mean"],r["server_nonmain_rate_mean"],
                     r["hard_minus_tv_mean"],r["hard_minus_tv_ci95"],r["seeds_hard_higher"]])
wcsv("f6_server_signal.csv",["setting","n_seeds","tv_mean_pct","rate_mean_pct","rate_minus_tv_pp","ci95","n_rate_higher"],rows)

# ================= primary performance + paired diffs (numbers doc) =================
print("primary performance table")
rows=[]
prim={"CIFAR-10 Schedule A":dict(n=5,tv="runs/phaseT1_disjoint/t1_A_dg_s*.json",ent="runs/phaseT1_disjoint/t1_A_ent_s*.json",
         fx40="runs/phaseT1_disjoint/t1_A_fx40_s*.json",fx30="runs/phaseT1_disjoint/t1_A_fx30_s*.json",
         fx50="runs/phaseT1_disjoint/t1_A_fx50_s*.json",a1="runs/phaseT1_disjoint/t1_A_a1_s*.json",
         fx20="runs/phaseT3_fixedref/t3_c10*_fx20_s*.json"),  # A fx20 lives elsewhere; handled below
      "CIFAR-10 medium mobility":dict(n=3,tv="runs/phaseT1_disjoint/t1_mob_dg_s*.json",ent="runs/phaseT1_disjoint/t1_mob_ent_s*.json",
         fx40="runs/phaseT1_disjoint/t1_mob_fx40_s*.json",a1="runs/phaseT1_disjoint/t1_mob_a1_s*.json")}
def mstats(pat):
    fs=famf(pat);
    if not fs: return None
    v=[integ(load(f)) for f in fs.values()]; return (np.mean(v)*100, np.std(v,ddof=1)*100 if len(v)>1 else float('nan'), len(v))
for setting,cfg in prim.items():
    for arm in ["tv","ent","fx40","fx30","fx50","a1"]:
        if arm not in cfg: continue
        m=mstats(cfg[arm])
        if m: rows.append([setting,arm,f"{m[0]:.4f}",f"{m[1]:.4f}",m[2]])
# pre-registered fixed l0.2 (disjoint) for A & mob from round2
for setting,pat in [("CIFAR-10 Schedule A","runs/phaseT2r2_signal/r2_A_fx20_s*.json"),
                    ("CIFAR-10 medium mobility","runs/phaseT2r2_signal/r2_mob_fx20_s*.json")]:
    m=mstats(pat)
    if m: rows.append([setting,"fx20_prereg",f"{m[0]:.4f}",f"{m[1]:.4f}",m[2]])
wcsv("primary_performance.csv",["setting","arm","mean_pct","sd_pct","n_seeds"],rows)

# paired diffs TV - reference
prows=[]
for setting,tvp,refs in [("CIFAR-10 Schedule A","runs/phaseT1_disjoint/t1_A_dg_s*.json",
                          [("fixed l0.4","runs/phaseT1_disjoint/t1_A_fx40_s*.json"),("entropy","runs/phaseT1_disjoint/t1_A_ent_s*.json")]),
                         ("CIFAR-10 medium mobility","runs/phaseT1_disjoint/t1_mob_dg_s*.json",
                          [("fixed l0.4","runs/phaseT1_disjoint/t1_mob_fx40_s*.json"),("entropy","runs/phaseT1_disjoint/t1_mob_ent_s*.json")])]:
    tv={s:integ(load(f)) for s,f in famf(tvp).items()}
    for lbl,rp in refs:
        ref={s:integ(load(f)) for s,f in famf(rp).items()}
        s,d=paired(tv,ref); lo,hi=ci95(d)
        prows.append([setting,f"TV - {lbl}",f"{d.mean()*100:+.4f}",f"[{lo*100:+.2f},{hi*100:+.2f}]",int((d>0).sum()),len(d),"disjoint"])
wcsv("primary_paired_diff.csv",["setting","comparison","mean_pp","ci95","n_pos","n_seeds","protocol"],prows)

print("\nDONE. data written to", DATA)
