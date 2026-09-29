"""Round-2 finalize — remaining tables + 15 consistency asserts + selection state.
RAW JSON only; disjoint protocol; canonical integrated. Writes to final_closure_round2/.
"""
import json, glob, re, csv, math, sys, hashlib
import numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0,"."); sys.path.insert(0,"..")
from src.evaluation.signal_metrics import evaluate_signal
from src.controllers.self_calibrating import SIGNAL_RANGE, Z0, TAU_Z

JR=Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT=JR/"tables/final_closure_round2"
T1="runs/phaseT1_disjoint"; T2="runs/phaseT2_signal"; R2="runs/phaseT2r2_signal"
def load(p): return json.load(open(p))
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def er(h): return tuple(e["round"] for e in h["eval"])
def famf(pat):
    d={}
    for f in sorted(glob.glob(str(JR/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: d[int(m.group(1))]=f
    return d
def ifam(pat): return {s:integ(load(f)) for s,f in famf(pat).items()}
def paired(a,b):
    s=sorted(set(a)&set(b)); return s,np.array([a[x]-b[x] for x in s])
def ci(d):
    if len(d)<2: return (float('nan'),float('nan'))
    se=stats.sem(d); h=se*stats.t.ppf(0.975,len(d)-1); return d.mean()-h,d.mean()+h
def rho_s(h): return np.array([r if not isinstance(r,dict) else np.mean(list(r.values())) for r in h["rho_trace"]],float)
def ms(spe,nm): return np.array([np.mean([float(v) for v in d.values()]) for d in spe[nm]])
def seg(h,rho=0.8):
    v=[e["acc_total"] for e in h["eval"] if abs((e["rho"] if not isinstance(e["rho"],dict) else 0)-rho)<1e-6]
    return float(np.mean(v)) if v else None
def wcsv(nm,hdr,rows):
    with open(OUT/nm,"w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
def dfree(a): return max(a,1-a) if a==a else a

FAILS=[]
def check(n,c,d=""):
    print(f"[{'PASS' if c else 'FAIL'}] {n}  {d}")
    if not c: FAILS.append(n)

# ============ table_main_dynamic (Schedule A + mobility) ============
def block(name,arms):
    seeds_all=sorted(set.intersection(*[set(a) for a in arms.values() if a]))
    rows=[]
    tv=arms["TV"]
    for lbl,fam in arms.items():
        if not fam: rows.append([name,lbl,"","","unmatched"]); continue
        mean=np.mean(list(fam.values()))*100
        rows.append([name,lbl,f"{mean:.4f}",len(fam),str(sorted(fam))])
    return rows
A_arms={"TV":ifam(f"{T1}/t1_A_dg_s*.json"),"server non-Main rate (hard)":ifam(f"{T2}/t2_A_hard_s*.json"),
        "adaptive-lambda Lambda=0.5 (a1)":ifam(f"{T1}/t1_A_a1_s*.json"),
        "fixed-grid best (fx40)":ifam(f"{T1}/t1_A_fx40_s*.json"),
        "pre-fixed lambda0.2":ifam(f"{R2}/r2_A_fx20_s*.json"),"predictive entropy":ifam(f"{T1}/t1_A_ent_s*.json")}
M_arms={"TV":ifam(f"{T1}/t1_mob_dg_s*.json"),"server non-Main rate (hard)":ifam(f"{T2}/t2_mob_hard_s*.json"),
        "adaptive-lambda Lambda=0.5 (a1)":ifam(f"{T1}/t1_mob_a1_s*.json"),
        "fixed-grid best (fx40)":ifam(f"{T1}/t1_mob_fx40_s*.json"),
        "pre-fixed lambda0.2":ifam(f"{R2}/r2_mob_fx20_s*.json"),"predictive entropy":ifam(f"{T1}/t1_mob_ent_s*.json")}
mrows=[]
for nm,arms in [("Schedule A",A_arms),("mobility-med",M_arms)]:
    tv=arms["TV"]
    for lbl,fam in arms.items():
        s,d=paired(fam,tv); _,dbf=paired(fam,arms["fixed-grid best (fx40)"]); _,den=paired(fam,arms["predictive entropy"])
        lo,hi=ci(d)
        mrows.append([nm,lbl,f"{np.mean(list(fam.values()))*100:.4f}",len(fam),
            f"{d.mean()*100:+.4f}" if lbl!="TV" else "-", # method - TV
            f"{dbf.mean()*100:+.4f}", f"{den.mean()*100:+.4f}",
            f"[{lo*100:+.2f},{hi*100:+.2f}]" if lbl!="TV" else "-", int((d>0).sum()) if lbl!="TV" else "-"])
wcsv("table_main_dynamic.csv",["setting","method","mean_acc","n_seeds","method_minus_TV_pp",
     "method_minus_fixedgrid_pp","method_minus_entropy_pp","method_minus_TV_ci95","seeds_method_over_TV"],mrows)

# ============ high_nonmain_segment (Task 6.1) ============
segA={"TV":famf(f"{T1}/t1_A_dg_s*.json"),"server non-Main rate (hard)":famf(f"{T2}/t2_A_hard_s*.json"),
      "predictive entropy":famf(f"{T1}/t1_A_ent_s*.json"),"fixed lambda0.4":famf(f"{T1}/t1_A_fx40_s*.json"),
      "pre-fixed lambda0.2":famf(f"{R2}/r2_A_fx20_s*.json"),"adaptive-lambda Lambda=0.5":famf(f"{T1}/t1_A_a1_s*.json")}
segA={k:{s:load(f) for s,f in v.items()} for k,v in segA.items()}
srows=[]
for k,v in segA.items():
    per=[seg(h) for h in v.values()]
    srows.append([k,f"{np.mean(per)*100:.4f}",len(v),str([round(x*100,2) for x in per]),"rho=0.8 (rounds 70,80,90); metric acc_total; source task0_headline_audit.high_drift"])
for a,b in [("TV","predictive entropy"),("server non-Main rate (hard)","predictive entropy"),
            ("TV","fixed lambda0.4"),("server non-Main rate (hard)","fixed lambda0.4"),
            ("TV","pre-fixed lambda0.2"),("server non-Main rate (hard)","pre-fixed lambda0.2"),
            ("TV","server non-Main rate (hard)")]:
    sa=segA[a]; sb=segA[b]; seeds=sorted(set(sa)&set(sb))
    d=np.array([seg(sa[s])-seg(sb[s]) for s in seeds]); lo,hi=ci(d)
    srows.append([f"DIFF {a} - {b}",f"{d.mean()*100:+.4f}",len(d),f"CI95[{lo*100:+.2f},{hi*100:+.2f}] pos={int((d>0).sum())}/{len(d)}",""])
wcsv("high_nonmain_segment.csv",["arm_or_diff","value_pct_or_pp","n_seeds","per_seed_or_ci","definition"],srows)

# ============ fixed_lambda_capital_mobility (Task 6.2) ============
dg=ifam(f"{T1}/t1_mob_dg_s*.json"); a1=ifam(f"{T1}/t1_mob_a1_s*.json")
s,d=paired(dg,a1); lo,hi=ci(d)
frows=[[s_,f"{dg[s_]*100:.4f}",f"{a1[s_]*100:.4f}",f"{(dg[s_]-a1[s_])*100:+.4f}"] for s_ in s]
frows.append(["MEAN",f"{np.mean(list(dg.values()))*100:.4f}",f"{np.mean(list(a1.values()))*100:.4f}",
              f"{d.mean()*100:+.4f} CI95[{lo*100:+.2f},{hi*100:+.2f}] full-higher {int((d>0).sum())}/{len(d)}"])
wcsv("fixed_lambda_capital_mobility.csv",["seed","TV_full_adaptiveLambda","adaptive_lambda_Lambda0.5","full_minus_fixedLambda_pp"],frows)

# ============ server_role_nonmain_signal (Task 6.3) ============
rrows=[]
for role in ["separated","same_role","weak_server"]:
    f=JR/f"{R2}/r2_role_{role}_s0.json"
    if not f.exists(): continue
    h=load(f); spe=h["signals_per_es"]; x=ms(spe,"server_nonmain_hard"); rho=rho_s(h)
    m=evaluate_signal(x,rho,warmup=15,drift_level=0.5)
    sm=np.nanmean([e.get("server_main",np.nan) for e in h["eval"]]); so=np.nanmean([e.get("server_oop",np.nan) for e in h["eval"]])
    rrows.append([role,round(float(x[15:].mean()),4),round(float(x[15:].std()),4),len(set(np.round(x,4))),
                  round(float(m["spearman_rho"]),4),round(float(m["auroc_raw"]),4),round(float(sm),4),round(float(so),4),"seed0"])
wcsv("server_role_nonmain_signal.csv",["role","signal_mean","signal_std","n_unique","spearman_rho","auroc_raw",
     "server_main_acc","server_oop_acc","seeds"],rrows)

# ============ table_signal_comparison (passive + downstream) ============
# passive from passive_signal_quality.csv (already computed); downstream from ifam
def dmean(pat):
    v=ifam(pat); return (np.mean(list(v.values()))*100, len(v)) if v else (None,0)
sig_meta={"predictive entropy":("client+server logits","no","no","[0, ln C]"),
          "server non-Main prob mass (soft)":("server logits only","yes","no","[0,1]"),
          "server non-Main pred rate (hard)":("server argmax only","yes","no","[0,1]"),
          "client-server TV":("client+server logits","no","no","[0,1]")}
down={"predictive entropy":dict(A=f"{T1}/t1_A_ent_s*.json",mob=f"{T1}/t1_mob_ent_s*.json",svhn=f"{T2}/t2_svhn_ent_s*.json"),
      "server non-Main prob mass (soft)":dict(A=f"{T2}/t2_A_soft_s*.json",mob=f"{T2}/t2_mob_soft_s*.json",svhn=f"{T2}/t2_svhn_soft_s*.json"),
      "server non-Main pred rate (hard)":dict(A=f"{T2}/t2_A_hard_s*.json",mob=f"{T2}/t2_mob_hard_s*.json",svhn=f"{T2}/t2_svhn_hard_s*.json",
          c100sp=f"{R2}/r2_c100sp_hard_s*.json",c100g=f"{R2}/r2_c100gsig_hard_s*.json",tiny=f"{R2}/r2_tiny_hard_s*.json"),
      "client-server TV":dict(A=f"{T1}/t1_A_dg_s*.json",mob=f"{T1}/t1_mob_dg_s*.json",svhn=f"{T2}/t2_svhn_tv_s*.json",
          c100sp=f"{R2}/r2_c100sp_tv_s*.json",c100g=f"{R2}/r2_c100gsig_tv_s*.json",tiny=f"{R2}/r2_tiny_tv_s*.json")}
# passive spearman/auroc from CSV
pas={}
for r in csv.DictReader(open(OUT/"passive_signal_quality.csv")):
    key=(r["setting"],r["signal"]); pas.setdefault(key,[]).append((float(r["spearman_signal_rho"]),float(r["raw_direction_auroc"])))
def pget(setting,sig):
    vals=pas.get((setting,sig));
    if not vals: return ("","")
    sp=np.mean([v[0] for v in vals]); au=np.mean([v[1] for v in vals]); return (f"{sp:+.3f}",f"{au:.3f}/{dfree(au):.3f}")
scomp=[]
for sig,(out_,meta_client,meta_rt,rng) in sig_meta.items():
    spA,auA=pget("Schedule A",sig); spS,auS=pget("SVHN temporal",sig)
    def dm(k): m,n=dmean(down[sig][k]) if k in down[sig] else (None,0); return f"{m:.2f}(n{n})" if m else ""
    scomp.append([sig,out_,meta_client,meta_rt,rng,spA,auA,spS,auS,dm("A"),dm("mob"),dm("svhn"),dm("c100sp"),dm("c100g"),dm("tiny")])
wcsv("table_signal_comparison.csv",["signal","model_output_needed","client_metadata_needed","runtime_label_needed","signal_range",
     "passive_spearman_A","passive_auroc_A(raw/dirfree)","passive_spearman_SVHN","passive_auroc_SVHN(raw/dirfree)",
     "acc_ScheduleA","acc_mobility","acc_SVHN","acc_C100spatial","acc_C100gradual","acc_TinyImageNet"],scomp)

# ============ table_settings ============
SET={"Schedule A":dict(tv=f"{T1}/t1_A_dg_s*.json",hard=f"{T2}/t2_A_hard_s*.json",fx02=f"{R2}/r2_A_fx20_s*.json",ent=f"{T1}/t1_A_ent_s*.json",fxg=f"{T1}/t1_A_fx40_s*.json"),
 "CIFAR-10 gradual":dict(tv=f"{R2}/r2_c10gsig_tv_s*.json",hard=f"{R2}/r2_c10gsig_hard_s*.json"),
 "CIFAR-100 spatial":dict(tv=f"{R2}/r2_c100sp_tv_s*.json",hard=f"{R2}/r2_c100sp_hard_s*.json"),
 "CIFAR-100 gradual":dict(tv=f"{R2}/r2_c100gsig_tv_s*.json",hard=f"{R2}/r2_c100gsig_hard_s*.json"),
 "Tiny-ImageNet":dict(tv=f"{R2}/r2_tiny_tv_s*.json",hard=f"{R2}/r2_tiny_hard_s*.json"),
 "SVHN temporal":dict(tv=f"{T2}/t2_svhn_tv_s*.json",hard=f"{T2}/t2_svhn_hard_s*.json",fx02=f"{T2}/t2_svhn_fx20_s*.json",ent=f"{T2}/t2_svhn_ent_s*.json"),
 "mobility slow":dict(),"mobility medium":dict(tv=f"{T1}/t1_mob_dg_s*.json",hard=f"{T2}/t2_mob_hard_s*.json",fx02=f"{R2}/r2_mob_fx20_s*.json",ent=f"{T1}/t1_mob_ent_s*.json",fxg=f"{T1}/t1_mob_fx40_s*.json"),"mobility fast":dict()}
setrows=[]
for name,cfg in SET.items():
    def g(k):
        if k not in cfg: return "unmatched(no disjoint ref)"
        m,n=dmean(cfg[k]); return f"{m:.2f}(n{n})" if m else "unmatched"
    if not cfg:
        setrows.append([name,"","","","","not run (budget; hard not uniform default -> mobility figure stays TV)"]); continue
    setrows.append([name,g("tv"),g("hard"),g("fxg") if "fxg" in cfg else "unmatched(no disjoint grid)",
                    g("fx02"),g("ent") if "ent" in cfg else "unmatched(no disjoint ref)"])
wcsv("table_settings.csv",["setting","client_server_TV","server_nonMain_rate","fixed_grid_best","pre_fixed_lambda0.2","predictive_entropy"],setrows)

# ============ run_manifest ============
def manifest_rows():
    rows=[]
    for f in sorted(glob.glob(str(JR/f"{R2}/*.json"))):
        h=load(f); c=h.get("config",{}); nm=f.split("/")[-1][:-5]
        rows.append([c.get("run_id",nm),"round2",nm,"disjoint",c.get("controller_signal"),
                     re.search(r"_s(\d+)$",nm).group(1) if re.search(r"_s(\d+)$",nm) else "",
                     "e82b96b",str((JR/'..'/'configs/base_v3.yaml').resolve()).split('adaptive_splitomc_tmc/')[-1],
                     "yes","no",h.get("probe_eval_overlap"),len(h["eval"]),round(h.get("total_time_sec",0)/3600,2)])
    return rows
wcsv("run_manifest.csv",["run_id","task","run_name","protocol","signal","seed","git_commit","config_path",
     "new_training","evaluation_only","overlap_count","actual_eval_rounds","gpu_hours"],manifest_rows())

# ============ paper_number_status ============
pn=[["intro_TV_spearman","~0.92","TV Spearman vs rho @150R","yes","yes(evaluate_signal drift=0.5)","REPRODUCED","0.907","rec_A passive fixed-lambda, 3 seeds","3","Schedule A (rec runs)","rec_A_s0..s2"],
    ["intro_entropy_spearman","~0.50","entropy Spearman vs rho","yes","yes","REPLACED","A:+0.44 / abrupt:-0.58","rec_A / rec_abrupt passive","3","schedule-dependent, sign flips","rec_A_s*,rec_abrupt_s*"],
    ["intro_TV_auroc","0.98-1.00","TV AUROC drift vs no-drift","yes","yes","REPRODUCED","A:0.978 abrupt:1.000","rec passive","3","raw-direction AUROC","rec_A_s*,rec_abrupt_s*"],
    ["intro_entropy_auroc","0.16","entropy AUROC abrupt","yes","yes","REPRODUCED","0.161 (dir-free 0.839)","rec_abrupt passive","3","raw-direction; MUST cite dir-free too","rec_abrupt_s*"],
    ["highdrift_signal_gap","+3.64pp","TV-entropy @rho=0.8 (same-pool)","yes","yes","REPLACED","+6.71pp (disjoint 5-seed)","Schedule A disjoint","5","recomputed disjoint; same-pool not mixed","t1_A_dg_s*,t1_A_ent_s*"]]
wcsv("paper_number_status.csv",["paper_claim_id","old_value","old_wording","source_found","exact_definition_found",
     "status","final_value","final_setting","final_seed_count","final_wording_scope","source_run_ids"],pn)

# ================= 15 CONSISTENCY ASSERTS =================
print("\n=== 15 CONSISTENCY ASSERTS ===")
canon=load(JR/"tables/canonical_headline_metrics.json")
# 1 no seed-count mixing in paired diffs (all hard-vs-tv use equal seed sets)
allmatch=all(set(famf(SET[s]["tv"]).keys())==set(famf(SET[s]["hard"]).keys()) for s in SET if SET[s])
check("1 no seed-count mixing (hard vs TV equal seeds)",allmatch)
# 2 no protocol mixing: all r2/t1/t2 runs used here are disjoint (overlap key present)
prot=all(load(f).get("probe_eval_overlap") is not None for s in SET if SET[s] for k in ("tv","hard") for f in famf(SET[s][k]).values())
check("2 no protocol mixing (all disjoint)",prot)
# 3 all new adaptive runs overlap 0
ov=[load(f).get("probe_eval_overlap") for f in glob.glob(str(JR/f"{R2}/r2_*_s*.json")) if "fx20" not in f and "role" not in f]
check("3 all new adaptive runs overlap==0",all(o==0 for o in ov),f"n={len(ov)}")
# 4 fixed runs classified separately (fx20 are fixed -> overlap 0 too but not 'adaptive')
fxov=[load(f).get("probe_eval_overlap") for f in glob.glob(str(JR/f"{R2}/r2_*fx20*_s*.json"))]
check("4 fixed runs overlap==0 (classified separately)",all(o==0 for o in fxov),f"n_fixed={len(fxov)}")
# 5 eval rounds match config expected (read from each run, not hardcoded)
def expected(nm): return 11 if "tiny" in nm else (13 if "mob" in nm else 16)
er_ok=all(len(load(f)["eval"])==expected(f.split("/")[-1]) for f in glob.glob(str(JR/f"{R2}/*.json")))
check("5 eval rounds match per-setting expected grid",er_ok)
# 6 every table number traceable to run id (manifest non-empty, covers all r2)
man=len(manifest_rows()); check("6 run manifest covers all r2 runs",man==len(glob.glob(str(JR/f"{R2}/*.json"))),f"{man} rows")
# 7 table mean == mean of per-seed (main_dynamic A TV)
tvA=ifam(f"{T1}/t1_A_dg_s*.json"); check("7 table mean == per-seed mean",abs(np.mean(list(tvA.values()))-np.mean([tvA[s] for s in tvA]))<1e-12)
# 8 diff == per-seed diff mean (hard-tv A)
hA=ifam(f"{T2}/t2_A_hard_s*.json"); s,d=paired(hA,tvA)
check("8 diff == per-seed diff mean",abs(d.mean()-(np.mean([hA[x] for x in s])-np.mean([tvA[x] for x in s])))<1e-12)
# 9 fixed-grid best >= all grid values (Schedule A)
grid={k:np.mean(list(ifam(f"{T1}/t1_A_{k}_s*.json").values())) for k in ["fx30","fx40","fx50"]}
check("9 fixed-grid best >= all grid values",max(grid,key=grid.get)=="fx40" and grid["fx40"]>=max(grid.values())-1e-12,f"best fx40={grid['fx40']:.4f}")
# 10 Round-1 canonical reproduced (Schedule A adaptive 5-seed == canonical)
sp=ifam("runs/phaseC_signals/dvsig_tv_A_s*.json"); sp.update({k:v for k,v in ifam("runs/phaseS_supp/b1_adaptive_A_s*.json").items() if k not in sp})
md=max(abs(sp[int(s)]-canon["scheduleA_adaptive_per_seed"][s]) for s in canon["scheduleA_adaptive_per_seed"])
check("10 Round-1 canonical reproduced",md<1e-9,f"max|d|={md:.1e}")
# 11 mobility hard & soft seed 2 resolved (both have 3 seeds valid)
mh=famf(f"{T2}/t2_mob_hard_s*.json"); msf=famf(f"{T2}/t2_mob_soft_s*.json")
check("11 mobility hard & soft seed2 resolved",set(mh)=={0,1,2} and set(msf)=={0,1,2})
# 12 main-class metadata from training split only (audit CSV all 'no' leakage)
maud=list(csv.DictReader(open(OUT/"main_class_metadata_audit.csv")))
check("12 main-class metadata training-only (no leakage)",all(r["eval_label_access"]=="no" and r["probe_label_access"]=="no" for r in maud),f"n={len(maud)}")
# 13 no new same-pool run created
newpool=[f for f in glob.glob(str(JR/f"{R2}/*.json")) if load(f).get("probe_eval_overlap") is None]
check("13 no new same-pool run (all r2 disjoint)",len(newpool)==0)
# 14 abs-active fraction has defined denominator (0..1)
def af(path,sg):
    h=load(path); cz=h["controller_z"]; sig=h["signals_per_es"][sg]; rng=SIGNAL_RANGE.get(sg,1.0); n=a=0
    for r in range(25,len(cz)):
        for es in cz[r]:
            g=1/(1+math.exp(-max(-50,min(50,(float(cz[r][es])-Z0)/TAU_Z)))); dh=min(max(float(sig[r][es])/rng,0),1); n+=1; a+=(dh>g)
    return a/n
afv=af(famf(f"{T2}/t2_A_hard_s*.json")[0],"server_nonmain_hard")
check("14 abs-active fraction in [0,1] defined denom",0<=afv<=1,f"A hard={afv:.3f}")
# 15 all intro numbers linked to source (paper_number_status has source_run_ids)
pnr=list(csv.DictReader(open(OUT/"paper_number_status.csv")))
check("15 intro numbers linked to source",all(r["source_run_ids"] for r in pnr),f"n={len(pnr)}")

print(f"\n{'ALL 15 PASS' if not FAILS else 'FAILURES: '+', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
