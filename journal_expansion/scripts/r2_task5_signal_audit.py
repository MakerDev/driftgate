"""Round-2 Task 5 — passive signal-quality audit.
Uses the EXISTING canonical definition src/evaluation/signal_metrics.evaluate_signal
(warmup=15, drift_level=0.5, raw-direction AUROC) — no new definition invented.
Part A: reproduce the Introduction numbers from the passive fixed-lambda rec_* runs.
Part B: final fair comparison of the four paper signals on a common backbone per
        setting (server non-Main hard/soft exist only on controller backbones, so
        the 4-way uses the entropy-controller run — NOT called a neutral backbone).
Writes tables/final_closure_round2/passive_signal_quality.csv (+ prints repro).
"""
import json, glob, re, csv, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "..")
from src.evaluation.signal_metrics import evaluate_signal
from scipy import stats

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT = JR/"tables/final_closure_round2"; OUT.mkdir(parents=True, exist_ok=True)
WARMUP=15; DRIFT=0.5

def mean_series(spe, name):
    return np.array([np.mean([float(v) for v in d.values()]) for d in spe[name]])
def rho_series(h):
    return np.array([r if not isinstance(r,dict) else np.mean(list(r.values())) for r in h["rho_trace"]],float)
def dfree(a): return max(a,1-a) if a==a else a

rows=[]
def add(setting,seed,signal,src,src_run,x,rho,note=""):
    m=evaluate_signal(x,rho,warmup=WARMUP,drift_level=DRIFT)
    rows.append(dict(setting=setting,seed=seed,signal=signal,model_trajectory_source=src,
        source_run_id=src_run,rho_source="get_rho deterministic (configured=realized)",
        rounds_used=f"{WARMUP+1}-{len(x)}",drift_label_definition="rho>=0.5 (evaluate_signal default)",
        spearman_signal_rho=round(float(m["spearman_rho"]),4),
        raw_direction_auroc=round(float(m["auroc_raw"]),4),
        direction_free_auroc=round(float(dfree(m["auroc_raw"])),4),
        signal_mean=round(float(np.mean(x[WARMUP:])),4),signal_std=round(float(np.std(x[WARMUP:])),4),
        reproduced_from_existing_log=("yes" if "rec_" in src_run else "no"),notes=note))
    return m

# ============ PART A: reproduce Introduction numbers from rec_* fixed backbone ============
print("### PART A — reproduce Introduction numbers (rec_* passive fixed-lambda backbone) ###")
NM_OLD={"tv":"tv_dist","entropy":"ent_client","delta_hard":"delta_hard"}
repro={}
for sched in ["A","abrupt"]:
    for sig,nm in NM_OLD.items():
        sp=[]; au=[]
        for s in range(3):
            f=JR/f"runs/signal_benchmark/rec_{sched}_s{s}.json"
            if not f.exists(): continue
            h=json.load(open(f)); x=mean_series(h["signals_per_es"],nm); rho=rho_series(h)
            m=add(f"rec_{sched}",s,sig,"passive fixed-lambda rec run",f"rec_{sched}_s{s}",x,rho,"historical repro")
            sp.append(m["spearman_rho"]); au.append(m["auroc_raw"])
        repro[(sched,sig)]=(np.mean(sp),np.mean(au))
        print(f"  {sched:7s} {sig:10s} Spearman={np.mean(sp):+.3f}  raw-AUROC={np.mean(au):.3f} (dir-free {dfree(np.mean(au)):.3f})  n={len(sp)}")

print("\n  vs Introduction claims:")
print(f"    TV Spearman claim ~0.92  -> reproduced A={repro[('A','tv')][0]:.3f}")
print(f"    entropy Spearman claim ~0.50 -> reproduced A={repro[('A','entropy')][0]:.3f}, abrupt={repro[('abrupt','entropy')][0]:.3f}")
print(f"    TV AUROC claim 0.98-1.00 -> reproduced A={repro[('A','tv')][1]:.3f}, abrupt={repro[('abrupt','tv')][1]:.3f}")
print(f"    entropy AUROC claim 0.16 -> reproduced abrupt raw={repro[('abrupt','entropy')][1]:.3f} (dir-free {dfree(repro[('abrupt','entropy')][1]):.3f})")

# ============ PART B: final fair 4-signal comparison on common controller backbone ============
print("\n### PART B — final fair comparison, 4 paper signals, common backbone per setting ###")
NM4={"client-server TV":"tv_dist","server non-Main prob mass (soft)":"server_nonmain_soft",
     "server non-Main pred rate (hard)":"server_nonmain_hard","predictive entropy":"ent_client"}
BACK={"Schedule A":("runs/phaseT1_disjoint/t1_A_ent_s{s}.json",5),
      "SVHN temporal":("runs/phaseT2_signal/t2_svhn_ent_s{s}.json",5)}
for setting,(patt,ns) in BACK.items():
    print(f"\n  {setting} (backbone = entropy-controller run; NOT a neutral backbone):")
    agg={k:{"sp":[],"au":[]} for k in NM4}
    for s in range(ns):
        f=JR/patt.format(s=s)
        if not f.exists(): continue
        h=json.load(open(f)); rho=rho_series(h); spe=h["signals_per_es"]
        for k,nm in NM4.items():
            if nm not in spe: continue
            x=mean_series(spe,nm)
            m=add(setting,s,k,"entropy-controller run (shared trajectory, entropy-shaped)",
                  h.get("config",{}).get("run_id",f.stem),x,rho,"4-way fair; backbone entropy-shaped")
            agg[k]["sp"].append(m["spearman_rho"]); agg[k]["au"].append(m["auroc_raw"])
    for k in NM4:
        if not agg[k]["sp"]: continue
        sp=np.mean(agg[k]["sp"]); au=np.mean(agg[k]["au"])
        print(f"    {k:34s} Spearman={sp:+.3f}  raw-AUROC={au:.3f}  dir-free={dfree(au):.3f}  n={len(agg[k]['sp'])}")

# write CSV
cols=["setting","seed","signal","model_trajectory_source","source_run_id","rho_source","rounds_used",
      "drift_label_definition","spearman_signal_rho","raw_direction_auroc","direction_free_auroc",
      "stationary_segment_spearman_signal_round","signal_mean","signal_std","reproduced_from_existing_log","notes"]
for r in rows: r.setdefault("stationary_segment_spearman_signal_round","")
with open(OUT/"passive_signal_quality.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
print(f"\n[written] {OUT}/passive_signal_quality.csv ({len(rows)} rows)")
