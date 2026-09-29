"""Task-2 FINAL — direct server non-Main signals vs TV dual-exit divergence.
Two views, both recomputed from RAW run JSON:
  (A) DOWNSTREAM accuracy: full controller runs, disjoint pools, A/mob/SVHN.
  (B) PASSIVE signal quality: Spearman(signal, rho) and AUROC(signal, drift-active)
      for tv/soft/hard/entropy read from ONE common neutral backbone per seed
      (the entropy-controller run: none of tv/soft/hard shaped it), on SVHN where
      rho is a clean 0->0.8 sigmoid ramp (Schedule A rho_trace is 0 -> not usable).
Confounds noted in the report. NO signal-specific tuning; identical DV pipeline.
"""
import json, glob, re
import numpy as np
from pathlib import Path
from scipy import stats

def roc_auc_score(label, score):
    """AUROC via the rank (Mann-Whitney U) identity; ties get average ranks."""
    label=np.asarray(label); score=np.asarray(score)
    order=np.argsort(score, kind="mergesort"); ranks=np.empty(len(score),float)
    sc=score[order]; i=0; r=1
    while i < len(sc):
        j=i
        while j+1 < len(sc) and sc[j+1]==sc[i]: j+=1
        ranks[order[i:j+1]]=(r + r+(j-i))/2.0; r+=(j-i+1); i=j+1
    npos=int(label.sum()); nneg=len(label)-npos
    if npos==0 or nneg==0: return float("nan")
    return (ranks[label==1].sum() - npos*(npos+1)/2.0)/(npos*nneg)

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
def load(p): return json.load(open(p))
def integrated(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def erounds(h): return tuple(e["round"] for e in h["eval"])

def fam(pat):
    out={}
    for f in sorted(glob.glob(str(JR/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: out[int(m.group(1))]=f
    return out

def integ_fam(pat):
    return {s:integrated(load(f)) for s,f in fam(pat).items()}

def paired(a,b):
    seeds=sorted(set(a)&set(b)); d=np.array([a[s]-b[s] for s in seeds]); return seeds,d

def mean_series(spe, name):
    """mean-over-ES per-round series for a raw signal name."""
    ser=spe[name]  # list of {es:val}
    return np.array([np.mean([float(v) for v in d.values()]) for d in ser])

L=[]
def p(s=""): L.append(s); print(s)

p("="*72); p("TASK 2 FINAL — direct server non-Main signal vs TV dual-exit divergence"); p("="*72)

# ---------------- (A) DOWNSTREAM ACCURACY ----------------
p("\n### (A) DOWNSTREAM integrated accuracy (disjoint pools) ###")
D="runs/phaseT1_disjoint"; T="runs/phaseT2_signal"
blocks = {
 "Schedule A (5 seeds)": dict(
    tv=integ_fam(f"{D}/t1_A_dg_s*.json"), soft=integ_fam(f"{T}/t2_A_soft_s*.json"),
    hard=integ_fam(f"{T}/t2_A_hard_s*.json"), ent=integ_fam(f"{D}/t1_A_ent_s*.json"),
    fixed=integ_fam(f"{D}/t1_A_fx40_s*.json")),
 "mobility-med (3 seeds)": dict(
    tv=integ_fam(f"{D}/t1_mob_dg_s*.json"), soft=integ_fam(f"{T}/t2_mob_soft_s*.json"),
    hard=integ_fam(f"{T}/t2_mob_hard_s*.json"), ent=integ_fam(f"{D}/t1_mob_ent_s*.json"),
    fixed=integ_fam(f"{D}/t1_mob_fx40_s*.json")),
 "SVHN temporal (5 seeds)": dict(
    tv=integ_fam(f"{T}/t2_svhn_tv_s*.json"), soft=integ_fam(f"{T}/t2_svhn_soft_s*.json"),
    hard=integ_fam(f"{T}/t2_svhn_hard_s*.json"), ent=integ_fam(f"{T}/t2_svhn_ent_s*.json"),
    fixed=integ_fam(f"{T}/t2_svhn_fx20_s*.json")),
}
down_summary={}
for blk,arms in blocks.items():
    p(f"\n{blk}")
    p(f"  {'arm':8s}  mean(%)  n   per-seed")
    for k in ["tv","soft","hard","ent","fixed"]:
        v=arms[k]; mean=np.mean(list(v.values()))*100 if v else float('nan')
        ps=[round(v[s]*100,2) for s in sorted(v)]
        p(f"  {k:8s}  {mean:6.2f}  {len(v):d}   {ps}")
    for direct in ["soft","hard"]:
        seeds,d=paired(arms["tv"],arms[direct])
        se=stats.sem(d) if len(d)>1 else 0; h=se*stats.t.ppf(0.975,len(d)-1) if len(d)>1 else 0
        try: _,pw=stats.wilcoxon(d)
        except Exception: pw=float('nan')
        _,pt=stats.ttest_rel([arms["tv"][s] for s in seeds],[arms[direct][s] for s in seeds]) if len(d)>1 else (0,float('nan'))
        p(f"    TV − {direct:4s}: {d.mean()*100:+.3f}pp  CI95[{(d.mean()-h)*100:+.2f},{(d.mean()+h)*100:+.2f}]  "
          f"TV-higher {int((d>0).sum())}/{len(d)}  t-p={pt:.3f} wilcox-p={pw:.3f}  per-seed={[round(x*100,2) for x in d]}")
        down_summary[(blk,direct)]=(d.mean()*100,int((d>0).sum()),len(d),pt)

# ---------------- (B) PASSIVE SIGNAL QUALITY (SVHN) ----------------
p("\n### (B) PASSIVE signal quality on SVHN (common neutral backbone = entropy run) ###")
p("Spearman(signal, rho) and AUROC(signal, drift-active := rho >= 0.4). Higher = better tracks drift.")
NAMEMAP={"tv":"tv_dist","soft":"server_nonmain_soft","hard":"server_nonmain_hard","ent":"ent_client"}
q_spear={k:[] for k in NAMEMAP}; q_auroc={k:[] for k in NAMEMAP}
for s in range(5):
    bpath=JR/f"{T}/t2_svhn_ent_s{s}.json"
    if not bpath.exists(): continue
    h=load(bpath); spe=h["signals_per_es"]; rho=np.array(h["rho_trace"])
    label=(rho>=0.4).astype(int)
    for k,nm in NAMEMAP.items():
        sig=mean_series(spe,nm)
        sp=stats.spearmanr(sig,rho).statistic
        q_spear[k].append(sp)
        if label.min()!=label.max():
            q_auroc[k].append(roc_auc_score(label,sig))
p(f"\n  {'signal':8s}  Spearman(vs rho)      AUROC(drift-active)")
for k in ["tv","soft","hard","ent"]:
    sm=np.mean(q_spear[k]); am=np.mean(q_auroc[k]) if q_auroc[k] else float('nan')
    p(f"  {k:8s}  {sm:+.4f} (n={len(q_spear[k])})    {am:.4f} (n={len(q_auroc[k])})   per-seed-spear={[round(x,3) for x in q_spear[k]]}")

# ---------------- VERDICT ----------------
p("\n### TASK 2 VERDICT (§5.8) ###")
p("Downstream TV−direct (positive => TV better):")
for (blk,direct),(m,npos,n,pt) in down_summary.items():
    tag = "TV better" if m>0.15 else ("direct better" if m<-0.15 else "≈ tie")
    p(f"  {blk:24s} TV−{direct:4s} = {m:+.3f}pp  ({npos}/{n} TV-higher, t-p={pt:.3f})  -> {tag}")
p("\nSignal quality (SVHN, Spearman vs rho):")
best=max(["tv","soft","hard"],key=lambda k:np.mean(q_spear[k]))
p(f"  ranking: " + " > ".join(sorted(["tv","soft","hard"],key=lambda k:-np.mean(q_spear[k]))) +
  f"  (best={best})")

Path(JR/"tables").mkdir(exist_ok=True)
open(JR/"tables/task2_final_analysis.txt","w").write("\n".join(L))
print("\n[written] tables/task2_final_analysis.txt")
