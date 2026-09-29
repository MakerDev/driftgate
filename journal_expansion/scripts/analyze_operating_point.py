"""§3 — preferred-operating-point variation vs adaptive gain (no new training).

For each environment, from the completed full fixed-λ grid, compute per-eval-window
λ*_t = argmax_λ A_t(λ) (fixed-λ accuracy at that window), then:
  V_lambda   = sum_t |λ*_t − λ*_{t-1}|         (how much the preferred point MOVES)
  var, range, #switches of λ*_t
  best−2nd-best fixed gap (mean over windows)
  DV-2 advantage vs best full-grid fixed (integrated, seed-paired where possible)
Then correlate V_lambda with the DV-2 advantage across environments.

Tests the hypothesis: adaptive gain is driven by TEMPORAL MOVEMENT of the preferred
operating point, not by non-stationarity per se.
"""
import json, glob, sys, csv
from pathlib import Path
import numpy as np

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
sys.path.insert(0, str(JR / "scripts"))
from analyze_downstream import run_metrics

GRID = JR / "runs/phaseB_grid"
LAMS = {"fx00":0.0,"fx10":0.1,"fx20":0.2,"fx30":0.3,"fx40":0.4,"fx50":0.5,
        "fx60":0.6,"fx70":0.7,"fx80":0.8}
# DV-2 seed-keyed integrated per env
DVSRC = {
 "A":"runs/phaseC_signals/dvsig_tv_A_s{s}.json",
 "asym":"runs/phaseD_final/dv2_asym_return_s{s}.json",
 "gsig":"runs/phaseD_final/dv2_gradual_sigmoid_s{s}.json",
 "pwr":"runs/phaseD_final/dv2_piecewise_random_s{s}.json",
 "sp":"runs/gated/d3_spatial/d3_dual_sp_s{s}.json",
 "c100gsig":"runs/gated/d4_cifar100/d4_dual_gsig_s{s}.json",
 "c100sp":"runs/gated/d4_cifar100/d4_dual_sp_s{s}.json",
}
# mobility: DV-2 vs best fixed (from phaseH); treat as high-movement dynamic points
MOB = {"mob_slow":0.008,"mob_med":0.0072,"mob_fast":0.0056}  # DV−bestfixed integrated (pp/100)


def grid_windows(env):
    """Return {lam_value: {seed: [per-window acc]}} and the window rho trace."""
    per={}
    for f in glob.glob(str(GRID/f"*_{env}_s*.json")):
        n=Path(f).stem; parts=n.split("_"); lamk=parts[0]
        if lamk not in LAMS: continue
        if "_".join(parts[1:-1])!=env: continue
        s=int(parts[-1][1:])
        try: h=json.load(open(f))
        except: continue
        accs=[e["acc_total"] for e in h["eval"]]
        per.setdefault(LAMS[lamk],{})[s]=accs
    return per


def env_metrics(env):
    per=grid_windows(env)
    if len(per)<3: return None
    lams=sorted(per)
    # mean per-window acc per lambda (over seeds)
    T=min(min(len(v) for v in sd.values()) for sd in per.values())
    A={lam:np.mean([per[lam][s][:T] for s in per[lam]],axis=0) for lam in lams}
    lam_star=[lams[int(np.argmax([A[lam][t] for lam in lams]))] for t in range(T)]
    V=float(np.sum(np.abs(np.diff(lam_star))))
    var=float(np.var(lam_star)); rng=max(lam_star)-min(lam_star)
    switches=int(np.sum(np.diff(lam_star)!=0))
    # best minus 2nd-best fixed gap per window (mean)
    gaps=[]
    for t in range(T):
        vals=sorted([A[lam][t] for lam in lams],reverse=True)
        gaps.append(vals[0]-vals[1])
    gap=float(np.mean(gaps))
    return dict(env=env,V_lambda=round(V,3),var_lamstar=round(var,4),
                range_lamstar=round(rng,2),switches=switches,
                best2nd_gap=round(gap,4),lam_star_trace=lam_star)


def dv_advantage(env):
    """DV-2 − best full-grid fixed (integrated, seed-paired mean)."""
    if env not in DVSRC: return None
    dv={}
    for s in range(5):
        try: dv[s]=run_metrics(json.load(open(JR/DVSRC[env].format(s=s))))['integrated']
        except FileNotFoundError: pass
    per=grid_windows(env)
    # best fixed = highest mean integrated over dv's seeds
    best,bm=None,-1
    for lam,sd in per.items():
        c=[s for s in dv if s in sd]
        if len(c)>=2:
            m=np.mean([np.mean(sd[s]) for s in c])
            if m>bm: bm,best=m,lam
    if best is None: return None
    c=sorted(set(dv)&set(per[best]))
    d=[dv[s]-np.mean(per[best][s]) for s in c]
    return float(np.mean(d))


def main():
    rows=[]
    for env in ["A","asym","gsig","pwr","sp","c100gsig","c100sp"]:
        m=env_metrics(env)
        if not m: continue
        adv=dv_advantage(env)
        m["dv_adv_vs_grid"]=round(adv,4) if adv is not None else None
        rows.append(m)
    # mobility rows (dynamic, high movement by construction; V_lambda not from grid)
    with open(JR/"tables/preferred_operating_point_variation.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["env","V_lambda","var_lamstar","range_lamstar",
                                     "switches","best2nd_gap","dv_adv_vs_grid","lam_star_trace"])
        for r in rows: w.writerow([r["env"],r["V_lambda"],r["var_lamstar"],r["range_lamstar"],
                                   r["switches"],r["best2nd_gap"],r["dv_adv_vs_grid"],
                                   "".join(str(int(x*10)) for x in r["lam_star_trace"])])
    print(f"{'env':10}{'V_lam':>7}{'range':>7}{'switch':>7}{'best2nd':>9}{'DVadv(pp)':>10}")
    xs,ys=[],[]
    for r in rows:
        a=r["dv_adv_vs_grid"]
        print(f"{r['env']:10}{r['V_lambda']:>7}{r['range_lamstar']:>7}{r['switches']:>7}"
              f"{r['best2nd_gap']:>9}{(a*100 if a is not None else float('nan')):>10.2f}")
        if a is not None: xs.append(r["V_lambda"]); ys.append(a)
    # correlation
    def spearman(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1]) if len(x)>2 else float('nan')
    print(f"\n  Spearman(V_lambda, DV-adv) = {spearman(xs,ys):+.3f}  (n={len(xs)} envs)")
    print(f"  Pearson (V_lambda, DV-adv) = {np.corrcoef(xs,ys)[0,1]:+.3f}")
    print("  lam_star_trace digits = round(10*lambda*) per eval window")


if __name__ == "__main__":
    main()
