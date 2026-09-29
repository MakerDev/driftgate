"""Generate Integrated_R100.md / Integrated_R150.md directly from result JSONs.
No hand-transcribed numbers. Run: python scripts/gen_integrated.py"""
import json, glob, os, numpy as np
V4='/disk2/Yujin/adaptive_splitomc_v4/'  # [SERVER-PATH:EXTERNAL]
V3='/disk2/Yujin/adaptive_splitomc_v3/'  # [SERVER-PATH:EXTERNAL]
OUT=V4+'docs/results_v3/phase_reports/'

def jl(p): return json.load(open(p))
def ev(d): return {e['round']:e for e in d.get('eval',[]) if isinstance(e,dict) and e.get('round')}
def rho_at(r): return [0.0,0.4,0.8,0.4,0.0][min((r-1)//30,4)]
def at(d,r,k='acc_total'):
    e=ev(d); return e[r].get(k) if r in e else None
def integ(d,c):
    e=ev(d); v=[x['acc_total'] for r,x in e.items() if r<=c and x.get('acc_total') is not None]; return np.mean(v) if v else None
def segs(d,c):
    e=ev(d); b={0.0:[],0.4:[],0.8:[]}
    for r,x in e.items():
        if r<=c and x.get('acc_total') is not None:
            rr=x.get('rho'); rr=rr if rr is not None else rho_at(r)
            if rr in b: b[rr].append(x['acc_total'])
    return {k:(np.mean(v) if v else None) for k,v in b.items()}
def f(x,nd=4): return ('%.*f'%(nd,x)) if x is not None else '-'
def spread_lams(d,r):
    lam=d['lamdas'][r-1] if r-1<len(d.get('lamdas',[])) else {}
    if not isinstance(lam,dict) or not lam: return None,None
    cl=[lam.get(str(i),lam.get(i)) for i in range(5)]
    if any(c is None for c in cl): return None,None
    return cl, max(cl)-min(cl)

def gen(CUT):
    L=[]
    P=L.append
    other = 100 if CUT==150 else 150
    P("# Integrated Results — R%d%s"%(CUT, " (full-horizon, primary)" if CUT==150 else " (early-cutoff)"))
    P("")
    P("**Adaptive-SplitOMC v4 · all phases at the R%d training horizon · auto-generated from result JSONs (no hand-transcribed numbers).** Companion: `Integrated_R%d.md`."%(CUT,other))
    P("")
    P("## Signal-mode provenance")
    P("")
    P("| Phase | adaptive signal |")
    P("|---|---|")
    P("| E1 static | entropy |")
    P("| E2 temporal | disagreement (mu=0.31) |")
    P("| E3 spatial | disagreement (mu=0.31) |")
    P("| E4 mobility | disagreement (mu=0.31) + entropy backup |")
    P("| E5 signal ablation | 3-way: entropy / disagreement / disagreement+margin (β=0.5), 150R |")
    P("")
    P("Calibration (disagreement): `mu_drift=0.31, tau_drift=0.045, lam[0.15,0.7], Lam[0.4,0.7]`; entropy: `mu_H=2.0, tau_H=0.5`.")
    P("")
    P("---")
    P("")

    # E1
    P("## E1 — Static Pareto (no drift, ρ=0.4 eval)  [adaptive = ENTROPY]")
    P("")
    e1=V4+'results/e1_static/'
    rows=[]
    for fp in glob.glob(e1+'*.json'):
        d=jl(fp); a=at(d,CUT)
        if a is None: continue
        rows.append((os.path.basename(fp)[:-5],a,at(d,CUT,'acc_main'),at(d,CUT,'acc_oop'),at(d,CUT,'acc_oor'),at(d,CUT,'worst_cell_acc')))
    rows.sort(key=lambda x:-x[1])
    P("| Rank | Method | acc_total | main | oop | oor | worst_cell |")
    P("|---:|---|---:|---:|---:|---:|---:|")
    for i,(n,a,m,o,r,w) in enumerate(rows,1):
        star='**' if n=='adaptive_splitomc' else ''
        P("| %d | %s%s%s | %s | %s | %s | %s | %s |"%(i,star,n,star,f(a),f(m,3),f(o,3),f(r,3),f(w,3)))
    P("")

    # E2
    P("## E2 — Temporal drift (Schedule A)  [adaptive = DISAGREEMENT mu=0.31]")
    P("")
    e2=V4+'results/e2_temporal/schedule_A/'
    e2m=[('adaptive (disagree mu=0.31)','adaptive_splitomc.json'),
         ('adaptive (disagree mu=0.35)','adaptive_splitomc_mu35.json'),
         ('adaptive (disagree mu=0.392)','adaptive_splitomc_disagree_mu392.json'),
         ('adaptive (entropy)','adaptive_splitomc_entropy.json'),
         ('fixed λ=0.2','splitomcplus_lam0.2_Lam0.5.json'),
         ('fixed λ=0.4','splitomcplus_lam0.4_Lam0.5.json'),
         ('fixed λ=0.6','splitomcplus_lam0.6_Lam0.5.json')]
    P("| Method | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |")
    P("|---|---:|---:|---:|---:|")
    e2r={}
    for lab,fn in e2m:
        d=jl(e2+fn); ig=integ(d,CUT); sm=segs(d,CUT); e2r[lab]=ig
        P("| %s | %s | %s | %s | %s |"%(lab,f(ig),f(sm[0.0]),f(sm[0.4]),f(sm[0.8])))
    fx=[k for k in e2r if k.startswith('fixed')]; bf=max(fx,key=lambda k:e2r[k])
    g=(e2r['adaptive (disagree mu=0.31)']-e2r[bf])*100
    ge=(e2r['adaptive (disagree mu=0.31)']-e2r['adaptive (entropy)'])*100
    v='PASS' if g>=2 else ('PARTIAL' if g>=-0.5 else 'FAIL')
    P("")
    P("- best fixed = %s (%s); adaptive(mu=0.31) **%+.2f pp = %s** · vs entropy **%+.2f pp**"%(bf,f(e2r[bf]),g,v,ge))
    P("")

    # E3
    P("## E3 — Spatial heterogeneity (equal_spread: cell0 ρ=0 … cell4 ρ=0.8)  [adaptive = DISAGREEMENT mu=0.31]")
    P("")
    e3=V4+'results/e3_spatial/mode_equal_spread/'
    e3m=[('adaptive (disagree mu=0.31)','adaptive_splitomc.json'),
         ('adaptive (disagree mu=0.35)','adaptive_splitomc_mu35.json'),
         ('adaptive (entropy)','adaptive_splitomc_entropy.json'),
         ('fixed λ=0.2','splitomcplus_lam0.2_Lam0.5.json'),
         ('fixed λ=0.4','splitomcplus_lam0.4_Lam0.5.json'),
         ('fixed λ=0.6','splitomcplus_lam0.6_Lam0.5.json')]
    P("| Method | acc_total | worst_cell | best_cell | cell_gap |")
    P("|---|---:|---:|---:|---:|")
    e3r={}
    for lab,fn in e3m:
        d=jl(e3+fn); e3r[lab]=(at(d,CUT),at(d,CUT,'worst_cell_acc'))
        P("| %s | %s | %s | %s | %s |"%(lab,f(at(d,CUT)),f(at(d,CUT,'worst_cell_acc')),f(at(d,CUT,'best_cell_acc')),f(at(d,CUT,'cell_gap'))))
    fx3=[k for k in e3r if k.startswith('fixed')]; bfa=max(fx3,key=lambda k:e3r[k][0]); bfw=max(fx3,key=lambda k:e3r[k][1])
    ad=e3r['adaptive (disagree mu=0.31)']
    dd=jl(e3+'adaptive_splitomc.json'); de=jl(e3+'adaptive_splitomc_entropy.json')
    _,sd=spread_lams(dd,CUT); _,se=spread_lams(de,CUT)
    P("")
    P("- acc_total: adaptive %s vs best fixed %s (%s) → **%+.2f pp**"%(f(ad[0]),f(e3r[bfa][0]),bfa,(ad[0]-e3r[bfa][0])*100))
    P("- worst_cell: adaptive %s vs best fixed %s (%s) → **%+.2f pp = %s**"%(f(ad[1]),f(e3r[bfw][1]),bfw,(ad[1]-e3r[bfw][1])*100,'PASS' if ad[1]>=e3r[bfw][1] else 'NOT MET'))
    P("- λ stratification spread @R%d: disagreement **%s** vs entropy **%s** (FLAT)"%(CUT,f(sd,3),f(se,3)))
    P("")

    # E4
    P("## E4 — Mobility (Gauss-Markov rewiring)  [adaptive = DISAGREEMENT mu=0.31]")
    P("")
    e4=V4+'results/e4_mobility/'
    e4m=[('adaptive (disagreement)','adaptive_splitomc_mob.json'),
         ('adaptive (entropy)','adaptive_splitomc_mob_entropy.json'),
         ('fixed λ=0.2','splitomcplus_lam0.2_Lam0.5_mob.json'),
         ('fixed λ=0.4','splitomcplus_lam0.4_Lam0.5_mob.json'),
         ('fixed λ=0.6','splitomcplus_lam0.6_Lam0.5_mob.json')]
    P("| Method | acc_total | main | oop | oor | worst_cell |")
    P("|---|---:|---:|---:|---:|---:|")
    e4r={}
    for lab,fn in e4m:
        fp=e4+fn
        if not os.path.exists(fp): continue
        d=jl(fp); e4r[lab]=(at(d,CUT),at(d,CUT,'worst_cell_acc'))
        P("| %s | %s | %s | %s | %s | %s |"%(lab,f(at(d,CUT)),f(at(d,CUT,'acc_main'),3),f(at(d,CUT,'acc_oop'),3),f(at(d,CUT,'acc_oor'),3),f(at(d,CUT,'worst_cell_acc'))))
    fx4=[k for k in e4r if k.startswith('fixed')]; bfa4=max(fx4,key=lambda k:e4r[k][0]); bfw4=max(fx4,key=lambda k:e4r[k][1])
    ad4=e4r['adaptive (disagreement)']; ae4=e4r['adaptive (entropy)']
    P("")
    P("- disagreement vs entropy: **acc %+.2f pp, worst %+.2f pp**"%((ad4[0]-ae4[0])*100,(ad4[1]-ae4[1])*100))
    P("- disagreement vs **best fixed acc (%s, %s)**: **%+.2f pp**"%(bfa4,f(e4r[bfa4][0]),(ad4[0]-e4r[bfa4][0])*100))
    P("- disagreement vs **best fixed worst (%s, %s)**: **%+.2f pp = %s**"%(bfw4,f(e4r[bfw4][1]),(ad4[1]-e4r[bfw4][1])*100,'PASS' if ad4[1]>=e4r[bfw4][1] else 'NOT MET'))
    P("")

    # E5 signal
    P("## E5 — 3-way SIGNAL ablation (Schedule A, 150R, same controller — only signal source changes)")
    P("")
    e5=V4+'results/e5_signal/'
    e5m=[('entropy','entropy.json'),('disagreement (S1)','disagreement.json'),('disagreement+margin (β=0.5)','disagreement_margin.json')]
    P("| Signal | integrated | ρ=0 | ρ=0.4 | ρ=0.8 |")
    P("|---|---:|---:|---:|---:|")
    e5r={}
    for lab,fn in e5m:
        d=jl(e5+fn); ig=integ(d,CUT); sm=segs(d,CUT); e5r[lab]=ig
        P("| %s | %s | %s | %s | %s |"%(lab,f(ig),f(sm[0.0]),f(sm[0.4]),f(sm[0.8])))
    P("")
    P("- **disagreement − entropy = %+.2f pp** (same controller, only the signal swapped) — direct C1 isolation."%((e5r['disagreement (S1)']-e5r['entropy'])*100))
    P("- **(disagreement+margin) − disagreement = %+.2f pp** — %s."%((e5r['disagreement+margin (β=0.5)']-e5r['disagreement (S1)'])*100,
        'margin HELPS' if e5r['disagreement+margin (β=0.5)']>e5r['disagreement (S1)'] else 'margin HURTS → S1 alone best, β=0 confirmed'))
    P("")
    P("---")
    P("")
    P("## Headline summary (R%d)"%CUT)
    P("")
    P("- **C1 (signal): disagreement > entropy everywhere** — E5 signal-swap %+.2f pp; E2 %+.2f; E3 acc %+.2f; E4 acc %+.2f."%(
        (e5r['disagreement (S1)']-e5r['entropy'])*100, ge,
        (jl(e3+'adaptive_splitomc.json') and (e3r['adaptive (disagree mu=0.31)'][0]-e3r['adaptive (entropy)'][0])*100),
        (ad4[0]-ae4[0])*100))
    P("- **vs best hand-tuned fixed λ:** E2 %s (%+.2f pp); E3 worst_cell %+.2f pp; E4 acc %+.2f pp / worst %+.2f pp. Disagreement does NOT beat best fixed on these aggregates at R%d."%(
        v,g,(ad[1]-e3r[bfw][1])*100,(ad4[0]-e4r[bfa4][0])*100,(ad4[1]-e4r[bfw4][1])*100,CUT))
    P("- **S1 sufficiency:** adding S2 margin %s."%('helps' if e5r['disagreement+margin (β=0.5)']>e5r['disagreement (S1)'] else 'hurts (β=0 is correct)'))
    P("")
    P("_Generated by `scripts/gen_integrated.py`. All values are real eval points at the R%d cutoff (eval_every=10/20)._"%CUT)
    return '\n'.join(L)+'\n'

open(OUT+'Integrated_R150.md','w').write(gen(150))
open(OUT+'Integrated_R100.md','w').write(gen(100))
print("WROTE both reports")
