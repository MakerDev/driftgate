"""DriftGate TMC — publication figures from tidy CSVs in ../data.
Vector PDF (fonttype 42) + 300dpi PNG. Column 89mm, double 181mm.
Colors fixed: TV #0072B2, entropy #D55E00, fixed0.4 #4D4D4D, fixed0.2 #999999, server #009E73.
Real recorded rounds get markers; connecting lines are visual only (no smoothing).
Run: python make_figures.py
"""
import csv, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans","font.size":8,
    "axes.labelsize":9,"axes.titlesize":9,"xtick.labelsize":8,"ytick.labelsize":8,
    "legend.fontsize":7.5,"axes.linewidth":0.8,"lines.linewidth":1.4,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,
    "grid.linewidth":0.4,"grid.alpha":0.35,
})
C={"TV":"#0072B2","entropy":"#D55E00","fixed l0.4":"#4D4D4D","fixed l0.2":"#999999","server":"#009E73","rho":"#BBBBBB"}
MM=1/25.4
DATA=Path(__file__).resolve().parent.parent/"data"
FIG=Path(__file__).resolve().parent.parent/"figures"; FIG.mkdir(parents=True,exist_ok=True)
def rd(name):
    return list(csv.DictReader(open(DATA/name)))
def parse_ci(ci):
    return [float(x) for x in re.split(r'[\s,]+', ci.strip('[]').strip())]
def save(fig,name):
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig); print("  saved",name)

# ---------------- F1: signal dynamics ----------------
def f1():
    rows=rd("f1_signal_dynamics.csv")
    scheds=["Schedule A","post-convergence abrupt"]
    fig,axes=plt.subplots(2,2,figsize=(181*MM,88*MM),height_ratios=[1,2.2],sharex="col")
    for j,sch in enumerate(scheds):
        rr=[r for r in rows if r["schedule"]==sch]
        x=[int(r["round"]) for r in rr]; rho=[float(r["rho"]) for r in rr]
        tv=np.array([float(r["tv_mean"]) for r in rr]); tvs=np.array([float(r["tv_sd"]) for r in rr])
        en=np.array([float(r["ent_norm_mean"]) for r in rr]); ens=np.array([float(r["ent_norm_sd"]) for r in rr])
        axr=axes[0,j]; axs=axes[1,j]
        axr.fill_between(x,rho,color=C["rho"],alpha=0.6,lw=0); axr.set_ylabel(r"$\rho$")
        axr.set_ylim(0,0.9); axr.set_title(sch,fontsize=9)
        axr.axvspan(1,15,color="0.9",alpha=0.5,lw=0)
        axs.axvspan(1,15,color="0.9",alpha=0.5,lw=0)
        axs.plot(x,tv,color=C["TV"],label="client-server TV")
        axs.fill_between(x,tv-tvs,tv+tvs,color=C["TV"],alpha=0.18,lw=0)
        axs.plot(x,en,color=C["entropy"],label=r"predictive entropy $H/\ln C$")
        axs.fill_between(x,en-ens,en+ens,color=C["entropy"],alpha=0.18,lw=0)
        axs.set_xlabel("Round"); axs.grid(True,axis="y")
        if j==0:
            axs.set_ylabel("Signal level"); axs.legend(loc="upper left",frameon=False)
        axs.set_ylim(0,None)
    fig.text(0.008,0.97,"(a)",fontsize=10,fontweight="bold")
    fig.text(0.52,0.97,"(b)",fontsize=10,fontweight="bold")
    fig.tight_layout(w_pad=1.6,h_pad=0.6)
    save(fig,"fig01_signal_dynamics")

# ---------------- F3: traffic-shift -> lambda adaptation (accuracy-vs-round panel removed) ----------------
def f3():
    rho=rd("f3_rho.csv"); lam=rd("f3_traj_lambda.csv")
    settings=["Schedule A","mobility-med"]
    methods=[("TV","TV"),("entropy","entropy"),("fixed l0.4","fixed l0.4")]
    fig,axes=plt.subplots(2,2,figsize=(181*MM,88*MM),height_ratios=[1,1.7],sharex="col")
    for j,st in enumerate(settings):
        rr=[r for r in rho if r["setting"]==st]; x=[int(r["round"]) for r in rr]; y=[float(r["rho"]) for r in rr]
        ax0=axes[0,j]; ax0.fill_between(x,y,color=C["rho"],alpha=0.6,lw=0); ax0.set_ylim(0,0.9)
        ax0.set_ylabel(r"$\rho$"); ax0.set_title(st,fontsize=9); ax0.axvspan(1,15,color="0.9",alpha=0.5,lw=0)
        ax1=axes[1,j]  # lambda
        for m,lbl in methods:
            lr=[r for r in lam if r["setting"]==st and r["method"]==m]
            if not lr: continue
            xr=[int(r["round"]) for r in lr]; lm=[float(r["lambda_mean"]) for r in lr]
            if m=="fixed l0.4":
                ax1.axhline(0.4,color=C["fixed l0.4"],ls=":",lw=1.1,label="fixed λ=0.4")
            else:
                ax1.plot(xr,lm,color=C[m],label={"TV":"TV (DriftGate)","entropy":"entropy"}[m])
        ax1.set_ylabel(r"Personalization weight $\lambda$"); ax1.axvspan(1,15,color="0.9",alpha=0.5,lw=0)
        ax1.grid(True,axis="y"); ax1.set_ylim(0.1,0.72); ax1.set_xlabel("Round")
        if j==0: ax1.legend(loc="lower left",frameon=False,ncol=1)
    for k,lab in enumerate(["(a)","(b)"]):
        axes[k,0].annotate(lab,xy=(-0.36,1.02),xycoords="axes fraction",
                           fontsize=10,fontweight="bold",annotation_clip=False)
    fig.tight_layout(w_pad=1.6,h_pad=0.6)
    save(fig,"fig03_lambda_adaptation")

# ---------------- F4: mechanism & views ----------------
def f4():
    role=rd("f4_role.csv"); absc=rd("f4_absolute_contribution.csv")
    fig,axes=plt.subplots(1,2,figsize=(181*MM,72*MM))
    # (a) role dot plot
    ax=axes[0]
    labs=[r["condition"] for r in role][::-1]; vals=[float(r["tv_rho_spearman_mean"]) for r in role][::-1]
    sds=[float(r["sd"]) if r["sd"] else 0 for r in role][::-1]
    ypos=np.arange(len(labs))
    ax.axvline(0,color="0.6",lw=0.8)
    for i,(v,s) in enumerate(zip(vals,sds)):
        col=C["TV"] if "standard" in labs[i] else "#888888"
        ax.errorbar(v,i,xerr=s,fmt="o",color=col,ms=5,capsize=2.5,lw=1.2)
    ax.set_yticks(ypos); ax.set_yticklabels([l.replace(" ","\n",1) if len(l)>16 else l for l in labs])
    ax.set_xlabel(r"TV–$\rho$ Spearman correlation"); ax.set_title("(a) Server-role ablation",fontsize=9)
    ax.grid(True,axis="x"); ax.set_xlim(-0.6,1.05)
    # (d) absolute-view contribution
    ax=axes[1]
    labs=[r["setting"] for r in absc][::-1]; vals=[float(r["full_minus_relative_pp"]) for r in absc][::-1]
    cis=[r["ci95"] for r in absc][::-1]
    for i,(v,ci) in enumerate(zip(vals,cis)):
        lo,hi=parse_ci(ci)
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=C["server"],ms=5,capsize=2.5,lw=1.2)
    ax.axvline(0,color="0.6",lw=0.8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("Full − relative-only (pp)"); ax.set_title("(b) Absolute-view contribution",fontsize=9)
    ax.grid(True,axis="x"); ax.set_xlim(0,5.2)
    fig.tight_layout(w_pad=2.0)
    save(fig,"fig04_mechanism_and_views")

# ---------------- F5: timing & Lambda ----------------
def f5():
    tim=rd("f5_timing.csv"); lam=rd("f5_lambda.csv")
    fig,axes=plt.subplots(1,2,figsize=(181*MM,70*MM),width_ratios=[1.5,1])
    ax=axes[0]
    ctrls=["global mean λ","per-cluster mean λ","shuffled λ"]; setts=["Schedule A","gradual","static spatial"]
    cols={"global mean λ":"#0072B2","per-cluster mean λ":"#56B4E9","shuffled λ":"#E69F00"}
    ypos=np.arange(len(setts)); off={"global mean λ":0.22,"per-cluster mean λ":0.0,"shuffled λ":-0.22}
    for c in ctrls:
        xs=[];ys=[];xe=[]
        for i,st in enumerate(setts):
            r=[x for x in tim if x["setting"]==st and x["control"]==c and x["adaptive_minus_control_pp"]]
            if not r: continue
            v=float(r[0]["adaptive_minus_control_pp"]); lo,hi=parse_ci(r[0]["ci95"])
            ax.errorbar(v,i+off[c],xerr=[[v-lo],[hi-v]],fmt="o",color=cols[c],ms=4.5,capsize=2,lw=1.1,
                        label=c if i==0 else None)
    ax.axvline(0,color="0.6",lw=0.8); ax.set_yticks(ypos); ax.set_yticklabels(setts)
    ax.set_xlabel("Adaptive − control (pp)"); ax.set_title("(a) Weight-timing controls",fontsize=9)
    ax.grid(True,axis="x"); ax.legend(loc="lower right",frameon=False)
    ax=axes[1]
    setts=[r["setting"] for r in lam][::-1]; vals=[float(r["full_minus_fixedLambda_pp"]) for r in lam][::-1]
    cis=[r["ci95"] for r in lam][::-1]
    for i,(v,ci) in enumerate(zip(vals,cis)):
        lo,hi=parse_ci(ci)
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=C["TV"],ms=5,capsize=2.5,lw=1.2)
    ax.axvline(0,color="0.6",lw=0.8); ax.set_yticks(range(len(setts))); ax.set_yticklabels(setts)
    ax.set_xlabel("Full − (Λ=0.5) (pp)"); ax.set_title("(b) Adaptive-Λ contribution",fontsize=9)
    ax.grid(True,axis="x"); ax.set_xlim(-0.35,0.35)
    fig.tight_layout(w_pad=2.2)
    save(fig,"fig05_timing_and_lambda")

# ---------------- F6: server signal vs TV ----------------
def f6():
    rows=rd("f6_server_signal.csv")
    labs=[r["setting"].replace("CIFAR-10 Schedule A","Schedule A") for r in rows][::-1]
    vals=[float(r["rate_minus_tv_pp"]) for r in rows][::-1]; cis=[r["ci95"] for r in rows][::-1]
    ns=[r["n_seeds"] for r in rows][::-1]
    fig,ax=plt.subplots(figsize=(89*MM,78*MM))
    ax.axvline(0,color="0.5",lw=0.9)
    for i,(v,ci,n) in enumerate(zip(vals,cis,ns)):
        lo,hi=parse_ci(ci)
        col=C["server"] if v>0 else C["TV"]
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=col,ms=5,capsize=2.5,lw=1.2)
        ax.text(hi+0.12,i,f"n={n}",va="center",fontsize=6.5,color="0.4")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("Server non-Main rate − TV (pp)")
    ax.set_title("Direct server signal vs dual-exit TV",fontsize=9)
    ax.grid(True,axis="x"); ax.set_xlim(-2.2,2.0)
    leg=[Line2D([0],[0],marker="o",color=C["server"],lw=0,label="rate higher"),
         Line2D([0],[0],marker="o",color=C["TV"],lw=0,label="TV higher")]
    ax.legend(handles=leg,loc="upper left",frameon=False,bbox_to_anchor=(0.0,0.62))
    fig.tight_layout()
    save(fig,"fig06_server_signal_comparison")

for fn in (f1,f3,f4,f5,f6):
    fn()
print("figures done ->",FIG)
