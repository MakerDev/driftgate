"""F2 — DriftGate system schematic (vector). Shows the code-verified structure:
input -> client block -> {client exit p_c} and {feature -> server block -> server exit p_s};
p_c,p_s -> per-client TV -> cluster signal -> three comparisons -> lambda,Lambda mixing.
Data/prediction flow = solid; parameter-sharing flow = dashed. No label/rho into controller.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans","font.size":8})
MM=1/25.4
FIG=Path(__file__).resolve().parent.parent/"figures"; FIG.mkdir(parents=True,exist_ok=True)
BLUE="#0072B2"; GREEN="#009E73"; GREY="#4D4D4D"; ORANGE="#D55E00"; LIL="#CDE3F0"; LGRN="#CDEBDF"

fig,ax=plt.subplots(figsize=(181*MM,96*MM)); ax.set_xlim(0,100); ax.set_ylim(0,54); ax.axis("off")
def box(x,y,w,h,fc,ec,txt,fs=7.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",
                 fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,c=GREY,ls="-",lw=1.2,rad=0.0,txt=None,tx=0,ty=0,fs=6.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,
                 color=c,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((x1+x2)/2+tx,(y1+y2)/2+ty,txt,ha="center",va="center",fontsize=fs,color=c)

# ---- ROW 1: on-device dual-exit data path (top) ----
ax.text(2,52,"(a) On-device dual-exit inference (one client)",fontsize=8.5,fontweight="bold")
box(2,42,12,7,"#F2F2F2",GREY,"Unlabeled\nrequest $x$")
box(19,42,13,7,LIL,BLUE,"Client\nblock")
box(37,47.5,13,5.2,LIL,BLUE,"Client exit\n$p_c(x)$",7)
box(37,40.3,13,5.2,LGRN,GREEN,"Server block\n+ exit  $p_s(x)$",7)
box(56,43.2,11,5.5,"#FBEEE6",ORANGE,r"$D_{TV}=\frac{1}{2}\sum_j|p_c-p_s|$",6.6)
arrow(14,45.5,19,45.5)
arrow(32,45.5,37,50.1,rad=-0.15)                     # client block -> client exit
arrow(32,45.5,37,42.9,rad=0.15,c=GREEN)              # feature rep -> server block
ax.text(34.5,47.6,"feature",fontsize=6,color=GREEN,rotation=18)
arrow(50,50.1,56,47.2,rad=-0.1,c=BLUE)
arrow(50,42.9,56,44.9,rad=0.1,c=GREEN)

# ---- ROW 2: two serving clusters + overlapping client + controller ----
ax.text(2,35,"(b) Cluster-level control & model sharing",fontsize=8.5,fontweight="bold")
# clusters
ax.add_patch(FancyBboxPatch((3,10),27,22,boxstyle="round,pad=0.4,rounding_size=1.5",
             fc="#F7FAFC",ec=BLUE,lw=1.3,ls="--"))
ax.text(16.5,30,"Serving cluster $z_1$",fontsize=7.5,color=BLUE,ha="center",fontweight="bold")
ax.add_patch(FancyBboxPatch((40,10),27,22,boxstyle="round,pad=0.4,rounding_size=1.5",
             fc="#F7FAFC",ec=GREEN,lw=1.3,ls="--"))
ax.text(53.5,30,"Serving cluster $z_2$",fontsize=7.5,color=GREEN,ha="center",fontweight="bold")
for cx,lab in [(8,"c1"),(16,"c2")]: box(cx-3,20,6,5,LIL,BLUE,lab,7)
for cx,lab in [(50,"c4"),(58,"c5")]: box(cx-3,20,6,5,LGRN,GREEN,lab,7)
box(31.5,20,6,5,"#EFE7F7","#8E6FC0","c3",7)          # overlapping client
ax.text(34.5,26.4,"overlapping\nclient",fontsize=5.8,ha="center",color="#8E6FC0")
arrow(31.5,22.5,30,22.5,c="#8E6FC0",rad=0); arrow(37.5,22.5,40,22.5,c="#8E6FC0",rad=0)
# per-client TV -> cluster signal
box(8,13,20,4.5,"#FBEEE6",ORANGE,r"cluster TV signal $d_{z_1}$",6.6)
box(45,13,20,4.5,"#FBEEE6",ORANGE,r"cluster TV signal $d_{z_2}$",6.6)
arrow(13,20,15,17.6,c=ORANGE,rad=0); arrow(53,20,53,17.6,c=ORANGE,rad=0)
# controller
box(72,17,26,13,"#FFF7E6","#B8860B",
    "DriftGate controller\n(per cluster $z$)\n\ntemporal · spatial · absolute\n"
    r"$\lambda_z=\min(\lambda_{rel},\lambda_{abs})$",6.4)
arrow(28,15.2,72,20,c=ORANGE,rad=-0.12,txt="scalar",tx=-3,ty=2.2)
arrow(65,15.2,72,19,c=ORANGE,rad=0.12)
# lambda/Lambda mixing outputs
box(72,3,26,10,"#EAF3FB",BLUE,
    r"$\lambda$: local $\leftrightarrow$ cluster model"+"\n"+
    r"$\Lambda$: cluster $\leftrightarrow$ global",6.6)
arrow(85,17,85,13,c=BLUE)
# sharing arrows (dashed) back to clusters
arrow(72,6,30,8,c=BLUE,ls="--",rad=0.10,txt="weight sharing",tx=4,ty=-2,fs=6)
arrow(72,5,67,8,c=GREEN,ls="--",rad=-0.10)

# legend
ax.plot([2,7],[1.5,1.5],color=GREY,lw=1.3); ax.text(8,1.5,"data / prediction & signal flow",va="center",fontsize=6.3)
ax.plot([44,49],[1.5,1.5],color=BLUE,lw=1.3,ls="--"); ax.text(50,1.5,"parameter-sharing flow (no labels / $\\rho$ enter controller)",va="center",fontsize=6.3)

fig.savefig(FIG/"fig02_system_overview.pdf",bbox_inches="tight")
fig.savefig(FIG/"fig02_system_overview.png",dpi=300,bbox_inches="tight")
plt.close(fig); print("saved fig02_system_overview")
