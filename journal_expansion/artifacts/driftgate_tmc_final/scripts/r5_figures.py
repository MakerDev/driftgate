"""Round 5 — the five final figures (vector PDF with embedded TrueType fonts + 300 dpi PNG).
Reads tables/*.csv written by r5_tables.py plus the passive rec_* runs (raw JSON) for Fig. 1.
Names in figures: Schedule A -> stepwise composition change, mobility-med -> client mobility,
post-convergence abrupt -> late abrupt change, relonly -> DriftGate. The absolute branch is not drawn.
Palette: DriftGate #0072B2, entropy #D55E00 (validated, CVD ΔE 21.9); fixed λ values use one
ordered gray ramp (contrast >= 3:1 on white) plus distinct markers and a legend.
"""
import csv, glob, json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "grid.linewidth": 0.4, "grid.alpha": 0.35})
MM = 1 / 25.4
C_DG, C_ENT, C_RHO, C_WARM = "#0072B2", "#D55E00", "#BBBBBB", "0.92"
# one gray per λ value (same in every panel), ordered light -> dark, all >= 3:1 contrast on white
GRAY = {"fixed 0.2": "#8C8C8C", "fixed 0.3": "#737373", "fixed 0.4": "#595959", "fixed 0.5": "#404040",
        "fixed 0.6": "#1F1F1F"}
MK = {"DriftGate": "o", "entropy": "s", "fixed 0.2": "v", "fixed 0.3": "^", "fixed 0.4": "D", "fixed 0.5": "P", "fixed 0.6": "X"}
HERE = Path(__file__).resolve().parent.parent
TAB, FIG = HERE / "tables", HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
RUNS = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs")  # [SERVER-PATH:REPO_ROOT]
def rd(n): return list(csv.DictReader(open(TAB / n)))
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved", name)
def lab(m): return m.replace("fixed ", "fixed λ=")

# ---------------------------------------------------------------- Fig 1 signal dynamics
def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(181 * MM, 88 * MM), height_ratios=[1, 2.2], sharex="col")
    for j, (sch, title) in enumerate([("A", "(a) stepwise composition change, 100-round variant"),
                                      ("abrupt", "(b) late abrupt change")]):
        hs = [json.load(open(f)) for f in sorted(glob.glob(str(RUNS / f"signal_benchmark/rec_{sch}_s*.json")))]
        ser = lambda h, k: np.array([np.mean([float(v) for v in d.values()]) for d in h["signals_per_es"][k]])
        tv = np.array([ser(h, "tv_dist") for h in hs]); en = np.array([ser(h, "ent_client") for h in hs]) / math.log(10)
        rho = np.array([r if not isinstance(r, dict) else np.mean(list(r.values())) for r in hs[0]["rho_trace"]])
        x = np.arange(1, len(rho) + 1)
        axr, axs = axes[0, j], axes[1, j]
        axr.fill_between(x, rho, color=C_RHO, alpha=0.7, lw=0); axr.set_ylim(0, 0.9); axr.set_ylabel("ρ")
        axr.set_title(title, fontsize=9, loc="left")
        for ax in (axr, axs): ax.axvspan(1, 15, color=C_WARM, lw=0)
        for cr in [i + 1 for i in range(1, len(rho)) if abs(rho[i] - rho[i - 1]) > 1e-9]:
            axr.axvline(cr, color="0.55", lw=0.6, ls="--"); axs.axvline(cr, color="0.55", lw=0.6, ls="--")
        for arr, col, name in ((tv, C_DG, "client-server TV"), (en, C_ENT, "predictive entropy H/ln C")):
            m, sd = arr.mean(0), arr.std(0, ddof=1)
            axs.plot(x, m, color=col, label=name); axs.fill_between(x, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        axs.set_xlabel("Round"); axs.grid(True, axis="y"); axs.set_ylim(0, 1.05)
        if j == 0: axs.set_ylabel("Signal value")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.6, h_pad=0.5); save(fig, "fig1_signal_dynamics")

# ---------------------------------------------------------------- Fig 2 system overview
def fig2():
    BLUE, GREEN, GREY, ORANGE, LIL, LGRN, PUR, GOLD = "#0072B2", "#009E73", "#4D4D4D", "#D55E00", "#CDE3F0", "#CDEBDF", "#8E6FC0", "#B8860B"
    fig, ax = plt.subplots(figsize=(181 * MM, 102 * MM)); ax.set_xlim(0, 100); ax.set_ylim(-1.5, 56); ax.axis("off")
    def box(x, y, w, h, fc, ec, txt, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
    def arrow(p, q, c=GREY, ls="-", rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=c, lw=1.2, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    ax.text(2, 54.3, "(a) Dual-exit inference (one client)", fontsize=8.5, fontweight="bold")
    box(2, 43, 12, 7, "#F2F2F2", GREY, "Unlabeled\nrequest $x$")
    box(19, 43, 13, 7, LIL, BLUE, "Client\nblock")
    box(38, 47.5, 13, 5.2, LIL, BLUE, "Client exit\n$p_c(x)$", 7)
    box(38, 39.6, 13, 5.2, LGRN, GREEN, "Server block\n+ exit  $p_s(x)$", 7)
    box(57, 43.3, 18, 5.5, "#FBEEE6", ORANGE, r"$D_{TV}=\frac{1}{2}\sum_j|p_c(j)-p_s(j)|$", 6.0)
    arrow((14, 46.5), (19, 46.5)); arrow((32, 46.5), (38, 50.1), c=BLUE, rad=-0.15); arrow((32, 46.5), (38, 42.2), c=GREEN, rad=0.15)
    ax.text(32.6, 43.0, "feature", fontsize=6, color=GREEN, rotation=-32)
    arrow((51, 50.1), (57, 47.3), c=BLUE, rad=-0.1); arrow((51, 42.2), (57, 45.0), c=GREEN, rad=0.1)
    ax.text(70.5, 40.6, "mean over 64 recent unlabeled requests", fontsize=5.8, ha="center", va="center", color="0.3")
    ax.text(2, 36.5, "(b) Per-cluster control and model sharing", fontsize=8.5, fontweight="bold")
    for cx, col, sub, cl in ((3, BLUE, "1", ("c1", "c2")), (40, GREEN, "2", ("c4", "c5"))):
        ax.add_patch(FancyBboxPatch((cx, 8), 27, 26, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#F7FAFC", ec=col, lw=1.3, ls="--"))
        ax.text(cx + 13.5, 32.3, f"Serving cluster $z_{sub}$", fontsize=7.5, color=col, ha="center", fontweight="bold")
        for k, nm in enumerate(cl): box(cx + 3 + 8 * k, 24, 6, 5, LIL if col == BLUE else LGRN, col, nm, 7)
        box(cx + 2, 17.5, 23, 4.0, "#FBEEE6", ORANGE, f"cluster TV signal $\\bar d_{{z_{sub}}}$", 6.4)
        box(cx + 2, 9.0, 23, 6.4, "#FFF7E6", GOLD, f"controller of $z_{sub}$: temporal and\nspatial comparison → $\\lambda_{{z_{sub}}}, \\Lambda_{{z_{sub}}}$", 6.0)
        arrow((cx + 8, 24), (cx + 12, 21.6), c=ORANGE); arrow((cx + 13.5, 17.5), (cx + 13.5, 15.6), c=ORANGE)
    box(31.5, 24, 6, 5, "#EFE7F7", PUR, "c3", 7); ax.text(34.5, 29.6, "in two\nclusters", fontsize=5.6, ha="center", va="bottom", color=PUR)
    arrow((31.5, 26.5), (30.2, 26.5), c=PUR); arrow((37.5, 26.5), (39.8, 26.5), c=PUR)
    arrow((28, 12.2), (42, 12.2), c=ORANGE, ls=":", style="<|-|>")
    ax.text(35, 14.0, "scalars\nbetween\nedges", fontsize=5.6, color=ORANGE, ha="center", va="bottom")
    box(72, 17, 26, 9.5, "#EAF3FB", BLUE, r"$\lambda_z$: local $\leftrightarrow$ cluster model" + "\n" + r"$\Lambda_z$: cluster $\leftrightarrow$ global model", 6.6)
    arrow((85, 17), (85, 5.5), c=BLUE, ls="--")
    ax.plot([16.5, 85], [5.5, 5.5], color=BLUE, lw=1.2, ls="--"); arrow((16.5, 5.5), (16.5, 8), c=BLUE, ls="--"); arrow((53.5, 5.5), (53.5, 8), c=GREEN, ls="--")
    ax.text(50, 3.6, "model mixing and sharing", fontsize=6, color=BLUE, ha="center", va="center")
    ax.plot([2, 7], [0.3, 0.3], color=GREY, lw=1.3); ax.text(8, 0.3, "prediction and signal flow", va="center", fontsize=6.3)
    ax.plot([40, 45], [0.3, 0.3], color=BLUE, lw=1.3, ls="--"); ax.text(46, 0.3, "parameter sharing (no labels and no ρ enter any controller)", va="center", fontsize=6.3)
    save(fig, "fig2_system_overview")

# ---------------------------------------------------------------- Fig 3 λ trajectories
def fig3():
    rows = rd("lambda_trajectories.csv")
    fig, axes = plt.subplots(2, 2, figsize=(181 * MM, 88 * MM), height_ratios=[1, 1.7], sharex="col")
    for j, st in enumerate(["stepwise composition change", "client mobility"]):
        dg = [r for r in rows if r["setting"] == st and r["method"] == "DriftGate"]
        en = [r for r in rows if r["setting"] == st and r["method"] == "entropy"]
        x = [int(r["round"]) for r in dg]; rho = [float(r["rho"]) for r in dg]
        a0 = axes[0, j]; a0.fill_between(x, rho, color=C_RHO, alpha=0.7, lw=0); a0.set_ylim(0, 0.9); a0.set_ylabel("ρ")
        a0.set_title(f"({'ab'[j]}) {st}", fontsize=9, loc="left"); a0.axvspan(1, 25, color=C_WARM, lw=0)
        a1 = axes[1, j]; a1.axvspan(1, 25, color=C_WARM, lw=0)
        for rr, col, name in ((dg, C_DG, f"DriftGate ({dg[0]['n_seeds']} seeds)"), (en, C_ENT, f"entropy ({en[0]['n_seeds']} seeds)")):
            m = np.array([float(r["lambda_mean"]) for r in rr]); sd = np.array([float(r["lambda_sd_over_seeds"]) for r in rr])
            a1.plot(x, m, color=col, label=name); a1.fill_between(x, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        a1.axhline(0.4, color=GRAY["fixed 0.4"], ls=":", lw=1.2, label="fixed λ=0.4")
        a1.axhline(0.2, color=GRAY["fixed 0.2"], ls=(0, (1.5, 2.5)), lw=1.2, label="fixed λ=0.2")
        a1.set_ylim(0.1, 0.72); a1.set_ylabel("Personalization weight λ"); a1.set_xlabel("Round"); a1.grid(True, axis="y")
    h, l = axes[1, 0].get_legend_handles_labels()
    l = [x.replace(" (5 seeds)", "").replace(" (3 seeds)", "") for x in l]
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=1.6, h_pad=0.5); save(fig, "fig3_lambda_trajectories")

# ---------------------------------------------------------------- Fig 4 segment accuracy
def fig4():
    rows = rd("T2a_segment_accuracy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(181 * MM, 66 * MM), width_ratios=[5, 2.2])
    for ax, st in zip(axes, ["stepwise composition change", "client mobility"]):
        rr = [r for r in rows if r["setting"] == st]
        segs = list(dict.fromkeys(r["segment"] for r in rr))
        methods = ["DriftGate", "entropy"] + sorted({r["method"] for r in rr if r["method"].startswith("fixed")})
        k = len(methods); w = 0.8 / k
        for i, m in enumerate(methods):
            d = {r["segment"]: r for r in rr if r["method"] == m}
            xs = np.arange(len(segs)) + (i - (k - 1) / 2) * w
            ys = [float(d[s]["acc_total_pct"]) for s in segs]; es = [float(d[s]["sd_pct"] or 0) for s in segs]
            col = C_DG if m == "DriftGate" else C_ENT if m == "entropy" else GRAY[m]
            ax.errorbar(xs, ys, yerr=es, fmt=MK[m], color=col, ms=4.2, capsize=1.6, lw=0.9, label=lab(m))
        ax.set_xticks(range(len(segs))); ax.set_xticklabels([s.replace(" (", "\n(") for s in segs], fontsize=7.5)
        ax.set_ylabel("Accuracy (%)"); ax.set_title(f"({'ab'[list(axes).index(ax)]}) {st}", fontsize=9, loc="left"); ax.grid(True, axis="y")
        ax.set_xlabel("segment of evaluation rounds")
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()): hl.setdefault(l, h)
    order = ["DriftGate", "entropy"] + sorted(l for l in hl if l.startswith("fixed"))
    fig.legend([hl[l] for l in order], order, loc="upper center", ncol=len(order), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.06), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=1.8); save(fig, "fig4_segment_accuracy")

# ---------------------------------------------------------------- Fig 5 role experiment (single column)
def fig5():
    rows = rd("T7_role_experiment.csv")[::-1]
    fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM)); ax.axvline(0, color="0.6", lw=0.8)
    for i, r in enumerate(rows):
        m, sd = float(r["tv_rho_spearman_mean"]), float(r["sd"] or 0); pts = [float(v) for v in r["per_seed"].split()]
        col = C_DG if r["condition"] == "standard" else "#6B6B6B"
        ax.scatter(pts, [i] * len(pts), s=12, facecolors="none", edgecolors=col, lw=0.8, zorder=2)
        ax.errorbar(m, i, xerr=sd, fmt="o", color=col, ms=5, capsize=2.5, lw=1.2, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["condition"].replace(" independent", "\nindependent") for r in rows])
    ax.set_xlabel("Spearman correlation of TV with ρ"); ax.set_xlim(-0.65, 1.05); ax.grid(True, axis="x")
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.2, top=0.97); save(fig, "fig5_role_experiment")

for fn in (fig1, fig2, fig3, fig4, fig5):
    fn()
print("figures ->", FIG)
