"""R4 figures (A7 revisions + A1 segment figure). Reads r3 export data + r4 data.
Terminology: Schedule A -> "stepwise composition change"; mobility-med -> "client mobility";
post-convergence abrupt -> "late abrupt change". Vector PDF (fonttype 42) + 300 dpi PNG.
F4(b) is drawn ONLY when B1 results exist (data/b1_view_removal.csv); no placeholder curves.
"""
import csv, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "grid.linewidth": 0.4, "grid.alpha": 0.35})
C = {"TV": "#0072B2", "entropy": "#D55E00", "fixed l0.4": "#4D4D4D", "fixed l0.2": "#999999", "server": "#009E73", "rho": "#BBBBBB"}
MM = 1 / 25.4
HERE = Path(__file__).resolve().parent.parent
D3 = HERE.parent / "driftgate_tmc_export" / "data"      # round-3 tidy data (read-only)
D4 = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAME = {"Schedule A": "stepwise composition change", "mobility-med": "client mobility",
        "post-convergence abrupt": "late abrupt change (round 51)", "A": "stepwise composition change", "mob": "client mobility"}
def rd(p): return list(csv.DictReader(open(p)))
def ci(s): return [float(x) for x in re.split(r"[\s,]+", s.strip("[]").strip())]
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved", name)

# ---------------- F1 (A4): relabelled schedules, factual caption values in a4_f1_stats ----------------
def f1():
    rows = rd(D3 / "f1_signal_dynamics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(181 * MM, 88 * MM), height_ratios=[1, 2.2], sharex="col")
    for j, (sch, title) in enumerate([("Schedule A", "shortened stepwise schedule (100 rounds)"),
                                      ("post-convergence abrupt", "late abrupt change (round 51)")]):
        rr = [r for r in rows if r["schedule"] == sch]
        x = [int(r["round"]) for r in rr]; rho = [float(r["rho"]) for r in rr]
        tv = np.array([float(r["tv_mean"]) for r in rr]); tvs = np.array([float(r["tv_sd"]) for r in rr])
        en = np.array([float(r["ent_norm_mean"]) for r in rr]); ens = np.array([float(r["ent_norm_sd"]) for r in rr])
        axr, axs = axes[0, j], axes[1, j]
        axr.fill_between(x, rho, color=C["rho"], alpha=0.6, lw=0); axr.set_ylabel(r"$\rho$"); axr.set_ylim(0, 0.9)
        axr.set_title(title, fontsize=9); axr.axvspan(1, 15, color="0.9", alpha=0.5, lw=0)
        axs.axvspan(1, 15, color="0.9", alpha=0.5, lw=0)
        chg = [i + 1 for i in range(1, len(rho)) if abs(rho[i] - rho[i - 1]) > 1e-9]
        for cr in chg:
            axr.axvline(cr, color="0.55", lw=0.6, ls="--"); axs.axvline(cr, color="0.55", lw=0.6, ls="--")
        axs.plot(x, tv, color=C["TV"], label="client-server TV"); axs.fill_between(x, tv - tvs, tv + tvs, color=C["TV"], alpha=0.18, lw=0)
        axs.plot(x, en, color=C["entropy"], label=r"predictive entropy $H/\ln C$"); axs.fill_between(x, en - ens, en + ens, color=C["entropy"], alpha=0.18, lw=0)
        axs.set_xlabel("Round"); axs.grid(True, axis="y"); axs.set_ylim(0, None)
        if j == 0: axs.set_ylabel("Signal level"); axs.legend(loc="upper left", frameon=False)
    fig.text(0.008, 0.97, "(a)", fontsize=10, fontweight="bold"); fig.text(0.52, 0.97, "(b)", fontsize=10, fontweight="bold")
    fig.tight_layout(w_pad=1.6, h_pad=0.6); save(fig, "fig01_signal_dynamics")

# ---------------- F2: dual-exit inference + per-cluster controllers (overlap fixes) ----------------
def f2():
    BLUE, GREEN, GREY, ORANGE, LIL, LGRN, PUR = "#0072B2", "#009E73", "#4D4D4D", "#D55E00", "#CDE3F0", "#CDEBDF", "#8E6FC0"
    fig, ax = plt.subplots(figsize=(181 * MM, 100 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    def box(x, y, w, h, fc, ec, txt, fs=7.2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.1, ls=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)
    def arrow(p, q, c=GREY, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=c, lw=1.2, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    # (a) dual-exit inference — title placed above boxes; exits stacked with clearance
    ax.text(2, 54.3, "(a) Dual-exit inference (one client)", fontsize=8.5, fontweight="bold")
    box(2, 43, 12, 7, "#F2F2F2", GREY, "Unlabeled\nrequest $x$")
    box(19, 43, 13, 7, LIL, BLUE, "Client\nblock")
    box(38, 47.5, 13, 5.2, LIL, BLUE, "Client exit\n$p_c(x)$", 7)
    box(38, 39.6, 13, 5.2, LGRN, GREEN, "Server block\n+ exit  $p_s(x)$", 7)
    box(57, 43.3, 12, 5.5, "#FBEEE6", ORANGE, r"$D_{TV}=\frac{1}{2}\sum_j|p_c-p_s|$", 6.5)
    arrow((14, 46.5), (19, 46.5)); arrow((32, 46.5), (38, 50.1), c=BLUE, rad=-0.15); arrow((32, 46.5), (38, 42.2), c=GREEN, rad=0.15)
    ax.text(32.6, 43.0, "feature", fontsize=6, color=GREEN, rotation=-32)   # along the green arrow, clear of the box text
    arrow((51, 50.1), (57, 47.3), c=BLUE, rad=-0.1); arrow((51, 42.2), (57, 45.0), c=GREEN, rad=0.1)
    # (b) per-cluster control: controllers are drawn INSIDE each cluster (computed per cluster)
    ax.text(2, 36.5, "(b) Per-cluster control & model sharing", fontsize=8.5, fontweight="bold")
    for cx, col, lab, cl in ((3, BLUE, "$z_1$", ("c1", "c2")), (40, GREEN, "$z_2$", ("c4", "c5"))):
        ax.add_patch(FancyBboxPatch((cx, 8), 27, 26, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#F7FAFC", ec=col, lw=1.3, ls="--"))
        ax.text(cx + 13.5, 32.3, f"Serving cluster {lab}", fontsize=7.5, color=col, ha="center", fontweight="bold")
        for k, nm in enumerate(cl): box(cx + 3 + 8 * k, 24, 6, 5, LIL if col == BLUE else LGRN, col, nm, 7)
        box(cx + 2, 17.5, 23, 4.0, "#FBEEE6", ORANGE, f"cluster TV signal $d_{{{lab[1:-1]}}}$", 6.4)
        box(cx + 2, 9.0, 23, 6.4, "#FFF7E6", "#B8860B",
            f"controller ({lab}): temporal ·\nspatial · absolute → $\\lambda_{{{lab[1:-1]}}},\\Lambda_{{{lab[1:-1]}}}$", 6.0)
        arrow((cx + 8, 24), (cx + 12, 21.6), c=ORANGE); arrow((cx + 13.5, 17.5), (cx + 13.5, 15.6), c=ORANGE)
    box(31.5, 24, 6, 5, "#EFE7F7", PUR, "c3", 7); ax.text(34.5, 30.4, "overlapping\nclient", fontsize=5.8, ha="center", color=PUR)
    arrow((31.5, 26.5), (30.2, 26.5), c=PUR); arrow((37.5, 26.5), (39.8, 26.5), c=PUR)
    # scalar exchange between neighbouring controllers (1-step consensus): dotted, label in the gap
    ax.add_patch(FancyArrowPatch((28, 12.2), (42, 12.2), arrowstyle="<|-|>", mutation_scale=8, color=ORANGE, lw=1.1, ls=":"))
    ax.text(35, 14.0, "scalar\nexchange", fontsize=5.6, color=ORANGE, ha="center", va="bottom")
    # global aggregate + sharing: routed BELOW the clusters so no line crosses a client box
    box(72, 17, 26, 9.5, "#EAF3FB", BLUE, r"$\lambda_z$: local $\leftrightarrow$ cluster model" + "\n" + r"$\Lambda_z$: cluster $\leftrightarrow$ global aggregate", 6.6)
    arrow((85, 17), (85, 5.5), c=BLUE, ls="--")
    ax.plot([16.5, 85], [5.5, 5.5], color=BLUE, lw=1.2, ls="--"); arrow((16.5, 5.5), (16.5, 8), c=BLUE, ls="--"); arrow((53.5, 5.5), (53.5, 8), c=GREEN, ls="--")
    ax.text(70, 6.4, "weight sharing", fontsize=6, color=BLUE, ha="center")
    ax.plot([2, 7], [2.2, 2.2], color=GREY, lw=1.3); ax.text(8, 2.2, "data / prediction & signal flow", va="center", fontsize=6.3)
    ax.plot([44, 49], [2.2, 2.2], color=BLUE, lw=1.3, ls="--"); ax.text(50, 2.2, "parameter sharing  (no labels / $\\rho$ enter any controller)", va="center", fontsize=6.3)
    save(fig, "fig02_system_overview")

# ---------------- F3: rho -> lambda, with fixed 0.2 and 0.4 reference lines ----------------
def f3():
    rho = rd(D3 / "f3_rho.csv"); lam = rd(D3 / "f3_traj_lambda.csv")
    fig, axes = plt.subplots(2, 2, figsize=(181 * MM, 88 * MM), height_ratios=[1, 1.7], sharex="col")
    for j, st in enumerate(["Schedule A", "mobility-med"]):
        rr = [r for r in rho if r["setting"] == st]; x = [int(r["round"]) for r in rr]; y = [float(r["rho"]) for r in rr]
        ax0 = axes[0, j]; ax0.fill_between(x, y, color=C["rho"], alpha=0.6, lw=0); ax0.set_ylim(0, 0.9)
        ax0.set_ylabel(r"$\rho$"); ax0.set_title(NAME[st], fontsize=9); ax0.axvspan(1, 25, color="0.9", alpha=0.5, lw=0)
        ax1 = axes[1, j]
        for m, lbl in (("TV", "TV (DriftGate)"), ("entropy", "entropy controller")):
            lr = [r for r in lam if r["setting"] == st and r["method"] == m]
            ax1.plot([int(r["round"]) for r in lr], [float(r["lambda_mean"]) for r in lr], color=C[m], label=lbl)
        ax1.axhline(0.4, color=C["fixed l0.4"], ls=":", lw=1.1, label="fixed λ=0.4")
        ax1.axhline(0.2, color=C["fixed l0.2"], ls=":", lw=1.1, label="fixed λ=0.2")
        ax1.set_ylabel(r"Personalization weight $\lambda$"); ax1.axvspan(1, 25, color="0.9", alpha=0.5, lw=0)
        ax1.grid(True, axis="y"); ax1.set_ylim(0.1, 0.72); ax1.set_xlabel("Round")
        if j == 0: ax1.legend(loc="lower left", frameon=False)
    for k, lab in enumerate(["(a)", "(b)"]):
        axes[k, 0].annotate(lab, xy=(-0.36, 1.02), xycoords="axes fraction", fontsize=10, fontweight="bold", annotation_clip=False)
    fig.tight_layout(w_pad=1.6, h_pad=0.6); save(fig, "fig03_lambda_adaptation")

# ---------------- F4: (a) role ablation, (b) B1 view removal (only if present) ----------------
def f4():
    role = rd(D3 / "f4_role.csv"); b1p = D4 / "b1_view_removal.csv"
    has_b1 = b1p.exists() and len(rd(b1p)) > 0
    fig, axes = plt.subplots(1, 2 if has_b1 else 1, figsize=((181 if has_b1 else 89) * MM, 72 * MM), squeeze=False); axes = axes[0]
    ax = axes[0]; labs = [r["condition"] for r in role][::-1]; vals = [float(r["tv_rho_spearman_mean"]) for r in role][::-1]
    sds = [float(r["sd"]) if r["sd"] else 0 for r in role][::-1]; ax.axvline(0, color="0.6", lw=0.8)
    for i, (v, s) in enumerate(zip(vals, sds)):
        ax.errorbar(v, i, xerr=s, fmt="o", color=C["TV"] if "standard" in labs[i] else "#888888", ms=5, capsize=2.5, lw=1.2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l.replace(" ", "\n", 1) if len(l) > 16 else l for l in labs])
    ax.set_xlabel(r"TV–$\rho$ Spearman correlation"); ax.set_title("(a) Server-role ablation (same-pool, 3 seeds)", fontsize=8.5)
    ax.grid(True, axis="x"); ax.set_xlim(-0.6, 1.05)
    if has_b1:
        ax = axes[1]; rows = rd(b1p); labs = [f"{r['setting']}: full − {r['arm']}" for r in rows][::-1]
        vals = [float(r["full_minus_arm_pp"]) for r in rows][::-1]; cis = [ci(r["ci95"]) for r in rows][::-1]
        for i, (v, (lo, hi)) in enumerate(zip(vals, cis)):
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=C["server"], ms=5, capsize=2.5, lw=1.2)
        ax.axvline(0, color="0.6", lw=0.8); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6.8)
        ax.set_xlabel("Full controller − view-removed variant (pp)"); ax.set_title("(b) View removal, same TV signal (disjoint)", fontsize=8.5); ax.grid(True, axis="x")
    fig.tight_layout(w_pad=2.0); save(fig, "fig04_mechanism_and_views")

# ---------------- F5: (a) only, single column ----------------
def f5():
    tim = rd(D3 / "f5_timing.csv"); fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
    ctrls = ["global mean λ", "per-cluster mean λ", "shuffled λ"]; setts = ["Schedule A", "gradual", "static spatial"]
    cols = {"global mean λ": "#0072B2", "per-cluster mean λ": "#56B4E9", "shuffled λ": "#E69F00"}; off = {"global mean λ": 0.22, "per-cluster mean λ": 0.0, "shuffled λ": -0.22}
    for c in ctrls:
        for i, st in enumerate(setts):
            r = [x for x in tim if x["setting"] == st and x["control"] == c and x["adaptive_minus_control_pp"]]
            if not r: continue
            v = float(r[0]["adaptive_minus_control_pp"]); lo, hi = ci(r[0]["ci95"])
            ax.errorbar(v, i + off[c], xerr=[[v - lo], [hi - v]], fmt="o", color=cols[c], ms=4.5, capsize=2, lw=1.1, label=c if i == 0 else None)
    ax.axvline(0, color="0.6", lw=0.8); ax.set_yticks(range(len(setts))); ax.set_yticklabels([NAME.get(s, s) for s in setts])
    ax.set_xlabel("Adaptive − control (pp)"); ax.set_title("Weight-timing controls (same-pool replay, 3 seeds)", fontsize=8.5)
    ax.grid(True, axis="x"); ax.legend(loc="lower right", frameon=False); fig.tight_layout(); save(fig, "fig05_timing_controls")

# ---------------- F6: legend moved clear of rows ----------------
def f6():
    rows = rd(D3 / "f6_server_signal.csv")
    labs = [NAME.get(r["setting"].replace("CIFAR-10 Schedule A", "Schedule A"), r["setting"]) for r in rows][::-1]
    vals = [float(r["rate_minus_tv_pp"]) for r in rows][::-1]; cis = [ci(r["ci95"]) for r in rows][::-1]; ns = [r["n_seeds"] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(89 * MM, 82 * MM)); ax.axvline(0, color="0.5", lw=0.9)
    for i, (v, (lo, hi), n) in enumerate(zip(vals, cis, ns)):
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=C["server"] if v > 0 else C["TV"], ms=5, capsize=2.5, lw=1.2)
        ax.text(hi + 0.12, i, f"n={n}", va="center", fontsize=6.5, color="0.4")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.set_xlabel("Server non-Main rate − TV (pp)")
    ax.set_title("Direct server signal vs dual-exit TV", fontsize=9); ax.grid(True, axis="x"); ax.set_xlim(-2.4, 2.6); ax.set_ylim(-0.7, len(labs) - 0.3 + 1.1)
    leg = [Line2D([0], [0], marker="o", color=C["server"], lw=0, label="rate higher"), Line2D([0], [0], marker="o", color=C["TV"], lw=0, label="TV higher")]
    ax.legend(handles=leg, loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.0))   # top strip above the first row
    fig.tight_layout(); save(fig, "fig06_server_signal_comparison")

# ---------------- NEW (A1): segment accuracy ----------------
def f7_segments():
    rows = rd(D4 / "a1_segment_acc.csv")
    arms = [("TV full", C["TV"], "o"), ("entropy", C["entropy"], "s"), ("fixed 0.4", C["fixed l0.4"], "^"), ("fixed 0.2", C["fixed l0.2"], "v")]
    fig, axes = plt.subplots(1, 2, figsize=(181 * MM, 62 * MM), width_ratios=[5, 2])
    for ax, st, segs in ((axes[0], "A", ["rho0 early", "rho0.4 up", "rho0.8", "rho0.4 down", "rho0 late"]),
                         (axes[1], "mob", ["pre-move", "post-move"])):
        for k, (arm, col, mk) in enumerate(arms):
            rr = {r["segment"]: r for r in rows if r["setting"] == st and r["arm"] == arm}
            if not rr: continue
            xs = np.arange(len(segs)) + (k - 1.5) * 0.16
            ys = [float(rr[s]["acc_mean_pct"]) for s in segs]; es = [float(rr[s]["acc_sd_pct"] or 0) for s in segs]
            ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, ms=4, capsize=2, lw=1.0, label=arm.replace("TV full", "TV (DriftGate)"))
        lab = {"pre-move": "ρ=0\n(R1–60)", "post-move": "ρ=0.8\n(R70–120)"}
        ax.set_xticks(range(len(segs)))
        ax.set_xticklabels([lab.get(s, s.replace("rho", "ρ=").replace(" up", "↑").replace(" down", "↓")) for s in segs], fontsize=7.5)
        ax.set_ylabel("Accuracy (%)"); ax.set_title(NAME[st], fontsize=9); ax.grid(True, axis="y")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[0].set_xlabel("segment (eval rounds)"); axes[1].set_xlabel("segment (eval rounds)")
    fig.tight_layout(w_pad=1.8); save(fig, "fig07_segment_accuracy")

for fn in (f1, f2, f3, f4, f5, f6, f7_segments):
    fn()
print("figures ->", FIG)
