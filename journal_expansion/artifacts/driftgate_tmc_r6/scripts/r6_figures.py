"""Round 6 figures (vector PDF + 300 dpi PNG) and English captions, from env files, run files and
the CSV tables written by r6_tables.py / latency_calc.py.

  fig1_scenario_commute / fig1_scenario_geolife   map, clients per cell, cell mean rho (by clock time)
  fig2_lambda_trajectories                        cell rho (top) and lambda of DriftGate / entropy controller
  fig3_accuracy_by_time                           accuracy by clock time: DriftGate, best fixed, entropy controller
  fig4_scale_robustness                           DriftGate - best fixed lambda with 95% CI per condition
  fig5_latency                                    offload rate and mean / p95 end-to-end latency
Colours (validated with the dataviz validator, light surface): DriftGate #2a78d6, entropy controller
#eb6834, best fixed #1baf7a; cells use slots 1-5; fixed-lambda reference lines are grey dashed.
Internal names are replaced by paper terms (S1 -> commute mobility, S2 -> GeoLife trace,
relonly -> DriftGate, Schedule A -> stepwise composition change).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parent.parent
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r6_env  # noqa: E402
import r6_tables as T  # noqa: E402

FIG = HERE / "figures"
TAB = HERE / "tables"
ENV = JR / "runs" / "phaseT6_env"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_DG, C_ENT, C_FIX = "#2a78d6", "#eb6834", "#1baf7a"
CELL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
S1_CELL = ["E (residential)", "N (residential)", "W (residential)", "S (residential)", "H (hub)"]
CAPS = []

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {name}")


def clock_axis(ax, label=True):
    ticks = [5 * 60, 10 * 60, 15 * 60, 20 * 60]
    ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
    ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
    ax.set_xlim(1, 151)
    if label:
        ax.set_xlabel("time of day")


def rows(name):
    p = TAB / name
    return list(csv.DictReader(open(p))) if p.exists() else []


# ------------------------------------------------------------------------ fig 1
def pick_clients_s1(env):
    """one worker from each residential cell (lowest id whose work cell is the hub) and one stay-home client."""
    picks = []
    for cell in range(4):
        for k in range(len(env["home_cell"])):
            if env["home_cell"][k] == cell and env["plan_type"][k] == "work":
                picks.append(k)
                break
    moved = [k for k in range(len(env["home_cell"])) if env["plan_type"][k] == "stay"
             and np.ptp(env["pos"][k, :, 0]) + np.ptp(env["pos"][k, :, 1]) > 0.5]
    if moved:
        picks.append(moved[0])
    return picks[:5]


def pick_clients_s2(env):
    """the five clients with the longest round-to-round path."""
    step = np.linalg.norm(np.diff(env["pos"], axis=1), axis=2).sum(axis=1)
    return list(np.argsort(-step)[:5])


def fig1(scen, seeds, name, title_scen):
    env0, meta = r6_env.load_env(ENV / f"{scen}_seed{seeds[0]}.npz")
    L = len(env0["cell_xy"])
    occ = np.zeros((len(seeds), 150, L))
    rho = np.full((len(seeds), 150, L), np.nan)
    for i, s in enumerate(seeds):
        env, _ = r6_env.load_env(ENV / f"{scen}_seed{s}.npz")
        for z in range(L):
            mem = (env["member"] == z).any(axis=2)                      # [K, T]
            occ[i, :, z] = mem.sum(axis=0)
            with np.errstate(invalid="ignore"):
                rho[i, :, z] = np.where(mem.any(axis=0), (env["rho"] * mem).sum(axis=0) / np.maximum(mem.sum(axis=0), 1), np.nan)
    labels = S1_CELL if scen == "S1" else [
        f"cell {z} ({n} resident{'s' if n != 1 else ''})" for z, n in enumerate(meta["residents_per_cell"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.1, 1, 1]},
                             layout="constrained")
    ax = axes[0]
    pos = env0["pos"]
    picks = pick_clients_s1(env0) if scen == "S1" else pick_clients_s2(env0)
    for z in range(L):
        x, y = env0["cell_xy"][z]
        if scen == "S1":
            ax.add_patch(Circle((x, y), r6_env.R_KM, facecolor=CELL_COLORS[z], alpha=0.12,
                                edgecolor=CELL_COLORS[z], lw=1.0))
        ax.plot([x], [y], marker="^", ms=6, color=CELL_COLORS[z], mec="white", mew=0.8, zorder=4, ls="none")
        if scen == "S1":
            d = np.array([x, y]) / (np.hypot(x, y) or 1.0)
            lx, ly = (np.array([x, y]) + 1.25 * d) if np.hypot(x, y) > 0 else (0.62, -0.62)
            ax.text(lx, ly, labels[z].split(" (")[0], fontsize=7, color=INK2, ha="center", va="center")
        else:
            ax.annotate(f"cell {z}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK2,
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    for i, k in enumerate(picks):
        ax.plot(pos[k, :, 0], pos[k, :, 1], color=INK, lw=0.9, ls=styles[i], alpha=0.85)
        ax.plot(pos[k, 0, 0], pos[k, 0, 1], marker="o", ms=3.5, color=INK, ls="none")
        ax.annotate(f"{i + 1}", (pos[k, 0, 0], pos[k, 0, 1]), xytext=(3, -8), textcoords="offset points",
                    fontsize=6.5, color=INK, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("(a) cells and five daily paths")
    ax.grid(False)
    if scen == "S1":
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
    x = np.arange(1, 151)
    for z in range(L):
        axes[1].plot(x, occ[:, :, z].mean(axis=0), color=CELL_COLORS[z], label=labels[z])
        axes[2].plot(x, np.nanmean(rho[:, :, z], axis=0), color=CELL_COLORS[z])
    axes[1].set_title("(b) clients in each cell")
    axes[1].set_ylabel("clients")
    axes[2].set_title("(c) cell mean rho")
    axes[2].set_ylabel("rho")
    axes[2].set_ylim(0.0, 0.85)
    for a in axes[1:]:
        clock_axis(a)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5 if scen == "S1" else 3, fontsize=6.5)
    save(fig, name)
    return occ, rho, picks, meta


# ------------------------------------------------------------------------ fig 2
def lam_series(scen, arm):
    out = {}
    for s, f in T.fam(scen, arm).items():
        out[s] = T.run(f, scen, s)
    return out


def fig2():
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2), sharex=True, layout="constrained")
    x = np.arange(1, 151)
    info = {}
    # S1: hub vs residential
    runs = {arm: lam_series("S1", arm) for arm in ("driftgate", "entropy")}
    if runs["driftgate"]:
        any_run = next(iter(runs["driftgate"].values()))
        hub = np.flatnonzero(any_run["env"]["cell_is_hub"])
        res = np.flatnonzero(~any_run["env"]["cell_is_hub"])
        rho = np.mean([r["rho_cell"] for r in runs["driftgate"].values()], axis=0)
        axes[0, 0].plot(x, np.nanmean(rho[:, hub], axis=1), color=INK, label="hub cell")
        axes[0, 0].plot(x, np.nanmean(rho[:, res], axis=1), color=MUTED, label="residential cells (mean)")
        axes[0, 0].legend(loc="lower center")
        for row, cells, lab in ((1, hub, "hub cell"), (2, res, "residential cells (mean)")):
            ax = axes[row, 0]
            for arm, col, nm in (("driftgate", C_DG, "DriftGate"), ("entropy", C_ENT, "entropy controller")):
                if runs[arm]:
                    lam = np.mean([r["lam"][:, cells].mean(axis=1) for r in runs[arm].values()], axis=0)
                    ax.plot(x, lam, color=col, label=nm)
                    info[(arm, lab)] = lam
            for v in (0.4, 0.2):
                ax.axhline(v, color=MUTED, ls="--", lw=0.9)
                ax.text(151, v, f" fixed {v:g}", va="center", fontsize=6.5, color=INK2)
            ax.set_ylabel(f"lambda, {lab}")
            ax.set_ylim(0.12, 0.73)
        n = len(runs["driftgate"])
        axes[0, 0].set_title(f"commute mobility ({n} seeds)")
    # S2: per cell rho, one lambda line per method (all cells share lambda)
    runs2 = {arm: lam_series("S2", arm) for arm in ("driftgate", "entropy")}
    if runs2["driftgate"]:
        rho = np.mean([r["rho_cell"] for r in runs2["driftgate"].values()], axis=0)
        L = rho.shape[1]
        meta = next(iter(runs2["driftgate"].values()))["h"]["env_meta"]
        for z in range(L):
            axes[0, 1].plot(x, rho[:, z], color=CELL_COLORS[z], lw=1.2,
                            label=f"cell {z} ({meta['residents_per_cell'][z]})")
        axes[0, 1].legend(loc="upper right", ncol=2, fontsize=6.5, title="cell (residents)", title_fontsize=6.5)
        for row in (1, 2):
            ax = axes[row, 1]
            for arm, col, nm in (("driftgate", C_DG, "DriftGate"), ("entropy", C_ENT, "entropy controller")):
                if runs2[arm]:
                    lams = np.stack([r["lam"] for r in runs2[arm].values()])       # [seeds, T, L]
                    same = float(np.abs(lams - lams[:, :, :1]).max())
                    info[(arm, "S2 max spread across cells")] = same
                    cells = [0] if row == 1 else list(range(1, L))
                    ax.plot(x, lams[:, :, cells].mean(axis=(0, 2)), color=col, label=nm)
            for v in (0.4, 0.2):
                ax.axhline(v, color=MUTED, ls="--", lw=0.9)
                ax.text(151, v, f" fixed {v:g}", va="center", fontsize=6.5, color=INK2)
            ax.set_ylabel("lambda, cell 0" if row == 1 else "lambda, cells 1-4 (mean)")
            ax.set_ylim(0.12, 0.73)
        axes[0, 1].set_title(f"GeoLife trace ({len(runs2['driftgate'])} seeds)")
    axes[0, 0].set_ylabel("cell mean rho")
    axes[0, 1].set_ylabel("cell mean rho")
    for a in axes[2]:
        clock_axis(a)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C_DG), Line2D([], [], color=C_ENT), Line2D([], [], color=MUTED, ls="--", lw=0.9)],
               ["DriftGate", "entropy controller", "fixed lambda (reference)"], loc="outside lower center", ncol=3)
    save(fig, "fig2_lambda_trajectories")
    return info


# ------------------------------------------------------------------------ fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
    out = {}
    for ax, scen in zip(axes, ("S1", "S2")):
        dg = T.metric(scen, "driftgate", "integ")
        if not dg:
            continue
        bf = T.best_fixed(scen, T.FIXED_ALL, sorted(dg))
        for arm, col, nm in (("driftgate", C_DG, "DriftGate"), (bf, C_FIX, f"best fixed ({T.ARM_NAME[bf]})" if bf else ""),
                             ("entropy", C_ENT, "entropy controller")):
            if arm is None:
                continue
            rr = {s: T.run(f, scen, s) for s, f in T.fam(scen, arm).items()}
            if not rr:
                continue
            er = next(iter(rr.values()))["rounds"]
            acc = np.mean([r["acc_round"] for r in rr.values()], axis=0) * 100
            ax.plot(er, acc, color=col, marker="o", ms=2.5, label=nm)
            out[(scen, arm)] = (er, acc)
        for name, clock, lo, hi in T.SLOTS[1:]:
            ax.axvline(lo - 0.5, color=AXIS, lw=0.6, zorder=0)
        ax.set_title(f"{T.SCEN[scen][2]} ({len(dg)} seeds)")
        ax.set_ylabel("accuracy (%)")
        clock_axis(ax)
        ax.legend(loc="lower right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_accuracy_by_time")
    return out


# ------------------------------------------------------------------------ fig 4
def fig4():
    items = []
    for r in rows("T7_scale.csv"):
        items.append((f"K={r['K']} (L={r['L']})", r, "scale"))
    for r in rows("T6_speed.csv"):
        if "vehicle" in r["setting"]:
            items.append(("vehicle speed", r, "speed"))
    for r in rows("T8_robustness.csv"):
        items.append((r["condition"], r, "robustness"))
    items = [(lab, r, g) for lab, r, g in items if r.get("DriftGate_minus_best_pp")]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 0.28 * len(items) + 0.8))
    ys = np.arange(len(items))[::-1]
    for y, (lab, r, g) in zip(ys, items):
        m = float(r["DriftGate_minus_best_pp"])
        lo, hi = [float(v) for v in r["ci95"].strip("[]").split(",")]
        ax.plot([lo, hi], [y, y], color=C_DG, lw=1.5, solid_capstyle="round")
        ax.plot([m], [y], marker="o", ms=5, color=C_DG, mec="white", mew=1.0)
        ax.text(max(hi, m) + 0.08, y, f"{m:+.2f} ({r['n_pos']}/{r['n']})", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab}  [vs {r['best_fixed']}]" for lab, r, g in items])
    ax.set_xlabel("DriftGate minus best fixed lambda (pp), 95% CI")
    ax.grid(axis="y", visible=False)
    xmin = min(float(r["ci95"].strip("[]").split(",")[0]) for _, r, _ in items)
    xmax = max(float(r["ci95"].strip("[]").split(",")[1]) for _, r, _ in items)
    ax.set_xlim(min(-0.2, xmin - 0.2), xmax + 1.0)
    fig.tight_layout()
    save(fig, "fig4_scale_robustness")
    return items


# ------------------------------------------------------------------------ fig 5
def fig5():
    lat = rows("T9_latency.csv")
    if not lat:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    out = {}
    order = ["driftgate"] + T.FIXED_ALL + ["entropy", "apfl001", "apfl010"]
    names = [T.ARM_NAME[a] for a in order]
    off = {}
    for r in lat:
        if r["rtt_ms"] in ("20", "20.0") and r["bandwidth_mbps"] in ("50", "50.0"):
            off[(r["scenario"], r["arm"])] = float(r["offload_rate"]) * 100
    y = np.arange(len(order))[::-1]
    for i, (scen, col) in enumerate((("S1", C_DG), ("S2", MUTED))):
        vals = [off.get((scen, a), np.nan) for a in order]
        axes[0].barh(y + (0.2 if i == 0 else -0.2), vals, height=0.38, color=col,
                     label=T.SCEN[scen][2])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("offloaded requests (%)")
    axes[0].set_title("(a) offload rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", visible=False)
    bf = T.best_fixed("S1", T.FIXED_ALL, sorted(T.metric("S1", "driftgate", "integ")))
    for ax, key, title in ((axes[1], "mean_e2e_ms", "(b) mean latency, commute mobility"),
                           (axes[2], "p95_e2e_ms", "(c) p95 latency, commute mobility")):
        for arm, col, nm in (("driftgate", C_DG, "DriftGate"), (bf, C_FIX, f"best fixed ({T.ARM_NAME[bf]})"),
                             ("entropy", C_ENT, "entropy controller")):
            pts = sorted((float(r["bandwidth_mbps"]), float(r[key])) for r in lat
                         if r["scenario"] == "S1" and r["arm"] == arm and r["rtt_ms"] in ("20", "20.0"))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=3.5, label=nm)
                out[(arm, key)] = pts
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 100])
        ax.set_xticklabels(["10", "50", "100"])
        ax.set_xlabel("uplink bandwidth (Mbps), RTT 20 ms")
        ax.set_ylabel("ms")
        ax.set_title(title)
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig5_latency")
    return out, off


def main():
    CAPS.clear()
    occ1, rho1, picks1, _ = fig1("S1", [0, 1, 2, 3, 4], "fig1_scenario_commute", "commute mobility")
    hub_peak = occ1[:, :, 4].mean(axis=0).max()
    CAPS.append(("fig1_scenario_commute",
                 "Commute mobility scenario with five cells (four residential cells and one hub). "
                 "(a) Cells with a 1 km edge range and the daily paths of five clients of seed 0 "
                 "(numbers mark the start at 05:00). (b) Number of clients in each cell by time of day. "
                 f"The hub holds up to {hub_peak:.0f} clients during the day. (c) Cell mean rho, the share parameter "
                 "of requests outside the client's own classes (0.1 at home, 0.8 away). "
                 "Values are means over five seeds."))
    occ2, rho2, picks2, meta2 = fig1("S2", [0, 1, 2], "fig1_scenario_geolife", "GeoLife trace")
    CAPS.append(("fig1_scenario_geolife",
                 f"GeoLife trace scenario built from {meta2['n_selected']} weekday user days of GeoLife Trajectories 1.3 "
                 f"({meta2['n_candidates']} candidate users). (a) Five edges placed by k-means and the paths of the five "
                 "clients that move the most. (b) Number of clients in each cell by time of day. (c) Cell mean rho. "
                 f"Residents per cell are {', '.join(map(str, meta2['residents_per_cell']))}."))
    info2 = fig2()
    if info2:
        CAPS.append(("fig2_lambda_trajectories",
                     "Cell mean rho (top) and lambda chosen by DriftGate and the entropy controller (middle and bottom). "
                     "Left: commute mobility, hub cell and the mean of the residential cells. Right: GeoLife trace. "
                     "In the GeoLife trace every pair of edges is a neighbour, so the one-step neighbour average gives every "
                     "cell the same score and the same lambda. Dashed lines mark fixed lambda 0.4 and 0.2. "
                     "Lines are means over seeds."))
    f3 = fig3()
    if f3:
        CAPS.append(("fig3_accuracy_by_time",
                     "Accuracy by time of day for DriftGate, the best fixed lambda of each scenario and the entropy "
                     "controller. Vertical lines separate the pre-commute, commute, daytime, return and evening periods. "
                     "Lines are means over seeds."))
    f4 = fig4()
    if f4:
        CAPS.append(("fig4_scale_robustness",
                     "Accuracy difference between DriftGate and the best fixed lambda with 95% confidence intervals over "
                     "matched seeds, for larger systems (K clients, L cells), vehicle speed, signal loss, signal delay, "
                     "a low request rate and partial participation. Numbers give the mean difference and the count of "
                     "seeds where DriftGate is higher."))
    f5 = fig5()
    if f5:
        CAPS.append(("fig5_latency",
                     "(a) Share of evaluation requests sent to the edge for each method. (b, c) Mean and 95th percentile "
                     "end-to-end latency in commute mobility for uplink bandwidths of 10, 50 and 100 Mbps with a 20 ms "
                     "round-trip time. Client-side timings are host placeholders until device measurements are added."))
    with open(FIG / "figure_captions.md", "w") as f:
        f.write("# Round 6 figure captions\n\nGenerated by scripts/r6_figures.py from the tables. "
                "No em dashes or semicolons.\n\n")
        for name, cap in CAPS:
            assert "—" not in cap and ";" not in cap, name
            f.write(f"## {name}\n\n{cap}\n\n")


if __name__ == "__main__":
    main()
