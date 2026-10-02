"""Round 7 gate: all values of directive 5.1, the pre-registered decision of 5.2, tables and two figures.
raw JSON + *_trace.npz (runs/phaseT7_gate) -> this script -> tables/*.csv, decision.json, figures/*.

Arms: F (fixed 0.4), C (cell DriftGate), CL (cell DriftGate, Lambda 0.5), DTV / DSR (device lambda from x_TV /
x_SR), O (oracle 0.70 home / 0.15 away). Seeds 5, 6, 7. Paired differences use the same seed (Student-t 95% CI,
n_pos = seeds where the first arm is higher). Home / away accuracy as in Round 6: per evaluation round the mean
over clients at home / away, averaged over the rounds where the group is non-empty.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
RUNS = JR / "runs" / "phaseT7_gate"
TAB, FIG = HERE / "tables", HERE / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

ARMS = ["F", "C", "CL", "DTV", "DSR", "O"]
NAME = {"F": "F (fixed 0.4)", "C": "C (cell DriftGate)", "CL": "C-Λ (cell, Λ=0.5)", "DTV": "D-TV (device, x_TV)",
        "DSR": "D-SR (device, x_SR)", "O": "O (oracle)"}
SEEDS = [5, 6, 7]
SLOTS = [("pre-commute", 1, 25), ("commute", 26, 45), ("daytime", 46, 110), ("return", 111, 140), ("evening", 141, 150)]
PAIRS = [("O", "F"), ("DTV", "F"), ("DSR", "F"), ("C", "F"), ("CL", "C"), ("DSR", "DTV")]


def load(arm, s):
    h = json.load(open(RUNS / f"r7_{arm}_s{s}.json"))
    z = np.load(RUNS / f"r7_{arm}_s{s}_trace.npz")
    er = np.array(h["eval_rounds"])
    acc = z["eval_correct"] / np.maximum(z["eval_n_total"], 1)
    home = z["at_home"][er - 1]                                        # [E, K]
    with np.errstate(invalid="ignore", divide="ignore"):
        am = np.where(z["eval_n_main"] > 0, z["eval_c_main"] / z["eval_n_main"], np.nan)
        an = np.where(z["eval_n_nonmain"] > 0, z["eval_c_nonmain"] / z["eval_n_nonmain"], np.nan)
    hm = [acc[e][home[e]].mean() for e in range(len(er)) if home[e].any()]
    aw = [acc[e][~home[e]].mean() for e in range(len(er)) if (~home[e]).any()]
    return dict(h=h, z=z, rounds=er, acc_round=np.array([e["acc_total"] for e in h["eval"]]),
                integ=float(np.mean([e["acc_total"] for e in h["eval"]])), home=float(np.mean(hm)),
                away=float(np.mean(aw)), main=float(np.mean(np.nanmean(am, axis=1))),
                nonmain=float(np.mean(np.nanmean(an, axis=1))),
                offload=float(z["eval_n_off"].sum() / z["eval_n_total"].sum()))


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS]) * 100
    h = stats.sem(d) * stats.t.ppf(0.975, len(d) - 1)
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), n_pos=int((d > 0).sum()),
                n=len(d), per_seed=[float(x) for x in d])


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def response_events(lam, home, start=25):
    """per client: leave events (home -> away) -> rounds until lambda < 0.425 while still away;
    return events (away -> home) -> rounds until lambda > 0.425 while still at home. Censored (None) if the
    client changes state again first or the day ends. Events from round 26 on (index >= 25).
    Each event is (rounds, already): already = lambda was past 0.425 in the last round before the move."""
    leave, back = [], []
    T, K = lam.shape
    for k in range(K):
        for t in range(max(start, 1), T):
            if home[t - 1, k] and not home[t, k]:
                hit = None
                for u in range(t, T):
                    if home[u, k]:
                        break
                    if lam[u, k] < 0.425:
                        hit = u - t
                        break
                leave.append((hit, bool(lam[t - 1, k] < 0.425)))
            if (not home[t - 1, k]) and home[t, k]:
                hit = None
                for u in range(t, T):
                    if not home[u, k]:
                        break
                    if lam[u, k] > 0.425:
                        hit = u - t
                        break
                back.append((hit, bool(lam[t - 1, k] > 0.425)))
    return leave, back


def main():
    R = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    # ---------- T1 per run
    rows = []
    for a in ARMS:
        for s in SEEDS:
            r = R[(a, s)]
            rows.append([NAME[a], s, r["h"]["config"]["run_id"], f"{r['integ'] * 100:.4f}", f"{r['home'] * 100:.4f}",
                         f"{r['away'] * 100:.4f}", f"{r['main'] * 100:.4f}", f"{r['nonmain'] * 100:.4f}",
                         f"{r['offload'] * 100:.4f}"])
        m = lambda key: np.mean([R[(a, s)][key] for s in SEEDS]) * 100
        sd = lambda key: np.std([R[(a, s)][key] for s in SEEDS], ddof=1) * 100
        rows.append([NAME[a], "mean (SD)", "", f"{m('integ'):.4f} ({sd('integ'):.4f})", f"{m('home'):.4f} ({sd('home'):.4f})",
                     f"{m('away'):.4f} ({sd('away'):.4f})", f"{m('main'):.4f}", f"{m('nonmain'):.4f}", f"{m('offload'):.4f}"])
    wcsv("T1_runs_and_means.csv", ["arm", "seed", "run_id", "integrated_acc_pct", "home_acc_pct", "away_acc_pct",
                                   "main_acc_pct", "nonmain_acc_pct", "offload_pct"], rows)
    # ---------- T2 paired differences (integrated, home, away)
    P = {}
    rows = []
    for a, b in PAIRS:
        for key in ("integ", "home", "away"):
            p = paired({s: R[(a, s)][key] for s in SEEDS}, {s: R[(b, s)][key] for s in SEEDS})
            P[(a, b, key)] = p
            rows.append([f"{NAME[a]} − {NAME[b]}", {"integ": "integrated", "home": "at home", "away": "away"}[key],
                         f"{p['mean']:+.4f}", f"[{p['lo']:+.2f}, {p['hi']:+.2f}]", p["n_pos"], p["n"],
                         " ".join(f"{x:+.2f}" for x in p["per_seed"])])
    wcsv("T2_paired_differences.csv", ["comparison", "accuracy", "mean_pp", "ci95", "n_pos", "n", "per_seed_pp(5,6,7)"], rows)
    # ---------- T3 lambda_k at home / away (device arms and oracle), overall and per slot
    rows = []
    for a in ("C", "CL", "DTV", "DSR", "O"):
        for name, lo, hi in [("all rounds", 1, 150), ("after warm-up (26-150)", 26, 150)] + SLOTS:
            vh, va = [], []
            for s in SEEDS:
                z = R[(a, s)]["z"]
                lam, home = z["client_lam"][lo - 1:hi], z["at_home"][lo - 1:hi].astype(bool)
                vh.append(lam[home].mean() if home.any() else np.nan)
                va.append(lam[~home].mean() if (~home).any() else np.nan)
            rows.append([NAME[a], name, f"{np.nanmean(vh):.4f}", f"{np.nanmean(va):.4f}", f"{np.nanmean(vh) - np.nanmean(va):+.4f}"])
    wcsv("T3_lambda_home_away.csv", ["arm", "rounds", "mean_lambda_home", "mean_lambda_away", "home_minus_away"], rows)
    # ---------- T4 AUROC of the signals for away (F runs, rounds >= 26)
    rows, auc_mean = [], {}
    for sig in ("x_tv", "x_sr"):
        per_seed = []
        for s in SEEDS:
            z = R[("F", s)]["z"]
            x, home = z[sig][25:], z["at_home"][25:].astype(bool)
            vals = [auroc(x[:, k], (~home[:, k]).astype(int)) for k in range(x.shape[1])
                    if home[:, k].any() and (~home[:, k]).any() and np.isfinite(x[:, k]).all()]
            per_seed.append(np.mean(vals))
            rows.append([sig, s, len(vals), x.shape[1], f"{np.mean(vals):.4f}", f"{np.std(vals, ddof=1):.4f}"])
        auc_mean[sig] = float(np.mean(per_seed))
        rows.append([sig, "mean of seeds", "", "", f"{np.mean(per_seed):.4f}", ""])
    wcsv("T4_signal_auroc_away.csv", ["signal", "seed", "clients_with_both_states", "clients", "mean_auroc", "sd_over_clients"], rows)
    # ---------- T5 reaction / recovery of lambda_k
    rows = []
    for a in ("C", "CL", "DTV", "DSR", "O"):
        L_all, B_all = [], []
        for s in SEEDS:
            z = R[(a, s)]["z"]
            lv, bk = response_events(z["client_lam"], z["at_home"].astype(bool))
            L_all += lv
            B_all += bk
        for ev, lst in (("leave home -> lambda < 0.425", L_all), ("return home -> lambda > 0.425", B_all)):
            hit = [h for h, _ in lst if h is not None]
            fresh = [(h, al) for h, al in lst if not al]
            fhit = [h for h, _ in fresh if h is not None]
            fmt = lambda v, f: f"{f(v) * 6:.1f}" if v else ""
            rows.append([NAME[a], ev, len(lst), len(hit), len(lst) - len(hit), fmt(hit, np.mean), fmt(hit, np.median),
                         sum(al for _, al in lst), len(fresh), len(fhit), fmt(fhit, np.mean), fmt(fhit, np.median)])
    wcsv("T5_lambda_response.csv", ["arm", "event", "events", "reached", "censored", "mean_min", "median_min",
                                    "already_past_before_move", "events_not_already_past", "reached_not_already_past",
                                    "mean_min_not_already_past", "median_min_not_already_past"], rows)
    # ---------- T7 (supplementary, not used by the decision): time-slot decomposition of the paired differences.
    # contribution of a slot = mean same-seed difference over its evaluation rounds x (its rounds / 31);
    # the contributions add up to the integrated difference.
    rows = []
    er = R[("F", SEEDS[0])]["rounds"]
    for a, b in [p for p in PAIRS if p[1] == "F"]:
        d = np.array([R[(a, s)]["acc_round"] - R[(b, s)]["acc_round"] for s in SEEDS]) * 100   # [3, 31]
        for name, lo, hi in SLOTS:
            m = (er >= lo) & (er <= hi)
            per_seed = d[:, m].mean(axis=1)
            cont = d[:, m].sum(axis=1) / len(er)
            rows.append([f"{NAME[a]} − {NAME[b]}", name, int(m.sum()), f"{per_seed.mean():+.4f}",
                         f"{cont.mean():+.4f}", " ".join(f"{x:+.2f}" for x in per_seed)])
        rows.append([f"{NAME[a]} − {NAME[b]}", "sum = integrated", len(er), "", f"{d.mean():+.4f}", ""])
    wcsv("T7_slot_decomposition.csv", ["comparison", "slot", "eval_rounds", "mean_diff_in_slot_pp",
                                       "contribution_to_integrated_pp", "per_seed_diff_in_slot_pp(5,6,7)"], rows)
    # away clients per round (how many clients the away curves average over)
    n_away = np.array([(~R[("F", s)]["z"]["at_home"].astype(bool)).sum(axis=1) for s in SEEDS])
    wcsv("T8_clients_away_by_slot.csv", ["slot", "mean_clients_away", "min_clients_away"],
         [[name, f"{n_away[:, lo - 1:hi].mean():.1f}", int(n_away[:, lo - 1:hi].min())] for name, lo, hi in SLOTS])
    # ---------- decision (5.2, fixed before the results)
    up = P[("O", "F", "integ")]["mean"]
    dec = dict(O_minus_F_pp=up, threshold_upper_pp=1.0)
    if up < 1.0:
        dec.update(room="small", verdict="O − F < 1.0 pp: little room for device-level control; no further decision")
    else:
        dec["room"] = "enough"
        res = {}
        for a in ("DTV", "DSR"):
            p = P[(a, "F", "integ")]
            res[a] = dict(minus_F_pp=p["mean"], needed_pp=0.5 * up, seeds_above_F=p["n_pos"],
                          passes=bool(p["mean"] >= 0.5 * up and p["n_pos"] >= 2))
        cand = max(("DTV", "DSR"), key=lambda a: np.mean([R[(a, s)]["integ"] for s in SEEDS]))
        sr_tv = P[("DSR", "DTV", "integ")]["mean"]
        if res["DTV"]["passes"] and res["DSR"]["passes"]:
            chosen = "DSR" if sr_tv >= 0.3 else "DTV"
        elif res[cand]["passes"]:
            chosen = cand
        else:
            chosen = None
        dec.update(signals=res, candidate=cand, DSR_minus_DTV_pp=sr_tv, chosen=chosen,
                   verdict=(f"pass: {chosen}" if chosen else f"no pass (candidate {cand})"))
    dec["C_Lambda_minus_C_pp"] = P[("CL", "C", "integ")]["mean"]
    dec["auroc_away"] = auc_mean
    json.dump(dec, open(HERE / "decision.json", "w"), indent=1)
    print(json.dumps(dec, indent=1))
    # ---------- manifest
    rows = []
    for a in ARMS:
        for s in SEEDS:
            h = R[(a, s)]["h"]
            prov = JR / "provenance" / f"{h['config']['run_id']}.json"
            pv = json.load(open(prov)) if prov.exists() else {}
            gpu_index = (SEEDS.index(s) * len(ARMS) + ARMS.index(a)) % 4   # launch_r7_gate.py: round-robin over GPUs 0-3
            rows.append([h["config"]["run_id"], f"r7_{a}_s{s}", NAME[a], s, 100 + s, f"runs/phaseT6_env/S1_seed{s}.npz",
                         "yes" if len(h["round"]) == 150 and len(h["eval"]) == 31 else "no", len(h["round"]),
                         h["probe_eval_overlap"], "yes" if h["probe_eval_overlap"] == 0 else "NO",
                         gpu_index, pv.get("env", {}).get("gpu", ""), pv.get("git_commit", "")[:7],
                         f"{h['total_time_sec'] / 60:.1f}"])
    wcsv("T6_run_manifest.csv", ["run_id", "run_name", "arm", "seed", "model_seed", "env_file", "complete", "rounds",
                                 "probe_eval_overlap", "overlap_ok", "gpu_index", "gpu", "git_commit", "wall_min"], rows)
    figures(R)


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    COL = {"C": "#2a78d6", "DTV": "#eb6834", "DSR": "#1baf7a", "O": "#4a3aa7", "F": INK2}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})

    def clock(ax):
        ticks = [5 * 60, 10 * 60, 15 * 60, 20 * 60]
        ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
        ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
        ax.set_xlim(1, 151)
        ax.set_xlabel("time of day")

    x = np.arange(1, 151)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, layout="constrained")
    for ax, grp, title in ((axes[0], True, "(a) clients at home"), (axes[1], False, "(b) clients away from home")):
        for a, nm in (("DTV", "D-TV"), ("DSR", "D-SR"), ("O", "oracle")):
            per = []
            for s in SEEDS:
                z = R[(a, s)]["z"]
                lam, home = z["client_lam"], z["at_home"].astype(bool)
                m = home if grp else ~home
                per.append(np.array([lam[t][m[t]].mean() if m[t].any() else np.nan for t in range(150)]))
            ax.plot(x, np.nanmean(per, axis=0), color=COL[a], label=nm, ls="--" if a == "O" else "-")
        ax.axhline(0.4, color=MUTED, ls=":", lw=0.9)
        ax.text(151, 0.4, " fixed 0.4", fontsize=6.5, color=INK2, va="center")
        ax.set_title(title)
        clock(ax)
    axes[0].set_ylabel("mean lambda_k")
    axes[0].legend(loc="lower left")
    fig.savefig(FIG / "figA_lambda_home_away.pdf")
    fig.savefig(FIG / "figA_lambda_home_away.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.8), layout="constrained")
    for a, nm in (("F", "F: fixed 0.4"), ("C", "C: cell DriftGate"), ("DTV", "D-TV"), ("DSR", "D-SR"), ("O", "oracle")):
        er = R[(a, SEEDS[0])]["rounds"]
        acc = np.mean([R[(a, s)]["acc_round"] for s in SEEDS], axis=0) * 100
        ax.plot(er, acc, color=COL[a], marker="o", ms=2.2, label=nm, ls="--" if a == "F" else "-")
    for _, lo, _ in SLOTS[1:]:
        ax.axvline(lo - 0.5, color=AXIS, lw=0.6, zorder=0)
    ax.set_ylabel("accuracy (%)")
    clock(ax)
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(FIG / "figB_accuracy_by_time.pdf")
    fig.savefig(FIG / "figB_accuracy_by_time.png", dpi=300)
    plt.close(fig)
    with open(FIG / "figure_captions.md", "w") as f:
        f.write("# Round 7 gate figure captions\n\n")
        f.write("## figA_lambda_home_away\n\nMean lambda_k of the clients at home (a) and away from home (b) by time of day "
                "for the device-level controller with the TV signal (D-TV), with the server-exit non-Main rate (D-SR) "
                "and the home/away oracle. Commute mobility, seeds 5 to 7. The dotted line marks fixed lambda 0.4. "
                "The device-level controllers hold lambda at 0.425 for the first 25 rounds, while the oracle uses its "
                "values from the first round. Few clients are away after 19:00, so the away curves are noisy there.\n\n")
        f.write("## figB_accuracy_by_time\n\nAccuracy by time of day for fixed lambda 0.4 (F), the cell-level DriftGate (C), "
                "the two device-level controllers and the oracle. Commute mobility, means over seeds 5 to 7. Vertical lines "
                "separate the pre-commute, commute, daytime, return and evening periods.\n")
    print("  figures written")


if __name__ == "__main__":
    main()
