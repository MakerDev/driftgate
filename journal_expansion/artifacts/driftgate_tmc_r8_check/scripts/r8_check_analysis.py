"""Round 8 check: inference-only mixing ratio lambda_inf. All values of directive 5.1 and 5.3, the decision of 5.2,
tables and one figure.  raw JSON + *_trace.npz (runs/phaseT8_check) -> this script -> tables/*.csv, decision.json,
figures/*.

Training arms: T40 (fixed lambda_t 0.4) and T15 (fixed lambda_t 0.15), Lambda 0.5, seeds 5, 6, 7.
Every evaluation round each client was evaluated with theta_inf = lambda_inf * theta_local + (1 - lambda_inf) * cell
average for lambda_inf in {0.15, 0.3, 0.4, 0.55, 0.7} (evinf_* arrays). A policy picks one lambda_inf per client and
evaluation round; integrated accuracy = mean over the 31 evaluation rounds of the mean over clients (equal weights).
  P0  lambda_inf = lambda_t
  P1  oracle: 0.70 at home, lambda_t away (env at_home)
  P2  x_SR switch: q_k recomputed from the recorded x_SR with the Round 7 device controller (same constants);
      rounds 1-25 -> 0.70; from round 26: q_k >= 1.5 -> lambda_t, else 0.70
  P3  as P2 with x_TV
Baseline B = T40 P0. Paired differences use the same seed (Student-t 95% CI, n_pos = seeds where the policy is higher).
Home / away and Main / non-Main accuracy as in Round 6 and 7.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
RUNS = JR / "runs" / "phaseT8_check"
R7RUNS = JR / "runs" / "phaseT7_gate"
TAB, FIG = HERE / "tables", HERE / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_controller import DeviceDriftGate  # noqa: E402

ARMS = ["T40", "T15"]
LT = {"T40": 0.40, "T15": 0.15}
SEEDS = [5, 6, 7]
LI = [0.15, 0.3, 0.4, 0.55, 0.7]
IH = LI.index(0.70)
Q_SWITCH = 1.5
POL = ["P0", "P1", "P2", "P3"]
PNAME = {"P0": "P0 (lambda_t)", "P1": "P1 (oracle)", "P2": "P2 (x_SR switch)", "P3": "P3 (x_TV switch)"}
SLOTS = [("pre-commute", 1, 25), ("commute", 26, 45), ("daytime", 46, 110), ("return", 111, 140), ("evening", 141, 150)]
KEYS = ("correct", "n_off", "c_main", "c_nonmain", "off_main", "off_nonmain")


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")


def recompute_q(x, cells):
    """q_k of the Round 7 device controller (DeviceDriftGate, same constants) from a recorded signal x [T, K]
    and the recorded memberships cells [T, K, 2]; the same reference sets as runner_r6 (members of the client's
    cells that sent a signal this round)."""
    T, K = x.shape
    dc = DeviceDriftGate(K)
    q = np.full((T, K), np.nan)
    for t in range(T):
        xs = {k: float(x[t, k]) for k in range(K) if np.isfinite(x[t, k])}
        mem = {}
        for k in range(K):
            for c in cells[t, k]:
                if c >= 0:
                    mem.setdefault(int(c), set()).add(k)
        ref = {k: sorted(set().union(*[mem[int(c)] for c in cells[t, k] if c >= 0])) for k in xs}
        dc.step(xs, ref)
        q[t] = [dc.q[k] for k in range(K)]
    return q


def load(arm, s):
    h = json.load(open(RUNS / f"r8_{arm}_s{s}.json"))
    z = np.load(RUNS / f"r8_{arm}_s{s}_trace.npz")
    assert np.allclose(z["infer_lambdas"], LI)
    er = np.array(h["eval_rounds"])
    nt = z["eval_n_total"].astype(float)                                   # [E, K]
    ev = {k: z[f"evinf_{k}"].astype(float) for k in KEYS}                   # [E, 5, K]
    assert (z["evinf_correct"] >= 0).all(), "a client was not evaluated (not participating)"
    it = LI.index(LT[arm])
    chk = {k: bool(np.array_equal(z[f"evinf_{k}"][:, it, :], z[f"eval_{k}"])) for k in KEYS}
    with np.errstate(invalid="ignore", divide="ignore"):
        amain = ev["c_main"] / z["eval_n_main"][:, None, :]
        anon = np.where(z["eval_n_nonmain"][:, None, :] > 0, ev["c_nonmain"] / z["eval_n_nonmain"][:, None, :], np.nan)
    home = z["at_home"].astype(bool)
    return dict(h=h, z=z, er=er, nt=nt, acc=ev["correct"] / nt[:, None, :], amain=amain, anon=anon,
                noff=ev["n_off"], it=it, chk=chk, home_e=home[er - 1],
                q_sr=recompute_q(z["x_sr"], z["cells"]), q_tv=recompute_q(z["x_tv"], z["cells"]))


def choices(R):
    E, K = R["nt"].shape
    it = R["it"]
    P = {"P0": np.full((E, K), it), "P1": np.where(R["home_e"], IH, it)}
    for name, key in (("P2", "q_sr"), ("P3", "q_tv")):
        q = R[key][R["er"] - 1]
        ch = np.where(q >= Q_SWITCH, it, IH)
        ch[R["er"] <= 25] = IH
        P[name] = ch
    return P


def metrics(R, ch):
    E, K = ch.shape
    ee, kk = np.arange(E)[:, None], np.arange(K)[None, :]
    acc = R["acc"][ee, ch, kk]
    home = R["home_e"]
    hm = [acc[e][home[e]].mean() for e in range(E) if home[e].any()]
    aw = [acc[e][~home[e]].mean() for e in range(E) if (~home[e]).any()]
    return dict(round=acc.mean(axis=1), integ=float(acc.mean(axis=1).mean()), home=float(np.mean(hm)),
                away=float(np.mean(aw)), main=float(np.mean(np.nanmean(R["amain"][ee, ch, kk], axis=1))),
                nonmain=float(np.mean(np.nanmean(R["anon"][ee, ch, kk], axis=1))),
                offload=float(R["noff"][ee, ch, kk].sum() / R["nt"].sum()))


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS]) * 100
    h = stats.sem(d) * stats.t.ppf(0.975, len(d) - 1)
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), n_pos=int((d > 0).sum()),
                n=len(d), per_seed=[float(x) for x in d])


def fmt_p(p):
    return [f"{p['mean']:+.4f}", f"[{p['lo']:+.2f}, {p['hi']:+.2f}]", p["n_pos"], p["n"],
            " ".join(f"{x:+.2f}" for x in p["per_seed"])]


def main():
    R = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    CH = {k: choices(v) for k, v in R.items()}
    M = {(a, s, p): metrics(R[(a, s)], CH[(a, s)][p]) for a in ARMS for s in SEEDS for p in POL}
    B = {s: M[("T40", s, "P0")] for s in SEEDS}
    # ---------- T1 installed model: lambda_inf = lambda_t check, T40 vs Round 7 F
    rows = []
    for a in ARMS:
        for s in SEEDS:
            r = R[(a, s)]
            row = [a, s, r["h"]["config"]["run_id"], f"{r['h']['integrated_acc'] * 100:.4f}",
                   f"{M[(a, s, 'P0')]['integ'] * 100:.4f}", "yes" if all(r["chk"].values()) else
                   "NO " + ",".join(k for k, v in r["chk"].items() if not v)]
            if a == "T40":
                f7 = json.load(open(R7RUNS / f"r7_F_s{s}.json"))
                z7 = np.load(R7RUNS / f"r7_F_s{s}_trace.npz")
                same_loss = f7["mean_loss"] == r["h"]["mean_loss"]
                same_eval = bool(np.array_equal(z7["eval_correct"], r["z"]["eval_correct"]))
                n_diff_loss = int(sum(x != y for x, y in zip(f7["mean_loss"], r["h"]["mean_loss"])))
                first = next((i + 1 for i, (x, y) in enumerate(zip(f7["mean_loss"], r["h"]["mean_loss"])) if x != y), "")
                row += [f"{f7['integrated_acc'] * 100:.4f}", f"{(r['h']['integrated_acc'] - f7['integrated_acc']) * 100:+.4f}",
                        "yes" if same_loss else "no", n_diff_loss, first, "yes" if same_eval else "no"]
            else:
                row += ["", "", "", "", "", ""]
            rows.append(row)
    wcsv("T1_installed_and_r7F.csv", ["arm", "seed", "run_id", "installed_integrated_pct(json)",
                                      "P0_integrated_pct(recomputed)", "lambda_inf=lambda_t_equals_installed",
                                      "r7_F_integrated_pct", "T40_minus_r7F_pp", "loss_identical_all_rounds",
                                      "rounds_with_different_loss", "first_round_different",
                                      "eval_correct_identical"], rows)
    # ---------- T2 policies per run and mean
    rows = []
    for a in ARMS:
        for p in POL:
            for s in SEEDS:
                m = M[(a, s, p)]
                rows.append([a, PNAME[p], s] + [f"{m[k] * 100:.4f}" for k in ("integ", "home", "away", "main", "nonmain", "offload")])
            mm = lambda k: np.mean([M[(a, s, p)][k] for s in SEEDS]) * 100
            sd = lambda k: np.std([M[(a, s, p)][k] for s in SEEDS], ddof=1) * 100
            rows.append([a, PNAME[p], "mean (SD)"] + [f"{mm(k):.4f} ({sd(k):.4f})" for k in ("integ", "home", "away", "main", "nonmain", "offload")])
    wcsv("T2_policies.csv", ["lambda_t_arm", "policy", "seed", "integrated_pct", "home_pct", "away_pct", "main_pct",
                             "nonmain_pct", "offload_pct"], rows)
    # ---------- T3 paired differences vs B
    P = {}
    rows = []
    for a in ARMS:
        for p in POL:
            if a == "T40" and p == "P0":
                continue
            for k in ("integ", "home", "away", "main", "nonmain"):
                d = paired({s: M[(a, s, p)][k] for s in SEEDS}, {s: B[s][k] for s in SEEDS})
                P[(a, p, k)] = d
                rows.append([a, PNAME[p], k] + fmt_p(d))
    for a in ARMS:   # P2 - P3 (rule 3)
        d = paired({s: M[(a, s, "P2")]["integ"] for s in SEEDS}, {s: M[(a, s, "P3")]["integ"] for s in SEEDS})
        P[(a, "P2-P3", "integ")] = d
        rows.append([a, "P2 - P3", "integ"] + fmt_p(d))
    wcsv("T3_paired_vs_B.csv", ["lambda_t_arm", "policy", "metric", "mean_pp", "ci95", "n_pos", "n", "per_seed_pp(5,6,7)"], rows)
    # ---------- decision (5.2, fixed before the results)
    up = {a: P[(a, "P1", "integ")]["mean"] for a in ARMS}
    lt = max(ARMS, key=lambda a: up[a])
    dec = dict(B="T40 P0", P1_minus_B_pp=up, chosen_lambda_t=lt, upper_pp=up[lt], threshold_upper_pp=1.0)
    if up[lt] < 1.0:
        dec.update(room="small", verdict=f"max(P1 - B) = {up[lt]:+.2f} pp < 1.0 pp ({lt}): little room; results only")
    else:
        dec["room"] = "enough"
        res = {}
        for p in ("P2", "P3"):
            d = P[(lt, p, "integ")]
            res[p] = dict(minus_B_pp=d["mean"], needed_pp=0.5 * up[lt], seeds_above_B=d["n_pos"],
                          passes=bool(d["mean"] >= 0.5 * up[lt] and d["n_pos"] >= 2))
        cand = max(("P2", "P3"), key=lambda p: np.mean([M[(lt, s, p)]["integ"] for s in SEEDS]))
        p2p3 = P[(lt, "P2-P3", "integ")]["mean"]
        if res["P2"]["passes"] and res["P3"]["passes"]:
            chosen = "x_SR (P2)" if p2p3 >= 0.3 else "x_TV (P3)"
        elif res[cand]["passes"]:
            chosen = {"P2": "x_SR (P2)", "P3": "x_TV (P3)"}[cand]
        else:
            chosen = None
        dec.update(signals=res, candidate=cand, P2_minus_P3_pp=p2p3, chosen=chosen,
                   verdict=(f"pass: {chosen}" if chosen else f"no pass (candidate {cand})"))
    json.dump(dec, open(HERE / "decision.json", "w"), indent=1)
    print(json.dumps(dec, indent=1))
    # ---------- T4 time-slot contributions vs B
    rows = []
    er = R[("T40", SEEDS[0])]["er"]
    for a in ARMS:
        for p in POL:
            if a == "T40" and p == "P0":
                continue
            d = np.array([M[(a, s, p)]["round"] - B[s]["round"] for s in SEEDS]) * 100
            for name, lo, hi in SLOTS:
                m = (er >= lo) & (er <= hi)
                rows.append([a, PNAME[p], name, int(m.sum()), f"{d[:, m].mean():+.4f}",
                             f"{(d[:, m].sum(axis=1) / len(er)).mean():+.4f}", " ".join(f"{x:+.2f}" for x in d[:, m].mean(axis=1))])
            rows.append([a, PNAME[p], "sum = integrated", len(er), "", f"{d.mean():+.4f}", ""])
    wcsv("T4_slot_contribution_vs_B.csv", ["lambda_t_arm", "policy", "slot", "eval_rounds", "mean_diff_in_slot_pp",
                                           "contribution_pp", "per_seed_diff_in_slot_pp(5,6,7)"], rows)
    # ---------- T5 lambda_t x lambda_inf (one lambda_inf for every client)
    rows = []
    for a in ARMS:
        for i, li in enumerate(LI):
            mm = {s: metrics(R[(a, s)], np.full(R[(a, s)]["nt"].shape, i)) for s in SEEDS}
            d = paired({s: mm[s]["integ"] for s in SEEDS}, {s: B[s]["integ"] for s in SEEDS})
            rows.append([a, li] + [f"{np.mean([mm[s][k] for s in SEEDS]) * 100:.4f}" for k in ("integ", "home", "away", "main", "nonmain", "offload")]
                        + fmt_p(d)[:3])
    wcsv("T5_lambda_t_by_lambda_inf.csv", ["lambda_t_arm", "lambda_inf", "integrated_pct", "home_pct", "away_pct",
                                           "main_pct", "nonmain_pct", "offload_pct", "minus_B_pp", "ci95", "n_pos"], rows)
    # ---------- T6 home / away grid (reference)
    rows = []
    for a in ARMS:
        for ih, lh in enumerate(LI):
            for ia, la in enumerate(LI):
                mm = {s: metrics(R[(a, s)], np.where(R[(a, s)]["home_e"], ih, ia)) for s in SEEDS}
                d = paired({s: mm[s]["integ"] for s in SEEDS}, {s: B[s]["integ"] for s in SEEDS})
                rows.append([a, lh, la, f"{np.mean([mm[s]['integ'] for s in SEEDS]) * 100:.4f}"] + fmt_p(d)[:3])
    wcsv("T6_home_away_grid.csv", ["lambda_t_arm", "lambda_inf_home", "lambda_inf_away", "integrated_pct",
                                   "minus_B_pp", "ci95", "n_pos"], rows)
    # ---------- T7 upper bound: best lambda_inf per client and evaluation round (reference)
    rows = []
    for a in ARMS:
        mm = {s: metrics(R[(a, s)], R[(a, s)]["acc"].argmax(axis=1)) for s in SEEDS}
        d = paired({s: mm[s]["integ"] for s in SEEDS}, {s: B[s]["integ"] for s in SEEDS})
        rows.append([a, " ".join(f"{mm[s]['integ'] * 100:.2f}" for s in SEEDS),
                     f"{np.mean([mm[s]['integ'] for s in SEEDS]) * 100:.4f}"] + fmt_p(d)[:3])
    wcsv("T7_upper_bound_best_per_client_round.csv", ["lambda_t_arm", "integrated_pct_per_seed(5,6,7)",
                                                      "integrated_pct", "minus_B_pp", "ci95", "n_pos"], rows)
    # ---------- T8 Main-aware routing on the installed model (reference)
    rows = []
    for a in ARMS:
        vals, inst = {}, {}
        for s in SEEDS:
            z = R[(a, s)]["z"]
            vals[s] = float((z["eval_ma_correct"] / z["eval_n_total"]).mean(axis=1).mean())
            inst[s] = M[(a, s, "P0")]["integ"]
            off = z["eval_ma_n_off"].sum() / z["eval_n_total"].sum()
            rows.append([a, s, f"{inst[s] * 100:.4f}", f"{vals[s] * 100:.4f}", f"{(vals[s] - inst[s]) * 100:+.4f}",
                         f"{z['eval_n_off'].sum() / z['eval_n_total'].sum() * 100:.2f}", f"{off * 100:.2f}"])
        d = paired(vals, inst)
        rows.append([a, "mean", "", "", f"{d['mean']:+.4f} {fmt_p(d)[1]} ({d['n_pos']}/3)", "", ""])
        d = paired(vals, {s: B[s]["integ"] for s in SEEDS})
        rows.append([a, "minus B", "", "", f"{d['mean']:+.4f} {fmt_p(d)[1]} ({d['n_pos']}/3)", "", ""])
    wcsv("T8_mainaware_route.csv", ["lambda_t_arm", "seed", "installed_integrated_pct", "mainaware_integrated_pct",
                                    "mainaware_minus_installed_pp", "offload_pct_installed", "offload_pct_mainaware"], rows)
    # ---------- T9 how P2 / P3 switch (evaluation rounds >= 30)
    rows = []
    for a in ARMS:
        for p in ("P2", "P3"):
            hh, aa = [], []
            for s in SEEDS:
                ch, home = CH[(a, s)][p], R[(a, s)]["home_e"]
                m = (R[(a, s)]["er"] >= 26)[:, None] & np.ones_like(home)
                hh.append((ch[m & home] == IH).mean())
                aa.append((ch[m & ~home] != IH).mean())
            rows.append([a, PNAME[p], f"{np.mean(hh):.4f}", f"{np.mean(aa):.4f}", " ".join(f"{x:.3f}" for x in hh),
                         " ".join(f"{x:.3f}" for x in aa)])
    wcsv("T9_switch_choices.csv", ["lambda_t_arm", "policy", "share_0.70_at_home", "share_lambda_t_away",
                                   "per_seed_home", "per_seed_away"], rows)
    # ---------- T10 the offline q_k recomputation reproduces the Round 7 device controller
    rows = []
    for arm, key in (("DTV", "x_tv"), ("DSR", "x_sr")):
        for s in SEEDS:
            z = np.load(R7RUNS / f"r7_{arm}_s{s}_trace.npz")
            q = recompute_q(z[key], z["cells"])
            qr = z["q_k"].astype(float)
            sl = slice(25, None)
            flips = int(((q[sl] >= Q_SWITCH) != (qr[sl] >= Q_SWITCH)).sum())
            rows.append([arm, s, f"{np.nanmax(np.abs(q - qr)):.2e}", flips, int(np.isfinite(qr[sl]).sum())])
    wcsv("T10_q_recompute_check.csv", ["r7_arm", "seed", "max_abs_diff_q", "switch_decisions_different_r26_150",
                                       "client_rounds_r26_150"], rows)
    # ---------- T11 manifest
    rows = []
    for a in ARMS:
        for s in SEEDS:
            h = R[(a, s)]["h"]
            prov = JR / "provenance" / f"{h['config']['run_id']}.json"
            pv = json.load(open(prov)) if prov.exists() else {}
            gpu_index = (SEEDS.index(s) * len(ARMS) + ARMS.index(a)) % 4   # launch_r8_check.py round-robin
            rows.append([h["config"]["run_id"], f"r8_{a}_s{s}", a, LT[a], s, 100 + s, f"runs/phaseT6_env/S1_seed{s}.npz",
                         "yes" if len(h["round"]) == 150 and len(h["eval"]) == 31 else "no", len(h["round"]),
                         h["probe_eval_overlap"], gpu_index, pv.get("env", {}).get("gpu", ""),
                         pv.get("git_commit", "")[:7], f"{h['total_time_sec'] / 60:.1f}"])
    wcsv("T11_run_manifest.csv", ["run_id", "run_name", "arm", "lambda_t", "seed", "model_seed", "env_file", "complete",
                                  "rounds", "probe_eval_overlap", "gpu_index", "gpu", "git_commit", "wall_min"], rows)
    figure(R, M, B, lt)


def figure(R, M, B, lt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    COL = {"B": INK2, "P1": "#4a3aa7", "P2": "#1baf7a", "P3": "#eb6834"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    er = R[("T40", SEEDS[0])]["er"]
    fig, ax = plt.subplots(figsize=(4.8, 2.8), layout="constrained")
    ax.plot(er, np.mean([B[s]["round"] for s in SEEDS], axis=0) * 100, color=COL["B"], ls="--", marker="o", ms=2.2,
            label="B: lambda_t 0.4 for training and inference")
    lab = {"P1": "P1: oracle", "P2": "P2: x_SR switch", "P3": "P3: x_TV switch"}
    for p in ("P1", "P2", "P3"):
        ax.plot(er, np.mean([M[(lt, s, p)]["round"] for s in SEEDS], axis=0) * 100, color=COL[p], marker="o", ms=2.2,
                label=f"{lab[p]} (lambda_t {LT[lt]:.2f})")
    for _, lo, _ in SLOTS[1:]:
        ax.axvline(lo - 0.5, color=AXIS, lw=0.6, zorder=0)
    ticks = [5 * 60, 10 * 60, 15 * 60, 20 * 60]
    ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
    ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
    ax.set_xlim(1, 151)
    ax.set_xlabel("time of day")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.savefig(FIG / "fig_accuracy_by_time.pdf")
    fig.savefig(FIG / "fig_accuracy_by_time.png", dpi=300)
    plt.close(fig)
    with open(FIG / "figure_captions.md", "w") as f:
        f.write("# Round 8 check figure caption\n\n## fig_accuracy_by_time\n\n"
                f"Accuracy by time of day for the baseline B (lambda 0.4 for training and inference) and for three "
                f"inference-time policies on the model trained with lambda_t {LT[lt]:.2f}: the home/away oracle (P1) and "
                "the switches driven by the server-exit non-Main rate (P2) and by the TV signal (P3). Commute mobility, "
                "means over seeds 5 to 7. Vertical lines separate the pre-commute, commute, daytime, return and evening "
                "periods.\n")
    print("  figure written")


if __name__ == "__main__":
    main()
