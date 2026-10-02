"""Round 10: correct the server exit with the device's request mix (prior correction), offline from the run records.

Inputs (runs/phaseT10_prior/r10_T40_s{5,6,7}): run JSON, *_trace.npz (train_hist_cell, train_hist_all, cells) and
*_evalprobs.npz (per evaluated request: client- and server-exit softmax probabilities, predictions, entropy, label,
kind, client, evaluation round, at_home, evaluation position, arrival order). Outputs: tables/*.csv, decision.json,
figures/*.

Notation (directive 3.1-3.3): M_k = the client's Main classes (rebuilt with the partition code), O_k = the others.
pi_tr = 0.5 * h_cell + 0.5 * h_all from the label counts of the training round of the evaluation round (two cells:
mean of the two); a = clip(pi_tr(M_k), 0.01, 0.99), b = 1 - a. Prior correction with target r (clip [0.02, 0.98]):
p'(c) ~ p_s(c) (1 - r) / a for c in M_k and p_s(c) r / b for c in O_k. EM for r over a request set W from r0:
q_i = (r/b) p_s(O_k|x_i) / [((1-r)/a) p_s(M_k|x_i) + (r/b) p_s(O_k|x_i)], r <- mean q_i, stop when the change
< 1e-4 or after 50 iterations; the result is clipped to [0.02, 0.98].
r variants: causal (request j of a client and evaluation round, in arrival order, uses EM over the earlier requests
1..j-1 of the same round started at r_prev; fewer than 8 earlier requests -> r_prev; r_prev = EM over all requests of
the client's previous evaluation round, 0.5 at the first); batch (EM over all requests of the round, started at r_prev);
oracle (true non-Main share of the client's requests in the round); fixed (causal up to training round 25, then the
batch value of evaluation round 25 for the rest of the day).
Rules: B0 entropy routing tau 0.8; B1 client exit; B2 server exit; B3 argmax (p_c + p_s)/2; M1 argmax p'; M2 M_k vs O_k
from p', the class inside M_k from p_c; variants of M1/M2 with batch / oracle / fixed r; device switch (causal r < 0.3
-> client exit, else server exit); kind oracle (Main -> client exit, else server exit).
Accuracy: per evaluation round the mean over clients, then the mean over the rounds of the window (full = 31 rounds;
late = evaluation rounds 30-150, 25 rounds). Decision on the late window, seed means (directive 4).
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
RUNS = Path(os.environ.get("R10_RUNS", JR / "runs" / "phaseT10_prior"))       # overrides only for smoke tests
OUT = Path(os.environ.get("R10_OUT", HERE))
LATE_MIN = int(os.environ.get("R10_LATE_MIN", 30))
SEEDS = [int(x) for x in os.environ.get("R10_SEEDS", "5,6,7").split(",")]
NAME = os.environ.get("R10_NAME", "r10_T40_s{seed}")
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_requests import build_partition  # noqa: E402

LAMBDA_BIG = 0.5
A_CLIP, R_CLIP = (0.01, 0.99), (0.02, 0.98)
EM_TOL, EM_MAX, MIN_PREV = 1e-4, 50, 8
SWITCH_R = 0.3
FIXED_LAST_ROUND = 25
TAU = np.float32(0.8)
R8_REF = {"B0": 62.54, "B1": 63.67, "B2": 62.59}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RULES = ["B0", "B1", "B2", "B3", "M1", "M2", "M1 batch", "M2 batch", "M1 oracle r", "M2 oracle r", "M1 fixed r",
         "M2 fixed r", "device switch", "kind oracle"]
KINDS = ["Main", "OOP", "OOR"]


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_TRAIN = None


def rebuild_mains(h):
    global _TRAIN
    if _TRAIN is None:
        from data.partition import get_cifar10
        _TRAIN = np.array(get_cifar10(str(ROOT / "data_cache"))[0].targets)
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    env, _ = r6_env.load_env(h["config"]["env_file"])
    _, mains, _, _ = build_partition(_TRAIN, env, int(h["config"]["partition_seed"]), cfg["num_classes"], cfg)
    return {int(k): sorted(int(c) for c in v) for k, v in mains.items()}


# ------------------------------------------------------------------------------------------------ EM
def em_full(Mv, Ov, a, b, r0):
    r = float(r0)
    for _ in range(EM_MAX):
        num = (r / b) * Ov
        rn = float(np.mean(num / (((1 - r) / a) * Mv + num)))
        done = abs(rn - r) < EM_TOL
        r = rn
        if done:
            break
    return float(np.clip(r, *R_CLIP))


def em_prefix(Mv, Ov, a, b, r0):
    """causal estimate for each request position j (arrival order): EM over requests 0..j-1, started at r0;
    positions with fewer than MIN_PREV earlier requests get r0."""
    n = len(Mv)
    out = np.full(n, float(r0))
    if n <= MIN_PREV:
        return out
    Mt = torch.as_tensor(Mv, dtype=torch.float64, device=DEV)
    Ot = torch.as_tensor(Ov, dtype=torch.float64, device=DEV)
    rows = torch.arange(MIN_PREV, n, device=DEV)
    mask = torch.arange(n, device=DEV)[None, :] < rows[:, None]
    cnt = rows.to(torch.float64)
    r = torch.full((len(rows),), float(r0), dtype=torch.float64, device=DEV)
    active = torch.ones(len(rows), dtype=torch.bool, device=DEV)
    for _ in range(EM_MAX):
        num = (r / b)[:, None] * Ot[None, :]
        q = torch.where(mask, num / (((1 - r) / a)[:, None] * Mt[None, :] + num), torch.zeros((), dtype=torch.float64, device=DEV))
        rn = q.sum(1) / cnt
        delta = (rn - r).abs()
        r = torch.where(active, rn, r)
        active = active & (delta >= EM_TOL)
        if not bool(active.any()):
            break
    out[MIN_PREV:] = np.clip(r.cpu().numpy(), *R_CLIP)
    return out


# ------------------------------------------------------------------------------------------------ per run
def load_run(seed):
    name = NAME.format(seed=seed)
    h = json.load(open(RUNS / f"{name}.json"))
    z = np.load(RUNS / f"{name}_trace.npz")
    src = RUNS / f"{name}_evalprobs.npz"
    q = dict(np.load(src))
    mains = rebuild_mains(h)
    prov = JR / "provenance" / f"{h['config']['run_id']}.json"
    prov_main = json.load(open(prov)).get("client_main") if prov.exists() else None
    K = 50
    er = np.array(q["eval_rounds"])
    E = len(er)
    M = np.zeros((K, 10), bool)
    for k, v in mains.items():
        M[k, v] = True
    e_i, k_i = q["req_eval_index"].astype(np.int64), q["req_client"].astype(np.int64)
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"].astype(np.int64)
    ps = q["ps"].astype(np.float64)
    pc = q["pc"].astype(np.float64)
    Mrow = M[k_i]
    checks = dict(mains_equal_provenance=(prov_main == {str(k): v for k, v in mains.items()}) if prov_main else None,
                  label_in_Mk_equals_kind_Main=bool(np.array_equal(kind == 0, M[k_i, y])),
                  argmax_ps_equals_sp=float((ps.argmax(1) == q["sp"]).mean()),
                  argmax_pc_equals_cp=float((pc.argmax(1) == q["cp"]).mean()),
                  source_sha256=sha256(src), source_bytes=src.stat().st_size, run_id=h["config"]["run_id"])
    # ---- pi_tr, a, b per (evaluation round, client)
    hc, ha, cells = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float), z["cells"]
    a_ek = np.zeros((E, K))
    empty_cells = 0
    for e, r in enumerate(er):
        t = int(r) - 1
        hall = ha[t] / ha[t].sum()
        for k in range(K):
            pis = []
            for zc in cells[t, k]:
                if zc < 0:
                    continue
                s_ = hc[t, int(zc)].sum()
                empty_cells += s_ == 0
                pis.append(LAMBDA_BIG * (hc[t, int(zc)] / s_ if s_ > 0 else hall) + (1 - LAMBDA_BIG) * hall)
            a_ek[e, k] = np.clip(np.mean(pis, axis=0)[M[k]].sum(), *A_CLIP)
    checks["cells_with_no_training_samples"] = int(empty_cells)
    Mv = (ps * Mrow).sum(1)
    Ov = (ps * ~Mrow).sum(1)
    # ---- r estimates
    N = len(y)
    r_causal = np.zeros(N)
    r_batch = np.zeros(N)
    r_oracle = np.zeros(N)
    r_fixed = np.zeros(N)
    batch_ek = np.zeros((E, K))
    order = np.lexsort((q["req_arrival"], k_i, e_i))     # by round, client, arrival
    bounds = np.flatnonzero(np.diff(e_i[order] * K + k_i[order])) + 1
    groups = np.split(order, bounds)
    gidx = {(int(e_i[g[0]]), int(k_i[g[0]])): g for g in groups}
    e25 = int(np.flatnonzero(er <= FIXED_LAST_ROUND)[-1])
    r_prev = np.full(K, 0.5)
    for e in range(E):
        for k in range(K):
            g = gidx[(e, k)]                                   # arrival order
            a, b = a_ek[e, k], 1 - a_ek[e, k]
            r_causal[g] = em_prefix(Mv[g], Ov[g], a, b, r_prev[k])
            batch_ek[e, k] = em_full(Mv[g], Ov[g], a, b, r_prev[k])
            r_batch[g] = batch_ek[e, k]
            r_oracle[g] = np.clip(np.mean(kind[g] != 0), *R_CLIP)
            r_prev[k] = batch_ek[e, k]
    late_fixed = er[e_i] > FIXED_LAST_ROUND
    r_fixed[:] = r_causal
    r_fixed[late_fixed] = batch_ek[e25, k_i[late_fixed]]
    D = dict(seed=seed, h=h, z=z, er=er, E=E, K=K, e=e_i, k=k_i, y=y, kind=kind, home_r=q["req_home"], checks=checks,
             r_causal=r_causal, n=np.bincount(e_i * K + k_i, minlength=E * K).reshape(E, K).astype(float))
    # true non-Main share per group (unclipped) for the estimation-quality table
    D["oracle_ek"] = np.bincount(e_i * K + k_i, weights=kind != 0, minlength=E * K).reshape(E, K) / D["n"]
    D["rhat_ek"] = np.bincount(e_i * K + k_i, weights=r_causal, minlength=E * K).reshape(E, K) / D["n"]
    home = np.zeros(E * K, bool)
    home[e_i * K + k_i] = q["req_home"]
    D["home"] = home.reshape(E, K)
    # ---- answers
    a_req = a_ek[e_i, k_i]

    def m1m2(r):
        b_req = 1 - a_req
        w = np.where(Mrow, ((1 - r) / a_req)[:, None], (r / b_req)[:, None])
        pp = ps * w
        pp /= pp.sum(1, keepdims=True)
        pM = (pp * Mrow).sum(1)
        pcM = (pc * Mrow).sum(1)
        s = np.where(Mrow, pM[:, None] * pc / (pcM[:, None] + 1e-8), pp)
        return pp.argmax(1), s.argmax(1)

    cp, sp = q["cp"].astype(np.int64), q["sp"].astype(np.int64)
    ans = {"B0": np.where(q["ent"] > TAU, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1)}
    srv = {"B0": q["ent"] > TAU, "B1": np.zeros(N, bool), "B2": np.ones(N, bool), "B3": np.ones(N, bool)}
    for tag, r in (("", r_causal), (" batch", r_batch), (" oracle r", r_oracle), (" fixed r", r_fixed)):
        m1, m2 = m1m2(r)
        ans["M1" + tag], ans["M2" + tag] = m1, m2
        srv["M1" + tag] = srv["M2" + tag] = np.ones(N, bool)
    sw = r_causal >= SWITCH_R
    ans["device switch"] = np.where(sw, sp, cp)
    srv["device switch"] = sw
    ans["kind oracle"] = np.where(kind == 0, cp, sp)
    srv["kind oracle"] = kind != 0
    D["correct"] = {rl: ans[rl] == y for rl in RULES}
    D["server"] = srv
    # B0 recomputation equals the installed-model counts of the run
    corr = np.bincount(e_i * K + k_i, weights=D["correct"]["B0"], minlength=E * K).reshape(E, K)
    checks["B0_counts_equal_eval_correct"] = bool(np.array_equal(np.rint(corr), z["eval_correct"]))
    print(f"  seed {seed}: {N} requests, checks {checks}", flush=True)
    return D


# ------------------------------------------------------------------------------------------------ metrics
def wmask(D, w):
    return np.ones(D["E"], bool) if w == "full" else D["er"] >= LATE_MIN


def per_round_acc(D, correct):
    g = D["e"] * D["K"] + D["k"]
    c = np.bincount(g, weights=correct, minlength=D["E"] * D["K"]).reshape(D["E"], D["K"])
    return (c / D["n"]).mean(axis=1), c / D["n"]


def metrics(D, rl, w):
    pr, a_ek = per_round_acc(D, D["correct"][rl])
    m = wmask(D, w)
    home = D["home"]
    out = dict(acc=pr[m].mean(),
               home=np.mean([a_ek[e][home[e]].mean() for e in np.flatnonzero(m) if home[e].any()]),
               away=np.mean([a_ek[e][~home[e]].mean() for e in np.flatnonzero(m) if (~home[e]).any()]))
    g = D["e"] * D["K"] + D["k"]
    G = D["E"] * D["K"]
    for i, kn in enumerate(KINDS):
        sel = D["kind"] == i
        ck = np.bincount(g, weights=D["correct"][rl] & sel, minlength=G).reshape(D["E"], D["K"])
        nk = np.bincount(g, weights=sel, minlength=G).reshape(D["E"], D["K"])
        with np.errstate(invalid="ignore", divide="ignore"):
            ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
        out[kn] = np.mean([np.nanmean(ak[e]) for e in np.flatnonzero(m) if np.isfinite(ak[e]).any()])
    inw = m[D["e"]]
    out["server_share"] = float(D["server"][rl][inw].mean())
    return out


def main():
    R = {s: load_run(s) for s in SEEDS}
    # ---------- checks (2.3) and run manifest
    rows = []
    for s, D in R.items():
        c = D["checks"]
        full = {rl: metrics(D, rl, "full")["acc"] * 100 for rl in ("B0", "B1", "B2")}
        rows.append([s, c["run_id"], c["mains_equal_provenance"], c["label_in_Mk_equals_kind_Main"],
                     c["B0_counts_equal_eval_correct"], f"{c['argmax_ps_equals_sp']:.6f}", f"{c['argmax_pc_equals_cp']:.6f}",
                     c["cells_with_no_training_samples"]] + [f"{full[rl]:.4f}" for rl in ("B0", "B1", "B2")]
                    + [c["source_sha256"], c["source_bytes"], f"{D['h']['total_time_sec'] / 60:.1f}"])
    mean_full = {rl: np.mean([metrics(R[s], rl, "full")["acc"] for s in SEEDS]) * 100 for rl in ("B0", "B1", "B2")}
    rows.append(["mean", "", "", "", "", "", "", ""] + [f"{mean_full[rl]:.4f}" for rl in ("B0", "B1", "B2")] + ["", "", ""])
    rows.append(["Round 8 report", "", "", "", "", "", "", ""] + [f"{R8_REF[rl]:.2f}" for rl in ("B0", "B1", "B2")] + ["", "", ""])
    rows.append(["difference", "", "", "", "", "", "", ""]
                + [f"{mean_full[rl] - R8_REF[rl]:+.4f}{' (>0.3 pp)' if abs(mean_full[rl] - R8_REF[rl]) > 0.3 else ''}"
                   for rl in ("B0", "B1", "B2")] + ["", "", ""])
    wcsv("R10_T0_checks_and_manifest.csv", ["seed", "run_id", "Mk_rebuilt_equals_provenance", "label_in_Mk_equals_kind_Main",
                                            "B0_counts_equal_installed_eval", "argmax_ps_fp16_equals_recorded_sp",
                                            "argmax_pc_fp16_equals_recorded_cp", "cells_with_no_training_samples",
                                            "B0_full_pct", "B1_full_pct", "B2_full_pct", "evalprobs_sha256",
                                            "evalprobs_bytes", "wall_min"], rows)
    # ---------- table 1
    MT = {(rl, w, s): metrics(R[s], rl, w) for rl in RULES for w in ("full", "late") for s in SEEDS}
    mean = lambda rl, w, key: float(np.mean([MT[(rl, w, s)][key] for s in SEEDS]))
    rows = []
    for rl in RULES:
        rows.append([rl, f"{mean(rl, 'full', 'acc') * 100:.2f}", f"{mean(rl, 'late', 'acc') * 100:.2f}"]
                    + [f"{mean(rl, 'late', key) * 100:.2f}" for key in ("home", "away", "Main", "OOP", "OOR")]
                    + [f"{mean(rl, 'late', 'server_share') * 100:.1f}", f"{mean(rl, 'full', 'server_share') * 100:.1f}"])
    wcsv("R10_T1_rules.csv", ["rule", "acc_full_pct", "acc_late_pct", "home_late_pct", "away_late_pct", "Main_late_pct",
                              "OOP_late_pct", "OOR_late_pct", "server_output_needed_late_pct",
                              "server_output_needed_full_pct"], rows)
    # ---------- table 2 per seed
    rows = []
    for rl in ("B0", "B1", "B2", "B3", "M1", "M2", "M1 oracle r", "M2 oracle r", "kind oracle"):
        for w in ("late", "full"):
            rows.append([rl, w] + [f"{MT[(rl, w, s)]['acc'] * 100:.2f}" for s in SEEDS] + [f"{mean(rl, w, 'acc') * 100:.2f}"])
    wcsv("R10_T2_per_seed.csv", ["rule", "window"] + [f"seed_{s}_pct" for s in SEEDS] + ["mean_pct"], rows)
    # ---------- table 3: estimation quality of the causal r
    rows = []
    for w in ("late", "full"):
        allx, ally = [], []
        for s in SEEDS:
            D = R[s]
            m = wmask(D, w)
            x, yv, hm = D["rhat_ek"][m].ravel(), D["oracle_ek"][m].ravel(), D["home"][m].ravel()
            allx.append(x)
            ally.append(yv)
            rows.append([w, s, f"{np.corrcoef(x, yv)[0, 1]:.4f}", f"{np.mean(np.abs(x - yv)):.4f}",
                         f"{x[hm].mean():.4f}", f"{yv[hm].mean():.4f}", f"{x[~hm].mean():.4f}", f"{yv[~hm].mean():.4f}"])
        x, yv = np.concatenate(allx), np.concatenate(ally)
        hm = np.concatenate([R[s]["home"][wmask(R[s], w)].ravel() for s in SEEDS])
        rows.append([w, "pooled", f"{np.corrcoef(x, yv)[0, 1]:.4f}", f"{np.mean(np.abs(x - yv)):.4f}",
                     f"{x[hm].mean():.4f}", f"{yv[hm].mean():.4f}", f"{x[~hm].mean():.4f}", f"{yv[~hm].mean():.4f}"])
    wcsv("R10_T3_r_estimation.csv", ["window", "seed", "pearson_rhat_vs_oracle", "mean_abs_error", "rhat_home",
                                     "oracle_home", "rhat_away", "oracle_away"], rows)
    # ---------- decision (late window, seed means)
    Bs = ("B0", "B1", "B2", "B3")
    bmax_late = max(Bs, key=lambda rl: mean(rl, "late", "acc"))
    bmax_full = max(Bs, key=lambda rl: mean(rl, "full", "acc"))
    mbest = max(("M1", "M2"), key=lambda rl: mean(rl, "late", "acc"))
    delta = (mean(mbest, "late", "acc") - mean(bmax_late, "late", "acc")) * 100
    mo = max(("M1 oracle r", "M2 oracle r"), key=lambda rl: mean(rl, "late", "acc"))
    delta_o = (mean(mo, "late", "acc") - mean(bmax_late, "late", "acc")) * 100
    full_ok = mean(mbest, "full", "acc") > mean(bmax_full, "full", "acc")
    if delta >= 2.0 and full_ok:
        verdict = "proceed"
    elif delta_o >= 2.0:
        verdict = "estimation is the bottleneck"
    else:
        verdict = "close"
    dec = dict(window="late (evaluation rounds 30-150)", verdict=verdict, delta_pp=delta, method=mbest,
               best_baseline_late=bmax_late, delta_oracle_r_pp=delta_o, oracle_method=mo,
               method_beats_best_baseline_full=bool(full_ok), best_baseline_full=bmax_full,
               means_late_pct={rl: mean(rl, "late", "acc") * 100 for rl in RULES},
               means_full_pct={rl: mean(rl, "full", "acc") * 100 for rl in RULES})
    json.dump(dec, open(OUT / "decision.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in dec.items() if not k.startswith("means")}, indent=1))
    figures(R, MT)


def figures(R, MT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    er = R[SEEDS[0]]["er"]

    def clock(ax):
        ticks = [5 * 60, 10 * 60, 15 * 60, 20 * 60]
        ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
        ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
        ax.set_xlim(1, 151)
        ax.set_xlabel("time of day")

    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
    style = [("B0", "entropy routing (tau 0.8)", INK, "--"), ("B1", "client exit only", INK2, ":"),
             ("B2", "server exit only", MUTED, "-."), ("M1", "M1: corrected server exit", BLUE, "-"),
             ("M2", "M2: corrected server exit + client exit", ORANGE, "-"), ("kind oracle", "request-kind oracle", PURPLE, "-")]
    for rl, lab, col, ls in style:
        y = np.mean([per_round_acc(R[s], R[s]["correct"][rl])[0] for s in SEEDS], axis=0) * 100
        ax.plot(er, y, color=col, ls=ls, marker="o", ms=2, label=lab)
    ax.axvline(LATE_MIN - 0.5, color=AXIS, lw=0.8, zorder=0)
    ax.set_ylabel("accuracy (%)")
    clock(ax)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.savefig(FIG / "R10_fig1_accuracy_by_time.pdf")
    fig.savefig(FIG / "R10_fig1_accuracy_by_time.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained", sharey=True)
    for ax, at_home, title in ((axes[0], True, "(a) clients at home"), (axes[1], False, "(b) clients away from home")):
        rh, ro = [], []
        for s in SEEDS:
            D = R[s]
            msk = D["home"] if at_home else ~D["home"]
            rh.append([D["rhat_ek"][e][msk[e]].mean() if msk[e].any() else np.nan for e in range(D["E"])])
            ro.append([D["oracle_ek"][e][msk[e]].mean() if msk[e].any() else np.nan for e in range(D["E"])])
        ax.plot(er, np.nanmean(rh, axis=0), color=BLUE, marker="o", ms=2, label="causal estimate of r")
        ax.plot(er, np.nanmean(ro, axis=0), color=INK, ls="--", marker="o", ms=2, label="true non-Main share")
        ax.axhline(SWITCH_R, color=MUTED, ls=":", lw=0.9)
        ax.set_title(title, fontsize=8)
        clock(ax)
    axes[0].set_ylabel("share of non-Main requests")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.savefig(FIG / "R10_fig2_r_by_time.pdf")
    fig.savefig(FIG / "R10_fig2_r_by_time.png", dpi=300)
    plt.close(fig)
    with open(FIG / "R10_figure_captions.md", "w") as f:
        f.write("# Round 10 figure captions\n\n## R10_fig1_accuracy_by_time\n\nAccuracy by time of day for entropy routing with "
                "threshold 0.8, the client exit alone, the server exit alone, the server exit corrected with the device's estimated "
                "request mix (M1), the corrected server exit combined with the client exit inside the client's own classes (M2), and "
                "the request-kind oracle. Commute mobility, training ratio 0.4, means over seeds 5 to 7. The vertical line marks the "
                "start of the late window used for the decision.\n\n## R10_fig2_r_by_time\n\nMean causal estimate of the share of "
                "requests outside the client's own classes and the true share, for clients at home (a) and away from home (b), by "
                "time of day. Means over seeds 5 to 7. The dotted line marks 0.3, the threshold of the device-level switch. Before 07:30 "
                "only a few clients are away, so the away curve rests on few clients there.\n")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
