"""Round 14: tables and figures for the results and motivation sections of the paper (no new run, full window).

Inputs: Round 12 records (runs/phaseT12_fusion/r12_*), the Round 13b part-B CIFAR-100 runs (runs/phaseT13b_arch/
r13b_c100_s{0,1,2}), the Round 13b logistic-regression coefficients (tables/R13b_dev_LR.csv, used as written), Round 7
table T2_paired_differences.csv (device-level oracle) and Round 11 table R11_T3_auroc.csv (reference only).
Per-run loading and the rule functions come from scripts/r13b_baselines_analysis.py of Round 13b (imported; M_k,
pi_tr, a, b, arrival order, home/away, the Round 6 client-id rebuild, windows, EM, features, interpolation).

Rules (directive 3; paper names in brackets). p_c, p_s: stored probabilities; H_c: the record's client-exit entropy;
H_s: entropy of the stored p_s; p'_s: p_s corrected with r = 0.5 (Round 12).
  B0 [SplitGP] client exit when H_c <= 0.8, else server exit | B1 [Device only] | B2 [Edge only] | B3 [Average]
  R-PoE [FedRoD] argmax log p_c + log p_s (floor 1e-8) | R-THE [FedTHE] exit with the smaller entropy (tie: client)
  R-ZTW [ZTW] the ZTW curve (client exit when max p_c >= q, else R-PoE; q = 0, 0.01, ..., 1 and the all-server end
      point) at the server use of B0, full window (Round 13b interpolation)
  R-EM [Label-shift EM] Round 13b R-EM (causal EM over earlier requests, Round 10 M1)
  R-LR [Learned per-request weight] P(Main) = sigmoid(sum_i coef_i (x_i - mean_i) / scale_i + intercept) with the 9
      Round 11 features and the coefficients of R13b_dev_LR.csv; argmax [P p_c + (1 - P) p_s]
  DriftGate argmax [w p_c + (1 - w) p'_s], w = Hbar_s / (Hbar_c + Hbar_s) over the window (= Round 13b F-auto)
  ablation: Fixed w = 0.2 (= Round 13b F), Fixed w = 0.5, No calibration w = 0.2, No calibration w = 0.5 (= B3),
      Calibration + product (argmax log p_c + log p'_s)
Window of a request: earlier requests of the same client and evaluation round (arrival order) when at least 8,
otherwise all requests of the client's last evaluation round with requests; none -> w = 0.5.
Curves (full window; server use = share of requests that need the server exit):
  SplitGP: B0 with tau = 0, 0.05, ..., 2.30 and infinity; ZTW: as above;
  DriftGate: client exit when H_c <= tau, otherwise DriftGate with a w computed only from what the edge has seen:
      Hbar_c over all requests of the window, Hbar_s over the window's requests that were sent (H_c > tau); when the
      window holds no sent request, the client's previous w (its last request in arrival order, across evaluation
      rounds); when there is none, 0.5; when the window does not exist, 0.5 (as in 3.2). Same tau grid as SplitGP.
  Accuracy at server use 0.1, ..., 1.0 by linear interpolation between neighbouring grid points (Round 12 interp);
  the first server use at which the DriftGate curve reaches the accuracy of B0 (Round 13b first_reach).
Metrics (full window = all evaluation rounds): accuracy = per evaluation round the mean over clients with requests,
then the mean over rounds; home / away / Main / OOP / OOR as in Round 13b over all rounds; bottom 10% = per round the
mean of the lowest ceil(0.1 n) client accuracies (n = clients with requests), then the mean over rounds; time of day by
the start of the evaluation round (05:00 + 6 (r - 1) min): pre-commute [05:00, 07:30), commute [07:30, 09:30), daytime
[09:30, 16:00), return [16:00, 19:00), evening [19:00, 20:00]; share at home = share of clients with requests that are
in their home cell; AUROC of H_c for non-Main requests over all requests of a run (ties: average rank).
Seed statistics: mean and sample SD (ddof 1) over seeds; differences are per seed (same seed) before the mean.
Start check (directive 1.1): B0, B3 and DriftGate (full window, seed means, the 10 settings) equal Round 13b table 1
(B0_full, B3_full, F-auto_full) within 0.05 pp; otherwise the comparison is written and the script stops.
Outputs: tables/*.csv, figures/* (IEEE column 3.5 in, two columns 7.16 in; PDF and PNG), cache/ (per-run summaries).
"""
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ART = JR / "artifacts"
os.environ.setdefault("R13B_OUT", str(HERE))      # the imported Round 13b module creates tables/, figures/, cache/ here
R13B_PATH = ART / "driftgate_tmc_r13b_baselines" / "scripts" / "r13b_baselines_analysis.py"
_spec = importlib.util.spec_from_file_location("r13b", R13B_PATH)
m13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m13)
TAB, FIG, CACHE = HERE / "tables", HERE / "figures", HERE / "cache"
for d in (TAB, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)

RUNS12, RUNS13 = JR / "runs" / "phaseT12_fusion", JR / "runs" / "phaseT13b_arch"
SMOKE_RUNS = os.environ.get("R14_SMOKE_RUNS")      # smoke tests only: "dir:name,dir:name"
WORKERS = int(os.environ.get("R14_WORKERS", 3))
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
R13B_SHA = hashlib.sha256(R13B_PATH.read_bytes()).hexdigest()
TAU0, RSTR, P_FLOOR = np.float32(0.8), 0.5, 1e-8
TAU, QG, OPTS = m13.TAU, m13.QG, m13.OPTS
DAY_START, ROUND_MIN = 300.0, 6.0
BINS = [("pre-commute", 300, 450), ("commute", 450, 570), ("daytime", 570, 960), ("return", 960, 1140),
        ("evening", 1140, 1201)]
# setting -> (runs dir, fixed-lambda run pattern, seeds, DriftGate-lambda run pattern or None)
SETTINGS = {
    "S1": (RUNS12, "r12_s1_fixed040_s{}", range(5), "r12_s1_driftgate_own_s{}"),
    "S2": (RUNS12, "r12_s2_fixed040_s{}", range(3), None),
    "S1-fast": (RUNS12, "r12_s1fast_fixed040_s{}", range(3), None),
    "participation 0.5": (RUNS12, "r12_s4part05_fixed040_s{}", range(3), None),
    "stepwise change": (RUNS12, "r12_t1_A_fx40_s{}", range(5), "r12_r0_b1_relonly_A_s{}"),
    "client mobility": (RUNS12, "r12_t1_mob_fx40_s{}", range(3), "r12_r0_b1_relonly_mob_s{}"),
    "CIFAR-100": (RUNS13, "r13b_c100_s{}", range(3), None),
    "ResNet-18": (RUNS12, "r12_e2_res_fx40_A_s{}", range(3), "r12_r0_e2_res_relonly_A_s{}"),
    "K=200": (RUNS12, "r12_s3k200_fixed040_s{}", range(3), None),
    "K=500": (RUNS12, "r12_s3k500_fixed040_s{}", range(3), None),
}
R13B_GROUP = {"S1": ("Round 12", "S1"), "S2": ("Round 12", "S2"), "S1-fast": ("Round 12", "S1-fast"),
              "participation 0.5": ("Round 12", "participation 0.5"), "stepwise change": ("Round 12", "Schedule A"),
              "client mobility": ("Round 12", "client mobility"), "CIFAR-100": ("part B", "CIFAR-100"),
              "ResNet-18": ("Round 12", "ResNet-18"), "K=200": ("Round 12", "K=200"), "K=500": ("Round 12", "K=500")}
BASELINES = ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "R-ZTW", "R-EM", "R-LR"]
PAPER = {"B0": "SplitGP", "B1": "Device only", "B2": "Edge only", "B3": "Average", "R-PoE": "FedRoD", "R-THE": "FedTHE",
         "R-ZTW": "ZTW", "R-EM": "Label-shift EM", "R-LR": "Learned per-request weight", "DriftGate": "DriftGate"}
ABL = ["DriftGate", "Fixed w = 0.2", "Fixed w = 0.5", "No calibration, w = 0.2", "No calibration, w = 0.5",
       "Calibration + product"]
RULES = BASELINES + ["DriftGate"] + ABL[1:]
SPLITS = ["home", "away", "Main", "OOP", "OOR"]
KINDS = ["Main", "OOP", "OOR"]
CURVES = ["SplitGP", "ZTW", "DriftGate"]


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


def read_lr():
    rows = {r["feature"]: r for r in csv.DictReader(open(ART / "driftgate_tmc_r13b_baselines" / "tables" / "R13b_dev_LR.csv"))}
    feats = m13.FEATURES
    return dict(mean=np.array([float(rows[f]["mean"]) for f in feats]), scale=np.array([float(rows[f]["scale"]) for f in feats]),
                coef=np.array([float(rows[f]["coefficient"]) for f in feats]), intercept=float(rows["intercept"]["coefficient"]))


# ------------------------------------------------------------------------------------------------ windows (vectorized)
def window_index(D):
    """groups (round, client) in arrival order, rank within the group, previous group of the same client."""
    K = D["K"]
    order = np.lexsort((D["arrival"], D["k"], D["e"]))
    gs = D["g"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    gid = gs[starts]
    e_g, k_g = gid // K, gid % K
    o2 = np.lexsort((e_g, k_g))
    prev = np.full(len(starts), -1)
    same = k_g[o2][1:] == k_g[o2][:-1]
    prev[o2[1:][same]] = o2[:-1][same]
    gi = np.repeat(np.arange(len(starts)), sizes)
    pos = np.arange(len(order)) - starts[gi]
    return dict(order=order, starts=starts, sizes=sizes, prev=prev, gi=gi, pos=pos)


def window_sum(ix, v):
    """per request: sum of v over its window and the window's size (requests); nan / 0 when there is no window."""
    order, starts, sizes, prev, gi, pos = (ix[k] for k in ("order", "starts", "sizes", "prev", "gi", "pos"))
    vs = np.asarray(v, np.float64)[order]
    tot = np.add.reduceat(vs, starts)
    cs = np.cumsum(vs) - vs
    pref = cs - cs[starts][gi]
    use = pos >= m13.MIN_PREV
    pg = prev[gi]
    s = np.where(use, pref, np.where(pg >= 0, tot[np.maximum(pg, 0)], np.nan))
    n = np.where(use, pos, np.where(pg >= 0, sizes[np.maximum(pg, 0)], 0))
    out_s, out_n = np.empty(len(vs)), np.empty(len(vs))
    out_s[order], out_n[order] = s, n
    return out_s, out_n


def client_order(D):
    """requests in (client, round, arrival) order and the first request of every client in that order."""
    oc = np.lexsort((D["arrival"], D["e"], D["k"]))
    return oc, np.r_[True, D["k"][oc][1:] != D["k"][oc][:-1]]


def ffill_client(co, w):
    """nan -> the client's previous value in (round, arrival) order; nan at the start -> 0.5."""
    oc, first = co
    wv = w[oc]
    idx = np.maximum.accumulate(np.where(~np.isnan(wv) | first, np.arange(len(wv)), 0))
    f = wv[idx]
    out = np.empty_like(w)
    out[oc] = np.where(np.isnan(f), 0.5, f)
    return out


def auroc(score, pos):
    """AUROC of score for the positive class (ties: average rank)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ per run
def analyse(runs, name, lr):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = m13.load(Path(runs), name)
    pc, ps, Mrow, a, y = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"]
    cp, sp, ent, kind = D["cp"], D["sp"], D["ent"], D["kind"]
    N, E, K = len(y), D["E"], D["K"]
    allr = np.ones(E, bool)
    wf = m13.weights(D, allr)
    psr = m13.corrected(ps, Mrow, a, RSTR)
    lpc = np.log(np.maximum(pc, P_FLOOR))
    poe = (lpc + np.log(np.maximum(ps, P_FLOOR))).argmax(1)
    calprod = (lpc + np.log(np.maximum(psr, P_FLOOR))).argmax(1)
    del lpc
    Hs = m13.server_entropy(ps)
    the = np.where(ent.astype(np.float64) <= Hs, cp, sp)
    rhat = m13.r_causal(D, dev)
    em = m13.corrected(ps, Mrow, a, rhat.astype(np.float32)).argmax(1)
    feats = m13.features(D)
    z = ((feats - lr["mean"]) / lr["scale"]) @ lr["coef"] + lr["intercept"]
    P = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)[:, None]
    lr_ans = (P * pc + (1 - P) * ps).argmax(1)
    del feats, z
    Hc_bar, Hs_bar = m13.window_means(D, [ent.astype(np.float64), Hs])     # Round 13b F-auto, unchanged
    with np.errstate(invalid="ignore", divide="ignore"):
        w_auto = np.where(np.isfinite(Hc_bar) & ((Hc_bar + Hs_bar) > 0), Hs_bar / (Hc_bar + Hs_bar), 0.5)
    w32 = w_auto.astype(np.float32)[:, None]
    dg = (w32 * pc + (1 - w32) * psr).argmax(1)
    ans = {"B0": np.where(ent > TAU0, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1), "R-PoE": poe, "R-THE": the,
           "R-EM": em, "R-LR": lr_ans, "DriftGate": dg, "Fixed w = 0.2": (0.2 * pc + 0.8 * psr).argmax(1),
           "Fixed w = 0.5": (0.5 * pc + 0.5 * psr).argmax(1), "No calibration, w = 0.2": (0.2 * pc + 0.8 * ps).argmax(1),
           "Calibration + product": calprod}
    ans["No calibration, w = 0.5"] = ans["B3"]
    corr = {rl: (v == y).astype(np.float32) for rl, v in ans.items()}
    # ---- curves (full window)
    cc = corr["B1"]
    curves = {}
    j0 = np.searchsorted(TAU, ent, side="left")              # client answers at tau_j  <=>  j >= j0
    NJ = len(TAU) + 1
    sc = corr["B2"]
    diff = np.bincount(j0, weights=wf * (cc - sc), minlength=NJ + 1)[:NJ]
    accj = (wf * sc).sum() + np.cumsum(diff)
    srvj = 1.0 - np.cumsum(np.bincount(j0, minlength=NJ + 1)[:NJ]) / N
    curves["SplitGP"] = dict(acc=accj, srv=srvj)
    jr = np.searchsorted(QG, pc.max(1), side="right")         # client answers at q_j  <=>  j < jr
    NQ = len(QG)
    sc = corr["R-PoE"]
    diff = np.bincount(jr, weights=wf * (cc - sc), minlength=NQ + 2)
    accq = (wf * sc).sum() + (diff.sum() - np.cumsum(diff)[:NQ])
    srvq = np.cumsum(np.bincount(jr, minlength=NQ + 2))[:NQ] / N
    curves["ZTW"] = dict(acc=np.append(accq, (wf * sc).sum())[::-1], srv=np.append(srvq, 1.0)[::-1])   # non-increasing srv
    ix = window_index(D)
    co = client_order(D)
    hc_s, hc_n = window_sum(ix, ent.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        hc_bar = hc_s / hc_n
    defined = hc_n > 0
    acc_dg, srv_dg, w_sent_mean = [], [], []
    for tau in list(TAU) + [np.inf]:
        sent = ent > tau
        if not sent.any():
            acc_dg.append(float((wf * cc).sum()))
            srv_dg.append(0.0)
            w_sent_mean.append(float("nan"))
            continue
        s_s, s_n = window_sum(ix, Hs * sent)
        _, n_sent = window_sum(ix, sent.astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            hs_bar = s_s / n_sent
            w = np.where(defined & (n_sent > 0), np.where(hc_bar + hs_bar > 0, hs_bar / (hc_bar + hs_bar), 0.5), np.nan)
        w = np.where(defined, w, 0.5)
        w = ffill_client(co, w)
        idx = np.flatnonzero(sent)
        wi = w[idx].astype(np.float32)[:, None]
        c_ = cc.copy()
        c_[idx] = ((wi * pc[idx] + (1 - wi) * psr[idx]).argmax(1) == y[idx])
        acc_dg.append(float((wf * c_).sum()))
        srv_dg.append(float(sent.mean()))
        w_sent_mean.append(float(w[idx].mean()))
    curves["DriftGate"] = dict(acc=np.array(acc_dg), srv=np.array(srv_dg))
    for cn in CURVES:
        curves[cn]["at"] = m13.interp(curves[cn]["srv"], curves[cn]["acc"], OPTS)
    b0_srv = float((ent > TAU0).mean())
    ztw_val = float(m13.interp(curves["ZTW"]["srv"], curves["ZTW"]["acc"], [b0_srv])[0])
    # ---- per (round, client) accuracy and the metrics
    g, home = D["g"], D["home"]
    present = D["present"]
    acc_ek = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for rl, c in corr.items():
            acc_ek[rl] = np.where(present, np.bincount(g, weights=c, minlength=E * K).reshape(E, K) / D["n_ek"], np.nan)
    er = D["er"]
    start_min = DAY_START + ROUND_MIN * (er - 1)
    res = {}
    for rl, A_ in acc_ek.items():
        per_round = np.nanmean(A_, axis=1)
        bottom = []
        for e in range(E):
            v = np.sort(A_[e][present[e]])
            bottom.append(v[: math.ceil(0.1 * len(v))].mean())
        r_ = dict(full=float(per_round.mean()), per_round=per_round, bottom10=float(np.mean(bottom)))
        if D["has_home"]:
            r_["home"] = float(np.mean([A_[e][home[e] == 1].mean() for e in range(E) if (home[e] == 1).any()]))
            r_["away"] = float(np.mean([A_[e][home[e] == 0].mean() for e in range(E) if (home[e] == 0).any()]))
        else:
            r_["home"] = r_["away"] = float("nan")
        for i, kn in enumerate(KINDS):
            selk = kind == i
            ck = np.bincount(g, weights=corr[rl] * selk, minlength=E * K).reshape(E, K)
            nk = np.bincount(g, weights=selk, minlength=E * K).reshape(E, K)
            with np.errstate(invalid="ignore", divide="ignore"):
                ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
            vals = [np.nanmean(ak[e]) for e in range(E) if np.isfinite(ak[e]).any()]
            r_[kn] = float(np.mean(vals)) if vals else float("nan")
        r_["time"] = {b: float(per_round[(start_min >= lo) & (start_min < hi)].mean()) if ((start_min >= lo) & (start_min < hi)).any()
                      else float("nan") for b, lo, hi in BINS}
        res[rl] = r_
    res["R-ZTW"] = dict(full=ztw_val)
    home_share = np.array([float((home[e][present[e]] == 1).mean()) if D["has_home"] else np.nan for e in range(E)])
    h = D["h"]
    out = dict(name=name, res=res, curves=curves, b0_server=b0_srv, reach_b0=m13.first_reach(curves["DriftGate"]["srv"],
               curves["DriftGate"]["acc"], res["B0"]["full"]), dg_curve_w_sent_mean=w_sent_mean,
               auroc_Hc_nonmain=auroc(ent.astype(np.float64), kind != 0), er=er, start_min=start_min,
               home_share=home_share, has_home=D["has_home"], checks=D["checks"], n_req=N, K=K, C=D["C"],
               run_id=h["config"]["run_id"], json_integrated=float(np.mean([e_["acc_total"] for e_ in h["eval"]])),
               dg_w_mean=float(w_auto.mean()))
    return out


def analyse_cached(args):
    runs, name, lr = args
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    key = hashlib.sha256(f"{SCRIPT_SHA}|{R13B_SHA}|{src}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cp_ = CACHE / f"{Path(runs).name}__{name}__{key}.pkl"
    if cp_.exists():
        return pickle.load(open(cp_, "rb"))
    out = analyse(runs, name, lr)
    out["sha"] = m13.sha256(src)
    pickle.dump(out, open(cp_, "wb"))
    print(f"  analysed {name}", flush=True)
    return out


def run_many(tasks):
    """records above 1 GB (K = 500) one at a time in their own worker, the others in WORKERS workers."""
    if WORKERS <= 1:
        return [analyse_cached(t) for t in tasks]
    big = {i for i, t in enumerate(tasks) if (Path(t[0]) / f"{t[1]}_evalprobs.npz").stat().st_size > 1e9}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as exb, ProcessPoolExecutor(WORKERS, mp_context=ctx) as exs:
        fut = {i: (exb if i in big else exs).submit(analyse_cached, t) for i, t in enumerate(tasks)}
        return [fut[i].result() for i in range(len(tasks))]


# ------------------------------------------------------------------------------------------------ main
def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def fmt(v, d=2):
    mu, sd = ms(v)
    return f"{mu:.{d}f} ({sd:.{d}f})"


def main():
    lr = read_lr()
    if SMOKE_RUNS:
        tasks = [(str(JR / "runs" / s.split(":")[0]), s.split(":")[1], lr) for s in SMOKE_RUNS.split(",")]
        for a in run_many(tasks):
            print(a["name"], {rl: round(a["res"][rl]["full"] * 100, 3) for rl in RULES}, "| B0 server", round(a["b0_server"], 4),
                  "| DG curve first/last", np.round(a["curves"]["DriftGate"]["acc"][[0, -1]] * 100, 3),
                  "srv", np.round(a["curves"]["DriftGate"]["srv"][[0, -1]], 4), "| reach", a["reach_b0"],
                  "| AUROC", round(a["auroc_Hc_nonmain"], 4), "| checks", a["checks"])
        return
    tasks, keys = [], []
    for st, (rd, pat, seeds, dgpat) in SETTINGS.items():
        for s in seeds:
            tasks.append((str(rd), pat.format(s), lr))
            keys.append((st, "fixed", s))
            if dgpat:
                tasks.append((str(rd), dgpat.format(s), lr))
                keys.append((st, "DriftGate lambda", s))
    A = dict(zip(keys, run_many(tasks)))
    R = {st: [A[(st, "fixed", s)] for s in seeds] for st, (_, _, seeds, _) in SETTINGS.items()}
    val = lambda st, rl, k="full": np.array([a["res"][rl][k] for a in R[st]]) * 100

    # ---- start check against Round 13b table 1
    t13 = {(r["part"], r["scenario"]): r for r in csv.DictReader(open(ART / "driftgate_tmc_r13b_baselines" / "tables" / "R13b_T1_rules.csv"))
           if r["training"] == "fixed 0.4"}
    rows, bad = [], []
    pairs = [("B0", "B0"), ("B3", "B3"), ("DriftGate", "F-auto"), ("R-PoE", "R-PoE"), ("R-THE", "R-THE"), ("R-ZTW", "R-ZTW"),
             ("R-EM", "R-EM"), ("R-LR", "R-LR"), ("Fixed w = 0.2", "F"), ("Fixed w = 0.5", "F-eq"),
             ("No calibration, w = 0.2", "F-nodebias"), ("Calibration + product", "R-PoE-bal")]
    for st in SETTINGS:
        ref = t13[R13B_GROUP[st]]
        for rl, rl13 in pairs:
            v, r0 = val(st, rl).mean(), float(ref[f"{rl13}_full"])
            used = rl in ("B0", "B3", "DriftGate")
            ok = abs(v - r0) <= 0.05
            rows.append([st, rl, rl13, f"{r0:.2f}", f"{v:.4f}", f"{v - r0:+.4f}", ok, "start check" if used else "information"])
            if used and not ok:
                bad.append((st, rl, r0, v))
    wcsv("R14_T0_start_check.csv", ["setting", "rule", "R13b_rule", "R13b_table1_full_pct", "recomputed_full_pct", "difference_pp",
                                   "within_0.05pp", "use"], rows)
    if bad:
        raise SystemExit(f"start check failed (directive 1.1): {bad}")
    rows = [[st, kk[1], kk[2], a["name"], a["run_id"], a["n_req"], a["checks"]["mk_consistent"], a["checks"]["mk_prov"],
             a["checks"]["ids_rebuilt"], f"{a['res']['B0']['full'] * 100:.4f}", f"{a['json_integrated'] * 100:.4f}", a["sha"]]
            for kk, a in A.items() for st in [kk[0]]]
    wcsv("R14_T0_runs.csv", ["setting", "training", "seed", "run", "run_id", "requests", "Mk_consistent", "Mk_equals_provenance",
                             "client_ids_rebuilt", "B0_full_recomputed_pct", "run_json_integrated_pct", "evalprobs_sha256"], rows)

    # ---- table 1
    ST = list(SETTINGS)
    rows, longr = [], []
    for rl in BASELINES + ["DriftGate"]:
        rows.append([PAPER[rl], rl] + [fmt(val(st, rl)) for st in ST])
        for st in ST:
            mu, sd = ms(val(st, rl))
            longr.append([st, rl, PAPER[rl], len(R[st]), f"{mu:.4f}", f"{sd:.4f}"])
    best = {}
    for st in ST:
        cand = {rl: val(st, rl).mean() for rl in BASELINES if rl != "B0"}
        best[st] = max(cand, key=cand.get)
    rows.append(["DriftGate - SplitGP", ""] + [fmt(val(st, "DriftGate") - val(st, "B0")) for st in ST])
    rows.append(["DriftGate - best other baseline", ""] + [f"{fmt(val(st, 'DriftGate') - val(st, best[st]))} vs {PAPER[best[st]]}" for st in ST])
    wcsv("R14_T1_main.csv", ["rule (paper)", "rule"] + [f"{st} mean (SD)" for st in ST], rows)
    wcsv("R14_T1_main_long.csv", ["setting", "rule", "paper_name", "seeds", "mean_pct", "sd_pct"], longr)
    # ---- table 2
    rows = [[rl] + [fmt(val(st, rl)) for st in ("S1", "client mobility", "ResNet-18")] for rl in ABL]
    wcsv("R14_T2_ablation.csv", ["rule", "S1 mean (SD)", "client mobility mean (SD)", "ResNet-18 mean (SD)"], rows)
    # ---- table 3
    T3R = ["B0", "B1", "B2", "B3", "R-PoE", "DriftGate"]
    rows = [[st, PAPER[rl]] + [fmt(val(st, rl, s)) for s in SPLITS] for st in ("S1", "S2") for rl in T3R]
    wcsv("R14_T3_home_away_kinds.csv", ["setting", "rule"] + [f"{s} mean (SD)" for s in SPLITS], rows)
    # ---- table 4
    rows = []
    for st in ST:
        for cn in CURVES:
            at = np.array([a["curves"][cn]["at"] for a in R[st]]) * 100
            rows.append([st, cn] + [f"{at[:, i].mean():.2f} ({at[:, i].std(ddof=1):.2f})" for i in range(len(OPTS))])
    wcsv("R14_T4_curves.csv", ["setting", "curve"] + [f"server_use_{o}" for o in OPTS], rows)
    rows = [[st, fmt([a["b0_server"] for a in R[st]], 3), fmt([a["reach_b0"] for a in R[st]], 3),
             " ".join(f"{a['reach_b0']:.3f}" for a in R[st])] for st in ST]
    wcsv("R14_T4_reach.csv", ["setting", "B0_server_use mean (SD)", "DriftGate_curve_first_server_use_at_B0_accuracy mean (SD)",
                              "per_seed"], rows)
    # ---- table 5
    T5S = ["S1", "S2", "S1-fast", "participation 0.5", "K=200", "K=500"]
    T5R = ["B0", "B3", "R-PoE", "DriftGate"]
    wcsv("R14_T5_bottom10.csv", ["setting"] + [PAPER[rl] + " mean (SD)" for rl in T5R],
         [[st] + [fmt(val(st, rl, "bottom10")) for rl in T5R] for st in T5S])
    # ---- table 6
    rows = []
    for st in ("S1", "S2"):
        for b, _, _ in BINS:
            hs = [float(np.nanmean(a["home_share"][(a["start_min"] >= lo) & (a["start_min"] < hi)]))
                  for a in R[st] for (bb, lo, hi) in BINS if bb == b]
            n_e = int(sum(1 for (bb, lo, hi) in BINS if bb == b for x in R[st][0]["start_min"] if lo <= x < hi))
            rows.append([st, b, n_e] + [fmt([a["res"][rl]["time"][b] * 100 for a in R[st]]) for rl in ("B0", "B1", "B2", "DriftGate")]
                        + [fmt(np.array(hs) * 100, 1)])
    wcsv("R14_T6_time_of_day.csv", ["setting", "time", "evaluation_rounds", "SplitGP mean (SD)", "Device only mean (SD)",
                                    "Edge only mean (SD)", "DriftGate mean (SD)", "share_at_home_pct mean (SD)"], rows)
    # ---- table 7
    rows = []
    for st in ("stepwise change", "client mobility", "ResNet-18", "S1"):
        seeds = SETTINGS[st][2]
        d = np.array([A[(st, "DriftGate lambda", s)]["res"]["DriftGate"]["full"] - A[(st, "fixed", s)]["res"]["DriftGate"]["full"]
                      for s in seeds]) * 100
        rows.append([st, len(d), fmt(d), " ".join(f"{x:+.2f}" for x in d)])
    wcsv("R14_T7_training_adjustment.csv", ["setting", "seeds", "DriftGate_rule: lambda_adjustment_minus_fixed_0.4 mean (SD) pp",
                                            "per_seed_pp"], rows)
    # ---- table 8
    r11 = [r for r in csv.DictReader(open(ART / "driftgate_tmc_r11_fusion" / "tables" / "R11_T3_auroc.csv"))
           if r["devices"] == "all devices" and r["feature"] == "client entropy"][0]
    rows = [["S1", PAPER[rl]] + [fmt(val("S1", rl, s)) for s in SPLITS] for rl in ("B1", "B2")]
    rows.append(["S1", "AUROC of the client-exit entropy for non-Main requests", fmt([a["auroc_Hc_nonmain"] for a in R["S1"]], 4)]
                + [""] * 4)
    rows.append(["S1 development records (Round 11, seeds 5-7, reference)", "AUROC of the client-exit entropy", r11["auroc_mean"]] + [""] * 4)
    rows.append(["S1", "SplitGP server use (%)", fmt([a["b0_server"] * 100 for a in R["S1"]], 1)] + [""] * 4)
    wcsv("R14_T8_motivation.csv", ["setting", "rule or quantity"] + [f"{s} mean (SD)" for s in SPLITS], rows)
    json.dump({"start_check": "passed", "best_other_baseline": {st: PAPER[best[st]] for st in ST},
               "runs": len(A), "dg_curve_w_sent_mean_S1_seed0": R["S1"][0]["dg_curve_w_sent_mean"]},
              open(TAB / "R14_summary.json", "w"), indent=1)
    figures(R, val)


# ------------------------------------------------------------------------------------------------ figures
COL = {"B0": "#3b3b3b", "B1": "#9a9a9a", "B2": "#6b6b6b", "B3": "#2e9e6b", "R-PoE": "#e08a1e", "R-THE": "#8c6bb1",
       "R-ZTW": "#c49a00", "R-EM": "#4f8fbf", "R-LR": "#a05a2c", "DriftGate": "#d7263d"}
MK = {"B0": "s", "B1": "v", "B2": "^", "B3": "D", "R-PoE": "P", "DriftGate": "o"}


def figures(R, val):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": "#b5b5b0", "axes.grid": True,
                         "grid.color": "#e6e5df", "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.3, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})
    W1, W2 = 3.5, 7.16
    save = lambda fig, n: (fig.savefig(FIG / f"{n}.pdf"), fig.savefig(FIG / f"{n}.png", dpi=300), plt.close(fig))
    # 1 motivation
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.7), layout="constrained", gridspec_kw=dict(width_ratios=[2, 3]))
    for ax, cats, title in ((axs[0], ["home", "away"], "(a) clients at home / away"), (axs[1], KINDS, "(b) request kind")):
        x = np.arange(len(cats))
        for i, rl in enumerate(("B1", "B2")):
            mu = [val("S1", rl, c).mean() for c in cats]
            sd = [val("S1", rl, c).std(ddof=1) for c in cats]
            ax.bar(x + (i - 0.5) * 0.38, mu, 0.36, yerr=sd, color=COL[rl], label=PAPER[rl], error_kw=dict(lw=0.6, capsize=1.5), zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_title(title, fontsize=7)
    axs[0].set_ylabel("accuracy (%)")
    axs[1].legend(loc="upper right")
    save(fig, "R14_fig1_motivation")
    # 2 device-level oracle (Round 7)
    t2 = [r for r in csv.DictReader(open(ART / "driftgate_tmc_r7_gate" / "tables" / "T2_paired_differences.csv"))
          if r["comparison"].startswith("O (oracle)")]
    fig, ax = plt.subplots(figsize=(W1 * 0.6, 1.6), layout="constrained")
    labs = {"integrated": "whole day", "at home": "at home", "away": "away"}
    for i, r in enumerate(t2):
        ps_ = np.array([float(x) for x in r["per_seed_pp(5,6,7)"].split()])
        ax.bar(i, ps_.mean(), 0.6, yerr=ps_.std(ddof=1), color="#7a7a7a", error_kw=dict(lw=0.6, capsize=2), zorder=2)
        ax.plot(np.full(len(ps_), i) + np.array([-0.12, 0, 0.12]), ps_, ls="none", marker="o", ms=2.5, color="#222222", zorder=3)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(t2)))
    ax.set_xticklabels([labs[r["accuracy"]] for r in t2])
    ax.set_ylabel("oracle - fixed ratio (pp)")
    save(fig, "R14_fig2_device_oracle")
    # 3 main result
    ST = list(SETTINGS)
    fig, ax = plt.subplots(figsize=(W2, 1.8), layout="constrained")
    d = [val(st, "DriftGate") - val(st, "B0") for st in ST]
    ax.bar(range(len(ST)), [v.mean() for v in d], 0.62, yerr=[v.std(ddof=1) for v in d], color=COL["DriftGate"],
           error_kw=dict(lw=0.7, capsize=2), zorder=2)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(range(len(ST)))
    ax.set_xticklabels(ST)
    ax.set_ylabel("DriftGate - SplitGP (pp)")
    save(fig, "R14_fig3_main")
    # 4 home / away scatter
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2), layout="constrained")
    for ax, st in zip(axs, ("S1", "S2")):
        for rl in ("B0", "B1", "B2", "B3", "R-PoE", "DriftGate"):
            ax.plot(val(st, rl, "home").mean(), val(st, rl, "away").mean(), ls="none", marker=MK[rl], color=COL[rl],
                    ms=8 if rl == "DriftGate" else 6, label=PAPER[rl], zorder=3 if rl == "DriftGate" else 2)
        ax.set_title(st, fontsize=7)
        ax.set_xlabel("accuracy of clients at home (%)")
        ax.set_ylabel("accuracy of clients away (%)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "R14_fig4_home_away")
    # 5 selective offload
    grid = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2), layout="constrained")
    for ax, st in zip(axs, ("S1", "S2")):
        for cn, c_, ls_, lab_ in (("SplitGP", COL["B0"], "-", "SplitGP (entropy threshold)"), ("ZTW", COL["R-ZTW"], "-.", "ZTW"),
                                  ("DriftGate", COL["DriftGate"], "-", "DriftGate")):
            v = np.nanmean([m13.interp(a["curves"][cn]["srv"], a["curves"][cn]["acc"], grid) for a in R[st]], axis=0) * 100
            ax.plot(grid, v, color=c_, ls=ls_, lw=1.8 if cn == "DriftGate" else 1.2, label=lab_)
        ax.axvline(float(np.mean([a["b0_server"] for a in R[st]])), color="#888888", lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(st, fontsize=7)
        ax.set_xlabel("server use (share of requests sent to the edge)")
        ax.set_ylabel("accuracy (%)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "R14_fig5_offload")
    # 6 time of day (S1)
    a0 = R["S1"][0]
    hrs = a0["start_min"] / 60.0
    fig, ax = plt.subplots(figsize=(W2, 2.0), layout="constrained")
    ax2 = ax.twinx()
    ax2.fill_between(hrs, np.mean([a["home_share"] for a in R["S1"]], axis=0) * 100, color="#d9d6cc", alpha=0.45, lw=0, step="mid")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("clients at home (%)", color="#8a877c")
    ax2.tick_params(axis="y", colors="#8a877c")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for rl in ("B0", "B1", "B2", "DriftGate"):
        v = np.mean([a["res"][rl]["per_round"] for a in R["S1"]], axis=0) * 100
        ax.plot(hrs, v, color=COL[rl], lw=1.8 if rl == "DriftGate" else 1.1, marker="o", ms=2, label=PAPER[rl])
    for _, lo, _ in BINS[1:]:
        ax.axvline(lo / 60.0, color="#bbbbbb", lw=0.5, ls="--")
    ax.set_xlim(5, 20)
    ax.set_xticks(range(5, 21))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower left", ncol=4)
    save(fig, "R14_fig6_time_of_day")
    with open(FIG / "R14_figure_captions.md", "w") as f:
        f.write("""# Round 14 figure captions

## R14_fig1_motivation
Accuracy of the device-only and edge-only exits in the commute scenario S1 (training ratio 0.4, seeds 0 to 4, whole day): (a) clients in their home cell and clients in another cell; (b) requests of the device's own training classes (Main), of classes trained only by other clients of the current cell (OOP), and of other classes (OOR). Error bars: standard deviation over seeds.

## R14_fig2_device_oracle
Accuracy gain of a device-level oracle that sets the personalization ratio from the true home/away state (0.70 at home, 0.15 away) over a fixed ratio of 0.4, for the whole day, clients at home and clients away (Round 7, S1, seeds 5 to 7). Bars: mean; error bars: standard deviation; dots: seeds.

## R14_fig3_main
Accuracy of DriftGate minus SplitGP (entropy routing with threshold 0.8) over the whole day in the ten settings. Error bars: standard deviation of the per-seed difference.

## R14_fig4_home_away
Accuracy of clients at home against accuracy of clients away for each inference rule in S1 (seeds 0 to 4) and S2 (GeoLife traces, seeds 0 to 2), whole day.

## R14_fig5_offload
Accuracy against the share of requests sent to the edge when only requests whose device-exit entropy exceeds a threshold are sent: SplitGP answers them with the edge exit, ZTW with the product of both exits (device confidence threshold), and DriftGate with its fusion computed from the requests the edge has received. Whole day, seed means. The dotted line marks the server use of SplitGP at threshold 0.8.

## R14_fig6_time_of_day
Accuracy over the day in S1 for SplitGP, the device-only exit, the edge-only exit and DriftGate (seed means per evaluation round). The shaded background is the share of clients in their home cell (right axis); dashed lines separate pre-commute, commute, daytime, return and evening.
""")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
