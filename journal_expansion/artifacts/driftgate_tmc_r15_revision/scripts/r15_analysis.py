"""Round 15: re-check of the comparison, windows, replay, correction/weight/window and offloading (plan: R15_analysis_plan_ko.md).

Per-run loading and the rule functions come from Round 13b (scripts/r13b_baselines_analysis.py) and Round 14
(scripts/r14_paper_analysis.py: vectorized windows, client order, AUROC, LR coefficients from R13b_dev_LR.csv); both are
imported unchanged. New here: Logit-entropy weighting (LEW), the windows round > 30 / round > 50, correction-matched
variants, r and fixed-length window sensitivity, the partial-offloading DriftGate with both mean entropies over
offloaded requests, the online budget controller, and the scale composition.

Rules (plan section 5): B0 Confidence-based offloading (tau 0.8) | B1 Device only | B2 Edge only | B3 Probability average |
R-PoE Logit sum | R-THE Lower-entropy exit | R-ZTW Geometric ensemble with early exit (at the B0 offloading ratio of the
same window) | R-EM Label-shift EM | R-LR Learned weight (R13b coefficients) | LEW Logit-entropy weighting | DriftGate.
LEW: per request w in {0, 0.01, ..., 1} minimizing H(softmax(w z_d + (1 - w) z_e)), ties -> |w - 0.5| smallest, then
the smaller w; z = float32 log-softmax when the record has it (lpc, lps), otherwise log(max(p, 1e-8)) from the float16
probabilities (not an exact reconstruction; the share of float16 zeros is reported).
B1 variants: no correction + DriftGate's w; correction + w 0.5; correction + w 0.2 (Round 11 development choice);
corrected edge only; correction + learned weight (R-LR features from p'_e, regression refitted on the development
records' round >= 30 requests as in Round 13b); correction + product (exponents 1, 1).
B2: DriftGate with r 0.3 / 0.7 (existing window); fixed-length windows N 8 / 32 / 128 over the device's last N
offloaded requests (across rounds, current request excluded, fewer than 8 -> w 0.5); at full offloading and under the
online controller (beta 0.5).
Partial offloading (plan 7): sent = H_d > tau; DriftGate-P computes both mean entropies over the sent requests of the
window (earlier sent requests of the round when at least 8, else the sent requests of the device's last evaluation
round with requests); no window -> the device's previous w (arrival order across rounds), else 0.5. DriftGate-R14 is
the Round 14 curve (Hbar_d over all window requests). Online controller: tau = median of the device's previous 128
device-exit entropies (current excluded; fewer than 16 previous -> 0.8); offloaded requests answered by B0 (edge),
B3, Logit sum, DriftGate-P and the fixed-length windows.
Windows: full, gt30 (evaluation round > 30), gt50 (> 50). Accuracy as in the paper (per round mean over devices with
requests, then mean over rounds); splits as Round 14; seed SD ddof 1; differences per seed.
Outputs: tables/*.csv, figures/*, cache/ (per-run summaries; gitignored).
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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ART = JR / "artifacts"
os.environ.setdefault("R13B_OUT", str(HERE))
R14_PATH = ART / "driftgate_tmc_r14_paper" / "scripts" / "r14_paper_analysis.py"
_spec = importlib.util.spec_from_file_location("r14", R14_PATH)
m14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m14)
m13 = m14.m13
TAB, FIG, CACHE = HERE / "tables", HERE / "figures", HERE / "cache"
for d in (TAB, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)
RUNS12, RUNS13, RUNS15 = JR / "runs" / "phaseT12_fusion", JR / "runs" / "phaseT13b_arch", JR / "runs" / "phaseT15_replay"
DEVRUNS = JR / "runs" / "phaseT10_prior"
WORKERS = int(os.environ.get("R15_WORKERS", 3))
SMOKE_RUNS = os.environ.get("R15_SMOKE_RUNS")      # smoke tests only: "abs_dir:name,..."
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEP_SHA = hashlib.sha256(R14_PATH.read_bytes() + m14.R13B_PATH.read_bytes()).hexdigest()
TAU0, P_FLOOR = np.float32(0.8), 1e-8
TAU, QG, OPTS = m13.TAU, m13.QG, m13.OPTS
WGRID = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
W_DEV, R_GRID, N_GRID, BETA = 0.2, (0.3, 0.5, 0.7), (8, 32, 128), 0.5
CTRL_W, CTRL_MIN, CTRL_INIT = 128, 16, 0.8
WINDOWS = {"full": 0, "gt30": 30, "gt50": 50}
CORE = ["S1", "S2", "S1-fast", "partial participation", "stepwise change", "random mobility", "CIFAR-100", "ResNet-18"]
SCALE = ["K=200", "K=500"]
SETTINGS = {   # name -> (runs dir, pattern, seeds)
    "S1": (RUNS12, "r12_s1_fixed040_s{}", range(5)), "S2": (RUNS12, "r12_s2_fixed040_s{}", range(3)),
    "S1-fast": (RUNS12, "r12_s1fast_fixed040_s{}", range(3)), "partial participation": (RUNS12, "r12_s4part05_fixed040_s{}", range(3)),
    "stepwise change": (RUNS12, "r12_t1_A_fx40_s{}", range(5)), "random mobility": (RUNS12, "r12_t1_mob_fx40_s{}", range(3)),
    "CIFAR-100": (RUNS13, "r13b_c100_s{}", range(3)), "ResNet-18": (RUNS12, "r12_e2_res_fx40_A_s{}", range(3)),
    "K=200": (RUNS12, "r12_s3k200_fixed040_s{}", range(3)), "K=500": (RUNS12, "r12_s3k500_fixed040_s{}", range(3)),
    "S1 day 1 (retrained)": (RUNS15, "r15_s1_fixed040_s{}", range(5)), "S1 frozen replay": (RUNS15, "r15_s1_replay_s{}", range(5)),
    "S2 day 1 (retrained)": (RUNS15, "r15_s2_fixed040_s{}", range(3)), "S2 frozen replay": (RUNS15, "r15_s2_replay_s{}", range(3)),
}
A2 = ["S1 day 1 (retrained)", "S1 frozen replay", "S2 day 1 (retrained)", "S2 frozen replay"]
B_SET = ["S1", "S2", "random mobility", "CIFAR-100", "ResNet-18"] + A2
COMPARE = ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "R-ZTW", "R-EM", "R-LR", "LEW"]
SHOW = {"B0": "Confidence-based offloading", "B1": "Device only", "B2": "Edge only", "B3": "Probability average",
        "R-PoE": "Logit sum", "R-THE": "Lower-entropy exit", "R-ZTW": "Geometric ensemble with early exit",
        "R-EM": "Label-shift EM", "R-LR": "Learned weight", "LEW": "Logit-entropy weighting", "DriftGate": "DriftGate"}
BVAR = ["DriftGate", "no correction + adaptive w", "correction + w 0.5", "correction + dev w 0.2", "corrected edge only",
        "correction + learned weight", "correction + product"]
SPLITS = ["home", "away", "Main", "OOP", "OOR"]
KINDS = ["Main", "OOP", "OOR"]


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def lew_answers(zd, ze, dev, chunk=None):
    """per request: argmax of softmax(w* z_d + (1 - w*) z_e) and w* (entropy minimum over WGRID, tie rule)."""
    import torch
    chunk = chunk or max(1024, int(4e7 / (len(WGRID) * zd.shape[1])))
    wg = torch.as_tensor(WGRID, dtype=torch.float64, device=dev)
    tie = torch.abs(wg - 0.5) + 1e-6 * wg
    ans = np.empty(len(zd), np.int64)
    wstar = np.empty(len(zd), np.float32)
    for i in range(0, len(zd), chunk):
        a = torch.as_tensor(zd[i:i + chunk], dtype=torch.float64, device=dev)
        b = torch.as_tensor(ze[i:i + chunk], dtype=torch.float64, device=dev)
        z = wg[None, :, None] * a[:, None, :] + (1 - wg)[None, :, None] * b[:, None, :]      # [n, W, C]
        lp = torch.log_softmax(z, dim=2)
        H = -(lp.exp() * lp).sum(2)                                                            # [n, W]
        hmin = H.min(1, keepdim=True).values
        score = torch.where(H == hmin, tie[None, :], torch.full_like(H, 9.0))
        j = score.argmin(1)
        ans[i:i + chunk] = z[torch.arange(len(j), device=dev), j].argmax(1).cpu().numpy()
        wstar[i:i + chunk] = wg[j].float().cpu().numpy()
    return ans, wstar


def window_sum_counted(ix, v, c):
    """per request: sum of v*c and count of c over the window built from the requests with c = 1 (earlier ones of the
    same round when at least 8, else those of the device's previous round with requests); count 0 = no window."""
    order, starts, sizes, prev, gi, pos = (ix[k] for k in ("order", "starts", "sizes", "prev", "gi", "pos"))
    vs = (np.asarray(v, np.float64) * c)[order]
    cs_ = np.asarray(c, np.float64)[order]
    tv, tc = np.add.reduceat(vs, starts), np.add.reduceat(cs_, starts)
    pv = np.cumsum(vs) - vs
    pc = np.cumsum(cs_) - cs_
    pv, pc = pv - pv[starts][gi], pc - pc[starts][gi]
    use = pc >= m13.MIN_PREV
    pg = prev[gi]
    s = np.where(use, pv, np.where(pg >= 0, tv[np.maximum(pg, 0)], 0.0))
    n = np.where(use, pc, np.where(pg >= 0, tc[np.maximum(pg, 0)], 0.0))
    out_s, out_n = np.empty(len(vs)), np.empty(len(vs))
    out_s[order], out_n[order] = s, n
    return out_s, out_n


def fixed_window_w(co, Hc, Hs, sent, N):
    """w from the device's last N offloaded requests (client order: round, arrival; current request excluded);
    fewer than 8 earlier offloaded requests -> nan (cold start). Returns (w, cold)."""
    oc, first = co
    s_ = sent[oc].astype(np.int64)
    client_id = np.cumsum(first) - 1
    hi = np.cumsum(s_) - s_                                     # offloaded requests before i in the whole array
    k_before = hi - hi[np.flatnonzero(first)][client_id]        # ... within the device
    idx_sent = np.flatnonzero(s_)
    cum_hc = np.r_[0.0, np.cumsum(Hc[oc][idx_sent])]
    cum_hs = np.r_[0.0, np.cumsum(Hs[oc][idx_sent])]
    lo = np.maximum(hi - N, hi - k_before)
    hcb, hsb = cum_hc[hi] - cum_hc[lo], cum_hs[hi] - cum_hs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where((k_before >= m13.MIN_PREV) & (hcb + hsb > 0), hsb / (hcb + hsb), np.nan)
    w_out, cold = np.empty(len(w)), np.empty(len(w), bool)
    w_out[oc], cold[oc] = w, k_before < m13.MIN_PREV
    return w_out, cold


def cells_matrix(runs, name, K, er):
    """[E, K, 2] cells of every device at the evaluation rounds (-1 = none), as in the Round 13b loader."""
    runs = Path(runs)
    if (runs / f"{name}_trace.npz").exists():
        return np.load(runs / f"{name}_trace.npz")["cells"][np.asarray(er) - 1]
    z = np.load(runs / f"{name}_rec.npz")
    ce = z["eval_cells"]
    cid = np.array(z["client_ids"]) if "client_ids" in z.files else np.arange(ce.shape[1])
    out = np.full((len(er), K, 2), -1, np.int64)
    out[:, cid, :] = ce
    return out


def online_offload(co, Hc):
    """controller: offload if H_d > the (1 - beta) quantile (linear interpolation) of the device's previous 128 H_d;
    fewer than 16 previous requests -> tau 0.8. Client order: round, arrival; the current request is excluded."""
    from numpy.lib.stride_tricks import sliding_window_view
    oc, first = co
    x_all = Hc[oc].astype(np.float64)
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x_all)]
    tau = np.full(len(x_all), CTRL_INIT)
    for s0, e0 in zip(starts, ends):
        x = x_all[s0:e0]
        n = len(x)
        for i in range(CTRL_MIN, min(CTRL_W, n)):            # 16..127 previous requests
            tau[s0 + i] = np.quantile(x[:i], 1 - BETA)
        if n > CTRL_W:                                      # exactly 128 previous requests
            tau[s0 + CTRL_W:e0] = np.quantile(sliding_window_view(x[:-1], CTRL_W), 1 - BETA, axis=1)
    out = np.empty(len(x_all), bool)
    out[oc] = x_all > tau
    return out


def mix_answer(w, pd_, pe_):
    w = np.asarray(w, np.float32)[:, None]
    return (w * pd_ + (1 - w) * pe_).argmax(1)


# ------------------------------------------------------------------------------------------------ per run
def analyse(runs, name, lr, lr_corr):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    D = m13.load(Path(runs), name)
    q = np.load(Path(runs) / f"{name}_evalprobs.npz")
    exact = "lpc" in q.files
    pc, ps, Mrow, a, y = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"]
    cp, sp, ent, kind = D["cp"], D["sp"], D["ent"], D["kind"]
    N, E, K = len(y), D["E"], D["K"]
    er = D["er"]
    wm = {wn: er > b for wn, b in WINDOWS.items()}
    wts = {wn: m13.weights(D, m) for wn, m in wm.items()}
    psr = m13.corrected(ps, Mrow, a, 0.5)
    lpc_ = np.log(np.maximum(pc, P_FLOOR))
    ans = {"B0": np.where(ent > TAU0, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1),
           "R-PoE": (lpc_ + np.log(np.maximum(ps, P_FLOOR))).argmax(1),
           "correction + product": (lpc_ + np.log(np.maximum(psr, P_FLOOR))).argmax(1)}
    Hs = m13.server_entropy(ps)
    Hc = ent.astype(np.float64)
    ans["R-THE"] = np.where(Hc <= Hs, cp, sp)
    rhat = m13.r_causal(D, dev)
    ans["R-EM"] = m13.corrected(ps, Mrow, a, rhat.astype(np.float32)).argmax(1)
    feats = m13.features(D)
    z = ((feats - lr["mean"]) / lr["scale"]) @ lr["coef"] + lr["intercept"]
    P = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    ans["R-LR"] = mix_answer(P, pc, ps)
    # correction + learned weight: R-LR features from p'_e, regression refitted on the development records
    Dc = dict(D, ps=psr, sp=psr.argmax(1).astype(np.int64))
    fc = m13.features(Dc)
    fc[:, 1] = Hc
    Pc = lr_corr[1].predict_proba(lr_corr[0].transform(fc))[:, 1].astype(np.float32)
    ans["correction + learned weight"] = mix_answer(Pc, pc, psr)
    del feats, fc, Dc, z
    # LEW
    t0 = time.perf_counter()
    if exact:
        lew, wstar = lew_answers(q["lpc"], q["lps"], dev)
        lew16, _ = lew_answers(lpc_, np.log(np.maximum(ps, P_FLOOR)), dev)
        lew_agree_fp16 = float((lew == lew16).mean())
    else:
        lew, wstar = lew_answers(lpc_, np.log(np.maximum(ps, P_FLOOR)), dev)
        lew_agree_fp16 = None
    t_lew = time.perf_counter() - t0
    ans["LEW"] = lew
    fp16_zero = dict(device=float((pc == 0).mean()), edge=float((ps == 0).mean()))
    del lpc_
    # DriftGate (Round 13b F-auto, unchanged) and variants
    Hc_bar, Hs_bar = m13.window_means(D, [Hc, Hs])
    with np.errstate(invalid="ignore", divide="ignore"):
        w_auto = np.where(np.isfinite(Hc_bar) & ((Hc_bar + Hs_bar) > 0), Hs_bar / (Hc_bar + Hs_bar), 0.5)
    ans["DriftGate"] = mix_answer(w_auto, pc, psr)
    ans["no correction + adaptive w"] = mix_answer(w_auto, pc, ps)
    ans["correction + w 0.5"] = mix_answer(np.full(N, 0.5), pc, psr)
    ans["correction + dev w 0.2"] = mix_answer(np.full(N, W_DEV), pc, psr)
    ans["corrected edge only"] = psr.argmax(1)
    for r_ in R_GRID:
        if r_ != 0.5:
            ans[f"DriftGate r {r_}"] = mix_answer(w_auto, pc, m13.corrected(ps, Mrow, a, r_))
    ix = m14.window_index(D)
    co = m14.client_order(D)
    allsent = np.ones(N, bool)
    wfix = {}
    for Nw in N_GRID:
        wf, cold = fixed_window_w(co, Hc, Hs, allsent, Nw)
        wf = np.where(np.isnan(wf), 0.5, wf)
        ans[f"DriftGate window N {Nw}"] = mix_answer(wf, pc, psr)
        wfix[Nw] = dict(w_mean=float(wf.mean()), w_sd=float(wf.std()), cold=float(cold.mean()))
    corr = {k: (v == y) for k, v in ans.items()}
    # ---- curves
    cc = corr["B1"].astype(np.float64)
    curves = {wn: {} for wn in WINDOWS}
    j0 = np.searchsorted(TAU, ent, side="left")
    NJ = len(TAU) + 1
    jr = np.searchsorted(QG, pc.max(1), side="right")
    NQ = len(QG)
    for wn in WINDOWS:
        wv, sel = wts[wn], wm[wn][D["e"]]
        nsel = sel.sum()
        sc = corr["B2"].astype(np.float64)
        diff = np.bincount(j0, weights=wv * (cc - sc), minlength=NJ + 1)[:NJ]
        curves[wn]["SplitGP"] = dict(acc=(wv * sc).sum() + np.cumsum(diff),
                                     srv=1.0 - np.cumsum(np.bincount(j0, weights=sel.astype(float), minlength=NJ + 1)[:NJ]) / nsel)
        sc = corr["R-PoE"].astype(np.float64)
        diff = np.bincount(jr, weights=wv * (cc - sc), minlength=NQ + 2)
        accq = (wv * sc).sum() + (diff.sum() - np.cumsum(diff)[:NQ])
        srvq = np.cumsum(np.bincount(jr, weights=sel.astype(float), minlength=NQ + 2))[:NQ] / nsel
        curves[wn]["Geometric ensemble"] = dict(acc=np.append(accq, (wv * sc).sum())[::-1], srv=np.append(srvq, 1.0)[::-1])
    # DriftGate partial offloading (both entropies over sent requests) and the Round 14 definition
    hc_all_s, hc_all_n = m14.window_sum(ix, Hc)
    with np.errstate(invalid="ignore", divide="ignore"):
        hc_all = hc_all_s / hc_all_n
    dgp = {key: {wn: [] for wn in WINDOWS} for key in ("DriftGate-P", "DriftGate-R14")}
    srv_tau = {wn: [] for wn in WINDOWS}
    b1acc = {wn: float((wts[wn] * corr["B1"]).sum()) for wn in WINDOWS}
    for ti, tau in enumerate(list(TAU) + [np.inf]):
        sent = ent > tau
        for wn in WINDOWS:
            srv_tau[wn].append(float(sent[wm[wn][D["e"]]].mean()))
        if not sent.any():
            for key in dgp:
                for wn in WINDOWS:
                    dgp[key][wn].append(b1acc[wn])
            continue
        idx = np.flatnonzero(sent)
        sd_s, sd_n = window_sum_counted(ix, Hc, sent)
        ss_s, _ = window_sum_counted(ix, Hs, sent)
        with np.errstate(invalid="ignore", divide="ignore"):
            wP = np.where(sd_n > 0, np.where(sd_s + ss_s > 0, ss_s / (sd_s + ss_s), 0.5), np.nan)
        wP = m14.ffill_client(co, wP)
        hs_s14, _ = m14.window_sum(ix, Hs * sent)
        _, ns14 = m14.window_sum(ix, sent.astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            hs14 = hs_s14 / ns14
            w14 = np.where((hc_all_n > 0) & (ns14 > 0), np.where(hc_all + hs14 > 0, hs14 / (hc_all + hs14), 0.5), np.nan)
        w14 = np.where(hc_all_n > 0, w14, 0.5)
        w14 = m14.ffill_client(co, w14)
        for key, wv_ in (("DriftGate-P", wP), ("DriftGate-R14", w14)):
            c_ = corr["B1"].copy()
            c_[idx] = mix_answer(wv_[idx], pc[idx], psr[idx]) == y[idx]
            for wn in WINDOWS:
                dgp[key][wn].append(float((wts[wn] * c_).sum()))
            if ti == 0 and key == "DriftGate-P":
                corr["DriftGate-P tau0"] = c_
    for wn in WINDOWS:
        for key in dgp:
            curves[wn][key] = dict(acc=np.array(dgp[key][wn]), srv=np.array(srv_tau[wn]))
        for cn in curves[wn]:
            curves[wn][cn]["at"] = m13.interp(curves[wn][cn]["srv"], curves[wn][cn]["acc"], OPTS)
    del dgp
    # R-ZTW at the B0 offloading ratio of each window (scalar per window)
    ztw = {}
    for wn in WINDOWS:
        b0s = float((ent > TAU0)[wm[wn][D["e"]]].mean())
        cu = curves[wn]["Geometric ensemble"]
        ztw[wn] = dict(acc=float(m13.interp(cu["srv"], cu["acc"], [b0s])[0]), srv=b0s)
    srv_asc = np.cumsum(np.bincount(jr, minlength=NQ + 2))[:NQ] / N
    srv_asc = np.append(srv_asc, 1.0)
    b0f = ztw["full"]["srv"]
    jhi = min(max(int(np.searchsorted(srv_asc, b0f, side="left")), 1), len(srv_asc) - 1)
    jlo = jhi - 1
    alpha = 0.0 if srv_asc[jhi] == srv_asc[jlo] else (b0f - srv_asc[jlo]) / (srv_asc[jhi] - srv_asc[jlo])
    zc = lambda j: corr["R-PoE"] if j >= NQ else np.where(jr > j, corr["B1"], corr["R-PoE"])
    corr["R-ZTW"] = (1 - alpha) * zc(jlo) + alpha * zc(jhi)    # soft correctness: per-round values and splits only
    # ---- online controller (beta 0.5)
    off = online_offload(co, Hc)
    on = {}
    on_ans = {"Confidence-based offloading": np.where(off, sp, cp), "Probability average": np.where(off, ans["B3"], cp),
              "Logit sum": np.where(off, ans["R-PoE"], cp)}
    sd_s, sd_n = window_sum_counted(ix, Hc, off)
    ss_s, _ = window_sum_counted(ix, Hs, off)
    with np.errstate(invalid="ignore", divide="ignore"):
        wP = np.where(sd_n > 0, np.where(sd_s + ss_s > 0, ss_s / (sd_s + ss_s), 0.5), np.nan)
    wP = m14.ffill_client(co, wP)
    on_ans["DriftGate"] = np.where(off, mix_answer(wP, pc, psr), cp)
    on_w = {"DriftGate": wP[off]}
    for Nw in N_GRID:
        wf, cold = fixed_window_w(co, Hc, Hs, off, Nw)
        wf = np.where(np.isnan(wf), 0.5, wf)
        on_ans[f"DriftGate window N {Nw}"] = np.where(off, mix_answer(wf, pc, psr), cp)
        wfix[f"online N {Nw}"] = dict(w_mean=float(wf[off].mean()), w_sd=float(wf[off].std()), cold=float(cold[off].mean()))
    for k_, v in on_ans.items():
        corr["online: " + k_] = v == y
    # ---- metrics
    g, home, present = D["g"], D["home"], D["present"]
    start_min = 300.0 + 6.0 * (er - 1)
    res = {}
    for rl, c in corr.items():
        cf = c.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            A_ = np.where(present, np.bincount(g, weights=cf, minlength=E * K).reshape(E, K) / D["n_ek"], np.nan)
        per_round = np.nanmean(A_, axis=1)
        r_ = dict(per_round=per_round)
        for wn, m in wm.items():
            r_[wn] = float(per_round[m].mean())
            bottom = [np.sort(A_[e][present[e]])[: math.ceil(0.1 * present[e].sum())].mean() for e in np.flatnonzero(m)]
            r_[wn + "_bottom10"] = float(np.mean(bottom))
            if D["has_home"]:
                r_[wn + "_home"] = float(np.mean([A_[e][home[e] == 1].mean() for e in np.flatnonzero(m) if (home[e] == 1).any()]))
                r_[wn + "_away"] = float(np.mean([A_[e][home[e] == 0].mean() for e in np.flatnonzero(m) if (home[e] == 0).any()]))
            else:
                r_[wn + "_home"] = r_[wn + "_away"] = float("nan")
            for i, kn in enumerate(KINDS):
                selk = kind == i
                ck = np.bincount(g, weights=cf * selk, minlength=E * K).reshape(E, K)
                nk = np.bincount(g, weights=selk, minlength=E * K).reshape(E, K)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
                vals = [np.nanmean(ak[e]) for e in np.flatnonzero(m) if np.isfinite(ak[e]).any()]
                r_[wn + "_" + kn] = float(np.mean(vals)) if vals else float("nan")
        res[rl] = r_
    for wn in WINDOWS:   # window values: the curve at that window's B0 ratio (Round 14 definition)
        res["R-ZTW"][wn] = ztw[wn]["acc"]
    srv = {wn: {"B0": float((ent > TAU0)[wm[wn][D["e"]]].mean()), "online": float(off[wm[wn][D["e"]]].mean())} for wn in WINDOWS}
    # ---- composition (scale study) and request accounting
    cells_e = cells_matrix(runs, name, K, er)
    home_share = np.array([float((home[e][present[e]] == 1).mean()) if D["has_home"] else np.nan for e in range(E)])
    kinds_share = [float((kind == i).mean()) for i in range(3)]
    out = dict(name=name, res=res, curves=curves, srv=srv, exact_logprobs=exact, lew_agree_fp16=lew_agree_fp16,
               lew_w=dict(mean=float(wstar.mean()), sd=float(wstar.std()), share_0=float((wstar == 0).mean()),
                          share_1=float((wstar == 1).mean())), lew_seconds=t_lew, fp16_zero=fp16_zero,
               dg_w=dict(mean=float(w_auto.mean()), sd=float(w_auto.std())), wfix=wfix,
               on_w=dict(mean=float(on_w["DriftGate"].mean()), sd=float(on_w["DriftGate"].std())),
               er=er, start_min=start_min, home_share=home_share, has_home=D["has_home"], K=K, C=D["C"], n_req=N,
               kinds_share=kinds_share, a_mean=float(D["a_ek"][present].mean()),
               mains_per_client=float(np.mean(D["M"].sum(1)[np.unique(D["k"])])),
               auroc=m14.auroc(Hc, kind != 0), checks=D["checks"], run_id=D["h"]["config"]["run_id"],
               json_integrated=float(np.mean([e_["acc_total"] for e_ in D["h"]["eval"]])), seconds=time.time() - t_start)
    if True:
        two = (cells_e >= 0).sum(2) == 2
        out["two_cell_share"] = float(two[present].mean())
        nreq_two = float((D["n_ek"] * two).sum() / D["n_ek"].sum())
        out["edge_calls_per_offloaded"] = {"full": 1 + nreq_two}
        on_two = float(np.mean(two[D["e"], D["k"]][off])) if off.any() else float("nan")
        out["edge_calls_per_offloaded"]["online"] = 1 + on_two
        cells_members = {}
        for e in range(E):
            for k in np.flatnonzero(present[e]):
                for z in cells_e[e, k]:
                    if z >= 0:
                        cells_members.setdefault((e, int(z)), []).append(int(k))
        out["devices_per_cell"] = float(np.mean([len(v) for v in cells_members.values()]))
        out["own_classes_per_cell"] = float(np.mean([D["M"][v].any(0).sum() for v in cells_members.values()]))
    return out


def analyse_cached(args):
    runs, name, lr, lr_corr = args
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    key = hashlib.sha256(f"{SCRIPT_SHA}|{DEP_SHA}|{src}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cp_ = CACHE / f"{Path(runs).name}__{name}__{key}.pkl"
    if cp_.exists():
        return pickle.load(open(cp_, "rb"))
    out = analyse(runs, name, lr, lr_corr)
    out["sha"] = m13.sha256(src)
    pickle.dump(out, open(cp_, "wb"))
    print(f"  analysed {name} ({out['seconds'] / 60:.1f} min)", flush=True)
    return out


def run_many(tasks):
    if WORKERS <= 1:
        return [analyse_cached(t) for t in tasks]
    big = {i for i, t in enumerate(tasks) if (Path(t[0]) / f"{t[1]}_evalprobs.npz").stat().st_size > 1e9}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as exb, ProcessPoolExecutor(WORKERS, mp_context=ctx) as exs:
        fut = {i: (exb if i in big else exs).submit(analyse_cached, t) for i, t in enumerate(tasks)}
        return [fut[i].result() for i in range(len(tasks))]


def fit_lr_corr():
    """correction + learned weight: R-LR features from p'_e on the development records (round >= 30), refitted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = CACHE / f"lr_corr__{DEP_SHA[:16]}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    Xs, ys = [], []
    for s in m13.DEV_SEEDS:
        D = m13.load(DEVRUNS, f"r10_T40_s{s}")
        psr = m13.corrected(D["ps"], D["Mrow"], D["a_req"], 0.5)
        Dc = dict(D, ps=psr, sp=psr.argmax(1).astype(np.int64))
        f = m13.features(Dc)
        f[:, 1] = D["ent"].astype(np.float64)
        msk = (D["er"] >= 30)[D["e"]]
        Xs.append(f[msk])
        ys.append((D["kind"] == 0)[msk].astype(int))
    X, yv = np.concatenate(Xs), np.concatenate(ys)
    sc = StandardScaler().fit(X)
    model = LogisticRegression().fit(sc.transform(X), yv)
    out = (sc, model)
    pickle.dump(out, open(p, "wb"))
    wcsv("R15_B1_learned_weight_corrected_coefficients.csv", ["feature", "mean", "scale", "coefficient"],
         [[f, f"{m_:.6g}", f"{s_:.6g}", f"{c_:.6g}"] for f, m_, s_, c_ in zip(m13.FEATURES, sc.mean_, sc.scale_, model.coef_[0])]
         + [["intercept", "", "", f"{model.intercept_[0]:.6g}"], ["training requests", "", "", len(yv)],
            ["lbfgs iterations", "", "", int(model.n_iter_[0])]])
    return out


def lew_timing():
    """CPU time per request of LEW (101 weights) and of the DriftGate combination, numpy float32, C = 10 and 100."""
    rng = np.random.default_rng(0)
    rows = []
    for C in (10, 100):
        zd = np.log(rng.dirichlet(np.ones(C), 2000)).astype(np.float32)
        ze = np.log(rng.dirichlet(np.ones(C), 2000)).astype(np.float32)
        wg = WGRID.astype(np.float32)[:, None]
        t0 = time.perf_counter()
        for i in range(2000):
            z = wg * zd[i] + (1 - wg) * ze[i]
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            H = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
            int(np.argmax(z[int(np.argmin(H))]))
        t_lew = (time.perf_counter() - t0) / 2000
        pdv, pev = np.exp(zd), np.exp(ze)
        t0 = time.perf_counter()
        for i in range(2000):
            int(np.argmax(0.42 * pdv[i] + 0.58 * pev[i]))
        t_dg = (time.perf_counter() - t0) / 2000
        rows.append([C, f"{t_lew * 1e6:.1f}", f"{t_dg * 1e6:.2f}"])
    return rows


# ------------------------------------------------------------------------------------------------ main
def ms(v):
    v = np.asarray(v, float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def fmt(v, d=2):
    mu, sd = ms(v)
    return f"{mu:.{d}f} ({sd:.{d}f})"


def main():
    lr = m14.read_lr()
    lr_corr = fit_lr_corr()
    if SMOKE_RUNS:
        tasks = [(s.split(":")[0], s.split(":")[1], lr, lr_corr) for s in SMOKE_RUNS.split(",")]
        for a in run_many(tasks):
            print(a["name"], {k: round(a["res"][k]["full"] * 100, 3) for k in COMPARE + ["DriftGate"] + BVAR[1:]},
                  "| DriftGate-P at tau index 0 (all sent):", round(a["res"]["DriftGate-P tau0"]["full"] * 100, 3),
                  "| online srv", round(a["srv"]["full"]["online"], 3), "| LEW w", a["lew_w"], "| exact", a["exact_logprobs"],
                  "| LEW agree fp16", a["lew_agree_fp16"], "| fp16 zero", a["fp16_zero"], "| min", round(a["seconds"] / 60, 1))
        return
    avail = {st: [s for s in seeds if (rd / f"{pat.format(s)}_evalprobs.npz").exists() and (rd / f"{pat.format(s)}.json").exists()]
             for st, (rd, pat, seeds) in SETTINGS.items()}
    tasks, keys = [], []
    for st, (rd, pat, _) in SETTINGS.items():
        for s in avail[st]:
            tasks.append((str(rd), pat.format(s), lr, lr_corr))
            keys.append((st, s))
    A = dict(zip(keys, run_many(tasks)))
    R = {st: [A[(st, s)] for s in avail[st]] for st in SETTINGS if avail[st]}
    print("  settings analysed:", {st: len(v) for st, v in R.items()}, flush=True)
    val = lambda st, rl, key="full": np.array([a["res"][rl][key] for a in R[st]]) * 100
    write_tables(R, val)
    write_figures(R, val)


def strongest(R, val, st, wn):
    cand = {rl: val(st, rl, wn).mean() for rl in COMPARE}
    return max(cand, key=cand.get)


def write_tables(R, val):
    # ---- start check against Round 14 (full window, same runs)
    t14 = {(r["setting"], r["rule"]): r for r in csv.DictReader(open(ART / "driftgate_tmc_r14_paper" / "tables" / "R14_T1_main_long.csv"))}
    name14 = {"partial participation": "participation 0.5", "stepwise change": "stepwise change", "random mobility": "client mobility"}
    rows, bad = [], []
    for st in CORE + SCALE:
        if st not in R:
            continue
        for rl in ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "R-ZTW", "R-EM", "R-LR", "DriftGate"]:
            ref = t14.get((name14.get(st, st), rl))
            if ref is None:
                continue
            v = val(st, rl).mean()
            d = v - float(ref["mean_pct"])
            rows.append([st, rl, ref["mean_pct"], f"{v:.4f}", f"{d:+.5f}", abs(d) <= 0.05])
            if abs(d) > 0.05:
                bad.append((st, rl, d))
    wcsv("R15_T0_start_check_vs_R14.csv", ["setting", "rule", "R14_full_mean_pct", "R15_full_mean_pct", "difference_pp", "within_0.05pp"], rows)
    json.dump({"start_check_failures": bad}, open(TAB / "R15_T0_start_check_summary.json", "w"), indent=1)
    # ---- A1: rules per window
    for wn in WINDOWS:
        rows, longr = [], []
        sts = [st for st in CORE + SCALE + A2 if st in R]
        for rl in COMPARE + ["DriftGate"]:
            rows.append([SHOW[rl], rl] + [fmt(val(st, rl, wn)) for st in sts])
            for st in sts:
                for a, v in zip(R[st], val(st, rl, wn)):
                    longr.append([st, wn, rl, SHOW[rl], a["name"], f"{v:.4f}"])
        best = {st: strongest(R, val, st, wn) for st in sts}
        rows.append(["DriftGate - Confidence-based offloading", ""] + [fmt(val(st, "DriftGate", wn) - val(st, "B0", wn)) for st in sts])
        rows.append(["DriftGate - strongest baseline", ""] + [f"{fmt(val(st, 'DriftGate', wn) - val(st, best[st], wn))} vs {SHOW[best[st]]}" for st in sts])
        rows.append(["Device only - Confidence-based offloading", ""] + [fmt(val(st, "B1", wn) - val(st, "B0", wn)) for st in sts])
        orc = []
        for st in sts:
            per = np.array([[a["res"][rl][wn] for rl in COMPARE] for a in R[st]]) * 100
            orc.append(fmt(val(st, "DriftGate", wn) - per.max(1)))
        rows.append(["DriftGate - best baseline of each seed (seed-wise oracle comparison)", ""] + orc)
        rank = []
        for st in sts:
            means = {rl: val(st, rl, wn).mean() for rl in COMPARE + ["DriftGate"]}
            rank.append(str(1 + sum(v > means["DriftGate"] for k, v in means.items() if k != "DriftGate")))
        rows.append(["rank of DriftGate among the 11 rules (by mean)", ""] + rank)
        er_list = {st: " ".join(str(int(x)) for x in R[st][0]["er"][R[st][0]["er"] > WINDOWS[wn]]) for st in sts}
        rows.append(["evaluation rounds in the window", ""] + [er_list[st] for st in sts])
        wcsv(f"R15_A1_rules_{wn}.csv", ["rule (display)", "rule id"] + [f"{st} mean (SD)" for st in sts], rows)
        wcsv(f"R15_A1_rules_{wn}_per_seed.csv", ["setting", "window", "rule", "display", "run", "accuracy_pct"], longr)
    # ---- A1: time of day and contributions (S1, S2, A2)
    BINS = m14.BINS
    rows = []
    for st in [s for s in ("S1", "S2") + tuple(A2) if s in R]:
        best = strongest(R, val, st, "full")
        comps = {"DriftGate - Confidence-based offloading": ("DriftGate", "B0"), f"DriftGate - strongest ({SHOW[best]})": ("DriftGate", best),
                 "Device only - Confidence-based offloading": ("B1", "B0")}
        for b, lo, hi in BINS:
            msk = [(a["start_min"] >= lo) & (a["start_min"] < hi) for a in R[st]]
            row = [st, b, int(msk[0].sum())]
            for rl in ("B0", "B1", "B2", "DriftGate", best):
                row.append(fmt([a["res"][rl]["per_round"][m_].mean() * 100 for a, m_ in zip(R[st], msk)]))
            for cname, (x, y_) in comps.items():
                d = np.array([(a["res"][x]["per_round"][m_] - a["res"][y_]["per_round"][m_]).mean() * 100 for a, m_ in zip(R[st], msk)])
                contrib = np.array([m_.sum() / len(m_) for m_ in msk]) * d
                row += [fmt(d), fmt(contrib, 3)]
            row.append(fmt([np.nanmean(a["home_share"][m_]) * 100 for a, m_ in zip(R[st], msk)], 1))
            rows.append(row)
        tot = [st, "whole day (sum of contributions)", int(len(R[st][0]["er"]))] + [""] * 5
        for cname, (x, y_) in comps.items():
            tot_d = np.array([(a["res"][x]["full"] - a["res"][y_]["full"]) * 100 for a in R[st]])
            s_c = np.array([sum((((a["start_min"] >= lo) & (a["start_min"] < hi)).sum() / len(a["er"])) *
                                (a["res"][x]["per_round"][(a["start_min"] >= lo) & (a["start_min"] < hi)] -
                                 a["res"][y_]["per_round"][(a["start_min"] >= lo) & (a["start_min"] < hi)]).mean() * 100
                                for _, lo, hi in BINS) for a in R[st]])
            tot += [fmt(tot_d), f"{fmt(s_c, 3)}; max |sum - total| {np.abs(s_c - tot_d).max():.2e}"]
        tot.append("")
        rows.append(tot)
    hdr = ["setting", "time", "evaluation_points", "Confidence-based mean (SD)", "Device only", "Edge only", "DriftGate", "strongest baseline (full window)"]
    for c in ("DriftGate - Confidence-based", "DriftGate - strongest", "Device only - Confidence-based"):
        hdr += [f"{c} difference pp mean (SD)", f"{c} contribution pp mean (SD)"]
    wcsv("R15_A1_time_of_day.csv", hdr + ["share_at_home_pct"], rows)
    # ---- A1: splits, bottom 10%, ablation per window
    rows = []
    for st in [s for s in ("S1", "S2") + tuple(A2) if s in R]:
        for wn in WINDOWS:
            for rl in ["B0", "B1", "B2", "B3", "R-PoE", "LEW", "DriftGate"]:
                rows.append([st, wn, SHOW[rl]] + [fmt(val(st, rl, f"{wn}_{s_}")) for s_ in SPLITS])
    wcsv("R15_A1_splits.csv", ["setting", "window", "rule"] + [f"{s_} mean (SD)" for s_ in SPLITS], rows)
    rows = [[st, wn] + [fmt(val(st, rl, f"{wn}_bottom10")) for rl in ("B0", "B3", "R-PoE", "LEW", "DriftGate")]
            for st in ["S1", "S2", "S1-fast", "partial participation", "K=200", "K=500"] + A2 if st in R for wn in WINDOWS]
    wcsv("R15_A1_bottom10.csv", ["setting", "window"] + [SHOW[rl] for rl in ("B0", "B3", "R-PoE", "LEW", "DriftGate")], rows)
    # ---- B1 and B2
    rows = []
    for st in [s for s in B_SET if s in R]:
        for wn in ("full", "gt30"):
            for v_ in BVAR:
                d = val(st, v_, wn) - val(st, "DriftGate", wn)
                rows.append([st, wn, v_, fmt(val(st, v_, wn)), fmt(d) if v_ != "DriftGate" else "",
                             " ".join(f"{x:+.2f}" for x in d) if v_ != "DriftGate" else ""])
    wcsv("R15_B1_correction_weight.csv", ["setting", "window", "variant", "accuracy mean (SD)", "variant - DriftGate pp mean (SD)", "per seed"], rows)
    rows = []
    for st in [s for s in B_SET if s in R]:
        items = [("full offloading", "r 0.3", "DriftGate r 0.3", None), ("full offloading", "r 0.5 (DriftGate)", "DriftGate", "dg"),
                 ("full offloading", "r 0.7", "DriftGate r 0.7", None)]
        items += [("full offloading", f"window N {n}", f"DriftGate window N {n}", n) for n in N_GRID]
        items += [("online beta 0.5", "round-based window (DriftGate-P)", "online: DriftGate", "on")]
        items += [("online beta 0.5", f"window N {n}", f"online: DriftGate window N {n}", f"online N {n}") for n in N_GRID]
        for cond, lab, rl, wkey in items:
            if wkey == "dg":
                wst = [a["dg_w"] | {"cold": float("nan")} for a in R[st]]
            elif wkey == "on":
                wst = [a["on_w"] | {"cold": float("nan")} for a in R[st]]
            elif wkey is not None:
                wst = [{"mean": a["wfix"][wkey]["w_mean"], "sd": a["wfix"][wkey]["w_sd"], "cold": a["wfix"][wkey]["cold"]} for a in R[st]]
            else:
                wst = None
            wcols = ([f"{np.mean([w['mean'] for w in wst]):.3f}", f"{np.mean([w['sd'] for w in wst]):.3f}",
                      f"{np.mean([w['cold'] for w in wst]):.4f}"] if wst else ["", "", ""])
            rows.append([st, cond, lab, fmt(val(st, rl)), fmt(val(st, rl, "gt30"))] + [fmt(val(st, rl, f"full_{k_}")) for k_ in KINDS]
                        + wcols + [f"{np.mean([a['n_req'] for a in R[st]]):.0f}",
                                   fmt([a["srv"]["full"]["online"] for a in R[st]], 3) if cond != "full offloading" else "1.000"])
    wcsv("R15_B2_sensitivity.csv", ["setting", "condition", "variant", "full mean (SD)", "round>30 mean (SD)", "Main full", "OOP full", "OOR full",
                                    "w mean", "w SD", "cold-start share", "requests per run", "realized offloading"], rows)
    # ---- C1 offloading evidence
    FULLSET = ["B2", "B3", "R-PoE", "R-THE", "R-EM", "R-LR", "LEW"]
    rows, rows_s = [], []
    for st in [s for s in CORE + SCALE + A2 if s in R]:
        for wn in ("full", "gt30"):
            fm = {rl: val(st, rl, wn).mean() for rl in FULLSET}
            fbest = max(fm, key=fm.get)
            dm = dict(fm, B0=val(st, "B0", wn).mean(), B1=val(st, "B1", wn).mean(), **{"R-ZTW": val(st, "R-ZTW", wn).mean()})
            dbest = max(dm, key=dm.get)
            cu = {cn: np.mean([a["curves"][wn][cn]["acc"] for a in R[st]], axis=0) * 100 for cn in ("DriftGate-P", "DriftGate-R14")}
            sv = np.mean([a["curves"][wn]["DriftGate-P"]["srv"] for a in R[st]], axis=0)
            res_ = []
            for cn in ("DriftGate-P", "DriftGate-R14"):
                for target, tname in ((fm[fbest], "A_full"), (dm[dbest], "A_default")):
                    o = np.argsort(sv, kind="stable")
                    s_, a_ = sv[o], cu[cn][o]
                    reach_grid = next((float(s_[j]) for j in range(len(s_)) if a_[j] >= target), float("nan"))
                    exceed_grid = next((float(s_[j]) for j in range(len(s_)) if a_[j] > target), float("nan"))
                    reach_int = m13.first_reach(sv, cu[cn], target)
                    after = a_[s_ >= reach_grid] if np.isfinite(reach_grid) else np.array([])
                    redrop = bool((after < target).any()) if after.size else False
                    res_ += [f"{reach_grid:.3f}", f"{exceed_grid:.3f}", f"{reach_int:.3f}", redrop]
            ge = np.mean([a["curves"][wn]["Geometric ensemble"]["at"] for a in R[st]], axis=0) * 100
            dp = np.mean([a["curves"][wn]["DriftGate-P"]["at"] for a in R[st]], axis=0) * 100
            under = [(float(s_), float(a_ - fm[fbest])) for s_, a_ in zip(sv, cu["DriftGate-P"]) if s_ <= 0.5 and a_ > fm[fbest]]
            rows.append([st, wn, f"{val(st, 'B0', wn).mean():.2f}", f"{np.mean([a['srv'][wn]['B0'] for a in R[st]]):.3f}",
                         SHOW[fbest], f"{fm[fbest]:.2f}", SHOW.get(dbest, dbest), f"{dm[dbest]:.2f}"] + res_
                        + [" ".join(f"{x:+.2f}" for x in dp - ge), len(under), (f"{min(u[0] for u in under):.3f}" if under else "none")])
            for a in R[st]:
                for cn in ("DriftGate-P",):
                    rows_s.append([st, wn, a["name"], cn, f"{m13.first_reach(a['curves'][wn][cn]['srv'], a['curves'][wn][cn]['acc'] * 100, fm[fbest]):.3f}",
                                   f"{m13.first_reach(a['curves'][wn][cn]['srv'], a['curves'][wn][cn]['acc'] * 100, dm[dbest]):.3f}"])
    hdr = ["setting", "window", "B0 accuracy", "B0 offloading", "best full-offload baseline", "A_full", "best default-operating baseline", "A_default"]
    for cn in ("DriftGate-P", "DriftGate-R14"):
        for tn in ("A_full", "A_default"):
            hdr += [f"{cn} reach {tn} (grid point)", f"{cn} exceed {tn} (grid point)", f"{cn} reach {tn} (interpolated)", f"{cn} drops below {tn} after reaching"]
    hdr += ["DriftGate-P minus geometric ensemble at offloading 0.1..1.0 (pp)", "grid points with offloading <= 0.5 above A_full", "smallest such offloading"]
    wcsv("R15_C1_offload_evidence.csv", hdr, rows)
    wcsv("R15_C1_offload_reach_per_seed.csv", ["setting", "window", "run", "curve", "reach A_full of the setting (interpolated)", "reach A_default (interpolated)"], rows_s)
    rows = []
    for st in [s for s in CORE + SCALE + A2 if s in R]:
        for wn in ("full", "gt30"):
            for cn in ("SplitGP", "Geometric ensemble", "DriftGate-P", "DriftGate-R14"):
                at = np.array([a["curves"][wn][cn]["at"] for a in R[st]]) * 100
                rows.append([st, wn, cn] + [f"{at[:, i].mean():.2f} ({at[:, i].std(ddof=1) if len(at) > 1 else float('nan'):.2f})" for i in range(len(OPTS))])
    wcsv("R15_C1_curves.csv", ["setting", "window", "curve"] + [f"offloading_{o}" for o in OPTS], rows)
    # ---- C2 online controller
    rows = []
    for st in [s for s in ("S1", "S2") + tuple(A2) if s in R]:
        for rl in ("Confidence-based offloading", "Probability average", "Logit sum", "DriftGate"):
            rows.append([st, rl, fmt(val(st, "online: " + rl)), fmt(val(st, "online: " + rl, "gt30")),
                         fmt([a["srv"]["full"]["online"] for a in R[st]], 3),
                         f"{np.mean([a['edge_calls_per_offloaded']['online'] for a in R[st]]):.3f}" if "edge_calls_per_offloaded" in R[st][0] else ""])
    wcsv("R15_C2_online_controller.csv", ["setting", "answer for offloaded requests", "accuracy full mean (SD)", "accuracy round>30 mean (SD)",
                                          "realized offloading mean (SD)", "edge calls per offloaded request"], rows)
    # ---- E scale composition
    rows = []
    for st in [s for s in ("S1", "K=200", "K=500") if s in R]:
        best = strongest(R, val, st, "full")
        row = [st, R[st][0]["K"]]
        for rl in ("B1", "B2", "B0", "B3", "DriftGate", best):
            row.append(fmt(val(st, rl)))
        row.append(SHOW[best])
        for rl in ("B0", "DriftGate"):
            row += [fmt(val(st, rl, f"full_{s_}")) for s_ in SPLITS]
        row += [f"{np.mean([np.nanmean(a['home_share']) for a in R[st]]) * 100:.1f}",
                " ".join(f"{np.mean([a['kinds_share'][i] for a in R[st]]) * 100:.1f}" for i in range(3)),
                f"{np.mean([a['a_mean'] for a in R[st]]):.3f}", f"{np.mean([a.get('devices_per_cell', np.nan) for a in R[st]]):.1f}",
                f"{np.mean([a.get('own_classes_per_cell', np.nan) for a in R[st]]):.2f}", f"{np.mean([a.get('two_cell_share', np.nan) for a in R[st]]) * 100:.1f}",
                fmt([a["srv"]["full"]["B0"] for a in R[st]], 3), fmt([a["auroc"] for a in R[st]], 4)]
        rows.append(row)
    hdr = ["setting", "devices", "Device only", "Edge only", "Confidence-based", "Probability average", "DriftGate", "strongest baseline", "strongest name"]
    for rl in ("Confidence-based", "DriftGate"):
        hdr += [f"{rl} {s_}" for s_ in SPLITS]
    hdr += ["share of device-rounds at home %", "request kind shares Main OOP OOR %", "a_k mean", "devices per cell", "own classes per cell",
            "devices in two cells %", "B0 offloading", "AUROC of device entropy (non-Main)"]
    wcsv("R15_E_scale.csv", hdr, rows)
    # ---- P2 LEW details and timing
    rows = [[st, fmt(val(st, "LEW")), f"{np.mean([a['lew_w']['mean'] for a in R[st]]):.3f}", f"{np.mean([a['lew_w']['share_0'] for a in R[st]]):.3f}",
             f"{np.mean([a['lew_w']['share_1'] for a in R[st]]):.3f}", R[st][0]["exact_logprobs"],
             f"{np.mean([a['lew_agree_fp16'] for a in R[st]]):.5f}" if R[st][0]["lew_agree_fp16"] is not None else "",
             f"{np.mean([a['fp16_zero']['device'] for a in R[st]]):.5f}", f"{np.mean([a['fp16_zero']['edge'] for a in R[st]]):.5f}",
             f"{np.mean([a['lew_seconds'] / a['n_req'] * 1e6 for a in R[st]]):.3f}"] for st in R]
    wcsv("R15_P2_logit_entropy_weighting.csv", ["setting", "accuracy full mean (SD)", "mean w*", "share w* = 0 (edge)", "share w* = 1 (device)",
                                                "exact float32 logits", "answer agreement exact vs float16 reconstruction",
                                                "float16 zero share device", "float16 zero share edge", "GPU microseconds per request (batch)"], rows)
    wcsv("R15_P2_cpu_time.csv", ["classes", "LEW per request (us, numpy, 101 weights)", "DriftGate combination per request (us)"], lew_timing())
    # ---- A2 summary (day 1 vs replay) and run manifest of analysed runs
    rows = []
    for sc in ("S1", "S2"):
        d1, rp = f"{sc} day 1 (retrained)", f"{sc} frozen replay"
        if d1 in R and rp in R:
            for rl in COMPARE + ["DriftGate", "correction + dev w 0.2", "correction + w 0.5"]:
                rows.append([sc, SHOW.get(rl, rl), fmt(val(d1, rl)), fmt(val(rp, rl)), fmt(val(rp, rl) - val(d1, rl))]
                            + [fmt(val(rp, rl, f"full_{s_}")) for s_ in SPLITS])
    wcsv("R15_A2_replay.csv", ["scenario", "rule", "day 1 (retrained) full mean (SD)", "frozen replay full mean (SD)", "replay - day 1 pp"]
         + [f"replay {s_}" for s_ in SPLITS], rows)
    rows = [[st, a["name"], a["run_id"], a["n_req"], a["K"], a["C"], a["checks"]["mk_consistent"], a["checks"]["mk_prov"],
             f"{a['res']['B0']['full'] * 100:.4f}", f"{a['json_integrated'] * 100:.4f}", a["sha"], f"{a['seconds'] / 60:.1f}"]
            for st in R for a in R[st]]
    wcsv("R15_T0_analysed_runs.csv", ["setting", "run", "run_id", "requests", "devices", "classes", "Mk_consistent", "Mk_equals_provenance",
                                      "B0_full_recomputed_pct", "run_json_integrated_pct", "evalprobs_sha256", "analysis_min"], rows)


def write_figures(R, val):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5, "axes.edgecolor": "#b5b5b0", "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.3,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    COL = m14.COL
    save = lambda fig, n: (fig.savefig(FIG / f"{n}.pdf"), fig.savefig(FIG / f"{n}.png", dpi=300), plt.close(fig))
    pairs = [(d, r) for d, r in (("S1 day 1 (retrained)", "S1 frozen replay"), ("S2 day 1 (retrained)", "S2 frozen replay")) if d in R and r in R]
    if pairs:
        fig, axs = plt.subplots(1, len(pairs), figsize=(7.16, 2.1), layout="constrained", squeeze=False)
        for ax, (d1, rp) in zip(axs[0], pairs):
            hrs = R[d1][0]["start_min"] / 60
            for rl in ("B0", "B1", "B2", "DriftGate"):
                for st, ls in ((d1, ":"), (rp, "-")):
                    v = np.mean([a["res"][rl]["per_round"] for a in R[st]], axis=0) * 100
                    ax.plot(hrs, v, color=COL[rl], ls=ls, lw=1.6 if rl == "DriftGate" else 1.0,
                            label=f"{SHOW[rl]}, {'day 1' if st == d1 else 'replay'}")
            ax.set_title(d1.split()[0], fontsize=7)
            ax.set_xlim(5, 20)
            ax.set_xlabel("time of day (h)")
            ax.set_ylabel("accuracy (%)")
        axs[0][-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6)
        save(fig, "R15_fig_replay_time_of_day")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
