"""Round 16 stage 1: oracles and simulated selectors from the existing records.

Rules: decision_rule.md; settings: r16_config.json (read here). The reference rules use the Round 15 definitions:
scripts/r15_analysis.py of driftgate_tmc_r15_revision is imported unchanged for its helpers and constants, and the code
that builds the reference answers below repeats r15_analysis.analyse line by line (start check against the Round 15 cache).

Subcommands (run from the repository root):
  manifest      evaluation_manifest.csv (evaluation rounds per row; no accuracy)
  check         phase A: per run, the 17 reference answers (cache/, gitignored) and the start check against the Round 15
                cache -> tables/R16_T0_start_check.csv, cache/START_CHECK.json
  run           phase B (only after a passed start check): grid, simulated calibration, Main-share estimate, selectors,
                oracles, online controller; then all stage-1 tables
  smoke NAME    phase B on one development record (runs/phaseT10_prior), printed only

Candidates: 90 fixed pairs (w, r) in priority order (w ascending; r 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, none)
and DriftGate (index 0 of every score matrix). Calibration of device k at evaluation round e: its requests of the most
recent earlier evaluation round with >= 8 Main and >= 8 non-Main requests (outputs, prior and cells of that round).
DriftGate's calibration score uses the weight of the current request: correctness of argmax(w p_d + (1 - w) p'_e) on a
calibration sample is an open interval of w (lo, hi), computed in float64 from the stored probabilities.
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
CFG = json.loads((HERE / "r16_config.json").read_text())
R15_PATH = ART / "driftgate_tmc_r15_revision" / "scripts" / "r15_analysis.py"
_spec = importlib.util.spec_from_file_location("r15", R15_PATH)
m15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m15)
m14, m13 = m15.m14, m15.m13
TAB, CACHE = HERE / "tables", HERE / "cache"
for d_ in (TAB, CACHE):
    d_.mkdir(parents=True, exist_ok=True)
WORKERS = int(os.environ.get("R16_WORKERS", 3))
VERBOSE = os.environ.get("R16_VERBOSE") == "1"


def tick(t0, msg):
    if VERBOSE:
        print(f"    [{time.time() - t0:7.1f} s] {msg}", flush=True)
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes() + (HERE / "r16_config.json").read_bytes()).hexdigest()
REFS_VERSION = "refs-v1"          # phase A cache: changes only when the reference code below changes

TAU0, P_FLOOR, TAU, QG = m15.TAU0, m15.P_FLOOR, m15.TAU, m15.QG
MIN_PREV = m13.MIN_PREV
W_GRID = [float(w) for w in CFG["w_grid"]]
R_ORDER = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, None]
assert sorted(x for x in R_ORDER if x is not None) == [x for x in CFG["r_grid"] if x != "none"]
GRID = [(w, r) for w in W_GRID for r in R_ORDER]                   # priority order; candidate index = 1 + grid index
NG = len(GRID)
R_NUM = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
R_PRI = sorted(range(9), key=lambda i: (round(abs(R_NUM[i] - 0.5), 9), R_NUM[i]))
TIE = CFG["score_tie_tolerance"]
MINC = CFG["calibration"]["min_main"]
assert MINC == CFG["calibration"]["min_nonmain"] == CFG["online"]["min_group"] == CFG["online"]["min_prior_offloads"] == MIN_PREV
CLIP = tuple(CFG["share_estimate"]["clip"])
MIN_CD = CFG["share_estimate"]["min_abs_c11_minus_c01"]
MIN_VD = CFG["online"]["min_abs_v_diff"]
SA = CFG["sA_candidates"]
PASS_TOL, HOLD_FLOOR = CFG["decision"]["pass_tolerance_pp"], CFG["decision"]["hold_floor_pp"]
WIN_DAY1 = ["day1_full", "day1_early", "day1_gt30", "day1_gt50"]
BINS = [tuple(b) for b in CFG["time_bins_minutes"]]
assert BINS == [tuple(b) for b in m14.BINS]

REF = ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "LEW", "R-ZTW", "R-EM", "R-LR", "DriftGate",
       "no correction + adaptive w", "correction + w 0.5", "correction + dev w 0.2", "corrected edge only",
       "correction + learned weight", "correction + product"]
SHOW = dict(m15.SHOW, **{k: k for k in m15.BVAR[1:]})
assert [SHOW[k] for k in REF] == CFG["reference_rules"]
ANS_KEYS = [k for k in REF if k != "R-ZTW"]
ONLINE_OPS = {   # online reference -> answer key used on offloaded requests (Device only: offloading 0)
    "Device only (offloading 0)": None, "Edge answer (Confidence-based offloading, Edge only)": "B2",
    "Probability average": "B3", "Logit sum (also Geometric ensemble)": "R-PoE", "Lower-entropy exit": "R-THE",
    "Logit-entropy weighting": "LEW", "Label-shift EM": "R-EM", "Learned weight": "R-LR",
    "DriftGate (DriftGate-P)": "DG-P", "no correction + adaptive w": "DG-P nocorr", "correction + w 0.5": "correction + w 0.5",
    "correction + dev w 0.2": "correction + dev w 0.2", "corrected edge only": "corrected edge only",
    "correction + learned weight": "correction + learned weight", "correction + product": "correction + product"}
assert list(ONLINE_OPS) == CFG["online_reference_operations"]
SELECTORS = ["S-A", "S-B0", "S-B5", "S-Bg", "S-C"]


def runs_dir(st):
    return JR / CFG["settings"][st]["runs"]


def run_name(st, s):
    return CFG["settings"][st]["pattern"].format(s)


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


def ms(v):
    v = np.asarray(v, float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def fmt(v, d=2):
    mu, sd = ms(v)
    return f"{mu:.{d}f} ({sd:.{d}f})"


# ------------------------------------------------------------------------------------------------ shared pieces
def window_masks(er, period):
    if period == "replay":
        return {"replay_full": np.ones(len(er), bool)}
    return {"day1_full": np.ones(len(er), bool), "day1_early": er <= 30, "day1_gt30": er > 30, "day1_gt50": er > 50}


def bin_masks(er):
    start = 300.0 + 6.0 * (er - 1)
    return {b: (start >= lo) & (start < hi) for b, lo, hi in BINS}


def per_round(D, c):
    """as r15_analysis: per evaluation round, mean over devices with requests."""
    E, K = D["E"], D["K"]
    cf = c.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ = np.where(D["present"], np.bincount(D["g"], weights=cf, minlength=E * K).reshape(E, K) / D["n_ek"], np.nan)
    return np.nanmean(A_, axis=1)


def per_round_groups(D, counts):
    """per-round mean over devices with requests of counts[E*K] / n_ek."""
    E, K = D["E"], D["K"]
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ = np.where(D["present"], counts.reshape(E, K) / D["n_ek"], np.nan)
    return np.nanmean(A_, axis=1)


def mix_scalar(w, pd_, pe_, chunk=1 << 21):
    """argmax(w p_d + (1 - w) p_e) with a scalar w, float32 as r15_analysis.mix_answer."""
    w32 = np.float32(w)
    one_m = np.float32(1) - w32
    out = np.empty(len(pd_), np.int64)
    for s in range(0, len(pd_), chunk):
        out[s:s + chunk] = (w32 * pd_[s:s + chunk] + one_m * pe_[s:s + chunk]).argmax(1)
    return out


def corrected_chunked(ps, Mrow, a, r, chunk=1 << 21):
    out = np.empty_like(ps)
    rr = np.asarray(r, np.float32)
    for s in range(0, len(ps), chunk):
        out[s:s + chunk] = m13.corrected(ps[s:s + chunk], Mrow[s:s + chunk], a[s:s + chunk], rr[s:s + chunk] if rr.ndim else rr)
    return out


def ztw(D, corr_b1, corr_poe, masks):
    """Geometric ensemble with early exit as r15_analysis: per window, the curve at that window's B0 ratio; soft
    correctness at the full-window B0 ratio for per-round values."""
    pc, ent, N = D["pc"], D["ent"], len(D["y"])
    jr = np.searchsorted(QG, pc.max(1), side="right")
    NQ = len(QG)
    cc, sc = corr_b1.astype(np.float64), corr_poe.astype(np.float64)
    vals = {}
    for wn, m in masks.items():
        wv, sel = m13.weights(D, m), m[D["e"]]
        nsel = sel.sum()
        diff = np.bincount(jr, weights=wv * (cc - sc), minlength=NQ + 2)
        accq = (wv * sc).sum() + (diff.sum() - np.cumsum(diff)[:NQ])
        srvq = np.cumsum(np.bincount(jr, weights=sel.astype(float), minlength=NQ + 2))[:NQ] / nsel
        acc, srv = np.append(accq, (wv * sc).sum())[::-1], np.append(srvq, 1.0)[::-1]
        b0s = float((ent > TAU0)[sel].mean())
        vals[wn] = float(m13.interp(srv, acc, [b0s])[0])
    srv_asc = np.append(np.cumsum(np.bincount(jr, minlength=NQ + 2))[:NQ] / N, 1.0)
    b0f = float((ent > TAU0).mean())
    jhi = min(max(int(np.searchsorted(srv_asc, b0f, side="left")), 1), len(srv_asc) - 1)
    jlo = jhi - 1
    alpha = 0.0 if srv_asc[jhi] == srv_asc[jlo] else (b0f - srv_asc[jlo]) / (srv_asc[jhi] - srv_asc[jlo])
    zc = lambda j: corr_poe if j >= NQ else np.where(jr > j, corr_b1, corr_poe)
    return vals, (1 - alpha) * zc(jlo) + alpha * zc(jhi)


def driftgate_weight(D):
    Hs = m13.server_entropy(D["ps"])
    Hc = D["ent"].astype(np.float64)
    Hc_bar, Hs_bar = m13.window_means(D, [Hc, Hs])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(Hc_bar) & ((Hc_bar + Hs_bar) > 0), Hs_bar / (Hc_bar + Hs_bar), 0.5), Hs


# ------------------------------------------------------------------------------------------------ phase A
def reference_answers(D, q, lr, lr_corr, dev):
    """the reference answers, repeating r15_analysis.analyse (lines 210-258) without the parts not needed here."""
    exact = "lpc" in q.files
    pc, ps, Mrow, a, y = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"]
    cp, sp, ent = D["cp"], D["sp"], D["ent"]
    N = len(y)
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
    ans["R-LR"] = m15.mix_answer(P, pc, ps)
    Dc = dict(D, ps=psr, sp=psr.argmax(1).astype(np.int64))
    fc = m13.features(Dc)
    fc[:, 1] = Hc
    Pc = lr_corr[1].predict_proba(lr_corr[0].transform(fc))[:, 1].astype(np.float32)
    ans["correction + learned weight"] = m15.mix_answer(Pc, pc, psr)
    del feats, fc, Dc, z
    if exact:
        ans["LEW"], _ = m15.lew_answers(q["lpc"], q["lps"], dev)
    else:
        ans["LEW"], _ = m15.lew_answers(lpc_, np.log(np.maximum(ps, P_FLOOR)), dev)
    del lpc_
    Hc_bar, Hs_bar = m13.window_means(D, [Hc, Hs])
    with np.errstate(invalid="ignore", divide="ignore"):
        w_auto = np.where(np.isfinite(Hc_bar) & ((Hc_bar + Hs_bar) > 0), Hs_bar / (Hc_bar + Hs_bar), 0.5)
    ans["DriftGate"] = m15.mix_answer(w_auto, pc, psr)
    ans["no correction + adaptive w"] = m15.mix_answer(w_auto, pc, ps)
    ans["correction + w 0.5"] = m15.mix_answer(np.full(N, 0.5), pc, psr)
    ans["correction + dev w 0.2"] = m15.mix_answer(np.full(N, m15.W_DEV), pc, psr)
    ans["corrected edge only"] = psr.argmax(1)
    return ans, w_auto


def refs_path(runs, name):
    return CACHE / f"{Path(runs).name}__{name}__{REFS_VERSION}.npz"


def phase_a(task):
    runs, name, lr, lr_corr = task
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    p, pm = refs_path(runs, name), refs_path(runs, name).with_suffix(".json")
    if p.exists() and pm.exists():
        meta = json.loads(pm.read_text())
        if meta["src_size"] == st.st_size and meta["src_mtime_ns"] == st.st_mtime_ns:
            return meta
    t0 = time.time()
    D = m13.load(Path(runs), name)
    q = np.load(src)
    ans, w_auto = reference_answers(D, q, lr, lr_corr, dev)
    y = D["y"]
    corr = {k: ans[k] == y for k in ANS_KEYS}
    full = {k: float(per_round(D, c).mean()) for k, c in corr.items()}
    vals, _ = ztw(D, corr["B1"], corr["R-PoE"], {"full": np.ones(D["E"], bool)})
    full["R-ZTW"] = vals["full"]
    assert max(int(ans[k].max()) for k in ANS_KEYS) < 256
    np.savez(p, w_auto=w_auto, **{k: ans[k].astype(np.uint8) for k in ANS_KEYS})
    meta = dict(runs=str(runs), name=name, full=full, src_size=st.st_size, src_mtime_ns=st.st_mtime_ns,
                seconds=time.time() - t0)
    pm.write_text(json.dumps(meta))
    print(f"  phase A {name} ({meta['seconds'] / 60:.1f} min)", flush=True)
    return meta


def r15_cache_value(runs, name):
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    key = hashlib.sha256(f"{m15.SCRIPT_SHA}|{m15.DEP_SHA}|{src}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    p = m15.CACHE / f"{Path(runs).name}__{name}__{key}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


# ------------------------------------------------------------------------------------------------ phase B helpers
def pick(J):
    """first candidate (priority order = column order) whose score is within TIE of the row maximum."""
    return np.argmax(J >= J.max(1, keepdims=True) - TIE, axis=1)


def r_index(shat):
    """index into R_NUM of the grid value nearest to 1 - shat (ties: closer to 0.5, then smaller)."""
    dp = np.abs((1.0 - shat)[:, None] - R_NUM[None, :])[:, R_PRI]
    return np.asarray(R_PRI)[np.argmax(dp <= dp.min(1, keepdims=True) + TIE, axis=1)]


GIDX = {(wi, R_ORDER.index(R_NUM[ri])): 1 + wi * len(R_ORDER) + R_ORDER.index(R_NUM[ri]) for wi in range(len(W_GRID)) for ri in range(9)}
COL_B0 = np.array([GIDX[(W_GRID.index(0.0), R_ORDER.index(R_NUM[ri]))] for ri in range(9)])
COL_B5 = np.array([GIDX[(W_GRID.index(0.5), R_ORDER.index(R_NUM[ri]))] for ri in range(9)])
COL_BG = np.array([[0] + [GIDX[(wi, R_ORDER.index(R_NUM[ri]))] for wi in range(len(W_GRID))] for ri in range(9)])


def intervals(pc, pe, y, chunk=1 << 19):
    """per request: the open interval (lo, hi) of w where argmax(w p_d + (1 - w) p_e) = y; empty -> (2, 2)."""
    N = len(y)
    lo, hi = np.empty(N), np.empty(N)
    for s in range(0, N, chunk):
        pd_, pe_, yy = pc[s:s + chunk].astype(np.float64), pe[s:s + chunk].astype(np.float64), y[s:s + chunk]
        r = np.arange(len(yy))
        A = pe_[r, yy][:, None] - pe_
        B = (pd_[r, yy][:, None] - pd_) - A
        A[r, yy], B[r, yy] = 1.0, 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = -A / B
        l_ = np.where(B > 0, ratio, -np.inf).max(1)
        h_ = np.where(B < 0, ratio, np.inf).min(1)
        bad = ((B == 0) & (A <= 0)).any(1) | (l_ >= h_)
        lo[s:s + chunk], hi[s:s + chunk] = np.where(bad, 2.0, l_), np.where(bad, 2.0, h_)
    return lo, hi


def count_in(lo_sorted, hi_sorted, w):
    """number of intervals (lo, hi) containing w (intervals non-empty or (2, 2))."""
    return np.searchsorted(lo_sorted, w, side="left") - np.searchsorted(hi_sorted, w, side="right")


def period_window(ix, v, prev_ok):
    """R15 period window restricted to the current model and cell set: earlier requests of the device's round when at
    least 8; else the requests of its previous round with requests when prev_ok[group]; else the earlier requests of the
    round (0-7). Returns per request (sum of v, count)."""
    order, starts, sizes, prev, gi, pos = (ix[k] for k in ("order", "starts", "sizes", "prev", "gi", "pos"))
    vs = np.asarray(v, np.float64)[order]
    tot = np.add.reduceat(vs, starts)
    cs = np.cumsum(vs) - vs
    pref = cs - cs[starts][gi]
    use = pos >= MIN_PREV
    pg = prev[gi]
    okp = (pg >= 0) & prev_ok[gi]
    s = np.where(use, pref, np.where(okp, tot[np.maximum(pg, 0)], pref))
    n = np.where(use, pos, np.where(okp, sizes[np.maximum(pg, 0)], pos))
    out_s, out_n = np.empty(len(vs)), np.empty(len(vs))
    out_s[order], out_n[order] = s, n
    return out_s, out_n


def controller(co, Hc):
    """r15_analysis.online_offload, also returning tau."""
    from numpy.lib.stride_tricks import sliding_window_view
    oc, first = co
    x_all = Hc[oc].astype(np.float64)
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x_all)]
    tau = np.full(len(x_all), m15.CTRL_INIT)
    for s0, e0 in zip(starts, ends):
        x = x_all[s0:e0]
        n = len(x)
        for i in range(m15.CTRL_MIN, min(m15.CTRL_W, n)):
            tau[s0 + i] = np.quantile(x[:i], 1 - m15.BETA)
        if n > m15.CTRL_W:
            tau[s0 + m15.CTRL_W:e0] = np.quantile(sliding_window_view(x[:-1], m15.CTRL_W), 1 - m15.BETA, axis=1)
    off, tau_o = np.empty(len(x_all), bool), np.empty(len(x_all))
    off[oc], tau_o[oc] = x_all > tau, tau
    return off, tau_o


def image_keys(D):
    """same image <=> same (label, rank of the request among the device-round's requests of that label in stored order);
    the requests of a device-round are stored contiguously in request-builder order (runner_r6)."""
    g, y = D["g"], D["y"]
    N = len(y)
    blocks = np.r_[True, g[1:] != g[:-1]]
    assert len(np.unique(g)) == blocks.sum(), "device-round requests not contiguous"
    o = np.lexsort((np.arange(N), y, g))
    st = np.r_[True, (g[o][1:] != g[o][:-1]) | (y[o][1:] != y[o][:-1])]
    first = np.maximum.accumulate(np.where(st, np.arange(N), 0))
    rank = np.empty(N, np.int64)
    rank[o] = np.arange(N) - first
    return y.astype(np.int64) * (1 << 24) + rank


# ------------------------------------------------------------------------------------------------ phase B
def analyse16(task):
    runs, name, period, online, use_refs = task
    t_start = time.time()
    D = m13.load(Path(runs), name)
    pc, ps, Mrow, a, y = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"]
    cp, sp, ent, kind = D["cp"], D["sp"], D["ent"], D["kind"]
    N, E, K = len(y), D["E"], D["K"]
    G = E * K
    er, g = D["er"], D["g"]
    e_i, k_i = D["e"], D["k"]
    main = kind == 0
    is_replay = D["h"]["config"].get("replay_from") is not None
    assert is_replay == (period == "replay"), name
    masks = window_masks(er, period)
    out = dict(name=name, period=period, er=er, n_req=N, K=K, C=D["C"], pr={}, win={}, checks=dict(D["checks"]))
    pr = out["pr"]
    # ---- reference answers (phase A cache) and DriftGate
    if use_refs:
        z = np.load(refs_path(runs, name))
        ans = {k: z[k].astype(np.int64) for k in ANS_KEYS}
        w_auto = z["w_auto"]
        Hs = m13.server_entropy(ps)
    else:
        w_auto, Hs = driftgate_weight(D)
        ans = {"DriftGate": m15.mix_answer(w_auto, pc, m13.corrected(ps, Mrow, a, 0.5))}
    corr_ref = {k: v == y for k, v in ans.items()}
    corr_dg = corr_ref["DriftGate"]
    for k, c in corr_ref.items():
        pr["ref:" + k] = per_round(D, c)
    if use_refs:
        vals, soft = ztw(D, corr_ref["B1"], corr_ref["R-PoE"], masks)
        pr["ref:R-ZTW"] = per_round(D, soft)
        out["win"]["ref:R-ZTW"] = vals
    tick(t_start, "references")
    # ---- grid
    CORR = np.empty((NG, N), bool)
    for r in R_ORDER:
        pe = ps if r is None else corrected_chunked(ps, Mrow, a, r)
        for wi, w in enumerate(W_GRID):
            CORR[wi * len(R_ORDER) + R_ORDER.index(r)] = mix_scalar(w, pc, pe) == y
        if r == 0.5:
            psr = pe
    out["checks"]["grid_vs_r15_rules"] = {   # fixed pairs that coincide with Round 15 rules
        k: float((CORR[GIDX_ALL[pair] - 1] == corr_ref[k]).mean()) for k, pair in
        (("B3", (0.5, None)), ("correction + w 0.5", (0.5, 0.5)), ("corrected edge only", (0.0, 0.5)),
         ("B1", (1.0, 0.5)), ("B2", (0.0, None)), ("correction + dev w 0.2", (0.2, 0.5))) if k in corr_ref}
    tick(t_start, "grid")
    lo, hi = intervals(pc, psr, y)
    out["checks"]["interval_agreement"] = {str(w): float((((lo < w) & (w < hi)) == CORR[GIDX_ALL[(w, 0.5)] - 1]).mean())
                                           for w in (0.2, 0.5, 0.8)}
    tick(t_start, "intervals")
    # ---- group statistics
    gm = g * 2 + main
    cnt = np.bincount(gm, minlength=2 * G).reshape(G, 2)
    nN, nM = cnt[:, 0].astype(float), cnt[:, 1].astype(float)
    cMN = np.empty((G, 2, NG))
    for ci in range(NG):
        cMN[:, :, ci] = np.bincount(gm, weights=CORR[ci], minlength=2 * G).reshape(G, 2)
    cN_, cM_ = cMN[:, 0], cMN[:, 1]
    T = Mrow[np.arange(N), sp]
    tt = np.bincount(gm, weights=T, minlength=2 * G).reshape(G, 2)
    tN, tM = tt[:, 0], tt[:, 1]
    for ci in range(NG):
        pr[f"grid:{ci}"] = per_round_groups(D, cN_[:, ci] + cM_[:, ci])
    with np.errstate(invalid="ignore", divide="ignore"):
        best = ((cN_ + cM_) / (nN + nM)[:, None]).max(1)
    pr["O2"] = per_round_groups(D, best * (nN + nM))
    s_true = np.where(nN + nM > 0, nM / np.maximum(nN + nM, 1), np.nan)
    r_true = np.clip(1.0 - s_true[g], 0.1, 0.9).astype(np.float32)
    pe = corrected_chunked(ps, Mrow, a, r_true)
    for w in W_GRID:
        pr[f"O3:{w}"] = per_round(D, mix_scalar(w, pc, pe) == y)
    del pe, r_true
    tick(t_start, "group statistics, O2, O3")
    # ---- calibration source per device-round
    ok_cal = ((nM >= MINC) & (nN >= MINC)).reshape(E, K)
    cal_src = np.full((E, K), -1)
    last = np.full(K, -1)
    for e in range(E):
        cal_src[e] = last
        last = np.where(ok_cal[e], e, last)
    cells_e = np.sort(m15.cells_matrix(runs, name, K, er), axis=2)
    ix = m14.window_index(D)
    order, starts, sizes, prev = ix["order"], ix["starts"], ix["sizes"], ix["prev"]
    gid = g[order[starts]]
    e_g, k_g = gid // K, gid % K
    gmap = np.full(G, -1)
    gmap[gid] = np.arange(len(gid))
    prev_ok = np.zeros(len(gid), bool)
    if is_replay:
        pv = np.maximum(prev, 0)
        prev_ok = (prev >= 0) & np.all(cells_e[e_g, k_g] == cells_e[e_g[pv], k_g], axis=1)
    q_s, q_n = period_window(ix, T, prev_ok)
    sw_s, _ = period_window(ix, main, prev_ok)
    src_req = cal_src[e_i, k_i]
    has_cal = src_req >= 0
    gs_req = np.where(has_cal, src_req * K + k_i, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c11 = np.where(has_cal, tM[gs_req] / nM[gs_req], np.nan)
        c01 = np.where(has_cal, tN[gs_req] / nN[gs_req], np.nan)
        q = q_s / q_n
        s_raw = (q - c01) / (c11 - c01)
        s_win = sw_s / q_n
    reason = np.full(N, "", dtype="<U10")          # priority: no_cal > no_obs > weak_c > nonfinite
    reason[~np.isfinite(s_raw)] = "nonfinite"
    reason[has_cal & (np.abs(c11 - c01) < MIN_CD)] = "weak_c"
    reason[q_n == 0] = "no_obs"
    reason[~has_cal] = "no_cal"
    valid = reason == ""
    shat = np.clip(np.where(valid, s_raw, 0.5), *CLIP)
    ri = r_index(shat)
    keys = image_keys(D)
    tick(t_start, "windows and share estimate")
    # ---- selectors (full offloading), per device-round
    choice = {s: np.full(N, -1, np.int16) for s in SELECTORS[1:] + [f"S-A {s_}" for s_ in SA]}
    rep = np.zeros(N, bool)
    for gi_ in range(len(gid)):
        rows = order[starts[gi_]:starts[gi_] + sizes[gi_]]
        e, k = int(e_g[gi_]), int(k_g[gi_])
        src = cal_src[e, k]
        if src < 0:
            continue
        gs = src * K + k
        cg = gmap[gs]
        crow = order[starts[cg]:starts[cg] + sizes[cg]]
        rep[rows] = np.isin(keys[rows], keys[crow])
        cmain = main[crow]
        loM, hiM = np.sort(lo[crow][cmain]), np.sort(hi[crow][cmain])
        loN, hiN = np.sort(lo[crow][~cmain]), np.sort(hi[crow][~cmain])
        wv = w_auto[rows]
        JM = np.empty((len(rows), NG + 1))
        JN = np.empty((len(rows), NG + 1))
        JM[:, 0] = count_in(loM, hiM, wv) / nM[gs]
        JN[:, 0] = count_in(loN, hiN, wv) / nN[gs]
        JM[:, 1:] = cM_[gs] / nM[gs]
        JN[:, 1:] = cN_[gs] / nN[gs]
        for s_ in SA:
            choice[f"S-A {s_}"][rows] = pick(s_ * JM + (1 - s_) * JN)
        v = valid[rows]
        if not v.any():
            continue
        rv = rows[v]
        sh = shat[rv][:, None]
        J = sh * JM[v] + (1 - sh) * JN[v]
        choice["S-C"][rv] = pick(J)
        rr = ri[rv]
        choice["S-B0"][rv] = COL_B0[rr]
        choice["S-B5"][rv] = COL_B5[rr]
        cols = COL_BG[rr]
        choice["S-Bg"][rv] = cols[np.arange(len(rv)), pick(np.take_along_axis(J, cols, 1))]
    ar = np.arange(N)
    for s_, ch in choice.items():
        pr["sel:" + s_] = per_round(D, np.where(ch <= 0, corr_dg, CORR[np.maximum(ch.astype(np.int64) - 1, 0), ar]))
    tick(t_start, "selectors")
    # ---- estimation statistics (full offloading)
    stale_e = np.where(has_cal, e_i - src_req, np.nan)
    stale_r = np.where(has_cal, er[e_i] - er[np.maximum(src_req, 0)], np.nan)
    cell_change = np.where(has_cal, np.any(cells_e[e_i, k_i] != cells_e[np.maximum(src_req, 0), k_i], axis=1), False)
    s_cur = s_true[g]
    chC = choice["S-C"].astype(np.int64)
    w_sel = np.where(chC == 0, w_auto, np.array([np.nan] + [w for w, _ in GRID])[np.maximum(chC, 0)])
    r_sel = np.array([np.nan] + [np.nan if r is None else r for _, r in GRID])[np.maximum(chC, 0)]
    r_none = np.array([False] + [r is None for _, r in GRID])[np.maximum(chC, 0)]
    stats = {}
    for wn, m in masks.items():
        sel = m[e_i]
        vs_ = sel & valid
        hc_ = sel & has_cal
        cv = vs_ & (chC > 0)
        st = dict(n=int(sel.sum()), fallback=float((~valid[sel]).mean()),
                  **{f"fallback_{r_}": float((reason[sel] == r_).mean()) for r_ in ("no_cal", "no_obs", "weak_c", "nonfinite")},
                  sA_fallback=float((~has_cal[sel]).mean()),
                  clip_low=float((s_raw[vs_] < CLIP[0]).mean()), clip_high=float((s_raw[vs_] > CLIP[1]).mean()),
                  err_window_mean=float((shat - s_win)[vs_].mean()), err_window_mae=float(np.abs(shat - s_win)[vs_].mean()),
                  err_current_mean=float((shat - s_cur)[vs_].mean()), err_current_mae=float(np.abs(shat - s_cur)[vs_].mean()),
                  s_true_current=float(s_cur[sel].mean()), c11=float(c11[hc_].mean()), c01=float(c01[hc_].mean()),
                  abs_c_diff=float(np.abs(c11 - c01)[hc_].mean()),
                  stale_eval_rounds=float(stale_e[hc_].mean()), stale_training_rounds=float(stale_r[hc_].mean()),
                  cell_change=float(cell_change[hc_].mean()), model_change=0.0 if is_replay else 1.0,
                  repeated_image=float(rep[hc_].mean()),
                  SC_share_driftgate=float((chC[vs_] == 0).mean()), SC_mean_w=float(w_sel[vs_].mean()),
                  SC_share_r_none=float(r_none[vs_].mean()),
                  SC_mean_numeric_r=float(r_sel[cv & ~r_none].mean()) if (cv & ~r_none).any() else float("nan"),
                  SBg_share_driftgate=float((choice["S-Bg"][vs_] == 0).mean()),
                  **{f"SA{s_}_share_driftgate": float((choice[f"S-A {s_}"][hc_] == 0).mean()) for s_ in SA})
        stats[wn] = st
    out["stats"] = stats
    # ---- online controller (beta 0.5)
    if online:
        out["online"] = analyse_online(D, ix, order, starts, sizes, prev, prev_ok, gid, e_g, k_g, gmap, cal_src, ans,
                                       corr_ref, Hs, psr, lo, hi, T, main, masks, CORR)
    del CORR
    tick(t_start, "statistics and online")
    out["seconds"] = time.time() - t_start
    return out


GIDX_ALL = {pair: 1 + i for i, pair in enumerate(GRID)}


def analyse_online(D, ix, order, starts, sizes, prev, prev_ok, gid, e_g, k_g, gmap, cal_src, ans, corr_ref, Hs,
                   psr, lo, hi, T, main, masks, CALL):
    """online controller (beta 0.5): offloaded requests answered by the reference operations and the selectors with the
    share among offloaded requests (decision_rule.md 4.5); requests kept on the device get the device-exit answer."""
    pc, ps, Mrow, a, y, cp, ent = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"], D["cp"], D["ent"]
    N, K = len(y), D["K"]
    e_i, k_i = D["e"], D["k"]
    Hc = ent.astype(np.float64)
    co = m14.client_order(D)
    off, tau = controller(co, Hc)
    assert np.array_equal(off, m15.online_offload(co, Hc))
    sd_s, sd_n = m15.window_sum_counted(ix, Hc, off)
    ss_s, _ = m15.window_sum_counted(ix, Hs, off)
    with np.errstate(invalid="ignore", divide="ignore"):
        wP = np.where(sd_n > 0, np.where(sd_s + ss_s > 0, ss_s / (sd_s + ss_s), 0.5), np.nan)
    wP = m14.ffill_client(co, wP)
    a_dgp = m15.mix_answer(wP, pc, psr)
    ops = dict(ans, **{"DG-P": a_dgp, "DG-P nocorr": m15.mix_answer(wP, pc, ps)})
    res = dict(pr={}, srv={wn: float(off[m[e_i]].mean()) for wn, m in masks.items()})
    corr_dev = cp == y
    for nm, key in ONLINE_OPS.items():
        if key is None:
            c = corr_dev
        elif key in ops:
            c = np.where(off, ops[key], cp) == y
        else:
            continue
        res["pr"]["ref:" + nm] = per_round(D, c)
    corr_dgp = np.where(off, a_dgp, cp) == y
    Z = off & T
    qz_s, qz_n = period_window(ix, Z, prev_ok)
    no_s, _ = period_window(ix, off, prev_ok)
    sw_s, _ = period_window(ix, main, prev_ok)
    pos = np.empty(N, np.int64)
    pos[order] = ix["pos"]
    gi_req = np.empty(N, np.int64)
    gi_req[order] = ix["gi"]
    reason = np.full(N, "", dtype="<U12")
    s_all = np.full(N, np.nan)
    s_off = np.full(N, np.nan)
    s_all_raw = np.full(N, np.nan)
    choice = {s: np.full(N, -1, np.int16) for s in SELECTORS[1:] + [f"S-A {s_}" for s_ in SA]}
    for gi_ in range(len(gid)):
        rows = order[starts[gi_]:starts[gi_] + sizes[gi_]]
        ro = rows[off[rows]]
        if len(ro) == 0:
            continue
        e, k = int(e_g[gi_]), int(k_g[gi_])
        src = cal_src[e, k]
        if src < 0:
            reason[ro] = "no_cal"
            continue
        cg = gmap[src * K + k]
        crow = order[starts[cg]:starts[cg] + sizes[cg]]
        cm = main[crow]
        HM, HN = Hc[crow][cm], Hc[crow][~cm]
        nMc, nNc = len(HM), len(HN)
        HMT, HNT = np.sort(HM[T[crow][cm]]), np.sort(HN[T[crow][~cm]])
        vM = lambda t: (len(HMT) - np.searchsorted(HMT, t, side="right")) / nMc
        vN = lambda t: (len(HNT) - np.searchsorted(HNT, t, side="right")) / nNc
        # vbar over the window of each offloaded request (same window rule as q_Z)
        tr = tau[rows]
        cvm = np.r_[0.0, np.cumsum(vM(tr))]
        cvn = np.r_[0.0, np.cumsum(vN(tr))]
        p_ = pos[ro]
        use = p_ >= MIN_PREV
        pg = prev[gi_]
        if pg >= 0 and prev_ok[gi_]:
            prow = order[starts[pg]:starts[pg] + sizes[pg]]
            pm_, pn_ = vM(tau[prow]).sum(), vN(tau[prow]).sum()
            sm = np.where(use, cvm[p_], pm_)
            sn = np.where(use, cvn[p_], pn_)
        else:
            sm, sn = cvm[p_], cvn[p_]
        nwin = qz_n[ro]
        with np.errstate(invalid="ignore", divide="ignore"):
            vbM, vbN = sm / nwin, sn / nwin
            sa_raw = (qz_s[ro] / nwin - vbN) / (vbM - vbN)
        sa = np.clip(sa_raw, *CLIP)
        to = tau[ro]
        HMs, HNs = np.sort(HM), np.sort(HN)
        iM, iN = np.searchsorted(HMs, to, side="right"), np.searchsorted(HNs, to, side="right")
        selM, selN = nMc - iM, nNc - iN
        uM, uN = selM / nMc, selN / nNc
        with np.errstate(invalid="ignore", divide="ignore"):
            den = sa * uM + (1 - sa) * uN
            so = np.clip(sa * uM / den, *CLIP)
        rs = np.full(len(ro), "", dtype="<U12")      # priority: few_selected > few_offloads > weak_v > nonfinite > denominator
        rs[~np.isfinite(so) | (den == 0)] = "denominator"
        rs[~np.isfinite(sa_raw)] = "nonfinite"
        rs[np.abs(vbM - vbN) < MIN_VD] = "weak_v"
        rs[no_s[ro] < MIN_PREV] = "few_offloads"
        rs[(selM < MINC) | (selN < MINC)] = "few_selected"
        reason[ro] = rs
        s_all_raw[ro], s_all[ro], s_off[ro] = sa_raw, sa, so
        # accuracies on the calibration samples with H_d > tau
        oM, oN = np.argsort(HM, kind="stable"), np.argsort(HN, kind="stable")
        CM = CALL[:, crow[cm][oM]].T.astype(np.float64)
        CN = CALL[:, crow[~cm][oN]].T.astype(np.float64)
        sufM = np.vstack([np.cumsum(CM[::-1], 0)[::-1], np.zeros((1, NG))])
        sufN = np.vstack([np.cumsum(CN[::-1], 0)[::-1], np.zeros((1, NG))])
        ok_sel = (selM >= MINC) & (selN >= MINC)
        if not ok_sel.any():
            continue
        r2 = ro[ok_sel]
        JM = np.empty((len(r2), NG + 1))
        JN = np.empty((len(r2), NG + 1))
        JM[:, 1:] = sufM[iM[ok_sel]] / selM[ok_sel][:, None]
        JN[:, 1:] = sufN[iN[ok_sel]] / selN[ok_sel][:, None]
        wv = wP[r2][:, None]
        t2 = to[ok_sel][:, None]
        loM, hiM = lo[crow][cm], hi[crow][cm]
        loN, hiN = lo[crow][~cm], hi[crow][~cm]
        JM[:, 0] = ((HM[None, :] > t2) & (loM[None, :] < wv) & (wv < hiM[None, :])).sum(1) / selM[ok_sel]
        JN[:, 0] = ((HN[None, :] > t2) & (loN[None, :] < wv) & (wv < hiN[None, :])).sum(1) / selN[ok_sel]
        for s_ in SA:
            choice[f"S-A {s_}"][r2] = pick(s_ * JM + (1 - s_) * JN)
        v = rs[ok_sel] == ""
        if not v.any():
            continue
        rv = r2[v]
        sh = so[ok_sel][v][:, None]
        J = sh * JM[v] + (1 - sh) * JN[v]
        choice["S-C"][rv] = pick(J)
        rr = r_index(so[ok_sel][v])
        choice["S-B0"][rv] = COL_B0[rr]
        choice["S-B5"][rv] = COL_B5[rr]
        cols = COL_BG[rr]
        choice["S-Bg"][rv] = cols[np.arange(len(rv)), pick(np.take_along_axis(J, cols, 1))]
    ar = np.arange(N)
    for s_, ch in choice.items():
        c = np.where(off, np.where(ch <= 0, corr_dgp, CALL[np.maximum(ch.astype(np.int64) - 1, 0), ar]), corr_dev)
        res["pr"]["sel:" + s_] = per_round(D, c)
    # estimation statistics over offloaded requests
    sel_off_true = np.full(N, np.nan)
    gm_ = D["g"]
    no_ = np.bincount(gm_, weights=off, minlength=D["E"] * K)
    nmo = np.bincount(gm_, weights=off & main, minlength=D["E"] * K)
    with np.errstate(invalid="ignore", divide="ignore"):
        sel_off_true = (nmo / no_)[gm_]
        s_win = sw_s / qz_n
    stats = {}
    for wn, m in masks.items():
        so_ = off & m[e_i]
        vs_ = so_ & (reason == "")
        stats[wn] = dict(n_offloaded=int(so_.sum()), fallback=float((reason[so_] != "").mean()),
                         **{f"fallback_{r_}": float((reason[so_] == r_).mean()) for r_ in
                            ("no_cal", "few_selected", "few_offloads", "weak_v", "nonfinite", "denominator")},
                         clip_all_low=float((s_all_raw[vs_] < CLIP[0]).mean()), clip_all_high=float((s_all_raw[vs_] > CLIP[1]).mean()),
                         err_all_window_mean=float((s_all - s_win)[vs_].mean()), err_all_window_mae=float(np.abs(s_all - s_win)[vs_].mean()),
                         err_off_current_mean=float((s_off - sel_off_true)[vs_].mean()),
                         err_off_current_mae=float(np.abs(s_off - sel_off_true)[vs_].mean()),
                         SC_share_driftgate=float((choice["S-C"][vs_] == 0).mean()))
    res["stats"] = stats
    return res


def analyse_cached(task):
    runs, name = task[0], task[1]
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    key = hashlib.sha256(f"{SCRIPT_SHA}|{src}|{st.st_size}|{st.st_mtime_ns}|{task[2:]}".encode()).hexdigest()[:16]
    p = CACHE / f"{Path(runs).name}__{name}__B__{key}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    out = analyse16(task)
    pickle.dump(out, open(p, "wb"))
    print(f"  phase B {name} ({out['seconds'] / 60:.1f} min)", flush=True)
    return out


def run_many(fn, tasks, workers=WORKERS):
    if workers <= 1:
        return [fn(t) for t in tasks]
    big = {i for i, t in enumerate(tasks) if (Path(t[0]) / f"{t[1]}_evalprobs.npz").stat().st_size > 1e9}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as exb, ProcessPoolExecutor(workers, mp_context=ctx) as exs:
        fut = {i: (exb if i in big else exs).submit(fn, t) for i, t in enumerate(tasks)}
        return [fut[i].result() for i in range(len(tasks))]


# ------------------------------------------------------------------------------------------------ subcommands
def all_runs(roles=("decision", "supplementary")):
    out = []
    for st, c in CFG["settings"].items():
        if c["role"] in roles:
            for s in c["seeds"]:
                out.append((st, s, runs_dir(st), run_name(st, s)))
    return out


def cmd_manifest():
    rows = []
    for st, c in CFG["settings"].items():
        ers = []
        for s in c["seeds"]:
            src = runs_dir(st) / f"{run_name(st, s)}_evalprobs.npz"
            ers.append(tuple(np.load(src)["eval_rounds"].tolist()) if src.exists() else None)
        ok = all(e is not None for e in ers) and len(set(ers)) == 1
        er = np.array(ers[0]) if ok else None
        wins = window_masks(er, c["period"]) if ok else {}
        for wn, m in wins.items():
            rows.append(["basic" if c["role"] == "decision" else c["role"], f"{st}|{wn}", st, c["period"], wn,
                         c["runs"], c["pattern"], " ".join(map(str, c["seeds"])), " ".join(map(str, er[m])), int(m.sum()),
                         "yes" if c["role"] == "decision" else "no"])
        if st in ("S1", "S2", "S1 replay", "S2 replay") and ok:
            for b, m in bin_masks(er).items():
                rows.append(["time", f"{st}|{b}", st, c["period"], b, c["runs"], c["pattern"], " ".join(map(str, c["seeds"])),
                             " ".join(map(str, er[m])), int(m.sum()), "yes"])
        if [st, "day1_full" if c["period"] == "day1" else "replay_full"] in CFG["online_table_rows"] and ok:
            wn = "day1_full" if c["period"] == "day1" else "replay_full"
            rows.append(["online", f"{st}|{wn}|online beta 0.5", st, c["period"], wn, c["runs"], c["pattern"],
                         " ".join(map(str, c["seeds"])), " ".join(map(str, er)), len(er), "yes"])
        if not ok:
            rows.append(["missing", st, st, c["period"], "", c["runs"], c["pattern"], " ".join(map(str, c["seeds"])), "", 0, "INCOMPLETE"])
    with open(HERE / "evaluation_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "row_id", "setting", "period", "window_or_bin", "runs_dir", "run_pattern", "seeds",
                    "evaluation_rounds", "n_evaluation_rounds", "judged"])
        w.writerows(rows)
    print(f"evaluation_manifest.csv: {len(rows)} rows; basic {sum(r[0] == 'basic' for r in rows)}, "
          f"time {sum(r[0] == 'time' for r in rows)}, online {sum(r[0] == 'online' for r in rows)}")


def load_lr():
    lr = m14.read_lr()
    p = m15.CACHE / f"lr_corr__{m15.DEP_SHA[:16]}.pkl"
    assert p.exists(), "Round 15 corrected-feature regression cache missing (refitting would rewrite a Round 15 table)"
    return lr, pickle.load(open(p, "rb"))


def cmd_check():
    lr, lr_corr = load_lr()
    runs = all_runs()
    metas = run_many(phase_a, [(str(r), n, lr, lr_corr) for _, _, r, n in runs])
    rows, worst, missing = [], 0.0, []
    for (st, s, r, n), meta in zip(runs, metas):
        c15 = r15_cache_value(str(r), n)
        for rl in REF:
            v16 = meta["full"][rl]
            if c15 is not None:
                v15, src = c15["res"][rl]["full"], "R15 cache"
                d = abs(v16 - v15)
            else:
                v15, src, d = float("nan"), "missing", float("nan")
                missing.append(n)
            worst = max(worst, d) if np.isfinite(d) else worst
            rows.append([st, s, n, rl, SHOW[rl], f"{v16:.17g}", f"{v15:.17g}", f"{d:.3g}", src])
    a1 = [st for st, c in CFG["settings"].items() if c["role"] in ("decision", "supplementary") and c["period"] == "day1"]
    ok = worst <= CFG["start_check"]["tolerance_fraction"] and not missing
    wcsv("R16_T0_start_check.csv", ["setting", "seed", "run", "rule", "display", "r16_value", "r15_cache_value", "abs_difference",
                                    "reference"], rows)
    res = dict(passed=bool(ok), max_abs_difference=worst, n_values=len(rows), missing_r15_cache=sorted(set(missing)),
               settings=sorted(set(st for st, *_ in runs)), first_day_settings=a1, script_sha=SCRIPT_SHA,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (CACHE / "START_CHECK.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_smoke(name):
    runs = runs_dir("S1 development")
    o = analyse16((str(runs), name, "day1", True, False))
    win = window_masks(o["er"], "day1")
    for k in ["ref:DriftGate"] + [f"sel:{s}" for s in SELECTORS[1:]] + [f"sel:S-A {s_}" for s_ in SA] + ["O2"]:
        print(k, {wn: round(float(o["pr"][k][m].mean()) * 100, 3) for wn, m in win.items()})
    print("online", {k: round(float(v.mean()) * 100, 3) for k, v in o["online"]["pr"].items()})
    print("checks", o["checks"])
    print("stats full", o["stats"]["day1_full"])
    print("online stats full", o["online"]["stats"]["day1_full"])
    print("minutes", round(o["seconds"] / 60, 1))


def cmd_run():
    chk = json.loads((CACHE / "START_CHECK.json").read_text())
    assert chk["passed"], "start check not passed"
    online_set = {st for st, _ in CFG["online_table_rows"]}
    tasks, keys = [], []
    for st, c in CFG["settings"].items():
        for s in c["seeds"]:
            tasks.append((str(runs_dir(st)), run_name(st, s), c["period"], st in online_set, c["role"] != "development"))
            keys.append((st, s))
    A = dict(zip(keys, run_many(analyse_cached, tasks)))
    R = {st: [A[(st, s)] for s in c["seeds"]] for st, c in CFG["settings"].items()}
    write_tables(R)


# ------------------------------------------------------------------------------------------------ tables
def wval(a, key, wn, m):
    if key in a["win"] and wn in a["win"][key]:
        return a["win"][key][wn]
    return float(a["pr"][key][m].mean())


def status(d):
    if d >= -PASS_TOL:
        return ">=0"
    return "neg >= -0.2" if d >= HOLD_FLOOR - PASS_TOL else "neg < -0.2"


def judge(diffs):
    if not diffs or any(not np.isfinite(d) for d in diffs):
        return "INCOMPLETE"
    if all(d >= -PASS_TOL for d in diffs):
        return "PASS"
    return "HOLD" if all(d >= HOLD_FLOOR - PASS_TOL for d in diffs) else "NO_GO"


def row_block(runs_, key_fn, ref_keys, methods, label):
    """runs_: list of run summaries (seeds); key_fn(a, key) -> value (fraction). Returns dict with references, strongest,
    per-method values and paired differences (pp)."""
    refv = {rk: np.array([key_fn(a, rk) for a in runs_]) * 100 for rk in ref_keys}
    means = {rk: v.mean() for rk, v in refv.items()}
    best = max(means.values())
    strongest = next(rk for rk in ref_keys if means[rk] >= best - TIE * 100)
    out = dict(refs=refv, strongest=strongest, sv=refv[strongest], methods={})
    for mk in methods:
        v = np.array([key_fn(a, mk) for a in runs_]) * 100
        out["methods"][mk] = dict(v=v, d=v - refv[strongest])
    return out


def write_tables(R):
    # ---- S-A: s from the development records
    dev = R["S1 development"]
    dv = {s_: np.array([a["pr"][f"sel:S-A {s_}"].mean() for a in dev]) * 100 for s_ in SA}
    sA = SA[0] if abs(dv[SA[1]].mean() - dv[SA[0]].mean()) <= TIE * 100 else max(SA, key=lambda s_: dv[s_].mean())
    wcsv("R16_S_dev_sA.csv", ["run", "s", "day1_full_accuracy_pct"],
         [[a["name"], s_, f"{v_:.4f}"] for s_ in SA for a, v_ in zip(dev, dv[s_])]
         + [["mean", s_, f"{dv[s_].mean():.4f}"] for s_ in SA] + [["chosen", sA, ""]])
    SEL = {"S-A": f"sel:S-A {sA}", "S-B0": "sel:S-B0", "S-B5": "sel:S-B5", "S-Bg": "sel:S-Bg", "S-C": "sel:S-C"}
    METHODS = ["ref:DriftGate"] + list(SEL.values())
    MNAME = {"ref:DriftGate": "DriftGate", **{v: k for k, v in SEL.items()}}
    REFK = ["ref:" + k for k in REF]
    judged = {}

    def emit(table, rows_def, ref_keys, ref_show, fname, METHODS=METHODS, MNAME=MNAME):
        summ, seed_rows, ref_rows = [], [], []
        diffs = {m_: [] for m_ in METHODS}
        for row_id, st, period, wnb, runs_, key_fn in rows_def:
            blk = row_block(runs_, key_fn, ref_keys, METHODS, row_id)
            ranks = sorted(ref_keys, key=lambda rk: -blk["refs"][rk].mean())
            for rk in ref_keys:
                ref_rows.append([table, row_id, st, period, wnb, ref_show(rk), fmt(blk["refs"][rk]), f"{blk['refs'][rk].mean():.6f}",
                                 ranks.index(rk) + 1, "yes" if rk == blk["strongest"] else ""])
            dg = blk["methods"][METHODS[0]]["v"]
            for mk in METHODS:
                mv = blk["methods"][mk]
                d_mean = float(mv["v"].mean() - blk["sv"].mean())
                diffs[mk].append(d_mean)
                summ.append([table, row_id, st, period, wnb, MNAME[mk], len(mv["v"]), fmt(mv["v"]), f"{mv['v'].mean():.6f}",
                             ref_show(blk["strongest"]), f"{blk['sv'].mean():.6f}", f"{d_mean:.6f}", fmt(mv["d"], 3),
                             fmt(mv["v"] - dg, 3), status(d_mean)])
                for a, x, xs, xd in zip(runs_, mv["v"], blk["sv"], dg):
                    seed_rows.append([table, row_id, st, period, wnb, MNAME[mk], a["name"], f"{x:.6f}", ref_show(blk["strongest"]),
                                      f"{xs:.6f}", f"{x - xs:.6f}", f"{xd:.6f}"])
        hdr = ["table", "row_id", "setting", "period", "window", "method", "n_seeds", "accuracy_pct mean (SD)", "accuracy_pct mean",
               "strongest reference", "strongest mean", "difference vs strongest pp (means)", "difference vs strongest pp mean (SD) per seed",
               "difference vs DriftGate pp mean (SD) per seed", "row result"]
        wcsv(fname + ".csv", hdr, summ)
        wcsv(fname + "_per_seed.csv", ["table", "row_id", "setting", "period", "window", "method", "run", "accuracy_pct",
                                       "strongest reference", "strongest_accuracy_pct", "difference_pp", "driftgate_accuracy_pct"], seed_rows)
        for mk in METHODS:
            judged[(table, MNAME[mk])] = (judge(diffs[mk]), diffs[mk], [r[0] for r in rows_def])
        return ref_rows


    # ---- basic table (42 rows) and supplementary rows
    def full_rows(settings):
        rd = []
        for st in settings:
            c = CFG["settings"][st]
            wins = WIN_DAY1 if c["period"] == "day1" else ["replay_full"]
            for wn in wins:
                m_ = window_masks(R[st][0]["er"], c["period"])[wn]
                rd.append((f"{st}|{wn}", st, c["period"], wn, R[st], lambda a, k, wn=wn, m_=m_: wval(a, k, wn, m_)))
        return rd
    dec = [st for st, c in CFG["settings"].items() if c["role"] == "decision"]
    basic_rows = full_rows(dec)
    assert len(basic_rows) == 42
    ref_rows = emit("basic", basic_rows, REFK, lambda rk: SHOW[rk[4:]], "R16_S_selectors")
    sup = [st for st, c in CFG["settings"].items() if c["role"] == "supplementary"]
    ref_rows += emit("supplementary", full_rows(sup), REFK, lambda rk: SHOW[rk[4:]], "R16_S_supplementary_R12")
    # ---- time table (20 rows)
    trows = []
    for base, (d1, rp) in CFG["time_table_settings"].items():
        for st in (d1, rp):
            for b, m_ in bin_masks(R[st][0]["er"]).items():
                trows.append((f"{st}|{b}", st, CFG["settings"][st]["period"], b, R[st], lambda a, k, m_=m_: float(a["pr"][k][m_].mean())))
    assert len(trows) == 20
    ref_rows += emit("time", trows, REFK, lambda rk: SHOW[rk[4:]], "R16_S_time")
    # ---- online table (4 rows)
    orows = []
    for st, wn in CFG["online_table_rows"]:
        orows.append((f"{st}|{wn}|online beta 0.5", st, CFG["settings"][st]["period"], wn, R[st],
                      lambda a, k: float(a["online"]["pr"][k].mean())))
    OREF = ["ref:" + k for k in ONLINE_OPS]
    ref_rows += emit("online", orows, OREF, lambda rk: rk[4:], "R16_S_online",
                     METHODS=["ref:DriftGate (DriftGate-P)"] + METHODS[1:],
                     MNAME=dict(MNAME, **{"ref:DriftGate (DriftGate-P)": "DriftGate (DriftGate-P)"}))
    wcsv("R16_R_references.csv", ["table", "row_id", "setting", "period", "window", "rule", "accuracy_pct mean (SD)", "mean",
                                  "rank", "strongest"], ref_rows)
    # ---- realized offloading of the online rows
    wcsv("R16_S_online_offloading.csv", ["setting", "window", "realized offloading mean (SD)"],
         [[st, wn, fmt([a["online"]["srv"][wn] for a in R[st]], 3)] for st, wn in CFG["online_table_rows"]])
    # ---- oracles (42 rows)
    orc, orc_seed = [], []
    for row_id, st, period, wn, runs_, key_fn in basic_rows:
        refm = {rk: np.mean([key_fn(a, rk) for a in runs_]) for rk in REFK}
        bestv = max(refm.values())
        strongest = next(rk for rk in REFK if refm[rk] >= bestv - TIE)
        gm_ = np.array([np.mean([key_fn(a, f"grid:{ci}") for a in runs_]) for ci in range(NG)])
        o1 = int(np.argmax(gm_ >= gm_.max() - TIE))
        o3m = np.array([np.mean([key_fn(a, f"O3:{w}") for a in runs_]) for w in W_GRID])
        o3b = W_GRID[int(np.argmax(o3m >= o3m.max() - TIE))]
        variants = [("O1 fixed grid pair", f"grid:{o1}", f"w={GRID[o1][0]}, r={GRID[o1][1] if GRID[o1][1] is not None else 'none'}"),
                    ("O2 device-round grid pair", "O2", ""), ("O3 true share, w=0", "O3:0.0", ""),
                    ("O3 true share, w=0.5", "O3:0.5", ""), ("O3 true share, best fixed w", f"O3:{o3b}", f"w={o3b}")]
        comp = [("DriftGate", "ref:DriftGate"), ("Probability average", "ref:B3"), ("corrected edge only", "ref:corrected edge only"),
                ("strongest reference", strongest)]
        for on, ok_, choice in variants:
            v = np.array([key_fn(a, ok_) for a in runs_]) * 100
            row = [row_id, st, period, wn, on, choice, fmt(v)]
            for cn, ck in comp:
                cv = np.array([key_fn(a, ck) for a in runs_]) * 100
                row.append(fmt(v - cv, 3))
            row.append(SHOW[strongest[4:]])
            orc.append(row)
            for a, x in zip(runs_, v):
                orc_seed.append([row_id, on, choice, a["name"], f"{x:.6f}"] +
                                [f"{x - key_fn(a, ck) * 100:.6f}" for _, ck in comp])
    wcsv("R16_O_oracles.csv", ["row_id", "setting", "period", "window", "oracle", "choice", "accuracy_pct mean (SD)",
                               "minus DriftGate pp", "minus Probability average pp", "minus corrected edge only pp",
                               "minus strongest reference pp", "strongest reference"], orc)
    wcsv("R16_O_oracles_per_seed.csv", ["row_id", "oracle", "choice", "run", "accuracy_pct", "minus DriftGate pp",
                                        "minus Probability average pp", "minus corrected edge only pp", "minus strongest reference pp"], orc_seed)
    # ---- estimation
    est = []
    for st, c in CFG["settings"].items():
        if c["role"] == "development":
            continue
        for wn in (WIN_DAY1 if c["period"] == "day1" else ["replay_full"]):
            keys_ = list(R[st][0]["stats"][wn])
            for kk in keys_:
                est.append([st, c["period"], "full offloading", wn, kk, fmt([a["stats"][wn][kk] for a in R[st]], 4)])
        if "online" in R[st][0]:
            wn = "day1_full" if c["period"] == "day1" else "replay_full"
            for kk in R[st][0]["online"]["stats"][wn]:
                est.append([st, c["period"], "online beta 0.5 (offloaded requests)", wn, kk,
                            fmt([a["online"]["stats"][wn][kk] for a in R[st]], 4)])
    wcsv("R16_S_estimation.csv", ["setting", "period", "mode", "window", "statistic", "mean (SD) over seeds"], est)
    chk = []
    for st, runs_ in R.items():
        for a in runs_:
            chk.append([st, a["name"], json.dumps(a["checks"].get("interval_agreement")), json.dumps(a["checks"].get("grid_vs_r15_rules")),
                        f"{a['seconds'] / 60:.1f}"])
    wcsv("R16_T1_run_checks.csv", ["setting", "run", "DriftGate-interval vs grid answer agreement", "grid pair vs Round 15 rule agreement",
                                   "phase B minutes"], chk)
    # ---- judgement
    jrows = []
    for (table, mn), (j, ds, ids) in judged.items():
        i_min = int(np.argmin(ds)) if ds else -1
        jrows.append([table, mn, len(ds), sum(d >= -PASS_TOL for d in ds), f"{min(ds):.6f}" if ds else "", ids[i_min] if ds else "",
                      "yes" if ds and all(abs(d) <= PASS_TOL for d in ds) else "", j])
    wcsv("R16_S_judgement.csv", ["table", "method", "rows", "rows with difference >= 0", "minimum difference pp", "row of the minimum",
                                 "all rows tied", "judgement"], jrows)
    json.dump(dict(sA=sA, sA_dev_means={str(s_): float(dv[s_].mean()) for s_ in SA},
                   judgement={f"{t}|{m_}": dict(judgement=j, min_difference=float(min(ds)) if ds else None,
                                               n_negative=int(sum(d < -PASS_TOL for d in ds)), n_rows=len(ds))
                              for (t, m_), (j, ds, ids) in judged.items()}, script_sha=SCRIPT_SHA),
              open(TAB / "R16_stage1_summary.json", "w"), indent=1)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "manifest":
        cmd_manifest()
    elif cmd == "check":
        cmd_check()
    elif cmd == "smoke":
        cmd_smoke(sys.argv[2])
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")


if __name__ == "__main__":
    main()
