"""Round 18 (own-class detector): per-run, per-condition analysis (decision_rule.md sections 1-3).

Conditions: first-day (training run, evaluation rounds <= 150), second-day (training run, rounds > 150) and frozen-replay
(the replay run). Each condition is analysed on its own requests; request histories (DriftGate windows, the detector rule's
previous 128 requests, the budget controller) start empty at the condition's first evaluation round, for every rule alike.

Per condition and seed the script stores (cache/<run>__<condition>.pkl):
  curves      macro accuracy on the ratio grid 0.05..1.00 for every (score, answer) pair:
              comparators: 5 scores x 5 answer rules; detectors: every stored detector column with the detector rule
              (edge-corrected-by-detector), edge-raw and edge-corrected-even; the main variant also with the detector rule
              averaged with the device output (weight 0.2); request-kind oracle with edge-raw and edge-corrected-even
  counts      correct counts per (evaluation round, device) at every grid ratio, and per request kind at 0.25/0.5/0.75
  auroc       non-Main detection, all / home / away, for every score
  extra (main variant and its schedules): fixed threshold 0.5, budget controller beta 0.25 / 0.5 (accuracy, ratio),
              estimation error of s-hat (fixed threshold, curve ratio 0.5, controller beta 0.5; home / away)
  reference   device only, edge only, confidence-based offloading (0.8 nats), entropy controller beta 0.25 / 0.5 with the
              five answer rules
The Round 13b-18 helper functions (record loading, correction, DriftGate and DriftGate-P weights, windows, entropy
controller) are imported unchanged, so the v32 rule is computed exactly as in Rounds 15-20.

  python .../r18oc_analysis.py run <run name> [<condition> ...]
  python .../r18oc_analysis.py check     # stage-1 checks (oracle substitution, agreement with the Round 20 code)
"""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ART = HERE.parent
JR = ART.parent
RUNS = JR / "runs" / "phaseT18oc"
CACHE = HERE / "cache"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


r19 = _load("r19", ART / "driftgate_tmc_r19_weights" / "scripts" / "r19_analysis.py")
r18, r17, r16, m15, m14, m13 = r19.r18, r19.r17, r19.r16, r19.m15, r19.m14, r19.m13

GRID = np.round(np.arange(1, 21) * 0.05, 2)
KEEP_KIND = (0.25, 0.5, 0.75)
SCORES = ["device-entropy", "device-max-prob", "device-energy", "own-mahalanobis", "own-knn"]
ANSWERS = ["edge-raw", "edge-corrected-even", "average-corrected-even", "probability-average", "logit-sum"]
NEW = "edge-corrected-by-detector"
NEW_AVG = "edge-corrected-by-detector + device 0.2"
ORACLE = "request-kind-oracle"
CONDITIONS = ("first-day", "second-day", "frozen-replay")
DAY = 150
W_PREV, MIN_PREV = 128, 16
S_CLIP, R_CLIP, TU_MIN = (0.02, 0.98), (0.05, 0.95), 0.1
BETAS = (0.25, 0.5)
P_FLOOR = 1e-8


# ------------------------------------------------------------------------------------------------ loading
def run_names(name):
    """(training run, replay run) of a run name of either kind."""
    base = name.replace("_replay", "")
    parts = base.rsplit("_s", 1)
    return base, f"{parts[0]}_replay_s{parts[1]}"


def load(name):
    D = m13.load(RUNS, name)
    Z = np.load(RUNS / f"{name}_r18oc.npz")
    X = {k: Z[k] for k in ("g", "energy", "maha", "knn", "hist_own", "hist_other", "det_round")}
    X["variants"] = [str(v) for v in Z["variants"]]
    assert len(X["g"]) == len(D["y"]), (len(X["g"]), len(D["y"]))
    assert np.array_equal(Z["eval_rounds"], D["er"])
    h = D["h"]
    X["seed"] = int(h["config"]["partition_seed"])
    X["meta"] = dict(name=name, model_config=h["config"].get("model_config"), seed=X["seed"])
    return D, X


def subset(D, X, emask):
    """the requests of the evaluation rounds in emask, re-indexed (D as returned by m13.load)."""
    sel = emask[D["e"]]
    remap = np.cumsum(emask) - 1
    S = dict(D)
    for k in ("e", "k", "y", "kind", "pc", "ps", "cp", "sp", "ent", "arrival", "a_req", "Mrow"):
        S[k] = D[k][sel]
    S["e"] = remap[S["e"]].astype(np.int64)
    S["er"], S["E"] = D["er"][emask], int(emask.sum())
    S["g"] = S["e"] * D["K"] + S["k"]
    for k in ("a_ek", "n_ek", "present", "home"):
        S[k] = D[k][emask]
    Y = dict(X)
    for k in ("g", "energy", "maha", "knn"):
        Y[k] = X[k][sel]
    for k in ("hist_own", "hist_other", "det_round"):
        Y[k] = X[k][emask]
    return S, Y


# ------------------------------------------------------------------------------------------------ helpers
def macro_rounds(cnt, n_ek, present):
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(present, cnt / n_ek, np.nan)
    return np.nanmean(A, axis=1)


def counts(D, c):
    return np.bincount(D["g"], weights=c, minlength=D["E"] * D["K"]).reshape(D["E"], D["K"]).astype(np.int32)


def macro(D, c):
    return float(np.mean(macro_rounds(counts(D, c), D["n_ek"], D["present"])))


def own_other_max(ps, Mrow):
    """edge probability and class of the most likely own class and of the most likely other class (float64)."""
    p = ps.astype(np.float64)
    po = np.where(Mrow, p, -1.0)
    pn = np.where(Mrow, -1.0, p)
    ao, an = po.argmax(1), pn.argmax(1)
    i = np.arange(len(p))
    return po[i, ao], ao, pn[i, an], an


def corr_argmax(oo, a, r):
    """argmax of the corrected edge exit (own classes x (1 - r) / a, others x r / (1 - a)); ties -> smaller class."""
    mo, ao, mn, an = oo
    so = mo * (1 - r) / a
    sn = mn * r / (1 - a)
    return np.where(so > sn, ao, np.where(sn > so, an, np.minimum(ao, an)))


def rank_offload(s, tb):
    """position of every request when the score s is sorted from the largest (ties by the fixed random key tb)."""
    order = np.lexsort((tb, -np.asarray(s, np.float64)))
    pos = np.empty(len(s), np.int64)
    pos[order] = np.arange(len(s))
    return pos, order


def theta_of(g, order, n_off):
    """detector threshold between the offloaded requests (smallest g first) and the rest."""
    if n_off >= len(g):
        return 1.0
    gs = g[order].astype(np.float64)
    lo = gs[n_off - 1] if n_off > 0 else -1.0
    return float(0.5 * (lo + gs[n_off]))


def cdf_hist(H, theta, e=None, k=None):
    """fraction of a 64-bin histogram on [0, 1] below theta (linear inside a bin).
    H [E, K, 64]; theta scalar -> [E, K]; theta per request (with the request's e, k) -> per request."""
    nb = H.shape[-1]
    H = H.astype(np.float64)
    tot = H.sum(-1)
    cum = np.concatenate([np.zeros(H.shape[:-1] + (1,)), np.cumsum(H, -1)], -1)
    th = np.clip(np.asarray(theta, np.float64), 0.0, 1.0)
    b = np.minimum((th * nb).astype(np.int64), nb - 1)
    frac = th * nb - b
    if e is None:
        below, inbin, T = cum[..., b], H[..., b], tot
    else:
        below, inbin, T = cum[e, k, b], H[e, k, b], tot[e, k]
    F = np.where(th >= 1.0, 1.0, (below + frac * inbin) / np.maximum(T, 1))
    return np.where(T > 0, F, np.nan)


def prev_share(co, v, W=W_PREV):
    """per request: mean of v over the device's previous W requests (current excluded) and the number of them."""
    oc, first = co
    x = np.asarray(v, np.float64)[oc]
    starts = np.flatnonzero(first)
    seg = np.repeat(starts, np.diff(np.r_[starts, len(x)]))
    i = np.arange(len(x))
    n_prev = np.minimum(i - seg, W)
    cs = np.r_[0.0, np.cumsum(x)]
    s = cs[i] - cs[i - n_prev]
    m = np.where(n_prev > 0, s / np.maximum(n_prev, 1), np.nan)
    out_m, out_n = np.empty(len(x)), np.empty(len(x), np.int64)
    out_m[oc], out_n[oc] = m, n_prev
    return out_m, out_n


def detector_r(D, co, off, theta, Hown, Hoth, has_det, nonmain=None):
    """r of edge-corrected-by-detector per request (decision_rule.md 1.3) and s-hat (nan where s-hat = 0.5 by rule);
    theta: scalar or per request; Hown/Hoth: [E, K, 64] of the variant. nonmain (optional) -> s_true per request."""
    q, n_prev = prev_share(co, off)
    e, k = D["e"], D["k"]
    if np.ndim(theta) == 0:
        t_ek, u_ek = cdf_hist(Hoth, theta), cdf_hist(Hown, theta)
        t, u = t_ek[e, k], u_ek[e, k]
    else:
        t, u = cdf_hist(Hoth, theta, e, k), cdf_hist(Hown, theta, e, k)
    ok = (n_prev >= MIN_PREV) & has_det & np.isfinite(t) & np.isfinite(u) & ((t - u) >= TU_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(ok, np.clip((q - u) / (t - u), *S_CLIP), 0.5)
        num, den = sh * t, sh * t + (1 - sh) * u
        r = np.where(has_det & np.isfinite(t) & np.isfinite(u) & (den > 0), np.clip(num / den, *R_CLIP), 0.5)
    out = dict(r=r, s_hat=np.where(ok, sh, np.nan))
    if nonmain is not None:
        out["s_true"], _ = prev_share(co, nonmain)
    return out


def g_controller(co, g, beta):
    """budget controller on the detector: theta = beta quantile of the previous 128 g (current excluded); 0.5 with fewer
    than 16 previous requests. Returns (offload mask, theta per request)."""
    from numpy.lib.stride_tricks import sliding_window_view
    oc, first = co
    x_all = np.asarray(g, np.float64)[oc]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x_all)]
    th = np.full(len(x_all), 0.5)
    for s0, e0 in zip(starts, ends):
        x = x_all[s0:e0]
        n = len(x)
        for i in range(MIN_PREV, min(W_PREV, n)):
            th[s0 + i] = np.quantile(x[:i], beta)
        if n > W_PREV:
            th[s0 + W_PREV:e0] = np.quantile(sliding_window_view(x[:-1], W_PREV), beta, axis=1)
    off, theta = np.empty(len(x_all), bool), np.empty(len(x_all))
    off[oc], theta[oc] = x_all < th, th
    return off, theta


def auroc(score, pos):
    from scipy.stats import rankdata
    if pos.all() or not pos.any():
        return float("nan")
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ one condition
def analyse(D, X, cond, main=None, detectors=None):
    t0 = time.time()
    N = len(D["y"])
    y, pc, ps, cp, sp, Mrow, a = D["y"], D["pc"], D["ps"], D["cp"], D["sp"], D["Mrow"], D["a_req"].astype(np.float64)
    nonmain = D["kind"] != 0
    tb = np.random.default_rng([X["seed"], 18, 7]).random(N)
    co, ix = m14.client_order(D), m14.window_index(D)
    Hc, Hs = D["ent"].astype(np.float64), m13.server_entropy(ps)
    w_auto, _ = r16.driftgate_weight(D)
    psr = m13.corrected(ps, Mrow, D["a_req"], 0.5)
    oo = own_other_max(ps, Mrow)
    lpc = np.log(np.maximum(pc, P_FLOOR))
    c_dev = cp == y
    fixed = {"edge-raw": sp == y, "edge-corrected-even": corr_argmax(oo, a, 0.5) == y,
             "probability-average": (pc + ps).argmax(1) == y,
             "logit-sum": (lpc + np.log(np.maximum(ps, P_FLOOR))).argmax(1) == y}
    del lpc
    out = dict(cond=cond, N=N, E=D["E"], K=D["K"], er=D["er"], n_ek=D["n_ek"], present=D["present"], home=D["home"],
               nonmain_share=float(nonmain.mean()), curves={}, cnt={}, cnt_kind={}, auroc={}, extra={}, reference={},
               checks={}, variants=X["variants"], meta=X["meta"])
    out["checks"]["corr_argmax_agree_m13"] = float((corr_argmax(oo, a, 0.5)[:200_000] == psr[:200_000].argmax(1)).mean())
    kinds = D["kind"]
    homeq = D["home"][D["e"], D["k"]] == 1

    def record(key, c_list):
        out["curves"][key] = np.array([macro(D, c) for c in c_list])
        out["cnt"][key] = np.stack([counts(D, c) for c in c_list])
        out["cnt_kind"][key] = {rho: np.stack([counts(D, c_list[list(GRID).index(rho)] & (kinds == kk)) for kk in range(3)])
                                for rho in KEEP_KIND}

    def masks(s):
        pos, order = rank_offload(s, tb)
        return [pos < int(np.ceil(rho * N - 1e-9)) for rho in GRID], order

    # ---- reference rows
    out["reference"]["Device only"] = macro(D, c_dev)
    out["reference"]["Edge only"] = macro(D, fixed["edge-raw"])
    off0 = D["ent"] > m15.TAU0
    out["reference"]["Confidence-based offloading"] = dict(acc=macro(D, np.where(off0, fixed["edge-raw"], c_dev)), ratio=float(off0.mean()))
    for beta in BETAS:
        offc = r18.controller(co, Hc, beta)
        wP = r17.dgp_weight(ix, co, Hc, Hs, offc)
        acg = m15.mix_answer(wP, pc, psr) == y
        res = {ans: macro(D, np.where(offc, (acg if ans == "average-corrected-even" else fixed[ans]), c_dev)) for ans in ANSWERS}
        out["reference"][f"entropy controller beta {beta}"] = dict(acc=res, ratio=float(offc.mean()))
    # ---- comparator scores
    sc = {"device-entropy": Hc, "device-max-prob": -pc.max(1).astype(np.float64), "device-energy": X["energy"].astype(np.float64),
          "own-mahalanobis": X["maha"].astype(np.float64), "own-knn": X["knn"].astype(np.float64)}
    for s_name, s in sc.items():
        out["auroc"][s_name] = {sp_: auroc(s[m_], nonmain[m_]) for sp_, m_ in
                                (("all", np.ones(N, bool)), ("home", homeq), ("away", ~homeq))}
        ms, _ = masks(s)
        for ans in ANSWERS:
            cl = []
            for rho, off in zip(GRID, ms):
                if ans == "average-corrected-even":
                    w = w_auto if rho >= 1.0 else r17.dgp_weight(ix, co, Hc, Hs, off)
                    c = m15.mix_answer(w, pc, psr) == y
                else:
                    c = fixed[ans]
                cl.append(np.where(off, c, c_dev))
            record((s_name, ans), cl)
        del ms
    # ---- request-kind oracle
    ms, _ = masks(nonmain.astype(np.float64))
    for ans in ("edge-raw", "edge-corrected-even"):
        record((ORACLE, ans), [np.where(off, fixed[ans], c_dev) for off in ms])
    out["auroc"][ORACLE] = {"all": 1.0}
    # ---- detectors
    V = X["variants"] if detectors is None else detectors
    for v in V:
        j = X["variants"].index(v)
        g = X["g"][:, j].astype(np.float64)
        has = g >= 0
        Hown, Hoth = X["hist_own"][:, :, j].astype(np.float64), X["hist_other"][:, :, j].astype(np.float64)
        out["auroc"][v] = {sp_: auroc(1 - g[m_], nonmain[m_]) for sp_, m_ in
                           (("all", np.ones(N, bool)), ("home", homeq), ("away", ~homeq))}
        ms, order = masks(-g)
        cl_new, cl_raw, cl_ce, cl_avg = [], [], [], []
        is_main = main is not None and v.split("@")[0] == main and "@" not in v
        for rho, off in zip(GRID, ms):
            th = theta_of(g, order, int(off.sum()))
            rr = detector_r(D, co, off, th, Hown, Hoth, has)
            cl_new.append(np.where(off, corr_argmax(oo, a, rr["r"]) == y, c_dev))
            cl_raw.append(np.where(off, fixed["edge-raw"], c_dev))
            cl_ce.append(np.where(off, fixed["edge-corrected-even"], c_dev))
            if is_main:
                pq = m13.corrected(ps, Mrow, D["a_req"], rr["r"].astype(np.float32))
                cl_avg.append(np.where(off, m15.mix_answer(np.full(N, 0.2, np.float32), pc, pq) == y, c_dev))
                del pq
        record((v, NEW), cl_new)
        record((v, "edge-raw"), cl_raw)
        record((v, "edge-corrected-even"), cl_ce)
        if is_main:
            record((v, NEW_AVG), cl_avg)
        del ms, cl_new, cl_raw, cl_ce, cl_avg
        # extras for the main variant and its schedules
        if main is not None and v.split("@")[0] == main:
            ex = {}
            off = g < 0.5
            rr = detector_r(D, co, off, 0.5, Hown, Hoth, has, nonmain)
            ex["fixed 0.5"] = dict(acc=macro(D, np.where(off, corr_argmax(oo, a, rr["r"]) == y, c_dev)), ratio=float(off.mean()),
                                   s_err=s_error(rr, homeq))
            ms_, order_ = masks(-g)
            off = ms_[list(GRID).index(0.5)]
            rr = detector_r(D, co, off, theta_of(g, order_, int(off.sum())), Hown, Hoth, has, nonmain)
            ex["curve 0.5"] = dict(s_err=s_error(rr, homeq), r_mean=float(rr["r"][off].mean()))
            del ms_
            for beta in BETAS:
                offb, thb = g_controller(co, g, beta)
                rr = detector_r(D, co, offb, thb, Hown, Hoth, has, nonmain)
                ex[f"controller {beta}"] = dict(acc=macro(D, np.where(offb, corr_argmax(oo, a, rr["r"]) == y, c_dev)),
                                                ratio=float(offb.mean()), s_err=s_error(rr, homeq))
            out["extra"][v] = ex
    out["seconds"] = time.time() - t0
    return out


def s_error(rr, homeq):
    d = np.abs(rr["s_hat"] - rr["s_true"])
    ok = np.isfinite(d)
    return {sp_: (float(d[ok & m_].mean()) if (ok & m_).any() else float("nan")) for sp_, m_ in
            (("all", np.ones(len(d), bool)), ("home", homeq), ("away", ~homeq))} | {"defined_share": float(ok.mean())}


def analyse_run(name, conds=None, detectors=None):
    CACHE.mkdir(exist_ok=True)
    tr, rp = run_names(name)
    todo = conds or CONDITIONS
    for cond in todo:
        f = CACHE / f"{tr}__{cond}.pkl"
        if f.exists():
            print(f"{f.name} exists", flush=True)
            continue
        D, X = load(rp if cond == "frozen-replay" else tr)
        main = None
        if len(X["variants"]) > 18:
            main = X["variants"][18].split("@")[0]
        if cond == "first-day":
            D, X = subset(D, X, D["er"] <= DAY)
        elif cond == "second-day":
            D, X = subset(D, X, D["er"] > DAY)
        out = analyse(D, X, cond, main=main, detectors=detectors)
        out["main"] = main
        pickle.dump(out, open(f, "wb"))
        print(json.dumps(dict(run=tr, cond=cond, N=out["N"], E=out["E"], seconds=round(out["seconds"], 1),
                              checks=out["checks"])), flush=True)
        del D, X, out


# ------------------------------------------------------------------------------------------------ stage-1 checks
def check():
    """1. detector replaced by the true request kind -> request-kind oracle; 2. v32 rule (device-entropy +
    average-corrected-even) from this script = Round 20 code on the same records (a Round 15 run and a Round 18 smoke run);
    3. two-max corrected argmax = m13.corrected argmax."""
    res = {}
    r20 = _load("r20", ART / "driftgate_tmc_r20_claims" / "scripts" / "r20_recompute.py")
    # (a) Round 12 S1 seed 0 (the v32 table run, first day): this script's v32 rule at full offload and under the Round 18 controller mask
    D = m13.load(JR / "runs" / "phaseT12_fusion", "r12_s1_fixed040_s0")
    co, ix = m14.client_order(D), m14.window_index(D)
    Hc, Hs = D["ent"].astype(np.float64), m13.server_entropy(D["ps"])
    psr = m13.corrected(D["ps"], D["Mrow"], D["a_req"], 0.5)
    c20 = pickle.load(open(ART / "driftgate_tmc_r20_claims" / "cache" / "r12_s1_fixed040_s0.pkl", "rb"))
    w_auto, _ = r16.driftgate_weight(D)
    ans1 = m15.mix_answer(w_auto, D["pc"], psr)
    res["v32_full_offload_counts_equal_r20"] = bool(np.array_equal(counts(D, ans1 == D["y"]), c20["cnt"]["DriftGate @ 1.0"]))
    offc = r18.controller(co, Hc, 0.5)
    wP = r17.dgp_weight(ix, co, Hc, Hs, offc)
    ansP = np.where(offc, m15.mix_answer(wP, D["pc"], psr), D["cp"])
    res["v32_beta0.5_counts_equal_r20"] = bool(np.array_equal(counts(D, ansP == D["y"]), c20["cnt"]["DriftGate @ 0.5"]))
    res["v32_full_offload_macro"] = macro(D, ans1 == D["y"])
    res["r20_macro"] = float(np.mean(r20.macro_rounds(c20["cnt"]["DriftGate @ 1.0"], c20["n_ek"], c20["present"])))
    a = D["a_req"].astype(np.float64)
    oo = own_other_max(D["ps"], D["Mrow"])
    res["corr_argmax_agree_m13_r0.5"] = float((corr_argmax(oo, a, 0.5) == psr.argmax(1)).mean())
    rr = np.random.default_rng(0).uniform(0.05, 0.95, len(a))
    res["corr_argmax_agree_m13_random_r"] = float((corr_argmax(oo, a, rr) == m13.corrected(D["ps"], D["Mrow"], D["a_req"], rr.astype(np.float32)).argmax(1)).mean())
    del D, psr
    # (b) a Round 18 smoke run (2 rounds, the same record format): this script's curve code vs direct computation
    smoke = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if smoke is not None:
        global RUNS
        RUNS = smoke.parent
        D, X = load(smoke.name)
        out = analyse(D, X, "smoke", main=None, detectors=[X["variants"][0]])
        co, ix = m14.client_order(D), m14.window_index(D)
        w_auto, _ = r16.driftgate_weight(D)
        psr = m13.corrected(D["ps"], D["Mrow"], D["a_req"], 0.5)
        direct = macro(D, m15.mix_answer(w_auto, D["pc"], psr) == D["y"])
        res["smoke_v32_full_offload_script"] = float(out["curves"][("device-entropy", "average-corrected-even")][-1])
        res["smoke_v32_full_offload_direct"] = direct
        # oracle substitution: detector column replaced by g = 1[Main] (0/1), threshold 0.5, answer edge-raw
        X2 = dict(X)
        X2["g"] = np.repeat((D["kind"] == 0).astype(np.float16)[:, None], len(X["variants"]), 1)
        o2 = analyse(D, X2, "smoke", main=None, detectors=[X["variants"][0]])
        nm_share = float((D["kind"] != 0).mean())
        rho_i = int(np.argmin(np.abs(GRID - nm_share)))
        off = (X2["g"][:, 0] < 0.5)
        res["oracle_sub_offload_equals_nonmain"] = bool(np.array_equal(off, D["kind"] != 0))
        direct_or = macro(D, np.where(D["kind"] != 0, D["sp"] == D["y"], D["cp"] == D["y"]))
        res["oracle_direct_at_nonmain_share"] = direct_or
        res["oracle_sub_threshold_0.5_edge_raw"] = macro(D, np.where(off, D["sp"] == D["y"], D["cp"] == D["y"]))
        res["nonmain_share"] = nm_share
        res["oracle_curve_grid_point"] = dict(rho=float(GRID[rho_i]), oracle=float(o2["curves"][(ORACLE, "edge-raw")][rho_i]),
                                              detector_sub=float(o2["curves"][(X["variants"][0], "edge-raw")][rho_i]))
        res["oracle_sub_curve_equal_oracle_curve"] = bool(np.array_equal(o2["curves"][(ORACLE, "edge-raw")],
                                                                         o2["curves"][(X["variants"][0], "edge-raw")]))
    print(json.dumps(res, indent=1))
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "tables" / "R18oc_stage1_checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        analyse_run(sys.argv[2], sys.argv[3:] or None)
    elif sys.argv[1] == "check":
        check()
