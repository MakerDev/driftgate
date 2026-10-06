"""Round 18: same-budget online comparison (A2), offloading claims (A3), action utility (B1/B3), D4 support (B2),
conditional combination (C), Main-share sensitivity (E1). Definitions: R18_plan.md.

Reference answers come from the Round 16 phase A cache (Round 15 definitions, start-checked bit-identical against the
Round 15 cache); the Round 15 helpers (DriftGate-P weight, windows, interpolation) and the Round 17 detector code are
imported unchanged.

Subcommands (repository root):
  run SETTING SEED     per-run analysis -> cache/<run>.pkl
  features SETTING SEED  detector diagnostics (S1 replay / S2 replay only; Round 17 records) -> cache/<run>__feat.pkl
  tables               all tables
"""
import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ART = JR / "artifacts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


r17 = _load("r17", ART / "driftgate_tmc_r17_featgate" / "scripts" / "r17_stage0.py")
r16, m15, m14, m13 = r17.r16, r17.m15, r17.m14, r17.m13
TAB, CACHE = HERE / "tables", HERE / "cache"
for d_ in (TAB, CACHE):
    d_.mkdir(parents=True, exist_ok=True)

R12, R13, R15 = JR / "runs" / "phaseT12_fusion", JR / "runs" / "phaseT13b_arch", JR / "runs" / "phaseT15_replay"
SETTINGS = {   # name -> (runs dir, pattern, seeds, period)
    "S1": (R12, "r12_s1_fixed040_s{}", range(5), "day1"), "S2": (R12, "r12_s2_fixed040_s{}", range(3), "day1"),
    "S1-fast": (R12, "r12_s1fast_fixed040_s{}", range(3), "day1"),
    "partial participation": (R12, "r12_s4part05_fixed040_s{}", range(3), "day1"),
    "stepwise change": (R12, "r12_t1_A_fx40_s{}", range(5), "day1"), "random mobility": (R12, "r12_t1_mob_fx40_s{}", range(3), "day1"),
    "CIFAR-100": (R13, "r13b_c100_s{}", range(3), "day1"), "ResNet-18": (R12, "r12_e2_res_fx40_A_s{}", range(3), "day1"),
    "K=200": (R12, "r12_s3k200_fixed040_s{}", range(3), "day1"), "K=500": (R12, "r12_s3k500_fixed040_s{}", range(3), "day1"),
    "S1 replay": (R15, "r15_s1_replay_s{}", range(5), "replay"), "S2 replay": (R15, "r15_s2_replay_s{}", range(3), "replay"),
}
DAY1 = [s for s, v in SETTINGS.items() if v[3] == "day1"]
REPLAY = ["S1 replay", "S2 replay"]
BETAS = [0.25, 0.5, 0.75, 1.0]
WINDOWS = {"full": None, "round<=30": (None, 30), "round>30": (30, None), "round>50": (50, None)}
BINS = [tuple(b) for b in m14.BINS]
SPLITS = ["all", "home", "away", "Main", "OOP", "OOR"]
FIXED = {"Raw edge only": "B2", "Probability average": "B3", "Logit sum": "R-PoE",
         "Correction + fixed w 0.5": "correction + w 0.5", "Correction + development w 0.2": "correction + dev w 0.2",
         "Corrected edge only": "corrected edge only", "Correction + product": "correction + product",
         "Correction + learned weight": "correction + learned weight"}
ADAPT = ["DriftGate-P", "No correction + adaptive w"]
REFS = list(FIXED)[:7] + ["No correction + adaptive w", "Correction + learned weight"]   # 9 rules for strongest reference
CANDS = ["conditional (adaptive w)", "conditional (fixed w 0.5)", "conditional (raw edge, adaptive w)"]
EPS = 1e-12
S_GRID = [0.3, 0.5, 0.65, 0.8, 0.9]
FEATURE_BYTES = 128 * 8 * 8 * 4      # CNN smashed data (float32); used for the CNN settings only


def run_name(st, s):
    return SETTINGS[st][1].format(s)


# ------------------------------------------------------------------------------------------------ helpers
def controller(co, Hc, beta):
    """Round 15 C2 controller with a target ratio beta (r15_analysis.online_offload with BETA replaced)."""
    if beta >= 1.0:
        return np.ones(len(Hc), bool)
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
            tau[s0 + i] = np.quantile(x[:i], 1 - beta)
        if n > m15.CTRL_W:
            tau[s0 + m15.CTRL_W:e0] = np.quantile(sliding_window_view(x[:-1], m15.CTRL_W), 1 - beta, axis=1)
    out = np.empty(len(x_all), bool)
    out[oc] = x_all > tau
    return out


def conditional(w, pd_, pe_, Mrow, chunk=1 << 20):
    """p(c) = E w d_c / D + (1 - w) e_c on M_k, e_c elsewhere; corrected-edge fallback when D or E < EPS."""
    N = len(pd_)
    w = np.broadcast_to(np.asarray(w, np.float64), (N,))
    out = np.empty(N, np.int64)
    fb = np.zeros(N, bool)
    for s in range(0, N, chunk):
        d = pd_[s:s + chunk].astype(np.float64)
        e = pe_[s:s + chunk].astype(np.float64)
        M = Mrow[s:s + chunk]
        Dm = (d * M).sum(1)
        Em = (e * M).sum(1)
        bad = (Dm < EPS) | (Em < EPS)
        ww = w[s:s + chunk][:, None]
        p = np.where(M, (Em / np.where(bad, 1.0, Dm))[:, None] * ww * d + (1 - ww) * e, e)
        a = p.argmax(1)
        a[bad] = e[bad].argmax(1)
        out[s:s + chunk] = a
        fb[s:s + chunk] = bad
    return out, fb


def pr_metrics(D, c):
    """per-round values of overall, home, away, Main, OOP, OOR accuracy (Round 15 definitions; nan when absent)."""
    E, K, g, present, home, kind = D["E"], D["K"], D["g"], D["present"], D["home"], D["kind"]
    cf = c.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ = np.where(present, np.bincount(g, weights=cf, minlength=E * K).reshape(E, K) / D["n_ek"], np.nan)
    out = {"all": np.nanmean(A_, axis=1)}
    hm = np.where(present & (home == 1), A_, np.nan)
    aw = np.where(present & (home == 0), A_, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["home"] = np.nanmean(hm, axis=1)
            out["away"] = np.nanmean(aw, axis=1)
            for i, kn in enumerate(["Main", "OOP", "OOR"]):
                selk = kind == i
                ck = np.bincount(g, weights=cf * selk, minlength=E * K).reshape(E, K)
                nk = np.bincount(g, weights=selk, minlength=E * K).reshape(E, K)
                ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
                out[kn] = np.nanmean(ak, axis=1)
    return out


def weights_all(D):
    return m13.weights(D, np.ones(D["E"], bool))


def err_groups(D, ans, wts):
    y, kind, Mrow = D["y"], D["kind"], D["Mrow"]
    wrong = ans != y
    tm = kind == 0
    pm = Mrow[np.arange(len(y)), ans]
    g = {"Main->non-Main": wrong & tm & ~pm, "non-Main->Main": wrong & ~tm & pm,
         "Main->other Main": wrong & tm & pm, "non-Main->other non-Main": wrong & ~tm & ~pm}
    out = {k: float((wts * v).sum()) for k, v in g.items()}
    out["total error"] = float((wts * wrong).sum())
    return out


def auroc_pm(score, u):
    """AUROC of score for u = +1 against u = -1 (requests with u = 0 excluded)."""
    m = u != 0
    s, p = np.asarray(score, np.float64)[m], u[m] > 0
    if p.all() or (~p).all():
        return float("nan")
    return m14.auroc(s, p)


def margin(P):
    t = np.partition(P, -2, axis=1)
    return t[:, -1] - t[:, -2]


# ------------------------------------------------------------------------------------------------ per run
def analyse(st, seed):
    t0 = time.time()
    runs, _, _, period = SETTINGS[st]
    name = run_name(st, seed)
    D = m13.load(runs, name)
    rp = r16.refs_path(str(runs), name)
    z = np.load(rp)
    ans = {k: z[k].astype(np.int16) for k in r16.ANS_KEYS}
    w_auto = z["w_auto"]
    y, pc, ps, cp, sp, ent, kind, Mrow, a = (D[k] for k in ("y", "pc", "ps", "cp", "sp", "ent", "kind", "Mrow", "a_req"))
    N, E, K = len(y), D["E"], D["K"]
    er = D["er"]
    psr = m13.corrected(ps, Mrow, a, 0.5)
    assert np.array_equal(psr.argmax(1), ans["corrected edge only"])
    Hc, Hs = ent.astype(np.float64), m13.server_entropy(ps)
    ix, co = m14.window_index(D), m14.client_order(D)
    cells = np.sort(m15.cells_matrix(runs, name, K, er), axis=2)
    two = ((cells >= 0).sum(2) == 2)[D["e"], D["k"]]
    out = dict(setting=st, seed=seed, name=name, period=period, er=er, n_req=N, refs_file=str(rp),
               refs_size=rp.stat().st_size, refs_mtime_ns=rp.stat().st_mtime_ns, checks={}, pr={}, srv={}, calls={})
    out["checks"]["two_cell_share"] = float(two.mean())
    dev = pr_metrics(D, cp == y)
    out["pr"]["Device only"] = dev
    wts = weights_all(D)
    # ---- online beta masks
    for beta in BETAS:
        off = controller(co, Hc, beta)
        if beta == 0.5:
            out["checks"]["beta0.5_equals_r15"] = bool(np.array_equal(off, m15.online_offload(co, Hc)))
        wP = r17.dgp_weight(ix, co, Hc, Hs, off)
        if beta == 1.0:
            out["checks"]["beta1_wP_minus_w_auto_maxabs"] = float(np.abs(wP - w_auto).max())
        A = {k_: ans[v] for k_, v in FIXED.items()}
        A["DriftGate-P"] = m15.mix_answer(wP, pc, psr).astype(np.int16)
        A["No correction + adaptive w"] = m15.mix_answer(wP, pc, ps).astype(np.int16)
        A["conditional (adaptive w)"], fb1 = conditional(wP, pc, psr, Mrow)
        A["conditional (fixed w 0.5)"], fb2 = conditional(0.5, pc, psr, Mrow)
        A["conditional (raw edge, adaptive w)"], fb3 = conditional(wP, pc, ps, Mrow)
        for k_ in CANDS:
            A[k_] = A[k_].astype(np.int16)
        if beta == 1.0:
            out["checks"]["beta1_DG-P_equals_DriftGate"] = float((A["DriftGate-P"] == ans["DriftGate"]).mean())
        out["checks"][f"fallback share beta {beta}"] = [float(fb1[off].mean()), float(fb2[off].mean()), float(fb3[off].mean())]
        for k_, v in A.items():
            fin = np.where(off, v, cp)
            out["pr"][f"{k_} @ {beta}"] = pr_metrics(D, fin == y)
            if beta in (0.5, 1.0) and k_ in ("DriftGate-P", "Corrected edge only", "conditional (adaptive w)", "Probability average"):
                out.setdefault("errors", {})[f"{k_} @ {beta}"] = err_groups(D, fin, wts)
        out["srv"][beta] = {wn: float(off[sel_rounds(er, wn)[D["e"]]].mean()) for wn in WINDOWS}
        out["calls"][beta] = float((off * (1 + two)).mean())
        if beta == 1.0:
            A1, wP1 = A, wP
    # ---- fixed-threshold calls (for A3)
    tg = list(m13.TAU) + [np.inf]
    out["tau_srv"] = np.array([float((ent > t).mean()) for t in tg])
    out["tau_calls"] = np.array([float(((ent > t) * (1 + two)).mean()) for t in tg])
    # ---- B1 / B3 at full offloading: base DriftGate, alternative corrected edge only
    base, alt = ans["DriftGate"].astype(np.int64), ans["corrected edge only"].astype(np.int64)
    u = (alt == y).astype(np.int8) - (base == y).astype(np.int8)
    dis = base != alt
    b = dict(w_plus=float(wts[u == 1].sum()), w_minus=float(wts[u == -1].sum()), w_zero=float(wts[u == 0].sum()),
             w_disagree=float(wts[dis].sum()),
             dis_plus=float(wts[dis & (u == 1)].sum() / max(wts[dis].sum(), 1e-300)),
             dis_minus=float(wts[dis & (u == -1)].sum() / max(wts[dis].sum(), 1e-300)),
             dis_zero=float(wts[dis & (u == 0)].sum() / max(wts[dis].sum(), 1e-300)),
             acc_base=float((wts * (base == y)).sum()), acc_alt=float((wts * (alt == y)).sum()),
             acc_oracle=float((wts * ((base == y) | (alt == y))).sum()))
    Dm = (pc * Mrow).sum(1, dtype=np.float64)
    Em = (psr * Mrow).sum(1, dtype=np.float64)
    qe = w_auto * (Dm - Em)
    mix = (w_auto.astype(np.float32)[:, None] * pc + (1 - w_auto.astype(np.float32))[:, None] * psr)
    mdiff = margin(psr) - margin(mix)
    del mix
    sig = {"device entropy": (Hc, Hc > 0.8), "corrected-edge Main probability sum": (-Em, Em < 0.5),
           "margin difference (corrected edge - DriftGate)": (mdiff, mdiff > 0), "Q_DG - E": (qe, qe > 0)}
    b["signals"] = {}
    for sn, (score, sw) in sig.items():
        b["signals"][sn] = dict(auroc=auroc_pm(score, u), switch=float(wts[sw].sum()),
                                rescued=float(wts[sw & (u == 1)].sum()), harmed=float(wts[sw & (u == -1)].sum()))
    b["mean_qe"] = float((wts * qe).sum())
    b["qe_by_u"] = {int(v): float(qe[u == v].mean()) if (u == v).any() else float("nan") for v in (-1, 0, 1)}
    b["mean_w"] = float((wts * w_auto).sum())
    out["utility"] = b
    # ---- E1 Main-share sensitivity (full offloading predictions)
    g = D["g"]
    main = kind == 0
    e1 = {}
    for k_ in ["DriftGate-P", "Corrected edge only", "Probability average", "Logit sum", "Raw edge only",
               "Correction + development w 0.2", "conditional (adaptive w)"]:
        c_ = A1[k_] == y
        cm = np.bincount(g, weights=c_ & main, minlength=E * K).reshape(E, K)
        cn = np.bincount(g, weights=c_ & ~main, minlength=E * K).reshape(E, K)
        nm_ = np.bincount(g, weights=main, minlength=E * K).reshape(E, K)
        nn_ = np.bincount(g, weights=~main, minlength=E * K).reshape(E, K)
        ok = (nm_ > 0) & (nn_ > 0)
        aM, aN = cm / np.maximum(nm_, 1), cn / np.maximum(nn_, 1)
        e1[k_] = {}
        for s_ in S_GRID:
            with np.errstate(invalid="ignore"):
                v = np.where(ok, s_ * aM + (1 - s_) * aN, np.nan)
                pr = np.nanmean(v, axis=1)
            e1[k_][s_] = float(np.nanmean(pr))
        e1["coverage_device_rounds"] = float(ok[D["present"]].mean())
        e1["coverage_requests"] = float(D["n_ek"][ok].sum() / D["n_ek"].sum())
        e1["observed_main_share"] = float(main.mean())
    out["e1"] = e1
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(CACHE / f"{name}.pkl", "wb"))
    print(json.dumps(dict(name=name, seconds=round(out["seconds"], 1), checks=out["checks"]), default=str))


def sel_rounds(er, wn):
    if wn == "full":
        return np.ones(len(er), bool)
    lo, hi = WINDOWS[wn]
    m = np.ones(len(er), bool)
    if lo is not None:
        m &= er > lo
    if hi is not None:
        m &= er <= hi
    return m


# ------------------------------------------------------------------------------------------------ detectors (B1/B2)
def features(st, seed):
    assert st in REPLAY
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rn = f"r17_{st.split()[0].lower()}_replay_s{seed}"
    runs17 = JR / "runs" / "phaseT17_featgate"
    D = m13.load(runs17, rn)
    feat = dict(np.load(runs17 / f"{rn}_features.npz"))
    own = dict(np.load(runs17 / f"{rn}_own.npz"))
    sc, th, nm, info = r17.detectors(D, feat, own, dev)
    del feat
    name = run_name(st, seed)
    z = np.load(r16.refs_path(str(SETTINGS[st][0]), name))
    q15 = np.load(SETTINGS[st][0] / f"{name}_evalprobs.npz")
    assert np.array_equal(q15["req_label"], D["y"]) and np.array_equal(q15["cp"], D["cp"])
    y, kind, ent = D["y"], D["kind"], D["ent"].astype(np.float64)
    base, alt = z["DriftGate"].astype(np.int64), z["corrected edge only"].astype(np.int64)
    u = (alt == y).astype(np.int8) - (base == y).astype(np.int8)
    wts = weights_all(D)
    out = dict(setting=st, seed=seed, name=name, signals={}, support={})
    for dn, s, f in (("D1", sc["D1"], nm["D1@0.1"]), ("D2", sc["D2"], nm["D2@0.1"]), ("D3", sc["D3"], nm["D3"]), ("D4", sc["D4"], nm["D4"])):
        fin = np.isfinite(s)
        uu = np.where(fin, u, 0)
        out["signals"][dn] = dict(auroc=auroc_pm(np.where(fin, s, 0.0), uu), switch=float(wts[f].sum()),
                                  rescued=float(wts[f & (u == 1)].sum()), harmed=float(wts[f & (u == -1)].sum()),
                                  coverage=float(fin.mean()))
    # B2: support where D4 is defined
    d4 = np.isfinite(sc["D4"])
    pos = kind != 0
    hm = D["home"][D["e"], D["k"]] == 1
    k_i = D["k"]
    for dn, s in (("entropy", ent), ("D1", sc["D1"]), ("D2", sc["D2"].astype(np.float64)), ("D3", sc["D3"]), ("D4", sc["D4"])):
        r_ = {}
        for sub, m in (("all", d4), ("home", d4 & hm), ("away", d4 & ~hm)):
            r_[sub] = m14.auroc(s[m], pos[m]) if (pos[m].any() and (~pos[m]).any()) else float("nan")
        per_dev, one_class, one_class_req = [], 0, 0
        for k in range(D["K"]):
            m = d4 & (k_i == k)
            if not m.any():
                continue
            if pos[m].all() or (~pos[m]).all():
                one_class += 1
                one_class_req += int(m.sum())
                continue
            per_dev.append(m14.auroc(s[m], pos[m]))
        r_["device_mean"] = float(np.mean(per_dev)) if per_dev else float("nan")
        r_["devices_with_one_kind"] = one_class
        r_["requests_of_one_kind_devices_share"] = float(one_class_req / max(d4.sum(), 1))
        out["support"][dn] = r_
    out["support"]["d4_coverage"] = float(d4.mean())
    out["support"]["d4_coverage_home"] = float(d4[hm].mean())
    out["support"]["d4_coverage_away"] = float(d4[~hm].mean())
    rb4 = np.where(nm["D4"], alt, base)
    out["support"]["Rb_D4_acc_defined"] = float((wts[d4] * (rb4 == y)[d4]).sum() / wts[d4].sum())
    out["support"]["DG_acc_defined"] = float((wts[d4] * (base == y)[d4]).sum() / wts[d4].sum())
    out["support"]["DG_acc_undefined"] = float((wts[~d4] * (base == y)[~d4]).sum() / max(wts[~d4].sum(), 1e-300))
    pickle.dump(out, open(CACHE / f"{name}__feat.pkl", "wb"))
    print(json.dumps(dict(name=name, d4_cov=out["support"]["d4_coverage"])))


# ------------------------------------------------------------------------------------------------ tables
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


def load_all():
    return {st: [pickle.load(open(CACHE / f"{run_name(st, s)}.pkl", "rb")) for s in SETTINGS[st][2]] for st in SETTINGS}


def rowval(a, key, split, rmask):
    v = a["pr"][key][split][rmask]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def row_defs(R):
    """(table, row id, setting, row mask fn, split) for the C rows and the A2 rows."""
    rows = []
    for st in DAY1:
        for wn in WINDOWS:
            rows.append(("window", f"{st} | {wn}", st, (lambda er, wn=wn: sel_rounds(er, wn)), "all"))
    for st in ("S1", "S2"):
        for bn, lo, hi in BINS:
            rows.append(("time of day", f"{st} | {bn}", st, (lambda er, lo=lo, hi=hi: ((300 + 6 * (er - 1)) >= lo) & ((300 + 6 * (er - 1)) < hi)), "all"))
    for st in REPLAY:
        for sp_ in ("all", "home", "away"):
            rows.append(("replay", f"{st} | {sp_}", st, (lambda er: np.ones(len(er), bool)), sp_))
    return rows


def tables():
    R = load_all()
    # ---- manifest
    man = []
    for st, runs_ in R.items():
        for a in runs_:
            man.append([st, a["name"], str(SETTINGS[st][0].relative_to(JR)), a["period"], a["n_req"], Path(a["refs_file"]).name,
                        a["refs_size"], a["refs_mtime_ns"], json.dumps(a["checks"], default=str)])
    wcsv("R18_reference_manifest.csv", ["setting", "run", "runs dir", "period", "requests", "reference answers (Round 16 cache)",
                                        "bytes", "mtime_ns", "checks"], man)
    # ---- A2: same budget
    a2, a2s = [], []
    rules = ["Device only"] + list(FIXED)[:7] + ["Correction + learned weight", "DriftGate-P", "No correction + adaptive w"] + CANDS
    for st, runs_ in R.items():
        er = runs_[0]["er"]
        full = np.ones(len(er), bool)
        for beta in BETAS:
            key = lambda r: r if r == "Device only" else f"{r} @ {beta}"
            vals = {r: np.array([rowval(a, key(r), "all", full) for a in runs_]) * 100 for r in rules}
            refm = {r: vals[r].mean() for r in REFS}
            strongest = max(REFS, key=lambda r: refm[r])
            for r in rules:
                a2.append([st, beta, r, fmt(vals[r]), fmt(vals[r] - vals["DriftGate-P"], 3) if r != "DriftGate-P" else "",
                           fmt(vals[r] - vals[strongest], 3), strongest,
                           fmt([a["srv"][beta]["full"] for a in runs_], 3) if r != "Device only" else "0",
                           fmt([a["calls"][beta] for a in runs_], 3) if r != "Device only" else "0"] +
                          [fmt(np.array([rowval(a, key(r), sp_, full) for a in runs_]) * 100) for sp_ in SPLITS[1:]])
                for a, x, xd in zip(runs_, vals[r], vals["DriftGate-P"]):
                    a2s.append([st, beta, r, a["name"], f"{x:.4f}", f"{x - xd:.4f}"])
    wcsv("R18_online_same_budget.csv", ["setting", "beta", "rule", "accuracy mean (SD)", "minus DriftGate-P pp", "minus strongest reference pp",
                                        "strongest reference", "realized offloading", "edge calls per request"] + SPLITS[1:], a2)
    wcsv("R18_online_same_budget_per_seed.csv", ["setting", "beta", "rule", "run", "accuracy pct", "minus DriftGate-P pp"], a2s)
    # offloading needed to reach the best beta = 1 reference
    need = []
    for st, runs_ in R.items():
        full = np.ones(len(runs_[0]["er"]), bool)
        dev = np.mean([rowval(a, "Device only", "all", full) for a in runs_]) * 100
        target_rule = max(REFS, key=lambda r: np.mean([rowval(a, f"{r} @ 1.0", "all", full) for a in runs_]))
        target = np.mean([rowval(a, f"{target_rule} @ 1.0", "all", full) for a in runs_]) * 100
        for r in REFS + ["DriftGate-P", "conditional (adaptive w)"]:
            xs = [0.0] + [np.mean([a["srv"][b]["full"] for a in runs_]) for b in BETAS]
            ys = [dev] + [np.mean([rowval(a, f"{r} @ {b}", "all", full) for a in runs_]) * 100 for b in BETAS]
            reach = "not reached"
            for j in range(1, len(xs)):
                if ys[j] >= target - 1e-12:
                    if ys[j - 1] >= target - 1e-12:
                        reach = f"{xs[j - 1]:.3f}"
                    else:
                        reach = f"{xs[j - 1] + (xs[j] - xs[j - 1]) * (target - ys[j - 1]) / (ys[j] - ys[j - 1]):.3f}"
                    break
            need.append([st, r, target_rule, f"{target:.2f}", reach] + [f"{v:.2f}" for v in ys])
    wcsv("R18_online_offload_needed.csv", ["setting", "rule", "target rule (best at beta 1)", "target accuracy", "offloading needed (interpolated)",
                                           "accuracy at offloading 0 (Device only)"] + [f"accuracy at beta {b}" for b in BETAS], need)
    # ---- C: conditional combination rows and judgements
    defs = row_defs(R)
    crow, cseed, judge = [], [], {}
    for beta in (1.0, 0.5, 0.75, 0.25):
        okstrict = True
        for tbl, rid, st, mfn, sp_ in defs:
            runs_ = R[st]
            m = mfn(runs_[0]["er"])
            v = lambda r: np.array([rowval(a, f"{r} @ {beta}", sp_, m) for a in runs_]) * 100
            dg = v("DriftGate-P")
            refv = {r: v(r) for r in REFS}
            strongest = max(REFS, key=lambda r: np.nanmean(refv[r]))
            for c_ in CANDS:
                cv = v(c_)
                d_dg, d_st = cv - dg, cv - refv[strongest]
                crow.append([beta, tbl, rid, c_, fmt(cv), fmt(d_dg, 3), fmt(d_st, 3), strongest, fmt(dg), fmt(refv[strongest])])
                for a, x, xd, xs_ in zip(runs_, cv, dg, refv[strongest]):
                    cseed.append([beta, rid, c_, a["name"], f"{x:.4f}", f"{x - xd:.4f}", f"{x - xs_:.4f}", strongest])
                if c_ == CANDS[0]:
                    judge.setdefault(beta, []).append((rid, float(d_dg.mean()), float(d_st.mean()), strongest))
    wcsv("R18_conditional_combination.csv", ["beta", "row group", "row", "candidate", "accuracy mean (SD)", "minus DriftGate pp", "minus strongest reference pp",
                                             "strongest reference", "DriftGate", "strongest reference accuracy"], crow)
    wcsv("R18_conditional_combination_per_seed.csv", ["beta", "row", "candidate", "run", "accuracy pct", "minus DriftGate pp",
                                                      "minus strongest reference pp", "strongest reference"], cseed)
    strict = {}
    for beta in (1.0, 0.5):
        rows_ = judge[beta]
        fails = [(r, d1, d2, s_) for r, d1, d2, s_ in rows_ if d1 < -1e-8 or d2 < -1e-8]
        strict[beta] = dict(passed=not fails, n_rows=len(rows_), n_fail=len(fails),
                            fails=[dict(row=r, minus_driftgate=d1, minus_strongest=d2, strongest=s_) for r, d1, d2, s_ in fails])
    core = {}
    for beta in (1.0, 0.5):
        for rid in ("S1 | full", "S2 | full", "S1 replay | all", "S2 replay | all"):
            r_ = next(x for x in judge[beta] if x[0] == rid)
            core[f"{rid} @ {beta}"] = dict(minus_driftgate=r_[1], minus_strongest=r_[2], strongest=r_[3])
    cand_ok = all(v["minus_driftgate"] >= -1e-8 for v in core.values()) and \
        all(v["minus_strongest"] >= 0.5 - 1e-8 for k_, v in core.items() if "replay" in k_)
    # ---- C3 error groups
    eg = []
    for st, runs_ in R.items():
        for beta in (1.0, 0.5):
            base = [a["errors"][f"DriftGate-P @ {beta}"] for a in runs_]
            for r in ("Corrected edge only", "conditional (adaptive w)", "Probability average"):
                oth = [a["errors"][f"{r} @ {beta}"] for a in runs_]
                gk = ["Main->non-Main", "non-Main->Main", "Main->other Main", "non-Main->other non-Main"]
                diffs = {k_: np.array([o[k_] - b_[k_] for o, b_ in zip(oth, base)]) * 100 for k_ in gk + ["total error"]}
                closure = max(abs(sum(diffs[k_][i] for k_ in gk) - diffs["total error"][i]) for i in range(len(runs_)))
                eg.append([st, beta, r, "DriftGate-P"] + [fmt(diffs[k_], 3) for k_ in gk + ["total error"]] + [f"{closure:.2e}"] +
                          [fmt(np.array([b_[k_] for b_ in base]) * 100) for k_ in gk])
    wcsv("R18_conditional_error_groups.csv", ["setting", "beta", "rule", "minus", "Main->non-Main pp", "non-Main->Main pp", "Main->other Main pp",
                                              "non-Main->other non-Main pp", "total error pp", "max closure error",
                                              "DriftGate-P Main->non-Main", "DriftGate-P non-Main->Main", "DriftGate-P Main->other Main",
                                              "DriftGate-P non-Main->other non-Main"], eg)
    # ---- B1 / B3 utility
    ut = []
    feat = {}
    for st in REPLAY:
        for s in SETTINGS[st][2]:
            p = CACHE / f"{run_name(st, s)}__feat.pkl"
            if p.exists():
                feat.setdefault(st, []).append(pickle.load(open(p, "rb")))
    for st, runs_ in R.items():
        U = [a["utility"] for a in runs_]
        ut.append([st, "structure", "", fmt([u["w_plus"] * 100 for u in U], 3), fmt([u["w_minus"] * 100 for u in U], 3),
                   fmt([(u["w_plus"] - u["w_minus"]) * 100 for u in U], 3), fmt([u["w_disagree"] * 100 for u in U], 3),
                   f"+1 {fmt([u['dis_plus'] for u in U], 3)} / -1 {fmt([u['dis_minus'] for u in U], 3)} / 0 {fmt([u['dis_zero'] for u in U], 3)}",
                   f"oracle - DriftGate {fmt([(u['acc_oracle'] - u['acc_base']) * 100 for u in U])}; oracle - corrected edge "
                   f"{fmt([(u['acc_oracle'] - u['acc_alt']) * 100 for u in U])}"])
        sigs = list(U[0]["signals"])
        for sn in sigs:
            S = [u["signals"][sn] for u in U]
            ut.append([st, "signal", sn, fmt([x["rescued"] * 100 for x in S], 3), fmt([x["harmed"] * 100 for x in S], 3),
                       fmt([(x["rescued"] - x["harmed"]) * 100 for x in S], 3), fmt([x["switch"] * 100 for x in S], 2),
                       f"AUROC +1 vs -1 {fmt([x['auroc'] for x in S], 3)}", ""])
        ut.append([st, "Q_DG - E", "", "", "", "", "", f"mean {fmt([u['mean_qe'] for u in U], 4)}; w mean {fmt([u['mean_w'] for u in U], 3)}",
                   "by u: " + " ".join(f"{k_}:{fmt([u['qe_by_u'][k_] for u in U], 4)}" for k_ in (-1, 0, 1))])
        if st in feat:
            for dn in ("D1", "D2", "D3", "D4"):
                S = [f_["signals"][dn] for f_ in feat[st]]
                ut.append([st, "signal (Round 17 detector)", dn, fmt([x["rescued"] * 100 for x in S], 3), fmt([x["harmed"] * 100 for x in S], 3),
                           fmt([(x["rescued"] - x["harmed"]) * 100 for x in S], 3), fmt([x["switch"] * 100 for x in S], 2),
                           f"AUROC +1 vs -1 {fmt([x['auroc'] for x in S], 3)} (score coverage {fmt([x['coverage'] for x in S], 3)})", ""])
    wcsv("R18_action_utility.csv", ["setting", "item", "signal", "rescued pp (u=+1 switched) / P(u=+1)", "harmed pp (u=-1 switched) / P(u=-1)",
                                    "net pp", "switched share pp / disagreement share pp", "AUROC or disagreement structure", "headroom"], ut)
    sup = []
    for st, fl in feat.items():
        for dn in ("entropy", "D1", "D2", "D3", "D4"):
            S = [f_["support"][dn] for f_ in fl]
            sup.append([st, dn, fmt([x["all"] for x in S], 3), fmt([x["home"] for x in S], 3), fmt([x["away"] for x in S], 3),
                        fmt([x["device_mean"] for x in S], 3), fmt([x["devices_with_one_kind"] for x in S], 1),
                        fmt([x["requests_of_one_kind_devices_share"] for x in S], 3)])
        S = [f_["support"] for f_ in fl]
        sup.append([st, "D4 coverage (all / home / away)", fmt([x["d4_coverage"] for x in S], 3), fmt([x["d4_coverage_home"] for x in S], 3),
                    fmt([x["d4_coverage_away"] for x in S], 3), "", "", ""])
        sup.append([st, "accuracy on D4-defined requests: R-b(D4) / DriftGate; DriftGate on undefined",
                    fmt([x["Rb_D4_acc_defined"] * 100 for x in S]), fmt([x["DG_acc_defined"] * 100 for x in S]),
                    fmt([x["DG_acc_undefined"] * 100 for x in S]), "", "", ""])
    wcsv("R18_detector_matched_support.csv", ["setting", "score", "AUROC Main vs non-Main (D4-defined requests)", "home", "away",
                                              "mean of device AUROC", "devices with one kind only", "their share of D4-defined requests"], sup)
    # ---- E1
    e1 = []
    for st, runs_ in R.items():
        ks = [k_ for k_ in runs_[0]["e1"] if isinstance(runs_[0]["e1"][k_], dict)]
        for k_ in ks:
            e1.append([st, k_] + [fmt([a["e1"][k_][s_] * 100 for a in runs_]) for s_ in S_GRID] +
                      [fmt([(a["e1"][k_][s_] - a["e1"]["DriftGate-P"][s_]) * 100 for a in runs_], 3) for s_ in S_GRID])
        e1.append([st, "coverage (device-rounds / requests) and observed Main share",
                   fmt([a["e1"]["coverage_device_rounds"] for a in runs_], 3), fmt([a["e1"]["coverage_requests"] for a in runs_], 3),
                   fmt([a["e1"]["observed_main_share"] for a in runs_], 3)] + [""] * 7)
    wcsv("R18_main_share_sensitivity.csv", ["setting", "rule"] + [f"s={s_}" for s_ in S_GRID] + [f"minus DriftGate s={s_}" for s_ in S_GRID], e1)
    # ---- A3 from the Round 15 cache
    a3()
    summary = dict(strict=strict, core=core, verification_candidate=bool(cand_ok), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (TAB / "R18_conditional_decision.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(strict={b_: (v["passed"], v["n_fail"], v["n_rows"]) for b_, v in strict.items()}, candidate=cand_ok), indent=1))


def a3():
    R = load_all()
    FULL0 = ["B2", "B3", "R-PoE", "R-THE", "R-EM", "R-LR", "LEW"]
    FULL1 = FULL0 + ["no correction + adaptive w", "correction + w 0.5", "correction + dev w 0.2", "corrected edge only",
                     "correction + learned weight", "correction + product"]
    m15set = {"S1": "S1", "S2": "S2", "S1-fast": "S1-fast", "partial participation": "partial participation", "stepwise change": "stepwise change",
              "random mobility": "random mobility", "CIFAR-100": "CIFAR-100", "ResNet-18": "ResNet-18", "K=200": "K=200", "K=500": "K=500",
              "S1 replay": "S1 frozen replay", "S2 replay": "S2 frozen replay"}
    rows = []
    for st, key15 in m15set.items():
        rd, pat, seeds = m15.SETTINGS[key15]
        C15 = [r16.r15_cache_value(str(rd), pat.format(s)) for s in seeds]
        assert all(c is not None for c in C15), st
        assert [pat.format(s) for s in seeds] == [a["name"] for a in R[st]], st
        val = lambda rl: np.array([c["res"][rl]["full"] for c in C15]) * 100
        sv = np.mean([c["curves"]["full"]["DriftGate-P"]["srv"] for c in C15], axis=0)
        cu = np.mean([c["curves"]["full"]["DriftGate-P"]["acc"] for c in C15], axis=0) * 100
        b0 = val("B0").mean()
        b0_srv = np.mean([c["srv"]["full"]["B0"] for c in C15])
        tau_calls = np.mean([a["tau_calls"] for a in R[st]], axis=0)
        tau_srv = np.mean([a["tau_srv"] for a in R[st]], axis=0)
        assert np.allclose(tau_srv, sv, atol=1e-9), st
        i08 = int(np.flatnonzero(np.isclose(list(m13.TAU) + [np.inf], 0.8))[0])
        for tag, fs in (("Round 15 set (7 rules)", FULL0), ("with corrected variants (13 rules)", FULL1)):
            fm = {rl: val(rl).mean() for rl in fs}
            fb = max(fm, key=fm.get)
            o = np.argsort(sv, kind="stable")
            j = next((jj for jj in range(len(o)) if cu[o[jj]] >= fm[fb]), None)
            if j is None:
                rows.append([st, tag, r16.SHOW.get(fb, fb), f"{fm[fb]:.2f}", "not reached", "", "", "", f"{b0_srv:.3f}", f"{tau_calls[i08]:.3f}"])
                continue
            ti = o[j]
            rows.append([st, tag, r16.SHOW.get(fb, fb), f"{fm[fb]:.2f}", f"{sv[ti]:.3f}", f"{1 - sv[ti] / b0_srv:.3f}", f"{tau_calls[ti]:.3f}",
                         f"{1 - tau_calls[ti] / tau_calls[i08]:.3f}", f"{b0_srv:.3f}", f"{tau_calls[i08]:.3f}"])
    wcsv("R18_A3_offload_claims.csv", ["setting", "full-offload comparison set", "best full-offload rule", "its accuracy",
                                       "DriftGate-P offloading at first grid point reaching it", "reduction of offloaded requests vs confidence-based (0.8 nats)",
                                       "edge calls per request at that point", "reduction of edge calls vs confidence-based",
                                       "confidence-based offloading", "confidence-based edge calls per request"], rows)


def main():
    cmd = sys.argv[1]
    if cmd == "run":
        analyse(sys.argv[2], int(sys.argv[3]))
    elif cmd == "features":
        features(sys.argv[2], int(sys.argv[3]))
    elif cmd == "tables":
        tables()
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
