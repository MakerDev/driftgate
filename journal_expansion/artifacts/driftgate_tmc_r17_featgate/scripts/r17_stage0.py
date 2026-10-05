"""Round 17 stage 0: feature-distance detectors and gate rules on the frozen-model replays (decision_rule.md).

The reference rules repeat the Round 15 definitions through r16_stage1.reference_answers (start-checked in Round 16
against the Round 15 cache); the Round 15 helpers (DriftGate-P weight, controller, curves, interpolation) are imported
unchanged. Settings: r17_config.json.

Subcommands (repository root):
  run NAME      per-run analysis of runs/phaseT17_featgate/NAME (GPU for LEW, Label-shift EM and k-NN) -> cache/NAME.pkl
  tables        all stage-0 tables and decision_stage0.json from the cached runs
"""
import csv
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ART = JR / "artifacts"
CFG = json.loads((HERE / "r17_config.json").read_text())
_spec = importlib.util.spec_from_file_location("r16", ART / "driftgate_tmc_r16_selector" / "scripts" / "r16_stage1.py")
r16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r16)
m15, m14, m13 = r16.m15, r16.m14, r16.m13
TAB, CACHE = HERE / "tables", HERE / "cache"
for d_ in (TAB, CACHE):
    d_.mkdir(parents=True, exist_ok=True)
TAU_GRID = [float(t) for t in m13.TAU] + [float("inf")]
OPTS = CFG["server_use_points"]
ALPHAS = CFG["D1"]["alpha_candidates"]
assert ALPHAS == CFG["D2"]["alpha_candidates"]
KINDS = ["Main", "OOP", "OOR"]
REF_KEYS = ["B0", "B1", "B2", "B3", "R-PoE", "R-THE", "LEW", "R-ZTW", "R-EM", "R-LR", "DriftGate", "corrected edge only"]
SHOW = dict(r16.SHOW)
OTHER = CFG["strongest_other"]
assert OTHER == [k for k in REF_KEYS if k != "DriftGate"]


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


# ------------------------------------------------------------------------------------------------ detectors
def maha_stats(X, y, classes, eps_factor):
    """class means and the whitening matrix L with (x - mu)^T (S + eps I)^-1 (x - mu) = |(x - mu) L|^2."""
    X = np.asarray(X, np.float64)
    mu = np.stack([X[y == c].mean(0) for c in classes])
    R = X - mu[np.searchsorted(classes, y)]
    S = R.T @ R / len(X)
    eps = eps_factor * np.trace(S) / S.shape[0]
    P = np.linalg.inv(S + eps * np.eye(S.shape[0]))
    L = np.linalg.cholesky(P)
    return mu, L, eps


def maha_dists(F, mu, L):
    """[n, C] squared Mahalanobis distances."""
    Z = np.asarray(F, np.float64) @ L
    M = mu @ L
    return (Z * Z).sum(1)[:, None] - 2 * Z @ M.T + (M * M).sum(1)[None, :]


def knn_scores(Q, B, k, dev, loo=False, chunk=8192):
    import torch
    Bt = torch.nn.functional.normalize(torch.as_tensor(B, dtype=torch.float32, device=dev), dim=1)
    out = np.empty(len(Q), np.float32)
    kk = k + 1 if loo else k
    for s in range(0, len(Q), chunk):
        q = torch.nn.functional.normalize(torch.as_tensor(np.asarray(Q[s:s + chunk], np.float32), device=dev), dim=1)
        sims = q @ Bt.T
        out[s:s + chunk] = (1.0 - sims.topk(min(kk, Bt.shape[0]), dim=1).values[:, -1]).cpu().numpy()
    return out


def detectors(D, feat, own, dev):
    """scores (larger = non-Main) and non-Main judgements of D1 (both alphas), D2 (both alphas), D3, D4."""
    from sklearn.linear_model import LogisticRegression
    N, K, C = len(D["y"]), D["K"], D["C"]
    k_i = D["k"]
    fd = feat["fd"]
    own_fd, own_k, own_y = own["own_fd"], own["own_k"].astype(np.int64), own["own_y"].astype(np.int64)
    main, home = own["main"], own["home_cell"]
    out = {"D1": np.empty(N), "D2": np.empty(N, np.float32), "D3": np.empty(N), "D4": np.full(N, np.nan)}
    th = {f"D1@{a}": np.full(K, np.nan) for a in ALPHAS} | {f"D2@{a}": np.full(K, np.nan) for a in ALPHAS}
    info = dict(d3_heads=0, d3_devices_without_negatives=0, d1_eps=[], d1_classes=[], d2_bank=[])
    for k in range(K):
        rq = np.flatnonzero(k_i == k)
        sel = own_k == k
        Xk, yk = own_fd[sel], own_y[sel]
        cls = np.flatnonzero(main[k])
        assert set(np.unique(yk)) == set(cls), k
        mu, L, eps = maha_stats(Xk, yk, cls, CFG["D1"]["eps_factor"])
        d_own = maha_dists(Xk, mu, L).min(1)
        out["D1"][rq] = maha_dists(fd[rq], mu, L).min(1)
        d2_own = knn_scores(Xk, Xk, CFG["D2"]["k"], dev, loo=True)
        out["D2"][rq] = knn_scores(fd[rq], Xk, CFG["D2"]["k"], dev)
        for a in ALPHAS:
            th[f"D1@{a}"][k] = np.quantile(d_own, 1 - a)
            th[f"D2@{a}"][k] = np.quantile(d2_own, 1 - a)
        info["d1_eps"].append(eps)
        info["d1_classes"].append(len(cls))
        info["d2_bank"].append(int(sel.sum()))
        # D3: positives own (h_k); negatives: other residents of k's home cell, labels outside M_k (their own h_j)
        res = [j for j in np.flatnonzero(home == home[k]) if j != k]
        negm = np.isin(own_k, res) & ~main[k][own_y]
        if negm.sum() == 0:
            info["d3_devices_without_negatives"] += 1
            out["D3"][rq] = 0.0
            continue
        X = np.concatenate([Xk, own_fd[negm]])
        t = np.r_[np.ones(len(Xk)), np.zeros(int(negm.sum()))]
        lr = LogisticRegression(C=CFG["D3"]["C"], class_weight="balanced", max_iter=CFG["D3"]["max_iter"]).fit(X, t)
        out["D3"][rq] = 1.0 - lr.predict_proba(np.asarray(fd[rq], np.float64))[:, 1]
        info["d3_heads"] += 1
    # D4: per cell, residents' training samples through the cell's server block
    cell_fe, cell_z, cell_y = own["cell_fe"], own["cell_z"].astype(np.int64), own["cell_y"].astype(np.int64)
    Mrow = D["Mrow"]
    s_sum, s_cnt = np.zeros(N), np.zeros(N)
    info["d4_classes_per_cell"] = []
    for z in np.unique(cell_z):
        selz = cell_z == z
        cls = np.unique(cell_y[selz])
        mu, L, _ = maha_stats(cell_fe[selz], cell_y[selz], cls, CFG["D4"]["eps_factor"])
        info["d4_classes_per_cell"].append(len(cls))
        for j in (0, 1):
            rq = np.flatnonzero(feat["fe_cell"][:, j] == z)
            if len(rq) == 0:
                continue
            dist = maha_dists(feat["fe"][rq, j], mu, L)
            A = Mrow[rq][:, cls]
            minA = np.where(A, dist, np.inf).min(1)
            minB = np.where(~A, dist, np.inf).min(1)
            ok = np.isfinite(minA) & np.isfinite(minB)
            s_sum[rq[ok]] += (minA - minB)[ok]
            s_cnt[rq[ok]] += 1
    out["D4"] = np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), np.nan)
    nm = {}
    for a in ALPHAS:
        nm[f"D1@{a}"] = out["D1"] > th[f"D1@{a}"][k_i]
        nm[f"D2@{a}"] = out["D2"] > th[f"D2@{a}"][k_i]
    nm["D3"] = out["D3"] > 1 - CFG["D3"]["threshold"]
    nm["D4"] = np.isfinite(out["D4"]) & (out["D4"] > 0)
    return out, th, nm, info


# ------------------------------------------------------------------------------------------------ rules
def metrics(D, c):
    """overall, home, away, Main, OOP, OOR as r15_analysis (whole replay)."""
    E, K, g, present, home, kind = D["E"], D["K"], D["g"], D["present"], D["home"], D["kind"]
    cf = c.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ = np.where(present, np.bincount(g, weights=cf, minlength=E * K).reshape(E, K) / D["n_ek"], np.nan)
    per_round = np.nanmean(A_, axis=1)
    r = dict(all=float(per_round.mean()), per_round=per_round,
             home=float(np.mean([A_[e][home[e] == 1].mean() for e in range(E) if (home[e] == 1).any()])),
             away=float(np.mean([A_[e][home[e] == 0].mean() for e in range(E) if (home[e] == 0).any()])))
    for i, kn in enumerate(KINDS):
        selk = kind == i
        ck = np.bincount(g, weights=cf * selk, minlength=E * K).reshape(E, K)
        nk = np.bincount(g, weights=selk, minlength=E * K).reshape(E, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
        vals = [np.nanmean(ak[e]) for e in range(E) if np.isfinite(ak[e]).any()]
        r[kn] = float(np.mean(vals)) if vals else float("nan")
    return r


def dgp_weight(ix, co, Hc, Hs, off):
    """Round 15 DriftGate-P weight over the offloaded requests of the window (r15_analysis lines 352-356)."""
    sd_s, sd_n = m15.window_sum_counted(ix, Hc, off)
    ss_s, _ = m15.window_sum_counted(ix, Hs, off)
    with np.errstate(invalid="ignore", divide="ignore"):
        wP = np.where(sd_n > 0, np.where(sd_s + ss_s > 0, ss_s / (sd_s + ss_s), 0.5), np.nan)
    return m14.ffill_client(co, wP)


def distance_controller(co, score, nm):
    """offload if the score exceeds the median of the device's previous 128 scores (current excluded); fewer than 16
    previous requests -> offload iff non-Main judgement."""
    from numpy.lib.stride_tricks import sliding_window_view
    oc, first = co
    x_all = np.asarray(score, np.float64)[oc]
    nm_o = nm[oc]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(x_all)]
    W, MIN = CFG["online"]["window"], CFG["online"]["min_previous"]
    tau = np.full(len(x_all), np.nan)
    for s0, e0 in zip(starts, ends):
        x = x_all[s0:e0]
        n = len(x)
        for i in range(MIN, min(W, n)):
            tau[s0 + i] = np.quantile(x[:i], 1 - CFG["online"]["beta"])
        if n > W:
            tau[s0 + W:e0] = np.quantile(sliding_window_view(x[:-1], W), 1 - CFG["online"]["beta"], axis=1)
    off_o = np.where(np.isnan(tau), nm_o, x_all > tau)
    off = np.empty(len(x_all), bool)
    off[oc] = off_o
    return off


def analyse(name):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    runs = JR / CFG["runs"]["S1"]["dir"]
    D = m13.load(runs, name)
    q = np.load(runs / f"{name}_evalprobs.npz")
    feat = dict(np.load(runs / f"{name}_features.npz"))
    own = dict(np.load(runs / f"{name}_own.npz"))
    out = dict(name=name, n_req=len(D["y"]), K=D["K"], C=D["C"])
    # ---- start check: bit-identical to the Round 15 replay record?
    st = name.split("_")[1].upper()
    r15name = CFG["r15_replay"]["pattern"][st].format(name.split("_s")[-1])
    b = np.load(JR / CFG["r15_replay"]["dir"] / f"{r15name}_evalprobs.npz")
    out["identical_to_r15"] = {k: bool(np.array_equal(q[k], b[k])) for k in b.files if k != "arrival_rng"}
    lr, lr_corr = r16.load_lr()
    ans, w_auto = r16.reference_answers(D, q, lr, lr_corr, dev)
    y, pc, ps, cp, sp, ent, kind = D["y"], D["pc"], D["ps"], D["cp"], D["sp"], D["ent"], D["kind"]
    psr = m13.corrected(ps, D["Mrow"], D["a_req"], 0.5)
    ce, dg = ans["corrected edge only"], ans["DriftGate"]
    res = {}
    for k_ in REF_KEYS:
        if k_ != "R-ZTW":
            res[k_] = metrics(D, ans[k_] == y)
    vals, soft = r16.ztw(D, ans["B1"] == y, ans["R-PoE"] == y, {"full": np.ones(D["E"], bool)})
    res["R-ZTW"] = metrics(D, soft)
    res["R-ZTW"]["all"] = vals["full"]          # Round 15: the curve at the B0 offloading ratio
    c15 = r16.r15_cache_value(str(JR / CFG["r15_replay"]["dir"]), r15name)
    out["dg_r15_cache"] = c15["res"]["DriftGate"]["full"] if c15 is not None else None
    res["kind oracle"] = metrics(D, np.where(kind == 0, cp, sp) == y)
    t_ref = time.time() - t0
    # ---- detectors
    t1 = time.time()
    sc, th, nm, info = detectors(D, feat, own, dev)
    t_det = time.time() - t1
    del feat
    pos = kind != 0
    hm = D["home"][D["e"], D["k"]] == 1
    au = {}
    for dn, s in (("D1", sc["D1"]), ("D2", sc["D2"]), ("D3", sc["D3"]), ("D4", sc["D4"]), ("entropy", ent.astype(np.float64))):
        fin = np.isfinite(s)
        au[dn] = {sub: (m14.auroc(s[fin & m], pos[fin & m]) if (pos[fin & m].any() and (~pos[fin & m]).any()) else float("nan"))
                  for sub, m in (("all", np.ones(len(s), bool)), ("home", hm), ("away", ~hm))}
        au[dn]["coverage"] = float(fin.mean())
    out["auroc"] = au
    chk = {}
    for v, f in nm.items():
        chk[v] = dict(nonmain_share=float(f.mean()), on_main=float(f[~pos].mean()), on_nonmain=float(f[pos].mean()))
    out["detector_checks"] = dict(chk, thresholds={v: float(np.nanmean(t)) for v, t in th.items()},
                                  **{k_: (float(np.mean(v)) if isinstance(v, list) else v) for k_, v in info.items()})
    # ---- full-offload gate rules
    for v in nm:
        res[f"R-b {v}"] = metrics(D, np.where(nm[v], ce, dg) == y)
    for v in [x for x in nm if x != "D4"]:
        res[f"R-c {v} inf"] = metrics(D, np.where(nm[v], ce, cp) == y)
    for a in ALPHAS:
        tk = th[f"D1@{a}"][D["k"]]
        d = sc["D1"]
        w = np.where(d <= tk, w_auto, w_auto * np.exp(-(d - tk) / tk))
        res[f"R-s D1@{a}"] = metrics(D, m15.mix_answer(w, pc, psr) == y)
    # ---- curves (R-a, R-c per detector; entropy DriftGate-P) and online controllers
    t2 = time.time()
    ix, co = m14.window_index(D), m14.client_order(D)
    Hc = ent.astype(np.float64)
    Hs = m13.server_entropy(ps)
    curves = {}
    dvars = [x for x in nm if x != "D4"]
    for v in dvars + ["entropy"]:
        pts = {"R-a": [], "R-c": [], "srv": []} if v != "entropy" else {"DriftGate-P": [], "srv": []}
        for tau in TAU_GRID:
            sent = ent > tau
            off = sent if v == "entropy" else (nm[v] | sent)
            pts["srv"].append(float(off.mean()))
            if not off.any():
                base = float(metrics(D, cp == y)["all"])
                for kk in pts:
                    if kk != "srv":
                        pts[kk].append(base)
                continue
            wP = dgp_weight(ix, co, Hc, Hs, off)
            mix = m15.mix_answer(wP, pc, psr)
            if v == "entropy":
                pts["DriftGate-P"].append(metrics(D, np.where(off, mix, cp) == y)["all"])
            else:
                pts["R-a"].append(metrics(D, np.where(off, mix, cp) == y)["all"])
                pts["R-c"].append(metrics(D, np.where(off, np.where(nm[v], ce, mix), cp) == y)["all"])
        curves[v] = {kk: np.array(vv) for kk, vv in pts.items()}
    out["curves"] = curves
    online = {}
    off_e = m15.online_offload(co, Hc)
    wPe = dgp_weight(ix, co, Hc, Hs, off_e)
    for nm_, a_ in (("entropy: edge answer", sp), ("entropy: Probability average", ans["B3"]),
                    ("entropy: Logit sum", ans["R-PoE"]), ("entropy: DriftGate-P", m15.mix_answer(wPe, pc, psr))):
        online[nm_] = dict(metrics(D, np.where(off_e, a_, cp) == y), srv=float(off_e.mean()))
    for v in dvars:
        s = sc[v.split("@")[0]]
        off = distance_controller(co, s, nm[v])
        wP = dgp_weight(ix, co, Hc, Hs, off)
        mix = m15.mix_answer(wP, pc, psr)
        online[f"distance {v}: R-a"] = dict(metrics(D, np.where(off, mix, cp) == y), srv=float(off.mean()))
        online[f"distance {v}: R-c"] = dict(metrics(D, np.where(off, np.where(nm[v], ce, mix), cp) == y), srv=float(off.mean()))
    out["online"] = online
    out["res"] = res
    out["seconds"] = dict(references=t_ref, detectors=t_det, curves_online=time.time() - t2, total=time.time() - t0)
    pickle.dump(out, open(CACHE / f"{name}.pkl", "wb"))
    print(json.dumps(dict(name=name, seconds=out["seconds"], identical=all(out["identical_to_r15"].values())), indent=1))


# ------------------------------------------------------------------------------------------------ tables
def ms(v):
    v = np.asarray(v, float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def fmt(v, d=2):
    mu, sd = ms(v)
    return f"{mu:.{d}f} ({sd:.{d}f})"


def load_runs():
    R = {}
    for st, c in CFG["runs"].items():
        for s in c["design_seeds"] + c["eval_seeds"]:
            R[(st, s)] = pickle.load(open(CACHE / f"{c['pattern'].format(s)}.pkl", "rb"))
    return R


def tables():
    R = load_runs()
    ev = {st: [R[(st, s)] for s in c["eval_seeds"]] for st, c in CFG["runs"].items()}
    # ---- start check
    rows, ok = [], True
    for st, c in CFG["runs"].items():
        runs_ = [R[(st, s)] for s in c["design_seeds"] + c["eval_seeds"]]
        for a in runs_:
            v = a["res"]["DriftGate"]["all"] * 100
            rows.append([st, a["name"], f"{v:.4f}", f"{a['dg_r15_cache'] * 100:.4f}" if a["dg_r15_cache"] is not None else "",
                         all(a["identical_to_r15"].values()), ""])
        m = np.mean([a["res"]["DriftGate"]["all"] for a in runs_]) * 100
        m15v = np.mean([a["dg_r15_cache"] for a in runs_]) * 100
        ref = CFG["r15_replay"]["a2_driftgate_pct"][st]
        good = abs(m - m15v) <= CFG["r15_replay"]["tolerance_pp"] and abs(round(m, 2) - ref) <= CFG["r15_replay"]["tolerance_pp"]
        ok &= bool(good)
        rows.append([st, "seed mean", f"{m:.4f}", f"{m15v:.4f} (table A2: {ref:.2f})", "", "within 0.05 pp" if good else "NOT within 0.05 pp"])
    wcsv("R17_T0_start_check.csv", ["setting", "run", "DriftGate replay accuracy pct", "Round 15 value", "replay arrays bit-identical to Round 15",
                                    "check"], rows)
    # ---- design choices (S1 seed 0)
    d0 = R[("S1", CFG["runs"]["S1"]["design_seeds"][0])]
    alpha = {}
    drows = []
    for dn in ("D1", "D2"):
        va = {a: d0["res"][f"R-b {dn}@{a}"]["all"] for a in ALPHAS}
        best = max(va.values())
        alpha[dn] = min(a for a in ALPHAS if va[a] >= best - 1e-12)
        drows += [[f"alpha of {dn}", f"R-b {dn}@{a}", f"{va[a] * 100:.4f}", "chosen" if a == alpha[dn] else ""] for a in ALPHAS]
    det = lambda v: {"D1": f"D1@{alpha['D1']}", "D2": f"D2@{alpha['D2']}", "D3": "D3", "D4": "D4"}[v]
    cand = {c_: (f"R-b {det(c_.split()[1])}" if c_.startswith("R-b") else f"R-c {det(c_.split()[1])} inf") for c_ in CFG["primary_candidates"]}
    cv = {c_: d0["res"][k_]["all"] for c_, k_ in cand.items()}
    best = max(cv.values())
    primary = next(c_ for c_ in CFG["primary_candidates"] if cv[c_] >= best - 1e-12)
    drows += [["primary candidate", c_, f"{cv[c_] * 100:.4f}", "chosen" if c_ == primary else ""] for c_ in CFG["primary_candidates"]]
    wcsv("R17_S0_design.csv", ["choice", "option", "S1 seed 0 replay accuracy pct", ""], drows)
    # ---- AUROC
    arows, aseed = [], []
    for st in ev:
        for dn in ("D1", "D2", "D3", "D4", "entropy"):
            arows.append([st, dn] + [fmt([a["auroc"][dn][sub] for a in ev[st]], 3) for sub in ("all", "home", "away", "coverage")])
        for s in CFG["runs"][st]["design_seeds"] + CFG["runs"][st]["eval_seeds"]:
            a = R[(st, s)]
            for dn in ("D1", "D2", "D3", "D4", "entropy"):
                aseed.append([st, s, "design" if s in CFG["runs"][st]["design_seeds"] else "evaluation", dn] +
                             [f"{a['auroc'][dn][sub]:.4f}" for sub in ("all", "home", "away", "coverage")])
    wcsv("R17_S0_auroc.csv", ["setting", "detector", "AUROC all mean (SD)", "home", "away", "score coverage"], arows)
    wcsv("R17_S0_auroc_per_seed.csv", ["setting", "seed", "role", "detector", "AUROC all", "home", "away", "coverage"], aseed)
    # ---- accuracy
    rules = REF_KEYS + ["kind oracle"] + [f"R-b {det(d)}" for d in ("D1", "D2", "D3", "D4")] + \
        [f"R-c {det(d)} inf" for d in ("D1", "D2", "D3")] + [f"R-s D1@{alpha['D1']}"]
    show = lambda r: SHOW.get(r, r)
    acc, accs, judge = [], [], {}
    for st in ev:
        val = lambda r, sub="all": np.array([a["res"][r][sub] for a in ev[st]]) * 100
        so = max(OTHER, key=lambda r: val(r).mean())
        for r in rules:
            acc.append([st, show(r)] + [fmt(val(r, sub)) for sub in ("all", "home", "away", "Main", "OOP", "OOR")] +
                       [show(so), fmt(val(r) - val(so)), fmt(val(r) - val("DriftGate"))])
            for a, x, xs, xd in zip(ev[st], val(r), val(so), val("DriftGate")):
                accs.append([st, a["name"], show(r), f"{x:.4f}", f"{x - xs:.4f}", f"{x - xd:.4f}"])
        judge[st] = dict(strongest_other=show(so), strongest_other_mean=float(val(so).mean()),
                         primary=primary, primary_rule=cand[primary], primary_mean=float(val(cand[primary]).mean()),
                         margin=float(val(cand[primary]).mean() - val(so).mean()),
                         margins_all_candidates={c_: float(val(k_).mean() - val(so).mean()) for c_, k_ in cand.items()},
                         auroc_D1=float(np.mean([a["auroc"]["D1"]["all"] for a in ev[st]])),
                         auroc_D3=float(np.mean([a["auroc"]["D3"]["all"] for a in ev[st]])))
    wcsv("R17_S0_accuracy.csv", ["setting", "rule", "all mean (SD)", "home", "away", "Main", "OOP", "OOR", "strongest other",
                                 "minus strongest other pp", "minus DriftGate pp"], acc)
    wcsv("R17_S0_accuracy_per_seed.csv", ["setting", "run", "rule", "all pct", "minus strongest other pp", "minus DriftGate pp"], accs)
    # ---- curves
    crows = []
    for st in ev:
        for v, keys in [(det(d), ("R-a", "R-c")) for d in ("D1", "D2", "D3")] + [("entropy", ("DriftGate-P",))]:
            for kk in keys:
                at = np.array([m13.interp(a["curves"][v]["srv"], a["curves"][v][kk], OPTS) for a in ev[st]]) * 100
                inf_srv = [a["curves"][v]["srv"][-1] for a in ev[st]]
                inf_acc = [a["curves"][v][kk][-1] * 100 for a in ev[st]]
                crows.append([st, f"{kk} ({v})"] + [f"{x:.2f}" if np.isfinite(x) else "" for x in
                                                   np.where(np.isfinite(at).all(0), at.mean(0), np.nan)] +
                             [fmt(inf_srv, 3), fmt(inf_acc)])
    wcsv("R17_S0_curves.csv", ["setting", "rule"] + [f"server use {o:.1f}" for o in OPTS] +
         ["tau = inf: server use mean (SD)", "tau = inf: accuracy mean (SD)"], crows)
    # ---- online
    orows = []
    for st in ev:
        keys = list(ev[st][0]["online"])
        keys = [k_ for k_ in keys if k_.startswith("entropy") or any(k_.startswith(f"distance {det(d)}:") for d in ("D1", "D2", "D3"))]
        base = np.array([a["online"]["entropy: DriftGate-P"]["all"] for a in ev[st]]) * 100
        for k_ in keys:
            v = np.array([a["online"][k_]["all"] for a in ev[st]]) * 100
            orows.append([st, k_, fmt(v), fmt([a["online"][k_]["srv"] for a in ev[st]], 3), fmt(v - base)] +
                         [fmt(np.array([a["online"][k_][sub] for a in ev[st]]) * 100) for sub in ("home", "away", "Main", "OOP", "OOR")])
    wcsv("R17_S0_online.csv", ["setting", "rule", "all mean (SD)", "realized offloading", "minus entropy DriftGate-P pp",
                               "home", "away", "Main", "OOP", "OOR"], orows)
    # ---- detector checks and cost
    krow, cost = [], []
    for st in ev:
        for v in [det(d) for d in ("D1", "D2", "D3", "D4")]:
            c_ = [a["detector_checks"][v] for a in ev[st]]
            krow.append([st, v, fmt([x["nonmain_share"] for x in c_], 3), fmt([x["on_main"] for x in c_], 3),
                         fmt([x["on_nonmain"] for x in c_], 3),
                         fmt([a["detector_checks"]["thresholds"][v] for a in ev[st]], 4) if v in ev[st][0]["detector_checks"]["thresholds"] else ""])
        dc = [a["detector_checks"] for a in ev[st]]
        krow.append([st, "D3 heads / devices without negatives", fmt([x["d3_heads"] for x in dc], 1),
                     fmt([x["d3_devices_without_negatives"] for x in dc], 1), "", ""])
        krow.append([st, "D4 score coverage", fmt([a["auroc"]["D4"]["coverage"] for a in ev[st]], 3), "", "", ""])
        nM = np.mean([x["d1_classes"] for x in dc])
        nk = np.mean([x["d2_bank"] for x in dc])
        cz = np.mean([np.mean(x["d4_classes_per_cell"]) for x in dc])
        d1_flop = 128 * 63 + 2 * 128 * 128 + nM * 3 * 128
        cost += [[st, "D1", "device", f"{d1_flop:.0f}", f"{128 * 128 * 4 + nM * 128 * 4 + 4:.0f}", "",
                  "pooling 128x64; whitening 128x128 matrix-vector; |M_k| distances of 128 values (whitened means stored)"],
                 [st, "D2", "device", f"{128 * 63 + 3 * 128 + 2 * 128 * nk + nk:.0f}", f"{nk * 128 * 4 + 4:.0f}", "",
                  f"pooling; normalisation; {nk:.0f} dot products of 128 values; selection of the 10th largest (float32 bank)"],
                 [st, "D3", "device", f"{128 * 63 + 2 * 128 + 1:.0f}", f"{129 * 4}", f"{129 * 4}",
                  "pooling; one dot product of 128 values plus bias; 129 float32 weights sent per training round"],
                 [st, "D4", "edge (per cell)", f"{2 * 128 * 128 + cz * 3 * 128:.0f}", f"{128 * 128 * 4 + cz * 128 * 4:.0f}", "",
                  f"per cell of the request (f_e already computed by the edge); {cz:.1f} class prototypes per cell"]]
    wcsv("R17_S0_detector_checks.csv", ["setting", "detector", "non-Main share of requests", "share on Main requests (false alarms)",
                                        "share on non-Main requests (detections)", "mean threshold"], krow)
    wcsv("R17_S0_cost.csv", ["setting", "detector", "where", "FLOP per request", "stored bytes", "downlink bytes per training round", "note"], cost)
    # ---- decision
    G, Hd = CFG["decision"]["auroc_go"], CFG["decision"]["auroc_hold"]
    marg = CFG["decision"]["margin_pp"]
    A = {st: max(judge[st]["auroc_D1"], judge[st]["auroc_D3"]) for st in judge}
    B = {st: judge[st]["margin"] for st in judge}
    if all(A[st] >= G and B[st] >= marg for st in judge):
        decision = "GO"
    elif all(B[st] >= marg and A[st] >= Hd for st in judge):
        decision = "HOLD"
    else:
        decision = "NO_GO"
    run_times = {a["name"]: a["seconds"] for a in R.values()}
    ext = {a["name"]: json.loads((JR / CFG["runs"]["S1"]["dir"] / f"{a['name']}_extract.json").read_text()) for a in R.values()}
    out = dict(round=17, stage=0, decision=decision, start_check_passed=bool(ok), alpha=alpha, primary_candidate=primary,
               primary_rule=cand[primary], per_setting={st: dict(judge[st], A=A[st], B=B[st]) for st in judge},
               other_candidates_with_margin={st: [c_ for c_, m_ in judge[st]["margins_all_candidates"].items() if m_ >= marg] for st in judge},
               analysis_seconds=run_times, replay_seconds={k_: v["replay_seconds"] for k_, v in ext.items()},
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE / "decision_stage0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k_: out[k_] for k_ in ("decision", "start_check_passed", "alpha", "primary_candidate")}, indent=1))
    print(json.dumps(out["per_setting"], indent=1))


def main():
    if sys.argv[1] == "run":
        analyse(sys.argv[2])
    elif sys.argv[1] == "tables":
        tables()
    else:
        raise SystemExit(sys.argv[1])


if __name__ == "__main__":
    main()
