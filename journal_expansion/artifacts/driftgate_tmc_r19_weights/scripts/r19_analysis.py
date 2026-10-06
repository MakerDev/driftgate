"""Round 19 stage 0: fixed device weights against the DriftGate weight (definitions: R19_plan.md).

Per run: correction r = 0.5 for every method; DriftGate (beta 1: Round 13b F-auto weight; beta 0.5: Round 15 DriftGate-P
weight on the controller's offload mask), 21 fixed weights w = 0, 0.05, ..., 1, the label-free weight fixed after the first
128 offloaded requests of a device, and the device constant (mean DriftGate weight; post hoc). Stored per device-round
correct counts so that the oracles, windows, transition device-rounds and request-weighted accuracies are computed from
the same numbers. The Round 15-18 helpers are imported unchanged.

Subcommands (repository root):
  run SETTING SEED  -> cache/<run>.pkl
  tables            -> tables/R19_*.csv, figures/R19_*.pdf/png, tables/R19_state.json
"""
import csv
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
_spec = importlib.util.spec_from_file_location("r18", ART / "driftgate_tmc_r18_compare" / "scripts" / "r18_analysis.py")
r18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r18)
r17, r16, m15, m14, m13 = r18.r17, r18.r16, r18.m15, r18.m14, r18.m13
TAB, FIG, CACHE = HERE / "tables", HERE / "figures", HERE / "cache"
for d_ in (TAB, FIG, CACHE):
    d_.mkdir(parents=True, exist_ok=True)

R12, R13, R15, R10 = (JR / "runs" / x for x in ("phaseT12_fusion", "phaseT13b_arch", "phaseT15_replay", "phaseT10_prior"))
SETTINGS = {   # name -> (runs dir, pattern, seeds, period, model)
    "S1": (R12, "r12_s1_fixed040_s{}", range(5), "day1", "CNN"), "S2": (R12, "r12_s2_fixed040_s{}", range(3), "day1", "CNN"),
    "S1-fast": (R12, "r12_s1fast_fixed040_s{}", range(3), "day1", "CNN"),
    "partial participation": (R12, "r12_s4part05_fixed040_s{}", range(3), "day1", "CNN"),
    "stepwise change": (R12, "r12_t1_A_fx40_s{}", range(5), "day1", "CNN"),
    "random mobility": (R12, "r12_t1_mob_fx40_s{}", range(3), "day1", "CNN"),
    "K=200": (R12, "r12_s3k200_fixed040_s{}", range(3), "day1", "CNN"), "K=500": (R12, "r12_s3k500_fixed040_s{}", range(3), "day1", "CNN"),
    "CIFAR-100": (R13, "r13b_c100_s{}", range(3), "day1", "CNN (100 classes)"),
    "ResNet-18": (R12, "r12_e2_res_fx40_A_s{}", range(3), "day1", "ResNet-18"),
    "ResNet-20 shallow": (R13, "r13b_res20_shallow_s{}", range(3), "day1", "ResNet-20 shallow"),
    "ResNet-20 middle": (R13, "r13b_res20_middle_s{}", range(3), "day1", "ResNet-20 middle"),
    "VGG-11 shallow": (R13, "r13b_vgg11_shallow_s{}", range(3), "day1", "VGG-11 shallow"),
    "VGG-11 middle": (R13, "r13b_vgg11_middle_s{}", range(3), "day1", "VGG-11 middle"),
    "S1 replay": (R15, "r15_s1_replay_s{}", range(5), "replay", "CNN"), "S2 replay": (R15, "r15_s2_replay_s{}", range(3), "replay", "CNN"),
    "S1 development": (R10, "r10_T40_s{}", [5, 6, 7], "day1", "CNN"),
}
EVAL = [s for s in SETTINGS if s != "S1 development"]
W_GRID = [round(0.05 * i, 2) for i in range(21)]
BETAS = [1.0, 0.5]
N_INIT = 128
WINDOWS = {"full": None, "round>30": 30, "round>50": 50}
BINS = [tuple(b) for b in m14.BINS]
TOL = 0.1   # pp, descriptive flatness threshold (R19_plan.md)


def mix_vec(w, pd_, pe_, chunk=1 << 21):
    """argmax(w p_d + (1 - w) p_e) with a per-request w, float32 as r15_analysis.mix_answer."""
    w = np.asarray(w, np.float32)
    out = np.empty(len(pd_), np.int64)
    for s in range(0, len(pd_), chunk):
        ww = w[s:s + chunk, None]
        out[s:s + chunk] = (ww * pd_[s:s + chunk] + (1 - ww) * pe_[s:s + chunk]).argmax(1)
    return out


def run_name(st, s):
    return SETTINGS[st][1].format(s)


def tie_order():
    """grid indices ordered by |w - 0.5|, then w (tie rule for argmax choices)."""
    return sorted(range(len(W_GRID)), key=lambda j: (round(abs(W_GRID[j] - 0.5), 9), W_GRID[j]))


TIE = tie_order()


def argbest(scores, axis=0):
    """index of the maximum along axis, ties (within 1e-12) -> closer to 0.5, then smaller w."""
    s = np.moveaxis(np.asarray(scores, np.float64), axis, 0)[TIE]
    best = np.nanmax(s, axis=0)
    j = np.argmax(s >= best - 1e-12, axis=0)
    return np.asarray(TIE)[j]


# ------------------------------------------------------------------------------------------------ per run
def analyse(st, seed):
    t0 = time.time()
    runs, _, _, period, _ = SETTINGS[st]
    name = run_name(st, seed)
    D = m13.load(runs, name)
    pc, ps, Mrow, a, y, cp, ent = (D[k] for k in ("pc", "ps", "Mrow", "a_req", "y", "cp", "ent"))
    N, E, K, g = len(y), D["E"], D["K"], D["g"]
    er = D["er"]
    psr = m13.corrected(ps, Mrow, a, 0.5)
    w_auto, Hs = r16.driftgate_weight(D)
    Hc = ent.astype(np.float64)
    ix, co = m14.window_index(D), m14.client_order(D)
    oc, first = co
    cells = np.sort(m15.cells_matrix(runs, name, K, er), axis=2)
    trans = np.zeros((E, K), bool)
    trans[1:] = np.any(cells[1:] != cells[:-1], axis=2)
    corr_dev = cp == y
    C = [r16.mix_scalar(w, pc, psr) == y for w in W_GRID]
    wts = m13.weights(D, np.ones(E, bool))
    out = dict(setting=st, seed=seed, name=name, period=period, er=er, E=E, K=K, n_ek=D["n_ek"], present=D["present"],
               home=D["home"], trans=trans, start_min=300.0 + 6.0 * (er - 1), n_req=N, beta={})
    for beta in BETAS:
        off = r18.controller(co, Hc, beta)
        wdg = w_auto if beta == 1.0 else r17.dgp_weight(ix, co, Hc, Hs, off)
        # label-free weight fixed after the first N_INIT offloaded requests of the device (client order)
        o_off = off[oc]
        dev_id = np.cumsum(first) - 1
        cum = np.cumsum(o_off)
        start_cum = np.r_[0, cum[np.flatnonzero(first)[1:] - 1]]
        n_before = cum - o_off - start_cum[dev_id]                       # offloaded requests of the device before i
        k_of = D["k"][oc]
        m_first = o_off & (n_before < N_INIT)
        nd = int(first.sum())
        cnt_f = np.bincount(dev_id, weights=m_first, minlength=nd)
        shc = np.bincount(dev_id, weights=np.where(m_first, Hc[oc], 0.0), minlength=nd)
        shs = np.bincount(dev_id, weights=np.where(m_first, Hs[oc], 0.0), minlength=nd)
        wfix = np.full(K, np.nan)
        kk = k_of[np.flatnonzero(first)]
        with np.errstate(invalid="ignore", divide="ignore"):
            wfix[kk] = np.where(cnt_f == N_INIT, shs / (shc + shs), np.nan)
        has_req = np.zeros(K, bool)
        has_req[kk] = True
        use_fix = np.zeros(N, bool)
        use_fix[oc] = (n_before >= N_INIT) & np.isfinite(wfix[k_of])
        w_init = np.where(use_fix, wfix[D["k"]], wdg)
        # device constant (post hoc): mean DriftGate weight over the device's offloaded requests
        s_w = np.bincount(D["k"], weights=np.where(off, wdg, 0.0), minlength=K)
        n_w = np.bincount(D["k"], weights=off, minlength=K)
        wconst = np.where(n_w > 0, s_w / np.maximum(n_w, 1), 0.5)
        corr = {"DriftGate": np.where(off, mix_vec(wdg, pc, psr) == y, corr_dev),
                "initial fixed": np.where(off, mix_vec(w_init, pc, psr) == y, corr_dev),
                "device constant": np.where(off, mix_vec(wconst[D["k"]], pc, psr) == y, corr_dev)}
        for j, w in enumerate(W_GRID):
            corr[f"w={w:.2f}"] = np.where(off, C[j], corr_dev)
        b = dict(cnt={k_: np.bincount(g, weights=v, minlength=E * K).reshape(E, K) for k_, v in corr.items()},
                 splits={k_: r18.pr_metrics(D, v) for k_, v in corr.items()},
                 srv=float(off.mean()), wdg_mean=float(wdg[off].mean()), wdg_sd=float(wdg[off].std()),
                 wdg_round=np.array([wdg[off & (D["e"] == e)].mean() if (off & (D["e"] == e)).any() else np.nan for e in range(E)]),
                 wfix=wfix[has_req], share_not_fixed=float(np.isnan(wfix[has_req]).mean()), wconst=wconst,
                 init_fixed_share_requests=float(use_fix[off].mean()))
        dg = corr["DriftGate"]
        rh = {}
        for wn, lo in WINDOWS.items():
            m = (er > lo)[D["e"]] if lo else np.ones(N, bool)
            ww = wts * m
            ww = ww / ww.sum()
            for k_ in [x for x in corr if x != "DriftGate"]:
                o = corr[k_]
                rh[(wn, k_)] = (float((ww * (dg & ~o)).sum()), float((ww * (~dg & o)).sum()))
        b["rescue_harm"] = rh
        out["beta"][beta] = b
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(CACHE / f"{name}.pkl", "wb"))
    print(json.dumps(dict(name=name, seconds=round(out["seconds"], 1),
                          share_not_fixed={b_: out["beta"][b_]["share_not_fixed"] for b_ in BETAS})))


# ------------------------------------------------------------------------------------------------ tables
def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


fmt, ms = r18.fmt, r18.ms


def acc_matrix(a, beta, key):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a["present"], a["beta"][beta]["cnt"][key] / a["n_ek"], np.nan)


def metric(A, rmask=None, dmask=None):
    """paper metric: per round mean over devices (optionally a device mask [E, K]), then mean over rounds."""
    import warnings
    M = A if dmask is None else np.where(dmask, A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = np.nanmean(M, axis=1)
        if rmask is not None:
            pr = pr[rmask]
        return float(np.nanmean(pr)) if np.isfinite(pr).any() else float("nan")


def req_acc(a, beta, key, rmask):
    c = a["beta"][beta]["cnt"][key][rmask].sum()
    n = a["n_ek"][rmask].sum()
    return float(c / n), float(n - c)


def wmask(a, wn):
    if wn == "full":
        return np.ones(a["E"], bool)
    return a["er"] > WINDOWS[wn]


def oracle_values(a, beta):
    """common fixed oracle, device fixed oracle, device x time-bin oracle: per-device-round accuracy matrices."""
    G = np.stack([acc_matrix(a, beta, f"w={w:.2f}") for w in W_GRID])          # [21, E, K]
    npres = a["present"].sum(1)
    contrib = np.where(a["present"][None], G, 0.0) / np.maximum(npres, 1)[None, :, None]
    common = argbest([metric(G[j]) for j in range(len(W_GRID))])
    dev_j = argbest(contrib.sum(1), axis=0)                                     # [K]
    Adev = np.take_along_axis(G, dev_j[None, None, :].repeat(G.shape[1], 1), 0)[0]
    Abin = np.full_like(G[0], np.nan)
    sizes = []
    for b, lo, hi in BINS:
        rm = (a["start_min"] >= lo) & (a["start_min"] < hi)
        if not rm.any():
            continue
        jb = argbest(contrib[:, rm].sum(1), axis=0)
        Abin[rm] = np.take_along_axis(G[:, rm], jb[None, None, :].repeat(rm.sum(), 1), 0)[0]
        sizes.append(np.where(a["present"][rm], a["n_ek"][rm], 0).sum(0)[a["present"][rm].any(0)])
    return dict(common=G[common], common_w=W_GRID[common], device=Adev, device_w=np.asarray(W_GRID)[dev_j], bin=Abin,
                bin_requests=float(np.mean(np.concatenate(sizes))) if sizes else float("nan"))


def tables():
    R = {st: [pickle.load(open(CACHE / f"{run_name(st, s)}.pkl", "rb")) for s in SETTINGS[st][2]] for st in SETTINGS}
    keys_grid = [f"w={w:.2f}" for w in W_GRID]
    # ---- development weight (S1 development seeds, beta 1, full window)
    dev_curve = np.array([[metric(acc_matrix(a, 1.0, k_)) for k_ in keys_grid] for a in R["S1 development"]]) * 100
    j_dev = int(argbest(dev_curve.mean(0)))
    w_dev = W_GRID[j_dev]
    k_dev = keys_grid[j_dev]
    # ---- curves (seed means) and near-optimal ranges
    curve_rows, near, best_grid = [], {}, {}
    for st in EVAL:
        for beta in BETAS:
            for wn in (["full", "round>30", "round>50"] if SETTINGS[st][3] == "day1" else ["full"]):
                vals = np.array([[metric(acc_matrix(a, beta, k_), wmask(a, wn)) for k_ in keys_grid] for a in R[st]]) * 100
                mean = vals.mean(0)
                bj = int(argbest(mean))
                ok = mean >= mean.max() - TOL
                near[(st, beta, wn)] = [W_GRID[j] for j in range(len(W_GRID)) if ok[j]]
                best_grid[(st, beta, wn)] = (W_GRID[bj], mean.max())
                dgv = np.array([metric(acc_matrix(a, beta, "DriftGate"), wmask(a, wn)) for a in R[st]]) * 100
                inv = np.array([metric(acc_matrix(a, beta, "initial fixed"), wmask(a, wn)) for a in R[st]]) * 100
                curve_rows.append([st, SETTINGS[st][4], beta, wn] + [f"{v:.3f}" for v in mean] +
                                  [f"{W_GRID[bj]:.2f}", f"{mean.max():.3f}", f"{min(near[(st, beta, wn)]):.2f}-{max(near[(st, beta, wn)]):.2f}",
                                   "yes" if len(near[(st, beta, wn)]) == (round((max(near[(st, beta, wn)]) - min(near[(st, beta, wn)])) / 0.05) + 1) else "no",
                                   f"{dgv.mean():.3f}", f"{inv.mean():.3f}",
                                   fmt([a["beta"][beta]["wdg_mean"] for a in R[st]], 3), fmt([a["beta"][beta]["wdg_sd"] for a in R[st]], 3),
                                   fmt([np.nanmean(a["beta"][beta]["wfix"]) for a in R[st]], 3),
                                   fmt([np.nanstd(a["beta"][beta]["wfix"]) for a in R[st]], 3),
                                   fmt([a["beta"][beta]["share_not_fixed"] for a in R[st]], 3)])
    wcsv("R19_weight_curves.csv", ["setting", "model", "beta", "window"] + [f"acc w={w:.2f}" for w in W_GRID] +
         ["best grid w", "best grid accuracy", "w within 0.1 pp of best (min-max)", "range contiguous", "DriftGate accuracy",
          "initial-fixed accuracy", "DriftGate w mean", "DriftGate w SD (requests)", "initial-fixed w mean over devices",
          "initial-fixed w SD over devices", "share of devices never fixed"], curve_rows)
    seed_rows = []
    for st in EVAL:
        for a in R[st]:
            for beta in BETAS:
                for wn in (["full", "round>30", "round>50"] if SETTINGS[st][3] == "day1" else ["full"]):
                    seed_rows.append([st, a["name"], beta, wn] + [f"{metric(acc_matrix(a, beta, k_), wmask(a, wn)) * 100:.4f}"
                                                                   for k_ in keys_grid + ["DriftGate", "initial fixed", "device constant"]])
    wcsv("R19_weight_curves_per_seed.csv", ["setting", "run", "beta", "window"] + [f"w={w:.2f}" for w in W_GRID] +
         ["DriftGate", "initial fixed", "device constant"], seed_rows)
    # ---- Table 1: deployable comparisons; Table 2: oracles
    t1, t2, state_in = [], [], {}
    for st in EVAL:
        for beta in BETAS:
            ors = [oracle_values(a, beta) for a in R[st]]
            for wn in (["full", "round>30", "round>50", "transition", "non-transition"] if SETTINGS[st][3] == "day1"
                       else ["full", "transition", "non-transition"]):
                def val(a, key, o=None):
                    if wn in ("transition", "non-transition"):
                        dm = a["trans"] if wn == "transition" else ~a["trans"]
                        A = acc_matrix(a, beta, key) if o is None else o
                        return metric(A, None, dm & a["present"])
                    A = acc_matrix(a, beta, key) if o is None else o
                    return metric(A, wmask(a, wn))
                dg = np.array([val(a, "DriftGate") for a in R[st]]) * 100
                comps = [("development w (S1 dev seeds) = " + f"{w_dev:.2f}", k_dev), ("Round 11 development w = 0.20", "w=0.20"),
                         ("model-specific development w", k_dev if SETTINGS[st][4] == "CNN" else None),
                         ("initial fixed (label-free)", "initial fixed"), ("corrected edge only (w = 0)", "w=0.00"),
                         ("fixed w = 0.50", "w=0.50")]
                row = [st, SETTINGS[st][4], beta, wn, fmt(dg)]
                for lab, k_ in comps:
                    if k_ is None:
                        row += ["no development data", ""]
                        continue
                    v = np.array([val(a, k_) for a in R[st]]) * 100
                    row += [fmt(v), fmt(dg - v, 3)]
                if wn in WINDOWS:
                    rh_i = [a["beta"][beta]["rescue_harm"][(wn, "initial fixed")] for a in R[st]]
                    rh_d = [a["beta"][beta]["rescue_harm"][(wn, k_dev)] for a in R[st]]
                    row += [f"{fmt([x[0] * 100 for x in rh_i], 3)} / {fmt([x[1] * 100 for x in rh_i], 3)}",
                            f"{fmt([x[0] * 100 for x in rh_d], 3)} / {fmt([x[1] * 100 for x in rh_d], 3)}"]
                    rq = [req_acc(a, beta, "DriftGate", wmask(a, wn)) for a in R[st]]
                    rqi = [req_acc(a, beta, "initial fixed", wmask(a, wn)) for a in R[st]]
                    row += [fmt([x[0] * 100 for x in rq]), fmt([(x[0] - z[0]) * 100 for x, z in zip(rq, rqi)], 3),
                            fmt([x[1] for x in rq], 0), fmt([x[1] - z[1] for x, z in zip(rq, rqi)], 0)]
                else:
                    row += ["", "", "", "", "", ""]
                t1.append(row)
                # oracles
                oc_ = np.array([val(a, None, o["common"]) for a, o in zip(R[st], ors)]) * 100
                od_ = np.array([val(a, None, o["device"]) for a, o in zip(R[st], ors)]) * 100
                ob_ = np.array([val(a, None, o["bin"]) for a, o in zip(R[st], ors)]) * 100
                dc_ = np.array([val(a, "device constant") for a in R[st]]) * 100
                t2.append([st, SETTINGS[st][4], beta, wn, fmt(dg), fmt(oc_ - dg, 3), fmt(od_ - dg, 3), fmt(ob_ - dg, 3), fmt(dc_ - dg, 3),
                           fmt(od_ - oc_, 3), fmt(ob_ - od_, 3), fmt([o["common_w"] for o in ors], 2),
                           fmt([np.std(o["device_w"]) for o in ors], 3), fmt([o["bin_requests"] for o in ors], 0)])
                state_in[(st, beta, wn)] = dict(dI=float((dg - np.array([val(a, "initial fixed") for a in R[st]]) * 100).mean()),
                                                H_time=float((ob_ - od_).mean()), dg=float(dg.mean()))
    hdr1 = ["setting", "model", "beta", "window", "DriftGate"]
    for lab in ["development w (S1 dev seeds)", "Round 11 development w = 0.20", "model-specific development w", "initial fixed (label-free)",
                "corrected edge only (w = 0)", "fixed w = 0.50"]:
        hdr1 += [lab, f"DriftGate minus {lab} pp"]
    hdr1 += ["DriftGate vs initial fixed: rescued / harmed pp", "DriftGate vs development w: rescued / harmed pp",
             "request-weighted DriftGate accuracy", "request-weighted DriftGate minus initial fixed pp", "DriftGate errors (requests)",
             "DriftGate minus initial fixed errors (requests)"]
    wcsv("R19_table1_deployable.csv", hdr1, t1)
    wcsv("R19_table2_oracles.csv", ["setting", "model", "beta", "window", "DriftGate", "common fixed oracle minus DriftGate pp",
                                    "device fixed oracle minus DriftGate pp", "device x time-bin oracle minus DriftGate pp",
                                    "device constant (mean DriftGate w) minus DriftGate pp", "device oracle minus common oracle pp",
                                    "time-bin oracle minus device oracle pp", "common oracle w", "SD of device-oracle w across devices",
                                    "requests per device and time bin"], t2)
    # ---- leave-one-seed-out model-specific w (diagnostic)
    lo_rows = []
    for st in EVAL:
        for beta in BETAS:
            vals = np.array([[metric(acc_matrix(a, beta, k_)) for k_ in keys_grid] for a in R[st]]) * 100
            dg = np.array([metric(acc_matrix(a, beta, "DriftGate")) for a in R[st]]) * 100
            pick, v = [], []
            for i in range(len(R[st])):
                j = int(argbest(np.delete(vals, i, 0).mean(0)))
                pick.append(W_GRID[j])
                v.append(vals[i, j])
            v = np.array(v)
            lo_rows.append([st, beta, " ".join(f"{p:.2f}" for p in pick), fmt(v), fmt(dg - v, 3)])
    wcsv("R19_leave_one_seed_out.csv", ["setting", "beta", "w chosen on the other seeds (per held-out seed)", "accuracy (held-out seeds)",
                                        "DriftGate minus it pp"], lo_rows)
    # ---- splits for the main comparisons (beta 1 and 0.5, full window)
    sp_rows = []
    for st in EVAL:
        for beta in BETAS:
            for k_, lab in (("DriftGate", "DriftGate"), ("initial fixed", "initial fixed"), (k_dev, f"development w {w_dev:.2f}"),
                            ("w=0.00", "corrected edge only")):
                row = [st, beta, lab]
                for sp_ in r18.SPLITS:
                    row.append(fmt([np.nanmean(a["beta"][beta]["splits"][k_][sp_]) * 100 for a in R[st]]))
                sp_rows.append(row)
    wcsv("R19_splits.csv", ["setting", "beta", "method"] + r18.SPLITS, sp_rows)
    # ---- decomposition of the full difference into round <= 30 and round > 30
    dec = []
    for st in [s for s in EVAL if SETTINGS[s][3] == "day1"]:
        for beta in BETAS:
            for lab, k_ in (("initial fixed", "initial fixed"), (f"development w {w_dev:.2f}", k_dev), ("corrected edge only", "w=0.00")):
                parts = []
                for a in R[st]:
                    er = a["er"]
                    d_round = np.nanmean(acc_matrix(a, beta, "DriftGate"), 1) - np.nanmean(acc_matrix(a, beta, k_), 1)
                    parts.append([d_round.mean() * 100, d_round[er <= 30].sum() / len(er) * 100, d_round[er > 30].sum() / len(er) * 100])
                parts = np.array(parts)
                dec.append([st, beta, lab, fmt(parts[:, 0], 3), fmt(parts[:, 1], 3), fmt(parts[:, 2], 3)])
    wcsv("R19_decomposition.csv", ["setting", "beta", "DriftGate minus", "full pp", "contribution of round <= 30 pp", "contribution of round > 30 pp"], dec)
    # ---- time bins: near-optimal w range and DriftGate w (figure 2 data)
    tb = []
    for st in ["S1", "S2", "S1 replay", "S2 replay", "ResNet-18", "ResNet-20 shallow", "ResNet-20 middle", "VGG-11 shallow", "VGG-11 middle"]:
        for beta in BETAS:
            for b, lo, hi in BINS:
                rm = (R[st][0]["start_min"] >= lo) & (R[st][0]["start_min"] < hi)
                if not rm.any():
                    continue
                vals = np.array([[metric(acc_matrix(a, beta, k_), rm) for k_ in keys_grid] for a in R[st]]) * 100
                mean = vals.mean(0)
                ok = [W_GRID[j] for j in range(len(W_GRID)) if mean[j] >= mean.max() - TOL]
                dg = np.array([metric(acc_matrix(a, beta, "DriftGate"), rm) for a in R[st]]) * 100
                wdg = [np.nanmean(a["beta"][beta]["wdg_round"][rm]) for a in R[st]]
                tb.append([st, beta, b, int(rm.sum()), f"{W_GRID[int(argbest(mean))]:.2f}", f"{min(ok):.2f}", f"{max(ok):.2f}",
                           fmt(wdg, 3), f"{mean.max():.3f}", fmt(dg - vals[:, int(argbest(mean))], 3)])
    wcsv("R19_time_bins.csv", ["setting", "beta", "time bin", "evaluation rounds", "best fixed w", "w within 0.1 pp: min", "max",
                               "DriftGate w mean", "best fixed accuracy", "DriftGate minus best fixed in the bin pp"], tb)
    # ---- state
    st_out = state(R, near, best_grid, state_in, w_dev)
    st_out["development_w"] = w_dev
    st_out["development_curve"] = {f"{w:.2f}": float(v) for w, v in zip(W_GRID, dev_curve.mean(0))}
    (TAB / "R19_state.json").write_text(json.dumps(st_out, indent=1, default=str))
    figures(R, near, w_dev)
    print(json.dumps({k: v for k, v in st_out.items() if k in ("A", "B", "C", "development_w")}, indent=1, default=str))


def state(R, near, best_grid, si, w_dev):
    out = {}
    for beta in BETAS:
        rows = {}
        for st in EVAL:
            bw, bv = best_grid[(st, beta, "full")]
            dI = si[(st, beta, "full")]["dI"]
            dG = si[(st, beta, "full")]["dg"] - bv
            dev_v = np.mean([metric(acc_matrix(a, beta, f"w={w_dev:.2f}")) for a in R[st]]) * 100
            init_v = si[(st, beta, "full")]["dg"] - dI
            rows[st] = dict(dI=dI, dG=dG, L_dev=bv - dev_v, H_time=si[(st, beta, "full")]["H_time"],
                            dI_transition=si[(st, beta, "transition")]["dI"], init_minus_best=init_v - bv,
                            near=near[(st, beta, "full")])
        common = set(W_GRID)
        for st in EVAL:
            common &= set(near[(st, beta, "full")])
        A = all(abs(r["dI"]) <= TOL and r["dG"] <= TOL for r in rows.values())
        frag = [st for st, r in rows.items() if r["L_dev"] > TOL]
        B_i = (len(common) == 0) or bool(frag)
        chk = frag if frag else list(rows)
        B_ii = all(max(rows[st]["init_minus_best"], rows[st]["dG"]) >= -TOL for st in chk)
        Cset = [st for st, r in rows.items() if r["H_time"] > TOL and r["dI"] > TOL and r["dI_transition"] > TOL]
        out[f"beta {beta}"] = dict(A=A, B=bool(B_i and B_ii), B_i=B_i, B_ii=B_ii, fragile_settings=frag,
                                   common_near_optimal_w=sorted(common), C=bool(Cset), C_settings=Cset, per_setting=rows)
    out["A"] = {b: out[f"beta {b}"]["A"] for b in BETAS}
    out["B"] = {b: out[f"beta {b}"]["B"] for b in BETAS}
    out["C"] = {b: out[f"beta {b}"]["C"] for b in BETAS}
    return out


def figures(R, near, w_dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [f"w={w:.2f}" for w in W_GRID]
    for tag, sets in (("main", ["S1", "S2", "ResNet-18", "S1 replay", "S2 replay"]),
                      ("models", ["S1", "ResNet-20 shallow", "ResNet-20 middle", "VGG-11 shallow", "VGG-11 middle"])):
        fig, axs = plt.subplots(2, len(sets), figsize=(3.0 * len(sets), 5.2), sharex=True)
        for c, st in enumerate(sets):
            for r, beta in enumerate(BETAS):
                ax = axs[r, c]
                vals = np.array([[metric(acc_matrix(a, beta, k_)) for k_ in keys] for a in R[st]]) * 100
                ax.plot(W_GRID, vals.mean(0), color="#2a6fbb", lw=1.6, label="fixed w (seed mean)")
                ax.fill_between(W_GRID, vals.min(0), vals.max(0), color="#2a6fbb", alpha=0.15, lw=0)
                dg = np.mean([metric(acc_matrix(a, beta, "DriftGate")) for a in R[st]]) * 100
                inv = np.mean([metric(acc_matrix(a, beta, "initial fixed")) for a in R[st]]) * 100
                ax.axhline(dg, color="#c0392b", lw=1.2, label="DriftGate")
                ax.axhline(inv, color="#7f8c8d", lw=1.0, ls="--", label="initial fixed")
                ax.axvline(w_dev, color="#555555", lw=0.8, ls=":", label=f"development w {w_dev:.2f}")
                nr = near[(st, beta, "full")]
                ax.axvspan(min(nr) - 0.025, max(nr) + 0.025, color="#f1c40f", alpha=0.25, lw=0)
                ax.set_title(f"{st}, beta {beta:g}", fontsize=8)
                ax.tick_params(labelsize=7)
                if c == 0:
                    ax.set_ylabel("accuracy (%)", fontsize=8)
                if r == 1:
                    ax.set_xlabel("device weight w", fontsize=8)
        axs[0, 0].legend(fontsize=6, loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"R19_fig1_accuracy_vs_w_{tag}.{ext}", dpi=200)
        plt.close(fig)
    # figure 2: time bins
    rows = list(csv.DictReader(open(TAB / "R19_time_bins.csv")))
    sets = ["S1", "S2", "S1 replay", "S2 replay", "ResNet-18"]
    fig, axs = plt.subplots(2, len(sets), figsize=(3.0 * len(sets), 5.0), sharex=True)
    for c, st in enumerate(sets):
        rr = [r for r in rows if r["setting"] == st and r["beta"] == "1.0"]
        x = np.arange(len(rr))
        lo = [float(r["w within 0.1 pp: min"]) for r in rr]
        hi = [float(r["max"]) for r in rr]
        wd = [float(r["DriftGate w mean"].split()[0]) for r in rr]
        axs[0, c].bar(x, np.array(hi) - np.array(lo) + 0.02, bottom=np.array(lo) - 0.01, color="#f1c40f", alpha=0.6, label="w within 0.1 pp")
        axs[0, c].plot(x, wd, "o-", color="#c0392b", ms=3, lw=1, label="DriftGate w")
        axs[0, c].set_ylim(-0.05, 1.05)
        axs[0, c].set_title(f"{st}, beta 1", fontsize=8)
        d = [float(r["DriftGate minus best fixed in the bin pp"].split()[0]) for r in rr]
        axs[1, c].bar(x, d, color=["#2a6fbb" if v >= 0 else "#c0392b" for v in d])
        axs[1, c].axhline(0, color="black", lw=0.6)
        axs[1, c].set_xticks(x)
        axs[1, c].set_xticklabels([r["time bin"].replace("pre-commute", "pre").replace("commute", "comm") for r in rr], fontsize=6, rotation=30)
        axs[0, c].tick_params(labelsize=7)
        axs[1, c].tick_params(labelsize=7)
    axs[0, 0].set_ylabel("device weight w", fontsize=8)
    axs[1, 0].set_ylabel("DriftGate - best fixed (pp)", fontsize=8)
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"R19_fig2_time_bins.{ext}", dpi=200)
    plt.close(fig)


def main():
    if sys.argv[1] == "run":
        analyse(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "tables":
        tables()
    else:
        raise SystemExit(sys.argv[1])


if __name__ == "__main__":
    main()
