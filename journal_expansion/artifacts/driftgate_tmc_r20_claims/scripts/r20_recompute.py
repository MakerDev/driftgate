"""Round 20: recompute the inference rules from the stored request outputs (R20_plan.md section 2).

No training and no new inference: every answer is computed from the stored device-exit and raw edge-exit probabilities,
request order, cell membership, M_k and a_k of the existing records. Per run and per beta in {0.25, 0.5, 0.75, 1} (Round 18
controller; beta = 0 is device only) the script stores correct counts per evaluation round and device for each rule, the
offloading ratio, edge calls per request and the two-cell share of offloaded requests. For S1, S2 and their frozen replays it
also stores the correction ablation of Task 3 (b = 0, constant a, current a_k) at beta 0.5 and 1. Reproduction checks against
the Round 15, 16, 18 and 19 caches are stored in out["checks"].

Subcommands: run SETTING SEED -> cache/<name>.pkl ; a0 -> cache/a0.json (median a_k over present device-rounds of the S1
development seeds 5-7; run after the three development runs)
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


r19 = _load("r19", ART / "driftgate_tmc_r19_weights" / "scripts" / "r19_analysis.py")
md = _load("r19md", ART / "driftgate_tmc_r19_weights" / "scripts" / "r19_multiday.py")
r18, r17, r16, m15, m14, m13 = r19.r18, r19.r17, r19.r16, r19.m15, r19.m14, r19.m13
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
R19C = ART / "driftgate_tmc_r19_weights" / "cache"
R18C = ART / "driftgate_tmc_r18_compare" / "cache"

SETTINGS = dict(r18.SETTINGS)
SETTINGS["S1 development"] = (JR / "runs" / "phaseT10_prior", "r10_T40_s{}", [5, 6, 7], "day1")
SETTINGS["S1 three-day"] = (None, "md_S1_s{}", range(5), "three-day")
SETTINGS["S2 three-day"] = (None, "md_S2_s{}", range(3), "three-day")
T3 = ["S1", "S2", "S1 replay", "S2 replay"]
BETAS = [0.25, 0.5, 0.75, 1.0]
A_DIAG = 0.25
P_FLOOR = m15.P_FLOOR
FIXED_W = {"Correction + w 0.5": 0.5, "Correction + w 0.2": 0.2, "Correction + w 0.45": 0.45}
# rule -> (Round 18 pr name, Round 15/16 key); None where the round has no such rule
R18_NAME = {"Device only": "Device only", "Raw edge only": "Raw edge only", "Probability average": "Probability average",
            "Logit sum": "Logit sum", "Correction + w 0.5": "Correction + fixed w 0.5", "Correction + w 0.2": "Correction + development w 0.2",
            "Corrected edge only": "Corrected edge only", "Correction + product": "Correction + product", "DriftGate": "DriftGate-P",
            "No correction + adaptive w": "No correction + adaptive w"}
R15_KEY = {"Device only": "B1", "Raw edge only": "B2", "Probability average": "B3", "Logit sum": "R-PoE", "Lower-entropy exit": "R-THE",
           "Correction + w 0.5": "correction + w 0.5", "Correction + w 0.2": "correction + dev w 0.2", "Corrected edge only": "corrected edge only",
           "Correction + product": "correction + product", "DriftGate": "DriftGate", "No correction + adaptive w": "no correction + adaptive w",
           "Confidence-based offloading": "B0", "Logit-entropy weighting": "LEW", "Label-shift EM": "R-EM", "Learned weight": "R-LR",
           "Correction + learned weight": "correction + learned weight"}
REFS_ONLY = ["Logit-entropy weighting", "Label-shift EM", "Learned weight", "Correction + learned weight"]


def run_name(st, seed):
    return SETTINGS[st][1].format(seed)


def counts(g, c, E, K):
    return np.bincount(g, weights=c, minlength=E * K).reshape(E, K).astype(np.int32)


def macro_rounds(cnt, n_ek, present):
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(present, cnt / n_ek, np.nan)
    return np.nanmean(A, axis=1)


def load_run(st, seed):
    runs, pat, _, period = SETTINGS[st]
    name = pat.format(seed)
    if period == "three-day":
        D, meta = md.combine(st.split()[0], seed)
        return D, meta["cells"], name, dict(day_of_round=meta["day_of_round"], start_min=meta["start_min"])
    D = m13.load(runs, name)
    cells = np.sort(m15.cells_matrix(runs, name, D["K"], D["er"]), axis=2)
    return D, cells, name, {}


def analyse(st, seed):
    t0 = time.time()
    runs, _, _, period = SETTINGS[st]
    D, cells, name, extra = load_run(st, seed)
    y, pc, ps, cp, sp, ent, Mrow, a = (D[k] for k in ("y", "pc", "ps", "cp", "sp", "ent", "Mrow", "a_req"))
    N, E, K, g = len(y), D["E"], D["K"], D["g"]
    two = ((cells >= 0).sum(2) == 2)[D["e"], D["k"]]
    out = dict(setting=st, seed=seed, name=name, period=period, er=D["er"], E=E, K=K, n_ek=D["n_ek"], present=D["present"],
               home=D["home"], n_req=N, cnt={}, cost={}, checks={}, extra=extra)
    psr = m13.corrected(ps, Mrow, a, 0.5)
    Hc, Hs = ent.astype(np.float64), m13.server_entropy(ps)
    ix, co = m14.window_index(D), m14.client_order(D)
    w_auto, _ = r16.driftgate_weight(D)
    # ---- answers that do not depend on the offloading mask
    lpc = np.log(np.maximum(pc, P_FLOOR))
    full = {"Raw edge only": sp, "Probability average": (pc + ps).argmax(1),
            "Logit sum": (lpc + np.log(np.maximum(ps, P_FLOOR))).argmax(1),
            "Correction + product": (lpc + np.log(np.maximum(psr, P_FLOOR))).argmax(1),
            "Lower-entropy exit": np.where(Hc <= Hs, cp, sp), "Corrected edge only": psr.argmax(1)}
    del lpc
    for k_, w in FIXED_W.items():
        full[k_] = r16.mix_scalar(w, pc, psr)
    refs = None
    if runs is not None and r16.refs_path(str(runs), name).exists():
        z = np.load(r16.refs_path(str(runs), name))
        refs = {k_: z[R15_KEY[k_]].astype(np.int64) for k_ in list(full) + REFS_ONLY + ["DriftGate", "No correction + adaptive w",
                                                                                       "Confidence-based offloading"]
                if k_ in R15_KEY and R15_KEY[k_] in z.files}
        out["checks"]["refs_w_auto_maxabs"] = float(np.abs(z["w_auto"] - w_auto).max())
        out["checks"]["answers_equal_refs"] = {k_: float((full[k_] == refs[k_]).mean()) for k_ in full if k_ in refs}
    # ---- device only and confidence-based offloading (0.8 nats)
    out["cnt"]["Device only"] = counts(g, cp == y, E, K)
    off0 = ent > m15.TAU0
    out["cnt"]["Confidence-based offloading"] = counts(g, np.where(off0, sp, cp) == y, E, K)
    out["cost"]["Confidence-based offloading"] = dict(srv=float(off0.mean()), calls=float((off0 * (1 + two)).mean()),
                                                      two_share=float(two[off0].mean()), srv_round=np.bincount(D["e"], weights=off0, minlength=E))
    off_c0 = r18.controller(co, Hc, 0.0)
    out["checks"]["controller_beta0_offloading"] = float(off_c0.mean())
    if refs is not None:
        for k_ in REFS_ONLY:
            if k_ in refs:
                out["cnt"][f"{k_} @ 1.0"] = counts(g, refs[k_] == y, E, K)
    # ---- beta masks
    for beta in BETAS:
        off = r18.controller(co, Hc, beta)
        wP = w_auto if beta == 1.0 else r17.dgp_weight(ix, co, Hc, Hs, off)
        A = dict(full)
        A["DriftGate"] = m15.mix_answer(wP, pc, psr)
        A["No correction + adaptive w"] = m15.mix_answer(wP, pc, ps)
        if beta == 1.0 and refs is not None:
            out["checks"]["DriftGate_equals_refs"] = float((A["DriftGate"] == refs["DriftGate"]).mean())
            out["checks"]["NoCorr_equals_refs"] = float((A["No correction + adaptive w"] == refs["No correction + adaptive w"]).mean())
        for k_, v in A.items():
            out["cnt"][f"{k_} @ {beta}"] = counts(g, np.where(off, v, cp) == y, E, K)
        out["cost"][beta] = dict(srv=float(off.mean()), calls=float((off * (1 + two)).mean()), two_share=float(two[off].mean()),
                                 srv_round=np.bincount(D["e"], weights=off, minlength=E))
        out.setdefault("wP_mean", {})[beta] = float(wP[off].mean())
        # ---- Task 3: correction ablation with the same mask and the same weight series
        if st in T3 and beta in (0.5, 1.0):
            a0 = json.loads((CACHE / "a0.json").read_text())["a0"]
            for tag, aval in (("a0", a0), ("a0.25", A_DIAG)):
                pq = m13.corrected(ps, Mrow, np.full(N, aval, np.float32), 0.5)
                T = {f"DriftGate [{tag}]": m15.mix_answer(wP, pc, pq), f"Corrected edge only [{tag}]": pq.argmax(1),
                     f"Correction + w 0.5 [{tag}]": r16.mix_scalar(0.5, pc, pq)}
                del pq
                for k_, v in T.items():
                    out["cnt"][f"{k_} @ {beta}"] = counts(g, np.where(off, v, cp) == y, E, K)
            out.setdefault("a0_used", a0)
    # ---- a_k and b_k (Task 3); identity of the correction and an own-class logit offset (first 200,000 requests)
    out["a_ek"] = D["a_ek"].astype(np.float32)
    s = slice(0, min(N, 200_000))
    b = np.log((1 - a[s]) / a[s]).astype(np.float64)
    with np.errstate(divide="ignore"):
        L = np.log(ps[s].astype(np.float64)) + b[:, None] * Mrow[s]
    P2 = np.exp(L - L.max(1, keepdims=True))
    P2 /= P2.sum(1, keepdims=True)
    out["checks"]["identity_maxabs"] = float(np.abs(P2 - psr[s]).max())
    out["checks"]["identity_argmax_equal"] = float((P2.argmax(1) == psr[s].argmax(1)).mean())
    # ---- reproduction checks
    nE, pr_ = D["n_ek"], D["present"]
    if period in ("day1", "replay") and st != "S1 development":
        c18 = pickle.load(open(R18C / f"{name}.pkl", "rb"))
        dif = {}
        for k_, n18 in R18_NAME.items():
            if k_ == "Device only":
                dif[k_] = float(np.nanmax(np.abs(macro_rounds(out["cnt"][k_], nE, pr_) - c18["pr"]["Device only"]["all"])))
                continue
            for beta in BETAS:
                dif[f"{k_} @ {beta}"] = float(np.nanmax(np.abs(macro_rounds(out["cnt"][f"{k_} @ {beta}"], nE, pr_) - c18["pr"][f"{n18} @ {beta}"]["all"])))
        out["checks"]["r18_macro_maxabs"] = max(dif.values())
        out["checks"]["r18_macro_maxabs_by_rule"] = dif
        out["checks"]["r18_cost_maxabs"] = max(max(abs(out["cost"][b_]["srv"] - c18["srv"][b_]["full"]),
                                                   abs(out["cost"][b_]["calls"] - c18["calls"][b_])) for b_ in BETAS)
    if period in ("day1", "replay"):
        c15 = r16.r15_cache_value(str(runs), name)
        if c15 is not None:
            dif = {}
            for k_, key in R15_KEY.items():
                ck = k_ if k_ in ("Device only", "Confidence-based offloading") else f"{k_} @ 1.0"
                if ck in out["cnt"] and key in c15["res"]:
                    dif[k_] = abs(float(np.nanmean(macro_rounds(out["cnt"][ck], nE, pr_))) - c15["res"][key]["full"])
            out["checks"]["r15_full_maxabs"] = max(dif.values())
            out["checks"]["r15_full_by_rule"] = dif
            out["r15_res_full"] = {k_: c15["res"][key]["full"] for k_, key in R15_KEY.items() if key in c15["res"]}
            out["r15_res_gt30"] = {k_: c15["res"][key]["gt30"] for k_, key in R15_KEY.items() if key in c15["res"]}
            out["r15_res_full"]["Geometric ensemble with early exit"] = c15["res"]["R-ZTW"]["full"]
            out["r15_res_gt30"]["Geometric ensemble with early exit"] = c15["res"]["R-ZTW"]["gt30"]
    r19f = R19C / f"{name}.pkl"
    if r19f.exists():
        c19 = pickle.load(open(r19f, "rb"))
        ok = {}
        for beta in (1.0, 0.5):
            for k_, k19 in (("DriftGate", "DriftGate"), ("Corrected edge only", "w=0.00"), ("Correction + w 0.5", "w=0.50"),
                            ("Correction + w 0.2", "w=0.20"), ("Correction + w 0.45", "w=0.45")):
                ok[f"{k_} @ {beta}"] = bool(np.array_equal(out["cnt"][f"{k_} @ {beta}"], np.rint(c19["beta"][beta]["cnt"][k19]).astype(np.int32)))
        out["checks"]["r19_counts_equal"] = ok
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(CACHE / f"{name}.pkl", "wb"))
    summ = {k_: v for k_, v in out["checks"].items() if not isinstance(v, dict)}
    summ["r19_all_equal"] = all(out["checks"].get("r19_counts_equal", {True: True}).values())
    print(json.dumps(dict(name=name, seconds=round(out["seconds"], 1), checks=summ)), flush=True)


def a0():
    v = []
    for s in SETTINGS["S1 development"][2]:
        o = pickle.load(open(CACHE / f"r10_T40_s{s}.pkl", "rb"))
        v.append(o["a_ek"][o["present"]])
    v = np.concatenate(v)
    res = dict(a0=float(np.median(v)), n_device_rounds=int(len(v)), source="median of a_k over present device-rounds, S1 development seeds 5-7")
    (CACHE / "a0.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        analyse(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "a0":
        a0()
