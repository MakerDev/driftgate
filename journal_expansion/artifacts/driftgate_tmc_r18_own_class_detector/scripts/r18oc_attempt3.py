"""Round 18 (own-class detector): improvement attempt 3 (directive 7.4 item 3; hypothesis, change and expected numbers in
decisions.md D18, written before this script was run).

Change (answer rule only): an offloaded request is answered with argmax(w * p_device + (1 - w) * p_edge(r)), where p_edge(r)
is the edge output corrected with the r of edge-corrected-by-detector (same theta, held-out histograms, window and clips) and
w is the weight of the v32 method at the same offloading ratio (r17.dgp_weight; r16.driftgate_weight at ratio 1.0). The
ranking (smallest g first), the detector and the answers of the requests kept on the device are unchanged.

  python .../r18oc_attempt3.py run <training run name> [condition ...]   # cache/<run>__<cond>__a3.pkl (new files only)
  python .../r18oc_attempt3.py dev <model config>                        # the D12 gate on S1 development seeds 8-10
  python .../r18oc_attempt3.py cells <model config>                      # stage-2 decision cells with the new answer rule
  python .../r18oc_attempt3.py check <training run name>                 # the recomputed edge-corrected-by-detector curve
                                                                          # equals the stage-2 cache (no attempt-3 numbers)
"""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tb", HERE / "scripts" / "r18oc_tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
an = T.an
GRID, NEW, CACHE, TABLES = an.GRID, an.NEW, T.CACHE, T.TABLES
A3 = "average-corrected-by-detector"
KEEP_KIND = an.KEEP_KIND


def load_cond(name, cond):
    tr, rp = an.run_names(name)
    D, X = an.load(rp if cond == "frozen-replay" else tr)
    if cond == "first-day":
        D, X = an.subset(D, X, D["er"] <= an.DAY)
    elif cond == "second-day":
        D, X = an.subset(D, X, D["er"] > an.DAY)
    return D, X


def curves_a3(D, X, main, with_a3=True):
    """curves, counts and per-kind counts of edge-corrected-by-detector (reproduction) and of the attempt-3 rule."""
    N = len(D["y"])
    y, pc, ps, cp, Mrow, a = D["y"], D["pc"], D["ps"], D["cp"], D["Mrow"], D["a_req"].astype(np.float64)
    tb = np.random.default_rng([X["seed"], 18, 7]).random(N)
    co, ix = an.m14.client_order(D), an.m14.window_index(D)
    Hc, Hs = D["ent"].astype(np.float64), an.m13.server_entropy(ps)
    w_auto, _ = an.r16.driftgate_weight(D)
    oo = an.own_other_max(ps, Mrow)
    c_dev = cp == y
    j = X["variants"].index(main)
    g = X["g"][:, j].astype(np.float64)
    has = g >= 0
    Hown, Hoth = X["hist_own"][:, :, j].astype(np.float64), X["hist_other"][:, :, j].astype(np.float64)
    pos, order = an.rank_offload(-g, tb)
    keys = [NEW] + ([A3] if with_a3 else [])
    cl = {k: [] for k in keys}
    for rho in GRID:
        off = pos < int(np.ceil(rho * N - 1e-9))
        th = an.theta_of(g, order, int(off.sum()))
        rr = an.detector_r(D, co, off, th, Hown, Hoth, has)
        cl[NEW].append(np.where(off, an.corr_argmax(oo, a, rr["r"]) == y, c_dev))
        if with_a3:
            w = w_auto if rho >= 1.0 else an.r17.dgp_weight(ix, co, Hc, Hs, off)
            pq = an.m13.corrected(ps, Mrow, D["a_req"], rr["r"].astype(np.float32))
            cl[A3].append(np.where(off, an.m15.mix_answer(w, pc, pq) == y, c_dev))
            del pq
    kinds = D["kind"]
    out = dict(curves={}, cnt={}, cnt_kind={})
    for k in keys:
        out["curves"][k] = np.array([an.macro(D, c) for c in cl[k]])
        out["cnt"][k] = np.stack([an.counts(D, c) for c in cl[k]])
        out["cnt_kind"][k] = {rho: np.stack([an.counts(D, cl[k][list(GRID).index(rho)] & (kinds == kk)) for kk in range(3)])
                              for rho in KEEP_KIND}
    return out


def run(name, conds):
    tr, _ = an.run_names(name)
    for cond in conds:
        f = CACHE / f"{tr}__{cond}__a3.pkl"
        if f.exists():
            print(f"{f.name} exists", flush=True)
            continue
        base = pickle.load(open(CACHE / f"{tr}__{cond}.pkl", "rb"))
        main = base["main"] or T.chosen_variant(cfg_of(tr))
        t0 = time.time()
        D, X = load_cond(name, cond)
        out = curves_a3(D, X, main)
        dev = float(np.max(np.abs(out["curves"][NEW] - base["curves"][(main, NEW)])))
        out.update(run=tr, cond=cond, main=main, check_new_max_abs_diff=dev, seconds=time.time() - t0)
        assert dev < 1e-9, dev
        pickle.dump(out, open(f, "wb"))
        print(json.dumps(dict(run=tr, cond=cond, main=main, check_new_max_abs_diff=dev, seconds=round(out["seconds"], 1))),
              flush=True)
        del D, X


def check(name):
    """reproduce edge-corrected-by-detector on the first-day condition without computing the attempt-3 rule."""
    tr, _ = an.run_names(name)
    base = pickle.load(open(CACHE / f"{tr}__first-day.pkl", "rb"))
    main = base["main"] or T.chosen_variant(cfg_of(tr))
    D, X = load_cond(name, "first-day")
    out = curves_a3(D, X, main, with_a3=False)
    print(json.dumps(dict(run=tr, main=main, max_abs_diff=float(np.max(np.abs(out["curves"][NEW] - base["curves"][(main, NEW)]))))))


def cfg_of(tr):
    tag = tr.split("_")[1]
    return {v: k for k, v in T.TAG.items()}[tag]


def a3_cache(cfg, st, seed, cond, att=None):
    """the stage-2 cache with the new method's curve (chosen variant, edge-corrected-by-detector) replaced by attempt 3."""
    o = T.cache(cfg, st, seed, cond)
    f = CACHE / f"r18oc_{T.TAG[cfg]}_{st}_s{seed}__{cond}__a3.pkl"
    if o is None or not f.exists():
        return None
    a = pickle.load(open(f, "rb"))
    o = dict(o)
    o["curves"] = dict(o["curves"])
    o["curves"][(a["main"], NEW)] = a["curves"][A3]
    return o


def dev(cfg):
    v = T.chosen_variant(cfg)
    j = list(GRID).index(0.5)
    rows, a, b = [], [], []
    for s in T.DEV:
        o0, o1 = T.cache(cfg, "s1", s, "frozen-replay"), a3_cache(cfg, "s1", s, "frozen-replay")
        if o0 is None or o1 is None:
            print(f"missing caches for seed {s}")
            return None
        a.append(100 * o0["curves"][(v, NEW)][j])
        b.append(100 * o1["curves"][(v, NEW)][j])
        rows.append([cfg, 3, v, s, f"{a[-1]:.2f}", f"{b[-1]:.2f}", f"{b[-1] - a[-1]:.2f}"])
    d = np.subtract(b, a)
    rows.append([cfg, 3, v, "mean (SD)", T.fmt(float(np.mean(a)), T.sd(a)), T.fmt(float(np.mean(b)), T.sd(b)),
                 T.fmt(float(np.mean(d)), T.sd(d))])
    T.write(f"R18oc_attempt3_{T.TAG[cfg]}_dev.csv", ["model config", "attempt", "detector variant", "seed",
                                                      "without the change: accuracy at 0.5 (%)", "attempt: accuracy at 0.5 (%)",
                                                      "difference pp"], rows)
    res = dict(model_config=cfg, attempt=3, change=A3, variant=v, base=float(np.mean(a)), attempt_acc=float(np.mean(b)),
               gain_pp=float(np.mean(d)), gain_sd=T.sd(d), threshold_pp=T.ATT_GAIN_PP,
               go_to_evaluation=bool(np.mean(d) >= T.ATT_GAIN_PP))
    (TABLES / f"R18oc_attempt3_{T.TAG[cfg]}_dev.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def cells(cfg):
    orig = T.cache
    T.cache = lambda c, st, s, cond, att=None: a3_cache(c, st, s, cond) if att == "a3" else orig(c, st, s, cond, att)
    try:
        rows, res = [], {}
        for st in ("s1", "s2"):
            for cond in ("second-day", "frozen-replay"):
                c0 = T.cell(cfg, st, cond, T.SEEDS[st])
                c = T.cell(cfg, st, cond, T.SEEDS[st], att="a3")
                if c is None:
                    print(f"missing attempt-3 caches: {st} {cond}")
                    return None
                n = c["new"]
                rows.append([T.SETTING_NAME[st], cfg, cond, c["main"], f"{c['A']:.2f}", f"{n['ratio']:.3f}" if n["ratio"] else "none",
                             f"{c['r_env']:.3f}", f"{n['S']:.3f}", T.fmt(n["G"][0.25], n["G_sd"][0.25]),
                             T.fmt(n["G"][0.5], n["G_sd"][0.5]), f"{n['acc'][0.5]:.2f}", f"{c0['new']['acc'][0.5]:.2f}",
                             "yes" if c["room"] else "no", n["judgement"], c0["new"]["judgement"], "yes" if c["border"] else "no"])
                res[f"{st} {cond}"] = dict(judgement=n["judgement"], room=c["room"], S=n["S"], G05=n["G"][0.5])
        T.write(f"R18oc_attempt3_{T.TAG[cfg]}_cells.csv",
                ["setting", "model config", "condition", "detector variant", "A* (%)", "ratio-at-target new",
                 "ratio-at-target comparator envelope", "S", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)",
                 "new accuracy at 0.5 (attempt 3)", "new accuracy at 0.5 (stage 2)", "room", "judgement (attempt 3)",
                 "judgement (stage 2)", "border"], rows)
        roomy = [v for v in res.values() if v["room"]]
        res = dict(model_config=cfg, attempt=3, cells=res,
                   all_roomy_partial_or_better=bool(roomy and all(v["judgement"] in ("pass", "partial") for v in roomy)))
        (TABLES / f"R18oc_attempt3_{T.TAG[cfg]}_cells.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return res
    finally:
        T.cache = orig


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3:] or list(an.CONDITIONS))
    elif cmd == "check":
        check(sys.argv[2])
    elif cmd == "dev":
        dev(sys.argv[2])
    elif cmd == "cells":
        cells(sys.argv[2])
