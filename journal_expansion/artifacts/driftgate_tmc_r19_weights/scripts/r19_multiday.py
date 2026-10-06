"""Round 19 section 5: analysis of the three-day frozen replay (protocol: R19_multiday_protocol.md).

Per seed the three replay days are concatenated into one stream: evaluation round index e + (day - 1) * 31, device order
(day, round, arrival). The DriftGate window, the controller history (beta 0.5), the DriftGate-P weight and the label-free
weight fixed after the first 128 offloaded requests therefore run across days without reset. The per-run computation is
the same as r19_analysis.analyse (copied, with the combined stream as input); fixed weights use correction r = 0.5.
Subcommands: run S1|S2 SEED -> cache/md_<sc>_s<seed>.pkl ; tables -> tables/R19_multiday_*.csv
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
_spec = importlib.util.spec_from_file_location("r19", HERE / "scripts" / "r19_analysis.py")
r19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r19)
r18, r17, r16, m15, m14, m13 = r19.r18, r19.r17, r19.r16, r19.m15, r19.m14, r19.m13
OUT = JR / "runs" / "phaseT19_multiday"
DAYS = (1, 2, 3)
SEEDS = {"S1": range(5), "S2": range(3)}


def combine(sc, seed):
    Ds, cells = [], []
    for d in DAYS:
        name = f"r19_{sc.lower()}_s{seed}_day{d}"
        D = m13.load(OUT, name)
        Ds.append(D)
        cells.append(np.sort(m15.cells_matrix(OUT, name, D["K"], D["er"]), axis=2))
    E1, K = Ds[0]["E"], Ds[0]["K"]
    assert all(D["E"] == E1 and D["K"] == K and np.array_equal(D["M"], Ds[0]["M"]) for D in Ds)
    cat = lambda key: np.concatenate([D[key] for D in Ds])
    e = np.concatenate([D["e"] + i * E1 for i, D in enumerate(Ds)])
    k = cat("k")
    E = E1 * len(Ds)
    g = e * K + k
    n_ek = np.bincount(g, minlength=E * K).reshape(E, K).astype(float)
    C = dict(h=Ds[0]["h"], E=E, K=K, C=Ds[0]["C"], e=e, k=k, g=g, y=cat("y"), kind=cat("kind"), M=Ds[0]["M"], pc=cat("pc"), ps=cat("ps"),
             cp=cat("cp"), sp=cat("sp"), ent=cat("ent"), arrival=cat("arrival"), a_req=cat("a_req"),
             a_ek=np.concatenate([D["a_ek"] for D in Ds]), n_ek=n_ek, present=n_ek > 0,
             home=np.concatenate([D["home"] for D in Ds]), has_home=True, checks=Ds[0]["checks"])
    C["Mrow"] = C["M"][k]
    er_day = Ds[0]["er"]
    C["er"] = np.concatenate([er_day + 150 * i for i in range(len(Ds))])
    meta = dict(day_of_round=np.repeat(np.arange(1, len(Ds) + 1), E1), start_min=np.tile(300.0 + 6.0 * (er_day - 1), len(Ds)),
                cells=np.concatenate(cells))
    return C, meta


def analyse(sc, seed):
    t0 = time.time()
    D, meta = combine(sc, seed)
    pc, ps, Mrow, a, y, cp, ent = (D[k] for k in ("pc", "ps", "Mrow", "a_req", "y", "cp", "ent"))
    N, E, K, g = len(y), D["E"], D["K"], D["g"]
    psr = m13.corrected(ps, Mrow, a, 0.5)
    w_auto, Hs = r16.driftgate_weight(D)
    Hc = ent.astype(np.float64)
    ix, co = m14.window_index(D), m14.client_order(D)
    oc, first = co
    cells = meta["cells"]
    trans = np.zeros((E, K), bool)
    trans[1:] = np.any(cells[1:] != cells[:-1], axis=2)
    corr_dev = cp == y
    Cg = [r16.mix_scalar(w, pc, psr) == y for w in r19.W_GRID]
    out = dict(setting=f"{sc} three-day replay", seed=seed, E=E, K=K, n_ek=D["n_ek"], present=D["present"], home=D["home"], trans=trans,
               er=D["er"], day_of_round=meta["day_of_round"], start_min=meta["start_min"], n_req=N, beta={})
    for beta in r19.BETAS:
        off = r18.controller(co, Hc, beta)
        wdg = w_auto if beta == 1.0 else r17.dgp_weight(ix, co, Hc, Hs, off)
        o_off = off[oc]
        dev_id = np.cumsum(first) - 1
        cum = np.cumsum(o_off)
        start_cum = np.r_[0, cum[np.flatnonzero(first)[1:] - 1]]
        n_before = cum - o_off - start_cum[dev_id]
        k_of = D["k"][oc]
        m_first = o_off & (n_before < r19.N_INIT)
        nd = int(first.sum())
        cnt_f = np.bincount(dev_id, weights=m_first, minlength=nd)
        shc = np.bincount(dev_id, weights=np.where(m_first, Hc[oc], 0.0), minlength=nd)
        shs = np.bincount(dev_id, weights=np.where(m_first, Hs[oc], 0.0), minlength=nd)
        wfix = np.full(K, np.nan)
        kk = k_of[np.flatnonzero(first)]
        with np.errstate(invalid="ignore", divide="ignore"):
            wfix[kk] = np.where(cnt_f == r19.N_INIT, shs / (shc + shs), np.nan)
        use_fix = np.zeros(N, bool)
        use_fix[oc] = (n_before >= r19.N_INIT) & np.isfinite(wfix[k_of])
        w_init = np.where(use_fix, wfix[D["k"]], wdg)
        s_w = np.bincount(D["k"], weights=np.where(off, wdg, 0.0), minlength=K)
        n_w = np.bincount(D["k"], weights=off, minlength=K)
        wconst = np.where(n_w > 0, s_w / np.maximum(n_w, 1), 0.5)
        corr = {"DriftGate": np.where(off, r19.mix_vec(wdg, pc, psr) == y, corr_dev),
                "initial fixed": np.where(off, r19.mix_vec(w_init, pc, psr) == y, corr_dev),
                "device constant": np.where(off, r19.mix_vec(wconst[D["k"]], pc, psr) == y, corr_dev)}
        for j, w in enumerate(r19.W_GRID):
            corr[f"w={w:.2f}"] = np.where(off, Cg[j], corr_dev)
        out["beta"][beta] = dict(cnt={k_: np.bincount(g, weights=v, minlength=E * K).reshape(E, K) for k_, v in corr.items()},
                                 srv=float(off.mean()), wdg_mean=float(wdg[off].mean()), wdg_sd=float(wdg[off].std()),
                                 wdg_day=[float(wdg[off & (meta["day_of_round"][D["e"]] == d)].mean()) for d in DAYS],
                                 wfix=wfix[kk], share_not_fixed=float(np.isnan(wfix[kk]).mean()))
    out["seconds"] = time.time() - t0
    pickle.dump(out, open(HERE / "cache" / f"md_{sc}_s{seed}.pkl", "wb"))
    print(json.dumps(dict(setting=sc, seed=seed, seconds=round(out["seconds"], 1))))


def tables():
    st = json.loads((HERE / "tables" / "R19_state.json").read_text())
    w_dev = st["development_w"]
    rows, seeds_rows, curve = [], [], []
    verdict = {}
    for sc, seeds in SEEDS.items():
        R = [pickle.load(open(HERE / "cache" / f"md_{sc}_s{s}.pkl", "rb")) for s in seeds]
        for beta in r19.BETAS:
            wins = {"three days": (None, None), "day 1": ({1}, None), "day 2": ({2}, None), "day 3": ({3}, None),
                    "days 2-3": ({2, 3}, None), "transition": (None, "t"), "non-transition": (None, "n"), "home": (None, "h"), "away": (None, "a")}
            for wn, (days, dm) in wins.items():
                def val(a, key):
                    A = r19.acc_matrix(a, beta, key)
                    rm = np.isin(a["day_of_round"], list(days)) if days else None
                    if dm is None:
                        return r19.metric(A, rm)
                    mask = {"t": a["trans"], "n": ~a["trans"], "h": a["home"] == 1, "a": a["home"] == 0}[dm] & a["present"]
                    return r19.metric(A, rm, mask)
                dg = np.array([val(a, "DriftGate") for a in R]) * 100
                grid = np.array([[val(a, f"w={w:.2f}") for w in r19.W_GRID] for a in R]) * 100
                jb = int(r19.argbest(grid.mean(0)))
                row = [sc, beta, wn, r19.fmt(dg)]
                for lab, key in ((f"development w {w_dev:.2f}", f"w={w_dev:.2f}"), ("Round 11 w 0.20", "w=0.20"), ("initial fixed", "initial fixed"),
                                 ("corrected edge only", "w=0.00"), ("device constant (post hoc)", "device constant")):
                    v = np.array([val(a, key) for a in R]) * 100
                    d = dg - v
                    row += [r19.fmt(v), r19.fmt(d, 3), int((d > 0).sum())]
                    if wn in ("three days", "days 2-3", "transition") and "post hoc" not in lab and "Round 11" not in lab:
                        verdict[(sc, beta, wn, lab)] = dict(mean=float(d.mean()), positive_seeds=int((d > 0).sum()), n=len(d),
                                                            improved=bool(d.mean() >= 0.1 and (d > 0).all()))
                row += [f"{r19.W_GRID[jb]:.2f}", f"{grid.mean(0)[jb]:.3f}", r19.fmt(dg - grid[:, jb], 3)]
                rows.append(row)
                if wn == "three days":
                    curve.append([sc, beta] + [f"{v:.3f}" for v in grid.mean(0)] + [f"{dg.mean():.3f}"])
                    for a, x in zip(R, dg):
                        seeds_rows.append([sc, beta, a["seed"], f"{x:.4f}"] + [f"{val(a, k_) * 100:.4f}" for k_ in
                                                                            ("initial fixed", f"w={w_dev:.2f}", "w=0.00")])
        R0 = R
        for beta in r19.BETAS:
            verdict[(sc, beta, "w")] = dict(wdg_day=[r19.fmt([a["beta"][beta]["wdg_day"][i] for a in R0], 3) for i in range(3)],
                                            wfix=r19.fmt([np.nanmean(a["beta"][beta]["wfix"]) for a in R0], 3),
                                            share_not_fixed=r19.fmt([a["beta"][beta]["share_not_fixed"] for a in R0], 3),
                                            transitions_share=r19.fmt([a["trans"][a["present"]].mean() for a in R0], 3))
    hdr = ["scenario", "beta", "window", "DriftGate"]
    for lab in (f"development w {w_dev:.2f}", "Round 11 w 0.20", "initial fixed", "corrected edge only", "device constant (post hoc)"):
        hdr += [lab, f"DriftGate minus {lab} pp", f"seeds with DriftGate higher than {lab}"]
    hdr += ["best fixed w (post hoc, seed mean)", "its accuracy", "DriftGate minus best fixed pp"]
    r19.wcsv("R19_multiday_table.csv", hdr, rows)
    r19.wcsv("R19_multiday_curves.csv", ["scenario", "beta"] + [f"w={w:.2f}" for w in r19.W_GRID] + ["DriftGate"], curve)
    r19.wcsv("R19_multiday_per_seed.csv", ["scenario", "beta", "seed", "DriftGate", "initial fixed", f"development w {w_dev:.2f}",
                                           "corrected edge only"], seeds_rows)
    (HERE / "tables" / "R19_multiday_verdict.json").write_text(json.dumps({" | ".join(map(str, k)): v for k, v in verdict.items()}, indent=1))
    for k, v in verdict.items():
        print(k, v)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        analyse(sys.argv[2], int(sys.argv[3]))
    else:
        tables()
