"""Round 6 — tables T1-T8, T10 and paper_numbers_r6.csv, from RAW run files only.
raw JSON + *_trace.npz + env files -> this script -> tables/*.csv (-> figures, report).

Definitions (R6 directive §2, §5.2):
  integrated accuracy  mean of acc_total over the evaluation rounds {1, 5, ..., 150}
  bottom-10% accuracy  per evaluation round the 10th percentile of client acc_total, then averaged
  home / away accuracy per evaluation round the mean over clients at home / not at home, averaged over
                       the rounds where that group is non-empty
  Main / non-Main acc  per evaluation round the mean over clients of their Main (non-Main) accuracy
  offload rate         offloaded evaluation requests / all evaluation requests (pooled)
  paired difference    same seeds only, Student-t 95% CI, n_pos = seeds where DriftGate is higher, SD ddof=1
  best fixed           the fixed lambda with the highest seed-mean integrated accuracy among the fixed
                       arms tested in that scenario, over the seeds used in the comparison
Missing runs leave empty cells (never imputed).
"""
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parent.parent
RUNS = JR / "runs"
TAB = HERE / "tables"
TAB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

# Schedule A column of T2: the Round-5 main table recomputed with the new DriftGate definition (R0, r6_r0_tables.py);
# fixed-lambda and APFL rows there are the unchanged Round-5 runs.
R5_T1 = HERE / "r0" / "tables" / "T1_main_results.csv"
SLOTS = [("pre-commute", "05:00-07:30", 1, 25), ("commute", "07:30-09:30", 26, 45),
         ("daytime", "09:30-16:00", 46, 110), ("return", "16:00-19:00", 111, 140),
         ("evening", "19:00-20:00", 141, 150)]
FIXED_ALL = ["fixed015", "fixed020", "fixed030", "fixed040", "fixed050", "fixed060"]
FIXED3 = ["fixed020", "fixed040", "fixed060"]
# 2026-10-01: DriftGate / entropy arms without the neighbour score average (arm names *_own). The earlier
# arms "driftgate" / "entropy" (with the average) are retired records and appear in no table.
DG, ENT = "driftgate_own", "entropy_own"
ARM_NAME = {DG: "DriftGate", ENT: "entropy controller", "apfl001": "APFL η=0.01",
            "apfl010": "APFL η=0.1", **{f: f"fixed {int(f[5:]) / 100:g}" for f in FIXED_ALL}}
SCEN = {  # label -> (run dir, run prefix, display name)
    "S1": ("phaseT6_S1", "s1", "commute mobility"),
    "S1fast": ("phaseT6_S1fast", "s1fast", "commute mobility (vehicle speed)"),
    "S2": ("phaseT6_S2", "s2", "GeoLife trace"),
    "S3_K200": ("phaseT6_S3", "s3k200", "commute mobility, K=200"),
    "S3_K500": ("phaseT6_S3", "s3k500", "commute mobility, K=500"),
    "S4_loss01": ("phaseT6_S4", "s4loss01", "signal loss p=0.1"),
    "S4_loss03": ("phaseT6_S4", "s4loss03", "signal loss p=0.3"),
    "S4_delay1": ("phaseT6_S4", "s4delay1", "signal delay 1 round"),
    "S4_delay3": ("phaseT6_S4", "s4delay3", "signal delay 3 rounds"),
    "S4_lowreq": ("phaseT6_S4", "s4lowreq", "low request rate"),
    "S4_part07": ("phaseT6_S4", "s4part07", "participation 0.7"),
    "S4_part05": ("phaseT6_S4", "s4part05", "participation 0.5"),
}
ENV_OF = {"S4_delay1": "S1", "S4_delay3": "S1"}

# ----------------------------------------------------------------------------- loading
_cache = {}


def fam(scen, arm):
    d, pre, _ = SCEN[scen]
    out = {}
    for f in sorted(glob.glob(str(RUNS / d / f"{pre}_{arm}_s*.json"))):
        m = re.search(rf"/{pre}_{arm}_s(\d+)\.json$", f)
        if m:
            out[int(m.group(1))] = f
    return out


def env_of(scen, seed):
    key = (ENV_OF.get(scen, scen), seed)
    if key not in _cache:
        _cache[key] = r6_env.load_env(RUNS / "phaseT6_env" / f"{key[0]}_seed{seed}.npz")[0]
    return _cache[key]


def run(path, scen, seed):
    if path in _cache:
        return _cache[path]
    h = json.load(open(path))
    z = np.load(path[:-5] + "_trace.npz")
    env = env_of(scen, seed)
    er = np.array(h["eval_rounds"])
    acc = z["eval_correct"] / np.maximum(z["eval_n_total"], 1)                   # [E, K]
    with np.errstate(invalid="ignore", divide="ignore"):
        am = np.where(z["eval_n_main"] > 0, z["eval_c_main"] / z["eval_n_main"], np.nan)
        an = np.where(z["eval_n_nonmain"] > 0, z["eval_c_nonmain"] / z["eval_n_nonmain"], np.nan)
    home = env["at_home"][:, er - 1].T                                            # [E, K]
    hm = [acc[e][home[e]].mean() for e in range(len(er)) if home[e].any()]
    aw = [acc[e][~home[e]].mean() for e in range(len(er)) if (~home[e]).any()]
    r = dict(
        acc_ck=acc,
        integ=float(np.mean([e["acc_total"] for e in h["eval"]])),
        acc_round=np.array([e["acc_total"] for e in h["eval"]]),
        rounds=er,
        p10=float(np.mean(np.percentile(acc, 10, axis=1))),
        home=float(np.mean(hm)) if hm else float("nan"),
        away=float(np.mean(aw)) if aw else float("nan"),
        main=float(np.mean(np.nanmean(am, axis=1))),
        nonmain=float(np.mean(np.nanmean(an, axis=1))),
        offload=float(z["eval_n_off"].sum() / z["eval_n_total"].sum()),
        slot={name: float(np.mean([a for a, rr in zip([e["acc_total"] for e in h["eval"]], er) if lo <= rr <= hi]))
              for name, _, lo, hi in SLOTS},
        lam=np.array([[d[str(zc)] for zc in range(h["L"])] for d in h["lamdas"]], float),     # [T, L]
        rho_cell=np.array([[np.nan if d[str(zc)] is None else d[str(zc)] for zc in range(h["L"])]
                           for d in h["rho_cell"]], float),
        ctrl_ns=z["ctrl_ns"] if "ctrl_ns" in z.files else None,
        run_id=h["config"]["run_id"], name=Path(path).stem, K=h["K"], L=h["L"],
        overlap=h["probe_eval_overlap"], n_rounds=len(h["round"]), h=h, env=env)
    _cache[path] = r
    return r


def metric(scen, arm, key, seeds=None):
    out = {}
    for s, f in fam(scen, arm).items():
        if seeds is None or s in seeds:
            out[s] = run(f, scen, s)[key]
    return out


# ----------------------------------------------------------------------------- statistics
def ci95(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    h = stats.sem(d) * stats.t.ppf(0.975, len(d) - 1)
    return d.mean() - h, d.mean() + h


def paired(a, b, scale=100.0):
    s = sorted(set(a) & set(b))
    if not s:
        return None
    d = np.array([a[x] - b[x] for x in s]) * scale
    lo, hi = ci95(d)
    return dict(mean=float(d.mean()), lo=lo, hi=hi, n_pos=int((d > 0).sum()), n=len(s), seeds=s, per_seed=d)


def msd(v, scale=100.0):
    x = np.array(list(v.values()), float) * scale
    if not len(x):
        return float("nan"), float("nan"), 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), len(x)


def f4(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def fci(p):
    return "" if p is None or math.isnan(p["lo"]) else f"[{p['lo']:+.2f}, {p['hi']:+.2f}]"


def fdiff(p):
    return "" if p is None else f"{p['mean']:+.4f}"


def seedset(d):
    return "{" + ",".join(map(str, sorted(d))) + "}"


def src(scen, arm, seeds):
    return f"{SCEN[scen][1]}_{arm}_s" + seedset(seeds)


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")


PN = []   # paper numbers: item, value, ci95, n_seeds, source runs


def pn(item, p, sources, unit="pp"):
    if p is None:
        return
    PN.append([item, f"{p['mean']:+.3f} {unit}", fci(p), p["n"], "; ".join(sources)])


def best_fixed(scen, arms, seeds):
    """argmax of the seed-mean integrated accuracy over `seeds` (all arms must have all seeds)."""
    best, bv = None, -1
    for a in arms:
        v = metric(scen, a, "integ", seeds)
        if len(v) != len(seeds):
            return None
        m = np.mean(list(v.values()))
        if m > bv:
            best, bv = a, m
    return best


# ============================================================================= T1 scenario definitions
def t1():
    rows = []
    sc = r6_env.SCENARIOS
    common = [("DriftGate", "relonly flags + --no_neighbor_avg: q = max(EMA temporal score, EMA spatial score) of the "
                             "cluster itself, no neighbour average; d-bar shared between edges for the spatial score"),
              ("rounds", "150 (05:00-20:00, 6 min per round)"), ("evaluation rounds", "1, 5, 10, ..., 150 (31)"),
              ("edge coverage radius R", "1.0 km, a client belongs to <= 2 nearest cells within R (else the nearest)"),
              ("rho at home / away", "0.1 / 0.8 (non-Main share 11.5% / 51%)"),
              ("request count", "N ~ Poisson(mu_k), probe n = min(N, 64)"),
              ("model / data", "CIFAR-10 split CNN, nd1 partition, local epochs 3, batch 32, lr 0.01"),
              ("disjoint pools", "controller 20% / evaluation 80% of each class, overlap 0")]
    for k, v in common:
        rows.append(["all", k, v])
    for name in ("S1", "S1fast", "S3_K200", "S3_K500", "S4_lowreq", "S4_part07", "S4_part05", "S4_loss01", "S4_loss03"):
        s = sc[name]
        env, meta = r6_env.load_env(RUNS / "phaseT6_env" / f"{name}_seed0.npz")
        rows.append([name, "K clients / L cells", f"{meta['K']} / {meta['L']}"])
        rows.append([name, "districts (hub + 4 residential cells, 4.5 km apart)", f"{s['grid'][0]} x {s['grid'][1]}"])
        rows.append([name, "speed (m/s)", f"U[{s_range(r6_env.SPEED_RANGE[s['speed']])}]"])
        rows.append([name, "mu_k (requests per round)", f"U[{s_range(r6_env.MU_RANGE[s['mu']])}]"])
        rows.append([name, "participation a", f"{s['avail']}"])
        rows.append([name, "signal loss p (client TV to an edge, d-bar shared between edges; each independent)", f"{s['loss']}"])
    rows.append(["S4_delay1 / S4_delay3", "signal delay d (rounds)", "1 / 3 (S1 environment)"])
    rows.append(["S1", "daily plans (residential)", "0.6 hub work, 0.2 other residential cell, 0.2 stay; hub residents 0.8 stay"])
    rows.append(["S3", "daily plans (residential)", "0.45 own hub, 0.15 nearest other district's hub, 0.2 other residential, 0.2 stay"])
    rows.append(["S1/S3", "commute / return departure", "N(07:45, 30 min) in [06:30, 09:30] / N(17:45, 45 min) in [16:00, 19:30]"])
    rows.append(["S1/S3", "lunch (30% of workers) / errand (30% of stayers)",
                 "N(12:15, 20 min), stay max(15, Exp(45)) / U[10:00, 16:00], stay max(20, Exp(60))"])
    env, meta = r6_env.load_env(RUNS / "phaseT6_env" / "S2_seed0.npz")
    rows += [["S2", "dataset", meta["dataset"]],
             ["S2", "points (all / after filters)", f"{meta['n_points']:,} / {meta['n_points_kept']:,}"],
             ["S2", "candidate users / selected user-days", f"{meta['n_candidates']} / {meta['n_selected']}"],
             ["S2", "edge positions (km, around the mean position)", "; ".join(f"({x:.2f}, {y:.2f})" for x, y in meta["edge_xy_km"])],
             ["S2", "edge positions (lat, lon)", "; ".join(f"({a:.4f}, {b:.4f})" for a, b in meta["edge_latlon"])],
             ["S2", "residents per cell", " ".join(map(str, meta["residents_per_cell"]))],
             ["S2", "membership", "nearest edge, plus the second if its distance <= 1.2 x the nearest"],
             ["S2", "mu_k / participation / loss", f"U[{s_range(meta['mu_range'])}] / 1.0 / 0"]]
    wcsv("T1_scenarios.csv", ["scenario", "parameter", "value"], rows)


def s_range(r):
    return f"{r[0]:g}, {r[1]:g}"


# ============================================================================= T2 main results
def t2():
    rows = []
    # Schedule A from Round 5 (read only)
    r5 = [r for r in csv.DictReader(open(R5_T1)) if r["setting"] == "stepwise composition change"] if R5_T1.exists() else []
    mapname = {"DriftGate": DG, "entropy": ENT, "APFL η=0.01": "apfl001", "APFL η=0.1": "apfl010"}
    for r in r5:
        m = r["method"].replace(" (best fixed)", "")
        arm = mapname.get(m) or ("fixed" + f"{int(round(float(m.split()[1]) * 100)):03d}" if m.startswith("fixed") else None)
        if arm is None:
            continue
        rows.append(["stepwise composition change (Round-5 runs re-run as R0)", ARM_NAME[arm], r["n_seeds"], r["integrated_acc_pct"],
                     r["sd_pct"], r["DriftGate_minus_method_pp"], r["ci95"], r["n_pos"], r["n_matched"],
                     "yes" if "(best fixed)" in r["method"] else "", r["source_runs"]])
    for scen in ("S1", "S2"):
        dg = metric(scen, DG, "integ")
        fx_seeds = sorted(dg)
        bf = best_fixed(scen, FIXED_ALL, fx_seeds)
        for arm in [DG] + FIXED_ALL + [ENT, "apfl001", "apfl010"]:
            v = metric(scen, arm, "integ")
            if not v:
                rows.append([SCEN[scen][2], ARM_NAME[arm], 0, "", "", "", "", "", "", "", "missing"])
                continue
            m, sd, n = msd(v)
            p = None if arm == DG else paired(dg, v)
            rows.append([SCEN[scen][2], ARM_NAME[arm], n, f4(m), f4(sd), fdiff(p), fci(p),
                         "" if p is None else p["n_pos"], "" if p is None else p["n"],
                         "yes" if arm == bf else "", src(scen, arm, v)])
            if p is not None:
                pn(f"{SCEN[scen][2]}: DriftGate − {ARM_NAME[arm]}" + (" (best fixed)" if arm == bf else ""), p,
                   [src(scen, DG, p["seeds"]), src(scen, arm, p["seeds"])])
    wcsv("T2_main_results.csv", ["setting", "method", "n_seeds", "integrated_acc_pct", "sd_pct",
                                 "DriftGate_minus_method_pp", "ci95", "n_pos", "n_matched", "best_fixed", "source_runs"], rows)


# ============================================================================= T3 extra metrics
EXTRA = [("p10", "bottom-10% client accuracy"), ("home", "accuracy at home"), ("away", "accuracy away from home"),
         ("main", "Main-request accuracy"), ("nonmain", "non-Main-request accuracy"), ("offload", "offload rate")]


def t3():
    rows, drows = [], []
    for scen in ("S1", "S2"):
        dgm = {k: metric(scen, DG, k) for k, _ in EXTRA}
        bf = best_fixed(scen, FIXED_ALL, sorted(dgm["p10"]))
        for arm in [DG] + FIXED_ALL + [ENT, "apfl001", "apfl010"]:
            vals = {k: metric(scen, arm, k) for k, _ in EXTRA}
            if not vals["p10"]:
                continue
            row = [SCEN[scen][2], ARM_NAME[arm], len(vals["p10"]), "yes" if arm == bf else ""]
            for k, _ in EXTRA:
                m, sd, _ = msd(vals[k])
                row += [f4(m), f4(sd)]
            rows.append(row)
            if arm == DG:
                continue
            for k, lab in EXTRA:
                p = paired(dgm[k], vals[k])
                drows.append([SCEN[scen][2], ARM_NAME[arm], "yes" if arm == bf else "", lab, fdiff(p), fci(p),
                              "" if p is None else p["n_pos"], "" if p is None else p["n"]])
                if arm == bf and k in ("home", "away", "p10"):
                    pn(f"{SCEN[scen][2]}: {lab}, DriftGate − {ARM_NAME[arm]} (best fixed)", p,
                       [src(scen, DG, p["seeds"]), src(scen, arm, p["seeds"])])
    hdr = ["setting", "method", "n_seeds", "best_fixed"]
    for k, lab in EXTRA:
        hdr += [f"{k}_pct" if k != "offload" else "offload_rate_pct", f"{k}_sd"]
    wcsv("T3a_extra_metrics.csv", hdr, rows)
    wcsv("T3b_extra_metrics_paired.csv", ["setting", "method", "best_fixed", "metric", "DriftGate_minus_method_pp",
                                          "ci95", "n_pos", "n"], drows)


# ============================================================================= T4 time-of-day
def t4():
    rows, crows = [], []
    for scen in ("S1", "S2"):
        dg = {s: run(f, scen, s) for s, f in fam(scen, DG).items()}
        if not dg:
            continue
        er = next(iter(dg.values()))["rounds"]
        nslot = {name: int(sum(lo <= r <= hi for r in er)) for name, _, lo, hi in SLOTS}
        for arm in [DG] + FIXED_ALL + [ENT, "apfl001", "apfl010"]:
            runs_ = {s: run(f, scen, s) for s, f in fam(scen, arm).items()}
            if not runs_:
                continue
            row = [SCEN[scen][2], ARM_NAME[arm], len(runs_)]
            for name, clock, lo, hi in SLOTS:
                row.append(f4(np.mean([r["slot"][name] for r in runs_.values()]) * 100))
            rows.append(row)
            if arm == DG:
                continue
            seeds = sorted(set(dg) & set(runs_))
            if not seeds:
                continue
            total = paired({s: dg[s]["integ"] for s in seeds}, {s: runs_[s]["integ"] for s in seeds})
            parts = []
            for name, clock, lo, hi in SLOTS:
                w = nslot[name] / len(er)
                p = paired({s: dg[s]["slot"][name] * w for s in seeds}, {s: runs_[s]["slot"][name] * w for s in seeds})
                parts.append(p["mean"])
                crows.append([SCEN[scen][2], ARM_NAME[arm], name, clock, nslot[name], fdiff(p), fci(p), p["n_pos"], p["n"]])
            crows.append([SCEN[scen][2], ARM_NAME[arm], "total (sum of slots)", "", len(er), f"{sum(parts):+.4f}",
                          fci(total), total["n_pos"], total["n"]])
            assert abs(sum(parts) - total["mean"]) < 1e-9, (scen, arm, sum(parts), total["mean"])
    wcsv("T4a_time_of_day_accuracy.csv", ["setting", "method", "n_seeds"] +
         [f"{n} {c} acc_pct" for n, c, _, _ in SLOTS], rows)
    wcsv("T4b_time_of_day_contribution.csv", ["setting", "method", "slot", "clock", "eval_rounds_in_slot",
                                              "contribution_to_DriftGate_minus_method_pp", "ci95", "n_pos", "n"], crows)


# ============================================================================= T5 lambda response
def response_times(lam, rho):
    """per cell: (reaction rounds, recovery rounds) following §5.2; None if the event did not happen."""
    T, L = lam.shape
    out = []
    for z in range(L):
        r = rho[:, z]
        react = recov = end = None
        start = next((t for t in range(25, T) if r[t] > 0.3), None)          # round >= 26 (index 25)
        if start is not None:
            hit = next((t for t in range(start, T) if lam[t, z] <= 0.425), None)
            react = None if hit is None else hit - start
            end = next((t for t in range(110, T) if r[t] < 0.3), None)       # round >= 111 (index 110)
            if end is not None:
                hit2 = next((t for t in range(end, T) if lam[t, z] > 0.425), None)
                recov = None if hit2 is None else hit2 - end
        out.append((start, react, recov, end))
    return out


def t5():
    rows, srows, lrows = [], [], []
    for scen in ("S1", "S2"):
        for arm in (DG, ENT):
            for s, f in fam(scen, arm).items():
                r = run(f, scen, s)
                ctype = ["hub" if r["env"]["cell_is_hub"][z] else ("cell " + str(z) if scen == "S2" else "residential")
                         for z in range(r["L"])]
                for z, (start, react, recov, end) in enumerate(response_times(r["lam"], r["rho_cell"])):
                    rows.append([SCEN[scen][2], ARM_NAME[arm], s, z, ctype[z],
                                 "" if start is None else start + 1,
                                 "never above 0.3" if start is None else ("not reached" if react is None else react),
                                 "" if react is None else react * 6,
                                 "" if recov is None else recov, "" if recov is None else recov * 6,
                                 "" if start is None else f"{r['lam'][start, z]:.3f}",
                                 "" if end is None else end + 1,
                                 "" if end is None else f"{r['lam'][end, z]:.3f}",
                                 f"{r['lam'][25:, z].min():.3f}"])
                for name, clock, lo, hi in SLOTS:
                    for typ in sorted(set(ctype)):
                        cells = [z for z in range(r["L"]) if ctype[z] == typ]
                        lrows.append([SCEN[scen][2], ARM_NAME[arm], s, name, clock, typ,
                                      f"{r['lam'][lo - 1:hi, cells].mean():.4f}",
                                      f"{np.nanmean(r['rho_cell'][lo - 1:hi, cells]):.4f}"])
    wcsv("T5a_lambda_response_per_cell.csv", ["setting", "method", "seed", "cell", "cell_type", "change_start_round",
                                              "reaction_rounds", "reaction_min", "recovery_rounds", "recovery_min",
                                              "lambda_at_change_start", "recovery_start_round",
                                              "lambda_at_recovery_start", "min_lambda_after_round_25"], rows)
    # summary by cell type (cells whose rho never exceeded 0.3 after round 25 are not counted)
    by = {}
    for r in rows:
        d = by.setdefault((r[0], r[1], r[4]), {"react": [], "recov": [], "n": 0, "cens": 0})
        if r[6] == "never above 0.3":
            continue
        d["n"] += 1
        if isinstance(r[6], int):
            d["react"].append(r[6])
        else:
            d["cens"] += 1
        if r[8] != "":
            d["recov"].append(r[8])
    for (setting, method, typ), d in sorted(by.items()):
        srows.append([setting, method, typ, d["n"], d["cens"],
                      f"{np.mean(d['react']):.2f}" if d["react"] else "", f"{np.mean(d['react']) * 6:.1f}" if d["react"] else "",
                      f"{np.median(d['react']):.1f}" if d["react"] else "",
                      len(d["recov"]), f"{np.mean(d['recov']):.2f}" if d["recov"] else "",
                      f"{np.mean(d['recov']) * 6:.1f}" if d["recov"] else ""])
        if method == "DriftGate" and d["react"]:
            PN.append([f"{setting}: DriftGate reaction time, {typ} cells", f"{np.mean(d['react']) * 6:.1f} min",
                       "", d["n"], f"{SCEN['S1' if setting == SCEN['S1'][2] else 'S2'][1]}_{DG} (all seeds)"])
            if d["recov"]:
                PN.append([f"{setting}: DriftGate recovery time, {typ} cells", f"{np.mean(d['recov']) * 6:.1f} min",
                           "", len(d["recov"]), "same runs"])
    wcsv("T5b_lambda_response_summary.csv", ["setting", "method", "cell_type", "cells_with_change", "reaction_not_reached",
                                             "reaction_mean_rounds", "reaction_mean_min", "reaction_median_rounds",
                                             "cells_with_recovery", "recovery_mean_rounds", "recovery_mean_min"], srows)
    # lambda per slot and cell type, seed mean
    agg = {}
    for r in lrows:
        agg.setdefault((r[0], r[1], r[3], r[4], r[5]), []).append((float(r[6]), float(r[7])))
    wcsv("T5c_lambda_by_time_and_cell.csv", ["setting", "method", "slot", "clock", "cell_type", "n_seeds",
                                            "mean_lambda", "mean_cell_rho"],
         [[k[0], k[1], k[2], k[3], k[4], len(v), f"{np.mean([a for a, _ in v]):.4f}", f"{np.mean([b for _, b in v]):.4f}"]
          for k, v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], [s[0] for s in SLOTS].index(kv[0][2]), kv[0][4]))])


# ============================================================================= T6-T8 DriftGate vs fixed
def dg_vs_fixed(scen, fixed_arms, seeds, bf_scen=None, bf_arms=None):
    """DriftGate accuracy, fixed accuracies, best fixed (selected over `seeds`) and the paired difference."""
    dg = metric(scen, DG, "integ", seeds)
    fs = bf_scen or scen
    arms = bf_arms or fixed_arms
    fx = {a: metric(fs, a, "integ", seeds) for a in arms}
    bf = best_fixed(fs, arms, seeds) if dg else None
    p = paired(dg, fx[bf]) if bf else None
    return dg, fx, bf, p


def t6():
    rows = []
    for scen in ("S1", "S1fast"):
        dg, fx, bf, p = dg_vs_fixed(scen, FIXED3, [0, 1, 2])
        rows.append([SCEN[scen][2], len(dg), f4(msd(dg)[0])] + [f4(msd(fx[a])[0]) for a in FIXED3] +
                    ["" if bf is None else ARM_NAME[bf], fdiff(p), fci(p), "" if p is None else p["n_pos"],
                     "" if p is None else p["n"]])
        pn(f"{SCEN[scen][2]}: DriftGate − best of fixed 0.2/0.4/0.6", p,
           [src(scen, DG, [0, 1, 2])] + ([src(scen, bf, [0, 1, 2])] if bf else []))
    wcsv("T6_speed.csv", ["setting", "n_seeds", "DriftGate_pct", "fixed_0.2_pct", "fixed_0.4_pct", "fixed_0.6_pct",
                          "best_fixed", "DriftGate_minus_best_pp", "ci95", "n_pos", "n"], rows)


def signal_bytes(env):
    """§5.2 per-round signal bytes (mean over rounds): client TV to each of its edges (4 B per edge) and
    d-bar sharing for the spatial score (4 (L-1) B per edge). The neighbour score exchange was removed
    with the neighbour average (2026-10-01)."""
    m = env["member"]
    tv = 4 * (m >= 0).sum(axis=2).sum(axis=0).mean()
    L = len(env["cell_xy"])
    share = 4 * L * (L - 1)
    return tv, share


def t7():
    rows = []
    for scen, K in (("S1", 50), ("S3_K200", 200), ("S3_K500", 500)):
        dg, fx, bf, p = dg_vs_fixed(scen, FIXED3, [0, 1, 2])
        env = env_of(scen, 0)
        tv, share = signal_bytes(env)
        L = len(env["cell_xy"])
        ns = [run(f, scen, s)["ctrl_ns"] for s, f in fam(scen, DG).items() if s in (0, 1, 2)]
        us = np.concatenate([n[25:].ravel() for n in ns]) / 1e3 if ns else np.array([])
        rows.append([K, L, SCEN[scen][2], len(dg), f4(msd(dg)[0])] + [f4(msd(fx[a])[0]) for a in FIXED3] +
                    ["" if bf is None else ARM_NAME[bf], fdiff(p), fci(p), "" if p is None else p["n_pos"],
                     "" if p is None else p["n"], f"{tv:.0f}", f"{share:.0f}", f"{tv + share:.0f}",
                     f"{(tv + share) / L:.1f}",
                     f"{np.median(us):.1f}" if len(us) else "", f"{np.percentile(us, 95):.1f}" if len(us) else ""])
        pn(f"K={K}: DriftGate − best of fixed 0.2/0.4/0.6", p,
           [src(scen, DG, [0, 1, 2])] + ([src(scen, bf, [0, 1, 2])] if bf else []))
    wcsv("T7_scale.csv", ["K", "L", "setting", "n_seeds", "DriftGate_pct", "fixed_0.2_pct", "fixed_0.4_pct",
                          "fixed_0.6_pct", "best_fixed", "DriftGate_minus_best_pp", "ci95", "n_pos", "n",
                          "signal_bytes_client_TV", "signal_bytes_dbar_sharing",
                          "signal_bytes_total_per_round", "signal_bytes_per_edge", "controller_us_per_edge_round_median",
                          "controller_us_per_edge_round_p95"], rows)


def t8():
    rows = []
    s1dg = metric("S1", DG, "integ", [0, 1, 2])
    for scen in ("S4_loss01", "S4_loss03", "S4_delay1", "S4_delay3", "S4_lowreq", "S4_part07", "S4_part05"):
        part = scen.startswith("S4_part")
        if part:
            dg, fx, bf, p = dg_vs_fixed(scen, FIXED3, [0, 1, 2])
            ref = "same participation, best of fixed 0.2/0.4/0.6"
        else:
            dg, fx, bf, p = dg_vs_fixed(scen, FIXED_ALL, [0, 1, 2], bf_scen="S1", bf_arms=FIXED_ALL)
            ref = "S1 fixed runs (seeds 0-2), best of 0.15-0.6"
        ch = paired(dg, s1dg)
        rows.append([SCEN[scen][2], len(dg), f4(msd(dg)[0]), fdiff(ch), fci(ch), ref,
                     "" if bf is None else ARM_NAME[bf], "" if bf is None else f4(msd(fx[bf])[0]),
                     fdiff(p), fci(p), "" if p is None else p["n_pos"], "" if p is None else p["n"]])
        pn(f"{SCEN[scen][2]}: DriftGate − best fixed ({ref})", p,
           [src(scen, DG, [0, 1, 2])] + ([src("S1" if not part else scen, bf, [0, 1, 2])] if bf else []))
        pn(f"{SCEN[scen][2]}: DriftGate change vs S1 DriftGate", ch, [src(scen, DG, [0, 1, 2]), src("S1", DG, [0, 1, 2])])
    wcsv("T8_robustness.csv", ["condition", "n_seeds", "DriftGate_pct", "change_vs_S1_DriftGate_pp", "ci95_change",
                               "fixed_reference", "best_fixed", "best_fixed_pct", "DriftGate_minus_best_pp", "ci95",
                               "n_pos", "n"], rows)


# ============================================================================= T11 best fixed by cell and time
def t11():
    """User request 2026-10-01 (no new runs): accuracy of every fixed-lambda run by cell group and time slot.
    A client's accuracy at an evaluation round counts for every cell it belongs to at that round
    (Z_k^t, two cells -> both). S1: hub cell vs residential cells (pooled); S2: each cell.
    Per (cell group, slot): mean over the slot's evaluation rounds of the mean client accuracy in the group
    (rounds with no client in the group are skipped), then the seed mean; the best fixed lambda is the
    arm with the highest seed mean; margin = best - runner-up (seed means)."""
    rows, brows = [], []
    for scen in ("S1", "S2"):
        groups = None
        acc = {}
        for arm in FIXED_ALL:
            for sd, f in fam(scen, arm).items():
                r = run(f, scen, sd)
                env = r["env"]
                if groups is None:
                    L = len(env["cell_xy"])
                    groups = ([("hub", [z for z in range(L) if env["cell_is_hub"][z]]),
                               ("residential", [z for z in range(L) if not env["cell_is_hub"][z]])]
                              if scen == "S1" else [(f"cell {z}", [z]) for z in range(L)])
                for gname, cells in groups:
                    for name, clock, lo, hi in SLOTS:
                        vals = []
                        for e, rr in enumerate(r["rounds"]):
                            if not lo <= rr <= hi:
                                continue
                            inc = np.isin(env["member"][:, rr - 1, :], cells).any(axis=1)
                            if inc.any():
                                vals.append(r["acc_ck"][e][inc].mean())
                        if vals:
                            acc.setdefault((gname, name, clock, arm), {})[sd] = float(np.mean(vals))
        if groups is None:
            continue
        for gname, _ in groups:
            for name, clock, _, _ in SLOTS:
                means = {}
                for arm in FIXED_ALL:
                    v = acc.get((gname, name, clock, arm), {})
                    if v:
                        m, sd_, n = msd(v)
                        means[arm] = m
                        rows.append([SCEN[scen][2], gname, name, clock, ARM_NAME[arm], n, f4(m), f4(sd_)])
                if len(means) == len(FIXED_ALL):
                    order = sorted(means, key=lambda a: -means[a])
                    brows.append([SCEN[scen][2], gname, name, clock, ARM_NAME[order[0]], f4(means[order[0]]),
                                  ARM_NAME[order[1]], f"{means[order[0]] - means[order[1]]:.4f}",
                                  ARM_NAME[order[-1]], f"{means[order[0]] - means[order[-1]]:.4f}"])
    wcsv("T11a_fixed_accuracy_by_cell_and_time.csv", ["setting", "cell_group", "slot", "clock", "fixed_lambda",
                                                      "n_seeds", "acc_pct", "sd_pct"], rows)
    summ = []
    for scen in ("S1", "S2"):
        b = [r for r in brows if r[0] == SCEN[scen][2]]
        if not b:
            continue
        for name, clock, _, _ in SLOTS:
            best = sorted({r[4] for r in b if r[2] == name})
            summ.append([SCEN[scen][2], "same time, across cells", f"{name} {clock}", " / ".join(best),
                         "differs" if len(best) > 1 else "same"])
        for g in sorted({r[1] for r in b}):
            best = [r[4] for r in b if r[1] == g]
            summ.append([SCEN[scen][2], "same cell, across time", g, " -> ".join(best),
                         "differs" if len(set(best)) > 1 else "same"])
    wcsv("T11b_best_fixed_by_cell_and_time.csv", ["setting", "cell_group", "slot", "clock", "best_fixed", "best_acc_pct",
                                                  "runner_up", "margin_to_runner_up_pp", "worst_fixed",
                                                  "margin_to_worst_pp"], brows)
    wcsv("T11c_best_fixed_changes.csv", ["setting", "comparison", "where", "best fixed", "verdict"], summ)


# ============================================================================= T10 manifest
def t10():
    rows = []
    qd = RUNS / "queue_r6"
    starts, ends = {}, {}
    for wl in glob.glob(str(qd / "logs" / "worker*.log")):
        for line in open(wl):
            m = re.match(r"\[(\S+) (\S+)\] START (\S+) \((\w+)\) (.+)$", line.strip())
            if m:
                starts[m.group(3)] = (m.group(5), m.group(2))
            m = re.match(r"\[(\S+) (\S+)\] END (\S+) exit=(\d+) (.+)$", line.strip())
            if m:
                ends[m.group(3)] = (m.group(5), m.group(4))
    for line in open(qd / "enqueued_snapshot_v3.txt"):
        cls, cmd = line.split(" ", 1)
        g = lambda k: re.search(rf"--{k} (\S+)", cmd).group(1)
        rn, od, seed = g("run_name"), g("output_dir"), int(g("seed"))
        if cls in ("R0", "R0H"):
            scen, arm, env = "R0 (Round-5 setting)", rn[3:].rsplit("_s", 1)[0], "none (Round-5 schedule)"
        else:
            scen, arm = g("scenario"), g("arm")
            env = f"runs/phaseT6_env/{g('env')}_seed{seed}.npz"
        f = Path(od) / f"{rn}.json"
        if f.exists():
            h = json.load(open(f))
            prov = JR / "provenance" / f"{h['config']['run_id']}.json"
            gpu = json.load(open(prov))["env"].get("gpu", "") if prov.exists() else ""
            want = int(re.search(r"--rounds (\d+)", cmd).group(1)) if "--rounds" in cmd else 150
            ok = len(h["round"]) == want and (cls in ("R0", "R0H") or len(h["eval"]) == 31)
            rows.append([h["config"]["run_id"], rn, scen, arm, seed, 100 + seed, env, cls, "yes" if ok else "no",
                         len(h["round"]), len(h["eval"]), h["probe_eval_overlap"],
                         "yes" if h["probe_eval_overlap"] == 0 else "NO", f"honeynaps, {gpu.replace('NVIDIA GeForce ', '')}, "
                         f"{starts.get(rn, ('', ''))[1].replace('cuda:0/', '')}", ends.get(rn, ("", ""))[1],
                         starts.get(rn, ("", ""))[0], ends.get(rn, ("", ""))[0], f"{h['total_time_sec'] / 60:.1f}"])
        else:
            rows.append(["", rn, scen, arm, seed, 100 + seed, env, cls, "no (queued or running)", "", "", "", "",
                         starts.get(rn, ("", ""))[1], ends.get(rn, ("", ""))[1], starts.get(rn, ("", ""))[0], "", ""])
    wcsv("T10_run_manifest.csv", ["run_id", "run_name", "scenario", "arm", "seed", "model_seed", "env_file", "size",
                                  "complete", "rounds", "eval_rounds", "probe_eval_overlap", "overlap_ok", "device",
                                  "worker_exit_code", "start", "end", "wall_min"], rows)
    return sum(1 for r in rows if r[8] == "yes"), len(rows)


def main():
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    t7()
    t8()
    t11()
    n, total = t10()
    with open(HERE / "paper_numbers_r6.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "ci95", "n_seeds", "source_runs"])
        w.writerows(PN)
    print(f"  [paper_numbers_r6.csv] {len(PN)} rows; complete runs {n}/{total}")


if __name__ == "__main__":
    main()
