"""Round 20 tables (R20_plan.md sections 3-5) from the recompute caches (cache/*.pkl), the Round 15 values stored in them,
the Round 19 caches (fixed-weight grid, post hoc reference only) and the Round 18 A3 table.

Writes tables/R20_checks.csv, R20_T1_accuracy.csv, R20_T1_accuracy_per_seed.csv, R20_T1_request_weighted.csv,
R20_T2_cost_curves.csv, R20_T2_targets.csv, R20_T2_dev_operating_points.csv, R20_T3_correction.csv, R20_T3_a_stats.csv,
R20_T0_reconcile.csv and R20_comparator_sets.csv.
"""
import csv
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

import importlib.util

HERE = Path(__file__).resolve().parent.parent
ART = HERE.parent
_spec = importlib.util.spec_from_file_location("rc", HERE / "scripts" / "r20_recompute.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
r19 = rc.r19
CACHE, TAB = rc.CACHE, HERE / "tables"
TAB.mkdir(exist_ok=True)
R19C = rc.R19C
warnings.simplefilter("ignore")

DAY1 = ["S1", "S2", "S1-fast", "partial participation", "stepwise change", "random mobility", "CIFAR-100", "ResNet-18", "K=200", "K=500"]
REPLAY = ["S1 replay", "S2 replay"]
THREE = ["S1 three-day", "S2 three-day"]
MAIN10 = ["Confidence-based offloading", "Device only", "Raw edge only", "Probability average", "Logit sum", "Lower-entropy exit",
          "Logit-entropy weighting", "Geometric ensemble with early exit", "Label-shift EM", "Learned weight"]
REQUIRED = ["Device only", "Raw edge only", "Confidence-based offloading", "Probability average", "Logit sum",
            "No correction + adaptive w", "Correction + w 0.5", "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only",
            "Correction + product", "DriftGate"]
APPENDIX = ["Lower-entropy exit", "Logit-entropy weighting", "Geometric ensemble with early exit", "Label-shift EM", "Learned weight",
            "Correction + learned weight"]
COST = ["DriftGate", "No correction + adaptive w", "Probability average", "Logit sum", "Raw edge only", "Correction + w 0.5",
        "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only", "Correction + product"]
C_GROUP = ["DriftGate", "Correction + w 0.5", "Correction + w 0.2", "Corrected edge only", "Correction + product"]
FULL0 = ["Raw edge only", "Probability average", "Logit sum", "Lower-entropy exit", "Logit-entropy weighting", "Label-shift EM", "Learned weight"]
FULL1 = FULL0 + ["No correction + adaptive w", "Correction + w 0.5", "Correction + w 0.2", "Corrected edge only", "Correction + learned weight",
                 "Correction + product"]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-9


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")


def fmt(v, d=2):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return "NA"
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} ({v.std(ddof=1):.{d}f})"


def fmtd(v, d=3):
    v = np.asarray(v, float)
    if np.all(np.isnan(v)):
        return "NA"
    return f"{v.mean():+.{d}f} ({v.std(ddof=1):.{d}f})"


_C = {}


def runs(st):
    if st not in _C:
        _C[st] = [pickle.load(open(CACHE / f"{rc.run_name(st, s)}.pkl", "rb")) for s in rc.SETTINGS[st][2]]
    return _C[st]


def rmask(a, wn):
    return np.ones(len(a["er"]), bool) if wn == "full" else a["er"] > 30


def key_of(rule, beta):
    if rule in ("Device only", "Confidence-based offloading"):
        return rule
    if beta == 0.0:
        return "Device only"
    return f"{rule} @ {beta}"


def macro(a, rule, beta, wn):
    if rule == "Geometric ensemble with early exit":
        if beta != 1.0 or "r15_res_full" not in a:
            return np.nan
        return a["r15_res_full" if wn == "full" else "r15_res_gt30"][rule]
    k = key_of(rule, beta)
    if k not in a["cnt"]:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(a["present"], a["cnt"][k] / a["n_ek"], np.nan)
    return float(np.nanmean(np.nanmean(A, axis=1)[rmask(a, wn)]))


def reqw(a, rule, beta, wn):
    k = key_of(rule, beta)
    if rule == "Geometric ensemble with early exit" or k not in a["cnt"]:
        return np.nan
    rm = rmask(a, wn)
    return float(a["cnt"][k][rm].sum() / a["n_ek"][rm].sum())


def vec(st, f, rule, beta, wn):
    return np.array([f(a, rule, beta, wn) for a in runs(st)]) * 100


def offl(st, rule, beta, wn="full"):
    """realized offloading ratio in the window (0 for device only)."""
    out = []
    for a in runs(st):
        if rule == "Device only" or beta == 0.0:
            out.append(0.0)
            continue
        c = a["cost"]["Confidence-based offloading" if rule == "Confidence-based offloading" else beta]
        rm = rmask(a, wn)
        out.append(float(c["srv_round"][rm].sum() / a["n_ek"].sum(1)[rm].sum()))
    return np.array(out)


def conditions():
    for st in DAY1:
        yield "first day", st, "full"
        yield "round > 30", st, "round>30"
    for st in REPLAY:
        yield "frozen replay", st, "full"
    for st in THREE:
        yield "three-day frozen replay", st, "full"


# ------------------------------------------------------------------------------------------------ checks
def checks():
    rows = []
    for st in list(rc.SETTINGS):
        for a in runs(st):
            c = a["checks"]
            ae = c.get("answers_equal_refs", {})
            rows.append([st, a["name"], c.get("r18_macro_maxabs", "NA"), c.get("r18_cost_maxabs", "NA"), c.get("r15_full_maxabs", "NA"),
                         ("all equal" if all(c["r19_counts_equal"].values()) else "MISMATCH") if "r19_counts_equal" in c else "NA",
                         min(ae.values()) if ae else "NA", c.get("DriftGate_equals_refs", "NA"), c.get("NoCorr_equals_refs", "NA"),
                         c.get("refs_w_auto_maxabs", "NA"), c["identity_maxabs"], c["identity_argmax_equal"], c["controller_beta0_offloading"]])
    wcsv("R20_checks.csv", ["setting", "run", "max abs difference of round macro accuracy vs Round 18 (all rules, beta 0.25-1)",
                            "max abs difference of offloading and edge calls vs Round 18", "max abs difference of beta 1 accuracy vs Round 15",
                            "correct counts vs Round 19 (DriftGate, w 0, 0.2, 0.45, 0.5; beta 1 and 0.5)",
                            "min share of identical answers vs Round 16 refs (fixed rules)", "DriftGate answers equal to refs (share)",
                            "no correction + adaptive w answers equal to refs (share)", "max abs difference of DriftGate weight vs refs",
                            "correction vs own-class logit offset: max abs probability difference", "same argmax (share)",
                            "offloading ratio of the controller at beta 0"], rows)
    return rows


# ------------------------------------------------------------------------------------------------ Task 1
def r19_best(st, beta, wn):
    """post hoc best fixed w of the Round 19 grid (seed mean), per seed values (pct)."""
    names = [rc.run_name(st, s) for s in rc.SETTINGS[st][2]]
    R = [pickle.load(open(R19C / f"{n}.pkl", "rb")) for n in names]
    grid = np.array([[r19.metric(r19.acc_matrix(a, beta, f"w={w:.2f}"), rmask(a, wn)) for w in r19.W_GRID] for a in R]) * 100
    j = int(r19.argbest(grid.mean(0)))
    return r19.W_GRID[j], grid[:, j]


def task1():
    rows, per, rw = [], [], []
    for cond, st, wn in conditions():
        for beta in (1.0, 0.5):
            dg = vec(st, macro, "DriftGate", beta, wn)
            rules = list(REQUIRED) + list(APPENDIX)
            if beta == 0.5:
                rules = [r for r in rules if r not in ("Confidence-based offloading", "Logit-entropy weighting", "Label-shift EM", "Learned weight",
                                                       "Correction + learned weight", "Geometric ensemble with early exit")]
            vals = {r: vec(st, macro, r, beta, wn) for r in rules}
            vals = {r: v for r, v in vals.items() if not np.all(np.isnan(v))}
            main_av = [r for r in MAIN10 if r in vals]
            strongest = max(main_av, key=lambda r: vals[r].mean()) if beta == 1.0 else None
            for r, v in vals.items():
                grp = "required" if r in REQUIRED else "appendix"
                bud = ("own threshold (0.8 nats)" if r == "Confidence-based offloading" else
                       "same share as confidence-based" if r == "Geometric ensemble with early exit" else
                       "none (device only)" if r == "Device only" else f"beta {beta:g}")
                src = "Round 15 cache" if r == "Geometric ensemble with early exit" else "R20 recompute (Round 15/18/19 values reproduced)"
                d = v - dg
                o = offl(st, r, beta, wn) if r != "Geometric ensemble with early exit" else np.full(len(v), np.nan)
                lab = r + (" [strongest main-table rule]" if r == strongest else "")
                rows.append([cond, st, f"beta {beta:g}", grp, lab, bud, fmt(v), fmtd(d) if r != "DriftGate" else "", f"{int((d > 0).sum())}/{len(d)}" if r != "DriftGate" else "",
                             fmt(o, 3), src])
                for a, x, dd in zip(runs(st), v, d):
                    per.append([cond, st, f"beta {beta:g}", r, a["seed"], f"{x:.4f}", f"{dd:+.4f}"])
            if st in THREE or rc.SETTINGS[st][3] in ("day1", "replay"):
                if (R19C / f"{rc.run_name(st, rc.SETTINGS[st][2][0])}.pkl").exists():
                    w, v = r19_best(st, beta, wn)
                    d = v - dg
                    rows.append([cond, st, f"beta {beta:g}", "post hoc reference", f"Correction + best fixed w on the evaluation seeds (w = {w:.2f})",
                                 f"beta {beta:g}", fmt(v), fmtd(d), f"{int((d > 0).sum())}/{len(d)}", fmt(offl(st, 'DriftGate', beta, wn), 3),
                                 "Round 19 cache (chosen with evaluation labels)"])
            # request-weighted auxiliary metric
            rv = {r: vec(st, reqw, r, beta, wn) for r in vals}
            rv = {r: v for r, v in rv.items() if not np.all(np.isnan(v))}
            mrank = sorted(rv, key=lambda r: -vals[r].mean())
            rrank = sorted(rv, key=lambda r: -rv[r].mean())
            for r in rv:
                d = rv[r] - rv["DriftGate"]
                rw.append([cond, st, f"beta {beta:g}", r, fmt(vals[r]), fmt(rv[r]), fmtd(d) if r != "DriftGate" else "", mrank.index(r) + 1, rrank.index(r) + 1])
    wcsv("R20_T1_accuracy.csv", ["condition", "setting", "block", "group", "rule", "budget", "accuracy mean (SD)", "minus DriftGate pp mean (SD)",
                                 "seeds above DriftGate", "offloading ratio", "source"], rows)
    wcsv("R20_T1_accuracy_per_seed.csv", ["condition", "setting", "block", "rule", "seed", "accuracy pct", "minus DriftGate pp"], per)
    wcsv("R20_T1_request_weighted.csv", ["condition", "setting", "block", "rule", "macro accuracy (paper metric)", "request-weighted accuracy",
                                         "request-weighted minus DriftGate pp", "rank by macro", "rank by request-weighted"], rw)
    # diagnostic for rank changes: share of requests and device-rounds away from home
    dg_rows = []
    for cond, st, wn in conditions():
        sh = []
        for a in runs(st):
            rm = rmask(a, wn)
            pres = a["present"][rm]
            away = (a["home"][rm] == 0) & pres
            if not (a["home"][rm] >= 0).any():
                sh.append([np.nan] * 3)
                continue
            n = a["n_ek"][rm]
            sh.append([away.sum() / pres.sum(), n[away].sum() / n[pres].sum(), n[away].mean() / n[pres & ~away].mean()])
        sh = np.array(sh)
        dg_rows.append([cond, st, fmt(sh[:, 0], 3), fmt(sh[:, 1], 3), fmt(sh[:, 2], 2)])
    wcsv("R20_T1_away_weight.csv", ["condition", "setting", "away share of device-rounds", "away share of requests",
                                    "requests per away device-round / per home device-round"], dg_rows)


# ------------------------------------------------------------------------------------------------ Task 2
def curve(st, m):
    acc = np.array([vec(st, macro, m, b, "full") for b in GRID])          # [grid, seeds]
    srv = np.array([[0.0 if b == 0 else a["cost"][b]["srv"] for a in runs(st)] for b in GRID])
    calls = np.array([[0.0 if b == 0 else a["cost"][b]["calls"] for a in runs(st)] for b in GRID])
    two = np.array([[np.nan if b == 0 else a["cost"][b]["two_share"] for a in runs(st)] for b in GRID])
    return acc, srv, calls, two


def targets(st):
    """A and B from the Round 15 full-offload values; where a run has no Round 15 cache (S1 development seeds 6 and 7),
    only the rules recomputed by R20 are used (logit-entropy weighting, label-shift EM and the two learned weights drop out)."""
    def full(r):
        if all("r15_res_full" in a for a in runs(st)):
            return np.array([a["r15_res_full"][r] for a in runs(st)]) * 100
        return vec(st, macro, r, 1.0, "full")
    A = {r: full(r).mean() for r in FULL0}
    B = {r: full(r).mean() for r in FULL1}
    A = {r: v for r, v in A.items() if np.isfinite(v)}
    B = {r: v for r, v in B.items() if np.isfinite(v)}
    ra, rb = max(A, key=A.get), max(B, key=B.get)
    cmax = {m: curve(st, m)[0].mean(1).max() for m in C_GROUP}
    rcm = min(cmax, key=cmax.get)
    return {"A": (A[ra], ra), "B": (B[rb], rb), "C": (cmax[rcm], f"lowest best-over-grid accuracy among the corrected group ({rcm})")}


def reach(acc_m, srv_m, calls_m, T):
    j = next((i for i in range(len(GRID)) if acc_m[i] >= T - EPS), None)
    if j is None:
        return None
    if j == 0:
        return j, 0.0, 0.0, 0.0
    x0, y0, x1, y1 = srv_m[j - 1], acc_m[j - 1], srv_m[j], acc_m[j]
    xi = x0 + (T - y0) / (y1 - y0) * (x1 - x0) if y1 > y0 else x1
    return j, srv_m[j], calls_m[j], xi


def task2():
    cur, tg = [], []
    a3 = {(r["setting"], r["full-offload comparison set"][:5]): r for r in csv.DictReader(open(ART / "driftgate_tmc_r18_compare" / "tables" / "R18_A3_offload_claims.csv"))}
    for st in DAY1 + REPLAY + ["S1 development"]:
        for m in COST:
            acc, srv, calls, two = curve(st, m)
            for i, b in enumerate(GRID):
                cur.append([st, m, b, fmt(acc[i]), f"{srv[i].mean():.3f}", f"{calls[i].mean():.3f}", "NA" if b == 0 else f"{np.nanmean(two[i]):.3f}"])
        T = targets(st)
        for tn, (tv, trule) in T.items():
            for m in COST:
                acc, srv, calls, _ = curve(st, m)
                r = reach(acc.mean(1), srv.mean(1), calls.mean(1), tv)
                if r is None:
                    tg.append([st, tn, f"{tv:.2f}", trule, m, "NA (not reached)", "", "", "", f"{acc.mean(1).max():.2f}"])
                else:
                    j, s_, c_, xi = r
                    tg.append([st, tn, f"{tv:.2f}", trule, m, f"{GRID[j]:g}", f"{s_:.3f}", f"{c_:.3f}", f"{xi:.3f} (interpolated)" if j > 0 else "0.000",
                               f"{acc.mean(1).max():.2f}"])
            key = {"A": "Round", "B": "with "}.get(tn)
            if key and (st, key) in a3:
                x = a3[(st, key)]
                tg.append([st, tn, x["its accuracy"], x["best full-offload rule"], "DriftGate-P, fixed-threshold sweep (Round 18 A3, Round 15 thresholds)", "",
                           x["DriftGate-P offloading at first grid point reaching it"], x["edge calls per request at that point"], "", ""])
    wcsv("R20_T2_cost_curves.csv", ["setting", "rule", "beta (0 = device only)", "accuracy mean (SD)", "offloading ratio", "edge calls per request",
                                    "two-cell share of offloaded requests"], cur)
    wcsv("R20_T2_targets.csv", ["setting", "target", "target accuracy", "target rule", "rule", "first grid beta reaching it", "offloading ratio there",
                                "edge calls per request there", "offloading ratio interpolated between grid points", "best accuracy of the rule over the grid"], tg)
    # S1 operating points from the development seeds
    Td, Tt = targets("S1 development"), targets("S1")
    rows = []
    for tn in ("A", "B", "C"):
        for m in COST:
            acc_d = curve("S1 development", m)[0].mean(1)
            j = next((i for i in range(len(GRID)) if acc_d[i] >= Td[tn][0] - EPS), None)
            if j is None:
                rows.append([tn, f"{Td[tn][0]:.2f} ({Td[tn][1]})", m, "NA (not reached on development)", "", "", "", f"{Tt[tn][0]:.2f}", ""])
                continue
            acc, srv, calls, _ = curve("S1", m)
            rows.append([tn, f"{Td[tn][0]:.2f} ({Td[tn][1]})", m, f"{GRID[j]:g}", fmt(acc[j]), f"{srv[j].mean():.3f}", f"{calls[j].mean():.3f}",
                         f"{Tt[tn][0]:.2f} ({Tt[tn][1]})", "yes" if acc[j].mean() >= Tt[tn][0] - EPS else "no"])
    wcsv("R20_T2_dev_operating_points.csv", ["target", "development target accuracy (rule)", "rule", "beta chosen on development seeds 5-7",
                                             "evaluation accuracy at that beta, seeds 0-4", "offloading ratio", "edge calls per request",
                                             "evaluation target (same definition)", "evaluation target reached"], rows)


# ------------------------------------------------------------------------------------------------ Task 3
def task3():
    a0 = json.loads((CACHE / "a0.json").read_text())
    rows = []
    fam = {"DriftGate": "No correction + adaptive w", "Corrected edge only": "Raw edge only", "Correction + w 0.5": "Probability average"}
    for st in rc.T3:
        prim = "a0" if st.startswith("S1") else "a0.25"
        for beta in (1.0, 0.5):
            for r, r0 in fam.items():
                cur = vec(st, macro, r, beta, "full")
                b0 = vec(st, macro, r0, beta, "full")
                ca = vec(st, macro, f"{r} [a0]", beta, "full")
                cq = vec(st, macro, f"{r} [a0.25]", beta, "full")
                cp = ca if prim == "a0" else cq
                d_corr, d_cell = cur - b0, cur - cp
                verdict = "yes" if (d_cell.mean() >= 0.10 and (d_cell > 0).all()) else "no"
                rows.append([st, f"{beta:g}", r, fmt(b0), fmt(cq), fmt(ca), fmt(cur), fmtd(d_corr), f"{int((d_corr > 0).sum())}/{len(d_corr)}",
                             "a0 = %.4f" % a0["a0"] if prim == "a0" else "a = 0.25", fmtd(d_cell), f"{int((d_cell > 0).sum())}/{len(d_cell)}",
                             fmtd(cur - (cq if prim == "a0" else ca)), verdict])
    wcsv("R20_T3_correction.csv", ["setting", "beta", "rule", "b = 0 (no correction)", "constant a = 0.25", f"constant a0 = {a0['a0']:.4f} (S1 development)",
                                   "current a_k (DriftGate)", "current a_k minus b = 0 pp", "seeds positive (vs b = 0)", "primary constant",
                                   "current a_k minus primary constant pp", "seeds positive (vs primary constant)", "current a_k minus the other constant pp",
                                   "current a_k contributes by the plan rule (>= +0.10 pp, all seeds positive)"], rows)
    st_rows = []
    for st in rc.T3 + ["S1 development"] + [s for s in DAY1 if s not in rc.T3]:
        A = np.concatenate([a["a_ek"][a["present"]] for a in runs(st)])
        Bk = np.log((1 - A) / A)
        within, between, mx = [], [], []
        hm, aw = [], []
        for a in runs(st):
            P = a["present"]
            X = np.where(P, a["a_ek"], np.nan)
            n = P.sum(0)
            sd = np.nanstd(X, axis=0, ddof=1)[n >= 2]
            within.append(np.nanmean(sd))
            mx.append(np.nanmax(sd))
            between.append(np.nanstd(np.nanmean(X, axis=0), ddof=1))
            hm.append(np.median(a["a_ek"][P & (a["home"] == 1)]) if (a["home"] == 1).any() else np.nan)
            aw.append(np.median(a["a_ek"][P & (a["home"] == 0)]) if (a["home"] == 0).any() else np.nan)
        q = np.percentile(A, [5, 25, 50, 75, 95])
        qb = np.percentile(Bk, [5, 25, 50, 75, 95])
        st_rows.append([st, len(A)] + [f"{x:.4f}" for x in q] + [f"{x:.3f}" for x in qb] +
                       [f"{np.mean(within):.4f}", f"{np.mean(mx):.4f}", f"{np.mean(between):.4f}", f"{np.nanmean(hm):.4f}", f"{np.nanmean(aw):.4f}",
                        f"{max(a['checks']['identity_maxabs'] for a in runs(st)):.2e}", f"{min(a['checks']['identity_argmax_equal'] for a in runs(st)):.6f}"])
    wcsv("R20_T3_a_stats.csv", ["setting", "present device-rounds (all seeds)", "a_k p5", "a_k p25", "a_k p50", "a_k p75", "a_k p95", "b_k p5", "b_k p25", "b_k p50", "b_k p75", "b_k p95",
                                "within-device SD of a_k over rounds (mean over devices)", "within-device SD (max over devices)",
                                "between-device SD of device-mean a_k", "median a_k at home", "median a_k away",
                                "correction vs logit offset: max abs probability difference", "same argmax (share)"], st_rows)


# ------------------------------------------------------------------------------------------------ Task 0
def task0():
    rows = []
    # (1) K=500: main/scale table probability average vs R18 best uncorrected rule
    for st in ("K=500", "K=200", "ResNet-18"):
        for r in ("Probability average", "Logit sum", "No correction + adaptive w", "DriftGate"):
            v = vec(st, macro, r, 1.0, "full")
            rows.append(["best uncorrected value by rule set", st, r, "first day, beta 1, seeds " + " ".join(str(s) for s in rc.SETTINGS[st][2]), fmt(v),
                         fmtd(vec(st, macro, "DriftGate", 1.0, "full") - v) if r != "DriftGate" else ""])
    # (3) the +0.60..+1.88 range: every first-day setting and beta
    for st in DAY1:
        for beta in (0.5, 1.0):
            unc = {r: vec(st, macro, r, beta, "full") for r in ("Raw edge only", "Probability average", "Logit sum", "No correction + adaptive w")}
            best = max(unc, key=lambda r: unc[r].mean())
            dg = vec(st, macro, "DriftGate", beta, "full")
            rows.append(["DriftGate minus best uncorrected rule (Round 18 group of 4)", st, best, f"first day, beta {beta:g}", fmt(unc[best]), fmtd(dg - unc[best])])
    # (4) S1 decomposition: window means and weighted contributions
    for st in ("S1", "S2"):
        for beta in (1.0, 0.5):
            parts = []
            for a in runs(st):
                er = a["er"]
                f = lambda r: np.array([np.nanmean(np.where(a["present"][e], a["cnt"][key_of(r, beta)][e] / np.maximum(a["n_ek"][e], 1), np.nan))
                                        for e in range(a["E"])])
                d = (f("DriftGate") - f("Corrected edge only")) * 100
                parts.append([d.mean(), d[er <= 30].mean(), d[er > 30].mean(), d[er <= 30].sum() / len(er), d[er > 30].sum() / len(er), (er <= 30).sum(), len(er)])
            P = np.array(parts)
            for i, lab in enumerate(["full window mean", "round <= 30 window mean", "round > 30 window mean", "round <= 30 contribution to full mean",
                                     "round > 30 contribution to full mean"]):
                rows.append(["DriftGate minus corrected edge only", st, lab, f"beta {beta:g}, {int(P[0, 5])} of {int(P[0, 6])} evaluation rounds are <= 30",
                             fmtd(P[:, i]), ""])
    # (2) original first-day runs against the retrained first-day runs of Round 15 (Round 15 cache, not recomputed here)
    m15, r16 = rc.m15, rc.r16
    for st, key in (("S1", "S1 day 1 (retrained)"), ("S2", "S2 day 1 (retrained)")):
        rd_, pat, seeds = m15.SETTINGS[key]
        C = [r16.r15_cache_value(str(rd_), pat.format(s)) for s in seeds]
        for r, k15 in (("DriftGate", "DriftGate"), ("Probability average", "B3"), ("Confidence-based offloading", "B0")):
            rows.append(["original first-day runs vs retrained first-day runs (Round 15 cache)", st, r, "Round 12 runs, R20 recompute",
                         fmt(vec(st, macro, r, 1.0, "full")), ""])
            rows.append(["original first-day runs vs retrained first-day runs (Round 15 cache)", st, r, "Round 15 retrained runs (" + pat.format("*") + ")",
                         fmt(np.array([c["res"][k15]["full"] for c in C]) * 100), ""])
    # (3b) S1 replay seed sets
    v = vec("S1 replay", macro, "DriftGate", 1.0, "full")
    rows.append(["seed sets", "S1 replay", "DriftGate", "seeds 0-4", fmt(v), ""])
    rows.append(["seed sets", "S1 replay", "DriftGate", "seeds 1-4 (Round 17 evaluation seeds)", fmt(v[1:]), ""])
    # (5) manuscript sentence on the beta 0.5 controller
    for st in ("S1", "S2", "S1 replay", "S2 replay"):
        for r in ("DriftGate", "Logit sum", "Probability average", "Raw edge only", "Corrected edge only", "Correction + w 0.2"):
            rows.append(["beta 0.5 controller values", st, r, "beta 0.5", fmt(vec(st, macro, r, 0.5, "full")),
                         fmtd(vec(st, macro, "DriftGate", 0.5, "full") - vec(st, macro, r, 0.5, "full")) if r != "DriftGate" else ""])
    wcsv("R20_T0_reconcile.csv", ["item", "setting", "rule or part", "condition", "value", "DriftGate minus it pp / note"], rows)


def comparator_sets():
    # rule: correction, weight, offloading at full budget, then membership flags
    R = [("Confidence-based offloading", "none", "one exit per request", "own threshold 0.8 nats (about 0.92 in S1)"),
         ("Device only", "none", "device exit", "no offloading"),
         ("Edge only (raw edge only)", "none", "edge exit", "all requests"),
         ("Probability average", "none", "fixed w 0.5", "all requests"),
         ("Logit sum", "none", "product of probabilities", "all requests"),
         ("Lower-entropy exit", "none", "one exit per request (lower entropy)", "all requests"),
         ("Logit-entropy weighting", "none", "per-request weight minimizing mixture entropy", "all requests"),
         ("Geometric ensemble with early exit", "none", "geometric mean", "same share as confidence-based"),
         ("Label-shift EM", "own correction (EM estimate of the share outside M_k)", "edge exit only", "all requests"),
         ("Learned weight", "none", "per-request logistic weight trained on development labels", "all requests"),
         ("DriftGate", "prior correction r 0.5", "adaptive entropy weight", "all requests (beta 1) or controller"),
         ("No correction + adaptive w", "none", "adaptive entropy weight", "all requests"),
         ("Correction + w 0.5", "prior correction r 0.5", "fixed w 0.5", "all requests"),
         ("Correction + w 0.2 (development, Round 11)", "prior correction r 0.5", "fixed w 0.2", "all requests"),
         ("Corrected edge only", "prior correction r 0.5", "fixed w 0", "all requests"),
         ("Correction + learned weight", "prior correction r 0.5", "per-request logistic weight", "all requests"),
         ("Correction + product", "prior correction r 0.5", "product of probabilities", "all requests"),
         ("Correction + w 0.45 (development, Round 19)", "prior correction r 0.5", "fixed w 0.45", "all requests"),
         ("Correction + fixed w grid 0, 0.05, ..., 1 (Round 19)", "prior correction r 0.5", "21 fixed weights", "all requests or controller")]
    S = {
        "manuscript main table (11 rules)": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        "manuscript ablation table (8 variants)": {10, 11, 3, 12, 13, 14, 15, 16},
        "manuscript offloading sentence / Round 18 A3 Round 15 set (7 full-offload rules)": {2, 3, 4, 5, 6, 8, 9},
        "Round 18 A3 set with corrected variants (13 rules)": {2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16},
        "Round 17 strongest other (11 rules)": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14},
        "Round 16-19 reference set (17 rules)": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        "Round 18 same-budget uncorrected group (4)": {2, 11, 3, 4},
        "Round 18 same-budget corrected group (5)": {12, 13, 14, 16, 15},
        "Round 18 strongest reference for the conditional rule (9)": {2, 3, 4, 12, 13, 14, 16, 11, 15},
        "Round 19 fixed-weight comparison": {10, 13, 14, 12, 17, 18},
        "Round 20 cost methods (10)": {10, 11, 3, 4, 2, 12, 13, 17, 14, 16},
        "Round 20 target C group (5)": {10, 12, 13, 14, 16},
    }
    rows = []
    for i, (name, corr, weight, budget) in enumerate(R):
        rows.append([name, corr, weight, budget] + ["yes" if i in s else "" for s in S.values()])
    wcsv("R20_comparator_sets.csv", ["rule", "edge correction", "combination or weight", "offloading in the full-offload comparison"] + list(S), rows)


# ------------------------------------------------------------------------------------------------ summaries for the report
CORR_DEPLOY = ["Correction + w 0.5", "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only", "Correction + product", "Correction + learned weight"]


def summaries():
    rows = []
    for cond, st, wn in conditions():
        for beta in (1.0, 0.5):
            dg = vec(st, macro, "DriftGate", beta, wn)
            main = [r for r in MAIN10 if not np.all(np.isnan(vec(st, macro, r, beta, wn)))]
            sm = max(main, key=lambda r: vec(st, macro, r, beta, wn).mean()) if beta == 1.0 else None
            cands = [r for r in CORR_DEPLOY if not np.all(np.isnan(vec(st, macro, r, beta, wn)))]
            bc = max(cands, key=lambda r: vec(st, macro, r, beta, wn).mean())
            allr = [r for r in REQUIRED + APPENDIX if not np.all(np.isnan(vec(st, macro, r, beta, wn)))]
            rank = 1 + sum(vec(st, macro, r, beta, wn).mean() > dg.mean() for r in allr if r != "DriftGate")
            d = lambda r: fmtd(dg - vec(st, macro, r, beta, wn))
            rows.append([cond, st, f"{beta:g}", fmt(dg), sm or "", d(sm) if sm else "", bc, d(bc), d("Corrected edge only"), d("Probability average"),
                         d("No correction + adaptive w"), f"{rank}/{len(allr)}"])
    wcsv("R20_T1_summary.csv", ["condition", "setting", "beta", "DriftGate", "strongest main-table rule (beta 1)", "DriftGate minus strongest main-table rule pp",
                                "best corrected deployable variant", "DriftGate minus best corrected deployable variant pp", "DriftGate minus corrected edge only pp",
                                "DriftGate minus probability average pp", "DriftGate minus no correction + adaptive w pp",
                                "DriftGate rank among the rules of this block"], rows)
    gap = []
    for st in DAY1 + REPLAY:
        line = [st]
        for b in GRID[1:]:
            dg = vec(st, macro, "DriftGate", b, "full")
            best = max(["Correction + w 0.5", "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only", "Correction + product"],
                       key=lambda r: vec(st, macro, r, b, "full").mean())
            unc = max(["Raw edge only", "Probability average", "Logit sum", "No correction + adaptive w"], key=lambda r: vec(st, macro, r, b, "full").mean())
            line += [best, fmtd(dg - vec(st, macro, best, b, "full")), unc, fmtd(dg - vec(st, macro, unc, b, "full"))]
        line += [f"{runs(st)[0]['cost'][0.5]['srv']:.3f}", f"{np.mean([a['cost'][b_]['calls'] for a in runs(st)]):.3f}" if False else
                 " / ".join(f"{np.mean([a['cost'][b_]['calls'] for a in runs(st)]):.3f}" for b_ in GRID[1:]),
                 f"{np.mean([a['cost'][1.0]['two_share'] for a in runs(st)]):.3f}"]
        gap.append(line)
    hdr = ["setting"]
    for b in GRID[1:]:
        hdr += [f"best corrected fixed rule at beta {b:g}", f"DriftGate minus best corrected fixed rule pp (beta {b:g})",
                f"best rule without prior correction at beta {b:g}", f"DriftGate minus best rule without prior correction pp (beta {b:g})"]
    hdr += ["offloading ratio at beta 0.5 (seed 0)", "edge calls per request at beta 0.25 / 0.5 / 0.75 / 1", "two-cell share of offloaded requests"]
    wcsv("R20_T2_same_budget_gap.csv", hdr, gap)


def main():
    checks()
    comparator_sets()
    task0()
    task1()
    task2()
    task3()
    summaries()


if __name__ == "__main__":
    main()
