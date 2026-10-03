"""Round 12: generality of the two-exit fusion F (w = 0.2, r = 0.5) and the lambda adjustment on top of it.

Inputs (runs/phaseT12_fusion/r12_<base run>): run JSON, *_evalprobs.npz (both exits' probabilities per evaluated
request) and the round records (Round 6 path: *_trace.npz with train_hist_cell, train_hist_all, cells, at_home;
Round 5 path: *_rec.npz with train_hist_cell, train_hist_all, eval_cells). Outputs: tables/*.csv, decision.json,
figures/*.

M_k: the classes of the client's Main requests (kind 0), identical in every evaluation round (checked); for the Round 6
path also compared with the provenance record. pi_tr = 0.5 h_cell + 0.5 h_all from the label counts of the training
round of the evaluation round (two cells: mean); a = clip(pi_tr(M_k), 0.01, 0.99), b = 1 - a.
Rules: B0 entropy routing tau 0.8; B1 client exit; B2 server exit; B3 argmax (p_c + p_s)/2;
F argmax [0.2 p_c + 0.8 p'_s], p'_s(c) ~ p_s(c) (1 - r)/a on M_k and p_s(c) r/b on O_k with r = 0.5;
kind oracle: Main -> client exit, otherwise server exit.
Curves: E (entropy routing) and SF (client exit when entropy <= tau, otherwise F), tau = 0, 0.05, ..., 2.30 and
infinity; server use = share of requests whose entropy exceeds tau; accuracy at server use 0.3, 0.5, 0.7, 0.9, 1.0
by linear interpolation between neighbouring grid points.
Accuracy: per evaluation round the mean over clients, then the mean over the rounds of the window; full = all
evaluation rounds, late = evaluation rounds from training round 30 on.
Decisions (late window, seed means; directive 4): (1) fusion as the central method: S1 F - B0 >= 2.0 pp and F - B0 > 0
in S2, S1-fast, K=200, K=500 and participation 0.5 (fixed lambda 0.4 runs); (2) keep the lambda adjustment: with F,
DriftGate - fixed 0.4 >= 0.5 pp in Schedule A and client mobility and >= -0.3 pp in S1.
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
RUNS = Path(os.environ.get("R12_RUNS", JR / "runs" / "phaseT12_fusion"))     # overrides only for smoke tests
OUT = Path(os.environ.get("R12_OUT", HERE))
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JR))

W, RSTR, TAU0 = 0.2, 0.5, np.float32(0.8)
LAMBDA_BIG, A_CLIP = 0.5, (0.01, 0.99)
LATE_FROM = int(os.environ.get("R12_LATE_FROM", 30))   # override only for smoke tests
TAU = np.round(np.arange(0, 2.30 + 1e-9, 0.05), 2).astype(np.float32)
OPTS = [0.3, 0.5, 0.7, 0.9, 1.0]
RULES = ["B0", "B1", "B2", "B3", "F", "kind oracle"]
KINDS = ["Main", "OOP", "OOR"]
INDIVIDUAL_OTHERS = ["S2", "S1-fast", "K=200", "K=500", "participation 0.5"]


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def interp(R, A, os_):
    """R non-increasing along the points; value at each o (largest bracketing pair), nan outside."""
    hi, lo, ah, al = R[:-1], R[1:], A[:-1], A[1:]
    O = np.asarray(os_)[None, :]
    inside = (lo[:, None] <= O + 1e-12) & (O <= hi[:, None] + 1e-12)
    span = (hi - lo)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.clip(np.where(span > 0, (O - lo[:, None]) / np.where(span > 0, span, 1), 1.0), 0, 1)
    v = np.where(span > 0, w * ah[:, None] + (1 - w) * al[:, None], np.maximum(ah, al)[:, None])
    v = np.where(inside, v, -np.inf).max(axis=0)
    v[np.isinf(v)] = np.nan
    return v


def analyse_run(job):
    name = job["name"]
    h = json.load(open(RUNS / f"{name}.json"))
    src = RUNS / f"{name}_evalprobs.npz"
    q = np.load(src)
    r6 = (RUNS / f"{name}_trace.npz").exists()
    if r6:
        z = np.load(RUNS / f"{name}_trace.npz")
        hc, ha = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float)
        er = np.array(q["eval_rounds"])
        cells_e = z["cells"][er - 1]                      # [E, K, 2]
    else:
        z = np.load(RUNS / f"{name}_rec.npz")
        hc, ha = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float)
        er = np.array(z["eval_rounds"])
        assert np.array_equal(er, q["eval_rounds"])
        cells_e = z["eval_cells"]                         # [E, clients, 2] in client order
    e_i = q["req_eval_index"].astype(np.int64)
    k_i = q["req_client"].astype(np.int64)
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"].astype(np.int64)
    C = q["pc"].shape[1]
    K = int(k_i.max()) + 1
    E = len(er)
    # ---- M_k from the Main requests, identical in every round
    M = np.zeros((K, C), bool)
    M[k_i[kind == 0], y[kind == 0]] = True
    mk_consistent = bool(np.array_equal(kind == 0, M[k_i, y]))
    mk_prov = None
    prov = JR / "provenance" / f"{h['config']['run_id']}.json"
    if prov.exists():
        pm = json.load(open(prov)).get("client_main")
        if pm:
            Mp = np.zeros_like(M)
            for kk, v in pm.items():
                if int(kk) < K:
                    Mp[int(kk), list(v)] = True
            mk_prov = bool(np.array_equal(M, Mp))
    # ---- a per (round, client)
    if not r6:   # cells_e is in client order of the run (client ids 0..K-1 when no client is empty)
        cid = np.array(z["client_ids"]) if "client_ids" in z.files else np.arange(cells_e.shape[1])
        tmp = np.full((E, K, 2), -1, np.int64)
        tmp[:, cid, :] = cells_e
        cells_e = tmp
    a_ek = np.full((E, K), 0.5)
    for e, r in enumerate(er):
        t = int(r) - 1
        hall = ha[t] / ha[t].sum()
        for k in range(K):
            pis = [LAMBDA_BIG * (hc[t, int(zc)] / hc[t, int(zc)].sum() if hc[t, int(zc)].sum() > 0 else hall)
                   + (1 - LAMBDA_BIG) * hall for zc in cells_e[e, k] if zc >= 0]
            if pis:
                a_ek[e, k] = np.clip(np.mean(pis, axis=0)[M[k]].sum(), *A_CLIP)
    pc = q["pc"].astype(np.float32)
    ps = q["ps"].astype(np.float32)
    cp, sp, ent = q["cp"].astype(np.int64), q["sp"].astype(np.int64), q["ent"]
    Mrow = M[k_i]
    a_r = a_ek[e_i, k_i].astype(np.float32)[:, None]
    wgt = np.where(Mrow, (1 - RSTR) / a_r, RSTR / (1 - a_r)).astype(np.float32)
    psr = ps * wgt
    psr /= psr.sum(1, keepdims=True)
    f_ans = (W * pc + (1 - W) * psr).argmax(1)
    ans = {"B0": np.where(ent > TAU0, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1), "F": f_ans,
           "kind oracle": np.where(kind == 0, cp, sp)}
    correct = {rl: ans[rl] == y for rl in RULES}
    g = e_i * K + k_i
    n_ek = np.bincount(g, minlength=E * K).reshape(E, K).astype(float)
    present = n_ek > 0
    home = np.full(E * K, -1, np.int8)
    home[g] = q["req_home"].astype(np.int8)
    home = home.reshape(E, K)
    has_home = bool((home >= 0).any())
    late = er >= LATE_FROM

    def per_ek(c):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(g, weights=c, minlength=E * K).reshape(E, K) / n_ek

    def acc_w(a_ek_, m):
        return float(np.mean([np.nanmean(np.where(present[e], a_ek_[e], np.nan)) for e in np.flatnonzero(m)]))

    res = {}
    for rl in RULES:
        a_ = per_ek(correct[rl])
        out = dict(full=acc_w(a_, np.ones(E, bool)), late=acc_w(a_, late), per_round=np.array(
            [np.nanmean(np.where(present[e], a_[e], np.nan)) for e in range(E)]))
        if has_home:
            out["home"] = float(np.mean([a_[e][home[e] == 1].mean() for e in np.flatnonzero(late) if (home[e] == 1).any()]))
            out["away"] = float(np.mean([a_[e][home[e] == 0].mean() for e in np.flatnonzero(late) if (home[e] == 0).any()]))
        for i, kn in enumerate(KINDS):
            sel = kind == i
            ck = np.bincount(g, weights=correct[rl] & sel, minlength=E * K).reshape(E, K)
            nk = np.bincount(g, weights=sel, minlength=E * K).reshape(E, K)
            with np.errstate(invalid="ignore", divide="ignore"):
                ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
            vals = [np.nanmean(ak[e]) for e in np.flatnonzero(late) if np.isfinite(ak[e]).any()]
            out[kn] = float(np.mean(vals)) if vals else np.nan
        res[rl] = out
    inl = late[e_i]
    res["B0"]["server_late"] = float((ent[inl] > TAU0).mean())
    res["B0"]["server_full"] = float((ent > TAU0).mean())
    # ---- curves E and SF
    j0 = np.searchsorted(TAU, ent, side="left")          # client answers at tau_j  <=>  j >= j0 (j = len(TAU): inf)
    NJ = len(TAU) + 1
    curves = {}
    for cname, srv_ans in (("E", sp), ("SF", f_ans)):
        cc = (cp == y).astype(float)
        sc = (srv_ans == y).astype(float)
        out = {}
        for wname, m in (("full", np.ones(E, bool)), ("late", late)):
            sel = m[e_i]
            npres = present.sum(1)   # weight = 1 / (requests of the client-round x clients in the round x rounds)
            wv = np.where(sel, 1.0 / (n_ek[e_i, k_i] * npres[e_i] * m.sum()), 0.0)
            idx = j0
            diff = np.bincount(idx, weights=wv * (cc - sc), minlength=NJ + 1)[:NJ]
            accj = (wv * sc).sum() + np.cumsum(diff)
            cnt = np.bincount(idx, weights=sel.astype(float), minlength=NJ + 1)[:NJ]
            srvj = 1.0 - np.cumsum(cnt) / sel.sum()
            out[wname] = dict(acc=accj, srv=srvj, at=interp(srvj, accj, OPTS))
        curves[cname] = out
    tm = h.get("r12_timing_sec", {})
    return dict(job=job, res=res, curves=curves, has_home=has_home, mk_consistent=mk_consistent, mk_prov=mk_prov,
                sha=sha256(src), bytes=src.stat().st_size, run_id=h["config"]["run_id"], n_req=len(y), C=C, K=K,
                er=er, timing=tm, total_min=h["total_time_sec"] / 60,
                json_integrated=float(np.mean([e_["acc_total"] for e_ in h["eval"]])),
                mismatch=h.get("r12_eval_record_mismatch"))


def fusion_cpu_timing():
    """CPU time of the fusion F for one request: prior correction + mixing + argmax (numpy, float32)."""
    rng = np.random.default_rng(0)
    rows = []
    for C in (10, 100):
        pc = rng.dirichlet(np.ones(C), 20000).astype(np.float32)
        ps = rng.dirichlet(np.ones(C), 20000).astype(np.float32)
        Mrow = np.zeros(C, bool)
        Mrow[: max(2, C // 5)] = True
        a = np.float32(0.3)
        wvec = np.where(Mrow, (1 - RSTR) / a, RSTR / (1 - a)).astype(np.float32)
        t0 = time.perf_counter()
        for i in range(20000):
            p2 = ps[i] * wvec
            p2 /= p2.sum()
            int(np.argmax(W * pc[i] + (1 - W) * p2))
        single = (time.perf_counter() - t0) / 20000
        t0 = time.perf_counter()
        for _ in range(20):
            p2 = ps * wvec
            p2 /= p2.sum(1, keepdims=True)
            (W * pc + (1 - W) * p2).argmax(1)
        batched = (time.perf_counter() - t0) / (20 * 20000)
        rows.append([C, f"{single * 1e6:.2f}", f"{batched * 1e9:.1f}"])
    return rows


def main():
    jobs = json.load(open(RUNS / "jobs.json"))
    A = [analyse_run(j) for j in jobs if (RUNS / f"{j['name']}.json").exists() and (RUNS / f"{j['name']}_evalprobs.npz").exists()]
    print(f"  analysed {len(A)} of {len(jobs)} runs", flush=True)
    keyf = lambda a: (a["job"]["scenario"], a["job"]["training"])
    groups = {}
    for a in A:
        groups.setdefault(keyf(a), []).append(a)
    order = []
    for j in jobs:
        k = (j["scenario"], j["training"])
        if k in groups and k not in order:
            order.append(k)
    mean = lambda L, rl, key: float(np.mean([a["res"][rl][key] for a in L]))
    # ---------- checks and manifest
    base_acc = {}
    rows = []
    for a in A:
        j = a["job"]
        bh = json.load(open(JR / "runs" / j["base_dir"] / f"{j['base']}.json"))
        base_int = float(bh.get("integrated_acc", np.mean([e_["acc_total"] for e_ in bh["eval"]])))
        b0 = a["res"]["B0"]["full"]
        flag = " (>0.3 pp)" if abs(b0 - base_int) * 100 > 0.3 else ""
        rows.append([j["name"], j["scenario"], j["training"], j["seed"], a["run_id"], j["base"], f"{base_int * 100:.4f}",
                     f"{b0 * 100:.4f}", f"{(b0 - base_int) * 100:+.4f}{flag}", f"{a['json_integrated'] * 100:.4f}",
                     a["mk_consistent"], a["mk_prov"], a["mismatch"], a["n_req"], a["sha"], a["bytes"],
                     f"{a['timing'].get('train', float('nan')) / 60:.1f}", f"{a['timing'].get('eval', float('nan')) / 60:.2f}",
                     f"{a['timing'].get('record', 0.0) / 60:.2f}", f"{a['total_min']:.1f}"])
    wcsv("R12_T0_checks_manifest.csv", ["run", "scenario", "training", "seed", "run_id", "base_run", "base_acc_full_pct",
                                        "B0_full_pct", "B0_minus_base_pp", "this_run_json_integrated_pct",
                                        "Mk_consistent_all_rounds", "Mk_equals_provenance", "record_vs_evaluator_mismatch",
                                        "requests", "evalprobs_sha256", "evalprobs_bytes", "train_min", "eval_min",
                                        "record_min", "total_min"], rows)
    # ---------- table 1 (+ per-seed csv)
    rows, rows_s = [], []
    for k in order:
        L = groups[k]
        rows.append([k[0], k[1], len(L)] + [f"{mean(L, rl, 'late') * 100:.2f}" for rl in RULES]
                    + [f"{mean(L, rl, 'full') * 100:.2f}" for rl in RULES]
                    + [f"{mean(L, 'B0', 'server_late') * 100:.1f}", f"{(mean(L, 'F', 'late') - mean(L, 'B0', 'late')) * 100:+.2f}",
                       f"{(mean(L, 'F', 'full') - mean(L, 'B0', 'full')) * 100:+.2f}"])
        for a in L:
            rows_s.append([k[0], k[1], a["job"]["seed"]] + [f"{a['res'][rl]['late'] * 100:.2f}" for rl in RULES]
                          + [f"{a['res'][rl]['full'] * 100:.2f}" for rl in RULES])
    hdr = [f"{rl}_late" for rl in RULES] + [f"{rl}_full" for rl in RULES]
    wcsv("R12_T1_rules.csv", ["scenario", "training", "seeds"] + hdr + ["B0_server_use_late_pct", "F_minus_B0_late_pp",
                                                                       "F_minus_B0_full_pp"], rows)
    wcsv("R12_T1_rules_per_seed.csv", ["scenario", "training", "seed"] + hdr, rows_s)
    # ---------- table 2: S1 and S2 splits
    rows = []
    for k in order:
        if k[0] not in ("S1", "S2"):
            continue
        L = groups[k]
        for rl in RULES:
            rows.append([k[0], k[1], rl] + [f"{mean(L, rl, key) * 100:.2f}" for key in ("home", "away", "Main", "OOP", "OOR")])
    wcsv("R12_T2_S1_S2_splits.csv", ["scenario", "training", "rule", "home_pct", "away_pct", "Main_pct", "OOP_pct", "OOR_pct"], rows)
    # ---------- table 3: lambda adjustment
    rows = []
    lam_eff = {}
    for scen in ("Schedule A", "client mobility", "ResNet-18", "S1"):
        if (scen, "DriftGate") not in groups or (scen, "fixed 0.4") not in groups:
            continue
        D, Fx = groups[(scen, "DriftGate")], groups[(scen, "fixed 0.4")]
        for rl in ("B0", "F"):
            for w_ in ("late", "full"):
                dlt = (mean(D, rl, w_) - mean(Fx, rl, w_)) * 100
                lam_eff[(scen, rl, w_)] = dlt
            seeds = sorted({a["job"]["seed"] for a in D} & {a["job"]["seed"] for a in Fx})
            per = [(next(a for a in D if a["job"]["seed"] == s)["res"][rl]["late"]
                    - next(a for a in Fx if a["job"]["seed"] == s)["res"][rl]["late"]) * 100 for s in seeds]
            rows.append([scen, rl, f"{lam_eff[(scen, rl, 'late')]:+.2f}", f"{lam_eff[(scen, rl, 'full')]:+.2f}",
                         " ".join(f"{x:+.2f}" for x in per)])
    wcsv("R12_T3_lambda_adjustment.csv", ["scenario", "rule", "DriftGate_minus_fixed_late_pp", "DriftGate_minus_fixed_full_pp",
                                          "per_seed_late_pp"], rows)
    # ---------- table 4: curves
    rows = []
    for k in order:
        L = groups[k]
        for cname in ("E", "SF"):
            for w_ in ("late", "full"):
                vals = np.nanmean([a["curves"][cname][w_]["at"] for a in L], axis=0) * 100
                rows.append([k[0], k[1], cname, w_] + [f"{v:.2f}" if np.isfinite(v) else "" for v in vals])
    wcsv("R12_T4_curves.csv", ["scenario", "training", "curve", "window"] + [f"acc_at_server_use_{o}" for o in OPTS], rows)
    # ---------- decisions
    F_B0 = {k: (mean(groups[k], "F", "late") - mean(groups[k], "B0", "late")) * 100 for k in order}
    s1 = F_B0.get(("S1", "fixed 0.4"))
    others = {sc: F_B0.get((sc, "fixed 0.4")) for sc in INDIVIDUAL_OTHERS}
    d1 = bool(s1 is not None and s1 >= 2.0 and all(v is not None and v > 0 for v in others.values()))
    la, lm, ls1 = (lam_eff.get(("Schedule A", "F", "late")), lam_eff.get(("client mobility", "F", "late")),
                   lam_eff.get(("S1", "F", "late")))
    d2 = bool(None not in (la, lm, ls1) and la >= 0.5 and lm >= 0.5 and ls1 >= -0.3)
    dec = dict(window="late (evaluation rounds from training round 30), seed means",
               decision_1_fusion_central=dict(answer="yes" if d1 else "no", S1_F_minus_B0_pp=s1,
                                              others_F_minus_B0_pp=others, runs="fixed lambda 0.4"),
               decision_2_keep_lambda_adjustment=dict(answer="yes" if d2 else "no", with_F_DriftGate_minus_fixed_pp=dict(
                   schedule_A=la, client_mobility=lm, S1=ls1)),
               F_minus_B0_late_pp={f"{k[0]} / {k[1]}": v for k, v in F_B0.items()},
               runs_analysed=len(A), runs_planned=len(jobs))
    json.dump(dec, open(OUT / "decision.json", "w"), indent=1)
    print(json.dumps(dec, indent=1))
    wcsv("R12_T5_fusion_cpu_time.csv", ["classes", "per_request_single_call_us", "per_request_batched_ns"], fusion_cpu_timing())
    figures(groups, order)


def figures(groups, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    mean = lambda L, rl, key: float(np.mean([a["res"][rl][key] for a in L])) * 100
    S1 = groups.get(("S1", "fixed 0.4"))
    if S1:
        fig, ax = plt.subplots(figsize=(3.6, 3.2), layout="constrained")
        lab = {"B0": "entropy routing (tau 0.8)", "B1": "client exit only", "B2": "server exit only",
               "B3": "mean of both exits", "F": "fusion F", "kind oracle": "request-kind oracle"}
        col = {"B0": INK, "B1": MUTED, "B2": INK2, "B3": GREEN, "F": BLUE, "kind oracle": PURPLE}
        mk = {"B0": "s", "B1": "v", "B2": "^", "B3": "D", "F": "o", "kind oracle": "*"}
        for rl in RULES:
            ax.plot(mean(S1, rl, "home"), mean(S1, rl, "away"), ls="none", marker=mk[rl], ms=7 if rl != "kind oracle" else 10,
                    color=col[rl], label=lab[rl])
        ax.set_xlabel("accuracy of clients at home (%)")
        ax.set_ylabel("accuracy of clients away (%)")
        ax.legend(fontsize=6.5, loc="lower right")
        fig.savefig(FIG / "R12_fig1_S1_home_away.pdf")
        fig.savefig(FIG / "R12_fig1_S1_home_away.png", dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.4, 3.0), layout="constrained")
        for cname, colr, lab_ in (("E", INK, "entropy routing (curve E)"), ("SF", BLUE, "selective fusion (curve SF)")):
            srv = S1[0]["curves"][cname]["late"]["srv"]
            acc = np.mean([a["curves"][cname]["late"]["acc"] for a in S1], axis=0) * 100
            srvm = np.mean([a["curves"][cname]["late"]["srv"] for a in S1], axis=0)
            ax.plot(srvm, acc, color=colr, marker="o", ms=2, label=lab_)
        ax.axvline(float(np.mean([a["res"]["B0"]["server_late"] for a in S1])), color=AXIS, lw=0.8, ls=":")
        ax.set_xlabel("server use (share of requests that need the server exit)")
        ax.set_ylabel("accuracy (%)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=6.5, loc="lower right")
        fig.savefig(FIG / "R12_fig2_S1_curves.pdf")
        fig.savefig(FIG / "R12_fig2_S1_curves.png", dpi=300)
        plt.close(fig)
    ks = [k for k in order]
    fig, ax = plt.subplots(figsize=(6.6, 2.8), layout="constrained")
    x = np.arange(len(ks))
    for off, w_, colr, lab_ in ((-0.2, "late", BLUE, "rounds from 30 on"), (0.2, "full", ORANGE, "all rounds")):
        vals = [mean(groups[k], "F", w_) - mean(groups[k], "B0", w_) for k in ks]
        ax.bar(x + off, vals, width=0.38, color=colr, label=lab_, zorder=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k[0]}\n{'DG' if k[1] == 'DriftGate' else 'fixed 0.4'}" for k in ks], fontsize=6.5)
    ax.set_ylabel("F - entropy routing (pp)")
    ax.legend(fontsize=7, loc="upper right")
    fig.savefig(FIG / "R12_fig3_F_minus_B0.pdf")
    fig.savefig(FIG / "R12_fig3_F_minus_B0.png", dpi=300)
    plt.close(fig)
    with open(FIG / "R12_figure_captions.md", "w") as f:
        f.write("# Round 12 figure captions\n\n## R12_fig1_S1_home_away\n\nAccuracy of clients at home against accuracy of clients "
                "away for each inference rule in the commute scenario (training ratio 0.4, seeds 0 to 4, evaluation rounds from "
                "training round 30 on): entropy routing with threshold 0.8, either exit alone, the mean of both exits, the fusion "
                "F (client weight 0.2, server exit corrected with strength 0.5) and the request-kind oracle.\n\n"
                "## R12_fig2_S1_curves\n\nAccuracy against the share of requests that need the server exit in the commute scenario. "
                "Entropy routing answers with the server exit above the entropy threshold; selective fusion answers with the fusion F "
                "there. Seeds 0 to 4, evaluation rounds from training round 30 on. The dotted line marks the server use of threshold 0.8.\n\n"
                "## R12_fig3_F_minus_B0\n\nAccuracy of the fusion F minus entropy routing with threshold 0.8 for every scenario and "
                "training (fixed ratio 0.4 or DriftGate), for evaluation rounds from training round 30 on and for all rounds. Seed means.\n")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
