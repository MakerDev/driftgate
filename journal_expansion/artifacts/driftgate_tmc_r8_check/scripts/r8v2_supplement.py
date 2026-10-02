"""Round 8 check v2, supplementary tables requested after the runs were launched (not used by any decision).

  python r8v2_supplement.py reduce   # requests.npz -> reduced/r8v2_*_exits.npz (per block, round, client, request kind)
  python r8v2_supplement.py tables   # -> tables/v2_S1_exit_accuracy*.csv, tables/v2_S2_server_use_at_reference.csv

S1 exit accuracy: for T15, T40, T60 and every lambda_inf block (and the installed model of T60, lambda 0.6), the accuracy
   when every request ends at the client exit (tau = infinity) and when every request is sent to the server exit;
   overall, at home / away (environment at_home) and on Main / OOP / OOR requests; plus home/away x kind.
   Accuracy as everywhere in Round 8: per evaluation round the mean over clients, then the mean over rounds; a split
   uses the clients (or client-rounds) that have requests of that group.
S2 server use at the reference accuracy: reference = accuracy of T40 with lambda_inf 0.4 and tau 0.8 (the installed
   model, same seed). For B*, the single-tau curves of T15 / T40 / T60 and Q1, Q2, Q3 (each lambda_t, and the best
   lambda_t), the lowest server use whose curve value is >= the reference (curves evaluated every 0.001 and linearly
   interpolated between the two neighbouring points); the same-seed difference from B*. Two ways of counting server use:
   (a) evaluation requests only (as in the decision), (b) the actual requests N of the environment with the probe
   requests of the signal-driven policies counted as server use (8, 16, 64 probes; B*, single tau and Q1 use no probe).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v2_analysis as A  # noqa: E402

KINDS = ["Main", "OOP", "OOR"]
OFINE = np.round(np.arange(0.0, 1.0 + 1e-9, 0.001), 3)


def reduce_exits():
    for a in A.ARMS:
        for s in A.SEEDS:
            name = f"r8v2_{a}_s{s}"
            q = np.load(A.RUNS / f"{name}_requests.npz")
            E, K = len(q["eval_rounds"]), 50
            g = q["req_eval_index"].astype(np.int64) * K + q["req_client"].astype(np.int64)
            gk = g * 3 + q["req_kind"].astype(np.int64)
            y = q["req_label"]
            n_kind = np.bincount(gk, minlength=E * K * 3).reshape(E, K, 3)
            nb = q["cp"].shape[0]
            cc = np.zeros((nb, E, K, 3), np.int32)
            sc = np.zeros((nb, E, K, 3), np.int32)
            for b in range(nb):
                cc[b] = np.bincount(gk, weights=q["cp"][b] == y, minlength=E * K * 3).reshape(E, K, 3)
                sc[b] = np.bincount(gk, weights=q["sp"][b] == y, minlength=E * K * 3).reshape(E, K, 3)
            np.savez_compressed(A.RED / f"{name}_exits.npz", n_kind=n_kind.astype(np.int32), cc_kind=cc, sc_kind=sc,
                                block_lambda=q["block_lambda"], installed_block=q["installed_block"])
            print(f"  {name}: exits by kind -> reduced/{name}_exits.npz", flush=True)


def exit_metrics(c, n, home):
    """c, n [E, K, 3] counts by kind -> overall, home, away, Main, OOP, OOR and home/away x kind accuracies."""
    E, K, _ = n.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = c.sum(2) / n.sum(2)
        ak = np.where(n > 0, c / np.maximum(n, 1), np.nan)
    hm = [acc[e][home[e]].mean() for e in range(E) if home[e].any()]
    aw = [acc[e][~home[e]].mean() for e in range(E) if (~home[e]).any()]
    out = dict(overall=acc.mean(), home=np.mean(hm), away=np.mean(aw))
    for i, kname in enumerate(KINDS):
        per_round = [np.nanmean(ak[e, :, i]) for e in range(E) if np.isfinite(ak[e, :, i]).any()]
        out[kname] = np.mean(per_round)
        for where, mask in (("home", home), ("away", ~home)):
            per_round = [np.nanmean(ak[e, mask[e], i]) for e in range(E) if np.isfinite(ak[e, mask[e], i]).any()]
            out[f"{where} {kname}"] = np.mean(per_round) if per_round else np.nan
    return out


def table_exits(R):
    cols = ["overall", "home", "away"] + KINDS
    cross = [f"{w} {k}" for w in ("home", "away") for k in KINDS]
    rows, rows_x, mean_rows = [], [], []
    for a in A.ARMS:
        X = {s: np.load(A.RED / f"r8v2_{a}_s{s}_exits.npz") for s in A.SEEDS}
        blk = [float(v) for v in X[A.SEEDS[0]]["block_lambda"]]
        ib = int(X[A.SEEDS[0]]["installed_block"])
        for b, lam in enumerate(blk):
            for exit_name, key in (("client exit only (tau = inf)", "cc_kind"), ("server exit only", "sc_kind")):
                ms = {s: exit_metrics(X[s][key][b].astype(float), X[s]["n_kind"].astype(float), R[(a, s)]["home"]) for s in A.SEEDS}
                tag = f"{lam:g}" + (" (= lambda_t, installed)" if b == ib else "")
                for s in A.SEEDS:
                    rows.append([a, tag, exit_name, s] + [f"{ms[s][c_] * 100:.4f}" for c_ in cols])
                    rows_x.append([a, tag, exit_name, s] + [f"{ms[s][c_] * 100:.4f}" for c_ in cross])
                mean_rows.append([a, tag, exit_name] + [f"{np.mean([ms[s][c_] for s in A.SEEDS]) * 100:.2f}" for c_ in cols + cross])
    A.wcsv("v2_S1_exit_accuracy_per_seed.csv", ["arm", "lambda_inf", "exit", "seed"] + [f"{c_}_pct" for c_ in cols], rows)
    A.wcsv("v2_S1_exit_accuracy_home_away_by_kind_per_seed.csv", ["arm", "lambda_inf", "exit", "seed"] + [f"{c_}_pct" for c_ in cross], rows_x)
    A.wcsv("v2_S1_exit_accuracy_mean.csv", ["arm", "lambda_inf", "exit"] + [f"{c_}_pct" for c_ in cols + cross], mean_rows)


def min_server_use(vals, ref):
    ok = np.isfinite(vals) & (vals >= ref - 1e-12)
    if not ok.any():
        return np.nan
    i = int(np.argmax(ok))
    if i == 0:
        return float(OFINE[0])
    v0, v1 = vals[i - 1], vals[i]
    if np.isfinite(v0) and v1 > v0:
        return float(OFINE[i - 1] + (ref - v0) / (v1 - v0) * (OFINE[i] - OFINE[i - 1]))
    return float(OFINE[i])


def table_server_use(R):
    POL = {"B": False, "Q1": True, "Q2": True, "Q3": True}

    def pc(D, name, mode, m=64):
        ib = D["ib"]
        H = {"B": np.ones_like(D["home"]), "Q1": D["sit"]["oracle"], "Q2": D["sit"][("sr", m)], "Q3": D["sit"][("tv", m)]}[name]
        return A.curve(D, H, ib, ib, POL[name], mode, OFINE)[0]

    ref = {s: R[("T40", s)]["h"]["integrated_acc"] for s in A.SEEDS}
    rows = []
    for count, variants in (("evaluation requests (probes not counted)", [(("std",), 64)]),
                            ("actual requests N, probes counted as server use", [(("probe", m), m) for m in A.PROBES])):
        for mode_sig, m in variants:
            base_mode = ("std",) if mode_sig[0] == "std" else ("Nw",)
            curves = {}
            for s in A.SEEDS:
                for a in A.ARMS:
                    D = R[(a, s)]
                    curves[("single tau", a, s)] = pc(D, "B", base_mode)
                    curves[("Q1", a, s)] = pc(D, "Q1", base_mode)
                    curves[("Q2", a, s)] = pc(D, "Q2", mode_sig, m)
                    curves[("Q3", a, s)] = pc(D, "Q3", mode_sig, m)
                curves[("B*", "best", s)] = np.nanmax([curves[("single tau", a, s)] for a in A.ARMS], axis=0)
                for p in ("Q1", "Q2", "Q3"):
                    curves[(p, "best", s)] = np.nanmax([curves[(p, a, s)] for a in A.ARMS], axis=0)
            mins = {k: min_server_use(v, ref[k[2]]) for k, v in curves.items()}
            labels = [("B*", "best")] + [("single tau", a) for a in A.ARMS] + [(p, a) for p in ("Q1", "Q2", "Q3") for a in A.ARMS + ["best"]]
            for p, a in labels:
                if mode_sig[0] == "probe" and p not in ("Q2", "Q3") and m != A.PROBES[0]:
                    continue   # curves without probes do not depend on m
                v = [mins[(p, a, s)] for s in A.SEEDS]
                d = [mins[(p, a, s)] - mins[("B*", "best", s)] for s in A.SEEDS]
                probes = m if (mode_sig[0] == "probe" and p in ("Q2", "Q3")) else 0
                if np.all(np.isfinite(d)) and p != "B*":
                    dd = np.array(d) * 100
                    h = stats.sem(dd) * stats.t.ppf(0.975, 2)
                    dtxt = [f"{dd.mean():+.2f}", f"[{dd.mean() - h:+.2f}, {dd.mean() + h:+.2f}]", int((dd < 0).sum())]
                else:
                    dtxt = ["", "", ""]
                rows.append([count, p, a, probes, " ".join(f"{ref[s] * 100:.2f}" for s in A.SEEDS)]
                            + [f"{x * 100:.2f}" if np.isfinite(x) else "not reached" for x in v]
                            + [f"{np.mean(v) * 100:.2f}" if np.all(np.isfinite(v)) else "not reached in every seed"]
                            + dtxt + [" ".join(f"{x * 100:+.2f}" if np.isfinite(x) else "nan" for x in d)])
    A.wcsv("v2_S2_server_use_at_reference.csv",
           ["server_use_counted_over", "curve", "lambda_t", "probes_counted", "reference_acc_pct(s5,s6,s7)",
            "min_server_use_pct_s5", "min_server_use_pct_s6", "min_server_use_pct_s7", "mean_pct",
            "minus_Bstar_pp", "ci95", "seeds_below_Bstar", "per_seed_minus_Bstar_pp"], rows)


def table_controls(R):
    """S3 (added after the v2 results were seen, to read the decisions; not used by them):
    (a) one lambda_inf for every client (single tau): how much of G(P) needs no home/away information;
    (b) placebo labels: per evaluation round the same number of "home" clients as the policy's own labels
        (at_home, x_SR or x_TV), chosen at random (50 draws, seed 20261003); G of the policy with these labels and the
        same-seed difference policy - placebo (mean over draws)."""
    OS = np.array(A.OPTS)
    rng = np.random.default_rng(20261003)
    NDRAW = 50

    def G(D, H, bh, ba, two, s):
        return float(np.mean(A.curve(D, H, bh, ba, two, ("std",), OS)[0] - Bstar[s]) * 100)

    Bstar = {s: np.max([A.curve(R[(a, s)], np.ones_like(R[(a, s)]["home"]), R[(a, s)]["ib"], R[(a, s)]["ib"], False,
                                ("std",), OS)[0] for a in A.ARMS], axis=0) for s in A.SEEDS}
    rows = []
    for a in A.ARMS:
        for b, lam in enumerate(R[(a, A.SEEDS[0])]["blk"]):
            g = [G(R[(a, s)], np.ones_like(R[(a, s)]["home"]), b, b, False, s) for s in A.SEEDS]
            st = A.tstat(g)
            rows.append(["one lambda_inf for every client", a, f"lambda_inf {lam:g}", f"{np.mean(g):+.4f}", "", "", "", "",
                         " ".join(f"{x:+.2f}" for x in g)])
    SPEC = [("P1", "oracle", "lambda_inf"), ("P2", ("sr", 64), "lambda_inf"), ("P3", ("tv", 64), "lambda_inf"),
            ("Q1", "oracle", "tau"), ("Q2", ("sr", 64), "tau"), ("Q3", ("tv", 64), "tau")]
    for p, key, what in SPEC:
        for a in (["T15", "T40"] if what == "lambda_inf" else A.ARMS):
            gp, gpl, gmax = [], [], []
            for s in A.SEEDS:
                D = R[(a, s)]
                ib = D["ib"]
                bh, two = (A.I70, False) if what == "lambda_inf" else (ib, True)
                Hs = D["sit"][key]
                gp.append(G(D, Hs, bh, ib, two, s))
                draws = []
                for _ in range(NDRAW):
                    P = np.zeros_like(Hs)
                    for e in range(Hs.shape[0]):
                        P[e, rng.permutation(Hs.shape[1])[:int(Hs[e].sum())]] = True
                    draws.append(G(D, P, bh, ib, two, s))
                gpl.append(float(np.mean(draws)))
                gmax.append(float(np.max(draws)))
            d = A.tstat(np.array(gp) - np.array(gpl))
            rows.append([f"{p} with placebo labels", a, f"{NDRAW} draws", f"{np.mean(gpl):+.4f}",
                         f"{np.mean(gp):+.4f}"] + A.fmt(d)[:3] + [" ".join(f"{x:+.2f}" for x in np.array(gp) - np.array(gpl))
                                                                 + " | placebo max over draws " + " ".join(f"{x:+.2f}" for x in gmax)])
    A.wcsv("v2_S3_controls.csv", ["control", "arm", "setting", "G_control_pp", "G_policy_pp", "policy_minus_control_pp",
                                  "ci95", "seeds_policy_higher", "per_seed"], rows)


def per_round_acc(D, H, bh, ba, cfg):
    """accuracy per evaluation round (mean over clients) at a curve point (two configurations mixed with weight w)."""
    E, K = D["n"].shape
    ee, kk = np.meshgrid(np.arange(E), np.arange(K), indexing="ij")
    def at(jh, ja):
        return D["A"][np.where(H, bh, ba), ee, kk, np.where(H, jh, ja)].mean(axis=1)
    kind, fixed, j, w = cfg
    if kind == "1d":
        return w * at(j, j) + (1 - w) * at(j + 1, j + 1)
    if kind == "fix_home":
        return w * at(fixed, j) + (1 - w) * at(fixed, j + 1)
    return w * at(j, fixed) + (1 - w) * at(j + 1, fixed)


def table_slots(R):
    """S4 (added after the v2 results were seen; not used by the decisions): G split over the Round 6 time slots.
    At each comparison point the policy and B* are evaluated at their own best configuration; the per-round
    difference is averaged over the three points; a slot's contribution = sum over its evaluation rounds / 31.
    The contributions add up to G."""
    OS = np.array(A.OPTS)
    curves = {}
    for a in A.ARMS:
        for s in A.SEEDS:
            D = R[(a, s)]
            curves[("B", a, s)] = A.curve(D, np.ones_like(D["home"]), D["ib"], D["ib"], False, ("std",), OS)
    er = R[("T40", A.SEEDS[0])]["er"]
    SPEC = [("P1", "T15", "oracle", True), ("P2", "T15", ("sr", 64), True), ("P3", "T15", ("tv", 64), True),
            ("P1", "T40", "oracle", True), ("P2", "T40", ("sr", 64), True),
            ("Q1", "T40", "oracle", False), ("Q2", "T40", ("sr", 64), False), ("Q3", "T40", ("tv", 64), False)]
    rows = []
    for p, a, key, is_p in SPEC:
        diffs = []
        for s in A.SEEDS:
            D = R[(a, s)]
            ib = D["ib"]
            H = D["sit"][key]
            bh, two = (A.I70, False) if is_p else (ib, True)
            _, cfgs = A.curve(D, H, bh, ib, two, ("std",), OS)
            d = np.zeros(len(er))
            for i in range(len(OS)):
                best = max(A.ARMS, key=lambda b_: curves[("B", b_, s)][0][i])
                Db = R[(best, s)]
                rb = per_round_acc(Db, np.ones_like(Db["home"]), Db["ib"], Db["ib"], curves[("B", best, s)][1][i])
                rp = per_round_acc(D, H, bh, ib, cfgs[i])
                d += (rp - rb) / len(OS)
            diffs.append(d * 100)
        diffs = np.array(diffs)
        for name, lo, hi in A.SLOTS:
            m = (er >= lo) & (er <= hi)
            cont = diffs[:, m].sum(axis=1) / len(er)
            rows.append([p, a, name, int(m.sum()), f"{cont.mean():+.4f}", " ".join(f"{x:+.2f}" for x in cont)])
        rows.append([p, a, "sum = G", len(er), f"{(diffs.sum(axis=1) / len(er)).mean():+.4f}",
                     " ".join(f"{x:+.2f}" for x in diffs.sum(axis=1) / len(er))])
    A.wcsv("v2_S4_G_by_time_slot.csv", ["policy", "arm", "slot", "eval_rounds", "contribution_to_G_pp", "per_seed_pp(5,6,7)"], rows)


def main():
    if sys.argv[1:] == ["slots"]:
        table_slots({(a, s): A.load(a, s) for a in A.ARMS for s in A.SEEDS})
        return
    if sys.argv[1:] == ["reduce"]:
        reduce_exits()
        return
    R = {(a, s): A.load(a, s) for a in A.ARMS for s in A.SEEDS}
    table_exits(R)
    table_server_use(R)
    table_controls(R)
    table_slots(R)


if __name__ == "__main__":
    main()
