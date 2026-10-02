"""Round 8 check v2, step 2: comparison at matched server use.
reduced counts (reduced/r8v2_*_reduced.npz, from r8v2_reduce.py) + run JSON / trace (runs/phaseT8_check) + env
(runs/phaseT6_env) -> tables/v2_*.csv, decision_v2.json, figures/v2_*.

Definitions (directive v2, section 5):
- accuracy: per evaluation round the mean over clients (equal weights), then the mean over the 31 rounds;
- server use: requests answered by the server exit / all evaluation requests (request weighted);
- tau grid 0, 0.05, ..., 2.30 and infinity; a policy with one tau gives one (server use, accuracy) point per tau;
  the curve value at a server use o is the linear interpolation between the two neighbouring grid points;
- policies with tau_home and tau_away: every (tau_home, tau_away) pair; the value at o is the largest value over
  the interpolations along tau_away with tau_home fixed and along tau_home with tau_away fixed;
- B*(o): the largest value at o over the single-tau curves of T15, T40, T60 with lambda_inf = lambda_t (same seed);
- G = mean over o in {0.5, 0.7, 0.9} of (policy(o) - B*(o)), per seed; mean, Student-t 95% CI, seeds with G > 0;
- situation from a signal: Round 7 device score q_k recomputed with the same constants; rounds 1-25 "home";
  from round 26 q_k >= 1.5 "away", otherwise "home". The oracle uses the environment's at_home.
Probe cost (5.4): N = the client's requests in that round (environment), n_p = min(probes drawn, m); a fraction
n_p / N of the requests is answered by the server exit (the probes), the rest follows the policy; server use =
sum(n_p + (N - n_p) * s) / sum(N) and accuracy per client = (n_p acc_server + (N - n_p) acc_policy) / N. B* for this
comparison uses the same N weighting without probes.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
RUNS = Path(os.environ.get("R8V2_RUNS", JR / "runs" / "phaseT8_check"))   # overrides only for smoke tests
R7RUNS = JR / "runs" / "phaseT7_gate"
ENV = JR / "runs" / "phaseT6_env"
RED = Path(os.environ.get("R8V2_RED", HERE / "reduced"))
OUT = Path(os.environ.get("R8V2_OUT", HERE))
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_controller import DeviceDriftGate  # noqa: E402

ARMS = ["T15", "T40", "T60"]
LT = {"T15": 0.15, "T40": 0.40, "T60": 0.60}
SEEDS = [5, 6, 7]
LI = [0.15, 0.3, 0.4, 0.55, 0.7]
I70 = LI.index(0.7)
OPTS = [0.5, 0.7, 0.9]
Q_SWITCH = 1.5
PROBES = [8, 16, 64]
OGRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
SLOTS = [("pre-commute", 1, 25), ("commute", 26, 45), ("daytime", 46, 110), ("return", 111, 140), ("evening", 141, 150)]
SIG = {"sr": ("x_sr", "probe_sr"), "tv": ("x_tv", "probe_tv")}


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows", flush=True)


def tstat(v):
    v = np.asarray(v, float)
    h = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)
    return dict(mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h), n_pos=int((v > 0).sum()),
                n=len(v), per_seed=[float(x) for x in v])


def fmt(st, scale=1.0):
    return [f"{st['mean'] * scale:+.4f}", f"[{st['lo'] * scale:+.2f}, {st['hi'] * scale:+.2f}]", st["n_pos"], st["n"],
            " ".join(f"{x * scale:+.2f}" for x in st["per_seed"])]


# ------------------------------------------------------------------------------------------------ signals
def recompute_q(x, cells):
    T, K = x.shape
    dc = DeviceDriftGate(K)
    q = np.full((T, K), np.nan)
    for t in range(T):
        xs = {k: float(x[t, k]) for k in range(K) if np.isfinite(x[t, k])}
        mem = {}
        for k in range(K):
            for c in cells[t, k]:
                if c >= 0:
                    mem.setdefault(int(c), set()).add(k)
        ref = {k: sorted(set().union(*[mem[int(c)] for c in cells[t, k] if c >= 0])) for k in xs}
        dc.step(xs, ref)
        q[t] = [dc.q[k] for k in range(K)]
    return q


def signal_series(z, sig, m):
    """x per round and client from the first m probe requests (m = 64: the value the runner recorded)."""
    key, pkey = SIG[sig]
    if m == 64:
        return z[key].astype(float)
    p = z[pkey][:, :, :m].astype(float)
    if sig == "sr":
        p[z[pkey][:, :, :m] < 0] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(p, axis=2)


# ------------------------------------------------------------------------------------------------ load
def load(arm, s):
    red = np.load(RED / f"r8v2_{arm}_s{s}_reduced.npz")
    h = json.load(open(RUNS / f"r8v2_{arm}_s{s}.json"))
    z = np.load(RUNS / f"r8v2_{arm}_s{s}_trace.npz")
    env = np.load(ENV / f"S1_seed{s}.npz")
    er = np.array(h["eval_rounds"])
    assert np.array_equal(er, red["eval_rounds"])
    n = red["n"].astype(float)
    ib = int(red["installed_block"])
    assert abs(float(red["block_lambda"][ib]) - LT[arm]) < 1e-12
    home = z["at_home"][er - 1].astype(bool)
    assert np.array_equal(home, red["home"])
    D = dict(arm=arm, seed=s, h=h, z=z, er=er, n=n, nmain=red["n_main"].astype(float), ib=ib,
             blk=[float(v) for v in red["block_lambda"]], home=home,
             A=red["corr"] / n[None, :, :, None], S=red["nsrv"].astype(float),
             cmain=red["cmain"].astype(float), cnon=red["cnon"].astype(float),
             ncc=red["n_cc"].astype(float), nsc=red["n_sc"].astype(float), ngain=red["n_gain"].astype(float),
             nloss=red["n_loss"].astype(float), red=red,
             N=env["n_req"][:, er - 1].T.astype(float), nprobe=z["nprobe"][er - 1].astype(float))
    D["nnon"] = n - D["nmain"]
    D["sit"] = {"oracle": home}
    D["x"], D["q"] = {}, {}
    for sig in SIG:
        for m in PROBES:
            x = signal_series(z, sig, m)
            q = recompute_q(x, z["cells"])
            D["x"][(sig, m)], D["q"][(sig, m)] = x, q
            D["sit"][(sig, m)] = (er <= 25)[:, None] | (q[er - 1] < Q_SWITCH)
    return D


# ------------------------------------------------------------------------------------------------ curves
def interp_rows(R, A, os):
    """R, A [M, P+1] with R non-increasing along axis 1; value at each o (largest bracketing pair) -> [M, n_o],
    plus the pair index and weight of the first-axis-wise best."""
    hi, lo, ah, al = R[:, :-1], R[:, 1:], A[:, :-1], A[:, 1:]
    O = np.asarray(os)[None, None, :]
    inside = (lo[..., None] <= O + 1e-12) & (O <= hi[..., None] + 1e-12)
    span = (hi - lo)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(span > 0, (O - lo[..., None]) / np.where(span > 0, span, 1.0), 1.0)
    w = np.clip(w, 0, 1)
    v = np.where(span > 0, w * ah[..., None] + (1 - w) * al[..., None], np.maximum(ah, al)[..., None])
    w = np.where(span > 0, w, (ah >= al)[..., None].astype(float))
    v = np.where(inside, v, -np.inf)
    jbest = v.argmax(axis=1)                                              # [M, n_o]
    vbest = np.take_along_axis(v, jbest[:, None, :], axis=1)[:, 0, :]
    wbest = np.take_along_axis(w, jbest[:, None, :], axis=1)[:, 0, :]
    vbest[np.isinf(vbest)] = np.nan
    return vbest, jbest, wbest


def transform(D, b, mode):
    """per-block accuracy [E,K,NJ] and server counts [E,K,NJ] and the denominator of server use, for
    mode = ("std",) | ("Nw",) | ("probe", m)."""
    A, S, n = D["A"][b], D["S"][b], D["n"]
    if mode[0] == "std":
        return A, S, n.sum()
    N = D["N"]
    if mode[0] == "Nw":
        return A, S / n[..., None] * N[..., None], N.sum()
    npr = np.minimum(D["nprobe"], mode[1])
    f = (npr / N)[..., None]
    acc_srv = (D["nsc"][b] / n)[..., None]
    return f * acc_srv + (1 - f) * A, npr[..., None] + (N - npr)[..., None] * S / n[..., None], N.sum()


def policy_arrays(D, H, bh, ba, mode):
    """home-set and away-set accuracy (summed over client-rounds / (E K)) and server counts per tau index."""
    E, K = D["n"].shape
    Ah, Sh, tot = transform(D, bh, mode)
    Aa, Sa, _ = transform(D, ba, mode)
    aH = (Ah * H[..., None]).sum((0, 1)) / (E * K)
    sH = (Sh * H[..., None]).sum((0, 1))
    aA = (Aa * ~H[..., None]).sum((0, 1)) / (E * K)
    sA = (Sa * ~H[..., None]).sum((0, 1))
    return aH, sH, aA, sA, tot


def curve(D, H, bh, ba, two_tau, mode, os):
    """values at os and, for each o, the best configuration (for split metrics)."""
    aH, sH, aA, sA, tot = policy_arrays(D, H, bh, ba, mode)
    if not two_tau:
        R, A = ((sH + sA) / tot)[None], (aH + aA)[None]
        v, j, w = interp_rows(R, A, os)
        cfg = [("1d", None, int(j[0, i]), float(w[0, i])) for i in range(len(os))]
        return v[0], cfg
    R1 = (sH[:, None] + sA[None, :]) / tot          # [jh, ja]
    A1 = aH[:, None] + aA[None, :]
    v1, j1, w1 = interp_rows(R1, A1, os)             # fixed jh (rows), sweep ja
    v2, j2, w2 = interp_rows(R1.T, A1.T, os)         # fixed ja (rows), sweep jh
    out, cfg = np.full(len(os), np.nan), []
    for i in range(len(os)):
        c1 = np.nanargmax(v1[:, i]) if np.isfinite(v1[:, i]).any() else None
        c2 = np.nanargmax(v2[:, i]) if np.isfinite(v2[:, i]).any() else None
        b1 = v1[c1, i] if c1 is not None else -np.inf
        b2 = v2[c2, i] if c2 is not None else -np.inf
        if c1 is None and c2 is None:
            cfg.append(None)
            continue
        if b1 >= b2:
            out[i] = b1
            cfg.append(("fix_home", int(c1), int(j1[c1, i]), float(w1[c1, i])))
        else:
            out[i] = b2
            cfg.append(("fix_away", int(c2), int(j2[c2, i]), float(w2[c2, i])))
    return out, cfg


def split_metrics(D, H, bh, ba, cfg):
    """accuracy at home / away (environment at_home) and on Main / non-Main requests for a curve point."""
    def at(jh, ja):
        E, K = D["n"].shape
        jj = np.where(H, jh, ja)
        bb = np.where(H, bh, ba)
        ee, kk = np.meshgrid(np.arange(E), np.arange(K), indexing="ij")
        acc = D["A"][bb, ee, kk, jj]
        with np.errstate(invalid="ignore", divide="ignore"):
            am = D["cmain"][bb, ee, kk, jj] / D["nmain"]
            an = np.where(D["nnon"] > 0, D["cnon"][bb, ee, kk, jj] / D["nnon"], np.nan)
        home = D["home"]
        hm = [acc[e][home[e]].mean() for e in range(E) if home[e].any()]
        aw = [acc[e][~home[e]].mean() for e in range(E) if (~home[e]).any()]
        return np.array([acc.mean(), np.mean(hm), np.mean(aw), np.mean(np.nanmean(am, 1)), np.mean(np.nanmean(an, 1))])
    kind, fixed, j, w = cfg
    if kind == "1d":
        return w * at(j, j) + (1 - w) * at(j + 1, j + 1)
    if kind == "fix_home":
        return w * at(fixed, j) + (1 - w) * at(fixed, j + 1)
    return w * at(j, fixed) + (1 - w) * at(j + 1, fixed)


def upper_bound(D, os):
    """routing upper bound for the installed model: send to the server first the requests where the client exit is
    wrong and the server exit right (largest accuracy weight first), then requests with no change, then requests
    where the client exit is right and the server exit wrong (smallest weight first)."""
    b = D["ib"]
    n = D["n"]
    E, K = n.shape
    wgt = (1.0 / (n * E * K)).ravel()
    base = (D["ncc"][b] / n).mean()
    tot = n.sum()
    g, l = D["ngain"][b].ravel(), D["nloss"][b].ravel()
    neutral = (n.ravel() - g - l).sum()
    og = np.argsort(-wgt, kind="stable")
    cg, cgv = np.cumsum(g[og]), np.cumsum(g[og] * wgt[og])
    ol = np.argsort(wgt, kind="stable")
    cl, clv = np.cumsum(l[ol]), np.cumsum(l[ol] * wgt[ol])
    G, Gv = cg[-1], cgv[-1]
    out = []
    for o in os:
        S = o * tot
        if S <= G:
            i = np.searchsorted(cg, S)
            prev_c, prev_v = (cg[i - 1], cgv[i - 1]) if i > 0 else (0.0, 0.0)
            out.append(base + prev_v + (S - prev_c) * wgt[og][i])
        elif S <= G + neutral:
            out.append(base + Gv)
        else:
            S2 = min(S - G - neutral, cl[-1])
            i = min(np.searchsorted(cl, S2), len(cl) - 1)
            prev_c, prev_v = (cl[i - 1], clv[i - 1]) if i > 0 else (0.0, 0.0)
            out.append(base + Gv - (prev_v + (S2 - prev_c) * wgt[ol][i]))
    return np.array(out)


def auroc_away(D, sig, m):
    from sklearn.metrics import roc_auc_score
    x = D["x"][(sig, m)][25:]
    away = ~D["z"]["at_home"][25:].astype(bool)
    vals = [roc_auc_score(away[:, k], x[:, k]) for k in range(x.shape[1])
            if away[:, k].any() and (~away[:, k]).any() and np.isfinite(x[:, k]).all()]
    return float(np.mean(vals)), len(vals)


# ------------------------------------------------------------------------------------------------ main
def main():
    R = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    J08 = int(np.flatnonzero(np.isclose(R[("T40", 5)]["red"]["tau"], 0.8))[0])

    def pol(D, name, mode=("std",), sig_m=64):
        """(H, block_home, block_away, two_tau) of a named policy for run D."""
        ib = D["ib"]
        sit = lambda key: D["sit"][key] if key == "oracle" else D["sit"][(key, sig_m)]
        table = {"B": (np.ones_like(D["home"]), ib, ib, False),
                 "P1": (sit("oracle"), I70, ib, False), "P2": (sit("sr"), I70, ib, False), "P3": (sit("tv"), I70, ib, False),
                 "Q1": (sit("oracle"), ib, ib, True), "Q2": (sit("sr"), ib, ib, True), "Q3": (sit("tv"), ib, ib, True),
                 "C1": (sit("oracle"), I70, ib, True), "C2": (sit("sr"), I70, ib, True)}
        return table[name]

    def pcurve(D, name, os, mode=("std",), sig_m=64):
        H, bh, ba, two = pol(D, name, mode, sig_m)
        return curve(D, H, bh, ba, two, mode, os)

    # ---------- checks (installed model at tau 0.8, probe means, installed-block mismatch, T40 vs Round 7 F)
    rows = []
    for a in ARMS:
        for s in SEEDS:
            D = R[(a, s)]
            b, z = D["ib"], D["z"]
            same_counts = bool(np.array_equal(D["red"]["corr"][b][:, :, J08], z["eval_correct"])
                               and np.array_equal(D["red"]["nsrv"][b][:, :, J08], z["eval_n_off"]))
            rec08 = float(D["A"][b][:, :, J08].mean())
            dx = {sig: float(np.nanmax(np.abs(np.nanmean(np.where(z[SIG[sig][1]] < 0, np.nan, z[SIG[sig][1]]).astype(float)
                                                          if sig == "sr" else z[SIG[sig][1]].astype(float), axis=2)
                                                - z[SIG[sig][0]]))) for sig in SIG}
            row = [a, s, D["h"]["config"]["run_id"], f"{D['h']['integrated_acc'] * 100:.6f}", f"{rec08 * 100:.6f}",
                   "yes" if same_counts else "NO", D["h"].get("installed_block_mismatch", ""), f"{dx['tv']:.1e}",
                   f"{dx['sr']:.1e}"]
            v1 = RUNS / f"r8_{a}_s{s}.json"
            row.append(f"{(D['h']['integrated_acc'] - json.load(open(v1))['integrated_acc']) * 100:+.4f}" if v1.exists() else "")
            if a == "T40":
                f7 = json.load(open(R7RUNS / f"r7_F_s{s}.json"))
                row += [f"{f7['integrated_acc'] * 100:.4f}", f"{(D['h']['integrated_acc'] - f7['integrated_acc']) * 100:+.4f}"]
            else:
                row += ["", ""]
            rows.append(row)
    wcsv("v2_T1_checks.csv", ["arm", "seed", "run_id", "installed_integrated_pct(json)", "recomputed_tau0.8_pct",
                              "counts_equal_installed(correct,n_off)", "installed_block_mismatch(client-rounds)",
                              "max|mean(probe_tv)-x_tv|", "max|mean(probe_sr)-x_sr|", "minus_round8_v1_same_arm_seed_pp",
                              "round7_F_integrated_pct", "T40_minus_round7_F_pp"], rows)
    # ---------- tau = 0.8 (the setting used so far)
    rows = []
    for a in ARMS:
        accs, srv = [], []
        for s in SEEDS:
            D = R[(a, s)]
            m = split_metrics(D, np.ones_like(D["home"]), D["ib"], D["ib"], ("1d", None, J08, 1.0))
            accs.append(m)
            srv.append(D["S"][D["ib"]][:, :, J08].sum() / D["n"].sum())
            rows.append([a, s, f"{m[0] * 100:.4f}", f"{srv[-1] * 100:.2f}", f"{m[1] * 100:.4f}", f"{m[2] * 100:.4f}",
                         f"{m[3] * 100:.4f}", f"{m[4] * 100:.4f}"])
        mm = np.mean(accs, axis=0)
        rows.append([a, "mean", f"{mm[0] * 100:.4f}", f"{np.mean(srv) * 100:.2f}", f"{mm[1] * 100:.4f}", f"{mm[2] * 100:.4f}",
                     f"{mm[3] * 100:.4f}", f"{mm[4] * 100:.4f}"])
    wcsv("v2_T2_tau08.csv", ["arm", "seed", "accuracy_pct", "server_use_pct", "home_pct", "away_pct", "main_pct",
                             "nonmain_pct"], rows)
    # ---------- curves at the comparison points and on the plotting grid
    OS = np.array(OPTS)
    arm_curve = {(a, s): pcurve(R[(a, s)], "B", OS) for a in ARMS for s in SEEDS}
    arm_grid = {(a, s): pcurve(R[(a, s)], "B", OGRID)[0] for a in ARMS for s in SEEDS}
    Bstar = {s: np.max([arm_curve[(a, s)][0] for a in ARMS], axis=0) for s in SEEDS}
    Bstar_arm = {s: [ARMS[i] for i in np.argmax([arm_curve[(a, s)][0] for a in ARMS], axis=0)] for s in SEEDS}
    Bstar_grid = {s: np.max([arm_grid[(a, s)] for a in ARMS], axis=0) for s in SEEDS}
    POLS = {"P1": ["T15", "T40"], "P2": ["T15", "T40"], "P3": ["T15", "T40"],
            "Q1": ARMS, "Q2": ARMS, "Q3": ARMS, "C1": ARMS, "C2": ARMS}
    PC = {(p, a, s): pcurve(R[(a, s)], p, OS) for p, arms in POLS.items() for a in arms for s in SEEDS}
    Gs = {(p, a): tstat([np.mean(PC[(p, a, s)][0] - Bstar[s]) * 100 for s in SEEDS]) for p, arms in POLS.items() for a in arms}
    for a in ARMS:
        Gs[("B", a)] = tstat([np.mean(arm_curve[(a, s)][0] - Bstar[s]) * 100 for s in SEEDS])
    rows = []
    for s in SEEDS:
        for i, o in enumerate(OPTS):
            rows.append(["B*", f"best of arms ({Bstar_arm[s][i]})", s, o, f"{Bstar[s][i] * 100:.4f}", ""])
    for a in ARMS:
        for s in SEEDS:
            for i, o in enumerate(OPTS):
                rows.append(["single tau, lambda_inf = lambda_t", a, s, o, f"{arm_curve[(a, s)][0][i] * 100:.4f}",
                             f"{(arm_curve[(a, s)][0][i] - Bstar[s][i]) * 100:+.4f}"])
    for (p, a, s), (v, _) in PC.items():
        for i, o in enumerate(OPTS):
            rows.append([p, a, s, o, f"{v[i] * 100:.4f}", f"{(v[i] - Bstar[s][i]) * 100:+.4f}"])
    for a in ARMS:
        for s in SEEDS:
            ub = upper_bound(R[(a, s)], OS)
            for i, o in enumerate(OPTS):
                rows.append(["routing upper bound", a, s, o, f"{ub[i] * 100:.4f}", f"{(ub[i] - Bstar[s][i]) * 100:+.4f}"])
    wcsv("v2_T3_points.csv", ["curve", "arm", "seed", "server_use", "accuracy_pct", "minus_Bstar_pp"], rows)
    # mean over seeds at the comparison points
    rows = []
    def addmean(label, a, vals):
        vals = np.array(vals)
        rows.append([label, a] + [f"{vals[:, i].mean() * 100:.4f}" for i in range(len(OPTS))]
                    + [f"{(vals[:, i] - np.array([Bstar[s][i] for s in SEEDS])).mean() * 100:+.4f}" for i in range(len(OPTS))])
    addmean("B*", "best", [Bstar[s] for s in SEEDS])
    for a in ARMS:
        addmean("single tau, lambda_inf = lambda_t", a, [arm_curve[(a, s)][0] for s in SEEDS])
    for p, arms in POLS.items():
        for a in arms:
            addmean(p, a, [PC[(p, a, s)][0] for s in SEEDS])
    for a in ARMS:
        addmean("routing upper bound", a, [upper_bound(R[(a, s)], OS) for s in SEEDS])
    wcsv("v2_T4_points_mean.csv", ["curve", "arm"] + [f"acc_pct_at_{o}" for o in OPTS] + [f"minus_Bstar_pp_at_{o}" for o in OPTS], rows)
    # ---------- G and decisions
    rows = []
    for (p, a), st in sorted(Gs.items(), key=lambda kv: (kv[0][0], ARMS.index(kv[0][1]))):
        rows.append([p, a] + fmt(st))
    wcsv("v2_T5_G.csv", ["policy", "arm", "G_mean_pp", "ci95", "n_pos", "n", "per_seed_pp(5,6,7)"], rows)

    def decide(oracle, sigs, arms):
        up = {a: Gs[(oracle, a)]["mean"] for a in arms}
        lt = max(arms, key=lambda a: up[a])
        d = dict(G_oracle_pp=up, chosen_lambda_t=lt, G_oracle_chosen_pp=up[lt], threshold_pp=1.0)
        if up[lt] < 1.0:
            d.update(room="small", verdict=f"G({oracle}) = {up[lt]:+.2f} pp < 1.0 pp at {lt}: little room")
            return d
        d["room"] = "enough"
        res = {p: dict(G_pp=Gs[(p, lt)]["mean"], needed_pp=0.5 * up[lt], seeds_positive=Gs[(p, lt)]["n_pos"],
                       passes=bool(Gs[(p, lt)]["mean"] >= 0.5 * up[lt] and Gs[(p, lt)]["n_pos"] >= 2)) for p in sigs}
        cand = max(sigs, key=lambda p: Gs[(p, lt)]["mean"])
        diff = Gs[(sigs[0], lt)]["mean"] - Gs[(sigs[1], lt)]["mean"]
        if res[sigs[0]]["passes"] and res[sigs[1]]["passes"]:
            chosen = "x_SR" if diff >= 0.3 else "x_TV"
        elif res[cand]["passes"]:
            chosen = "x_SR" if cand == sigs[0] else "x_TV"
        else:
            chosen = None
        d.update(signals=res, candidate=cand, SR_minus_TV_pp=diff, chosen=chosen,
                 verdict=f"pass: {chosen}" if chosen else f"no pass (candidate {cand})")
        return d
    dec = {"question1_inference_lambda": decide("P1", ("P2", "P3"), ["T15", "T40"]),
           "question2_threshold": decide("Q1", ("Q2", "Q3"), ARMS),
           "definition": "G = mean over server use 0.5, 0.7, 0.9 of (policy - B*), B* = best single-tau curve of T15/T40/T60"}
    json.dump(dec, open(OUT / "decision_v2.json", "w"), indent=1)
    print(json.dumps(dec, indent=1))
    lt1, lt2 = dec["question1_inference_lambda"]["chosen_lambda_t"], dec["question2_threshold"]["chosen_lambda_t"]
    # ---------- splits at server use 0.9
    rows = []
    i9 = OPTS.index(0.9)
    for label, getter in (("B*", None), (f"P1 ({lt1})", ("P1", lt1)), (f"Q1 ({lt2})", ("Q1", lt2)), (f"Q2 ({lt2})", ("Q2", lt2))):
        vals = []
        for s in SEEDS:
            if getter is None:
                a = Bstar_arm[s][i9]
                D = R[(a, s)]
                H, bh, ba, _ = pol(D, "B")
                cfg = arm_curve[(a, s)][1][i9]
            else:
                p, a = getter
                D = R[(a, s)]
                H, bh, ba, _ = pol(D, p)
                cfg = PC[(p, a, s)][1][i9]
            vals.append(split_metrics(D, H, bh, ba, cfg))
            rows.append([label, s, a] + [f"{v * 100:.4f}" for v in vals[-1]])
        mm = np.mean(vals, axis=0)
        rows.append([label, "mean", ""] + [f"{v * 100:.4f}" for v in mm])
    wcsv("v2_T6_splits_at_0.9.csv", ["curve", "seed", "arm", "accuracy_pct", "home_pct", "away_pct", "main_pct", "nonmain_pct"], rows)
    # ---------- probe cost: AUROC and G with 8 / 16 / 64 probes
    rows = []
    for a in ARMS:
        for sig in SIG:
            for m in PROBES:
                vals = [auroc_away(R[(a, s)], sig, m) for s in SEEDS]
                rows.append([a, f"x_{sig.upper()}", m] + [f"{v[0]:.4f}" for v in vals] + [f"{np.mean([v[0] for v in vals]):.4f}",
                                                                                     " ".join(str(v[1]) for v in vals)])
    wcsv("v2_T7_probe_auroc.csv", ["arm", "signal", "probes", "auroc_s5", "auroc_s6", "auroc_s7", "auroc_mean",
                                   "clients_with_both_states"], rows)
    rows, PG = [], {}
    BstarN = {s: np.max([pcurve(R[(a, s)], "B", OS, ("Nw",))[0] for a in ARMS], axis=0) for s in SEEDS}
    for p, arms in (("P2", ["T15", "T40"]), ("P3", ["T15", "T40"]), ("Q2", ARMS), ("Q3", ARMS)):
        for a in arms:
            for m in PROBES:
                g_std = [np.mean(pcurve(R[(a, s)], p, OS, ("std",), m)[0] - Bstar[s]) * 100 for s in SEEDS]
                g_nw = [np.mean(pcurve(R[(a, s)], p, OS, ("Nw",), m)[0] - BstarN[s]) * 100 for s in SEEDS]
                pv = {s: pcurve(R[(a, s)], p, OS, ("probe", m), m)[0] - BstarN[s] for s in SEEDS}
                reach = [o for i, o in enumerate(OPTS) if all(np.isfinite(pv[s][i]) for s in SEEDS)]
                g_pr = [np.nanmean([pv[s][i] for i, o in enumerate(OPTS) if o in reach]) * 100 if reach else np.nan for s in SEEDS]
                PG[(p, a, m)] = (tstat(g_std), tstat(g_nw), tstat(g_pr) if reach else None, reach)
                rows.append([p, a, m] + fmt(tstat(g_std))[:3] + fmt(tstat(g_nw))[:3]
                            + (fmt(tstat(g_pr))[:3] if reach else ["", "", ""]) + [" ".join(map(str, reach)) or "none"]
                            + [" ".join(f"{(pv[s][i] * 100):+.2f}" if np.isfinite(pv[s][i]) else "nan" for s in SEEDS) for i in range(len(OPTS))])
    wcsv("v2_T8_probe_G.csv", ["policy", "arm", "probes", "G_pp", "ci95", "n_pos", "G_Nweighted_noprobecost_pp", "ci95_Nw",
                               "n_pos_Nw", "G_with_probe_cost_pp", "ci95_cost", "n_pos_cost", "server_use_points_reachable"]
         + [f"with_cost_minus_BstarN_pp_at_{o}(s5,s6,s7)" for o in OPTS], rows)
    # ---------- upper bound per arm (also in T3/T4) and the curves themselves on a grid
    rows = []
    grid = {}
    grid["B*"] = np.mean([Bstar_grid[s] for s in SEEDS], axis=0)
    for a in ARMS:
        grid[f"single tau {a}"] = np.mean([arm_grid[(a, s)] for s in SEEDS], axis=0)
        grid[f"upper bound {a}"] = np.mean([upper_bound(R[(a, s)], OGRID) for s in SEEDS], axis=0)
    for p, arms in POLS.items():
        for a in arms:
            grid[f"{p} {a}"] = np.mean([pcurve(R[(a, s)], p, OGRID)[0] for s in SEEDS], axis=0)
    for i, o in enumerate(OGRID):
        rows.append([o] + [f"{grid[k][i] * 100:.4f}" if np.isfinite(grid[k][i]) else "" for k in grid])
    wcsv("v2_T9_curves_grid.csv", ["server_use"] + [f"{k} acc_pct" for k in grid], rows)
    # ---------- manifest
    rows = []
    for a in ARMS:
        for s in SEEDS:
            D = R[(a, s)]
            h = D["h"]
            prov = JR / "provenance" / f"{h['config']['run_id']}.json"
            pv = json.load(open(prov)) if prov.exists() else {}
            gpu_index = (SEEDS.index(s) * len(ARMS) + ARMS.index(a)) % 4
            rows.append([h["config"]["run_id"], f"r8v2_{a}_s{s}", a, LT[a], s, 100 + s, f"runs/phaseT6_env/S1_seed{s}.npz",
                         "yes" if len(h["round"]) == 150 and len(h["eval"]) == 31 else "no", h["probe_eval_overlap"],
                         gpu_index, pv.get("env", {}).get("gpu", ""), pv.get("git_commit", "")[:7],
                         f"{h['total_time_sec'] / 60:.1f}", str(D["red"]["source_sha256"]), int(D["red"]["source_bytes"])])
    wcsv("v2_T10_run_manifest.csv", ["run_id", "run_name", "arm", "lambda_t", "seed", "model_seed", "env_file", "complete",
                                     "probe_eval_overlap", "gpu_index", "gpu", "git_commit", "wall_min",
                                     "requests_npz_sha256", "requests_npz_bytes"], rows)
    figures(grid, lt1, lt2, PG, R)


def figures(grid, lt1, lt2, PG, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    # (a) accuracy versus server use, and the difference from B*
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), layout="constrained")
    lines = [("B*", "B*: best fixed lambda and tau", INK, "-"),
             (f"upper bound {lt2}", f"routing upper bound (lambda_t {LT[lt2]:.2f})", MUTED, ":"),
             (f"P1 {lt1}", f"P1: lambda_inf by at_home (lambda_t {LT[lt1]:.2f})", BLUE, "--"),
             (f"P2 {lt1}", f"P2: lambda_inf by x_SR (lambda_t {LT[lt1]:.2f})", BLUE, "-"),
             (f"Q1 {lt2}", f"Q1: tau by at_home (lambda_t {LT[lt2]:.2f})", ORANGE, "--"),
             (f"Q2 {lt2}", f"Q2: tau by x_SR (lambda_t {LT[lt2]:.2f})", ORANGE, "-")]
    m = OGRID >= 0.3
    for key, lab, col, ls in lines:
        axes[0].plot(OGRID[m], grid[key][m] * 100, color=col, ls=ls, label=lab, lw=1.6 if key == "B*" else 1.3)
        if key not in ("B*",) and not key.startswith("upper"):
            axes[1].plot(OGRID[m], (grid[key][m] - grid["B*"][m]) * 100, color=col, ls=ls, label=lab)
    axes[1].axhline(0, color=INK2, lw=0.8)
    for ax in axes:
        for o in OPTS:
            ax.axvline(o, color=AXIS, lw=0.6, zorder=0)
        ax.set_xlabel("server use (share of requests answered by the server exit)")
        ax.set_xlim(0.3, 1.0)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].set_ylabel("difference from B* (pp)")
    axes[0].set_title("(a) accuracy", fontsize=8)
    axes[1].set_title("(b) difference from B*", fontsize=8)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(FIG / "v2_figA_accuracy_vs_server_use.pdf")
    fig.savefig(FIG / "v2_figA_accuracy_vs_server_use.png", dpi=300)
    plt.close(fig)
    # (b) probes: AUROC and G
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), layout="constrained")
    for sig, col, lab in (("sr", GREEN, "x_SR"), ("tv", PURPLE, "x_TV")):
        for a, ls in ((lt1, "-"), (lt2, "--")) if lt1 != lt2 else ((lt1, "-"),):
            vals = [np.mean([auroc_away(R[(a, s)], sig, mm)[0] for s in SEEDS]) for mm in PROBES]
            axes[0].plot(PROBES, vals, color=col, ls=ls, marker="o", ms=3, label=f"{lab} (lambda_t {LT[a]:.2f})")
    axes[0].set_ylabel("AUROC for away")
    axes[0].set_title("(a) signal AUROC", fontsize=8)
    for idx, title in ((0, "(b) G, server use of the evaluation requests"), (2, "(c) G with probes counted as server use")):
        ax = axes[1] if idx == 0 else axes[2]
        for p, a, col, ls in (("P2", lt1, BLUE, "-"), ("P3", lt1, BLUE, "--"), ("Q2", lt2, ORANGE, "-"), ("Q3", lt2, ORANGE, "--")):
            ys = [PG[(p, a, mm)][idx]["mean"] if PG[(p, a, mm)][idx] is not None else np.nan for mm in PROBES]
            ax.plot(PROBES, ys, color=col, ls=ls, marker="o", ms=3, label=f"{p} (lambda_t {LT[a]:.2f})")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("G (pp)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(PROBES)
        ax.set_xticklabels([str(p) for p in PROBES])
        ax.set_xlabel("probe requests per round")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[1].legend(fontsize=6, loc="best")
    fig.savefig(FIG / "v2_figB_probes.pdf")
    fig.savefig(FIG / "v2_figB_probes.png", dpi=300)
    plt.close(fig)
    with open(FIG / "v2_figure_captions.md", "w") as f:
        f.write("# Round 8 check v2 figure captions\n\n## v2_figA_accuracy_vs_server_use\n\n"
                "Accuracy against the share of requests answered by the server exit (a) and the difference from B* (b). "
                "B* is the best curve over the fixed training ratios 0.15, 0.4 and 0.6 with one entropy threshold. "
                f"P1 and P2 change the inference mixing ratio by the true home state and by the x_SR signal (training ratio {LT[lt1]:.2f}). "
                f"Q1 and Q2 use separate thresholds at home and away by the true home state and by x_SR (training ratio {LT[lt2]:.2f}). "
                "The dotted curve is the routing upper bound. Commute mobility, means over seeds 5 to 7. "
                "Vertical lines mark the comparison points 0.5, 0.7 and 0.9.\n\n"
                "## v2_figB_probes\n\n"
                "AUROC of the two signals for detecting that a client is away (a) and the gain G of the signal-driven "
                "policies (b, c) when the signals use 8, 16 or 64 probe requests per round. In (c) the probe requests are "
                "counted as requests answered by the server exit, with the actual request counts of the environment; "
                "G is averaged over the comparison points that the policy can reach. Means over seeds 5 to 7.\n")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
