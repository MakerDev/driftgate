"""Round 9: which exit answers each request, computed from the Round 8 per-request records (no new run).

Inputs: runs/phaseT8_check/r8v2_{T15,T40,T60}_s{5,6,7}.json and *_requests.npz (Round 8 v2). Outputs: tables/*.csv,
decision.json, figures/*.

Per request i of client k in evaluation round r and inference block b (lambda_inf in {0.15, 0.3, 0.4, 0.55, 0.7};
T60 also has its installed model, lambda 0.6): client-exit prediction c_b(i), entropy e_b(i), server-exit prediction
s_b(i), label y(i), kind (Main / OOP / OOR). M_k = the client's Main classes, rebuilt with the partition code
(src.r6_requests.build_partition) from each run's own configuration; rules use M_k, never the label or the kind
(oracles excepted).

Placement A (server block copy on the device; both exits for every request):
  A-fixed     entropy routing with one tau (grid 0..2.30 step 0.05, infinity = client exit only, and server exit only)
  A-scope     s_b if s_b not in M_k, else c_b
  A-scope-tau s_b if s_b not in M_k and e_b > tau, else c_b (whole tau grid, infinity = client exit only)
  A-two-block c from block a, s from block b, then A-scope (all pairs of the lambda_inf grid)
  A-kind      oracle: c_b for Main requests, s_b otherwise (one block, and the best pair a, b)
Placement B (server exit only for requests sent to the edge):
  B-fixed     entropy routing, best (lambda_t, lambda_inf, tau) at each server use (Round 8 B* with lambda_inf free)
  B-scope     send if c_b not in M_k or e_b > tau (tau grid; infinity = send only c_b not in M_k; plus send all)
  B-kind      oracle: send exactly the non-Main requests (one point)
Accuracy: per evaluation round the mean over clients, then the mean over the rounds of the window.
Windows: late = evaluation rounds 30..150 (25 rounds; used by the decision), full = all 31 rounds.
Server use: requests answered by the server exit / all requests of the window.
G_A = A-scope at its best (lambda_t, lambda_inf) - A-fixed at its best (lambda_t, lambda_inf, tau); both settings are
      chosen by the seed mean, then the same-seed differences at those settings.
G_B = mean over server use 0.3, 0.5, 0.7 of (best B-scope curve - B-fixed), maxima over (lambda_t, lambda_inf) per seed.
Decision (late window): A passes if mean G_A >= 3.0 pp and G_A > 0 in all three seeds; B passes if mean G_B >= 2.0 pp
and G_B > 0 in all three seeds.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
R8RUNS = Path(os.environ.get("R9_RUNS", JR / "runs" / "phaseT8_check"))     # overrides only for smoke tests
R8RED = JR / "artifacts" / "driftgate_tmc_r8_check" / "reduced"
OUT = Path(os.environ.get("R9_OUT", HERE))
LATE_MIN = int(os.environ.get("R9_LATE_MIN", 30))
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_requests import build_partition  # noqa: E402

ARMS = ["T15", "T40", "T60"]
LT = {"T15": 0.15, "T40": 0.4, "T60": 0.6}
SEEDS = [5, 6, 7]
GRID = [0.15, 0.3, 0.4, 0.55, 0.7]
TAU = np.round(np.arange(0, 2.30 + 1e-9, 0.05), 2)
TAU32 = TAU.astype(np.float32)
NJ = len(TAU) + 1            # tau grid + infinity (index 47)
J_INF, J_SRV = NJ - 1, NJ    # column 48: server exit only / send everything
OPTS_B = [0.3, 0.5, 0.7]
WINDOWS = ["late", "full"]
KINDS = ["Main", "OOP", "OOR"]
OGRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


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
                per_seed=[float(x) for x in v])


def fmt(st):
    return [f"{st['mean']:+.4f}", f"[{st['lo']:+.2f}, {st['hi']:+.2f}]", st["n_pos"], " ".join(f"{x:+.2f}" for x in st["per_seed"])]


# ------------------------------------------------------------------------------------------------ Main classes
_TRAIN = None


def rebuild_mains(h):
    """M_k for a run from its own configuration (partition seed, environment file) with the partition code."""
    global _TRAIN
    if _TRAIN is None:
        from data.partition import get_cifar10
        _TRAIN = np.array(get_cifar10(str(ROOT / "data_cache"))[0].targets)
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    seed = int(h["config"]["partition_seed"])
    env, _ = r6_env.load_env(h["config"]["env_file"])
    _, mains, _, _ = build_partition(_TRAIN, env, seed, cfg["num_classes"], cfg)
    return {int(k): sorted(int(c) for c in v) for k, v in mains.items()}


# ------------------------------------------------------------------------------------------------ per run
def process_run(arm, s):
    h = json.load(open(R8RUNS / f"r8v2_{arm}_s{s}.json"))
    q = dict(np.load(R8RUNS / f"r8v2_{arm}_s{s}_requests.npz"))
    mains = rebuild_mains(h)
    prov = JR / "provenance" / f"{h['config']['run_id']}.json"
    prov_main = json.load(open(prov)).get("client_main") if prov.exists() else None
    K = 50
    er = np.array(q["eval_rounds"])
    E = len(er)
    ek, kk = q["req_eval_index"].astype(np.int64), q["req_client"].astype(np.int64)
    g = ek * K + kk
    G = E * K
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"].astype(np.int64)
    M = np.zeros((K, 10), bool)
    for k, v in mains.items():
        M[k, v] = True
    checks = dict(mains_equal_provenance=(prov_main == {str(k): v for k, v in mains.items()}) if prov_main else None,
                  label_in_Mk_equals_kind_Main=bool(np.array_equal(kind == 0, M[kk, y])))
    blocks = [float(v) for v in q["block_lambda"]]
    ib = int(q["installed_block"])
    D = dict(arm=arm, seed=s, h=h, er=er, E=E, K=K, blocks=blocks, ib=ib, checks=checks, mains=mains,
             n=np.bincount(g, minlength=G).reshape(E, K).astype(float),
             nkind=np.stack([np.bincount(g, weights=kind == i, minlength=G) for i in range(3)], -1).reshape(E, K, 3),
             home=np.zeros(G, bool))
    D["home"][g] = q["req_home"]
    D["home"] = D["home"].reshape(E, K)
    W1 = NJ + 1

    def curve_counts(j0, cc, sc):
        """client answers at column j (j < NJ) iff j >= j0; j0 = NJ: never; column NJ: all to the server."""
        idx = g * W1 + j0
        dh = np.bincount(idx, weights=cc.astype(np.int64) - sc.astype(np.int64), minlength=G * W1).reshape(G, W1)
        nh = np.bincount(idx, minlength=G * W1).reshape(G, W1)
        scg = np.bincount(g, weights=sc, minlength=G)
        corr = np.empty((G, W1))
        srv = np.empty((G, W1))
        corr[:, :NJ] = scg[:, None] + np.cumsum(dh, axis=1)[:, :NJ]
        srv[:, :NJ] = D["n"].ravel()[:, None] - np.cumsum(nh, axis=1)[:, :NJ]
        corr[:, NJ] = scg
        srv[:, NJ] = D["n"].ravel()
        return corr.reshape(E, K, W1), srv.reshape(E, K, W1)

    D["fix_c"], D["fix_s"], D["bsc_c"], D["bsc_s"], D["ast_c"], D["ast_s"] = [], [], [], [], [], []
    D["asc_c"], D["asc_s"], D["kor_c"] = [], [], []
    keep = {}
    for b in range(len(blocks)):
        cp, sp, ent = q["cp"][b].astype(np.int64), q["sp"][b].astype(np.int64), q["ent"][b]
        cc, sc = cp == y, sp == y
        j0 = np.searchsorted(TAU32, ent, side="left")
        cinM, sinM = M[kk, cp], M[kk, sp]
        c, sv = curve_counts(j0, cc, sc)                                   # A-fixed / B-fixed
        D["fix_c"].append(c), D["fix_s"].append(sv)
        c, sv = curve_counts(np.where(cinM, j0, NJ), cc, sc)               # B-scope (send if c not in M or e > tau)
        D["bsc_c"].append(c), D["bsc_s"].append(sv)
        c, sv = curve_counts(np.where(sinM, 0, j0), cc, sc)                # A-scope-tau (server iff s not in M, e > tau)
        D["ast_c"].append(c[..., :NJ]), D["ast_s"].append(sv[..., :NJ])
        D["asc_c"].append(np.bincount(g, weights=np.where(sinM, cc, sc), minlength=G).reshape(E, K))   # A-scope
        D["asc_s"].append(np.bincount(g, weights=~sinM, minlength=G).reshape(E, K))
        D["kor_c"].append(np.bincount(g, weights=np.where(kind == 0, cc, sc), minlength=G).reshape(E, K))  # A-kind
        keep[b] = (cc, sc, sinM)
    gb = [i for i, lam in enumerate(blocks) if any(abs(lam - x) < 1e-9 for x in GRID)]
    D["grid_blocks"] = gb
    D["pair_scope"], D["pair_kind"] = {}, {}
    for a in gb:
        for b in gb:
            cca, (_, scb, sinMb) = keep[a][0], keep[b]
            D["pair_scope"][(a, b)] = np.bincount(g, weights=np.where(sinMb, cca, scb), minlength=G).reshape(E, K)
            D["pair_kind"][(a, b)] = np.bincount(g, weights=np.where(kind == 0, cca, scb), minlength=G).reshape(E, K)
    # recomputation check against the Round 8 reduced counts (every block and tau) and the installed model
    red = R8RED / f"r8v2_{arm}_s{s}_reduced.npz"
    if red.exists():
        r = np.load(red)
        D["checks"]["fixed_counts_equal_round8_reduced"] = bool(all(
            np.array_equal(np.rint(D["fix_c"][b][..., :NJ]), r["corr"][b]) and np.array_equal(np.rint(D["fix_s"][b][..., :NJ]), r["nsrv"][b])
            for b in range(len(blocks))))
    print(f"  {arm} s{s}: {len(y)} requests, {len(blocks)} blocks, checks {D['checks']}", flush=True)
    return D


# ------------------------------------------------------------------------------------------------ accuracy helpers
def wmask(D, w):
    return np.ones(D["E"], bool) if w == "full" else D["er"] >= LATE_MIN


def acc(D, corr, w):
    """corr [E, K] or [E, K, J] -> accuracy (or per column) over the window."""
    m = wmask(D, w)
    a = corr[m] / (D["n"][m][..., None] if corr.ndim == 3 else D["n"][m])
    return a.mean(axis=(0, 1))


def srv_use(D, srv, w):
    m = wmask(D, w)
    return srv[m].sum(axis=(0, 1)) / D["n"][m].sum()


def interp(R, A, os):
    """R, A [P] with R non-increasing; largest linear interpolation value at each o (nan if outside)."""
    hi, lo, ah, al = R[:-1], R[1:], A[:-1], A[1:]
    O = np.asarray(os)[None, :]
    inside = (lo[:, None] <= O + 1e-12) & (O <= hi[:, None] + 1e-12)
    span = (hi - lo)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.clip(np.where(span > 0, (O - lo[:, None]) / np.where(span > 0, span, 1), 1.0), 0, 1)
    v = np.where(span > 0, w * ah[:, None] + (1 - w) * al[:, None], np.maximum(ah, al)[:, None])
    v = np.where(inside, v, -np.inf)
    out = v.max(axis=0)
    out[np.isinf(out)] = np.nan
    return out


ORDER = [J_SRV] + list(range(NJ))   # server only / send all, then tau ascending (server use non-increasing)


def bcurve(D, key, b, w, os):
    c, sv = D[f"{key}_c"][b], D[f"{key}_s"][b]
    return interp(srv_use(D, sv, w)[ORDER], acc(D, c, w)[ORDER], os)


# ------------------------------------------------------------------------------------------------ split metrics
def split_from_answers(D, R, w, ans_client, correct):
    """home / away and Main / OOP / OOR accuracy (Round 6 definitions) and wrong-exit shares for one rule setting,
    from per-request booleans (the client exit answers; the answer is correct)."""
    q, (g, kind) = R
    E, K = D["E"], D["K"]
    m = wmask(D, w)
    G = E * K
    corr = np.bincount(g, weights=correct, minlength=G).reshape(E, K)[m]
    n = D["n"][m]
    home = D["home"][m]
    a = corr / n
    out = dict(acc=a.mean(), home=np.mean([a[e][home[e]].mean() for e in range(len(a)) if home[e].any()]),
               away=np.mean([a[e][~home[e]].mean() for e in range(len(a)) if (~home[e]).any()]))
    for i, kn in enumerate(KINDS):
        ck = np.bincount(g, weights=correct & (kind == i), minlength=G).reshape(E, K)[m]
        nk = D["nkind"][m][..., i]
        with np.errstate(invalid="ignore", divide="ignore"):
            ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
        out[kn] = np.mean([np.nanmean(ak[e]) for e in range(len(ak)) if np.isfinite(ak[e]).any()])
    inw = m[q["req_eval_index"]]
    main = kind == 0
    out["main_to_server"] = float((~ans_client & main & inw).sum() / max((main & inw).sum(), 1))
    out["nonmain_at_client"] = float((ans_client & ~main & inw).sum() / max((~main & inw).sum(), 1))
    out["server_use"] = float((~ans_client & inw).sum() / max(inw.sum(), 1))
    return out


def rule_answers(D, q, rule, b, a=None, j=None):
    """per-request (client exit answers, correct) for a rule setting."""
    K = D["K"]
    kk = q["req_client"].astype(np.int64)
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"]
    M = np.zeros((K, 10), bool)
    for k, v in D["mains"].items():
        M[k, v] = True
    cpb, spb, eb = q["cp"][b].astype(np.int64), q["sp"][b].astype(np.int64), q["ent"][b]
    cpa = q["cp"][a].astype(np.int64) if a is not None else cpb
    def client_if_tau(j):
        return np.ones(len(y), bool) if j == J_INF else (np.zeros(len(y), bool) if j == J_SRV else eb <= TAU32[j])
    if rule == "A-fixed" or rule == "B-fixed":
        ac = client_if_tau(j)
    elif rule == "A-scope":
        ac = M[kk, spb]
    elif rule == "A-scope-tau":
        ac = M[kk, spb] | client_if_tau(j)
    elif rule == "A-two-block":
        ac = M[kk, spb]
    elif rule == "A-kind":
        ac = kind == 0
    elif rule == "B-scope":
        ac = M[kk, cpb] & client_if_tau(j) if j != J_SRV else np.zeros(len(y), bool)
    elif rule == "B-kind":
        ac = kind == 0
    else:
        raise ValueError(rule)
    correct = np.where(ac, cpa == y, spb == y)
    return ac, correct


# ------------------------------------------------------------------------------------------------ main
def main():
    R = {(a, s): process_run(a, s) for a in ARMS for s in SEEDS}
    D0 = R[(ARMS[0], SEEDS[0])]
    rows = []
    for (a, s), D in R.items():
        rows.append([a, s, D["h"]["config"]["run_id"]] + [str(D["checks"].get(k)) for k in
                    ("mains_equal_provenance", "label_in_Mk_equals_kind_Main", "fixed_counts_equal_round8_reduced")])
    wcsv("R9_T0_checks.csv", ["arm", "seed", "run_id", "Mk_rebuilt_equals_provenance_client_main",
                              "label_in_Mk_equals_kind_Main_all_requests", "tau_counts_equal_round8_reduced"], rows)
    # ---------- recomputation of Round 8 at lambda_inf = lambda_t, tau = 0.8
    j08 = int(np.flatnonzero(np.isclose(TAU, 0.8))[0])
    rows = []
    for a in ARMS:
        vals = {w: [(acc(R[(a, s)], R[(a, s)]["fix_c"][R[(a, s)]["ib"]], w)[j08],
                     srv_use(R[(a, s)], R[(a, s)]["fix_s"][R[(a, s)]["ib"]], w)[j08]) for s in SEEDS] for w in WINDOWS}
        rows.append([a] + [f"{np.mean([v[0] for v in vals[w]]) * 100:.4f}" for w in WINDOWS]
                    + [f"{np.mean([v[1] for v in vals[w]]) * 100:.2f}" for w in WINDOWS]
                    + [" ".join(f"{v[0] * 100:.4f}" for v in vals["full"]),
                       " ".join(f"{R[(a, s)]['h']['integrated_acc'] * 100:.4f}" for s in SEEDS)])
    wcsv("R9_T1_recompute_tau08.csv", ["arm", "acc_late_pct", "acc_full_pct", "server_use_late_pct", "server_use_full_pct",
                                       "acc_full_per_seed_pct", "round8_json_integrated_per_seed_pct"], rows)
    # ---------- placement A: candidates (accuracy per seed) and the best settings by the seed mean
    res = {}
    best = {}
    for w in WINDOWS:
        cand = {}
        for a in ARMS:
            for b, lam in enumerate(R[(a, SEEDS[0])]["blocks"]):
                key = (a, lam)
                for j in range(NJ + 1):
                    cand[("A-fixed", a, lam, None, j)] = [acc(R[(a, s)], R[(a, s)]["fix_c"][b], w)[j] for s in SEEDS]
                cand[("A-scope", a, lam, None, None)] = [acc(R[(a, s)], R[(a, s)]["asc_c"][b], w) for s in SEEDS]
                for j in range(NJ):
                    cand[("A-scope-tau", a, lam, None, j)] = [acc(R[(a, s)], R[(a, s)]["ast_c"][b], w)[j] for s in SEEDS]
                cand[("A-kind", a, lam, None, None)] = [acc(R[(a, s)], R[(a, s)]["kor_c"][b], w) for s in SEEDS]
            gb = R[(a, SEEDS[0])]["grid_blocks"]
            for ia in gb:
                for ibb in gb:
                    la, lb = R[(a, SEEDS[0])]["blocks"][ia], R[(a, SEEDS[0])]["blocks"][ibb]
                    cand[("A-two-block", a, lb, la, None)] = [acc(R[(a, s)], R[(a, s)]["pair_scope"][(ia, ibb)], w) for s in SEEDS]
                    cand[("A-kind-pair", a, lb, la, None)] = [acc(R[(a, s)], R[(a, s)]["pair_kind"][(ia, ibb)], w) for s in SEEDS]
        res[w] = cand
        for rule in ("A-fixed", "A-scope", "A-scope-tau", "A-two-block", "A-kind", "A-kind-pair"):
            ks = [k for k in cand if k[0] == rule]
            best[(w, rule)] = max(ks, key=lambda k: np.mean(cand[k]))
    # ---------- placement B: per seed, at each server use, maxima over (lambda_t, lambda_inf)
    bres = {}
    for w in WINDOWS:
        for s in SEEDS:
            for key in ("fix", "bsc"):
                curves = {(a, lam): bcurve(R[(a, s)], key, b, w, OGRID) for a in ARMS
                          for b, lam in enumerate(R[(a, s)]["blocks"])}
                bres[(w, s, key)] = curves
    # ---------- G_A, G_B and the decision
    def tau_txt(j):
        return "inf (client exit only)" if j == J_INF else ("server exit only" if j == J_SRV else (f"{TAU[j]:.2f}" if j is not None else ""))

    iopt = [int(np.flatnonzero(np.isclose(OGRID, o))[0]) for o in OPTS_B]
    G = {}
    rows = []
    for w in WINDOWS:
        bf, bs = best[(w, "A-fixed")], best[(w, "A-scope")]
        G[(w, "A")] = tstat((np.array(res[w][bs]) - np.array(res[w][bf])) * 100)
        for rule in ("A-scope-tau", "A-two-block", "A-kind", "A-kind-pair"):
            G[(w, rule)] = tstat((np.array(res[w][best[(w, rule)]]) - np.array(res[w][bf])) * 100)
        gB, diffs = [], []
        for s in SEEDS:
            fx = np.nanmax([v for v in bres[(w, s, "fix")].values()], axis=0)
            sc = np.nanmax([v for v in bres[(w, s, "bsc")].values()], axis=0)
            d = (sc[iopt] - fx[iopt]) * 100
            diffs.append(d)
            gB.append(np.mean(d))
        G[(w, "B")] = tstat(gB)
        G[(w, "B_points")] = np.array(diffs)
        for rule in ("A-fixed", "A-scope", "A-scope-tau", "A-two-block", "A-kind", "A-kind-pair"):
            k = best[(w, rule)]
            rows.append([w, rule, k[1], k[2], k[3] if k[3] is not None else "", tau_txt(k[4]),
                         " ".join(f"{x * 100:.2f}" for x in res[w][k]), f"{np.mean(res[w][k]) * 100:.4f}"]
                        + (fmt(G[(w, "A")]) if rule == "A-scope" else fmt(G[(w, rule)]) if rule != "A-fixed" else ["", "", "", ""]))
    wcsv("R9_T2_placementA.csv", ["window", "rule", "lambda_t_arm", "lambda_inf(server block for pairs)",
                                  "lambda_inf_client_block(pairs)", "tau", "acc_pct_s5_s6_s7", "acc_mean_pct",
                                  "minus_best_A_fixed_pp", "ci95", "seeds_positive", "per_seed"], rows)
    rows = []
    for w in WINDOWS:
        for s in SEEDS:
            fx = {k: v[iopt] for k, v in bres[(w, s, "fix")].items()}
            sc = {k: v[iopt] for k, v in bres[(w, s, "bsc")].items()}
            for i, o in enumerate(OPTS_B):
                kf = max(fx, key=lambda k: -np.inf if np.isnan(fx[k][i]) else fx[k][i])
                ks = max(sc, key=lambda k: -np.inf if np.isnan(sc[k][i]) else sc[k][i])
                rows.append([w, s, o, f"{fx[kf][i] * 100:.4f}", f"{kf[0]} lambda_inf {kf[1]:g}", f"{sc[ks][i] * 100:.4f}",
                             f"{ks[0]} lambda_inf {ks[1]:g}", f"{(sc[ks][i] - fx[kf][i]) * 100:+.4f}"])
        rows.append([w, "G_B", ""] + ["", "", "", ""] + [f"{G[(w, 'B')]['mean']:+.4f} {fmt(G[(w, 'B')])[1]} ({G[(w, 'B')]['n_pos']}/3)"])
    wcsv("R9_T3_placementB.csv", ["window", "seed", "server_use", "B_fixed_acc_pct", "B_fixed_setting", "B_scope_acc_pct",
                                  "B_scope_setting", "B_scope_minus_B_fixed_pp"], rows)
    gA, gB = G[("late", "A")], G[("late", "B")]
    dec = dict(window="late (evaluation rounds 30-150)",
               A=dict(G_A_pp=gA["mean"], ci95=[gA["lo"], gA["hi"]], per_seed=gA["per_seed"], threshold_pp=3.0,
                      passes=bool(gA["mean"] >= 3.0 and gA["n_pos"] == 3)),
               B=dict(G_B_pp=gB["mean"], ci95=[gB["lo"], gB["hi"]], per_seed=gB["per_seed"], threshold_pp=2.0,
                      passes=bool(gB["mean"] >= 2.0 and gB["n_pos"] == 3)),
               full_window=dict(G_A_pp=G[("full", "A")]["mean"], G_B_pp=G[("full", "B")]["mean"]),
               best_settings={w: {r: list(map(str, best[(w, r)])) for r in ("A-fixed", "A-scope")} for w in WINDOWS})
    dec["verdict"] = ("A passes" if dec["A"]["passes"] else "A does not pass") + "; " + \
                     ("B passes" if dec["B"]["passes"] else "B does not pass")
    if not dec["A"]["passes"] and not dec["B"]["passes"]:
        dec["verdict"] += " (neither passes)"
    json.dump(dec, open(OUT / "decision.json", "w"), indent=1)
    print(json.dumps(dec, indent=1))
    # ---------- splits and wrong-exit shares at the best settings
    rows = []
    cache = {}

    def runq(a, s):
        if (a, s) not in cache:
            q = dict(np.load(R8RUNS / f"r8v2_{a}_s{s}_requests.npz"))
            cache[(a, s)] = (q, (q["req_eval_index"].astype(np.int64) * 50 + q["req_client"].astype(np.int64), q["req_kind"]))
        return cache[(a, s)]

    def bidx(D, lam):
        return [i for i, x in enumerate(D["blocks"]) if abs(x - lam) < 1e-9][0]

    for w in WINDOWS:
        specs = [(r, best[(w, r)]) for r in ("A-fixed", "A-scope", "A-scope-tau", "A-two-block", "A-kind", "A-kind-pair")]
        bk = max([(a, lam) for a in ARMS for lam in R[(a, SEEDS[0])]["blocks"]],
                 key=lambda k: np.mean([acc(R[(k[0], s)], R[(k[0], s)]["kor_c"][bidx(R[(k[0], s)], k[1])], w) for s in SEEDS]))
        for rule, k in specs + [("B-kind", ("B-kind", bk[0], bk[1], None, None))]:
            ms = []
            for s in SEEDS:
                D = R[(k[1], s)]
                q, Rg = runq(k[1], s)
                b = bidx(D, k[2])
                a_ = bidx(D, k[3]) if k[3] is not None else None
                r_ = {"A-kind-pair": "A-kind"}.get(rule, rule)
                ac, correct = rule_answers(D, q, r_, b, a_, k[4])
                ms.append(split_from_answers(D, (q, Rg), w, ac, correct))
            mm = {key: np.mean([m[key] for m in ms]) for key in ms[0]}
            rows.append([w, rule, k[1], k[2], k[3] if k[3] is not None else "", tau_txt(k[4])]
                        + [f"{mm[c] * 100:.2f}" for c in ("acc", "home", "away", "Main", "OOP", "OOR", "server_use",
                                                          "main_to_server", "nonmain_at_client")])
        # placement B at each server use: best setting per seed, interpolated between its two tau columns
        for key, rule in (("fix", "B-fixed"), ("bsc", "B-scope")):
            for i, o in enumerate(OPTS_B):
                ms, labs = [], []
                for s in SEEDS:
                    curves = bres[(w, s, key)]
                    kbest = max(curves, key=lambda kk_: -np.inf if np.isnan(curves[kk_][iopt[i]]) else curves[kk_][iopt[i]])
                    D = R[(kbest[0], s)]
                    b = bidx(D, kbest[1])
                    Rs = srv_use(D, D[f"{key}_s"][b], w)[ORDER]
                    js = np.array(ORDER)
                    pair = [p for p in range(len(Rs) - 1) if Rs[p + 1] - 1e-12 <= o <= Rs[p] + 1e-12]
                    As = acc(D, D[f"{key}_c"][b], w)[ORDER]
                    p = max(pair, key=lambda p_: (o - Rs[p_ + 1]) / max(Rs[p_] - Rs[p_ + 1], 1e-15) * As[p_]
                            + (1 - (o - Rs[p_ + 1]) / max(Rs[p_] - Rs[p_ + 1], 1e-15)) * As[p_ + 1])
                    wt = (o - Rs[p + 1]) / max(Rs[p] - Rs[p + 1], 1e-15)
                    q, Rg = runq(kbest[0], s)
                    m1 = split_from_answers(D, (q, Rg), w, *rule_answers(D, q, rule, b, None, int(js[p])))
                    m2 = split_from_answers(D, (q, Rg), w, *rule_answers(D, q, rule, b, None, int(js[p + 1])))
                    ms.append({kx: wt * m1[kx] + (1 - wt) * m2[kx] for kx in m1})
                    labs.append(f"s{s}:{kbest[0]}/{kbest[1]:g}/tau {tau_txt(int(js[p]))}-{tau_txt(int(js[p + 1]))}")
                mm = {kx: np.mean([m[kx] for m in ms]) for kx in ms[0]}
                rows.append([w, f"{rule} at server use {o}", "per seed", "", "", "; ".join(labs)]
                            + [f"{mm[c] * 100:.2f}" for c in ("acc", "home", "away", "Main", "OOP", "OOR", "server_use",
                                                              "main_to_server", "nonmain_at_client")])
    wcsv("R9_T4_splits.csv", ["window", "rule", "lambda_t_arm", "lambda_inf(server block)", "lambda_inf_client_block",
                              "tau", "acc_pct", "home_pct", "away_pct", "Main_pct", "OOP_pct", "OOR_pct", "server_use_pct",
                              "Main_sent_to_server_pct", "nonMain_kept_at_client_pct"], rows)
    # ---------- A-scope-tau curve and B-kind points (reference)
    rows = []
    for w in WINDOWS:
        k = best[(w, "A-scope-tau")]
        D = R[(k[1], SEEDS[0])]
        b = bidx(D, k[2])
        for j in range(NJ):
            v = [acc(R[(k[1], s)], R[(k[1], s)]["ast_c"][b], w)[j] for s in SEEDS]
            rows.append([w, k[1], k[2], tau_txt(j), f"{np.mean(v) * 100:.4f}"])
    wcsv("R9_T5_A_scope_tau_curve.csv", ["window", "lambda_t_arm", "lambda_inf", "tau", "acc_pct_mean"], rows)
    rows = []
    for w in WINDOWS:
        for a in ARMS:
            for b, lam in enumerate(R[(a, SEEDS[0])]["blocks"]):
                av = [acc(R[(a, s)], R[(a, s)]["kor_c"][b], w) for s in SEEDS]
                su = [R[(a, s)]["nkind"][wmask(R[(a, s)], w)][..., 1:].sum() / R[(a, s)]["n"][wmask(R[(a, s)], w)].sum() for s in SEEDS]
                bf = [np.nanmax([interp(srv_use(R[(a2, s)], R[(a2, s)]["fix_s"][b2], w)[ORDER],
                                        acc(R[(a2, s)], R[(a2, s)]["fix_c"][b2], w)[ORDER], [su[i]])[0]
                                 for a2 in ARMS for b2 in range(len(R[(a2, s)]["blocks"]))]) for i, s in enumerate(SEEDS)]
                rows.append([w, a, lam, f"{np.mean(su) * 100:.2f}", f"{np.mean(av) * 100:.4f}", f"{np.mean(bf) * 100:.4f}",
                             f"{(np.mean(av) - np.mean(bf)) * 100:+.4f}"])
    wcsv("R9_T6_B_kind_oracle_points.csv", ["window", "lambda_t_arm", "lambda_inf", "server_use_pct(non-Main share)",
                                            "acc_pct", "B_fixed_acc_at_same_server_use_pct", "difference_pp"], rows)
    # ---------- curves for the figure and a table
    rows = []
    grid = {}
    for w in WINDOWS:
        grid[(w, "B-fixed")] = np.mean([np.nanmax(list(bres[(w, s, "fix")].values()), axis=0) for s in SEEDS], axis=0)
        grid[(w, "B-scope envelope")] = np.mean([np.nanmax(list(bres[(w, s, "bsc")].values()), axis=0) for s in SEEDS], axis=0)
        cfgs = list(bres[(w, SEEDS[0], "bsc")].keys())
        kbest = max(cfgs, key=lambda c: np.nanmean([np.nanmean(bres[(w, s, "bsc")][c][iopt]) for s in SEEDS]))
        grid[(w, "B-scope best")] = np.mean([bres[(w, s, "bsc")][kbest] for s in SEEDS], axis=0)
        grid[(w, "B-scope best setting")] = kbest
        for i, o in enumerate(OGRID):
            rows.append([w, o] + [f"{grid[(w, nm)][i] * 100:.4f}" if np.isfinite(grid[(w, nm)][i]) else ""
                                  for nm in ("B-fixed", "B-scope envelope", "B-scope best")])
    wcsv("R9_T7_B_curves.csv", ["window", "server_use", "B_fixed_acc_pct", "B_scope_envelope_acc_pct",
                                "B_scope_best_setting_acc_pct"], rows)
    figures(res, best, grid, G)


def figures(res, best, grid, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    # (a) placement A bars
    rules = [("A-fixed", "fixed tau\n(best)"), ("A-scope", "scope rule"), ("A-scope-tau", "scope rule\nwith tau"),
             ("A-two-block", "scope rule,\ntwo blocks"), ("A-kind", "kind oracle"), ("A-kind-pair", "kind oracle,\ntwo blocks")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
    x = np.arange(len(rules))
    for off, w, col, lab in ((-0.2, "late", BLUE, "rounds 30-150"), (0.2, "full", ORANGE, "all rounds")):
        vals = np.array([res[w][best[(w, r)]] for r, _ in rules]) * 100      # [rules, seeds]
        m = vals.mean(axis=1)
        ax.bar(x + off, m, width=0.38, color=col, label=lab, zorder=2)
        for i in range(len(rules)):
            ax.plot([x[i] + off] * 3, vals[i], ls="none", marker="o", ms=2.2, color=INK, zorder=3)
            ax.text(x[i] + off, m[i] + 0.3, f"{m[i]:.1f}", ha="center", va="bottom", fontsize=6, color=INK2)
    lo = min(np.min([res[w][best[(w, r)]] for r, _ in rules for w in WINDOWS]) * 100 - 3, 55)
    ax.set_ylim(lo, None)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in rules], fontsize=7)
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(FIG / "R9_figA_placementA.pdf")
    fig.savefig(FIG / "R9_figA_placementA.png", dpi=300)
    plt.close(fig)
    # (b) placement B curves (late window) and kind-oracle point
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), layout="constrained", sharey=True)
    for ax, w, title in ((axes[0], "late", "(a) rounds 30-150"), (axes[1], "full", "(b) all rounds")):
        ax.plot(OGRID, grid[(w, "B-fixed")] * 100, color=INK, label="fixed tau (best lambda_t, lambda_inf, tau)")
        kb = grid[(w, "B-scope best setting")]
        ax.plot(OGRID, grid[(w, "B-scope best")] * 100, color=BLUE,
                label=f"scope rule ({kb[0]}, lambda_inf {kb[1]:g})")
        ax.plot(OGRID, grid[(w, "B-scope envelope")] * 100, color=BLUE, ls="--", lw=1.0, label="scope rule, best setting per point")
        pts = []
        with open(TAB / "R9_T6_B_kind_oracle_points.csv") as f:
            for r in csv.DictReader(f):
                if r["window"] == w:
                    pts.append((float(r["acc_pct"]), float(r["server_use_pct(non-Main share)"]), r["lambda_t_arm"], r["lambda_inf"]))
        pa = max(pts)
        ax.plot([pa[1] / 100], [pa[0]], marker="*", ms=9, color=ORANGE, ls="none",
                label=f"kind oracle ({pa[2]}, lambda_inf {float(pa[3]):g})")
        for o in OPTS_B:
            ax.axvline(o, color=AXIS, lw=0.6, zorder=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("server use (share of requests sent to the server exit)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("accuracy (%)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(FIG / "R9_figB_placementB.pdf")
    fig.savefig(FIG / "R9_figB_placementB.png", dpi=300)
    plt.close(fig)
    with open(FIG / "R9_figure_captions.md", "w") as f:
        f.write("# Round 9 figure captions\n\n## R9_figA_placementA\n\nAccuracy of the exit rules when the device holds a copy of "
                "the server block (placement A): the best fixed entropy threshold, the scope rule (use the server exit when its "
                "prediction is outside the client's own classes), the scope rule with a threshold, the scope rule with the client "
                "and server exits taken from two mixing ratios, and the request-kind oracle with one or two mixing ratios. Bars are "
                "means over seeds 5 to 7 at the setting with the best seed mean; dots are the seeds. Commute mobility, Round 8 "
                "records.\n\n## R9_figB_placementB\n\nAccuracy against the share of requests sent to the server exit when the server "
                "block stays on the edge (placement B), for evaluation rounds 30 to 150 (a) and all rounds (b). The fixed-threshold "
                "curve is the best over training ratio, inference ratio and threshold at each point. The scope rule sends a request "
                "when the client exit predicts a class outside the client's own classes or its entropy exceeds the threshold. The "
                "star is the request-kind oracle that sends exactly the non-Main requests. Means over seeds 5 to 7; vertical lines "
                "mark 0.3, 0.5 and 0.7.\n")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
