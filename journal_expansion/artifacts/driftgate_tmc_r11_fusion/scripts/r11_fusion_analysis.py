"""Round 11: two-exit fusion with device-level and request-level adjustment, offline from the Round 10 records.

Inputs: runs/phaseT10_prior/r10_T40_s{5,6,7} (JSON, trace, *_evalprobs.npz). M_k, pi_tr (hence a, b), cells and the
arrival order are obtained as in the Round 10 analysis. Outputs: tables/*.csv, decision.json, figures/*.

Rules (directive 4):
  B0 entropy routing tau 0.8, B1 client exit, B2 server exit, B3 argmax (p_c + p_s)/2 (as in Round 10)
  F(w, r): argmax [w p_c + (1-w) p'_s], p'_s = server probabilities corrected with strength r (r = none: p_s);
           w in {0, 0.1, ..., 1}, r in {none, 0.05, 0.1, 0.2, 0.3, 0.5}
  G        one F(w, r) for every device, chosen by seed cross-selection
  D-oracle F(w, r) per state (true at_home), chosen on all three seeds (upper bound of device-level adjustment)
  D-SR / D-TV  state = signal >= theta (signal from the window of earlier requests), theta and the per-state F(w, r)
           chosen by seed cross-selection; theta candidates = 30/40/50/60/70th percentiles of the signal over all requests
           of the two selection seeds
  R-LR     logistic regression (standardized 9 features, L2, default settings) for P(Main), trained on the late-window
           requests of the two other seeds; R-fusion: argmax [P p_c + (1-P) p_s]
  kind oracle  Main -> client exit, otherwise server exit
Seed cross-selection: hold out one seed, choose the setting with the best mean late-window accuracy over the two
others, evaluate the held-out seed; report the three held-out values and their mean.
Accuracy: per evaluation round the mean over clients, then the mean over the rounds of the window; computed as a
weighted sum over requests with weight 1 / (requests of the client-round x clients x rounds in the window).
Device signal for request j of a client-round (arrival order): window = earlier requests of the same round when there
are at least 8, otherwise all requests of the client's previous evaluation round; in the first evaluation round with
fewer than 8 earlier requests the request is in the home state (signal -inf; feature value 0).
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
RUNS = Path(os.environ.get("R11_RUNS", JR / "runs" / "phaseT10_prior"))       # overrides only for smoke tests
OUT = Path(os.environ.get("R11_OUT", HERE))
LATE_MIN = int(os.environ.get("R11_LATE_MIN", 30))
SMOKE = os.environ.get("R11_SMOKE") == "1"
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402
from src.r6_requests import build_partition  # noqa: E402

SEEDS = [5, 6, 7]
SHA = {5: "8f2171d8fea95a108f0b7e8c5bd37572a323f634085eb68b1cf752aeb534e0d5",
       6: "4795a512ef1324bca26d87d6d168e29db31b6081d97607808c7e6c2381e339e8",
       7: "7992cb32a976f81f4fb8caf8b31219be747b0e58d51c5eada22e85aec9861542"}
R10_LATE = {"B0": 63.69, "B1": 61.52, "B2": 63.75, "B3": 65.86}
LAMBDA_BIG, A_CLIP = 0.5, (0.01, 0.99)
TAU = np.float32(0.8)
W_GRID = [round(0.1 * i, 1) for i in range(11)]
R_GRID = [None, 0.05, 0.1, 0.2, 0.3, 0.5]
CFGS = [(w, r) for r in R_GRID for w in W_GRID]
PCTS = [30, 40, 50, 60, 70]
MIN_PREV = 8
FEATURES = ["client max prob", "client entropy", "server max prob", "server entropy", "p_s(M_k)", "p_c(M_k)",
            "TV(p_c, p_s)", "exits agree", "device x_SR"]
KINDS = ["Main", "OOP", "OOR"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def cfg_txt(c):
    return f"w={c[0]:.1f}, r={'none' if c[1] is None else c[1]}"


_TRAIN = None


def rebuild_mains(h):
    global _TRAIN
    if _TRAIN is None:
        from data.partition import get_cifar10
        _TRAIN = np.array(get_cifar10(str(ROOT / "data_cache"))[0].targets)
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    env, _ = r6_env.load_env(h["config"]["env_file"])
    _, mains, _, _ = build_partition(_TRAIN, env, int(h["config"]["partition_seed"]), cfg["num_classes"], cfg)
    return {int(k): sorted(int(c) for c in v) for k, v in mains.items()}


# ------------------------------------------------------------------------------------------------ per seed
def load_seed(seed):
    name = f"r10_T40_s{seed}"
    h = json.load(open(RUNS / f"{name}.json"))
    z = np.load(RUNS / f"{name}_trace.npz")
    src = RUNS / f"{name}_evalprobs.npz"
    digest = sha256(src)
    if not SMOKE:
        assert digest == SHA[seed], f"sha256 of {src} differs: {digest}"
    q = dict(np.load(src))
    mains = rebuild_mains(h)
    K = 50
    er = np.array(q["eval_rounds"])
    E = len(er)
    M = np.zeros((K, 10), bool)
    for k, v in mains.items():
        M[k, v] = True
    e_i, k_i = q["req_eval_index"].astype(np.int64), q["req_client"].astype(np.int64)
    N = len(e_i)
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"].astype(np.int64)
    Mrow = M[k_i]
    # pi_tr and a per (round, client), as in the Round 10 analysis
    hc, ha, cells = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float), z["cells"]
    a_ek = np.zeros((E, K))
    for e, r in enumerate(er):
        t = int(r) - 1
        hall = ha[t] / ha[t].sum()
        for k in range(K):
            pis = [LAMBDA_BIG * (hc[t, int(zc)] / hc[t, int(zc)].sum() if hc[t, int(zc)].sum() > 0 else hall)
                   + (1 - LAMBDA_BIG) * hall for zc in cells[t, k] if zc >= 0]
            a_ek[e, k] = np.clip(np.mean(pis, axis=0)[M[k]].sum(), *A_CLIP)
    pc = q["pc"].astype(np.float64)
    ps = q["ps"].astype(np.float64)
    cp, sp = q["cp"].astype(np.int64), q["sp"].astype(np.int64)
    G_ = E * K
    g = e_i * K + k_i
    n_ek = np.bincount(g, minlength=G_).reshape(E, K).astype(float)
    late = er >= LATE_MIN
    wt = {"full": 1.0 / (n_ek[e_i, k_i] * K * E), "late": np.where(late[e_i], 1.0 / (n_ek[e_i, k_i] * K * late.sum()), 0.0)}
    home_r = q["req_home"].astype(bool)
    # ---- device signals over the window of earlier requests (arrival order)
    srout = (~M[k_i, sp]).astype(np.float64)
    tv = 0.5 * np.abs(pc - ps).sum(1)
    x_sr = np.full(N, -np.inf)
    x_tv = np.full(N, -np.inf)
    order = np.lexsort((q["req_arrival"], k_i, e_i))
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    groups = {(int(e_i[gg[0]]), int(k_i[gg[0]])): gg for gg in np.split(order, bounds)}
    for k in range(K):
        prev = None
        for e in range(E):
            gg = groups[(e, k)]
            for vals, out in ((srout, x_sr), (tv, x_tv)):
                cs = np.cumsum(vals[gg])
                j = np.arange(len(gg))
                sig = np.full(len(gg), -np.inf)
                okj = j >= MIN_PREV
                sig[okj] = cs[j[okj] - 1] / j[okj]
                if prev is not None:
                    sig[~okj] = vals[prev].mean()
                out[gg] = sig
            prev = gg
    # ---- features for the request-level model
    ent_s = -(ps * np.log(np.clip(ps, 1e-12, 1))).sum(1)
    feats = np.stack([pc.max(1), q["ent"].astype(np.float64), ps.max(1), ent_s, (ps * Mrow).sum(1), (pc * Mrow).sum(1),
                      tv, (cp == sp).astype(np.float64), np.where(np.isfinite(x_sr), x_sr, 0.0)], axis=1)
    D = dict(seed=seed, h=h, er=er, E=E, K=K, N=N, e=e_i, k=k_i, y=y, kind=kind, home=home_r, wt=wt, n_ek=n_ek,
             x_sr=x_sr, x_tv=x_tv, feats=feats, sha=digest, bytes=src.stat().st_size, run_id=h["config"]["run_id"],
             mains_equal_provenance=None)
    prov = JR / "provenance" / f"{h['config']['run_id']}.json"
    if prov.exists():
        D["mains_equal_provenance"] = json.load(open(prov)).get("client_main") == {str(k): v for k, v in mains.items()}
    # ---- answers of the fixed rules and of every F(w, r) (bool correct vectors)
    D["ans"] = {"B0": np.where(q["ent"] > TAU, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1),
                "kind oracle": np.where(kind == 0, cp, sp)}
    yt = torch.as_tensor(y, device=DEV)
    pct, pst = torch.as_tensor(pc, device=DEV), torch.as_tensor(ps, device=DEV)
    Mt = torch.as_tensor(Mrow, device=DEV)
    at = torch.as_tensor(a_ek[e_i, k_i], device=DEV)[:, None]
    C = torch.zeros((len(CFGS), N), dtype=torch.bool, device=DEV)
    for r in R_GRID:
        if r is None:
            psr = pst
        else:
            wgt = torch.where(Mt, (1 - r) / at, r / (1 - at))
            psr = pst * wgt
            psr = psr / psr.sum(1, keepdim=True)
        for w in W_GRID:
            C[CFGS.index((w, r))] = (w * pct + (1 - w) * psr).argmax(1) == yt
    D["C"] = C
    D["pc"], D["ps"] = pc, ps
    D["b3_equals_F05"] = float((C[CFGS.index((0.5, None))].cpu().numpy() == (D["ans"]["B3"] == y)).mean())
    print(f"  seed {seed}: {N} requests, sha256 ok, F(0.5, none) = B3 on {D['b3_equals_F05']:.6f} of requests", flush=True)
    return D


# ------------------------------------------------------------------------------------------------ helpers
def acc_vec(D, correct, w):
    return float(np.dot(np.asarray(correct, float), D["wt"][w]))


def cfg_scores(D, mask, w):
    """weighted accuracy contribution of every F config over the requests in mask (bool numpy) -> [n_cfg]."""
    wv = torch.as_tensor(D["wt"][w] * mask, device=DEV)
    return np.array([float((D["C"][i] * wv).sum()) for i in range(D["C"].shape[0])])


def split_metrics(D, correct, w="late"):
    E, K = D["E"], D["K"]
    m = D["er"] >= LATE_MIN if w == "late" else np.ones(E, bool)
    g = D["e"] * K + D["k"]
    c = np.bincount(g, weights=correct, minlength=E * K).reshape(E, K) / D["n_ek"]
    home = np.zeros(E * K, bool)
    home[g] = D["home"]
    home = home.reshape(E, K)
    out = dict(acc=c[m].mean(), home=np.mean([c[e][home[e]].mean() for e in np.flatnonzero(m) if home[e].any()]),
               away=np.mean([c[e][~home[e]].mean() for e in np.flatnonzero(m) if (~home[e]).any()]))
    for i, kn in enumerate(KINDS):
        sel = D["kind"] == i
        ck = np.bincount(g, weights=correct & sel, minlength=E * K).reshape(E, K)
        nk = np.bincount(g, weights=sel, minlength=E * K).reshape(E, K)
        with np.errstate(invalid="ignore", divide="ignore"):
            ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
        out[kn] = np.mean([np.nanmean(ak[e]) for e in np.flatnonzero(m) if np.isfinite(ak[e]).any()])
    return out


def per_round(D, correct):
    E, K = D["E"], D["K"]
    g = D["e"] * K + D["k"]
    return (np.bincount(g, weights=correct, minlength=E * K).reshape(E, K) / D["n_ek"]).mean(1)


# ------------------------------------------------------------------------------------------------ main
def main():
    R = {s: load_seed(s) for s in SEEDS}
    y = {s: R[s]["y"] for s in SEEDS}
    # ---------- precheck: B0-B3 late values against Round 10
    rows, ok = [], True
    for rl in ("B0", "B1", "B2", "B3"):
        v = np.mean([acc_vec(R[s], R[s]["ans"][rl] == y[s], "late") for s in SEEDS]) * 100
        d = v - R10_LATE[rl]
        ok &= abs(d) <= 0.05
        rows.append([rl, f"{v:.4f}", f"{R10_LATE[rl]:.2f}", f"{d:+.4f}", "yes" if abs(d) <= 0.05 else "NO"])
    rows += [[f"seed {s}", R[s]["run_id"], R[s]["sha"], R[s]["bytes"], f"Mk=provenance {R[s]['mains_equal_provenance']}; "
              f"F(0.5,none)=B3 on {R[s]['b3_equals_F05']:.6f}"] for s in SEEDS]
    wcsv("R11_T0_precheck.csv", ["rule_or_seed", "late_pct_or_run_id", "round10_pct_or_sha256", "difference_pp_or_bytes",
                                 "within_0.05_or_notes"], rows)
    if not ok and not SMOKE:
        print("precheck failed: B0-B3 differ from Round 10 by more than 0.05 pp; stopping")
        sys.exit(2)
    res = {}           # (rule, seed) -> correct bool vector (held-out evaluation)
    chosen = {}        # (rule, seed) -> setting text
    for s in SEEDS:
        for rl in ("B0", "B1", "B2", "B3", "kind oracle"):
            res[(rl, s)] = R[s]["ans"][rl] == y[s]
    # ---------- G: one F(w, r), seed cross-selection
    allmask = {s: np.ones(R[s]["N"], bool) for s in SEEDS}
    sc_all = {s: cfg_scores(R[s], allmask[s], "late") for s in SEEDS}
    for s in SEEDS:
        tr = [t for t in SEEDS if t != s]
        ci = int(np.argmax(np.mean([sc_all[t] for t in tr], axis=0)))
        res[("G", s)] = R[s]["C"][ci].cpu().numpy()
        chosen[("G", s)] = cfg_txt(CFGS[ci])
    # ---------- D-oracle: per-state F on all three seeds
    hm = {s: R[s]["home"] for s in SEEDS}
    sh = np.mean([cfg_scores(R[s], hm[s], "late") for s in SEEDS], axis=0)
    sa = np.mean([cfg_scores(R[s], ~hm[s], "late") for s in SEEDS], axis=0)
    ch, ca = int(np.argmax(sh)), int(np.argmax(sa))
    for s in SEEDS:
        C = R[s]["C"]
        res[("D-oracle", s)] = np.where(hm[s], C[ch].cpu().numpy(), C[ca].cpu().numpy())
        chosen[("D-oracle", s)] = f"home {cfg_txt(CFGS[ch])}; away {cfg_txt(CFGS[ca])} (all three seeds)"
    # ---------- D-SR / D-TV: theta and per-state F, seed cross-selection
    state_acc = {}
    for rl, key in (("D-SR", "x_sr"), ("D-TV", "x_tv")):
        for s in SEEDS:
            tr = [t for t in SEEDS if t != s]
            pool = np.concatenate([R[t][key][np.isfinite(R[t][key])] for t in tr])
            thetas = [float(np.percentile(pool, p)) for p in PCTS]
            best = None
            for pi, th in zip(PCTS, thetas):
                awy = {t: R[t][key] >= th for t in tr}
                shh = np.mean([cfg_scores(R[t], ~awy[t], "late") for t in tr], axis=0)
                saa = np.mean([cfg_scores(R[t], awy[t], "late") for t in tr], axis=0)
                tot = shh.max() + saa.max()
                if best is None or tot > best[0]:
                    best = (tot, pi, th, int(np.argmax(shh)), int(np.argmax(saa)))
            _, pi, th, ih, ia = best
            awy_s = R[s][key] >= th
            C = R[s]["C"]
            res[(rl, s)] = np.where(awy_s, C[ia].cpu().numpy(), C[ih].cpu().numpy())
            chosen[(rl, s)] = f"theta = {pi}th percentile ({th:.4f}); home {cfg_txt(CFGS[ih])}; away {cfg_txt(CFGS[ia])}"
            inw = R[s]["wt"]["late"] > 0
            truly_away = ~R[s]["home"]
            state_acc[(rl, s)] = (float((awy_s & truly_away & inw).sum() / max((truly_away & inw).sum(), 1)),
                                  float((~awy_s & ~truly_away & inw).sum() / max((~truly_away & inw).sum(), 1)),
                                  float(awy_s[inw].mean()))
    # ---------- request-level: AUROC per feature, R-LR, R-fusion
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    auc_rows = []
    P = {}
    lr_info = {}
    for s in SEEDS:
        tr = [t for t in SEEDS if t != s]
        Xtr = np.concatenate([R[t]["feats"][R[t]["wt"]["late"] > 0] for t in tr])
        ytr = np.concatenate([(R[t]["kind"] == 0)[R[t]["wt"]["late"] > 0] for t in tr]).astype(int)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression().fit(scaler.transform(Xtr), ytr)
        P[s] = model.predict_proba(scaler.transform(R[s]["feats"]))[:, 1]
        lr_info[s] = dict(coef=[float(c) for c in model.coef_[0]], intercept=float(model.intercept_[0]), n_iter=int(model.n_iter_[0]))
        pc, ps = R[s]["pc"], R[s]["ps"]
        res[("R-fusion", s)] = (P[s][:, None] * pc + (1 - P[s][:, None]) * ps).argmax(1) == y[s]
        chosen[("R-fusion", s)] = f"no setting; logistic regression trained on seeds {tr} (n_iter {lr_info[s]['n_iter']})"
    for group in ("all devices", "away devices"):
        for fi, fname in enumerate(FEATURES + ["R-LR (1 - P(Main))"]):
            vals, flipped = [], False
            for s in SEEDS:
                D = R[s]
                m = (D["wt"]["late"] > 0) & (np.ones(D["N"], bool) if group == "all devices" else ~D["home"])
                pos = (D["kind"] != 0)[m]
                score = (1 - P[s])[m] if fi == len(FEATURES) else D["feats"][m, fi]
                if fname == "device x_SR":
                    score = np.where(np.isfinite(D["x_sr"][m]), D["x_sr"][m], 0.0)
                v = roc_auc_score(pos, score) if pos.any() and (~pos).any() else np.nan
                vals.append(v)
            mean_v = float(np.mean(vals))
            if mean_v < 0.5:
                vals, mean_v, flipped = [1 - v for v in vals], 1 - mean_v, True
            auc_rows.append([group, fname, "yes" if flipped else "no"] + [f"{v:.4f}" for v in vals] + [f"{mean_v:.4f}"])
    wcsv("R11_T3_auroc.csv", ["devices", "feature", "direction_flipped", "auroc_s5", "auroc_s6", "auroc_s7", "auroc_mean"], auc_rows)
    # ---------- table 1 and table 2
    RULES = ["B0", "B1", "B2", "B3", "G", "D-oracle", "D-SR", "D-TV", "R-fusion", "kind oracle"]
    MET = {(rl, s): split_metrics(R[s], res[(rl, s)]) for rl in RULES for s in SEEDS}
    FULL = {(rl, s): acc_vec(R[s], res[(rl, s)], "full") for rl in RULES for s in SEEDS}
    mean = lambda rl, key: float(np.mean([MET[(rl, s)][key] for s in SEEDS]))
    mfull = lambda rl: float(np.mean([FULL[(rl, s)] for s in SEEDS]))
    rows = [[rl, f"{mean(rl, 'acc') * 100:.2f}", f"{mfull(rl) * 100:.2f}"]
            + [f"{mean(rl, k_) * 100:.2f}" for k_ in ("home", "away", "Main", "OOP", "OOR")] for rl in RULES]
    wcsv("R11_T1_rules.csv", ["rule", "acc_late_pct", "acc_full_pct", "home_late_pct", "away_late_pct", "Main_late_pct",
                              "OOP_late_pct", "OOR_late_pct"], rows)
    rows = []
    for rl in ("G", "D-oracle", "D-SR", "D-TV", "R-fusion"):
        for s in SEEDS:
            rows.append([rl, s, f"{MET[(rl, s)]['acc'] * 100:.2f}", f"{FULL[(rl, s)] * 100:.2f}", chosen[(rl, s)]])
        rows.append([rl, "mean", f"{mean(rl, 'acc') * 100:.2f}", f"{mfull(rl) * 100:.2f}", ""])
    wcsv("R11_T2_held_out.csv", ["rule", "held_out_seed", "acc_late_pct", "acc_full_pct", "chosen_setting"], rows)
    rows = []
    for rl in ("D-SR", "D-TV"):
        for s in SEEDS:
            a_, h_, f_ = state_acc[(rl, s)]
            rows.append([rl, s, f"{a_ * 100:.2f}", f"{h_ * 100:.2f}", f"{f_ * 100:.2f}"])
        rows.append([rl, "mean"] + [f"{np.mean([state_acc[(rl, s)][i] for s in SEEDS]) * 100:.2f}" for i in range(3)])
    wcsv("R11_T4_state_accuracy.csv", ["rule", "held_out_seed", "away_judged_away_pct", "home_judged_home_pct",
                                       "requests_in_away_state_pct"], rows)
    # ---------- decision
    b3, b3f, g_ = mean("B3", "acc") * 100, mfull("B3") * 100, mean("G", "acc") * 100
    dev = {rl: dict(late=mean(rl, "acc") * 100, full=mfull(rl) * 100,
                    passes=bool(mean(rl, "acc") * 100 >= b3 + 1.5 and mean(rl, "acc") * 100 >= g_ + 0.5 and mfull(rl) * 100 > b3f))
           for rl in ("D-SR", "D-TV")}
    rq = dict(late=mean("R-fusion", "acc") * 100, full=mfull("R-fusion") * 100)
    rq["passes"] = bool(rq["late"] >= b3 + 2.0 and rq["full"] > b3f)
    d_or = mean("D-oracle", "acc") * 100 - b3
    verdict = []
    if any(v["passes"] for v in dev.values()):
        verdict.append("device-level: proceed (" + ", ".join(k for k, v in dev.items() if v["passes"]) + ")")
    if rq["passes"]:
        verdict.append("request-level: proceed")
    if not verdict:
        verdict = ["close"]
    dec = dict(window="late (evaluation rounds 30-150), mean of the three held-out seeds", verdict="; ".join(verdict),
               B3_late=b3, B3_full=b3f, G_late=g_, device_level=dev, request_level=rq, D_oracle_minus_B3_pp=d_or,
               device_level_small_from_upper_bound=bool(d_or < 1.5),
               chosen={f"{rl} held-out {s}": chosen[(rl, s)] for (rl, s) in chosen},
               logistic_regression={str(s): lr_info[s] for s in SEEDS},
               means_late_pct={rl: mean(rl, "acc") * 100 for rl in RULES}, means_full_pct={rl: mfull(rl) * 100 for rl in RULES})
    json.dump(dec, open(OUT / "decision.json", "w"), indent=1)
    print(json.dumps({k: dec[k] for k in ("verdict", "B3_late", "G_late", "device_level", "request_level", "D_oracle_minus_B3_pp")}, indent=1))
    figure(R, res)


def figure(R, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    er = R[SEEDS[0]]["er"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
    for rl, lab, col, ls in (("B0", "entropy routing (tau 0.8)", INK, "--"), ("B3", "mean of both exits", INK2, ":"),
                             ("G", "one tuned fusion for all devices", MUTED, "-."), ("D-SR", "fusion per device state (x_SR)", BLUE, "-"),
                             ("R-fusion", "fusion weighted per request", GREEN, "-"), ("kind oracle", "request-kind oracle", PURPLE, "-")):
        yv = np.mean([per_round(R[s], res[(rl, s)]) for s in SEEDS], axis=0) * 100
        ax.plot(er, yv, color=col, ls=ls, marker="o", ms=2, label=lab)
    ax.axvline(LATE_MIN - 0.5, color=AXIS, lw=0.8, zorder=0)
    ticks = [5 * 60, 10 * 60, 15 * 60, 20 * 60]
    ax.set_xticks([(t - 300) / 6 + 1 for t in ticks])
    ax.set_xticklabels([r6_env.hhmm(t) for t in ticks])
    ax.set_xlim(1, 151)
    ax.set_xlabel("time of day")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.savefig(FIG / "R11_fig1_accuracy_by_time.pdf")
    fig.savefig(FIG / "R11_fig1_accuracy_by_time.png", dpi=300)
    plt.close(fig)
    with open(FIG / "R11_figure_captions.md", "w") as f:
        f.write("# Round 11 figure caption\n\n## R11_fig1_accuracy_by_time\n\nAccuracy by time of day for entropy routing with "
                "threshold 0.8, the mean of the client-exit and server-exit probabilities, one tuned fusion for every device, a fusion "
                "chosen per device state from the server-exit non-Main rate, a fusion weighted per request by a logistic model of "
                "the request being in the client's own classes, and the request-kind oracle. Tuned rules are evaluated on the held-out "
                "seed. Commute mobility, training ratio 0.4, means over seeds 5 to 7. The vertical line marks the start of the late "
                "window used for the decision.\n")
    print("  figure written", flush=True)


if __name__ == "__main__":
    main()
