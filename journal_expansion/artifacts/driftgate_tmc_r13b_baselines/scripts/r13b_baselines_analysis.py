"""Round 13b: related-work baselines, complements of the fusion F, and new structures and conditions.

Part A (no new run) and part B (new runs in runs/phaseT13b_arch) are computed by this one script, fixed before any
Round 13b result. Runs of part B that have not finished are skipped (their groups are then absent).

Inputs
  development: runs/phaseT10_prior/r10_T40_s{5,6,7} (Round 10 records; S1, fixed lambda 0.4, seeds 5-7). Only these
      records choose a value (the F-gate threshold t) or fit a model (R-LR).
  test: runs/phaseT12_fusion/r12_* (Round 12, 50 runs) and runs/phaseT13b_arch/r13b_* (part B); each is used once.
  Per run, M_k (classes of the Main requests), pi_tr = 0.5 h_cell + 0.5 h_all of the training round of the evaluation
  round (two cells: mean), a = clip(pi_tr(M_k), 0.01, 0.99), b = 1 - a, the arrival order and home/away (Round 6 path
  only) are obtained as in scripts/r12_fusion_eval_analysis.py (Round 6 client ids rebuilt from the stored order).

Rules (directive 1.2, 2, 3). p_c, p_s: stored softmax probabilities (float16 -> float32); cp, sp: the exits' answers;
H_c: the client-exit entropy of the record (the value B0 compares with 0.8); H_s: entropy of the stored p_s.
p'_s(r): p'_s(c) ~ p_s(c) (1 - r)/a on M_k and p_s(c) r/b on O_k, normalised.
  B0 entropy routing tau 0.8 | B1 client exit | B2 server exit | B3 argmax (p_c + p_s)/2
  F argmax [0.2 p_c + 0.8 p'_s(0.5)]
  R-PoE argmax [log p_c + log p_s]; R-PoE-bal argmax [log p_c + log p'_s(0.5)] (probabilities below 1e-8 set to 1e-8)
  R-THE the answer of the exit with the smaller entropy (client exit when H_c <= H_s)
  R-ZTW curve: client exit when max p_c >= q, otherwise the R-PoE answer; q = 0, 0.01, ..., 1, and the end point where
      every request goes to the server (= R-PoE). Table 1 value: the curve at the server use of B0 (same window).
  R-EM (Round 10 M1): r_hat for request j of a client and evaluation round = EM over the earlier requests of that
      client-round in arrival order, started at r_prev; fewer than 8 earlier requests -> r_prev; r_prev = EM over all
      requests of the client's previous evaluation round with requests (0.5 before the first); EM as in Round 10
      (q_i = (r/b) p_s(O_k) / [((1-r)/a) p_s(M_k) + (r/b) p_s(O_k)], r <- mean q_i, stop when the change < 1e-4 or after
      50 iterations, clip [0.02, 0.98]); answer argmax p'_s(r_hat).
  R-LR (Round 11 R-fusion): P(Main) from a logistic regression (scikit-learn defaults) on 9 standardized features
      (client max prob, H_c, server max prob, H_s, p_s(M_k), p_c(M_k), TV(p_c, p_s), exits agree, x_SR), fitted once on
      all late-window requests of the three development runs; answer argmax [P p_c + (1 - P) p_s]. x_SR = share of
      requests whose server answer is outside M_k over the window of the request (0 when there is no window).
  F-nodebias F(0.2, no correction) | F-eq F(0.5, 0.5)
  F-auto F(w, 0.5) with w = Hbar_s / (Hbar_c + Hbar_s), the mean entropies over the window of the request; w = 0.5
      when there is no window (or both means are 0)
  F-gate the server answer sp when p_s(M_k) < t, otherwise F; t from {0.02, 0.05, 0.1, 0.15, 0.2, 0.3} with the highest
      late-window seed-mean accuracy on the development runs (ties -> the smaller t)
  kind oracle (reference): Main -> client exit, otherwise server exit
Window of a request (Rounds 10 and 11): the earlier requests of the same client and evaluation round in arrival order
when there are at least 8; otherwise all requests of the client's previous evaluation round. When a client has no
requests in some evaluation rounds (participation 0.5, client mobility, structures with absent clients), "previous" is
the client's last evaluation round with requests.
Server use (share of requests that need the server exit's output): B1 0, B0 H_c > 0.8, kind oracle the non-Main
requests, R-ZTW the server use of B0, every other rule 1.
Curves (directive 2): E (B0 with tau), SF (client exit when H_c <= tau, otherwise F), SB3 (client exit when H_c <= tau,
otherwise B3), tau = 0, 0.05, ..., 2.30 and infinity; R-ZTW over q. Accuracy at server use 0.1, 0.2, ..., 1.0 by linear
interpolation between neighbouring grid points. SF ratios per run (directive 5; late window): the first server use at
which the piecewise-linear curve SF reaches (1) the accuracy of B0, (2) the highest accuracy of curve E over its grid.
Accuracy: per evaluation round the mean over the clients with requests, then the mean over the evaluation rounds of
the window; full = all evaluation rounds, late = evaluation rounds from training round 30 on. Splits (late): home and
away (Round 6 path), Main, OOP, OOR.
Start check (directive 1.3): B0, B3 and F recomputed from the Round 12 records equal Round 12 table 1 (late window,
seed means) within 0.05 pp; otherwise the script writes the comparison and stops.
Decisions (directive 6; test runs, late window, seed means):
  1 baselines: rules among R-PoE, R-PoE-bal, R-THE, R-EM, R-LR whose accuracy >= F in S1 or S2 (fixed 0.4); the same
    check for every part-B group, listed separately.
  2 F-auto adopted when (i) in every group with F < max(B1, B3) it is > max(B1, B3) and (ii) in every other group it is
    >= F - 0.3 pp. Groups: every group of table 1 (the 14 Round 12 groups including the DriftGate ones, as the directive
    names both ResNet-18 groups, and the part-B groups). The groups with F < max(B1, B3) follow from that definition;
    the directive's parenthesis names only the two Round 12 ResNet-18 groups, and the result for the parenthesis' set
    is written to decision.json for information only.
  3 F-gate adopted when in S1 and in S2 (fixed 0.4) its accuracy of clients away is >= F + 1.0 pp and its accuracy
    (late) >= F.
  4 structures 1-4: F - B0 and F - B3 (no decision). 5 conditions: F - B3 next to the default S1 (Round 12, seeds 0-4).
Outputs: tables/*.csv, decision.json, figures/*. Cached per-run summaries in cache/ (keyed by the record file and this
script); R13B_OUT, R13B_RUNS13, R13B_LATE_FROM and R13B_SMOKE (run names) are overrides for smoke tests only.
"""
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
RUNS12 = JR / "runs" / "phaseT12_fusion"
RUNS13 = Path(os.environ.get("R13B_RUNS13", JR / "runs" / "phaseT13b_arch"))   # override only for smoke tests
DEVRUNS = JR / "runs" / "phaseT10_prior"
OUT = Path(os.environ.get("R13B_OUT", HERE))
TAB, FIG, CACHE = OUT / "tables", OUT / "figures", OUT / "cache"
for d in (TAB, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(JR))

LATE_FROM = int(os.environ.get("R13B_LATE_FROM", 30))    # override only for smoke tests
SMOKE = [s for s in os.environ.get("R13B_SMOKE", "").split(",") if s]
WORKERS = int(os.environ.get("R13B_WORKERS", 4))
DEV_SEEDS = [5, 6, 7]
DEV_SHA = {5: "8f2171d8fea95a108f0b7e8c5bd37572a323f634085eb68b1cf752aeb534e0d5",
           6: "4795a512ef1324bca26d87d6d168e29db31b6081d97607808c7e6c2381e339e8",
           7: "7992cb32a976f81f4fb8caf8b31219be747b0e58d51c5eada22e85aec9861542"}
W, RSTR, TAU0 = 0.2, 0.5, np.float32(0.8)
LAMBDA_BIG, A_CLIP = 0.5, (0.01, 0.99)
TAU = np.round(np.arange(0, 2.30 + 1e-9, 0.05), 2).astype(np.float32)
QG = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
OPTS = [round(0.1 * i, 1) for i in range(1, 11)]
T_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
MIN_PREV, EM_MAX, EM_TOL, R_CLIP, P_FLOOR = 8, 50, 1e-4, (0.02, 0.98), 1e-8
RULES = ["B0", "B1", "B2", "B3", "F", "R-PoE", "R-PoE-bal", "R-THE", "R-ZTW", "R-EM", "R-LR", "F-nodebias", "F-eq",
         "F-auto", "F-gate", "kind oracle"]
BASELINES = ["R-PoE", "R-PoE-bal", "R-THE", "R-EM", "R-LR"]
SPLITS = ["home", "away", "Main", "OOP", "OOR"]
KINDS = ["Main", "OOP", "OOR"]
CURVES = ["E", "SF", "SB3", "R-ZTW"]
FEATURES = ["client max prob", "client entropy", "server max prob", "server entropy", "p_s(M_k)", "p_c(M_k)",
            "TV(p_c, p_s)", "exits agree", "device x_SR"]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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
    """R non-increasing along the points; value at each o (largest bracketing pair), nan outside (Round 12)."""
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


def first_reach(srv, acc, target):
    """first server use at which the piecewise-linear curve (points sorted by server use) reaches target; nan if never."""
    o = np.argsort(srv, kind="stable")
    s, a = srv[o], acc[o]
    if a[0] >= target:
        return float(s[0])
    for j in range(1, len(s)):
        if a[j] >= target:
            if a[j] == a[j - 1]:
                return float(s[j])
            return float(s[j - 1] + (target - a[j - 1]) / (a[j] - a[j - 1]) * (s[j] - s[j - 1]))
    return float("nan")


# ------------------------------------------------------------------------------------------------ loading
def load(runs, name):
    """request arrays of one run (Round 12 loading: M_k, a, cells, client ids)."""
    h = json.load(open(runs / f"{name}.json"))
    src = runs / f"{name}_evalprobs.npz"
    q = np.load(src)
    r6 = (runs / f"{name}_trace.npz").exists()
    if r6:
        z = np.load(runs / f"{name}_trace.npz")
        hc, ha = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float)
        er = np.array(q["eval_rounds"])
        cells_e = z["cells"][er - 1]
    else:
        z = np.load(runs / f"{name}_rec.npz")
        hc, ha = z["train_hist_cell"].astype(float), z["train_hist_all"].astype(float)
        er = np.array(z["eval_rounds"])
        assert np.array_equal(er, q["eval_rounds"])
        cells_e = z["eval_cells"]
    e_i = q["req_eval_index"].astype(np.int64)
    k_i = q["req_client"].astype(np.int64)
    ids_rebuilt = None
    if r6:   # Round 12 fix: uint8 client ids of older Round 6 records wrap at 256; rebuild from the stored order
        nt = z["eval_n_total"].astype(np.int64)
        k_reb = np.concatenate([np.repeat(np.arange(nt.shape[1]), nt[e]) for e in range(nt.shape[0])])
        e_reb = np.repeat(np.arange(nt.shape[0]), nt.sum(1))
        assert len(k_reb) == len(k_i) and np.array_equal(k_reb % 256, k_i % 256) and np.array_equal(e_reb, e_i), name
        ids_rebuilt = bool(not np.array_equal(k_reb, k_i))
        k_i = k_reb
    y = q["req_label"].astype(np.int64)
    kind = q["req_kind"].astype(np.int64)
    C = q["pc"].shape[1]
    K = int(k_i.max()) + 1
    E = len(er)
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
    if not r6:
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
    g = e_i * K + k_i
    n_ek = np.bincount(g, minlength=E * K).reshape(E, K).astype(float)
    present = n_ek > 0
    home = np.full(E * K, -1, np.int8)
    home[g] = q["req_home"].astype(np.int8)
    home = home.reshape(E, K)
    D = dict(name=name, h=h, src=src, r6=r6, er=er, E=E, K=K, C=C, e=e_i, k=k_i, g=g, y=y, kind=kind, M=M,
             pc=q["pc"].astype(np.float32), ps=q["ps"].astype(np.float32), cp=q["cp"].astype(np.int64),
             sp=q["sp"].astype(np.int64), ent=q["ent"].astype(np.float32), arrival=q["req_arrival"].astype(np.int64),
             a_req=a_ek[e_i, k_i].astype(np.float32), a_ek=a_ek, n_ek=n_ek, present=present, home=home,
             has_home=bool((home >= 0).any()), late=er >= LATE_FROM,
             checks=dict(mk_consistent=mk_consistent, mk_prov=mk_prov, ids_rebuilt=ids_rebuilt))
    D["Mrow"] = M[k_i]
    return D


def weights(D, m):
    """per-request weight: 1 / (requests of the client-round x clients with requests in the round x rounds in m)."""
    npres = D["present"].sum(1)
    sel = m[D["e"]]
    return np.where(sel, 1.0 / (D["n_ek"][D["e"], D["k"]] * npres[D["e"]] * m.sum()), 0.0)


def groups_in_order(D):
    """request index arrays per (round, client), arrival order, rounds ascending."""
    order = np.lexsort((D["arrival"], D["k"], D["e"]))
    bounds = np.flatnonzero(np.diff(D["g"][order])) + 1
    return np.split(order, bounds)


def window_means(D, vals_list):
    """mean of each value over the window of every request (nan when there is no window)."""
    outs = [np.full(len(D["y"]), np.nan) for _ in vals_list]
    prev = {}
    for gg in groups_in_order(D):
        kk = int(D["k"][gg[0]])
        j = np.arange(len(gg))
        okj = j >= MIN_PREV
        for vals, out in zip(vals_list, outs):
            cs = np.cumsum(vals[gg])
            sig = np.full(len(gg), np.nan)
            sig[okj] = cs[j[okj] - 1] / j[okj]
            if kk in prev:
                sig[~okj] = vals[prev[kk]].mean()
            out[gg] = sig
        prev[kk] = gg
    return outs


def server_entropy(ps):
    out = np.empty(len(ps), np.float64)
    for i in range(0, len(ps), 1 << 20):
        p = ps[i:i + (1 << 20)].astype(np.float64)
        out[i:i + (1 << 20)] = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return out


def features(D):
    """the 9 Round 11 features (float64 [N, 9])."""
    pc, ps, Mrow = D["pc"], D["ps"], D["Mrow"]
    srout = (~D["M"][D["k"], D["sp"]]).astype(np.float64)
    (x_sr,) = window_means(D, [srout])
    tv = 0.5 * np.abs(pc - ps).sum(1, dtype=np.float64)
    return np.stack([pc.max(1).astype(np.float64), D["ent"].astype(np.float64), ps.max(1).astype(np.float64),
                     server_entropy(ps), (ps * Mrow).sum(1, dtype=np.float64), (pc * Mrow).sum(1, dtype=np.float64),
                     tv, (D["cp"] == D["sp"]).astype(np.float64), np.where(np.isfinite(x_sr), x_sr, 0.0)], axis=1)


# ------------------------------------------------------------------------------------------------ EM (Round 10)
def em_full(Mv, Ov, a, b, r0):
    r = float(r0)
    for _ in range(EM_MAX):
        num = (r / b) * Ov
        rn = float(np.mean(num / (((1 - r) / a) * Mv + num)))
        done = abs(rn - r) < EM_TOL
        r = rn
        if done:
            break
    return float(np.clip(r, *R_CLIP))


def em_prefix(Mv, Ov, a, b, r0, dev):
    import torch
    n = len(Mv)
    out = np.full(n, float(r0))
    if n <= MIN_PREV:
        return out
    Mt = torch.as_tensor(Mv, dtype=torch.float64, device=dev)
    Ot = torch.as_tensor(Ov, dtype=torch.float64, device=dev)
    rows = torch.arange(MIN_PREV, n, device=dev)
    mask = torch.arange(n, device=dev)[None, :] < rows[:, None]
    cnt = rows.to(torch.float64)
    r = torch.full((len(rows),), float(r0), dtype=torch.float64, device=dev)
    active = torch.ones(len(rows), dtype=torch.bool, device=dev)
    zero = torch.zeros((), dtype=torch.float64, device=dev)
    for _ in range(EM_MAX):
        num = (r / b)[:, None] * Ot[None, :]
        qq = torch.where(mask, num / (((1 - r) / a)[:, None] * Mt[None, :] + num), zero)
        rn = qq.sum(1) / cnt
        delta = (rn - r).abs()
        r = torch.where(active, rn, r)
        active = active & (delta >= EM_TOL)
        if not bool(active.any()):
            break
    out[MIN_PREV:] = np.clip(r.cpu().numpy(), *R_CLIP)
    return out


def r_causal(D, dev):
    Mv = (D["ps"] * D["Mrow"]).sum(1, dtype=np.float64)
    Ov = (D["ps"] * ~D["Mrow"]).sum(1, dtype=np.float64)
    out = np.zeros(len(Mv))
    r_prev = {}
    for gg in groups_in_order(D):
        e, k = int(D["e"][gg[0]]), int(D["k"][gg[0]])
        a = float(D["a_ek"][e, k])
        b = 1 - a
        r0 = r_prev.get(k, 0.5)
        out[gg] = em_prefix(Mv[gg], Ov[gg], a, b, r0, dev)
        r_prev[k] = em_full(Mv[gg], Ov[gg], a, b, r0)
    return out


# ------------------------------------------------------------------------------------------------ per run
def corrected(ps, Mrow, a, r):
    """p'_s(r) with per-request r (array [N]) or scalar r."""
    r = np.asarray(r, np.float32)
    r = r[:, None] if r.ndim else r
    p = ps * np.where(Mrow, (1 - r) / a[:, None], r / (1 - a[:, None])).astype(np.float32)
    return p / p.sum(1, keepdims=True)


def analyse(runs, name, lr_model, t_gate, all_t=False):
    """summary of one run: per rule accuracy (full, late), splits, server use; curves; F-auto w; checks."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load(runs, name)
    pc, ps, Mrow, a, y = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"]
    cp, sp, ent, kind = D["cp"], D["sp"], D["ent"], D["kind"]
    N = len(y)
    wl, wf = weights(D, D["late"]), weights(D, np.ones(D["E"], bool))
    psr = corrected(ps, Mrow, a, RSTR)
    f_ans = (W * pc + (1 - W) * psr).argmax(1)
    lpc = np.log(np.maximum(pc, P_FLOOR))
    poe = (lpc + np.log(np.maximum(ps, P_FLOOR))).argmax(1)
    poe_bal = (lpc + np.log(np.maximum(psr, P_FLOOR))).argmax(1)
    del lpc
    Hs = server_entropy(ps)
    the = np.where(ent.astype(np.float64) <= Hs, cp, sp)
    rhat = r_causal(D, dev)
    em_ans = corrected(ps, Mrow, a, rhat.astype(np.float32)).argmax(1)
    feats = features(D)
    if lr_model is not None:
        scaler, model = lr_model
        P = model.predict_proba(scaler.transform(feats))[:, 1].astype(np.float32)
        lr_ans = (P[:, None] * pc + (1 - P[:, None]) * ps).argmax(1)
    else:
        lr_ans = np.full(N, -1)
    Hc_bar, Hs_bar = window_means(D, [ent.astype(np.float64), Hs])
    with np.errstate(invalid="ignore", divide="ignore"):
        w_auto = np.where(np.isfinite(Hc_bar) & ((Hc_bar + Hs_bar) > 0), Hs_bar / (Hc_bar + Hs_bar), 0.5)
    w_auto = w_auto.astype(np.float32)[:, None]
    fauto = (w_auto * pc + (1 - w_auto) * psr).argmax(1)
    psM = feats[:, 4]
    ans = {"B0": np.where(ent > TAU0, sp, cp), "B1": cp, "B2": sp, "B3": (pc + ps).argmax(1), "F": f_ans,
           "R-PoE": poe, "R-PoE-bal": poe_bal, "R-THE": the, "R-EM": em_ans, "R-LR": lr_ans,
           "F-nodebias": (W * pc + (1 - W) * ps).argmax(1), "F-eq": (0.5 * pc + 0.5 * psr).argmax(1),
           "F-auto": fauto, "F-gate": np.where(psM < t_gate, sp, f_ans) if t_gate is not None else np.full(N, -1),
           "kind oracle": np.where(kind == 0, cp, sp)}
    corr = {rl: (v == y).astype(np.float32) for rl, v in ans.items()}
    srv_req = {"B0": ent > TAU0, "B1": np.zeros(N, bool), "kind oracle": kind != 0}
    # ---- curves
    curves = {}
    cc = corr["B1"]
    j0 = np.searchsorted(TAU, ent, side="left")          # client answers at tau_j  <=>  j >= j0
    NJ = len(TAU) + 1
    for cname, sc in (("E", corr["B2"]), ("SF", corr["F"]), ("SB3", corr["B3"])):
        out = {}
        for wname, wv, m in (("full", wf, np.ones(D["E"], bool)), ("late", wl, D["late"])):
            sel = m[D["e"]]
            diff = np.bincount(j0, weights=wv * (cc - sc), minlength=NJ + 1)[:NJ]
            accj = (wv * sc).sum() + np.cumsum(diff)
            cnt = np.bincount(j0, weights=sel.astype(float), minlength=NJ + 1)[:NJ]
            srvj = 1.0 - np.cumsum(cnt) / sel.sum()
            out[wname] = dict(acc=accj, srv=srvj, at=interp(srvj, accj, OPTS))
        curves[cname] = out
    mpc = pc.max(1)
    jr = np.searchsorted(QG, mpc, side="right")           # client answers at q_j  <=>  j < jr
    NQ = len(QG)
    sc = corr["R-PoE"]
    for wname, wv, m in (("full", wf, np.ones(D["E"], bool)), ("late", wl, D["late"])):
        sel = m[D["e"]]
        diff = np.bincount(jr, weights=wv * (cc - sc), minlength=NQ + 2)
        accj = (wv * sc).sum() + (diff.sum() - np.cumsum(diff)[:NQ])
        cnt = np.bincount(jr, weights=sel.astype(float), minlength=NQ + 2)
        srvj = np.cumsum(cnt)[:NQ] / sel.sum()
        srvj, accj = np.append(srvj, 1.0), np.append(accj, (wv * sc).sum())   # end point: every request to the server
        curves.setdefault("R-ZTW", {})[wname] = dict(acc=accj, srv=srvj, at=interp(srvj[::-1], accj[::-1], OPTS))
    # ---- R-ZTW at the server use of B0 (same window); per-request soft correctness for the late splits
    res = {}
    b0_srv = {w_: float(srv_req["B0"][(D["late"] if w_ == "late" else np.ones(D["E"], bool))[D["e"]]].mean())
              for w_ in ("full", "late")}
    ztw_val = {w_: float(interp(curves["R-ZTW"][w_]["srv"][::-1], curves["R-ZTW"][w_]["acc"][::-1], [b0_srv[w_]])[0])
               for w_ in ("full", "late")}
    s_l = curves["R-ZTW"]["late"]["srv"]
    jhi = int(np.searchsorted(s_l, b0_srv["late"], side="left"))
    jhi = min(max(jhi, 1), len(s_l) - 1)
    jlo = jhi - 1
    alpha = 0.0 if s_l[jhi] == s_l[jlo] else (b0_srv["late"] - s_l[jlo]) / (s_l[jhi] - s_l[jlo])

    def ztw_correct(j):
        if j >= NQ:
            return sc
        return np.where(jr > j, cc, sc)
    corr["R-ZTW"] = (1 - alpha) * ztw_correct(jlo) + alpha * ztw_correct(jhi)
    # ---- metrics
    E, K = D["E"], D["K"]
    g = D["g"]
    home = D["home"]
    late = D["late"]

    def splits(c):
        with np.errstate(invalid="ignore", divide="ignore"):
            a_ = np.bincount(g, weights=c, minlength=E * K).reshape(E, K) / D["n_ek"]
        out = {}
        if D["has_home"]:
            out["home"] = float(np.mean([a_[e][home[e] == 1].mean() for e in np.flatnonzero(late) if (home[e] == 1).any()]))
            out["away"] = float(np.mean([a_[e][home[e] == 0].mean() for e in np.flatnonzero(late) if (home[e] == 0).any()]))
        else:
            out["home"] = out["away"] = float("nan")
        for i, kn in enumerate(KINDS):
            selk = kind == i
            ck = np.bincount(g, weights=c * selk, minlength=E * K).reshape(E, K)
            nk = np.bincount(g, weights=selk, minlength=E * K).reshape(E, K)
            with np.errstate(invalid="ignore", divide="ignore"):
                ak = np.where(nk > 0, ck / np.maximum(nk, 1), np.nan)
            vals = [np.nanmean(ak[e]) for e in np.flatnonzero(late) if np.isfinite(ak[e]).any()]
            out[kn] = float(np.mean(vals)) if vals else float("nan")
        return out
    for rl in RULES:
        c = corr[rl]
        if rl == "R-ZTW":
            r_ = dict(full=ztw_val["full"], late=ztw_val["late"], server_full=b0_srv["full"], server_late=b0_srv["late"])
        else:
            sr = srv_req.get(rl)
            r_ = dict(full=float((wf * c).sum()), late=float((wl * c).sum()),
                      server_full=float(sr.mean()) if sr is not None else 1.0,
                      server_late=float(sr[late[D["e"]]].mean()) if sr is not None else 1.0)
        if (rl == "R-LR" and lr_model is None) or (rl == "F-gate" and t_gate is None):
            r_.update(full=float("nan"), late=float("nan"))
        r_.update(splits(c))
        res[rl] = r_
    gate_t = {}
    if all_t:
        for t in T_GRID:
            cg = (np.where(psM < t, sp, f_ans) == y).astype(np.float32)
            gate_t[t] = dict(late=float((wl * cg).sum()), full=float((wf * cg).sum()))
    # ---- SF ratios (late)
    E_l, SF_l = curves["E"]["late"], curves["SF"]["late"]
    ratios = dict(B0_acc=res["B0"]["late"], E_max_acc=float(E_l["acc"].max()),
                  SF_reaches_B0_at=first_reach(SF_l["srv"], SF_l["acc"], res["B0"]["late"]),
                  SF_reaches_Emax_at=first_reach(SF_l["srv"], SF_l["acc"], float(E_l["acc"].max())))
    h = D["h"]
    tm = h.get("r12_timing_sec", {})
    lr_used = None
    prov = JR / "provenance" / f"{h['config']['run_id']}.json"
    if prov.exists():
        cmd = json.load(open(prov)).get("command")
        toks = cmd if isinstance(cmd, list) else str(cmd).split()
        lr_used = float(toks[toks.index("--learning_rate") + 1]) if "--learning_rate" in toks else "config"

    mains_per_client = float(np.mean(D["M"].sum(1)[np.unique(D["k"])]))
    out = dict(name=name, res=res, curves=curves, gate_t=gate_t, ratios=ratios, has_home=D["has_home"],
               w_auto=dict(mean=float(w_auto.mean()), min=float(w_auto.min()), max=float(w_auto.max()),
                           mean_late=float(w_auto[late[D["e"]], 0].mean())),
               rhat_mean=float(rhat.mean()), checks=D["checks"], n_req=N, C=D["C"], K=D["K"], run_id=h["config"]["run_id"],
               json_integrated=float(np.mean([e_["acc_total"] for e_ in h["eval"]])), mismatch=h.get("r12_eval_record_mismatch"),
               timing=tm, total_min=h["total_time_sec"] / 60, arch=h.get("arch_stats"),
               mains_per_client=mains_per_client, a_mean=float(D["a_ek"][D["present"]].mean()),
               lr=lr_used)
    return out


def analyse_cached(args):
    runs, name, lr_model, t_gate, all_t = args
    src = Path(runs) / f"{name}_evalprobs.npz"
    st = src.stat()
    key = hashlib.sha256(f"{SCRIPT_SHA}|{src}|{st.st_size}|{st.st_mtime_ns}|{t_gate}|{lr_model is not None}|{all_t}|{LATE_FROM}".encode()).hexdigest()[:16]
    cp_ = CACHE / f"{Path(runs).name}__{name}__{key}.pkl"
    if cp_.exists():
        return pickle.load(open(cp_, "rb"))
    out = analyse(Path(runs), name, lr_model, t_gate, all_t)
    out["sha"] = sha256(src)
    out["bytes"] = st.st_size
    pickle.dump(out, open(cp_, "wb"))
    print(f"  analysed {name}", flush=True)
    return out


def run_many(tasks):
    """records above 1 GB (K = 500, CIFAR-100) one at a time in their own worker (about 20 GB of memory each), the
    others in WORKERS workers."""
    if WORKERS <= 1:
        return [analyse_cached(t) for t in tasks]
    big = {i for i, t in enumerate(tasks) if (Path(t[0]) / f"{t[1]}_evalprobs.npz").stat().st_size > 1e9}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=ctx) as exb, ProcessPoolExecutor(WORKERS, mp_context=ctx) as exs:
        fut = {i: (exb if i in big else exs).submit(analyse_cached, t) for i, t in enumerate(tasks)}
        return [fut[i].result() for i in range(len(tasks))]


# ------------------------------------------------------------------------------------------------ main
def main():
    # ---- development: R-LR fitted on all late-window requests of the three runs, then t of F-gate
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    lr_p = CACHE / f"dev_lr__{SCRIPT_SHA[:16]}__{LATE_FROM}.pkl"
    if lr_p.exists():
        lr_model, lr_info = pickle.load(open(lr_p, "rb"))
    else:
        Xs, ys = [], []
        for s in DEV_SEEDS:
            name = f"r10_T40_s{s}"
            dg = sha256(DEVRUNS / f"{name}_evalprobs.npz")
            assert SMOKE or dg == DEV_SHA[s], f"sha256 of the development record {name} differs"
            D = load(DEVRUNS, name)
            F_ = features(D)
            m = D["late"][D["e"]]
            Xs.append(F_[m])
            ys.append((D["kind"] == 0)[m].astype(int))
            del D, F_
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        scaler = StandardScaler().fit(Xtr)
        model = LogisticRegression().fit(scaler.transform(Xtr), ytr)
        lr_model = (scaler, model)
        lr_info = dict(n_train=int(len(ytr)), share_main=float(ytr.mean()), coef=[float(c) for c in model.coef_[0]],
                       intercept=float(model.intercept_[0]), n_iter=int(model.n_iter_[0]),
                       mean=[float(v) for v in scaler.mean_], scale=[float(v) for v in scaler.scale_])
        pickle.dump((lr_model, lr_info), open(lr_p, "wb"))
    wcsv("R13b_dev_LR.csv", ["feature", "mean", "scale", "coefficient"],
         [[f, f"{m_:.6g}", f"{s_:.6g}", f"{c_:.6g}"] for f, m_, s_, c_ in zip(FEATURES, lr_info["mean"], lr_info["scale"], lr_info["coef"])]
         + [["intercept", "", "", f"{lr_info['intercept']:.6g}"], ["training requests", "", "", lr_info["n_train"]],
            ["share of Main requests", "", "", f"{lr_info['share_main']:.4f}"], ["lbfgs iterations", "", "", lr_info["n_iter"]]])
    DEVA = run_many([(str(DEVRUNS), f"r10_T40_s{s}", lr_model, T_GRID[0], True) for s in DEV_SEEDS])
    tmean = {t: float(np.mean([a["gate_t"][t]["late"] for a in DEVA])) for t in T_GRID}
    best = max(tmean.values())
    t_gate = min(t for t in T_GRID if tmean[t] == best)
    wcsv("R13b_dev_Fgate_t.csv", ["t"] + [f"late_pct_seed{s}" for s in DEV_SEEDS] + ["late_pct_mean", "chosen"],
         [[t] + [f"{a['gate_t'][t]['late'] * 100:.3f}" for a in DEVA] + [f"{tmean[t] * 100:.3f}", "yes" if t == t_gate else ""]
          for t in T_GRID])
    print(f"  F-gate threshold chosen on the development runs: t = {t_gate}", flush=True)
    DEVA = run_many([(str(DEVRUNS), f"r10_T40_s{s}", lr_model, t_gate, True) for s in DEV_SEEDS])
    wcsv("R13b_dev_rules.csv", ["rule", "late_pct_mean", "full_pct_mean"] + [f"late_pct_seed{s}" for s in DEV_SEEDS] + ["note"],
         [[rl, f"{np.mean([a['res'][rl]['late'] for a in DEVA]) * 100:.2f}", f"{np.mean([a['res'][rl]['full'] for a in DEVA]) * 100:.2f}"]
          + [f"{a['res'][rl]['late'] * 100:.2f}" for a in DEVA]
          + ["fitted on these runs (in-sample)" if rl == "R-LR" else ("t chosen on these runs" if rl == "F-gate" else "")]
          for rl in RULES])

    # ---- test runs
    jobs12 = json.load(open(RUNS12 / "jobs.json"))
    for j in jobs12:
        j.update(part="Round 12", runs=str(RUNS12))
    jobs13 = json.load(open(RUNS13 / "jobs.json")) if (RUNS13 / "jobs.json").exists() else []
    for j in jobs13:
        j.update(part="part B", runs=str(RUNS13), scenario=j["label"], training="fixed 0.4")
    jobs = [j for j in jobs12 + jobs13 if (Path(j["runs"]) / f"{j['name']}.json").exists()
            and (Path(j["runs"]) / f"{j['name']}_evalprobs.npz").exists()]
    if SMOKE:
        jobs = [j for j in jobs if j["name"] in SMOKE]
    A = run_many([(j["runs"], j["name"], lr_model, t_gate, False) for j in jobs])
    for a, j in zip(A, jobs):
        a["job"] = j
    print(f"  analysed {len(A)} test runs ({sum(j['part'] == 'part B' for j in jobs)} of part B, {len(jobs13)} planned)", flush=True)
    G = {}
    for a in A:
        G.setdefault((a["job"]["part"], a["job"]["scenario"], a["job"]["training"]), []).append(a)
    order = list(dict.fromkeys((j["part"], j["scenario"], j["training"]) for j in jobs))
    mean = lambda key, rl, w_: float(np.mean([a["res"][rl][w_] for a in G[key]])) * 100

    # ---- start check against Round 12 table 1
    t12 = {(r["scenario"], r["training"]): r for r in csv.DictReader(open(HERE.parent / "driftgate_tmc_r12_fusion_eval" / "tables" / "R12_T1_rules.csv"))}
    rows, bad = [], []
    for key in order:
        if key[0] != "Round 12" or (key[1], key[2]) not in t12:
            continue
        for rl in ("B0", "B3", "F"):
            ref = float(t12[(key[1], key[2])][f"{rl}_late"])
            v = mean(key, rl, "late")
            ok = abs(v - ref) <= 0.05
            rows.append([key[1], key[2], rl, f"{ref:.2f}", f"{v:.4f}", f"{v - ref:+.4f}", ok])
            if not ok:
                bad.append((key, rl, ref, v))
    wcsv("R13b_T0_r12_recompute_check.csv", ["scenario", "training", "rule", "R12_table1_late_pct", "recomputed_late_pct",
                                            "difference_pp", "within_0.05pp"], rows)
    if bad and not SMOKE:
        raise SystemExit(f"start check failed (directive 1.3): {bad}")

    # ---- table 0: checks, manifest, sha256, times
    rows = []
    for key in order:
        for a in G[key]:
            j, tm = a["job"], a["timing"]
            rows.append([j["name"], key[0], key[1], key[2], j["seed"], a["run_id"], a["n_req"], a["K"], a["C"],
                         a["checks"]["mk_consistent"], a["checks"]["mk_prov"], a["checks"]["ids_rebuilt"], a["mismatch"],
                         f"{a['res']['B0']['full'] * 100:.4f}", f"{a['json_integrated'] * 100:.4f}",
                         f"{tm.get('train', float('nan')) / 60:.1f}", f"{tm.get('eval', float('nan')) / 60:.2f}",
                         f"{tm.get('record', 0.0) / 60:.2f}", f"{a['total_min']:.1f}", a["lr"], a["sha"], a["bytes"]])
    wcsv("R13b_T0_checks_manifest.csv", ["run", "part", "scenario", "training", "seed", "run_id", "requests", "clients",
                                        "classes", "Mk_consistent_all_rounds", "Mk_equals_provenance",
                                        "client_ids_rebuilt_from_order", "record_vs_evaluator_mismatch",
                                        "B0_full_recomputed_pct", "run_json_integrated_pct", "train_min", "eval_min",
                                        "record_min", "total_min", "learning_rate", "evalprobs_sha256", "evalprobs_bytes"], rows)

    # ---- table 1
    hdr = [f"{rl}_late" for rl in RULES] + [f"{rl}_full" for rl in RULES]
    rows, rows_s = [], []
    for key in order:
        rows.append(list(key) + [len(G[key])] + [f"{mean(key, rl, 'late'):.2f}" for rl in RULES]
                    + [f"{mean(key, rl, 'full'):.2f}" for rl in RULES]
                    + [f"{np.mean([a['res']['B0']['server_late'] for a in G[key]]) * 100:.1f}",
                       f"{mean(key, 'F', 'late') - mean(key, 'B0', 'late'):+.2f}"])
        for a in G[key]:
            rows_s.append(list(key) + [a["job"]["seed"]] + [f"{a['res'][rl]['late'] * 100:.2f}" for rl in RULES]
                          + [f"{a['res'][rl]['full'] * 100:.2f}" for rl in RULES])
    wcsv("R13b_T1_rules.csv", ["part", "scenario", "training", "seeds"] + hdr + ["B0_server_use_late_pct", "F_minus_B0_late_pp"], rows)
    wcsv("R13b_T1_rules_per_seed.csv", ["part", "scenario", "training", "seed"] + hdr, rows_s)
    wcsv("R13b_T1_server_use.csv", ["part", "scenario", "training"] + [f"{rl}_server_use_late_pct" for rl in RULES],
         [list(key) + [f"{np.mean([a['res'][rl]['server_late'] for a in G[key]]) * 100:.1f}" for rl in RULES] for key in order])

    # ---- table 2: splits for S1, S2 (fixed 0.4) and part B
    t2_keys = [k for k in order if (k[0] == "Round 12" and k[1] in ("S1", "S2") and k[2] == "fixed 0.4") or k[0] == "part B"]
    rows = []
    for key in t2_keys:
        for rl in RULES:
            rows.append(list(key[1:]) + [rl] + [f"{np.mean([a['res'][rl][s] for a in G[key]]) * 100:.2f}" for s in SPLITS])
    wcsv("R13b_T2_splits.csv", ["scenario", "training", "rule"] + [f"{s}_pct" for s in SPLITS], rows)

    # ---- table 3: curves and SF ratios
    rows = []
    for key in order:
        for cn in CURVES:
            for w_ in ("late", "full"):
                vals = np.array([a["curves"][cn][w_]["at"] for a in G[key]]) * 100
                rows.append(list(key[1:]) + [cn, w_] + [f"{v:.2f}" if np.isfinite(v) else "" for v in vals.mean(0)])
    wcsv("R13b_T3_curves.csv", ["scenario", "training", "curve", "window"] + [f"acc_at_server_use_{o}" for o in OPTS], rows)
    rows, rows_g = [], []
    for key in t2_keys:
        for a in G[key]:
            r_ = a["ratios"]
            rows.append(list(key[1:]) + [a["job"]["seed"], f"{r_['B0_acc'] * 100:.2f}", f"{r_['SF_reaches_B0_at']:.3f}",
                                         f"{r_['E_max_acc'] * 100:.2f}", f"{r_['SF_reaches_Emax_at']:.3f}"])
        rows_g.append(list(key[1:]) + ["mean"] + [f"{np.mean([a['ratios'][x] for a in G[key]]) * (100 if 'acc' in x else 1):.{2 if 'acc' in x else 3}f}"
                                                  for x in ("B0_acc", "SF_reaches_B0_at", "E_max_acc", "SF_reaches_Emax_at")])
    wcsv("R13b_T3_SF_ratios.csv", ["scenario", "training", "seed", "B0_late_pct", "SF_first_server_use_at_B0_acc",
                                   "E_max_late_pct", "SF_first_server_use_at_E_max"], rows + rows_g)

    # ---- table 4: part B structures and conditions
    rows = []
    s1k = ("Round 12", "S1", "fixed 0.4")
    for key in [k for k in order if k[0] == "part B"] + ([s1k] if s1k in G else []):
        L_ = G[key]
        ar = L_[0]["arch"]
        if ar is None:   # the default split CNN of S1 (Round 12 runs do not record it)
            from models.architectures import ModelFactory
            from src.models_ext import arch_stats
            ar = arch_stats(ModelFactory(num_classes=L_[0]["C"]))
        tmean_ = lambda f_: float(np.mean([a["timing"].get(f_, float("nan")) for a in L_])) / 60
        rows.append([key[1] if key[0] == "part B" else "S1 default (Round 12)", len(L_),
                     ar.get("client_block_params", ""), ar.get("client_exit_params", ""), ar.get("server_block_and_exit_params", ""),
                     ar.get("client_block_flops", ""), ar.get("client_exit_flops", ""), ar.get("server_block_and_exit_flops", ""),
                     "x".join(map(str, ar.get("smashed_shape", []))), ar.get("smashed_bytes_float32", ""),
                     f"{tmean_('train'):.1f}", f"{tmean_('eval'):.2f}", f"{np.mean([a['total_min'] for a in L_]):.1f}",
                     f"{np.mean([a['mains_per_client'] for a in L_]):.2f}", f"{np.mean([a['a_mean'] for a in L_]):.3f}",
                     ";".join(sorted({str(a['lr']) for a in L_}))])
    wcsv("R13b_T4_structures.csv", ["group", "runs", "client_block_params", "client_exit_params", "server_block_and_exit_params",
                                    "client_block_flops_per_image", "client_exit_flops_per_image",
                                    "server_block_and_exit_flops_per_image", "smashed_shape", "smashed_bytes_float32",
                                    "train_min_mean", "eval_min_mean", "total_min_mean", "Main_classes_per_client_mean",
                                    "a_mean", "learning_rate"], rows)
    wcsv("R13b_Fauto_w.csv", ["run", "scenario", "training", "w_mean_all_requests", "w_mean_late", "w_min", "w_max", "rhat_mean_R-EM"],
         [[a["job"]["name"], key[1], key[2], f"{a['w_auto']['mean']:.3f}", f"{a['w_auto']['mean_late']:.3f}",
           f"{a['w_auto']['min']:.3f}", f"{a['w_auto']['max']:.3f}", f"{a['rhat_mean']:.3f}"] for key in order for a in G[key]])

    # ---- decisions
    fx = lambda sc: ("Round 12", sc, "fixed 0.4")
    dec = {"F_gate_t_chosen_on_development": t_gate, "runs_analysed": len(A),
           "part_B_runs_analysed": sum(1 for a in A if a["job"]["part"] == "part B"), "part_B_runs_planned": len(jobs13),
           "condition_2": "not run: Main classes per client = max(1, int(10 * 0.2)) = 2; one fewer is 1, below the floor of 2, "
                          "so the value stays 2 (S1 default); the partition has no Dirichlet coefficient"}
    d1 = {}
    for sc in ("S1", "S2"):
        if fx(sc) in G:
            Fv = mean(fx(sc), "F", "late")
            d1[sc] = {rl: round(mean(fx(sc), rl, "late") - Fv, 3) for rl in BASELINES}
    hits = [f"{rl} ({sc})" for sc, v in d1.items() for rl, d in v.items() if d >= 0]
    dB = {}
    for key in [k for k in order if k[0] == "part B"]:
        Fv = mean(key, "F", "late")
        dB[key[1]] = {rl: round(mean(key, rl, "late") - Fv, 3) for rl in BASELINES}
    hitsB = [f"{rl} ({sc})" for sc, v in dB.items() for rl, d in v.items() if d >= 0]
    dec["decision_1_baselines"] = {"answer": hits if hits else "none", "rule_minus_F_late_pp": d1,
                                   "part_B_answer": hitsB if hitsB else ("none" if dB else "no part-B run analysed"),
                                   "part_B_rule_minus_F_late_pp": dB}
    sets = {"definition": [], "parenthesis": []}
    for key in order:
        mx = max(mean(key, "B1", "late"), mean(key, "B3", "late"))
        Fv, Fa = mean(key, "F", "late"), mean(key, "F-auto", "late")
        base_item = dict(group=" / ".join(key[1:]) + ("" if key[0] == "Round 12" else " (part B)"), F=round(Fv, 3),
                         max_B1_B3=round(mx, 3), F_auto=round(Fa, 3))
        for reading, is_low in (("definition", Fv < mx),
                                ("parenthesis", (key[0] == "Round 12" and key[1] == "ResNet-18") or (key[0] == "part B" and Fv < mx))):
            sets[reading].append(dict(base_item, F_below_max_B1_B3=bool(is_low),
                                      ok=bool(Fa > mx) if is_low else bool(Fa >= Fv - 0.3)))
    ans = {r_: "yes" if sets[r_] and all(i["ok"] for i in sets[r_]) else "no" for r_ in sets}
    dec["decision_2_F_auto"] = {"answer": ans["definition"],
                                "groups_F_below_max_B1_B3": [i for i in sets["definition"] if i["F_below_max_B1_B3"]],
                                "other_groups": [i for i in sets["definition"] if not i["F_below_max_B1_B3"]],
                                "for_information_parenthesis_reading": {
                                    "answer": ans["parenthesis"],
                                    "groups_with_condition_i": [i["group"] for i in sets["parenthesis"] if i["F_below_max_B1_B3"]],
                                    "failing_groups": [i["group"] for i in sets["parenthesis"] if not i["ok"]]}}
    d3 = {}
    for sc in ("S1", "S2"):
        if fx(sc) in G:
            L_ = G[fx(sc)]
            ga = float(np.mean([a["res"]["F-gate"]["away"] - a["res"]["F"]["away"] for a in L_])) * 100
            gl = mean(fx(sc), "F-gate", "late") - mean(fx(sc), "F", "late")
            d3[sc] = dict(away_F_gate_minus_F_pp=round(ga, 3), late_F_gate_minus_F_pp=round(gl, 3), ok=bool(ga >= 1.0 and gl >= 0))
    dec["decision_3_F_gate"] = {"answer": "yes" if d3 and all(v["ok"] for v in d3.values()) and len(d3) == 2 else "no",
                                "t": t_gate, "values": d3}
    dec["item_4_structures"] = {key[1]: dict(F_minus_B0_pp=round(mean(key, "F", "late") - mean(key, "B0", "late"), 3),
                                            F_minus_B3_pp=round(mean(key, "F", "late") - mean(key, "B3", "late"), 3))
                                for key in order if key[0] == "part B" and key[1] != "CIFAR-100"}
    base = round(mean(s1k, "F", "late") - mean(s1k, "B3", "late"), 3) if s1k in G else None
    dec["item_5_conditions"] = {"S1 default (Round 12, seeds 0-4) F_minus_B3_pp": base,
                                "condition 1 CIFAR-100 F_minus_B3_pp": (round(mean(("part B", "CIFAR-100", "fixed 0.4"), "F", "late")
                                                                             - mean(("part B", "CIFAR-100", "fixed 0.4"), "B3", "late"), 3)
                                                                       if ("part B", "CIFAR-100", "fixed 0.4") in G else None),
                                "condition 2": "not run (see condition_2)"}
    with open(OUT / "decision.json", "w") as f:
        json.dump(dec, f, indent=1)
    print(json.dumps({k: dec[k] for k in ("decision_1_baselines", "decision_3_F_gate")}, indent=1)[:3000], flush=True)
    print("  decision 2:", dec["decision_2_F_auto"]["answer"], flush=True)
    figures(G, order, t2_keys)


def figures(G, order, t2_keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0a", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BLUE, ORANGE, GREEN, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c8323c"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
                         "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    lab = {"B0": "entropy routing (tau 0.8)", "B1": "client exit only", "B2": "server exit only", "B3": "mean of both exits",
           "F": "fusion F", "R-PoE": "product of experts (FedRoD-like)", "R-PoE-bal": "product of experts, corrected server",
           "R-THE": "lower-entropy exit (FedTHE-like)", "R-ZTW": "confidence exit + product (ZTW-like), B0 server use",
           "R-EM": "EM prior correction", "R-LR": "per-request weight (logistic regression)", "F-nodebias": "F without correction",
           "F-eq": "F with equal weights", "F-auto": "F with entropy weights", "F-gate": "F with server gate",
           "kind oracle": "request-kind oracle"}
    sty = {"B0": (INK, "s"), "B1": (MUTED, "v"), "B2": (INK2, "^"), "B3": (GREEN, "D"), "F": (BLUE, "o"),
           "R-PoE": (ORANGE, "P"), "R-PoE-bal": (ORANGE, "X"), "R-THE": (ORANGE, "<"), "R-ZTW": (ORANGE, ">"),
           "R-EM": (RED, "h"), "R-LR": (RED, "p"), "F-nodebias": (BLUE, "1"), "F-eq": (BLUE, "2"), "F-auto": (BLUE, "3"),
           "F-gate": (BLUE, "4"), "kind oracle": (PURPLE, "*")}
    keys = [k for k in (("Round 12", "S1", "fixed 0.4"), ("Round 12", "S2", "fixed 0.4")) if k in G]
    if keys:
        fig, axs = plt.subplots(1, len(keys), figsize=(4.0 * len(keys) + 2.6, 3.2), layout="constrained", squeeze=False)
        for ax, key in zip(axs[0], keys):
            for rl in RULES:
                hx = np.mean([a["res"][rl]["home"] for a in G[key]]) * 100
                ay = np.mean([a["res"][rl]["away"] for a in G[key]]) * 100
                c_, m_ = sty[rl]
                ax.plot(hx, ay, ls="none", marker=m_, ms=6 if rl != "kind oracle" else 9, color=c_, label=lab[rl], mew=1.2)
            ax.set_title(key[1], fontsize=8, color=INK2)
            ax.set_xlabel("accuracy of clients at home (%)")
            ax.set_ylabel("accuracy of clients away (%)")
        axs[0][-1].legend(fontsize=6.2, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.savefig(FIG / "R13b_fig1_home_away.pdf")
        fig.savefig(FIG / "R13b_fig1_home_away.png", dpi=300)
        plt.close(fig)
        fig, axs = plt.subplots(1, len(keys), figsize=(4.0 * len(keys) + 1.6, 3.0), layout="constrained", squeeze=False)
        grid = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
        for ax, key in zip(axs[0], keys):
            for cn, c_, ls_, lab_ in (("E", INK, "-", "entropy routing (curve E)"), ("SF", BLUE, "-", "selective fusion (curve SF)"),
                                      ("SB3", GREEN, "--", "selective mean of exits (curve SB3)"),
                                      ("R-ZTW", ORANGE, "-.", "confidence exit + product (ZTW-like)")):
                vals = []
                for a in G[key]:
                    cu = a["curves"][cn]["late"]
                    srv, acc = (cu["srv"][::-1], cu["acc"][::-1]) if cn == "R-ZTW" else (cu["srv"], cu["acc"])
                    vals.append(interp(srv, acc, grid))
                ax.plot(grid, np.nanmean(vals, axis=0) * 100, color=c_, ls=ls_, label=lab_)
            ax.axvline(float(np.mean([a["res"]["B0"]["server_late"] for a in G[key]])), color=AXIS, lw=0.8, ls=":")
            ax.set_title(key[1], fontsize=8, color=INK2)
            ax.set_xlabel("server use (share of requests that need the server exit)")
            ax.set_ylabel("accuracy (%)")
            ax.set_xlim(0, 1)
        axs[0][-1].legend(fontsize=6.2, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.savefig(FIG / "R13b_fig2_curves.pdf")
        fig.savefig(FIG / "R13b_fig2_curves.png", dpi=300)
        plt.close(fig)
    bk = [("Round 12", "S1", "fixed 0.4"), ("Round 12", "ResNet-18", "fixed 0.4")] + \
         [k for k in order if k[0] == "part B" and k[1] != "CIFAR-100"]
    bk = [k for k in bk if k in G]
    names = {("Round 12", "S1", "fixed 0.4"): "default CNN, S1\n(Round 12)",
             ("Round 12", "ResNet-18", "fixed 0.4"): "ResNet-18 middle,\nschedule A (Round 12)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    x = np.arange(len(bk))
    br = [("B0", INK), ("B1", MUTED), ("B2", INK2), ("B3", GREEN), ("F", BLUE)]
    for i, (rl, c_) in enumerate(br):
        ax.bar(x + (i - 2) * 0.16, [np.mean([a["res"][rl]["late"] for a in G[k]]) * 100 for k in bk], width=0.15,
               color=c_, label=lab[rl], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(k, k[1].replace(", ", "\n")) for k in bk], fontsize=6.5)
    ax.set_ylabel("accuracy, rounds from 30 on (%)")
    lo_ = min(np.mean([a["res"][rl]["late"] for a in G[k]]) * 100 for k in bk for rl, _ in br)
    ax.set_ylim(max(0, lo_ - 5), None)
    ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(FIG / "R13b_fig3_structures.pdf")
    fig.savefig(FIG / "R13b_fig3_structures.png", dpi=300)
    plt.close(fig)
    with open(FIG / "R13b_figure_captions.md", "w") as f:
        f.write("# Round 13b figure captions\n\n## R13b_fig1_home_away\n\nAccuracy of clients at home against accuracy of clients "
                "away for each inference rule in the commute scenario S1 (seeds 0 to 4) and the GeoLife scenario S2 (seeds 0 to 2), "
                "training ratio 0.4, evaluation rounds from training round 30 on. Related-work rules are computed from the recorded "
                "probabilities of both exits; the confidence-exit rule is read at the server use of entropy routing.\n\n"
                "## R13b_fig2_curves\n\nAccuracy against the share of requests that need the server exit in S1 and S2: entropy routing, "
                "selective fusion (client exit below the entropy threshold, otherwise the fusion F), selective mean of both exits, and "
                "the confidence exit with a product of both exits beyond it. Evaluation rounds from training round 30 on, seed means. "
                "The dotted line marks the server use of entropy threshold 0.8.\n\n"
                "## R13b_fig3_structures\n\nAccuracy of entropy routing, either exit alone, the mean of both exits and the fusion F for "
                "each model and split point, evaluation rounds from training round 30 on, seed means. The default CNN (S1, seeds 0 to 4) "
                "and ResNet-18 (middle split, stepwise schedule A, seeds 0 to 2) come from Round 12; ResNet-20 and VGG-11 run S1 "
                "with seeds 0 to 2.\n")
    print("  figures written", flush=True)


if __name__ == "__main__":
    main()
