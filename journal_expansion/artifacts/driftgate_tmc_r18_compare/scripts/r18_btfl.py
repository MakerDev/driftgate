"""Round 18 D: BTFL inference adaptation on the S1 / S2 frozen replays (one run with fixed defaults; R18_BTFL_compatibility.md).

BTFL (Zhou & Liu, KDD 2025; arXiv 2503.06633, code github.com/ZhouYuCS/BTFL) mixes a global head and a personal head on a
shared feature z with weight e from a Beta prior updated by entropy/likelihood events (HBU) and a per-sample likelihood
ratio (CBU). Adaptation to the split model (not full BTFL; no training):
  personal head  = device exit (p_d), global head = edge exit (raw p_e, two-cell devices: averaged logits as recorded)
  z              = f_d, the device-block output pooled over space (128, non-negative), from the Round 17 records
  local DLE      = per-dimension frequencies of round(tanh(z)) over device k's own training samples (through h_k)
  global DLE     = mean of the other devices' local DLE tables (code: leave-one-out mean). These tables come from
                   other devices' blocks h_j: the shared-feature assumption of BTFL does NOT hold here.
  Hbar_l, Hbar_g = mean entropies of the device exit and of the home-cell edge exit over device k's training samples
                   (forward pass of the frozen Round 15 checkpoint; no training)
Paper equations (Eq. 5-21) with the numerical choices of the official code where the paper is silent or would fail:
smoothed likelihood log(0.6 P + 0.2) (q = 2), integral over m in [0.01, 0.99] (100-point trapezoid, numpy.trapezoid), EXD events counted
in alpha and IND events in beta (code assignment, m = P(EXD)), prior state updated after the prediction, one request at a
time, reset at the start of the day per device; pruning lambda = 16 (paper, Algorithm 1). Final answer: argmax of
e p_e + (1 - e) p_d (paper Eq. 21, probability mixing; the code mixes log-probabilities).
Usage: python r18_btfl.py SETTING SEED  (SETTING in {S1, S2}) -> cache/btfl_<setting>_s<seed>.pkl
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
for p in (str(ROOT), str(JR)):
    if p not in sys.path:
        sys.path.insert(0, p)
from data.partition import get_cifar10            # noqa: E402
from models.architectures import ModelFactory     # noqa: E402
from src.r6_env import load_env                    # noqa: E402
from src.r6_requests import build_partition        # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("r18", HERE / "scripts" / "r18_analysis.py")
r18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r18)
m13, m14 = r18.m13, r18.m14
LAMBDA, GRID = 16, np.linspace(0.01, 0.99, 100)


def ent(logits):
    lp = torch.log_softmax(logits.float(), 1)
    return -(lp.exp() * lp).sum(1)


def train_stats(st, seed, dev):
    """Hbar_l, Hbar_g per device and the pooled training features, from the Round 15 checkpoint."""
    ck = torch.load(JR / "runs" / "phaseT15_replay" / f"r15_{st.lower()}_fixed040_s{seed}_ckpt.pt", map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    env, _ = load_env(JR / "runs" / "phaseT6_env" / f"{st}_seed{seed}.npz")
    train_ds, _ = get_cifar10(data_root=str(ROOT / "data_cache"))
    labels = np.array(train_ds.targets)
    indices, mains, _, _ = build_partition(labels, env, seed, 10, cfg)
    X = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(dev)
    mf = ModelFactory(dataset="cifar10", num_classes=10)
    home = np.asarray(env["home_cell"]).astype(int)
    servers = {}
    for z, sd in ck["server_avg"].items():
        sm = mf.make_server().to(dev)
        sm.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        servers[int(z)] = sm.eval()
    K = ck["K"]
    Hl, Hg, fd, kk, yy = np.zeros(K), np.zeros(K), [], [], []
    with torch.no_grad():
        for k in range(K):
            cm = mf.make_client().to(dev)
            cm.load_state_dict({kk_: v.to(dev) for kk_, v in ck["client_states"][k].items()})
            cm.eval()
            idx = torch.as_tensor(np.asarray(indices[k], dtype=np.int64), device=dev)
            hl, hg, f = [], [], []
            for s in range(0, len(idx), 512):
                lg, rep = cm(X[idx[s:s + 512]])
                hl.append(ent(lg).cpu())
                hg.append(ent(servers[int(home[k])](rep)[0]).cpu())
                f.append(rep.float().mean((2, 3)).cpu())
            Hl[k], Hg[k] = float(torch.cat(hl).mean()), float(torch.cat(hg).mean())
            fd.append(torch.cat(f).numpy())
            kk.append(np.full(len(idx), k))
            yy.append(labels[np.asarray(indices[k])])
    return Hl, Hg, np.concatenate(fd), np.concatenate(kk), np.concatenate(yy), mains


def beta_pdf(x, a, b):
    from scipy.special import gammaln
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - (gammaln(a) + gammaln(b) - gammaln(a + b)))


def run(st, seed):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Hl_bar, Hg_bar, own_fd, own_k, own_y, mains = train_stats(st, seed, dev)
    runs17 = JR / "runs" / "phaseT17_featgate"
    rn = f"r17_{st.lower()}_replay_s{seed}"
    own17 = np.load(runs17 / f"{rn}_own.npz")
    check_partition = bool(np.array_equal(own17["own_k"], own_k) and np.array_equal(own17["own_y"], own_y))
    check_fd = float(np.abs(own17["own_fd"] - own_fd).max())
    D = m13.load(runs17, rn)
    fd = np.load(runs17 / f"{rn}_features.npz")["fd"].astype(np.float32)
    K, N = D["K"], len(D["y"])
    zt = np.rint(np.tanh(own_fd)).astype(np.int8)          # q = 2: round(tanh(z) * (q - 1))
    assert zt.min() >= 0
    P0 = np.stack([(zt[own_k == k] == 0).mean(0) for k in range(K)])   # [K, d] frequency of bit 0
    P0g = (P0.sum(0)[None, :] - P0) / (K - 1)                          # leave-one-out mean of the other devices
    zr = np.rint(np.tanh(fd)).astype(np.int8)
    k_i = D["k"]

    def logq(P):
        p_bit = np.where(zr == 0, P[k_i], 1 - P[k_i])
        return np.log(0.6 * p_bit + 0.2).sum(1)
    lq_l, lq_g = logq(P0), logq(P0g)
    d = fd.shape[1]
    H_l = D["ent"].astype(np.float64)
    H_g = m13.server_entropy(D["ps"])
    hl, hg = Hl_bar[k_i], Hg_bar[k_i]
    tau = np.exp(lq_l - lq_g)
    exd = (tau < 1) & (H_g < hg) & (H_l > hl)
    ind = (tau > 1) & (H_l < hl) & (H_g > hg)
    u_l, u_g = np.exp((H_l - hl) / hl), np.exp((H_g - hg) / hg)
    log_tau_hat = (u_l * lq_l - u_g * lq_g) / d
    # HBU scan per device in (round, arrival) order; state before the request's own update
    oc, first = m14.client_order(D)
    al, be = np.empty(N), np.empty(N)
    a_, b_ = 1.0, 1.0
    ex_o, in_o, fi = exd[oc], ind[oc], first
    for j in range(N):
        if fi[j]:
            a_, b_ = 1.0, 1.0
        al[oc[j]], be[oc[j]] = a_, b_
        a_ += ex_o[j]
        b_ += in_o[j]
        if a_ + b_ > LAMBDA:
            s_ = a_ + b_
            a_, b_ = 1 + a_ / s_, 1 + b_ / s_
    # CBU / DPI: e = integral of Beta(m; alpha, beta) m / (m + (1 - m) tau_hat) dm on [0.01, 0.99]
    e = np.empty(N)
    th = np.exp(np.clip(log_tau_hat, -50, 50))
    for s in range(0, N, 200000):
        pdf = beta_pdf(GRID[None, :], al[s:s + 200000, None], be[s:s + 200000, None])
        f = pdf * GRID[None, :] / (GRID[None, :] + (1 - GRID[None, :]) * th[s:s + 200000, None])
        e[s:s + 200000] = np.trapezoid(f, GRID, axis=1)
    ans = (e[:, None].astype(np.float32) * D["ps"] + (1 - e[:, None].astype(np.float32)) * D["pc"]).argmax(1)
    res = r18.pr_metrics(D, ans == D["y"])
    out = dict(setting=f"{st} replay", seed=seed, name=rn, pr=res, e_mean=float(e.mean()), e_sd=float(e.std()),
               e_by_kind={kn: float(e[D["kind"] == i].mean()) for i, kn in enumerate(["Main", "OOP", "OOR"])},
               exd_share=float(exd.mean()), ind_share=float(ind.mean()), tau_gt1=float((tau > 1).mean()),
               bit1_share_requests=float(zr.mean()), bit1_share_train=float(zt.mean()),
               auroc_tau_nonmain=float(m14.auroc(-(lq_l - lq_g), D["kind"] != 0)),
               Hl_bar=float(Hl_bar.mean()), Hg_bar=float(Hg_bar.mean()), check_partition=check_partition,
               check_train_fd_maxabs=check_fd, seconds=time.time() - t0)
    pickle.dump(out, open(HERE / "cache" / f"btfl_{st}_s{seed}.pkl", "wb"))
    print(json.dumps({k: v for k, v in out.items() if k != "pr"}, indent=1))


def table():
    """tables/R18_BTFL_adaptation.csv: the adaptation against the full-offloading rules on the same replay requests."""
    import csv
    rows = []
    for st, seeds in (("S1", range(5)), ("S2", range(3))):
        B = [pickle.load(open(HERE / "cache" / f"btfl_{st}_s{s}.pkl", "rb")) for s in seeds]
        A = [pickle.load(open(HERE / "cache" / f"{r18.run_name(st + ' replay', s)}.pkl", "rb")) for s in seeds]
        full = np.ones(len(A[0]["er"]), bool)
        val = lambda a, key, sp: r18.rowval(a, key, sp, full) * 100
        refs = {r: np.mean([val(a, f"{r} @ 1.0", "all") for a in A]) for r in r18.REFS}
        strongest = max(refs, key=refs.get)
        for sp in r18.SPLITS:
            bt = np.array([float(np.nanmean(b["pr"][sp])) * 100 for b in B])
            row = [f"{st} replay", sp, r18.fmt(bt)]
            for r in ("DriftGate-P", "Probability average", "Corrected edge only", strongest):
                v = np.array([val(a, f"{r} @ 1.0", sp) for a in A])
                row += [r18.fmt(v), r18.fmt(bt - v, 3)]
            rows.append(row + [strongest])
        rows.append([f"{st} replay", "e (weight of the edge head)", r18.fmt([b["e_mean"] for b in B], 3),
                     "Main " + r18.fmt([b["e_by_kind"]["Main"] for b in B], 3), "OOP " + r18.fmt([b["e_by_kind"]["OOP"] for b in B], 3),
                     "OOR " + r18.fmt([b["e_by_kind"]["OOR"] for b in B], 3), "EXD events " + r18.fmt([b["exd_share"] for b in B], 3),
                     "IND events " + r18.fmt([b["ind_share"] for b in B], 3), "AUROC of -log(Q_l/Q_g) for non-Main " + r18.fmt([b["auroc_tau_nonmain"] for b in B], 3),
                     "bit-1 share requests / training " + r18.fmt([b["bit1_share_requests"] for b in B], 3) + " / " + r18.fmt([b["bit1_share_train"] for b in B], 3),
                     "Hbar_l / Hbar_g " + r18.fmt([b["Hl_bar"] for b in B], 3) + " / " + r18.fmt([b["Hg_bar"] for b in B], 3), ""])
        rows.append([f"{st} replay", "checks", "partition equal to Round 17: " + str(all(b["check_partition"] for b in B)),
                     "training f_d max difference to Round 17: " + str(max(b["check_train_fd_maxabs"] for b in B)),
                     "seconds per run " + r18.fmt([b["seconds"] for b in B], 1), "", "", "", "", "", "", ""])
    with open(HERE / "tables" / "R18_BTFL_adaptation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "split", "BTFL adaptation mean (SD)", "DriftGate", "BTFL - DriftGate pp", "Probability average", "BTFL - Probability average pp",
                    "Corrected edge only", "BTFL - corrected edge only pp", "strongest reference (beta 1)", "BTFL - strongest pp", "strongest reference"])
        w.writerows(rows)
    print(f"  [R18_BTFL_adaptation.csv] {len(rows)} rows")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    else:
        run(sys.argv[1], int(sys.argv[2]))
