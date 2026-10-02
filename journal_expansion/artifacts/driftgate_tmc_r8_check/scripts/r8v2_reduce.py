"""Round 8 check v2, step 1: per-request records (runs/phaseT8_check/r8v2_*_requests.npz, kept on the server, not in
git) -> per client and evaluation round counts on the tau grid (reduced/r8v2_*_reduced.npz, in git).

tau grid: 0, 0.05, ..., 2.30 (47 values) and infinity (index 47; every request ends at the client exit).
Routing as in the runner: the client exit answers when entropy <= tau, compared in float32 (the runner compares
the float32 entropy tensor with the Python float 0.8, i.e. in float32).
For every block b (one per lambda_inf; plus the installed model when lambda_t is not a lambda_inf value),
evaluation round e and client k:
  n, n_main, n_nonmain                         requests (block independent)
  corr[b, e, k, j], nsrv[b, e, k, j]           correct answers / requests answered by the server exit at tau_j
  cmain[b, e, k, j], cnon[b, e, k, j]          correct answers among Main / non-Main requests at tau_j
  n_cc, n_sc                                   client-exit correct, server-exit correct (all requests)
  n_gain, n_loss                               client wrong and server right / client right and server wrong
  home[e, k]                                   at_home of the client in that evaluation round
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
RUNS = JR / "runs" / "phaseT8_check"
RED = HERE / "reduced"
RED.mkdir(parents=True, exist_ok=True)
TAU = np.round(np.arange(0, 2.30 + 1e-9, 0.05), 2)            # 47 finite values
TAU32 = TAU.astype(np.float32)
NJ = len(TAU) + 1                                              # + infinity
ARMS, SEEDS = ["T15", "T40", "T60"], [5, 6, 7]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_run(name, src=None, outdir=RED):
    src = Path(src) if src else RUNS / f"{name}_requests.npz"
    q = np.load(src)
    E, K = len(q["eval_rounds"]), 50
    e, k = q["req_eval_index"].astype(np.int64), q["req_client"].astype(np.int64)
    g = e * K + k
    G = E * K
    y = q["req_label"]
    kind = q["req_kind"]
    assert (kind <= 2).all(), "request outside Main / OOP / OOR"
    main = kind == 0
    n = np.bincount(g, minlength=G)
    n_main = np.bincount(g, weights=main, minlength=G)
    home = np.zeros(G, bool)
    home[g] = q["req_home"]
    nb = q["cp"].shape[0]
    out = {k_: np.zeros((nb, E, K, NJ), np.int32) for k_ in ("corr", "nsrv", "cmain", "cnon")}
    cnt = {k_: np.zeros((nb, E, K), np.int32) for k_ in ("n_cc", "n_sc", "n_gain", "n_loss")}
    for b in range(nb):
        cc = (q["cp"][b] == y)
        sc = (q["sp"][b] == y)
        j0 = np.searchsorted(TAU32, q["ent"][b], side="left")   # client answers at tau_j  <=>  j >= j0
        idx = g * NJ + j0
        def cum(w):   # sum over requests with j0 <= j, for every j
            h = np.bincount(idx, weights=w, minlength=G * NJ).reshape(G, NJ)
            return np.cumsum(h, axis=1)
        d = cc.astype(np.int64) - sc.astype(np.int64)
        sc_g = np.bincount(g, weights=sc, minlength=G)
        out["corr"][b] = np.rint(sc_g[:, None] + cum(d)).reshape(E, K, NJ)
        out["nsrv"][b] = np.rint(n[:, None] - cum(np.ones_like(d))).reshape(E, K, NJ)
        out["cmain"][b] = np.rint(np.bincount(g, weights=sc & main, minlength=G)[:, None] + cum(d * main)).reshape(E, K, NJ)
        out["cnon"][b] = np.rint(np.bincount(g, weights=sc & ~main, minlength=G)[:, None] + cum(d * ~main)).reshape(E, K, NJ)
        cnt["n_cc"][b] = np.bincount(g, weights=cc, minlength=G).reshape(E, K)
        cnt["n_sc"][b] = sc_g.reshape(E, K)
        cnt["n_gain"][b] = np.bincount(g, weights=~cc & sc, minlength=G).reshape(E, K)
        cnt["n_loss"][b] = np.bincount(g, weights=cc & ~sc, minlength=G).reshape(E, K)
    np.savez_compressed(Path(outdir) / f"{name}_reduced.npz", tau=TAU, eval_rounds=q["eval_rounds"],
                        block_lambda=q["block_lambda"], installed_block=q["installed_block"],
                        n=n.reshape(E, K).astype(np.int32), n_main=np.rint(n_main).reshape(E, K).astype(np.int32),
                        home=home.reshape(E, K), source_sha256=np.array(sha256(src)),
                        source_bytes=np.int64(src.stat().st_size), **out, **cnt)
    print(f"  {name}: {len(y)} requests, {nb} blocks -> reduced/{name}_reduced.npz", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or [f"r8v2_{a}_s{s}" for a in ARMS for s in SEEDS]
    for nm in names:
        reduce_run(nm)
