"""Round 16 stage 1 diagnostic (written after the stage-1 tables, to examine the partial-participation rows; not part of
the decision): how well does the previous-round calibration choose a grid pair for the current device-round?

Per device-round with a calibration source (decision_rule.md 4.2), with the TRUE current Main share s (so that the share
estimate plays no role): a_cal = argmax_a s Acc_M,prev(a) + (1 - s) Acc_N,prev(a) over the 90 grid pairs (priority order).
Reported, as means over device-rounds in pp: current accuracy of a_cal minus DriftGate, the device-round oracle (O2) minus
a_cal, and the same for the pair that is best on the current requests' previous-round copies; split by whether the device
took part in training in the evaluation round (trace 'participated'; all devices take part in S1).
Output: tables/R16_D_stale_calibration.csv
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("r16", HERE / "scripts" / "r16_stage1.py")
r16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r16)
m13 = r16.m13
SETS = ["S1", "partial participation", "S1 replay"]


def one(st, s):
    runs, name = r16.runs_dir(st), r16.run_name(st, s)
    D = m13.load(Path(runs), name)
    pc, ps, Mrow, a, y, kind = D["pc"], D["ps"], D["Mrow"], D["a_req"], D["y"], D["kind"]
    E, K, g, er = D["E"], D["K"], D["g"], D["er"]
    G = E * K
    main = kind == 0
    z = np.load(r16.refs_path(runs, name))
    corr_dg = z["DriftGate"].astype(np.int64) == y
    gm = g * 2 + main
    cnt = np.bincount(gm, minlength=2 * G).reshape(G, 2).astype(float)
    cMN = np.empty((G, 2, r16.NG))
    for r in r16.R_ORDER:
        pe = ps if r is None else r16.corrected_chunked(ps, Mrow, a, r)
        for wi, w in enumerate(r16.W_GRID):
            ci = wi * len(r16.R_ORDER) + r16.R_ORDER.index(r)
            cMN[:, :, ci] = np.bincount(gm, weights=r16.mix_scalar(w, pc, pe) == y, minlength=2 * G).reshape(G, 2)
    dg = np.bincount(g, weights=corr_dg, minlength=G)
    n = cnt.sum(1)
    part = np.ones((E, K), bool)
    tr = Path(runs) / f"{name}_trace.npz"
    if tr.exists() and "participated" in np.load(tr).files and D["h"]["config"].get("replay_from") is None:
        part = np.load(tr)["participated"][er - 1].astype(bool)
    ok = ((cnt[:, 1] >= 8) & (cnt[:, 0] >= 8)).reshape(E, K)
    out = []
    last = np.full(K, -1)
    for e in range(E):
        for k in range(K):
            gi = e * K + k
            src = last[k]
            if src >= 0 and n[gi] > 0:
                gs = src * K + k
                s_true = cnt[gi, 1] / n[gi]
                J = s_true * cMN[gs, 1] / cnt[gs, 1] + (1 - s_true) * cMN[gs, 0] / cnt[gs, 0]
                a_cal = int(np.argmax(J >= J.max() - r16.TIE))
                acc_cur = (cMN[gi, 0] + cMN[gi, 1]) / n[gi]
                out.append((bool(part[e, k]), acc_cur[a_cal] - dg[gi] / n[gi], acc_cur.max() - acc_cur[a_cal], er[e]))
        last = np.where(ok[e], e, last)
    return out


def main():
    rows = []
    for st in SETS:
        per_seed = {}
        for s in r16.CFG["settings"][st]["seeds"]:
            per_seed[s] = one(st, s)
        for subset, f in (("all", lambda p: True), ("took part in the evaluation round", lambda p: p),
                          ("did not take part", lambda p: not p)):
            for wn, wf in (("all rounds", lambda r: True), ("round <= 30", lambda r: r <= 30), ("round > 30", lambda r: r > 30)):
                v1, v2, nn = [], [], []
                for s, o in per_seed.items():
                    sel = [x for x in o if f(x[0]) and wf(x[3])]
                    if sel:
                        v1.append(np.mean([x[1] for x in sel]) * 100)
                        v2.append(np.mean([x[2] for x in sel]) * 100)
                        nn.append(len(sel))
                if v1:
                    rows.append([st, subset, wn, int(np.sum(nn)), r16.fmt(v1, 3), r16.fmt(v2, 3)])
        print(st, "done", flush=True)
    r16.wcsv("R16_D_stale_calibration.csv", ["setting", "device-rounds", "rounds", "n device-rounds (all seeds)",
                                             "a_cal minus DriftGate pp mean (SD over seeds)", "O2 minus a_cal pp mean (SD over seeds)"], rows)


if __name__ == "__main__":
    main()
