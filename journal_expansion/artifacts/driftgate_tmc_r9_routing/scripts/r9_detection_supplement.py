"""Round 9 supplement (added after the Round 9 results were seen; not used by the decision): how well the device-side
quantities separate Main from non-Main requests, per training arm and inference block, rounds 30-150 and all rounds.

  s not in M_k   (server-exit prediction outside the client's Main classes; used by A-scope)
  c not in M_k   (client-exit prediction outside the client's Main classes; used by B-scope)
  entropy e      (client exit; AUROC for non-Main = positive)
Rates are pooled over requests: share flagged among non-Main requests (detected) and among Main requests (false alarm).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_routing_analysis as A  # noqa: E402


def auroc(score, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos, score))


def main():
    rows = []
    for a in A.ARMS:
        for s in A.SEEDS:
            h = __import__("json").load(open(A.R8RUNS / f"r8v2_{a}_s{s}.json"))
            mains = A.rebuild_mains(h)
            q = dict(np.load(A.R8RUNS / f"r8v2_{a}_s{s}_requests.npz"))
            M = np.zeros((50, 10), bool)
            for k, v in mains.items():
                M[k, v] = True
            kk = q["req_client"].astype(np.int64)
            er = np.array(q["eval_rounds"])
            nonmain = q["req_kind"] != 0
            for w in A.WINDOWS:
                inw = np.ones(len(kk), bool) if w == "full" else (er >= A.LATE_MIN)[q["req_eval_index"]]
                for b, lam in enumerate(q["block_lambda"]):
                    sout = ~M[kk, q["sp"][b].astype(np.int64)]
                    cout = ~M[kk, q["cp"][b].astype(np.int64)]
                    rows.append((w, a, float(lam), s,
                                 sout[inw & nonmain].mean(), sout[inw & ~nonmain].mean(),
                                 cout[inw & nonmain].mean(), cout[inw & ~nonmain].mean(),
                                 auroc(q["ent"][b][inw], nonmain[inw])))
            print(f"  {a} s{s} done", flush=True)
    out = []
    keys = sorted({(r[0], r[1], r[2]) for r in rows}, key=lambda k: (A.WINDOWS.index(k[0]), A.ARMS.index(k[1]), k[2]))
    for w, a, lam in keys:
        rs = [r for r in rows if r[:3] == (w, a, lam)]
        out.append([w, a, f"{lam:g}"] + [f"{np.mean([r[i] for r in rs]) * (100 if i < 8 else 1):.{2 if i < 8 else 4}f}" for i in range(4, 9)])
    A.wcsv("R9_S1_detection.csv", ["window", "lambda_t_arm", "lambda_inf", "s_out_of_Mk_among_nonMain_pct",
                                   "s_out_of_Mk_among_Main_pct", "c_out_of_Mk_among_nonMain_pct",
                                   "c_out_of_Mk_among_Main_pct", "entropy_auroc_nonMain"], out)


if __name__ == "__main__":
    main()
