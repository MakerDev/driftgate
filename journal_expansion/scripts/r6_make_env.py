"""Round 6: write environment files (R6 directive §3.8).

  python journal_expansion/scripts/r6_make_env.py                 # all synthetic scenarios
  python journal_expansion/scripts/r6_make_env.py --scenarios S1 --seeds 0

Writes runs/phaseT6_env/<scenario>_seed<s>.npz and plan_<scenario>_seed<s>.csv.
Existing files are never overwritten (an identical rebuild is verified instead).
S2 (trace) files are written by r6_trace_env.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

JR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

OUT = JR / "runs" / "phaseT6_env"
SEEDS = {"S1": [0, 1, 2, 3, 4]}   # S1 main arms use seeds 0-4; everything else 0-2 (§2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(r6_env.SCENARIOS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in args.scenarios:
        for s in (args.seeds if args.seeds is not None else SEEDS.get(sc, [0, 1, 2])):
            env, meta, plans, trips = r6_env.build_synthetic(sc, s)
            f = OUT / f"{sc}_seed{s}.npz"
            if f.exists():
                old, old_meta = r6_env.load_env(f)
                same = old_meta == meta and all(np.array_equal(old[k], env[k]) for k in env)
                print(f"{f.name}: exists, identical rebuild = {same}")
                if not same:
                    raise SystemExit(f"{f} differs from a fresh build; refusing to overwrite")
                continue
            r6_env.save_env(f, env, meta)
            r6_env.write_plan_csv(OUT / f"plan_{sc}_seed{s}.csv", plans, trips)
            print(f"{f.name}: K={meta['K']} L={meta['L']} at_home={env['at_home'].mean():.3f} "
                  f"two_cells={(env['member'][:, :, 1] >= 0).mean():.3f} "
                  f"n0={(env['n_req'] == 0).mean():.4f} avail={env['avail'].mean():.3f}")


if __name__ == "__main__":
    main()
