"""Round 18 (own-class detector): environment files for the new S1 development seeds (directive 2.4).

S1 seeds 5-7 were development seeds of earlier rounds (Rounds 7, 8, 10, 19, 20), so the development seeds of this round are
the smallest S1 seeds never used: 8, 9, 10 (no provenance record has partition_seed >= 8). The files are built with the
unchanged Round 6 generator (src/r6_env.build_synthetic) into runs/phaseT18oc_env/; existing environment files are not
touched. Before writing, the generator is checked on an existing file (S1 seed 5 must rebuild identically).

  python journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/scripts/r18oc_make_env.py
"""
import json
import sys
from pathlib import Path

import numpy as np

JR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

OLD = JR / "runs" / "phaseT6_env"
OUT = JR / "runs" / "phaseT18oc_env"
DEV_SEEDS = [8, 9, 10]


def same(env, meta, f):
    old, old_meta = r6_env.load_env(f)
    return old_meta == json.loads(json.dumps(meta)) and set(old) == set(env) and all(np.array_equal(old[k], env[k]) for k in env)


def main():
    env, meta, _, _ = r6_env.build_synthetic("S1", 5)
    assert same(env, meta, OLD / "S1_seed5.npz"), "generator does not rebuild S1 seed 5"
    print("S1 seed 5 rebuilds identically")
    OUT.mkdir(parents=True, exist_ok=True)
    for s in DEV_SEEDS:
        f = OUT / f"S1_seed{s}.npz"
        env, meta, plans, trips = r6_env.build_synthetic("S1", s)
        if f.exists():
            assert same(env, meta, f), f"{f} differs from a fresh build"
            print(f"{f.name}: exists, identical")
            continue
        r6_env.save_env(f, env, meta)
        r6_env.write_plan_csv(OUT / f"plan_S1_seed{s}.csv", plans, trips)
        print(f"{f.name}: K={meta['K']} L={meta['L']} at_home={env['at_home'].mean():.3f} "
              f"two_cells={(env['member'][:, :, 1] >= 0).mean():.3f}")


if __name__ == "__main__":
    main()
