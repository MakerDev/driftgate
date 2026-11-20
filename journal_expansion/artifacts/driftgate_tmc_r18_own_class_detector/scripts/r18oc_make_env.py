"""Round 18 (own-class detector): environment files for the new S1 development seeds (directive 2.4).

S1 seeds 5-7 were development seeds of earlier rounds (Rounds 7, 8, 10, 19, 20), so the development seeds of this round are
the smallest S1 seeds never used: 8, 9, 10 (no provenance record has partition_seed >= 8). The files are built with the
unchanged Round 6 generator (src/r6_env.build_synthetic) into runs/phaseT18oc_env/; existing environment files are not
touched. Before writing, the generator is checked on an existing file (S1 seed 5 must rebuild identically).

  python journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/scripts/r18oc_make_env.py
  python journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/scripts/r18oc_make_env.py s2 SCRATCH_DIR

`s2`: S2 (GeoLife) environment files for seeds 3 and 4 (stage 3 uses S2 seeds 0-4; Round 6 built 0-2). The unchanged
Round 6 generator (scripts/r6_trace_env.py) is run with its seed loop and output directory replaced: first seeds 0-2 into
SCRATCH_DIR, which must equal the existing runs/phaseT6_env files (and S2_selection.csv), then seeds 3-4 into
runs/phaseT18oc_env/. Only the request counts depend on the seed; the paths are the same for every seed.
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


def s2(scratch):
    src = (JR / "scripts" / "r6_trace_env.py").read_text()
    assert src.count("    for s in (0, 1, 2):") == 1
    src = src.replace("    for s in (0, 1, 2):", "    for s in SEEDS:")
    for seeds, out in (((0, 1, 2), Path(scratch)), ((3, 4), OUT)):
        ns = {"__name__": "r6_trace_env_r18oc", "__file__": str(JR / "scripts" / "r6_trace_env.py")}
        exec(compile(src, "r6_trace_env.py", "exec"), ns)
        ns["OUT"], ns["SEEDS"] = out, seeds
        ns["main"]()
        if out != OUT:
            for s in seeds:
                env, meta = r6_env.load_env(out / f"S2_seed{s}.npz")
                assert same(env, meta, OLD / f"S2_seed{s}.npz"), f"S2 seed {s} does not rebuild identically"
            assert (out / "S2_selection.csv").read_bytes() == (OLD / "S2_selection.csv").read_bytes()
            print("S2 seeds 0-2 and the selection rebuild identically")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "s2":
        s2(sys.argv[2])
    else:
        main()
