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

`stepwise`, `randmob`: the v32 settings "stepwise change" and "random mobility" (v32 Section 6, Table settings), which the
v32 code ran in the older runner (src/runner.py: --schedule A; --mobility --mobility_speed_level med --schedule abrupt),
written as Round 6 environment files so that the Round 18 runner, detectors and analysis are unchanged (decisions.md D14).
Both use the S1 partition (district 0, client k in static cell k // 10), the S1 request counts of the seed (stream 3),
full participation and no signal loss. One day = 150 rounds; runs repeat it (--days 2).
  stepwise: membership fixed to the older runner's static topology (make_es_topology(50, 5, 50): clients 10g+5..10g+9 of
            cells 0-3 are also in cell g+1), rho of every device 0, 0.4, 0.8, 0.4, 0 in five 30-round phases
            (src/schedules.get_rho("A")); cell positions of the S1 layout.
  randmob:  the older runner's Gauss-Markov model (network/mobility.GaussMarkovMobility: 1 km map, five edges on a 3 x 3
            grid, alpha 0.9, 12 m/s, 10 s per round, seed 42 as in v32, the same paths for every seed); rounds 1-4 use the
            static topology, then every 5th round each device joins its two nearest edges (as in src/runner.py); rho of
            every device 0 for rounds 1-75 and 0.8 for rounds 76-150 (get_rho("abrupt")).
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


def _common(seed, member, pos, cell_xy, cell_is_hub, rho, scenario, extra_meta):
    K, T = member.shape[:2]
    home_cell = np.arange(K) // 10
    mu, n_req = r6_env.request_counts(seed, "base", K, "normal")
    avail, lost_c2e, lost_dbar, lost_nbr = r6_env.participation_and_loss(seed, "base", K, len(cell_xy), 1.0, 0.0)
    at_home = (member == home_cell[:, None, None]).any(axis=2)
    env = dict(pos=pos.astype(np.float32), member=member.astype(np.int16), at_home=at_home, rho=rho.astype(np.float32),
               n_req=n_req, avail=avail, mu=mu.astype(np.float32), lost_c2e=lost_c2e, lost_dbar=lost_dbar,
               lost_nbr=lost_nbr, home_cell=home_cell, home_pos=pos[:, 0].astype(np.float32),
               plan_type=np.array(["fixed" if scenario == "stepwise" else "gauss_markov"] * K),
               cell_xy=cell_xy.astype(np.float32), cell_local=np.arange(5), cell_district=np.zeros(5, int),
               cell_is_hub=cell_is_hub, neighbors=r6_env.neighbor_matrix(cell_xy),
               client_district=np.zeros(K, int), client_local=np.arange(K), client_group=home_cell,
               round_start_min=np.array([r6_env.round_start_min(r) for r in range(1, T + 1)]))
    meta = dict(scenario=scenario, env_seed=int(seed), K=int(K), L=5, T=int(T), grid=[1, 1], layout="base",
                mu_range=r6_env.MU_RANGE["normal"], avail=1.0, loss=0.0,
                partition="district 0: nd1_partition on the static 50-client topology, seed = env_seed", **extra_meta)
    return env, meta


def build_stepwise(seed):
    from data.partition import make_es_topology
    from src.schedules import get_rho
    T = r6_env.T_ROUNDS
    c2es, _ = make_es_topology(50, 5, 50, seed=0)
    member = np.full((50, T, 2), -1, np.int16)
    for k, es in c2es.items():
        for j, z in enumerate(es):
            member[k, :, j] = z
    cell_xy, cell_local, _, _ = r6_env.district_layout((1, 1))
    pos = np.repeat(cell_xy[np.arange(50) // 10][:, None, :], T, axis=1)
    rho = np.tile(np.array([get_rho("A", r, T) for r in range(1, T + 1)], float), (50, 1))
    return _common(seed, member, pos, cell_xy, cell_local == r6_env.HUB_LOCAL, rho, "stepwise",
                   dict(schedule="A", rho_phases=[0.0, 0.4, 0.8, 0.4, 0.0], topology="make_es_topology(50, 5, 50, seed=0)"))


def build_randmob(seed):
    from data.partition import make_es_topology
    from network.mobility import GaussMarkovMobility
    from src.schedules import get_rho
    T = r6_env.T_ROUNDS
    c2es, _ = make_es_topology(50, 5, 50, seed=0)
    mob = GaussMarkovMobility(num_clients=50, num_es=5, map_size=1000.0, alpha=0.9, speed=12.0, dt=10.0, seed=42)
    member = np.full((50, T, 2), -1, np.int16)
    pos = np.zeros((50, T, 2))
    cur = {k: list(v) for k, v in c2es.items()}
    for r in range(1, T + 1):
        mob.step()
        if r % 5 == 0:
            cur = mob.get_topk_es(k=2)
        for k in range(50):
            for j, z in enumerate(cur[k]):
                member[k, r - 1, j] = z
        pos[:, r - 1] = mob.client_pos / 1000.0
    rho = np.tile(np.array([get_rho("abrupt", r, T) for r in range(1, T + 1)], float), (50, 1))
    return _common(seed, member, pos, mob.es_pos / 1000.0, np.zeros(5, bool), rho, "randmob",
                   dict(mobility="GaussMarkovMobility(map 1000 m, alpha 0.9, 12 m/s, dt 10 s, seed 42), top-2 every 5 rounds",
                        schedule="abrupt", rho_phases=[0.0, 0.8]))


def write_new(scenario, seeds=(0, 1, 2)):
    OUT.mkdir(parents=True, exist_ok=True)
    build = {"stepwise": build_stepwise, "randmob": build_randmob}[scenario]
    for s in seeds:
        f = OUT / f"{scenario}_seed{s}.npz"
        env, meta = build(s)
        if f.exists():
            assert same(env, meta, f), f"{f} differs from a fresh build"
            print(f"{f.name}: exists, identical")
            continue
        r6_env.save_env(f, env, meta)
        m = env["member"]
        print(f"{f.name}: K={meta['K']} at_home={env['at_home'].mean():.3f} two_cells={(m[:, :, 1] >= 0).mean():.3f} "
              f"rho_mean={env['rho'].mean():.3f} cell_changes={int((m[:, 1:] != m[:, :-1]).any(2).sum())}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "s2":
        s2(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] in ("stepwise", "randmob"):
        write_new(sys.argv[1])
    else:
        main()
