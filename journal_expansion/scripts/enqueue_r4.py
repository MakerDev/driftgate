"""Round-4 Part B — enqueue all 111 runs (conditions fixed BEFORE any result is seen).
No gate runs: every arm at its full seed set. Order B1,B2,B3 -> B4 -> B5,B6 (scheduling only).
All runs disjoint; same seeds/model_seed (100+seed) as the reference runs; eth=0.8 default.
Names match artifacts/driftgate_tmc_export_r4/scripts/r4_partA.py globs.
"""
from pathlib import Path
JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RV2 = f"python -u {JR}/scripts/run_v2.py"
Q = JR / "runs/queue/queue.txt"
DJ = "--disjoint_pools"
FULL = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap"
ENV = {  # env -> (flags, seeds)
    "A":        ("--schedule A --rounds 150", [0, 1, 2, 3, 4]),
    "mob":      ("--mobility --mobility_speed_level med --schedule abrupt --rounds 120", [0, 1, 2]),
    "svhn":     ("--dataset svhn --schedule gradual_sigmoid --rounds 150", [0, 1, 2, 3, 4]),
    "c10gsig":  ("--dataset cifar10 --schedule gradual_sigmoid --rounds 150", [0, 1, 2]),
    "c100gsig": ("--dataset cifar100 --schedule gradual_sigmoid --rounds 150", [0, 1, 2]),
    "c100sp":   ("--dataset cifar100 --schedule static --spatial equal_spread --rounds 150", [0, 1, 2]),
    "tiny":     ("--dataset tinyimagenet --schedule gradual_sigmoid --rounds 100", [0, 1, 2]),
}
jobs = []  # (out_dir, run_name, cmd)
def add(out, name, arm_flags, env, probe=64, seeds=None):
    flags, ss = ENV[env]
    for s in (seeds or ss):
        jobs.append((out, name.format(s=s),
                     f"{RV2} {arm_flags} {DJ} {flags} --probe_n {probe} --seed {s} --model_seed {100+s}"))

# ---- B1 view removal (25): absonly/relonly on A,mob,c100gsig ; nospatial on c100sp
B1 = "runs/phaseT4_B1_views"
ABSONLY = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --abs_only"
RELONLY = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm"      # no --abs_cap
NOSPAT  = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --abs_cap"           # no --spatial_norm
for env in ("A", "mob", "c100gsig"):
    add(B1, f"b1_absonly_{env}_s{{s}}", ABSONLY, env)
    add(B1, f"b1_relonly_{env}_s{{s}}", RELONLY, env)
add(B1, "b1_nospatial_c100sp_s{s}", NOSPAT, "c100sp")

# ---- B2 entnorm (8): entropy/lnC with the FULL controller
B2 = "runs/phaseT4_B2_entnorm"
ENTNORM = "--mode selfcal --signal ent_client_norm --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap"
for env in ("A", "mob"):
    add(B2, f"b2_entnorm_{env}_s{{s}}", ENTNORM, env)

# ---- B3 fixed grid (28), Lambda=0.5
B3 = "runs/phaseT4_B3_fixedgrid"
def fixed(lam): return f"--mode fixed --lambda_val {lam} --big_lambda_val 0.5"
for lam, tag in ((0.5, "fx50"), (0.6, "fx60")):
    add(B3, f"b3_mob_{tag}_s{{s}}", fixed(lam), "mob")
for lam, tag in ((0.4, "fx40"), (0.15, "fx15")):
    add(B3, f"b3_svhn_{tag}_s{{s}}", fixed(lam), "svhn")
for env in ("c10gsig", "c100gsig", "c100sp", "tiny"):
    add(B3, f"b3_{env}_fx15_s{{s}}", fixed(0.15), env)

# ---- B4 APFL-style lambda_k (16): eta in {lr=0.01, 10*lr=0.1}, Lambda=0.5, no probe/signal
B4 = "runs/phaseT4_B4_apfl"
for eta, tag in ((0.01, "eta001"), (0.1, "eta010")):
    for env in ("A", "mob"):
        add(B4, f"b4_apfl_{tag}_{env}_s{{s}}", f"--mode apfl --apfl_eta {eta} --big_lambda_val 0.5", env)

# ---- B5 probe count (16): full TV with probe 16 / 32
B5 = "runs/phaseT4_B5_probe"
for p in (16, 32):
    for env in ("A", "mob"):
        add(B5, f"b5_probe{p}_{env}_s{{s}}", FULL, env, probe=p)

# ---- B6 constant sensitivity (18): guard {0.25,1.0} on A(0-2); lam_max {0.60,0.80} on A(0-2),mob(0-2)
B6 = "runs/phaseT4_B6_const"
for g, tag in ((0.25, "guard025"), (1.0, "guard100")):
    add(B6, f"b6_{tag}_A_s{{s}}",
        f"--mode selfcal --signal tv_dist --burn_in 10 --z_guard {g} --spatial_norm --abs_cap", "A", seeds=[0, 1, 2])
for lm, tag in ((0.60, "lmax060"), (0.80, "lmax080")):
    for env in ("A", "mob"):
        add(B6, f"b6_{tag}_{env}_s{{s}}", f"{FULL} --lam_max {lm}", env, seeds=[0, 1, 2])

assert len(jobs) == 111, len(jobs)
existing = set(Q.read_text().splitlines()) if Q.exists() else set()
n = 0
with open(Q, "a") as f:
    for out, name, cmd in jobs:
        (JR / out).mkdir(parents=True, exist_ok=True)
        line = f"{cmd} --run_name {name} --output_dir {JR/out} --device {{DEV}}"
        if (JR / out / f"{name}.json").exists() or line in existing:
            continue
        f.write(line + "\n"); n += 1
print(f"enqueued {n} of {len(jobs)} R4 jobs")
