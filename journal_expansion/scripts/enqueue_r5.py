"""Round 5 — enqueue every run (conditions fixed BEFORE any result is seen; no gate runs).
GPU 0 only: jobs go to runs/queue_r5/{heavy,light}.txt, served by scripts/r5_worker.py.

  B3 re-run (10)  fixed λ=0.15, Λ=0.5 — the in-flight copies were SIGSTOPped for 3 days and had
                  to be killed (5 of them sat on GPU 1, which is now off-limits). CIFAR-100 spatial
                  is NOT re-run (excluded by the Round-5 directive).
  B4 (16)         APFL-style λ_k, η ∈ {0.01 (=lr), 0.1 (=10·lr)}, Λ=0.5, no probe/signal.
                  Same command lines as the Round-4 queue (moved here).
  E1 (11)         relonly on CIFAR-10 gradual (0–2), Tiny-ImageNet (0–2), SVHN temporal (0–4).
  E2 (9)          ResNet-18 middle split (existing Phase-F definition: --model_family resnet18
                  --split_point middle --num_clients 16), CIFAR-10 Schedule A 150 rounds,
                  arms relonly / fixed 0.2 / fixed 0.4 (Λ=0.5), seeds 0–2.
  E3 (18)         relonly constant sensitivity: guard {0.25, 1.0} on A(0–2);
                  λ_max {0.60, 0.80} on A(0–2) and mobility(0–2).
relonly = the exact Round-4 B1 flag set (no --abs_cap):
  --mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm
All runs: --disjoint_pools, probe 64, model_seed = 100 + seed, default eval rounds / eth=0.8.
Order inside each queue: longest jobs first.
"""
from pathlib import Path

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RV2 = f"python -u {JR}/scripts/run_v2.py"
QD = JR / "runs/queue_r5"; QD.mkdir(parents=True, exist_ok=True); (QD / "logs").mkdir(exist_ok=True)
DJ = "--disjoint_pools"
RELONLY = "--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm"
def FIXED(lam): return f"--mode fixed --lambda_val {lam} --big_lambda_val 0.5"
ENV = {
    "A":        "--schedule A --rounds 150",
    "mob":      "--mobility --mobility_speed_level med --schedule abrupt --rounds 120",
    "svhn":     "--dataset svhn --schedule gradual_sigmoid --rounds 150",
    "c10gsig":  "--dataset cifar10 --schedule gradual_sigmoid --rounds 150",
    "c100gsig": "--dataset cifar100 --schedule gradual_sigmoid --rounds 150",
    "tiny":     "--dataset tinyimagenet --schedule gradual_sigmoid --rounds 100",
    "resA":     "--model_family resnet18 --split_point middle --num_clients 16 --schedule A --rounds 150",
}
heavy, light = [], []   # (priority, out_dir, run_name, cmd)
def job(q, prio, out, name, arm, env, seed):
    cmd = f"{RV2} {arm} {DJ} {ENV[env]} --probe_n 64 --seed {seed} --model_seed {100 + seed}"
    q.append((prio, out, name, cmd))

B3, B4 = "runs/phaseT4_B3_fixedgrid", "runs/phaseT4_B4_apfl"
E1, E2, E3 = "runs/phaseT5_E1_relonly_transfer", "runs/phaseT5_E2_resnet", "runs/phaseT5_E3_const"

# ---- heavy (Tiny ~20 h, ResNet-18 ~9–19 h) : Tiny first
for s in (0, 1, 2):
    job(heavy, 0, B3, f"b3_tiny_fx15_s{s}", FIXED(0.15), "tiny", s)
    job(heavy, 0, E1, f"e1_relonly_tiny_s{s}", RELONLY, "tiny", s)
for s in (0, 1, 2):
    job(heavy, 1, E2, f"e2_res_relonly_A_s{s}", RELONLY, "resA", s)
    job(heavy, 1, E2, f"e2_res_fx20_A_s{s}", FIXED(0.2), "resA", s)
    job(heavy, 1, E2, f"e2_res_fx40_A_s{s}", FIXED(0.4), "resA", s)

# ---- light: SVHN (~7.5 h) > CIFAR-10 gradual (~6 h) > Schedule A (~4.5 h) > mobility (~3.5 h) > CIFAR-100 gradual
for s in (2, 3, 4):
    job(light, 0, B3, f"b3_svhn_fx15_s{s}", FIXED(0.15), "svhn", s)
for s in (0, 1, 2, 3, 4):
    job(light, 0, E1, f"e1_relonly_svhn_s{s}", RELONLY, "svhn", s)
for s in (0, 1):
    job(light, 1, B3, f"b3_c10gsig_fx15_s{s}", FIXED(0.15), "c10gsig", s)
for s in (0, 1, 2):
    job(light, 1, E1, f"e1_relonly_c10gsig_s{s}", RELONLY, "c10gsig", s)
for tag, eta in (("eta001", 0.01), ("eta010", 0.1)):
    for s in (0, 1, 2, 3, 4):
        job(light, 2, B4, f"b4_apfl_{tag}_A_s{s}", f"--mode apfl --apfl_eta {eta} --big_lambda_val 0.5", "A", s)
for g, tag in ((0.25, "guard025"), (1.0, "guard100")):
    for s in (0, 1, 2):
        job(light, 2, E3, f"e3_{tag}_A_s{s}",
            f"--mode selfcal --signal tv_dist --burn_in 10 --z_guard {g} --spatial_norm", "A", s)
for lm, tag in ((0.60, "lmax060"), (0.80, "lmax080")):
    for s in (0, 1, 2):
        job(light, 2, E3, f"e3_{tag}_A_s{s}", f"{RELONLY} --lam_max {lm}", "A", s)
for tag, eta in (("eta001", 0.01), ("eta010", 0.1)):
    for s in (0, 1, 2):
        job(light, 3, B4, f"b4_apfl_{tag}_mob_s{s}", f"--mode apfl --apfl_eta {eta} --big_lambda_val 0.5", "mob", s)
for lm, tag in ((0.60, "lmax060"), (0.80, "lmax080")):
    for s in (0, 1, 2):
        job(light, 3, E3, f"e3_{tag}_mob_s{s}", f"{RELONLY} --lam_max {lm}", "mob", s)
for s in (1, 2):
    job(light, 4, B3, f"b3_c100gsig_fx15_s{s}", FIXED(0.15), "c100gsig", s)

assert len(heavy) == 15 and len(light) == 49, (len(heavy), len(light))
import sys
if "--snapshot-only" in sys.argv:      # write all 64 command lines for re-queueing; queues untouched
    for name, q in (("heavy", heavy), ("light", light)):
        q.sort(key=lambda t: t[0])
        with open(QD / f"enqueued_{name}_snapshot.txt", "w") as f:
            for _, out, rn, cmd in q:
                f.write(f"{cmd} --run_name {rn} --output_dir {JR / out} --device {{DEV}}\n")
    print("snapshot written"); sys.exit(0)
counts = {}
for name, q in (("heavy", heavy), ("light", light)):
    q.sort(key=lambda t: t[0])           # stable: keeps the listed order inside a priority
    existing = (QD / f"{name}.txt").read_text().splitlines() if (QD / f"{name}.txt").exists() else []
    n = 0
    with open(QD / f"{name}.txt", "a") as f:
        for _, out, rn, cmd in q:
            (JR / out).mkdir(parents=True, exist_ok=True)
            line = f"{cmd} --run_name {rn} --output_dir {JR / out} --device {{DEV}}"
            if (JR / out / f"{rn}.json").exists() or line in existing:
                continue
            f.write(line + "\n"); n += 1
    counts[name] = n
print(f"enqueued heavy={counts['heavy']} light={counts['light']} (total {sum(counts.values())} of 64)")
