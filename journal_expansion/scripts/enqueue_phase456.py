"""Enqueue the pre-registered pilot matrices (priority order) into the file
queue consumed by queue_worker.sh. Skips jobs whose result JSON already exists.

Priority: v3b downstream (6) -> Gate C fairness (16) -> Phase 4 causal
baselines (22) -> Phase 6 network stress (21).
"""
import sys
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
QDIR = JOURNAL_ROOT / "runs/queue"
SEEDS = [(0, 100), (1, 101), (2, 102)]

RV2 = f"python -u {JOURNAL_ROOT}/scripts/run_v2.py"


def jobs_v3b():
    out = JOURNAL_ROOT / "runs/downstream_pilot"
    for sched in ("A", "abrupt"):
        for ps, ms in SEEDS:
            name = f"sc_delta_b10_{sched}_s{ps}"
            yield out, name, (f"{RV2} --mode selfcal --signal delta_hard --burn_in 10 "
                              f"--schedule {sched} --seed {ps} --model_seed {ms} "
                              f"--rounds 100 --probe_n 64 --run_name {name} "
                              f"--output_dir {out} --device {{DEV}}")


def jobs_gatec():
    out = JOURNAL_ROOT / "runs/gatec_fairness"
    base = (f"{RV2} --schedule static --spatial equal_spread --rounds 100 "
            f"--probe_n 64 --output_dir {out} --device {{DEV}} ")
    for ps, ms in SEEDS:
        s = f"--seed {ps} --model_seed {ms} "
        yield out, f"sc_delta_b10_sp_s{ps}", base + s + \
            f"--run_name sc_delta_b10_sp_s{ps} --mode selfcal --signal delta_hard --burn_in 10"
        yield out, f"fair_deploy_sp_s{ps}", base + s + \
            f"--run_name fair_deploy_sp_s{ps} --mode selfcal --signal delta_hard --burn_in 10 --fairness deployable"
        yield out, f"fair_oracle_sp_s{ps}", base + s + \
            f"--run_name fair_oracle_sp_s{ps} --mode selfcal --signal delta_hard --burn_in 10 --fairness oracle"
        yield out, f"fixed02_sp_s{ps}", base + s + \
            f"--run_name fixed02_sp_s{ps} --mode fixed --lambda_val 0.2"
        yield out, f"fixed04_sp_s{ps}", base + s + \
            f"--run_name fixed04_sp_s{ps} --mode fixed --lambda_val 0.4"
    yield out, "fair_random_sp_s0", base + "--seed 0 --model_seed 100 " + \
        "--run_name fair_random_sp_s0 --mode selfcal --signal delta_hard --burn_in 10 --fairness random"


def jobs_phase4():
    out = JOURNAL_ROOT / "runs/phase4_baselines"
    for sched in ("A", "abrupt"):
        for mode in ("ucb", "exp3", "oracle_greedy"):
            for ps, ms in SEEDS:
                name = f"{mode}_{sched}_s{ps}"
                yield out, name, (f"{RV2} --mode {mode} --schedule {sched} "
                                  f"--seed {ps} --model_seed {ms} --rounds 100 "
                                  f"--probe_n 64 --run_name {name} "
                                  f"--output_dir {out} --device {{DEV}}")
        for mode in ("proxy_grid",):
            name = f"{mode}_{sched}_s0"
            yield out, name, (f"{RV2} --mode {mode} --schedule {sched} --seed 0 "
                              f"--model_seed 100 --rounds 100 --probe_n 64 "
                              f"--run_name {name} --output_dir {out} --device {{DEV}}")
        name = f"sc_updatenorm_{sched}_s0"
        yield out, name, (f"{RV2} --mode selfcal --signal update_norm --burn_in 10 "
                          f"--schedule {sched} --seed 0 --model_seed 100 --rounds 100 "
                          f"--probe_n 64 --run_name {name} --output_dir {out} --device {{DEV}}")


def jobs_phase6():
    out = JOURNAL_ROOT / "runs/phase6_network"
    base = (f"{RV2} --mode selfcal --signal delta_hard --burn_in 10 "
            f"--schedule abrupt --rounds 100 --probe_n 64 "
            f"--output_dir {out} --device {{DEV}} ")
    for d in (2, 5, 10):
        for ps, ms in SEEDS:
            name = f"delay{d}_s{ps}"
            yield out, name, base + f"--seed {ps} --model_seed {ms} --signal_delay {d} --run_name {name}"
    for loss in (0.1, 0.2):
        for ps, ms in SEEDS:
            name = f"loss{int(loss*100)}_s{ps}"
            yield out, name, base + f"--seed {ps} --model_seed {ms} --signal_loss {loss} --run_name {name}"
    for topo in ("ring", "star", "dynamic"):
        name = f"topo_{topo}_s0"
        yield out, name, base + f"--seed 0 --model_seed 100 --topology {topo} --run_name {name}"
    for ps, ms in SEEDS:
        name = f"part50_s{ps}"
        yield out, name, base + f"--seed {ps} --model_seed {ms} --participation 0.5 --run_name {name}"


def main():
    QDIR.mkdir(parents=True, exist_ok=True)
    qfile = QDIR / "queue.txt"
    existing = set(qfile.read_text().strip().split("\n")) if qfile.exists() else set()
    added = 0
    with open(qfile, "a") as f:
        for gen in (jobs_v3b, jobs_gatec, jobs_phase4, jobs_phase6):
            for out, name, cmd in gen():
                out.mkdir(parents=True, exist_ok=True)
                if (out / f"{name}.json").exists() or cmd in existing:
                    continue
                f.write(cmd + "\n")
                added += 1
    print(f"enqueued {added} jobs -> {qfile}")


if __name__ == "__main__":
    main()
