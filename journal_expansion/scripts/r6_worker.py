#!/usr/bin/env python3
"""Round-6 file-queue worker (one per slot; started by scripts/supervisor_r6.sh).

Queue runs/queue_r6/queue.txt: "<K50|K200|K500|R0|R0H> <command with {DEV}>" per line, served in
file order. A worker takes the FIRST line it may run: at most R6_MAX_K500 K=500 jobs (3),
R6_MAX_K200 K=200 jobs (3) and R6_MAX_R0H heavy R0 jobs (1) run at once on its GPU
(GPU memory with expandable segments: K=500 ~7.9 GB, Tiny-ImageNet ~4 GB, K=50 ~2 GB), and at most
R6_MAX_JOBS_PER_GPU (4) jobs in total, counted from live processes (see running_on_gpu). Pops and the per-GPU counters are updated under one flock.
Every job runs with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<R6_GPU>, so the job's
cuda:0 is physical GPU R6_GPU. Stops when runs/queue_r6/STOP exists.
Usage: R6_GPU=<index> r6_worker.py <worker_id>
"""
import datetime
import fcntl
import os
import re
import subprocess
import sys
import time

WID = sys.argv[1]
QDIR = "/home/honeynaps/data/driftgate/journal_expansion/runs/queue_r6"  # [SERVER-PATH:REPO_ROOT]
GPU = os.environ.get("R6_GPU", "0")  # physical GPU index (nvidia-smi, PCI order)  # [SERVER-GPU]
CAPS = {"K500": int(os.environ.get("R6_MAX_K500", "2")), "K200": int(os.environ.get("R6_MAX_K200", "3")),
        "R0H": int(os.environ.get("R6_MAX_R0H", "1"))}   # R0H = Round-5 Tiny-ImageNet / ResNet-18 re-runs (R0)
# 2026-10-01: three K=500 runs on one 24 GB GPU ran out of memory (~7.9 GB each) -> at most 2 per GPU.
MAX_JOBS = int(os.environ.get("R6_MAX_JOBS_PER_GPU", "4"))   # all jobs on this GPU, whoever started them
# 2026-10-02: two K=200 runs ran out of memory next to two K=500 runs -> admit a job only if the GPU memory
# of the live jobs plus the new one stays within R6_GPU_BUDGET_GB (nvidia-smi per-process memory, measured).
COST_GB = {"K500": 9.0, "K200": 4.4, "K50": 2.1, "R0": 2.2, "R0H": 4.5}
BUDGET_GB = float(os.environ.get("R6_GPU_BUDGET_GB", "23.0"))
LOCK, LOGDIR, RUND = f"{QDIR}/queue.lock", f"{QDIR}/logs", f"{QDIR}/running_gpu{GPU}"
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU,
           JX_THREADS=os.environ.get("JX_THREADS", "2"),
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")   # K=500 ~6 GB instead of ~8.7 GB (allocator only)
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RUND, exist_ok=True)
os.chdir("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]


def log(msg):
    with open(f"{LOGDIR}/worker{WID}.log", "a") as f:
        f.write(f"[w{WID} cuda:0/GPU{GPU}] {msg} {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")


def live_tags():
    """run names of processes alive on this machine (python argv or the worker's /bin/sh -c string)."""
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
        except OSError:
            continue
        for m in re.finditer(r"--run_name[\x00 ](\S+?)(?=[\x00 ]|$)", cmd):
            out.add(m.group(1))
    return out


def running_on_gpu():
    """markers of this GPU whose job is alive (a marker younger than 120 s counts as alive: its job may
    not have started yet). Stale markers (job gone) are removed. Jobs started by a previous worker
    generation keep their markers and are counted while they run."""
    live, out = live_tags(), []
    for f in os.listdir(RUND):
        tag, cls = f.rsplit(".", 1)
        young = time.time() - os.path.getmtime(f"{RUND}/{f}") < 120
        if tag in live or young:
            out.append(cls)
        else:
            os.remove(f"{RUND}/{f}")
    return out


def pop():
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            path = f"{QDIR}/queue.txt"
            if not os.path.exists(path):
                return None, None, None
            lines = [l for l in open(path).read().splitlines() if l.strip()]
            running = running_on_gpu()
            if len(running) >= MAX_JOBS:
                return None, None, None
            used = sum(COST_GB.get(c, 2.1) for c in running)
            for i, line in enumerate(lines):
                cls, cmd = line.split(" ", 1)
                if cls in CAPS and running.count(cls) >= CAPS[cls]:
                    continue
                if used + COST_GB.get(cls, 2.1) > BUDGET_GB:
                    continue
                with open(path, "w") as f:
                    f.write("".join(l + "\n" for l in lines[:i] + lines[i + 1:]))
                m = re.search(r"--run_name (\S+)", cmd)
                tag = m.group(1) if m else f"job_{int(time.time())}"
                open(f"{RUND}/{tag}.{cls}", "w").close()
                return cls, cmd, tag
            return None, None, None
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


while not os.path.exists(f"{QDIR}/STOP"):
    cls, cmd, tag = pop()
    if cmd is None:
        time.sleep(60)
        continue
    log(f"START {tag} ({cls})")
    t_start = time.time()
    with open(f"{LOGDIR}/{tag}.log", "a") as out:
        rc = subprocess.call(cmd.replace("{DEV}", "cuda:0"), shell=True, stdout=out,
                             stderr=subprocess.STDOUT, env=ENV)
    try:
        os.remove(f"{RUND}/{tag}.{cls}")
    except FileNotFoundError:
        pass
    log(f"END {tag} exit={rc}")
    if rc != 0 and time.time() - t_start < 300:
        # 2026-10-02: one worker failed five K=200 jobs in two minutes (OOM at start-up) -> back off
        log("fast failure, pausing 10 min before the next job")
        time.sleep(600)
log("STOP file found, exiting")
