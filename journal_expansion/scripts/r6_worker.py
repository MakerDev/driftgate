#!/usr/bin/env python3
"""Round-6 file-queue worker (one per slot; started by scripts/supervisor_r6.sh).

Queue runs/queue_r6/queue.txt: "<K50|K200|K500> <command with {DEV}>" per line, served in
file order. A worker takes the FIRST line it may run: at most R6_MAX_K500 K=500 jobs and
R6_MAX_K200 K=200 jobs run at once on its GPU (GPU memory: K=500 ~8.7 GB, K=200 ~4 GB,
K=50 ~2 GB). Pops and the per-GPU counters are updated under one flock.
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
CAPS = {"K500": int(os.environ.get("R6_MAX_K500", "2")), "K200": int(os.environ.get("R6_MAX_K200", "3"))}
LOCK, LOGDIR, RUND = f"{QDIR}/queue.lock", f"{QDIR}/logs", f"{QDIR}/running_gpu{GPU}"
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU,
           JX_THREADS=os.environ.get("JX_THREADS", "2"))
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RUND, exist_ok=True)
os.chdir("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]


def log(msg):
    with open(f"{LOGDIR}/worker{WID}.log", "a") as f:
        f.write(f"[w{WID} cuda:0/GPU{GPU}] {msg} {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")


def pop():
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            path = f"{QDIR}/queue.txt"
            if not os.path.exists(path):
                return None, None, None
            lines = [l for l in open(path).read().splitlines() if l.strip()]
            running = [f.rsplit(".", 1)[-1] for f in os.listdir(RUND)]
            for i, line in enumerate(lines):
                cls, cmd = line.split(" ", 1)
                if cls in CAPS and running.count(cls) >= CAPS[cls]:
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
    with open(f"{LOGDIR}/{tag}.log", "a") as out:
        rc = subprocess.call(cmd.replace("{DEV}", "cuda:0"), shell=True, stdout=out,
                             stderr=subprocess.STDOUT, env=ENV)
    try:
        os.remove(f"{RUND}/{tag}.{cls}")
    except FileNotFoundError:
        pass
    log(f"END {tag} exit={rc}")
log("STOP file found, exiting")
