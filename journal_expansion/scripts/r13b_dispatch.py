"""Round 13b dispatcher: the new runs of directive part B (S1, fixed lambda 0.4, seeds 0-2) on GPUs 0-3.

Each job copies the command of the Round 12 run r12_s1_fixed040_s<seed> (provenance; already carries the Round 12
record flags --record_device_signals --record_eval_probs --record_train_label_hist) and changes only: run name, output
directory runs/phaseT13b_arch, and the item of its group:
  structure 1-4: --model_family resnet20 / vgg11 with --split_point shallow / middle
  condition 1:   --dataset cifar100 (default model)
Condition 2 (stronger class skew) is not run: the partition parameter that sets the Main classes per client gives
max(1, int(10 * 0.2)) = 2 classes; one fewer would be 1, below the floor of 2 in the directive, so the value stays 2
(the S1 default), and the partition has no Dirichlet coefficient (directive 4.2, last case).
Order: structure 1-4, condition 1. Scheduling as in r12_dispatch.py (free GPU memory minus jobs launched in the last 4
minutes, margin 2.5 GB, at most MAX_PER_GPU of our jobs per GPU, retry after 10 minutes, at most 3 attempts; STOP file
-> stop launching). A diverged run (non-finite loss) is not retried here; it is rerun by hand once with
--learning_rate 0.005 (directive 4.1).

  python journal_expansion/scripts/r13b_dispatch.py --dry-run
  setsid nohup python journal_expansion/scripts/r13b_dispatch.py < /dev/null > runs/phaseT13b_arch/dispatch.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
JR = ROOT / "journal_expansion"
OUT = JR / "runs/phaseT13b_arch"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU]
MAX_PER_GPU = 5
MARGIN_GB = 2.5
REC = ["--record_device_signals", "--record_eval_probs", "--record_train_label_hist"]

# (group, label, extra flags, cost GB)
PLAN = [
    ("structure 1", "ResNet-20, shallow split", ["--model_family", "resnet20", "--split_point", "shallow"], 3.0),
    ("structure 2", "ResNet-20, middle split", ["--model_family", "resnet20", "--split_point", "middle"], 3.0),
    ("structure 3", "VGG-11, shallow split", ["--model_family", "vgg11", "--split_point", "shallow"], 5.0),
    ("structure 4", "VGG-11, middle split", ["--model_family", "vgg11", "--split_point", "middle"], 5.0),
    ("condition 1", "CIFAR-100", ["--dataset", "cifar100"], 3.0),
]
TAG = {"structure 1": "res20_shallow", "structure 2": "res20_middle", "structure 3": "vgg11_shallow",
       "structure 4": "vgg11_middle", "condition 1": "c100"}
SEEDS = [0, 1, 2]


def base_command(seed):
    h = json.load(open(JR / "runs" / "phaseT12_fusion" / f"r12_s1_fixed040_s{seed}.json"))
    prov = json.load(open(JR / "provenance" / f"{h['config']['run_id']}.json"))
    cmd = prov["command"]
    return (cmd if isinstance(cmd, list) else cmd.split()), h["config"]["run_id"]


def jobs():
    out = []
    for group, label, extra, cost in PLAN:
        for s in SEEDS:
            toks, base_id = base_command(s)
            assert all(r in toks for r in REC) and toks[toks.index("--lambda_val") + 1] == "0.4", toks
            assert not any(t in toks for t in ("--model_family", "--split_point", "--dataset", "--learning_rate"))
            name = f"r13b_{TAG[group]}_s{s}"
            toks[toks.index("--run_name") + 1] = name
            toks[toks.index("--output_dir") + 1] = str(OUT)
            toks[toks.index("--device") + 1] = "cuda:0"
            cmd = ["python", "-u"] + toks + extra
            out.append(dict(name=name, base=f"r12_s1_fixed040_s{s}", base_run_id=base_id, group=group, label=label,
                            seed=s, cost=cost, cmd=" ".join(cmd)))
    assert len(out) == 15 and len({j["name"] for j in out}) == 15
    return out


def alive_names():
    names = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="ignore") for a in args]
        if "--run_name" in args and any(a.endswith(("run_r6.py", "run_v2.py")) for a in args):
            names.add(args[args.index("--run_name") + 1])
    return names


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {int(a): float(b) / 1024 for a, b in (l.split(",") for l in out)}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "dispatch.log", "a") as f:
        f.write(line + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    J = jobs()
    with open(OUT / "jobs.json", "w") as f:
        json.dump(J, f, indent=1)
    if "--dry-run" in sys.argv:
        for j in J:
            print(j["name"], "|", j["cmd"])
        return
    state_p = OUT / "dispatch_state.json"
    state = json.load(open(state_p)) if state_p.exists() else {}
    log(f"dispatcher start (pid {os.getpid()}), {len(J)} jobs")
    while True:
        alive = alive_names()
        done = {j["name"] for j in J if (OUT / f"{j['name']}.json").exists()}
        now = time.time()
        pending = []
        for j in J:
            n = j["name"]
            st = state.setdefault(n, {"attempts": 0, "last": 0, "gpu": None})
            if n in done or n in alive:
                continue
            if st["attempts"] >= 3:
                continue
            if st["attempts"] > 0 and now - st["last"] < 600:
                continue
            pending.append(j)
        if len(done) == len(J):
            log("all jobs done")
            (OUT / "ALL_DONE").write_text(time.strftime("%F %T") + "\n")
            break
        if not pending and not alive & {j["name"] for j in J}:
            log("no job can run (failed jobs reached 3 attempts); stopping")
            break
        if (OUT / "STOP").exists():
            log("STOP file found; not launching (running jobs continue)")
            time.sleep(60)
            continue
        free = free_gb()
        ours = {g: 0 for g in GPUS}
        recent = {g: 0.0 for g in GPUS}
        for j in J:
            st = state[j["name"]]
            if j["name"] in alive and st["gpu"] is not None:
                ours[st["gpu"]] += 1
                if now - st["last"] < 240:
                    recent[st["gpu"]] += j["cost"]
        for j in pending:
            eff = {g: free.get(g, 0) - recent[g] for g in GPUS if ours[g] < MAX_PER_GPU}
            if not eff:
                break
            g = max(eff, key=eff.get)
            if eff[g] - j["cost"] < MARGIN_GB:
                continue   # a smaller job later in the list may still fit
            st = state[j["name"]]
            st["attempts"] += 1
            st["last"] = now
            st["gpu"] = g
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), JX_THREADS="2",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            lf = open(OUT / "logs" / f"{j['name']}.log", "a")
            lf.write(f"\n=== attempt {st['attempts']} on GPU {g} at {time.strftime('%F %T')}\n")
            lf.flush()
            subprocess.Popen(j["cmd"], shell=True, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
            log(f"launch {j['name']} on GPU {g} (attempt {st['attempts']}, free {eff[g]:.1f} GB, cost {j['cost']})")
            ours[g] += 1
            recent[g] += j["cost"]
        with open(state_p, "w") as f:
            json.dump(state, f, indent=1)
        time.sleep(45)


if __name__ == "__main__":
    main()
