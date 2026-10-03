"""Round 12 dispatcher: the 50 runs of directive 2.2, in priority order, on GPUs 0-3 shared with other users.

Each job copies its base run's command (provenance) and changes only: run name r12_<base>, output directory
runs/phaseT12_fusion, the Round 12 records on (Round 6 path: --record_device_signals --record_eval_probs
--record_train_label_hist; Round 5 path: --record_eval_probs --record_train_label_hist --record_device_signals) and,
for base runs from the old server, the repository path. Jobs without a base run of lambda 0.4 do not exist here
(every cell of the run list has one).

State is read from the file system and the process table, so the dispatcher can be restarted at any time:
done = result JSON exists; running = a process with "--run_name <name> " is alive; otherwise pending.
A job is launched on the GPU with the most free memory (nvidia-smi, minus the cost of jobs launched in the last
4 minutes) when free - cost >= 2.5 GB and fewer than MAX_PER_GPU of our jobs run there. A job that ends without a
JSON is retried after 10 minutes, at most 3 attempts. STOP file in the output directory -> stop launching.

  python journal_expansion/scripts/r12_dispatch.py --dry-run
  setsid nohup python journal_expansion/scripts/r12_dispatch.py < /dev/null > runs/phaseT12_fusion/dispatch.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
JR = ROOT / "journal_expansion"
OUT = JR / "runs/phaseT12_fusion"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU]
MAX_PER_GPU = 5
MARGIN_GB = 2.5
OLD_ROOT = "/disk2/Yujin/adaptive_splitomc_tmc"
R6_REC = ["--record_device_signals", "--record_eval_probs", "--record_train_label_hist"]
R5_REC = ["--record_eval_probs", "--record_train_label_hist", "--record_device_signals"]

# (group, scenario, training, base directory, base name pattern, seeds, cost GB)
PLAN = [
    ("individual", "S1", "fixed 0.4", "phaseT6_S1", "s1_fixed040_s{s}", range(5), 2.5),
    ("individual", "S1", "DriftGate", "phaseT6_S1", "s1_driftgate_own_s{s}", range(5), 2.5),
    ("joint", "Schedule A", "fixed 0.4", "phaseT1_disjoint", "t1_A_fx40_s{s}", range(5), 2.5),
    ("joint", "Schedule A", "DriftGate", "phaseT6_R0", "r0_b1_relonly_A_s{s}", range(5), 2.5),
    ("joint", "client mobility", "fixed 0.4", "phaseT1_disjoint", "t1_mob_fx40_s{s}", range(3), 2.5),
    ("joint", "client mobility", "DriftGate", "phaseT6_R0", "r0_b1_relonly_mob_s{s}", range(3), 2.5),
    ("individual", "K=500", "fixed 0.4", "phaseT6_S3", "s3k500_fixed040_s{s}", range(3), 9.5),
    ("individual", "K=200", "fixed 0.4", "phaseT6_S3", "s3k200_fixed040_s{s}", range(3), 5.0),
    ("architecture", "ResNet-18", "fixed 0.4", "phaseT5_E2_resnet", "e2_res_fx40_A_s{s}", range(3), 5.0),
    ("architecture", "ResNet-18", "DriftGate", "phaseT6_R0", "r0_e2_res_relonly_A_s{s}", range(3), 5.0),
    ("data", "CIFAR-100 gradual", "fixed 0.4", "phaseT3_fixedref", "t3_c100gsig_fx40_s{s}", range(3), 2.5),
    ("individual", "S2", "fixed 0.4", "phaseT6_S2", "s2_fixed040_s{s}", range(3), 2.5),
    ("individual", "S1-fast", "fixed 0.4", "phaseT6_S1fast", "s1fast_fixed040_s{s}", range(3), 2.5),
    ("individual", "participation 0.5", "fixed 0.4", "phaseT6_S4", "s4part05_fixed040_s{s}", range(3), 2.5),
]


def base_command(base_dir, base_name):
    h = json.load(open(JR / "runs" / base_dir / f"{base_name}.json"))
    prov = json.load(open(JR / "provenance" / f"{h['config']['run_id']}.json"))
    cmd = prov["command"]
    toks = cmd if isinstance(cmd, list) else cmd.split()
    return [t.replace(OLD_ROOT, str(ROOT)) for t in toks], h["config"]["run_id"]


def jobs():
    out = []
    for group, scen, train, bdir, pat, seeds, cost in PLAN:
        for s in seeds:
            base = pat.format(s=s)
            toks, base_id = base_command(bdir, base)
            name = f"r12_{base}"
            toks[toks.index("--run_name") + 1] = name
            toks[toks.index("--output_dir") + 1] = str(OUT)
            toks[toks.index("--device") + 1] = "cuda:0"
            rec = R6_REC if toks[0].endswith("run_r6.py") else R5_REC
            assert all(r not in toks for r in rec), (base, toks)
            cmd = ["python", "-u"] + toks + rec
            out.append(dict(name=name, base=base, base_dir=bdir, base_run_id=base_id, group=group, scenario=scen,
                            training=train, seed=s, cost=cost, cmd=" ".join(cmd)))
    assert len(out) == 50, len(out)
    assert len({j["name"] for j in out}) == 50
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
