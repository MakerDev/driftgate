"""Round 15 dispatcher: A2 runs (plan section 4) on GPUs 0-3.

Day-1 runs (8): the Round 12 commands of r12_s1_fixed040_s{0..4} and r12_s2_fixed040_s{0..2} (provenance) with run name
r15_<s1|s2>_fixed040_s<seed>, output runs/phaseT15_replay, and --save_checkpoint --record_eval_logprobs added.
Replay runs (8): the same commands with run name r15_<s1|s2>_replay_s<seed>, --record_eval_logprobs and
--replay_from <day-1 checkpoint>; a replay job waits until its checkpoint exists.
Scheduling as in r13b_dispatch.py (free memory minus jobs launched in the last 4 minutes, margin 2.5 GB, at most
MAX_PER_GPU jobs per GPU, retry after 10 minutes, at most 3 attempts; STOP file -> stop launching).

  python journal_expansion/scripts/r15_dispatch.py --dry-run
  setsid nohup python -u journal_expansion/scripts/r15_dispatch.py < /dev/null > runs/phaseT15_replay/dispatch.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/honeynaps/data/driftgate")  # [SERVER-PATH:REPO_ROOT]
JR = ROOT / "journal_expansion"
OUT = JR / "runs/phaseT15_replay"
GPUS = [0, 1, 2, 3]  # [SERVER-GPU]
MAX_PER_GPU = 5
MARGIN_GB = 2.5
REC = ["--record_device_signals", "--record_eval_probs", "--record_train_label_hist"]
PLAN = [("S1", "r12_s1_fixed040_s{}", "s1", range(5)), ("S2", "r12_s2_fixed040_s{}", "s2", range(3))]


def base_command(base):
    h = json.load(open(JR / "runs" / "phaseT12_fusion" / f"{base}.json"))
    prov = json.load(open(JR / "provenance" / f"{h['config']['run_id']}.json"))
    cmd = prov["command"]
    return (cmd if isinstance(cmd, list) else cmd.split()), h["config"]["run_id"]


def jobs():
    out = []
    for kind in ("day1", "replay"):
        for scen, pat, tag, seeds in PLAN:
            for s in seeds:
                base = pat.format(s)
                toks, base_id = base_command(base)
                assert all(r in toks for r in REC) and toks[toks.index("--lambda_val") + 1] == "0.4", toks
                day1 = f"r15_{tag}_fixed040_s{s}"
                name = day1 if kind == "day1" else f"r15_{tag}_replay_s{s}"
                toks[toks.index("--run_name") + 1] = name
                toks[toks.index("--output_dir") + 1] = str(OUT)
                toks[toks.index("--device") + 1] = "cuda:0"
                extra = (["--save_checkpoint", "--record_eval_logprobs"] if kind == "day1"
                         else ["--record_eval_logprobs", "--replay_from", str(OUT / f"{day1}_ckpt.pt")])
                out.append(dict(name=name, kind=kind, scenario=scen, seed=s, base=base, base_run_id=base_id, cost=3.0,
                                requires=None if kind == "day1" else str(OUT / f"{day1}_ckpt.pt"),
                                after=None if kind == "day1" else day1,
                                cmd=" ".join(["python", "-u"] + toks + extra)))
    assert len(out) == 16 and len({j["name"] for j in out}) == 16
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
            if j["requires"] and (not Path(j["requires"]).exists() or j["after"] in alive):   # day-1 run finished
                continue
            pending.append(j)
        if len(done) == len(J):
            log("all jobs done")
            (OUT / "ALL_DONE").write_text(time.strftime("%F %T") + "\n")
            break
        waiting = [j for j in J if j["name"] not in done and j["requires"] and not Path(j["requires"]).exists()
                   and state[j["name"]]["attempts"] < 3]
        if not pending and not alive & {j["name"] for j in J} and not waiting:
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
