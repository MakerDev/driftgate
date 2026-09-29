"""Round 5 run manifest: every run executed in Round 5 (B3 re-runs, B4, E1, E2, E3).
Expected runs come from runs/queue_r5/enqueued_*_snapshot.txt (the exact command lines).
Columns: run_id, run_name, experiment, arm, setting, seed, output_dir, complete, rounds,
eval_rounds, probe_eval_overlap, overlap_ok, device, worker exit code, start, end.
"""
import csv, glob, json, re
from pathlib import Path
JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
QD = JR / "runs/queue_r5"
OUT = Path(__file__).resolve().parent.parent / "tables" / "run_manifest.csv"
ENVN = {"A": "stepwise composition change", "mob": "client mobility", "svhn": "SVHN temporal",
        "c10gsig": "CIFAR-10 gradual", "c100gsig": "CIFAR-100 gradual", "tiny": "Tiny-ImageNet"}
starts, ends = {}, {}
for wl in glob.glob(str(QD / "logs/worker*.log")):
    for line in open(wl):
        m = re.match(r"\[(\w+) (\S+)\] START (\S+) \(\w+\) (.+)$", line.strip())
        if m: starts[m.group(3)] = (m.group(4), m.group(2))
        m = re.match(r"\[(\w+) (\S+)\] END (\S+) exit=(\d+) (.+)$", line.strip())
        if m: ends[m.group(3)] = (m.group(5), m.group(4))
rows = []
for snap in ("heavy", "light"):
    for line in open(QD / f"enqueued_{snap}_snapshot.txt"):
        rn = re.search(r"--run_name (\S+)", line).group(1); od = Path(re.search(r"--output_dir (\S+)", line).group(1))
        parts = rn.split("_"); exp = parts[0].upper()
        if exp == "E2":   arm, env = parts[2], "resA"            # e2_res_{arm}_A_s{s}
        elif exp == "B3": arm, env = parts[2], parts[1]          # b3_{env}_fx15_s{s}
        elif exp == "B4": arm, env = "_".join(parts[1:3]), parts[3]   # b4_apfl_{eta}_{env}_s{s}
        else:             arm, env = parts[1], parts[2]          # e1_relonly_{env}_s{s} / e3_{tag}_{env}_s{s}
        arm = {"fx15": "fixed 0.15", "fx20": "fixed 0.2", "fx40": "fixed 0.4", "relonly": "DriftGate (relonly)",
               "apfl_eta001": "APFL η=0.01", "apfl_eta010": "APFL η=0.1", "guard025": "DriftGate guard 0.25",
               "guard100": "DriftGate guard 1.0", "lmax060": "DriftGate λ_max 0.60", "lmax080": "DriftGate λ_max 0.80"}.get(arm, arm)
        setting = "ResNet-18 middle split, stepwise composition change" if env == "resA" else ENVN.get(env, env)
        seed = int(re.search(r"_s(\d+)$", rn).group(1))
        f = od / f"{rn}.json"
        if f.exists():
            h = json.load(open(f)); ov = h.get("probe_eval_overlap")
            want_rounds = int(re.search(r"--rounds (\d+)", line).group(1))
            complete = len(h["round"]) == want_rounds
            rows.append([h.get("config", {}).get("run_id", ""), rn, exp, arm, setting, seed, str(od.relative_to(JR)),
                         "yes" if complete else "no", len(h["round"]), len(h["eval"]), ov, "yes" if ov == 0 else "NO",
                         "cuda:0 = physical GPU 0 (CUDA_VISIBLE_DEVICES=0)", ends.get(rn, ("", ""))[1],  # [SERVER-GPU]
                         starts.get(rn, ("", ""))[0], ends.get(rn, ("", ""))[0]])
        else:
            rows.append(["", rn, exp, arm, "" if not rn else setting, seed, str(od.relative_to(JR)), "no (running or queued)",
                         "", "", "", "", "cuda:0 = physical GPU 0", ends.get(rn, ("", ""))[1], starts.get(rn, ("", ""))[0], ""])
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run_id", "run_name", "experiment", "arm", "setting", "seed", "output_dir", "complete", "rounds",
                "eval_rounds", "probe_eval_overlap", "overlap_ok", "device", "worker_exit_code", "start", "end"])
    w.writerows(rows)
done = sum(1 for r in rows if r[7] == "yes")
print(f"manifest: {len(rows)} runs, {done} complete, overlap_ok {sum(1 for r in rows if r[11] == 'yes')}")
