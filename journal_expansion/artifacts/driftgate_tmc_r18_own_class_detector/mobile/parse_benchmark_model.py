"""Round 18, stage 4: TFLite benchmark_model logs (Android, adb) -> latency fragment of the measurement JSON.

  python parse_benchmark_model.py LOG_DIR --out measurement_part.json

LOG_DIR holds one log per model, named <model>.log (device_block, device_exit, detector, device_side), written by
  adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/<model>_float32.tflite --num_threads=4 \
      --warmup_runs=50 --num_runs=500 --enable_op_profiling=false > <model>.log
The parser reads the "Inference timings in us" line (avg, std, min, max per run).
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(runtime="TFLite benchmark_model", models={})
    for f in sorted(Path(a.log_dir).glob("*.log")):
        txt = f.read_text(errors="ignore")
        m = re.findall(r"Inference timings in us: Init: [\d.e+]+, First inference: [\d.e+]+, Warmup \(avg\): [\d.e+]+, Inference \(avg\): ([\d.e+]+)", txt)
        s = re.findall(r"count=(\d+) first=\d+ curr=\d+ min=(\d+) max=(\d+) avg=([\d.e+]+) std=(\d+)", txt)
        out = {}
        if m:
            out["mean_ms"] = float(m[-1]) / 1e3
        if s:
            cnt, mn, mx, avg, std = s[-1]
            out.update(n=int(cnt), min_ms=int(mn) / 1e3, max_ms=int(mx) / 1e3, std_ms=int(std) / 1e3,
                       mean_ms=out.get("mean_ms", float(avg) / 1e3))
        res["models"][f.stem] = out
        print(f.stem, out)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
