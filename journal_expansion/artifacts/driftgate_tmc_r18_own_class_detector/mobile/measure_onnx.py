"""Round 18, stage 4: latency (and optionally power on Jetson) of the exported device-side models with ONNX Runtime.

  python measure_onnx.py EXPORT_DIR --out measurement_part.json [--providers CUDAExecutionProvider,CPUExecutionProvider]
                         [--warmup 50] [--repeats 500] [--tegrastats]

Runs device_block, device_exit, detector and device_side one request at a time (batch 1) on the reference inputs and
writes mean / p95 / p99 latency in ms. With --tegrastats (Jetson) it samples the board power rails every 100 ms while
each model runs in a loop for --power_seconds and reports average power and energy per request (idle power, measured
first for the same time, is subtracted and stated). The output is one fragment of the measurement JSON (see
measurement_template.json); copy its values into that file.
"""
import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_model(sess, feeds, warmup, repeats):
    names = [o.name for o in sess.get_outputs()]
    n = len(next(iter(feeds.values())))
    for i in range(warmup):
        sess.run(names, {k: v[i % n:i % n + 1] for k, v in feeds.items()})
    t = []
    for i in range(repeats):
        f = {k: v[i % n:i % n + 1] for k, v in feeds.items()}
        t0 = time.perf_counter()
        sess.run(names, f)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return dict(mean_ms=float(t.mean()), p95_ms=float(np.percentile(t, 95)), p99_ms=float(np.percentile(t, 99)), n=int(repeats))


class Tegra:
    """average total board power (mW) from tegrastats lines while active (VDD_IN or the sum of the POM_5V rails)."""

    def __init__(self):
        self.vals, self.stop = [], False

    def start(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self):
        for line in self.p.stdout:
            if self.stop:
                break
            m = re.findall(r"(VDD_IN|POM_5V_IN)\s+(\d+)mW", line)
            if m:
                self.vals.append(float(m[0][1]))

    def finish(self):
        self.stop = True
        self.p.terminate()
        return float(np.mean(self.vals)) if self.vals else float("nan")


def power_loop(sess, feeds, seconds):
    names = [o.name for o in sess.get_outputs()]
    n = len(next(iter(feeds.values())))
    tg = Tegra()
    tg.start()
    t0, cnt = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        sess.run(names, {k: v[cnt % n:cnt % n + 1] for k, v in feeds.items()})
        cnt += 1
    el = time.perf_counter() - t0
    return tg.finish(), cnt, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--tegrastats", action="store_true")
    ap.add_argument("--power_seconds", type=float, default=30.0)
    a = ap.parse_args()
    d = Path(a.export_dir)
    man = json.loads((d / "export_manifest.json").read_text())
    ref = np.load(d / "reference.npz")
    prov = a.providers.split(",")
    res = dict(runtime=f"onnxruntime {ort.__version__}", providers=prov, models={},
               bytes_per_offloaded_request=man["bytes_per_offloaded_request"])
    idle = None
    if a.tegrastats:
        tg = Tegra()
        tg.start()
        time.sleep(a.power_seconds)
        idle = tg.finish()
        res["idle_power_mW"] = idle
    for nm, m in man["models"].items():
        sess = ort.InferenceSession(str(d / m["file"]), providers=prov)
        feeds = {n: ref[f"{nm}__in__{n}"].astype(np.float32) for n in m["inputs"]}
        r = run_model(sess, feeds, a.warmup, a.repeats)
        if a.tegrastats:
            p, cnt, el = power_loop(sess, feeds, a.power_seconds)
            r.update(power_mW=p, requests=cnt, seconds=el,
                     energy_per_request_mJ=(p - idle) * el / cnt if np.isfinite(p) else float("nan"),
                     energy_note="(average board power while running - idle power) x time / requests")
        res["models"][nm] = r
        print(nm, json.dumps(r))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
