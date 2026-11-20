"""Round 18, stage 4: ONNX -> TFLite (float32) with onnx2tf, and the check against the PyTorch outputs (reference.npz).

Run with the separate conversion environment (~/venvs/r18_tflite: tensorflow-cpu 2.17.1, onnx2tf 1.26.3):
  ~/venvs/r18_tflite/bin/python r18oc_tflite_check.py EXPORT_DIR

For every model of export_manifest.json: onnx2tf -b 1 converts it (4-D tensors become NHWC); the TFLite interpreter runs the
64 reference inputs one at a time; outputs are compared with PyTorch after transposing NHWC back to NCHW (tolerance 1e-4).
The result is added to export_manifest.json (tflite_max_abs_diff, tflite_ok, tflite file).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

TOL = 1e-4


def to_nhwc(a):
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def main():
    d = Path(sys.argv[1])
    man = json.loads((d / "export_manifest.json").read_text())
    ref = np.load(d / "reference.npz")
    onnx2tf = str(Path(sys.executable).parent / "onnx2tf")
    # onnx2tf validates 4-D float inputs with a sample-image file that it otherwise downloads; a local seeded file is used
    calib = d / "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not calib.exists():
        np.save(calib, np.random.default_rng(18).uniform(0, 1, (20, 128, 128, 3)).astype(np.float32))
    for nm, m in man["models"].items():
        od = d / f"tf_{nm}"
        if od.exists():
            shutil.rmtree(od)
        r = subprocess.run([onnx2tf, "-i", str(d / m["file"]), "-o", str(od), "-b", "1", "-n"], capture_output=True, text=True,
                           cwd=str(d))
        tfl = od / f"{nm}_float32.tflite"
        if r.returncode != 0 or not tfl.exists():
            m.update(tflite_ok=False, tflite_error=(r.stdout + r.stderr)[-2000:])
            print(nm, "conversion failed")
            continue
        shutil.copy(tfl, d / f"{nm}_float32.tflite")
        it = tf.lite.Interpreter(model_path=str(tfl))
        it.allocate_tensors()
        ins, outs = it.get_input_details(), it.get_output_details()
        in_names = list(m["inputs"])
        out_names = list(m["outputs"])
        # match interpreter tensors to names by shape (onnx2tf keeps the order for these small graphs; checked by shape)
        n = ref[f"{nm}__in__{in_names[0]}"].shape[0]
        diff = 0.0
        for i in range(n):
            for j, nmi in enumerate(in_names):
                x = to_nhwc(ref[f"{nm}__in__{nmi}"][i:i + 1]).astype(np.float32)
                det = [t for t in ins if list(t["shape"]) == list(x.shape)]
                det = det[j] if len(det) > j else det[0]
                it.set_tensor(det["index"], x)
            it.invoke()
            used = set()
            for nmo in out_names:
                want = ref[f"{nm}__out__{nmo}"][i:i + 1]
                cand = [t for t in outs if t["index"] not in used and list(t["shape"]) == list(to_nhwc(want).shape)]
                best = None
                for t in cand:
                    got = it.get_tensor(t["index"])
                    got = np.transpose(got, (0, 3, 1, 2)) if got.ndim == 4 else got
                    dd = float(np.abs(got.reshape(want.shape) - want).max())
                    if best is None or dd < best[0]:
                        best = (dd, t["index"])
                used.add(best[1])
                diff = max(diff, best[0])
        m.update(tflite_file=f"{nm}_float32.tflite", tflite_max_abs_diff=diff, tflite_ok=diff <= TOL,
                 tflite_input_layout="NHWC for 4-D inputs (onnx2tf)")
        print(nm, "TFLite max |diff|", diff)
    (d / "export_manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
