"""§6.3 model files from the final model of S1 DriftGate seed 0.

  python journal_expansion/artifacts/driftgate_tmc_r6/device/export_models.py

Input : runs/phaseT6_S1/s1_driftgate_s0_final_models.pt (client 0's client block + exit, and the
        server block + exit of the hub cell and of client 0's home cell, after round 150)
Output: device/models/
  driftgate_s1_s0_weights.pth            state dicts for r6_device_models.py (client, hub server)
  client_block_exit.{pt,ptl,onnx}        x [B,3,32,32] -> (client_logits [B,10], feature [B,128,8,8])
  server_block_exit.{pt,ptl,onnx}        feature [B,128,8,8] -> server_logits [B,10]
  MANIFEST.csv                           file, bytes, sha256, max |diff| against the eager model
.pt = TorchScript (torch.jit.trace, eval mode), .ptl = PyTorch Mobile lite interpreter
(optimize_for_mobile), .onnx = opset 17 with a variable batch axis.
"""
import csv
import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
JR = HERE.parents[2]
sys.path.insert(0, str(HERE))
from r6_device_models import CIFARClient, CIFARServer  # noqa: E402

SRC = JR / "runs" / "phaseT6_S1" / "s1_driftgate_s0_final_models.pt"
OUT = HERE / "models"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src = torch.load(SRC, map_location="cpu")
    c, s = CIFARClient(), CIFARServer()
    c.load_state_dict(src["client_block_and_exit"])
    s.load_state_dict(src["server_block_and_exit_hub_cell"])
    c.eval(), s.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"client_block_and_exit": c.state_dict(), "server_block_and_exit": s.state_dict(),
                "source": str(SRC.relative_to(JR)), "run_id": src["run_id"], "client": src["client"],
                "server_cell": src["hub_cell"], "round": src["round"]}, OUT / "driftgate_s1_s0_weights.pth")
    x = torch.randn(8, 3, 32, 32)
    f = torch.randn(8, 128, 8, 8)
    rows = []
    with torch.no_grad():
        ref_c, ref_f = c(x)
        ref_s = s(f)
        tc = torch.jit.trace(c, x)
        ts = torch.jit.trace(s, f)
        tc.save(str(OUT / "client_block_exit.pt"))
        ts.save(str(OUT / "server_block_exit.pt"))
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimize_for_mobile(tc)._save_for_lite_interpreter(str(OUT / "client_block_exit.ptl"))
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(OUT / "server_block_exit.ptl"))
        torch.onnx.export(c, x, str(OUT / "client_block_exit.onnx"), opset_version=17, dynamo=False,
                          input_names=["image"], output_names=["client_logits", "feature"],
                          dynamic_axes={"image": {0: "batch"}, "client_logits": {0: "batch"}, "feature": {0: "batch"}})
        torch.onnx.export(s, f, str(OUT / "server_block_exit.onnx"), opset_version=17, dynamo=False,
                          input_names=["feature"], output_names=["server_logits"],
                          dynamic_axes={"feature": {0: "batch"}, "server_logits": {0: "batch"}})
        diffs = {}
        a, b = torch.jit.load(str(OUT / "client_block_exit.pt"))(x)
        diffs["client_block_exit.pt"] = max((a - ref_c).abs().max().item(), (b - ref_f).abs().max().item())
        diffs["server_block_exit.pt"] = (torch.jit.load(str(OUT / "server_block_exit.pt"))(f) - ref_s).abs().max().item()
        from torch.jit.mobile import _load_for_lite_interpreter
        a, b = _load_for_lite_interpreter(str(OUT / "client_block_exit.ptl"))(x)
        diffs["client_block_exit.ptl"] = max((a - ref_c).abs().max().item(), (b - ref_f).abs().max().item())
        diffs["server_block_exit.ptl"] = (_load_for_lite_interpreter(str(OUT / "server_block_exit.ptl"))(f) - ref_s).abs().max().item()
    import onnx
    for name in ("client_block_exit.onnx", "server_block_exit.onnx"):
        onnx.checker.check_model(onnx.load(str(OUT / name)))
        diffs[name] = "onnx.checker ok"
    try:
        import numpy as np
        import onnxruntime as ort
        worst = {"client_block_exit.onnx": 0.0, "server_block_exit.onnx": 0.0}
        for b in (1, 64):
            xb, fb = torch.randn(b, 3, 32, 32), torch.randn(b, 128, 8, 8)
            with torch.no_grad():
                rc, rf = c(xb)
                rs = s(fb)
            oc = ort.InferenceSession(str(OUT / "client_block_exit.onnx")).run(None, {"image": xb.numpy()})
            osv = ort.InferenceSession(str(OUT / "server_block_exit.onnx")).run(None, {"feature": fb.numpy()})
            worst["client_block_exit.onnx"] = max(worst["client_block_exit.onnx"], float(np.abs(oc[0] - rc.numpy()).max()),
                                                  float(np.abs(oc[1] - rf.numpy()).max()))
            worst["server_block_exit.onnx"] = max(worst["server_block_exit.onnx"], float(np.abs(osv[0] - rs.numpy()).max()))
        for k, v in worst.items():
            diffs[k] = f"{v:.3g} (onnxruntime {ort.__version__}, batch 1 and 64)"
    except ImportError:
        pass
    for p in sorted(OUT.iterdir()):
        if p.name == "MANIFEST.csv":
            continue
        rows.append([p.name, p.stat().st_size, sha(p), diffs.get(p.name, "")])
    with open(OUT / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "bytes", "sha256", "max_abs_diff_vs_eager"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
