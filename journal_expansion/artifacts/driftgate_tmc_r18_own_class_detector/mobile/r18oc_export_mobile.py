"""Round 18 (own-class detector), stage 4: export the device side of one device (block, exit, detector) to ONNX and check it.

  python r18oc_export_mobile.py CKPT MODEL_CONFIG VARIANT [--device 0] [--dataset cifar10] [--out DIR]

CKPT is a round-150 checkpoint of a Round 18 run (runs/phaseT18oc/<run>_ckpt_r150.pt); VARIANT is the detector
"input|model|negatives". Four float32 ONNX models (opset 17) are written:
  device_block.onnx      image [1, 3, 32, 32] (CIFAR-normalised) -> smashed [1, Cb, H, W] (the feature sent when offloading)
  device_exit.onnx       smashed -> device_logits [1, C], exit_vector [1, De]
  detector.onnx          block_vector [1, Db] (+) exit_vector -> g [1]  (inputs as the variant needs them)
  device_side.onnx       image -> device_logits, smashed, g  (the whole device path)
  edge_block.onnx        smashed -> edge_logits  (the saved edge model of the device's first cell; for the edge time)
Every model is compared with PyTorch on 64 test images (max |difference| <= 1e-4 required); the inputs and PyTorch outputs
are saved in reference.npz for the TFLite check (r18oc_tflite_check.py). export_manifest.json lists shapes, parameters,
bytes of one offloaded request (float32 smashed data, uplink) and of the response (float32 edge logits, downlink).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

JR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src.r18oc_detector import INPUTS, MODELS, NEGS  # noqa: E402
from src.r18oc_models import _frozen_block, block_vector, exit_vector, make_factory  # noqa: E402

TOL = 1e-4


class Block(nn.Module):
    """device block; for MobileNetV2 the 32 -> 64 bilinear resize is written with scale factor 2 (identical to size 64,
    checked below), because onnx2tf converts the size form of ONNX Resize to a wrong shape."""

    def __init__(self, cm):
        super().__init__()
        self.cm = cm
        self.mbv2 = cm.__class__.__name__ == "MBV2Client"
        if self.mbv2 and cm.freeze:
            self.frozen = _frozen_block(cm.n_dev, "cpu")

    def forward(self, x):
        if not self.mbv2:
            return self.cm(x)[1]
        x = x * self.cm.in_scale + self.cm.in_shift
        x = F.interpolate(x, scale_factor=self.cm.size / 32.0, mode="bilinear", align_corners=False)
        return (self.frozen if hasattr(self, "frozen") else self.cm.block)(x)


class Exit(nn.Module):
    def __init__(self, cm):
        super().__init__()
        self.cm = cm

    def forward(self, rep):
        v = exit_vector(self.cm, rep)
        fc = self.cm.exit_fc if hasattr(self.cm, "exit_fc") else self.cm.aux_fc
        logits = fc(v)
        if getattr(self.cm, "other", False):
            logits = logits[:, :self.cm.C]
        return logits, v


class Detector(nn.Module):
    def __init__(self, ent, mdl, slot, cols):
        super().__init__()
        self.mdl, self.cols = mdl, cols
        self.register_buffer("mu", ent["mu"][slot].float().clone())
        self.register_buffer("sd", ent["sd"][slot].float().clone())
        if mdl == "logreg":
            self.register_buffer("w", ent["w"][slot].float().clone())
            self.register_buffer("b", ent["b"][slot].float().reshape(1).clone())
        else:
            for nm in ("W1", "b1", "w2"):
                self.register_buffer(nm, ent[nm][slot].float().clone())
            self.register_buffer("b2", ent["b2"][slot].float().reshape(1).clone())

    def forward(self, vb, ve):
        x = torch.cat([vb, ve], 1)[:, self.cols]
        x = (x - self.mu) / self.sd
        if self.mdl == "logreg":
            z = x @ self.w + self.b
        else:
            z = torch.relu(x @ self.W1 + self.b1) @ self.w2 + self.b2
        return torch.sigmoid(z)


class Edge(nn.Module):
    def __init__(self, sm):
        super().__init__()
        self.sm = sm

    def forward(self, rep):
        return self.sm(rep)[0]


class DeviceSide(nn.Module):
    def __init__(self, blk, ex, det):
        super().__init__()
        self.blk, self.ex, self.det = blk, ex, det

    def forward(self, x):
        rep = self.blk(x)
        logits, v = self.ex(rep)
        return logits, rep, self.det(block_vector(rep), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("model_config")
    ap.add_argument("variant")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "export"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    C = 100 if a.dataset == "cifar100" else 10
    w_fc = ck["client_states"][a.device].get("exit_fc.weight")
    other = w_fc is not None and w_fc.shape[0] == C + 1
    mf = make_factory(a.model_config, C, dataset=a.dataset, other=other)
    cm = mf.make_client()
    cm.load_state_dict(ck["client_states"][a.device])
    cm.eval()
    inp, mdl, neg = a.variant.split("|")
    st = ck["r18oc"]
    ent = st["det"][a.device][(inp, mdl)]
    slot = NEGS.index(neg)
    assert bool(ent["valid"][slot]), "the device has no trained detector for this variant"
    db, de = st["dims"]
    cols = {"block": list(range(db)) + ([db + de - 1] if other else []), "exit": list(range(db, db + de)),
            "concat": list(range(db + de))}[inp]
    blk, ex, det = Block(cm).eval(), Exit(cm).eval(), Detector(ent, mdl, slot, torch.tensor(cols)).eval()
    full = DeviceSide(blk, ex, det).eval()
    zc = int(ck["client_cells"][a.device][0])
    sm = mf.make_server()
    sm.load_state_dict(ck["server_avg"][zc])
    edge = Edge(sm.eval()).eval()
    # test images (the evaluation transform of the runs)
    if a.dataset == "cifar10":
        from data.partition import get_cifar10
        _, te = get_cifar10(str(JR.parent / "data_cache"))
    else:
        from src.datasets_ext import get_dataset
        _, te, _ = get_dataset(a.dataset)
    X = torch.stack([te[i][0] for i in range(64)])
    with torch.no_grad():
        rep = blk(X)
        assert torch.equal(rep, cm(X)[1]), "export block differs from the run's device block"
        logits, v = ex(rep)
        vb = block_vector(rep)
        if other:
            v_in = torch.cat([v, torch.softmax(cm.exit_fc(v), 1)[:, -1:]], 1)
        else:
            v_in = v
        g = det(vb, v_in)
        lf, rf, gf = full(X)
        el = edge(rep)
    assert torch.allclose(lf, logits) and torch.allclose(gf, g), "combined model differs from the parts"
    import onnxruntime as ort
    specs = {"device_block": (blk, (X[:1],), ["image"], ["smashed"], {"image": X.numpy()}, [rep.numpy()]),
             "device_exit": (ex, (rep[:1],), ["smashed"], ["device_logits", "exit_vector"], {"smashed": rep.numpy()},
                             [logits.numpy(), v.numpy()]),
             "detector": (det, (vb[:1], v_in[:1]), ["block_vector", "exit_vector"], ["g"],
                          {"block_vector": vb.numpy(), "exit_vector": v_in.numpy()}, [g.numpy()]),
             "device_side": (full, (X[:1],), ["image"], ["device_logits", "smashed", "g"], {"image": X.numpy()},
                             [lf.numpy(), rf.numpy(), gf.numpy()]),
             "edge_block": (edge, (rep[:1],), ["smashed"], ["edge_logits"], {"smashed": rep.numpy()}, [el.numpy()])}
    man = dict(checkpoint=str(a.ckpt), model_config=a.model_config, variant=a.variant, device=a.device, dataset=a.dataset,
               edge_cell=zc, other_class=other, tolerance=TOL, models={})
    ref = {}
    for nm, (mod, args, ins, outs, feeds, want) in specs.items():
        f = out / f"{nm}.onnx"
        torch.onnx.export(mod, args, str(f), input_names=ins, output_names=outs, opset_version=17,
                          dynamic_axes={n: {0: "batch"} for n in ins + outs}, dynamo=False)
        sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        got = sess.run(outs, feeds)
        diff = max(float(np.abs(gv - wv).max()) for gv, wv in zip(got, want))
        man["models"][nm] = dict(file=f.name, inputs={n: list(feeds[n].shape[1:]) for n in ins},
                                 outputs={n: list(w.shape[1:]) for n, w in zip(outs, want)}, onnx_max_abs_diff=diff,
                                 onnx_ok=diff <= TOL, parameters=int(sum(p.numel() for p in mod.parameters()))
                                 + int(sum(b.numel() for b in mod.buffers())))
        for n in ins:
            ref[f"{nm}__in__{n}"] = feeds[n]
        for n, w in zip(outs, want):
            ref[f"{nm}__out__{n}"] = w
        print(nm, "ONNX max |diff|", diff)
    np.savez(out / "reference.npz", **ref)
    man["bytes_per_offloaded_request"] = dict(uplink_float32_smashed=int(np.prod(rep.shape[1:]) * 4),
                                              downlink_float32_edge_logits=int(C * 4),
                                              note="application payload only; protocol headers not included")
    (out / "export_manifest.json").write_text(json.dumps(man, indent=1))
    assert all(m["onnx_ok"] for m in man["models"].values()), "ONNX outputs differ from PyTorch by more than 1e-4"
    print(json.dumps(man["bytes_per_offloaded_request"]))


if __name__ == "__main__":
    main()
