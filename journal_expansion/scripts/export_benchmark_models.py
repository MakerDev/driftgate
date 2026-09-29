"""Phase J — export the split blocks for on-device benchmarking.

Exports (per requested format): client block, server block, and a probe-path
wrapper (client forward -> rep -> server forward -> signal tensors). No latency
numbers are produced here; measurement happens on the user's device via
scripts/benchmark_ondevice.py.

Usage:
  python journal_expansion/scripts/export_benchmark_models.py \
      --family cnn --dataset cifar10 --split middle \
      --formats eager torchscript onnx --outdir journal_expansion/exported
"""
import argparse
import json
import sys
from pathlib import Path

import torch

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    sys.path.insert(0, p)

from src.models_ext import FlexModelFactory, split_stats
from src.datasets_ext import META


class ClientExport(torch.nn.Module):
    """Tensor-only outputs (tracing rejects the None in (logits, None) tuples)."""

    def __init__(self, client):
        super().__init__()
        self.client = client

    def forward(self, x):
        client_logits, rep = self.client(x)
        return client_logits, rep


class ServerExport(torch.nn.Module):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def forward(self, rep):
        return self.server(rep)[0]


class ProbePath(torch.nn.Module):
    """client forward + server tail + the tensors the signal library consumes."""

    def __init__(self, client, server):
        super().__init__()
        self.client = client
        self.server = server

    def forward(self, x):
        client_logits, rep = self.client(x)
        server_logits = self.server(rep)[0]
        return client_logits, server_logits, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="cnn", choices=["cnn", "resnet18", "mobilenetv2"])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--split", default="middle", choices=["early", "middle", "late"])
    ap.add_argument("--formats", nargs="+", default=["eager", "torchscript"],
                    choices=["eager", "torchscript", "onnx"])
    ap.add_argument("--outdir", default=str(JOURNAL_ROOT / "exported"))
    args = ap.parse_args()

    ncls, in_spatial, _ = META.get(args.dataset, (10, 8, 2))
    input_size = 64 if in_spatial == 16 else 32
    f = FlexModelFactory(family=args.family, num_classes=ncls,
                         in_spatial=in_spatial, split=args.split)
    client, server = f.make_client().eval(), f.make_server().eval()
    probe = ProbePath(client, server).eval()

    out = Path(args.outdir) / f"{args.family}_{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    ex = torch.randn(1, 3, input_size, input_size)

    manifest = {"family": args.family, "dataset": args.dataset, "split": args.split,
                "input_size": input_size,
                "stats": split_stats(f, input_size=input_size), "exports": {}}

    for fmt in args.formats:
        try:
            if fmt == "eager":
                torch.save({"client": client.state_dict(), "server": server.state_dict(),
                            "meta": manifest["stats"]}, out / "eager_state.pt")
                manifest["exports"]["eager"] = "eager_state.pt"
            elif fmt == "torchscript":
                torch.jit.trace(ClientExport(client), ex).save(str(out / "client_block.pt"))
                with torch.no_grad():
                    _, rep = client(ex)
                torch.jit.trace(ServerExport(server), rep).save(str(out / "server_block.pt"))
                torch.jit.trace(probe, ex).save(str(out / "probe_path.pt"))
                manifest["exports"]["torchscript"] = ["client_block.pt", "server_block.pt",
                                                     "probe_path.pt"]
            elif fmt == "onnx":
                torch.onnx.export(ClientExport(client), ex, str(out / "client_block.onnx"),
                                  input_names=["x"],
                                  output_names=["client_logits", "rep"],
                                  dynamic_axes={"x": {0: "batch"}})
                with torch.no_grad():
                    _, rep = client(ex)
                torch.onnx.export(ServerExport(server), rep, str(out / "server_block.onnx"),
                                  input_names=["rep"], output_names=["server_logits"],
                                  dynamic_axes={"rep": {0: "batch"}})
                manifest["exports"]["onnx"] = ["client_block.onnx", "server_block.onnx"]
        except Exception as e:  # report per-format failure, keep going
            manifest["exports"][fmt] = f"FAILED: {e}"

    with open(out / "export_manifest.json", "w") as fo:
        json.dump(manifest, fo, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
