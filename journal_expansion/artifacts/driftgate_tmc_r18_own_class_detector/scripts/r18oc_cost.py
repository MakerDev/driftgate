"""Round 18 (own-class detector): device and edge cost of the MobileNetV2 split before and after improvement attempt 2
(directive 7.4 item 2, `deeper-split`: "the device computation and the feature size change, so write them in the cost
table"). Counts from the model definitions (src/r18oc_models.py), CIFAR input (32 x 32 resized to 64 x 64), one image.

  python .../r18oc_cost.py   # tables/R18oc_attempt2_cost.csv

MACs: Conv2d = output elements x (input channels / groups) x kernel area; Linear = input x output. Batch norm, activations,
pooling and the input resize are not counted. Bytes: float32 smashed data (uplink) and float32 edge logits (downlink) of one
offloaded request, as in mobile/r18oc_export_mobile.py.
"""
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src.r18oc_models import SPLITS, MBV2Client, MBV2Server, _pretrained_features  # noqa: E402


def macs(module, x):
    tot = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            tot[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            tot[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        y = module(x)
    for h in hs:
        h.remove()
    return tot[0], y


def params(module):
    return sum(p.numel() for p in module.parameters())


class Exit(nn.Module):
    def __init__(self, cm):
        super().__init__()
        self.exit_conv, self.exit_bn, self.exit_fc = cm.exit_conv, cm.exit_bn, cm.exit_fc

    def forward(self, rep):
        return self.exit_fc(torch.nn.functional.relu6(self.exit_bn(self.exit_conv(rep))).mean(dim=(2, 3)))


def main():
    C = 10
    x = torch.zeros(1, 3, 64, 64)   # the 64 x 64 input after the resize in MBV2Client.prep
    rows = []
    for split, n_dev in SPLITS.items():
        blk = _pretrained_features()[:n_dev].eval()
        cm = MBV2Client(C, n_dev, freeze=False).eval()
        sm = MBV2Server(C, n_dev).eval()
        m_blk, rep = macs(blk, x)
        m_exit, _ = macs(Exit(cm).eval(), rep)
        m_edge, _ = macs(sm, rep)
        rows.append([split, "attempt 2 (deeper-split)" if split == "stride16" else "stage 2", n_dev, "x".join(map(str, rep.shape[1:])),
                     params(blk), params(Exit(cm)), f"{m_blk / 1e6:.2f}", f"{m_exit / 1e6:.3f}", params(sm), f"{m_edge / 1e6:.2f}",
                     int(rep[0].numel()) * 4, C * 4])
    out = HERE / "tables" / "R18oc_attempt2_cost.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device split", "used in", "features[] stages on the device", "smashed data shape (C x H x W)",
                    "device block parameters", "device exit parameters", "device block MMACs", "device exit MMACs",
                    "edge parameters", "edge MMACs", "uplink bytes per offloaded request (float32)",
                    "downlink bytes per offloaded request (float32)"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
