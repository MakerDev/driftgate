"""Model family + split-point variants (Phase 3 / 5.3).

Every client module returns (aux_logits, representation); every server module
maps representation -> (logits, None) — the contract of the v4 stack.

Families:
  cnn        — the v4 4-conv CNN (FlexCNN generalizes it to any input size)
  resnet18   — BasicBlock ResNet-18-style; split early/middle/late
  mobilenetv2— inverted-residual lite; split early/middle/late

split_stats(factory) reports client/server params and activation size per
sample — the raw numbers behind the split-point vs activation-traffic
trade-off table.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ flex CNN
class FlexCNNClient(nn.Module):
    """v4 CIFARClient generalized to arbitrary input size."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        c = base_channels
        self.conv1 = nn.Conv2d(3, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 2)
        self.conv4 = nn.Conv2d(c * 2, c * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 4)
        self.aux_conv = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.aux_bn = nn.BatchNorm2d(c * 4)
        self.aux_fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        rep = F.max_pool2d(x, 2)
        a = F.relu(self.aux_bn(self.aux_conv(rep)))
        a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        return self.aux_fc(a), rep


class FlexCNNServer(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, in_spatial=8):
        super().__init__()
        c = base_channels
        self.conv = nn.Conv2d(c * 4, c * 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(c * 8)
        spatial = in_spatial // 2
        self.fc1 = nn.Linear(c * 8 * spatial * spatial, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, rep):
        x = F.relu(self.bn(self.conv(rep)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x), None


# ------------------------------------------------------------------ resnet18
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        s = self.short(x) if self.short else x
        return F.relu(out + s)


def _res_layers():
    """(name, module-builder, out_channels) for the 4 ResNet-18 stages."""
    return [("layer1", lambda: nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64)), 64),
            ("layer2", lambda: nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128)), 128),
            ("layer3", lambda: nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256)), 256),
            ("layer4", lambda: nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512)), 512)]


RESNET_SPLITS = {"early": 1, "middle": 2, "late": 3}  # stages on the client


class ResNetClient(nn.Module):
    def __init__(self, num_classes=10, split="middle"):
        super().__init__()
        k = RESNET_SPLITS[split]
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())
        specs = _res_layers()[:k]
        self.stages = nn.Sequential(*[b() for _, b, _ in specs])
        cout = specs[-1][2]
        self.aux_fc = nn.Linear(cout, num_classes)

    def forward(self, x):
        rep = self.stages(self.stem(x))
        a = F.adaptive_avg_pool2d(rep, 1).flatten(1)
        return self.aux_fc(a), rep


class ResNetServer(nn.Module):
    def __init__(self, num_classes=10, split="middle"):
        super().__init__()
        k = RESNET_SPLITS[split]
        specs = _res_layers()[k:]
        self.stages = nn.Sequential(*[b() for _, b, _ in specs])
        self.fc = nn.Linear(specs[-1][2], num_classes)

    def forward(self, rep):
        x = self.stages(rep)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x), None


# -------------------------------------------------------------- mobilenetv2
class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6()]
        layers += [nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(),
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


MBV2_CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
MBV2_SPLITS = {"early": 2, "middle": 4, "late": 6}  # block groups on the client


def _mbv2_blocks(start, end, cin):
    blocks, c = [], cin
    for t, cout, n, s in MBV2_CFG[start:end]:
        for i in range(n):
            blocks.append(InvertedResidual(c, cout, s if i == 0 else 1, t))
            c = cout
    return nn.Sequential(*blocks), c


class MobileNetClient(nn.Module):
    def __init__(self, num_classes=10, split="middle"):
        super().__init__()
        k = MBV2_SPLITS[split]
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU6())
        self.blocks, cout = _mbv2_blocks(0, k, 32)
        self.aux_fc = nn.Linear(cout, num_classes)

    def forward(self, x):
        rep = self.blocks(self.stem(x))
        a = F.adaptive_avg_pool2d(rep, 1).flatten(1)
        return self.aux_fc(a), rep


class MobileNetServer(nn.Module):
    def __init__(self, num_classes=10, split="middle"):
        super().__init__()
        k = MBV2_SPLITS[split]
        cin = _mbv2_blocks(0, k, 32)[1]
        self.blocks, cout = _mbv2_blocks(k, len(MBV2_CFG), cin)
        self.head = nn.Sequential(nn.Conv2d(cout, 1280, 1, bias=False),
                                  nn.BatchNorm2d(1280), nn.ReLU6())
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, rep):
        x = self.head(self.blocks(rep))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x), None


# ------------------------------------------------------------------ factory
class FlexModelFactory:
    def __init__(self, family="cnn", num_classes=10, in_spatial=8, split="middle"):
        self.family = family
        self.num_classes = num_classes
        self.in_spatial = in_spatial
        self.split = split

    def make_client(self):
        if self.family == "cnn":
            return FlexCNNClient(self.num_classes)
        if self.family == "resnet18":
            return ResNetClient(self.num_classes, self.split)
        if self.family == "mobilenetv2":
            return MobileNetClient(self.num_classes, self.split)
        raise ValueError(self.family)

    def make_server(self):
        if self.family == "cnn":
            return FlexCNNServer(self.num_classes, in_spatial=self.in_spatial)
        if self.family == "resnet18":
            return ResNetServer(self.num_classes, self.split)
        if self.family == "mobilenetv2":
            return MobileNetServer(self.num_classes, self.split)
        raise ValueError(self.family)


def split_stats(factory, input_size=32):
    """client/server params + activation floats per sample at the cut."""
    c, s = factory.make_client(), factory.make_server()
    x = torch.randn(2, 3, input_size, input_size)
    c.eval(), s.eval()
    with torch.no_grad():
        logits, rep = c(x)
        out, _ = s(rep)
    return {
        "client_params": sum(p.numel() for p in c.parameters()),
        "server_params": sum(p.numel() for p in s.parameters()),
        "activation_floats_per_sample": int(rep[0].numel()),
        "rep_shape": list(rep.shape[1:]),
        "out_classes": out.shape[1],
    }
