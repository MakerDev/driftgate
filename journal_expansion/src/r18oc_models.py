"""Round 18 (own-class detector): split models and per-request vectors.

Model configurations (directive 2.3):
  cnn-scratch                 the default split CNN (models/architectures.py), random initialisation
  pretrained-finetuned-block  torchvision MobileNetV2 (ImageNet IMAGENET1K_V1), device block = features[0..6]
                              (32 channels, stride 8), edge block = features[7..18] + pooling + classifier
  pretrained-frozen-block     the same, but the device block is never trained: one shared, frozen block in eval mode
                              (not part of the device state, so it is never averaged or mixed); the device trains its
                              exit only
  deeper split (directive 7.4, attempt 2): device block = features[0..10] (64 channels, stride 16)

Inputs are the cached CIFAR tensors (ToTensor + Normalize with the CIFAR statistics). The MobileNetV2 device model
undoes the CIFAR normalisation, applies the ImageNet normalisation and resizes 32x32 -> 64x64 (bilinear,
align_corners=False) before its block.

Device exit of MobileNetV2: 1x1 conv (192 channels, no bias), batch normalisation, ReLU6, spatial mean, linear.
Vectors used by the detectors (directive 3.5):
  block vector  = device-block output averaged over space (CNN 128, MobileNetV2 32 / 64)
  exit vector   = the input of the last linear layer of the device exit (CNN 128, MobileNetV2 192)
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architectures import CIFARClient

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
               "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))}
SPLITS = {"stride8": 7, "stride16": 11}   # number of features[] stages on the device
EXIT_CH = 192
MODEL_CONFIGS = ("cnn-scratch", "pretrained-finetuned-block", "pretrained-frozen-block")

_PRETRAINED = {}
_FROZEN_BLOCKS = {}


def _pretrained_features():
    if "f" not in _PRETRAINED:
        import torchvision
        m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        _PRETRAINED["f"] = m.features.eval()
    return _PRETRAINED["f"]


def _frozen_block(n_dev, device):
    key = (n_dev, str(device))
    if key not in _FROZEN_BLOCKS:
        b = copy.deepcopy(_pretrained_features()[:n_dev]).to(device).eval()
        for p in b.parameters():
            p.requires_grad_(False)
        _FROZEN_BLOCKS[key] = b
    return _FROZEN_BLOCKS[key]


class MBV2Client(nn.Module):
    def __init__(self, num_classes, n_dev, freeze, dataset="cifar10", size=64):
        super().__init__()
        self.n_dev, self.freeze, self.size = n_dev, freeze, size
        if not freeze:
            self.block = copy.deepcopy(_pretrained_features()[:n_dev])
        cin = _block_out_channels(n_dev)
        self.block_ch = cin
        self.exit_conv = nn.Conv2d(cin, EXIT_CH, 1, bias=False)
        self.exit_bn = nn.BatchNorm2d(EXIT_CH)
        self.exit_fc = nn.Linear(EXIT_CH, num_classes)
        cm, cs = CIFAR_STATS[dataset]
        # x_imagenet = (x_cifar * std_c + mean_c - mean_i) / std_i = x_cifar * scale + shift
        scale = torch.tensor([cs[i] / IMAGENET_STD[i] for i in range(3)]).view(1, 3, 1, 1)
        shift = torch.tensor([(cm[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i] for i in range(3)]).view(1, 3, 1, 1)
        self.register_buffer("in_scale", scale, persistent=False)
        self.register_buffer("in_shift", shift, persistent=False)

    def prep(self, x):
        x = x * self.in_scale + self.in_shift
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)

    def block_forward(self, x):
        x = self.prep(x)
        if self.freeze:
            with torch.no_grad():
                return _frozen_block(self.n_dev, x.device)(x)
        return self.block(x)

    def exit_vec(self, rep):
        return F.relu6(self.exit_bn(self.exit_conv(rep))).mean(dim=(2, 3))

    def forward(self, x):
        rep = self.block_forward(x)
        return self.exit_fc(self.exit_vec(rep)), rep


def _block_out_channels(n_dev):
    if ("ch", n_dev) not in _PRETRAINED:
        with torch.no_grad():
            _PRETRAINED[("ch", n_dev)] = int(_pretrained_features()[:n_dev](torch.zeros(1, 3, 64, 64)).shape[1])
    return _PRETRAINED[("ch", n_dev)]


class MBV2Server(nn.Module):
    def __init__(self, num_classes, n_dev):
        super().__init__()
        self.block = copy.deepcopy(_pretrained_features()[n_dev:])
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, rep):
        x = self.block(rep).mean(dim=(2, 3))
        return self.fc(self.drop(x)), None


class MBV2Factory:
    """make_client / make_server like models.architectures.ModelFactory. New layers (device exit, edge classifier) use
    the global torch RNG, as ModelFactory does; the pretrained layers are copied."""

    def __init__(self, num_classes, freeze_block, split="stride8", dataset="cifar10"):
        self.num_classes, self.freeze, self.split, self.dataset = num_classes, freeze_block, split, dataset
        self.n_dev = SPLITS[split]

    def make_client(self):
        return MBV2Client(self.num_classes, self.n_dev, self.freeze, self.dataset)

    def make_server(self):
        return MBV2Server(self.num_classes, self.n_dev)


def make_factory(model_config, num_classes, dataset="cifar10", split="stride8"):
    if model_config == "cnn-scratch":
        from models.architectures import ModelFactory
        return ModelFactory(dataset="cifar10", num_classes=num_classes)
    if model_config == "pretrained-finetuned-block":
        return MBV2Factory(num_classes, False, split, dataset)
    if model_config == "pretrained-frozen-block":
        return MBV2Factory(num_classes, True, split, dataset)
    raise ValueError(model_config)


def exit_vector(model, rep):
    """Input of the last linear layer of the device exit, computed from a block output rep with this model's exit."""
    if isinstance(model, CIFARClient):
        a = F.relu(model.aux_bn(model.aux_conv(rep)))
        return F.adaptive_avg_pool2d(a, 1).flatten(1)
    return model.exit_vec(rep)


def block_vector(rep):
    return rep.float().mean(dim=(2, 3))


def device_param_bytes(model, part):
    """float32 bytes of the device block ('block') or of the device exit ('exit') of one device model."""
    if isinstance(model, CIFARClient):
        names = ("aux_conv", "aux_bn", "aux_fc")
    else:
        names = ("exit_conv", "exit_bn", "exit_fc")
    n = 0
    for nm, p in model.state_dict().items():
        in_exit = nm.split(".")[0] in names
        if p.dtype.is_floating_point and (in_exit == (part == "exit")):
            n += p.numel()
    return 4 * n
