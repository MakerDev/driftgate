"""Phase G — tensor-space image corruptions with time-varying severity.

Corruptions operate on NORMALIZED tensors (post-transform), parameterized by
severity in [0, 1]; severity 0 is the identity. Implemented (5): gaussian_noise,
motion_blur (horizontal box blur), brightness, contrast, jpeg-like blockiness
(coarse quantization via avg-pool/upsample — a differentiable stand-in for JPEG,
documented as such).

CorruptedView wraps a dataset; a mutable SeverityState lets the runner set the
current round's severity so probe AND eval batches see the same corruption.
This is COVARIATE drift (p(x) changes; labels untouched) — never called
concept drift.
"""
import torch
import torch.nn.functional as F


def gaussian_noise(x, sev, gen=None):
    return x + torch.randn(x.shape, generator=gen) * (0.6 * sev)


def motion_blur(x, sev, gen=None):
    k = 1 + 2 * int(round(4 * sev))
    if k <= 1:
        return x
    kernel = torch.zeros(3, 1, 1, k)
    kernel[:, :, 0, :] = 1.0 / k
    pad = k // 2
    return F.conv2d(x.unsqueeze(0), kernel, padding=(0, pad), groups=3).squeeze(0)


def brightness(x, sev, gen=None):
    return x + 1.5 * sev


def contrast(x, sev, gen=None):
    factor = 1.0 - 0.8 * sev
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return (x - mean) * factor + mean


def jpeg_blockiness(x, sev, gen=None):
    if sev <= 0:
        return x
    f = max(1, int(round(1 + 3 * sev)))
    if f == 1:
        return x
    h, w = x.shape[-2:]
    down = F.avg_pool2d(x.unsqueeze(0), f)
    return F.interpolate(down, size=(h, w), mode="nearest").squeeze(0)


CORRUPTIONS = {"gaussian_noise": gaussian_noise, "motion_blur": motion_blur,
               "brightness": brightness, "contrast": contrast,
               "jpeg": jpeg_blockiness}


def severity_schedule(name, round_idx, total_rounds):
    """Severity in [0,1] per round. Shapes mirror the rho schedules."""
    frac = (round_idx - 1) / max(total_rounds, 1)
    if name == "abrupt_sev":
        return 0.0 if frac < 0.5 else 0.8
    if name == "gradual_sev":
        return min(0.8, max(0.0, 0.8 * (frac - 0.3) / 0.4)) if frac >= 0.3 else 0.0
    if name == "recurring_sev":
        seg = min(int(frac * 5), 4)
        return [0.0, 0.8, 0.0, 0.8, 0.0][seg]
    raise ValueError(name)


class SeverityState:
    """Mutable holder the runner updates each round."""

    def __init__(self):
        self.severity = 0.0


class CorruptedView(torch.utils.data.Dataset):
    """Applies `corruption` at the CURRENT severity to every fetched image.
    Deterministic per (index, round-severity) via a seeded generator, so probe
    and eval draws are reproducible."""

    def __init__(self, base, corruption, state, seed=0):
        self.base = base
        self.fn = CORRUPTIONS[corruption]
        self.state = state
        self.seed = seed
        self.targets = base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        sev = self.state.severity
        if sev > 0:
            gen = torch.Generator().manual_seed(
                self.seed * 1000003 + i * 97 + int(sev * 1000))
            x = self.fn(x, sev, gen)
        return x, y
