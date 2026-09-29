"""Multi-dataset support + partition-severity generators (Phase 3).

Datasets (all expose .targets as a list/array of int labels):
  cifar10 (32px/10c), cifar100 (32px/100c), tinyimagenet (64px/200c),
  gtsrb (real-world traffic-sign sensing, resized 32px/43c), stl10 (64px/10c)

Per-dataset ND partition knobs are NOT forced to a common class count; each
dataset gets its own main-classes-per-client so that clients remain
role-separated specialists (the property the disagreement signal relies on).

Partition severities:
  nd1 with classes-per-client in {1, 2, 5} (via frac = k / num_classes)
  dirichlet with alpha in {0.1, 0.3, 1.0} (main classes := smallest set
    covering >= 90% of the client's samples — keeps the Main/OOP/OOR evaluator
    machinery intact)
  cluster-scope overlap in {0, 25, 50, 75}% (make_es_topology overlap knob)
"""
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_ROOT = Path("/home/honeynaps/data/driftgate_datasets")  # [SERVER-PATH:DATA_ROOT]

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    "gtsrb": ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}

# dataset -> (num_classes, in_spatial after transform, main classes per client)
META = {
    "cifar10": (10, 8, 2),
    "cifar100": (100, 8, 5),
    "tinyimagenet": (200, 16, 10),
    "gtsrb": (43, 8, 3),
    "stl10": (10, 16, 2),
    "svhn": (10, 8, 2),  # frozen holdout (Phase E): never used in any controller design
}


class TinyImageNet(Dataset):
    """tiny-imagenet-200 (train split from train/, test split from val/)."""

    def __init__(self, root, train=True, transform=None):
        self.root = Path(root) / "tiny-imagenet-200"
        self.transform = transform
        wnids = sorted((self.root / "train").iterdir())
        self.wnid_to_idx = {w.name: i for i, w in enumerate(wnids)}
        self.samples = []
        if train:
            for w in wnids:
                for img in sorted((w / "images").glob("*.JPEG")):
                    self.samples.append((str(img), self.wnid_to_idx[w.name]))
        else:
            ann = self.root / "val" / "val_annotations.txt"
            for line in ann.read_text().strip().split("\n"):
                parts = line.split("\t")
                self.samples.append((str(self.root / "val" / "images" / parts[0]),
                                     self.wnid_to_idx[parts[1]]))
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


class _WithTargets(Dataset):
    """Wrap a dataset lacking .targets (e.g. GTSRB) with a materialized label list."""

    def __init__(self, base, targets):
        self.base = base
        self.targets = targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i]


def get_dataset(name, data_root=None):
    """Returns (train_ds, test_ds, meta_dict). All train/test ds have .targets."""
    root = Path(data_root) if data_root else DATA_ROOT
    mean, std = STATS[name]
    ncls, in_spatial, per_client = META[name]

    def tf(size=None):
        ops = []
        if size:
            ops.append(transforms.Resize((size, size)))
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(ops)

    if name == "cifar10":
        tr = datasets.CIFAR10(root / "cifar10", train=True, download=True, transform=tf())
        te = datasets.CIFAR10(root / "cifar10", train=False, download=True, transform=tf())
    elif name == "cifar100":
        tr = datasets.CIFAR100(root / "cifar100", train=True, download=True, transform=tf())
        te = datasets.CIFAR100(root / "cifar100", train=False, download=True, transform=tf())
    elif name == "tinyimagenet":
        tr = TinyImageNet(root, train=True, transform=tf())
        te = TinyImageNet(root, train=False, transform=tf())
    elif name == "gtsrb":
        tr = datasets.GTSRB(root / "gtsrb", split="train", download=True, transform=tf(32))
        te = datasets.GTSRB(root / "gtsrb", split="test", download=True, transform=tf(32))
        tr = _WithTargets(tr, [s[1] for s in tr._samples])
        te = _WithTargets(te, [s[1] for s in te._samples])
    elif name == "stl10":
        tr = datasets.STL10(root / "stl10", split="train", download=True, transform=tf(64))
        te = datasets.STL10(root / "stl10", split="test", download=True, transform=tf(64))
        tr = _WithTargets(tr, list(tr.labels))
        te = _WithTargets(te, list(te.labels))
    elif name == "svhn":
        tr = datasets.SVHN(root / "svhn", split="train", download=True, transform=tf())
        te = datasets.SVHN(root / "svhn", split="test", download=True, transform=tf())
        tr = _WithTargets(tr, list(tr.labels))
        te = _WithTargets(te, list(te.labels))
    else:
        raise ValueError(f"unknown dataset {name}")

    meta = {"num_classes": ncls, "in_spatial": in_spatial,
            "classes_per_client": per_client,
            "classes_per_client_frac": per_client / ncls}
    return tr, te, meta


# ---------------------------------------------------------------- partitions

def dirichlet_partition(train_labels, num_classes, num_clients, alpha, seed=0,
                        main_cover=0.9):
    """Dirichlet(alpha) non-IID partition. Returns (client_indices,
    client_main_classes) with main = smallest class set covering >= main_cover
    of the client's samples."""
    rng = np.random.default_rng(seed)
    train_labels = np.asarray(train_labels)
    idx_by_class = {c: np.where(train_labels == c)[0] for c in range(num_classes)}
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])
    client_indices = {cid: [] for cid in range(num_clients)}
    for c in range(num_classes):
        props = rng.dirichlet([alpha] * num_clients)
        counts = (props * len(idx_by_class[c])).astype(int)
        pos = 0
        for cid in range(num_clients):
            client_indices[cid].extend(idx_by_class[c][pos:pos + counts[cid]].tolist())
            pos += counts[cid]
    main_classes = {}
    for cid in range(num_clients):
        labs = train_labels[client_indices[cid]]
        if len(labs) == 0:
            main_classes[cid] = set()
            continue
        vals, cnts = np.unique(labs, return_counts=True)
        order = np.argsort(-cnts)
        cum = np.cumsum(cnts[order]) / cnts.sum()
        k = int(np.searchsorted(cum, main_cover)) + 1
        main_classes[cid] = set(int(v) for v in vals[order[:k]])
    return client_indices, main_classes


def scope_from_mains(client_main_classes, client_to_es, num_es):
    """ES scope := union of member clients' main classes (for Dirichlet mode,
    where scope isn't drawn first)."""
    scope = {z: set() for z in range(num_es)}
    for cid, mains in client_main_classes.items():
        for z in client_to_es.get(cid, []):
            scope[z] |= mains
    return scope
