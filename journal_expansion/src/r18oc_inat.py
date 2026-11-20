"""Round 18 (own-class detector), directive 6.4: the iNaturalist setting with real observation locations.

Data: runs of artifacts/driftgate_tmc_r18_own_class_detector/scripts/r18oc_inat_prepare.py (100 species, five North
American regions, 96 x 96 uint8 images). Construction (fixed before any Round 18 result; decisions.md D7, D8):
  - images of each (species, region) group: 30% request pool (at least 1 when the group has 2 or more images), the rest
    the training pool; numpy default_rng([partition seed, 18, 60])
  - cell -> region: cells ranked by device-round membership over the day (ties: smaller index) take the regions ranked by
    images of the 100 species (largest first)
  - device k: home region = region of its home cell; own classes M_k = 10 species drawn without replacement with
    probability proportional to the home region's training-pool counts (default_rng([seed, 18, 61, k])); training
    samples = training-pool images of those species in the home region
  - requests of an evaluation round (N_REQ per device, default_rng([seed, 18, 62, round, k]), drawn with replacement):
    at home 90% own species (request-pool images of the species in the home region; the species' whole request pool if
    the home region has none) and 10% from the home region's distribution (species proportional to the region's
    request-pool counts, image uniform); away (or in two cells) every request from the current cells' distribution
    (two cells: each request picks one of the two cells with equal probability)
  - request kinds: Main = species in M_k; OOP = species owned by a resident of a current cell; OOR = the rest
The images stay uint8 on the GPU; the MobileNetV2 device model divides by 255 and applies the ImageNet normalisation
(dataset "inat", no resize).
"""
from pathlib import Path

import numpy as np
import torch

DATA = Path("/home/honeynaps/data/driftgate_datasets/inat2021_r18/inat_r18_images.npz")  # [SERVER-PATH:DATA_ROOT]
N_CLASSES = 100
N_OWN = 10
N_REQ = 400
HOME_OWN = 0.9
REQ_FRAC = 0.3


class _DS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.targets = X, [int(v) for v in y]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.X[i], self.targets[i]


def load(seed):
    z = np.load(DATA)
    assert bool(z["found"].all()), "the iNaturalist extraction is incomplete"
    img = np.transpose(z["images"], (0, 3, 1, 2))
    sp, rg = z["species"].astype(np.int64), z["region"].astype(np.int64)
    rng = np.random.default_rng([int(seed), 18, 60])
    is_req = np.zeros(len(sp), bool)
    for s in range(N_CLASSES):
        for r in range(5):
            ix = np.flatnonzero((sp == s) & (rg == r))
            if len(ix) >= 2:
                n = max(1, int(round(REQ_FRAC * len(ix))))
                is_req[rng.choice(ix, n, replace=False)] = True
    tr, te = np.flatnonzero(~is_req), np.flatnonzero(is_req)
    train_ds = _DS(torch.as_tensor(img[tr]), sp[tr])
    test_ds = _DS(torch.as_tensor(img[te]), sp[te])
    return train_ds, test_ds, dict(train_region=rg[tr], test_region=rg[te], n_train=len(tr), n_test=len(te))


def cell_region_map(env, n_img_region):
    K, T = env["member"].shape[:2]
    L = len(env["cell_xy"])
    visits = np.array([int((env["member"] == z).any(2).sum()) for z in range(L)])
    cells = sorted(range(L), key=lambda z: (-visits[z], z))
    regions = sorted(range(L), key=lambda r: (-n_img_region[r], r))
    return {c: r for c, r in zip(cells, regions)}, visits


def build_partition(meta, train_labels, env, seed):
    tr_rg = meta["train_region"]
    L = len(env["cell_xy"])
    n_img = np.bincount(np.r_[tr_rg, meta["test_region"]], minlength=L)
    c2r, visits = cell_region_map(env, n_img)
    K = len(env["home_cell"])
    indices, mains = {}, {}
    for k in range(K):
        r = c2r[int(env["home_cell"][k])]
        cnt = np.bincount(train_labels[tr_rg == r], minlength=N_CLASSES).astype(float)
        p = cnt / cnt.sum()
        own = np.random.default_rng([int(seed), 18, 61, int(k)]).choice(N_CLASSES, N_OWN, replace=False, p=p)
        mains[k] = set(int(s) for s in own)
        indices[k] = [int(i) for i in np.flatnonzero((tr_rg == r) & np.isin(train_labels, own))]
    cell_groups = {z: set() for z in range(L)}
    for k in range(K):
        cell_groups[int(env["home_cell"][k])] |= mains[k]
    all_used = set().union(*mains.values())
    return indices, mains, cell_groups, all_used, dict(cell_region=c2r, cell_visits=visits.tolist())


class RequestBuilder:
    def __init__(self, test_labels, test_region, cell_region, cell_groups, seed, home_region):
        self.y, self.rg = np.asarray(test_labels), np.asarray(test_region)
        self.c2r, self.groups, self.seed, self.home_region = cell_region, cell_groups, int(seed), home_region
        self.pool = {(s, r): np.flatnonzero((self.y == s) & (self.rg == r)) for s in range(N_CLASSES) for r in range(5)}
        self.pool_all = {s: np.flatnonzero(self.y == s) for s in range(N_CLASSES)}
        self.dist = {r: np.bincount(self.y[self.rg == r], minlength=N_CLASSES) / max((self.rg == r).sum(), 1) for r in range(5)}

    def _draw(self, rng, s, r):
        ix = self.pool[(s, r)]
        if len(ix) == 0:
            ix = self.pool_all[s]
        return int(ix[rng.integers(len(ix))])

    def build(self, main, cells, rho=None, k=None, r=None, home=None):
        rng = np.random.default_rng([self.seed, 18, 62, int(r), int(k)])
        main = sorted(main)
        cur = [int(z) for z in cells if z >= 0]
        idx = []
        hr = self.home_region[int(k)]
        for _ in range(N_REQ):
            if home:
                if rng.random() < HOME_OWN:
                    s = int(main[rng.integers(len(main))])
                else:
                    s = int(rng.choice(N_CLASSES, p=self.dist[hr]))
                idx.append(self._draw(rng, s, hr))
            else:
                rr = self.c2r[cur[rng.integers(len(cur))]]
                s = int(rng.choice(N_CLASSES, p=self.dist[rr]))
                idx.append(self._draw(rng, s, rr))
        scope = set()
        for z in cur:
            scope |= self.groups[z]
        oop = scope - set(main)
        oor = set(range(N_CLASSES)) - scope - set(main)
        return np.array(idx, np.int64), oop, oor, ""
