"""Round 18 (own-class detector), directive 6.4: the iNaturalist data with real observation locations.

Source: iNaturalist 2021 `train` (annotations with latitude and longitude). The Hsu et al. (ECCV 2020) geographic splits are
defined on the iNaturalist 2017 images (about 186 GB), which do not fit the free disk; the directive's fallback (2021 with
coordinates) is used. `train_mini` has 50 images per species (5,000 images for 100 species, per region and species median
0), too few for 50 devices, so the full 2021 `train` is read as a stream and only the selected images are kept (decisions.md
D7). Construction fixed before any Round 18 result:
  1. continent with the most observations: North America (longitude -130..-50, latitude 7..84; contiguous, Hawaii excluded)
  2. five regions: k-means (k = 5, n_init 10, random_state 0) on (latitude, longitude x cos(mean latitude))
  3. 100 species: most images over the five regions; ties (1,095 species at the 300 cap) by numpy default_rng(18).permutation
  4. images of those species inside the five regions, centre-cropped to a square and resized to 96 x 96 (bilinear,
     antialias), uint8
Outputs (DATA): inat_r18_images.npz (images [N, 96, 96, 3] uint8, species index 0..99, region, latitude, longitude,
original file name), and CSVs in the artifact tables: regions, species with counts per region.

  python -I .../r18oc_inat_prepare.py select     # selection and CSVs (annotations only)
  python -I .../r18oc_inat_prepare.py extract    # stream train.tar.gz and keep the selected images
"""
import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path("/home/honeynaps/data/driftgate_datasets/inat2021_r18")
ANN = DATA / "ann" / "train.json"
URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"
ART = Path(__file__).resolve().parent.parent
SIZE = 96


def select():
    from sklearn.cluster import KMeans
    a = json.load(open(ANN))
    cat = {an["image_id"]: an["category_id"] for an in a["annotations"]}
    names = {c["id"]: c["name"] for c in a["categories"]}
    im = a["images"]
    lat = np.array([x["latitude"] if x["latitude"] is not None else np.nan for x in im], float)
    lon = np.array([x["longitude"] if x["longitude"] is not None else np.nan for x in im], float)
    y = np.array([cat[x["id"]] for x in im])
    na = (lon >= -130) & (lon <= -50) & (lat >= 7) & (lat <= 84)
    c0 = float(np.cos(np.deg2rad(lat[na].mean())))
    X = np.c_[lat[na], lon[na] * c0]
    km = KMeans(5, n_init=10, random_state=0).fit(X)
    reg = np.full(len(im), -1)
    reg[na] = km.labels_
    cnt = np.bincount(y[na], minlength=len(names))
    order = np.lexsort((np.random.default_rng(18).permutation(len(names)), -cnt))
    top = order[:100]
    sp_index = {int(s): i for i, s in enumerate(top)}
    sel = np.flatnonzero(na & np.isin(y, top))
    man = dict(files=[im[i]["file_name"] for i in sel], species=[sp_index[int(y[i])] for i in sel],
               region=[int(reg[i]) for i in sel], lat=[float(lat[i]) for i in sel], lon=[float(lon[i]) for i in sel],
               category_id=[int(y[i]) for i in sel], cos_lat0=c0, n_na=int(na.sum()))
    (DATA / "selection.json").write_text(json.dumps(man))
    T = ART / "tables"
    T.mkdir(exist_ok=True)
    with open(T / "R18oc_inat_regions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "centre latitude", "centre longitude", "observations (all species)", "images of the 100 species"])
        for r in range(5):
            w.writerow([r, f"{km.cluster_centers_[r, 0]:.3f}", f"{km.cluster_centers_[r, 1] / c0:.3f}", int((reg == r).sum()),
                        int(sum(1 for z in man["region"] if z == r))])
    with open(T / "R18oc_inat_species.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species index", "category id", "name", "images in the five regions"] + [f"region {r}" for r in range(5)])
        for i, s in enumerate(top):
            w.writerow([i, int(s), names[int(s)], int(cnt[s])] + [int(((y == s) & (reg == r)).sum()) for r in range(5)])
    print(f"selected {len(sel)} images of 100 species; regions {np.bincount(man['region'], minlength=5).tolist()}")


def extract():
    from PIL import Image
    man = json.loads((DATA / "selection.json").read_text())
    want = {f: i for i, f in enumerate(man["files"])}
    imgs = np.zeros((len(want), SIZE, SIZE, 3), np.uint8)
    got = np.zeros(len(want), bool)
    t0, nb = time.time(), 0
    with urllib.request.urlopen(URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                nb += 1
                if nb % 200000 == 0:
                    print(f"{time.time() - t0:.0f}s members {nb} kept {got.sum()}", flush=True)
                if not m.isfile() or m.name not in want:
                    continue
                data = tf.extractfile(m).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                w_, h_ = img.size
                s = min(w_, h_)
                img = img.crop(((w_ - s) // 2, (h_ - s) // 2, (w_ - s) // 2 + s, (h_ - s) // 2 + s))
                img = img.resize((SIZE, SIZE), Image.BILINEAR, reducing_gap=None)
                i = want[m.name]
                imgs[i] = np.asarray(img, np.uint8)
                got[i] = True
                if got.all():
                    break
    np.savez(DATA / "inat_r18_images.npz", images=imgs, found=got, species=np.array(man["species"], np.int16),
             region=np.array(man["region"], np.int8), lat=np.array(man["lat"]), lon=np.array(man["lon"]),
             files=np.array(man["files"]))
    print(f"done in {time.time() - t0:.0f}s: {got.sum()} of {len(got)} images", flush=True)


if __name__ == "__main__":
    {"select": select, "extract": extract}[sys.argv[1]]()
