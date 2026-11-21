"""Round 18 (own-class detector): sha256 of every run output kept on the 8T disk (decisions.md D9: run outputs are not
committed; their sha256 is recorded instead, as in Round 13b), and of the analysis caches and mobile exports.

  python .../r18oc_manifest.py   # tables/R18oc_files_sha256.csv (area, file, bytes, sha256)

Areas: runs (runs/phaseT18oc, the dispatcher logs included), cache (analysis caches), mobile_export (exported models).
"""
import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STORE = Path("/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120")
AREAS = {"runs": STORE / "runs", "cache": STORE / "analysis_cache", "mobile_export": STORE / "mobile_export"}


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    rows = []
    for area, root in AREAS.items():
        if not root.exists():
            continue
        for p in sorted(x for x in root.rglob("*") if x.is_file() and not x.is_symlink()):
            rows.append([area, str(p.relative_to(root)), p.stat().st_size, sha256(p)])
    out = HERE / "tables" / "R18oc_files_sha256.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "file", "bytes", "sha256"])
        w.writerows(rows)
    print(f"{out.name}: {len(rows)} files, {sum(r[2] for r in rows) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
