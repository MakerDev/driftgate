"""Check that the code on this server is the exact code the Round-5 runs executed.

journal_expansion/artifacts/driftgate_tmc_final/precheck/code_identity.txt lists the sha256 of
every source file the Round-5 runs execute (taken on the old server). After migration, marker
comments ("  # [SERVER-...]") and rewritten paths change the raw sha256 of some files
(currently only journal_expansion/src/datasets_ext.py). This tool undoes exactly those two
changes in memory (strips the marker comment, maps the new roots back to the old roots on
marker lines only) and compares the result with the recorded sha256.

Usage (run from the repository root, with the same roots you gave rewrite_server_paths.py):
  python MIGRATION/tools/verify_code_identity.py --repo-root /new/repo --data-root /new/data [--git-root ...]
Before any rewrite, run it without options. Exit code 0 = all files identical.
"""
import argparse, hashlib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--repo-root", default="/disk2/Yujin/adaptive_splitomc_tmc")
ap.add_argument("--data-root", default="/disk2/Yujin/datasets")
ap.add_argument("--git-root", default=None)
ap.add_argument("--identity", default="journal_expansion/artifacts/driftgate_tmc_final/precheck/code_identity.txt")
a = ap.parse_args()
OLD = {"REPO_ROOT": "/disk2/Yujin/adaptive_splitomc_tmc", "DATA_ROOT": "/disk2/Yujin/datasets", "GIT_ROOT": "/disk2/Yujin"}
NEW = {"REPO_ROOT": str(Path(a.repo_root).resolve()) if a.repo_root != OLD["REPO_ROOT"] else OLD["REPO_ROOT"],
       "DATA_ROOT": str(Path(a.data_root).resolve()) if a.data_root != OLD["DATA_ROOT"] else OLD["DATA_ROOT"]}
NEW["GIT_ROOT"] = (str(Path(a.git_root).resolve()) if a.git_root else NEW["REPO_ROOT"]) \
    if (a.git_root or a.repo_root != OLD["REPO_ROOT"]) else OLD["GIT_ROOT"]


def normalize(text):
    out = []
    for l in text.split("\n"):
        if "  # [SERVER-" in l:
            code, _, comment = l.partition("  # [SERVER-")
            for cat in re.findall(r"SERVER-PATH:([A-Z_]+)", "[SERVER-" + comment):
                if cat == "GIT_ROOT":
                    for q in ('"', "'"):
                        code = code.replace(f"{q}{NEW[cat]}{q}", f"{q}{OLD[cat]}{q}")
                elif cat in NEW:
                    code = code.replace(NEW[cat], OLD[cat])
            l = code
        out.append(l)
    return "\n".join(out)


bad = 0
for line in (ROOT / a.identity).read_text().splitlines():
    m = re.match(r"^([0-9a-f]{64})  (\S+)$", line)
    if not m:
        continue
    want, rel = m.groups()
    raw = (ROOT / rel).read_bytes()
    got_raw = hashlib.sha256(raw).hexdigest()
    got_norm = hashlib.sha256(normalize(raw.decode()).encode()).hexdigest()
    status = "identical" if got_raw == want else ("identical after removing markers/path rewrite" if got_norm == want else "DIFFERENT")
    bad += status == "DIFFERENT"
    print(f"{status:48s} {rel}")
print(f"\n{'OK: all files match the Round-5 code' if not bad else f'FAIL: {bad} file(s) differ from the Round-5 code'}")
sys.exit(1 if bad else 0)
