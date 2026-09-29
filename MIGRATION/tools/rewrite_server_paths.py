"""Rewrite old-server paths on the NEW server (run from the repository root).

Only lines that carry a marker written by mark_server_paths.py are changed:
  [SERVER-PATH:REPO_ROOT]  /disk2/Yujin/adaptive_splitomc_tmc  -> --repo-root
  [SERVER-PATH:DATA_ROOT]  /disk2/Yujin/datasets               -> --data-root
  [SERVER-PATH:GIT_ROOT]   "/disk2/Yujin"                      -> --git-root (default: --repo-root)
The markers stay in place, so a later move can run this tool again with --old-* options.

Not rewritten automatically (reported for a manual decision):
  [SERVER-PATH:EXTERNAL]  sibling projects v3/v4 that were not migrated (legacy ICTC script only)
  [SERVER-GPU]            GPU count / device assumptions (see MIGRATION/START_HERE.md, step 6)
Run records (runs/**, provenance/**, logs, reports, queue snapshots) are never touched.

Usage:
  python MIGRATION/tools/rewrite_server_paths.py --repo-root /new/path/driftgate --data-root /new/data   # dry run
  python MIGRATION/tools/rewrite_server_paths.py --repo-root ... --data-root ... --apply              # write
"""
import argparse, ast, re, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo-root", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--git-root", default=None)
ap.add_argument("--old-repo-root", default="/disk2/Yujin/adaptive_splitomc_tmc")
ap.add_argument("--old-data-root", default="/disk2/Yujin/datasets")
ap.add_argument("--old-git-root", default="/disk2/Yujin")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
new_repo = str(Path(a.repo_root).resolve()); new_data = str(Path(a.data_root).resolve())
new_git = str(Path(a.git_root).resolve()) if a.git_root else new_repo
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) != new_repo:
    print(f"WARNING: this checkout is at {ROOT}, but --repo-root is {new_repo}")

SUBS = {
    "SERVER-PATH:REPO_ROOT": [(a.old_repo_root, new_repo)],
    "SERVER-PATH:DATA_ROOT": [(a.old_data_root, new_data)],
    "SERVER-PATH:GIT_ROOT": [(f'"{a.old_git_root}"', f'"{new_git}"'), (f"'{a.old_git_root}'", f"'{new_git}'")],
}
SKIP = ("/runs/", "/provenance/", "/.git/", "/MIGRATION/", "/logs/")
changes, manual = [], []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file() or f.suffix not in {".py", ".sh", ".yaml", ".yml"} or any(s in f.as_posix() for s in SKIP):
        continue
    text = f.read_text(); lines = text.split("\n"); new = list(lines)
    for i, l in enumerate(lines):
        if "[SERVER-" not in l:
            continue
        code, sep, comment = l.partition("  # [SERVER-")
        tags = re.findall(r"\[(SERVER-[A-Z:_]+)\]", l)
        for t in tags:
            if t in SUBS:
                for old, nw in SUBS[t]:
                    code = code.replace(old, nw)
            else:
                manual.append((f.relative_to(ROOT).as_posix(), i + 1, t, l.strip()[:150]))
        cand = code + sep + comment
        if cand != l:
            new[i] = cand
            changes.append((f.relative_to(ROOT).as_posix(), i + 1, l.strip()[:110], cand.strip()[:110]))
    if new != lines and a.apply:
        out = "\n".join(new)
        if f.suffix == ".py":
            ast.parse(out)  # raises if the rewrite broke syntax
        f.write_text(out)
        if f.suffix == ".sh" and subprocess.run(["bash", "-n", str(f)]).returncode:
            f.write_text(text); sys.exit(f"bash -n failed after rewrite: {f}")

print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {len(changes)} line(s) {'rewritten' if a.apply else 'would change'}")
for c in changes[:400]:
    print(f"  {c[0]}:{c[1]}\n      - {c[2]}\n      + {c[3]}")
print(f"\nManual decisions needed ({len(manual)} line(s)):")
for m in manual:
    print(f"  {m[0]}:{m[1]} [{m[2]}] {m[3]}")
# leftovers: any old path still present in executable code outside the skipped folders
left = []
for f in ROOT.rglob("*"):
    if f.is_file() and f.suffix in {".py", ".sh", ".yaml", ".yml"} and not any(s in f.as_posix() for s in SKIP):
        for i, l in enumerate(f.read_text().split("\n")):
            if (a.old_repo_root in l or a.old_data_root in l) and a.apply and "[SERVER-PATH:EXTERNAL]" not in l:
                left.append(f"{f.relative_to(ROOT)}:{i + 1}")
if a.apply:
    print(f"\nOld paths left in executable code: {len(left)}", *left[:20], sep="\n  ")
