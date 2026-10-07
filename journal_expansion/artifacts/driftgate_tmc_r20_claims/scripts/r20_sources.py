"""Round 20 source manifest (Task 0): every file the Round 20 tables read, with size, sha256 and the last commit that touched it
(git-ignored records and caches are listed with their hash only). Writes tables/R20_sources.csv.
"""
import csv
import hashlib
import importlib.util
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("rc", HERE / "scripts" / "r20_recompute.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
JR, ART = rc.JR, rc.ART
REPO = JR.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def commit(p):
    r = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(p)], capture_output=True, text=True, cwd=REPO).stdout.strip()
    ign = subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=REPO).returncode == 0
    return r if r else ("git-ignored" if ign else "untracked")


def main():
    items = []
    for m, kind in ((rc, "script (Round 20)"), (rc.r19, "script (Round 19)"), (rc.md, "script (Round 19)"), (rc.r18, "script (Round 18)"),
                    (rc.r17, "script (Round 17)"), (rc.r16, "script (Round 16)"), (rc.m15, "script (Round 15)"), (rc.m14, "script (Round 14)"),
                    (rc.m13, "script (Round 13b)")):
        items.append((Path(m.__file__), kind))
    items.append((HERE / "scripts" / "r20_tables.py", "script (Round 20)"))
    items.append((HERE / "scripts" / "r20_report_tables.py", "script (Round 20)"))
    for p in sorted((ART / "driftgate_tmc_r16_selector").glob("*.json")) + sorted((ART / "driftgate_tmc_r17_featgate").glob("r17_config.json")):
        items.append((p, "configuration"))
    items.append((ART / "driftgate_tmc_r18_compare" / "tables" / "R18_A3_offload_claims.csv", "table (Round 18)"))
    items.append((ART / "driftgate_tmc_r19_weights" / "tables" / "R19_state.json", "table (Round 19)"))
    for p in sorted((ART / "driftgate_tmc_r19_1_integrated" / "tables").glob("*.csv")):
        items.append((p, "table (Round 19.1)"))
    items.append((ART / "driftgate_tmc_r19_1_integrated" / "R19_1_integrated_report_ko.md", "report (Round 19.1)"))
    for p in sorted((ART / "driftgate_tmc_r15_revision" / "manuscript" / "v28").rglob("*.tex")):
        items.append((p, "manuscript v28"))
    for st, (runs, pat, seeds, period) in rc.SETTINGS.items():
        for s in seeds:
            name = pat.format(s)
            if period == "three-day":
                sc = st.split()[0].lower()
                for d in (1, 2, 3):
                    for suf in (".json", "_evalprobs.npz", "_trace.npz"):
                        items.append((JR / "runs" / "phaseT19_multiday" / f"r19_{sc}_s{s}_day{d}{suf}", f"record ({st})"))
            else:
                for suf in (".json", "_evalprobs.npz", "_trace.npz", "_rec.npz"):
                    p = runs / f"{name}{suf}"
                    if p.exists():
                        items.append((p, f"record ({st})"))
                rp = rc.r16.refs_path(str(runs), name)
                if rp.exists():
                    items.append((rp, "Round 16 refs cache"))
                c18 = rc.R18C / f"{name}.pkl"
                if c18.exists():
                    items.append((c18, "Round 18 cache"))
                src = runs / f"{name}_evalprobs.npz"
                k = __import__("hashlib").sha256(f"{rc.m15.SCRIPT_SHA}|{rc.m15.DEP_SHA}|{src}|{src.stat().st_size}|{src.stat().st_mtime_ns}".encode()).hexdigest()[:16]
                c15 = rc.m15.CACHE / f"{runs.name}__{name}__{k}.pkl"
                if c15.exists():
                    items.append((c15, "Round 15 cache"))
            c19 = rc.R19C / f"{name}.pkl"
            if c19.exists():
                items.append((c19, "Round 19 cache"))
            items.append((rc.CACHE / f"{name}.pkl", "Round 20 recompute cache"))
    items.append((rc.CACHE / "a0.json", "Round 20 recompute cache"))
    rows = []
    for p, kind in items:
        rows.append([str(p.relative_to(REPO)), kind, p.stat().st_size, sha(p), commit(p)])
    with open(HERE / "tables" / "R20_sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path (repository relative)", "kind", "bytes", "sha256", "last commit or git status"])
        w.writerows(rows)
    print(f"R20_sources.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
