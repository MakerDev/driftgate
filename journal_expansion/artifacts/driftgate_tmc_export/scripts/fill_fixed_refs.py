"""Fill the disjoint matched-fixed-reference cells for the 4 transfer settings once
runs/phaseT3_fixedref/ completes. Writes data/fixed_refs_transfer.csv and updates
tables/final_closure_round2/table_settings.csv. Idempotent; safe to run partial.
"""
import json, glob, re, csv
import numpy as np
from pathlib import Path
JR=Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
DATA=Path(__file__).resolve().parent.parent/"data"
T3=JR/"runs/phaseT3_fixedref"
def integ(f):
    h=json.load(open(f)); return float(np.mean([e["acc_total"] for e in h["eval"]]))
def fam(pat):
    d={}
    for f in sorted(glob.glob(str(T3/pat))):
        m=re.search(r"_s(\d+)\.json$",f)
        if m: d[int(m.group(1))]=f
    return d
SET={"CIFAR-10 gradual":"c10gsig","CIFAR-100 gradual":"c100gsig",
     "CIFAR-100 spatial":"c100sp","Tiny-ImageNet":"tiny"}
rows=[]
for name,env in SET.items():
    for lk in ["fx20","fx40"]:
        fs=fam(f"t3_{env}_{lk}_s*.json")
        if not fs: rows.append([name,lk,"(running)","",0]); continue
        v=[integ(f) for f in fs.values()]
        rows.append([name,lk,f"{np.mean(v)*100:.4f}",f"{np.std(v,ddof=1)*100:.4f}" if len(v)>1 else "",len(v)])
with open(DATA/"fixed_refs_transfer.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["setting","fixed_arm","mean_pct","sd_pct","n_seeds"]); w.writerows(rows)
done=sum(1 for r in rows if r[2]!="(running)")
print(f"fixed-ref cells filled: {done}/{len(rows)}  -> data/fixed_refs_transfer.csv")
for r in rows: print("  ",r[0],r[1],r[2],f"(n={r[4]})")
