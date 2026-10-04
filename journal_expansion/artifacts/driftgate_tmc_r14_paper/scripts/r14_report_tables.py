"""Round 14: markdown versions of tables 1, 4 and 5 for the Korean report (no computation; values copied from the CSVs).

python r14_report_tables.py --assemble DRAFT OUT replaces every line "<<T1>>", "<<T4>>", "<<T5>>" of DRAFT.
"""
import csv
import sys
from pathlib import Path

TAB = Path(__file__).resolve().parent.parent / "tables"
KO = {"S1": "S1", "S2": "S2", "S1-fast": "S1-fast", "participation 0.5": "참여율 0.5", "stepwise change": "단계적 구성 변화",
      "client mobility": "client mobility", "CIFAR-100": "CIFAR-100", "ResNet-18": "ResNet-18", "K=200": "K = 200",
      "K=500": "K = 500"}
read = lambda n: list(csv.DictReader(open(TAB / n)))
blocks = {}


def table(hdr, rows):
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


T1 = read("R14_T1_main.csv")
cols = [c for c in T1[0] if c.endswith("mean (SD)")]
half = (len(cols) + 1) // 2
parts = []
for cs in (cols[:half], cols[half:]):
    parts.append(table(["규칙 (원고의 이름)"] + [KO[c.replace(" mean (SD)", "")] for c in cs],
                       [[r["rule (paper)"] + (f" ({r['rule']})" if r["rule"] and r["rule"] != "DriftGate" else "")] + [r[c] for c in cs] for r in T1]))
blocks["T1"] = "\n\n".join(parts)
T4 = read("R14_T4_curves.csv")
opts = [c for c in T4[0] if c.startswith("server_use_")]
blocks["T4"] = table(["설정", "곡선"] + [c.replace("server_use_", "") for c in opts],
                     [[KO[r["setting"]], r["curve"]] + [r[c] for c in opts] for r in T4 if r["setting"] in ("S1", "S2")])
RE = read("R14_T4_reach.csv")
blocks["T4"] += "\n\n" + table(["설정", "B0 (SplitGP)의 server 사용 비율", "DriftGate 곡선이 B0의 정확도에 처음 닿는 server 사용 비율", "seed별"],
                               [[KO[r["setting"]], r["B0_server_use mean (SD)"],
                                 r["DriftGate_curve_first_server_use_at_B0_accuracy mean (SD)"], r["per_seed"]] for r in RE])
T5 = read("R14_T5_bottom10.csv")
blocks["T5"] = table(["설정"] + [c.replace(" mean (SD)", "") for c in list(T5[0])[1:]],
                     [[KO[r["setting"]]] + [r[c] for c in list(T5[0])[1:]] for r in T5])
if len(sys.argv) == 4 and sys.argv[1] == "--assemble":
    out = []
    for line in open(sys.argv[2]).read().split("\n"):
        s = line.strip()
        out.append(blocks[s[2:-2]] if s.startswith("<<") and s.endswith(">>") else line)
    Path(sys.argv[3]).write_text("\n".join(out))
    print("assembled", sys.argv[3])
