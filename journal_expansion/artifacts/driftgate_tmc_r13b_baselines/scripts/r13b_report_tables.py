"""Round 13b: markdown versions of the CSV tables for the Korean report (no computation; values copied from tables/*.csv).

Writes tables/R13b_report_tables.md and, with --assemble DRAFT OUT, replaces every line "<<key>>" of the report draft
by the block of that key (keys: T1, T2, T3, T4, MANIFEST, FAUTO, DEVT).
"""
import csv
import sys
from pathlib import Path

TAB = Path(__file__).resolve().parent.parent / "tables"
RULES = ["B0", "B1", "B2", "B3", "F", "R-PoE", "R-PoE-bal", "R-THE", "R-ZTW", "R-EM", "R-LR", "F-nodebias", "F-eq",
         "F-auto", "F-gate", "kind oracle"]
KO = {"S1": "S1 (출퇴근)", "Schedule A": "단계적 구성 변화", "S2": "S2 (GeoLife)", "participation 0.5": "참여율 0.5",
      "ResNet-20, shallow split": "ResNet-20 얕은 분할", "ResNet-20, middle split": "ResNet-20 중간 분할",
      "VGG-11, shallow split": "VGG-11 얕은 분할", "VGG-11, middle split": "VGG-11 중간 분할",
      "CIFAR-100": "조건 1: CIFAR-100", "CIFAR-100 gradual": "CIFAR-100 gradual", "fixed 0.4": "고정 0.4",
      "S1 default (Round 12)": "S1 기본 CNN (Round 12, seed 0–4)"}
ko = lambda s: KO.get(s, s)
read = lambda n: list(csv.DictReader(open(TAB / n)))
out = []
blocks = {}


def block(key):
    start = len(out)
    return lambda: blocks.__setitem__(key, blocks.get(key, "") + "\n".join(out[start:]) + "\n")


def table(hdr, rows):
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")


end = block("T1")
T1 = read("R13b_T1_rules.csv")
for w, title in (("late", "후반 구간"), ("full", "전체 구간")):
    for part, rules in (("기존 규칙과 F", ["B0", "B1", "B2", "B3", "F", "kind oracle"]),
                        ("관련 연구의 기준선", ["R-PoE", "R-PoE-bal", "R-THE", "R-ZTW", "R-EM", "R-LR"]),
                        ("F의 절제와 보완", ["F", "F-nodebias", "F-eq", "F-auto", "F-gate"])):
        out.append(f"**표 1, {title}: {part}**\n")
        table(["시나리오", "학습", "seed 수"] + rules,
              [[ko(r["scenario"]) + (" (B부)" if r["part"] == "part B" else ""), ko(r["training"]), r["seeds"]]
               + [r[f"{x}_{w}"] for x in rules] for r in T1])

end()
end = block("T2")
T2 = read("R13b_T2_splits.csv")
for g in dict.fromkeys((r["scenario"], r["training"]) for r in T2):
    out.append(f"**표 2: {ko(g[0])} ({ko(g[1])})**\n")
    table(["규칙", "집", "밖", "Main", "OOP", "OOR"],
          [[r["rule"], r["home_pct"], r["away_pct"], r["Main_pct"], r["OOP_pct"], r["OOR_pct"]] for r in T2
           if (r["scenario"], r["training"]) == g])

end()
end = block("T3")
T3 = read("R13b_T3_curves.csv")
opts = [f"{0.1 * i:.1f}" for i in range(1, 11)]
out.append("**표 3: 곡선 (후반 구간)**\n")
table(["시나리오", "곡선"] + opts,
      [[ko(r["scenario"]), r["curve"]] + [r[f"acc_at_server_use_{o}"] for o in opts] for r in T3
       if r["window"] == "late" and r["training"] == "fixed 0.4"
       and r["scenario"] in ("S1", "S2")])
R = read("R13b_T3_SF_ratios.csv")
out.append("**표 3, 지시문 5절의 두 비율 (후반 구간, seed 평균)**\n")
table(["시나리오", "B0 정확도", "SF가 B0의 정확도에 처음 닿는 server 사용", "곡선 E의 최고 정확도", "SF가 E의 최고에 처음 닿는 server 사용"],
      [[ko(r["scenario"]), r["B0_late_pct"], r["SF_first_server_use_at_B0_acc"], r["E_max_late_pct"], r["SF_first_server_use_at_E_max"]]
       for r in R if r["seed"] == "mean"])

end()
end = block("T4")
T4 = read("R13b_T4_structures.csv")
out.append("**표 4: 구조, 조건별 크기와 시간**\n")
mega = lambda v: f"{int(v):,}" if v not in ("", None) else ""
table(["묶음", "client block parameter", "client exit parameter", "server block과 exit parameter",
       "client block FLOPs", "client exit FLOPs", "server block과 exit FLOPs", "smashed data 모양",
       "smashed data (byte, float32)", "학습 시간 (분)", "평가 시간 (분)", "클라이언트당 Main class", "a 평균"],
      [[ko(r["group"]), mega(r["client_block_params"]), mega(r["client_exit_params"]), mega(r["server_block_and_exit_params"]),
        mega(r["client_block_flops_per_image"]), mega(r["client_exit_flops_per_image"]),
        mega(r["server_block_and_exit_flops_per_image"]), r["smashed_shape"], r["smashed_bytes_float32"],
        r["train_min_mean"], r["eval_min_mean"], r["Main_classes_per_client_mean"], r["a_mean"]] for r in T4])

end()
end = block("MANIFEST")
M = [r for r in read("R13b_T0_checks_manifest.csv") if r["part"] == "part B"]
out.append("**B부 run의 시간과 요청별 원본의 sha256**\n")
table(["run", "run_id", "학습 (분)", "평가 (분)", "전체 (분)", "크기 (MB)", "sha256"],
      [[r["run"], r["run_id"], r["train_min"], r["eval_min"], r["total_min"], f"{int(r['evalprobs_bytes']) / 1e6:.1f}",
        r["evalprobs_sha256"]] for r in M])
end()
end = block("FAUTO")
W_ = read("R13b_Fauto_w.csv")
out.append("**F-auto의 w (요청 전체의 평균, 후반 구간의 평균, 최솟값, 최댓값)**\n")
table(["run", "평균", "후반 평균", "최소", "최대"],
      [[r["run"], r["w_mean_all_requests"], r["w_mean_late"], r["w_min"], r["w_max"]] for r in W_])
end()
end = block("DEVT")
D = read("R13b_dev_Fgate_t.csv")
out.append("**개발용 기록에서 F-gate의 t 고르기 (후반 구간, seed 5–7)**\n")
table(["t", "seed 5", "seed 6", "seed 7", "평균", "선택"],
      [[r["t"], r["late_pct_seed5"], r["late_pct_seed6"], r["late_pct_seed7"], r["late_pct_mean"], r["chosen"]] for r in D])
end()
(TAB / "R13b_report_tables.md").write_text("\n".join(out))
print("written", TAB / "R13b_report_tables.md")
if len(sys.argv) == 4 and sys.argv[1] == "--assemble":
    lines = []
    for line in open(sys.argv[2]).read().split("\n"):
        key = line.strip()[2:-2] if line.strip().startswith("<<") and line.strip().endswith(">>") else None
        lines.append(blocks[key].rstrip("\n") if key else line)
    Path(sys.argv[3]).write_text("\n".join(lines))
    print("assembled", sys.argv[3])
