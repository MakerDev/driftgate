"""Round 16 stage 1: decision_stage1.json from the stage-1 tables and the rule of decision_rule.md section 3.

Reads tables/R16_stage1_summary.json (written by r16_stage1.py run), cache/START_CHECK.json and the route status of
calibration_route.md (section 6, fixed before any result). Writes decision_stage1.json.
"""
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROUTE_STATUS = "A_AVAILABLE_PENDING_APPROVAL"     # calibration_route.md section 6
ROUTE_EXISTS = True                               # route A can be implemented; its data assumption needs the user's approval
MAIN = "S-C"
TABLES = ["basic", "time", "online"]


def main():
    summ = json.loads((HERE / "tables" / "R16_stage1_summary.json").read_text())
    chk = json.loads((HERE / "cache" / "START_CHECK.json").read_text())
    J = summ["judgement"]
    main_j = {t: J[f"{t}|{MAIN}"] for t in TABLES}
    js = [main_j[t]["judgement"] for t in TABLES]
    all_tied = all(main_j[t]["n_negative"] == 0 and abs(main_j[t]["min_difference"]) <= 1e-8 for t in TABLES)
    if not chk["passed"]:
        decision = "BLOCKED_START_CHECK"
    elif all(j == "PASS" for j in js):
        decision = "GO_TO_IMPLEMENTATION_CHECK" if ROUTE_EXISTS else "BLOCKED_CALIBRATION"
    elif any(j == "NO_GO" for j in js):
        decision = "SIMULATION_BELOW_CRITERION"
    elif any(j == "INCOMPLETE" for j in js):
        decision = "INCOMPLETE"
    else:
        decision = "HOLD"
    others = {m: {t: J[f"{t}|{m}"]["judgement"] for t in TABLES if f"{t}|{m}" in J}
              for m in ["S-A", "S-B0", "S-B5", "S-Bg", "DriftGate"]}
    others["DriftGate"]["online"] = J.get("online|DriftGate (DriftGate-P)", {}).get("judgement")
    commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(HERE / "decision_rule.md")],
                            capture_output=True, text=True).stdout.strip()
    out = dict(round=16, stage=1, written=time.strftime("%Y-%m-%d %H:%M:%S"), plan_commit=commit, main_candidate=MAIN,
               start_check=dict(passed=chk["passed"], max_abs_difference=chk["max_abs_difference"], n_values=chk["n_values"]),
               sA_fixed_share=summ["sA"], sA_development_means_pct=summ["sA_dev_means"],
               tables={t: main_j[t] for t in TABLES}, all_rows_tied=all_tied,
               calibration_route=ROUTE_STATUS, stage1_decision=decision,
               note="Simulation decision only (previous-round requests and labels stand in for calibration data); not a paper decision.",
               other_candidates_for_reference=others, supplementary_R12_S1_S2=J.get(f"supplementary|{MAIN}"))
    (HERE / "decision_stage1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
