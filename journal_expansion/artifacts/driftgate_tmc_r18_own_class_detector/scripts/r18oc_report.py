"""Round 18 (own-class detector): markdown tables for the reports, built from the CSVs in tables/ and the run JSONs.

  python .../r18oc_report.py stage2   > (stdout) markdown blocks used in report_stage2.md
  python .../r18oc_report.py stage3   > (stdout) markdown blocks used in report_stage3.md
  python .../r18oc_report.py times <name prefix> ...   # run wall times (training and replay) as a markdown table

Only tables and numbers; the surrounding Korean text is written in the report itself.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TABLES = HERE / "tables"
RUNS = HERE.parents[1] / "runs" / "phaseT18oc"


def rows(name):
    r = list(csv.reader(open(TABLES / name)))
    return r[0], r[1:]


def md(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(x.replace("|", "\\|") for x in r) + " |" for r in body]
    return "\n".join(out)


def pick(name, cols, where=None):
    h, b = rows(name)
    ix = [h.index(c) for c in cols]
    sel = [r for r in b if where is None or where(dict(zip(h, r)))]
    return md(cols, [[r[i] for i in ix] for r in sel])


def times(prefixes):
    out = []
    for f in sorted(RUNS.glob("r18oc_*.json")):
        nm = f.stem
        if not any(nm.startswith(p) for p in prefixes):
            continue
        h = json.load(open(f))
        out.append([nm, "replay" if "_replay_" in nm else "training", f"{h['total_time_sec'] / 3600:.2f}"])
    return md(["run", "kind", "wall time (h)"], out)


def stage2():
    print("### cells\n")
    print(pick("R18oc_stage2_cells.csv", ["setting", "model config", "condition", "detector variant", "A* (%)", "ratio-at-target new",
                                          "ratio-at-target comparator envelope", "S", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)",
                                          "B(0.5) rule", "oracle S", "oracle G(0.5) pp", "room", "judgement", "border"]))
    print("\n### accuracy\n")
    print(pick("R18oc_stage2_cells.csv", ["setting", "model config", "condition", "new accuracy at 0.25", "new accuracy at 0.5",
                                          "new accuracy at 0.75", "B(0.25)", "B(0.5)", "B(0.75)", "B(1.0) rule"]))
    print("\n### auroc (all requests)\n")
    print(pick("R18oc_stage2_auroc.csv", ["setting", "model config", "condition", "score", "AUROC all", "AUROC home", "AUROC away"],
               lambda d: "@" not in d["score"]))
    print("\n### reference rows\n")
    print(pick("R18oc_stage2_reference.csv", ["setting", "model config", "condition", "rule", "answer rule", "accuracy mean (SD)",
                                              "offloading ratio"],
               lambda d: d["rule"] in ("Device only", "Edge only", "Confidence-based offloading")
               or d["answer rule"] == "average-corrected-even"))
    print("\n### extras (main variant)\n")
    print(pick("R18oc_stage2_extra.csv", ["setting", "model config", "condition", "detector column", "operating point",
                                          "accuracy mean (SD)", "offloading ratio", "s-hat error all", "s-hat error home",
                                          "s-hat error away"]))
    print("\n### held-out effect\n")
    print(pick("R18oc_holdout_effect.csv", rows("R18oc_holdout_effect.csv")[0]))
    for tag in ("cnn", "ft", "fz"):
        print(f"\n### design selection {tag} (frozen-replay)\n")
        print(pick(f"R18oc_design_selection_{tag}.csv", ["variant", "accuracy at ratio 0.5 (detector rule) mean (SD)",
                                                        "per seed (8, 9, 10)", "AUROC (all requests)"],
                   lambda d: d["condition"] == "frozen-replay"))


def stage3():
    print("### cells\n")
    print(pick("R18oc_stage3_cells.csv", ["setting", "model config", "condition", "ratio-at-target new",
                                          "ratio-at-target comparator envelope", "S", "new accuracy at 0.25", "new accuracy at 0.5",
                                          "new accuracy at 0.75", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)", "B(0.5) rule",
                                          "oracle S", "room", "judgement"]))
    for name, title in (("R18oc_stage3_auroc.csv", "auroc"), ("R18oc_stage3_breakdown.csv", "breakdown"),
                        ("R18oc_stage3_controller.csv", "controller"), ("R18oc_stage3_scale.csv", "scale"),
                        ("R18oc_stage3_ablation.csv", "ablation"), ("R18oc_stage3_cost.csv", "cost")):
        if (TABLES / name).exists():
            print(f"\n### {title}\n")
            print(pick(name, rows(name)[0]))


if __name__ == "__main__":
    {"stage2": stage2, "stage3": stage3}.get(sys.argv[1], lambda: print(times(sys.argv[2:])))()
