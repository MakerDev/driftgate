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
TAGS = {"cnn-scratch": "cnn", "pretrained-finetuned-block": "ft", "pretrained-frozen-block": "fz"}


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


def _d(name):
    h, b = rows(name)
    return [dict(zip(h, r)) for r in b]


def _num(x):
    return x.split(" (")[0] if x else ""


def stage2_summary():
    """compact pivots of the stage-2 CSVs for report_stage2.md."""
    cells = _d("R18oc_stage2_cells.csv")
    print("### cells (compact)\n")
    print(md(["setting", "model config", "condition", "S", "G(0.25)", "G(0.5)", "new at 0.5", "B(0.5)", "oracle S", "room", "judgement"],
             [[c["setting"], c["model config"], c["condition"], c["S"], c["G(0.25) pp mean (SD)"], c["G(0.5) pp mean (SD)"],
               c["new accuracy at 0.5"], c["B(0.5)"], c["oracle S"], c["room"], c["judgement"]] for c in cells]))
    au = _d("R18oc_stage2_auroc.csv")
    key = lambda d: (d["setting"], d["model config"], d["condition"])
    groups = list(dict.fromkeys(key(d) for d in au))
    sc = ["exit|mlp64|both", "device-entropy", "device-max-prob", "device-energy", "own-mahalanobis", "own-knn"]
    print("\n### AUROC, all requests (compact)\n")
    print(md(["setting", "model config", "condition"] + sc,
             [list(g) + [next((_num(d["AUROC all"]) for d in au if key(d) == g and d["score"] == s), "") for s in sc] for g in groups]))
    ref = _d("R18oc_stage2_reference.csv")
    print("\n### reference rows (compact; accuracy %, ratio)\n")
    out = []
    for g in groups:
        R = [d for d in ref if key(d) == g]
        get = lambda rule, ans="": next((d for d in R if d["rule"] == rule and d["answer rule"] == ans), None)
        cb, v32 = get("Confidence-based offloading"), get("entropy controller beta 0.5", "average-corrected-even")
        out.append(list(g) + [_num(get("Device only")["accuracy mean (SD)"]), _num(get("Edge only")["accuracy mean (SD)"]),
                              f"{_num(cb['accuracy mean (SD)'])} ({cb['offloading ratio']})",
                              f"{_num(v32['accuracy mean (SD)'])} ({v32['offloading ratio']})"])
    print(md(["setting", "model config", "condition", "Device only", "Edge only", "Confidence-based offloading (ratio)",
              "v32 method, entropy controller 0.5 (ratio)"], out))
    ex = _d("R18oc_stage2_extra.csv")
    cur = _d("R18oc_stage2_curves.csv")
    print("\n### extras and schedules of the chosen variant (accuracy %, ratio)\n")
    out = []
    for g in groups:
        E = {(d["detector column"], d["operating point"]): d for d in ex if key(d) == g}
        mv = next(d["detector column"] for d in ex if key(d) == g and "@" not in d["detector column"])
        C = {d["score or detector"]: d["ratio 0.50"] for d in cur if key(d) == g and d["answer rule"] == "edge-corrected-by-detector"}
        f = lambda col, op: f"{_num(E[(col, op)]['accuracy mean (SD)'])} ({E[(col, op)]['offloading ratio']})"
        out.append(list(g) + [C[mv], C[mv + "@frozen-at-home"], C[mv + "@every-10"], f(mv, "fixed 0.5"), f(mv, "controller 0.25"),
                              f(mv, "controller 0.5"), E[(mv, "curve 0.5")]["s-hat error all"]])
    print(md(["setting", "model config", "condition", "curve 0.5 (every round)", "curve 0.5 (frozen-at-home)", "curve 0.5 (every-10)",
              "fixed threshold 0.5", "controller 0.25", "controller 0.5", "s-hat error at curve 0.5"], out))
    dg = _d("R18oc_stage2_diagnosis_rules.csv")
    pairs = [("detector", "edge-corrected-by-detector"), ("detector", "edge-raw"), ("detector", "edge-corrected-even"),
             ("detector", "edge-corrected-by-detector + device 0.2"), ("device-max-prob", "edge-raw"),
             ("device-max-prob", "edge-corrected-even"), ("device-max-prob", "average-corrected-even"), ("device-max-prob", "logit-sum"),
             ("oracle", "edge-raw"), ("oracle", "edge-corrected-even"), ("envelope B", "max over the 25 comparators")]
    print("\n### diagnosis: accuracy at ratio 0.5 by ranking and answer rule (pretrained, S1 and S2)\n")
    gs = [g for g in dict.fromkeys(key(d) for d in dg) if g[0] in ("S1", "S2") and g[2] != "first-day"]
    out = []
    for rk, ans in pairs:
        out.append([rk, ans] + [_num(next(d["accuracy at 0.5 mean (SD)"] for d in dg if key(d) == g and d["ranking"] == rk
                                          and d["answer rule"] == ans)) for g in gs])
    print(md(["ranking", "answer rule"] + [f"{g[0]} {TAGS[g[1]]} {g[2]}" for g in gs], out))
    dk = _d("R18oc_stage2_diagnosis_kind.csv")
    print("\n### diagnosis: accuracy by request kind, frozen-replay (Main / OOP / OOR, %)\n")
    out = []
    for d in dk:
        if d["condition"] != "frozen-replay" or (d["ranking"], d["answer rule"]) not in pairs:
            continue
        out.append([d["setting"], d["model config"], d["ranking"], d["answer rule"]]
                   + [" / ".join(d[f"{k} accuracy at {r}"] for k in ("Main", "OOP", "OOR")) for r in (0.25, 0.5, 0.75)])
    print(md(["setting", "model config", "ranking", "answer rule", "ratio 0.25", "ratio 0.5", "ratio 0.75"], out))
    print("\n### diagnosis: request shares and r\n")
    print(pick("R18oc_stage2_diagnosis_r.csv", rows("R18oc_stage2_diagnosis_r.csv")[0]))


def times_summary(prefixes):
    """mean and range of the run wall times per (model configuration, setting, kind)."""
    G = {}
    for f in sorted(RUNS.glob("r18oc_*.json")):
        nm = f.stem
        if not any(nm.startswith(p) for p in prefixes):
            continue
        parts = nm.split("_")
        kind = "replay" if "replay" in parts else "training"
        seed = int(parts[-1][1:])
        G.setdefault((parts[1], parts[2], kind, "development" if seed >= 8 else "evaluation"), []).append(
            json.load(open(f))["total_time_sec"] / 3600)
    out = [[*k, str(len(v)), f"{sum(v) / len(v):.2f}", f"{min(v):.2f}-{max(v):.2f}"] for k, v in sorted(G.items())]
    return md(["config tag", "setting", "kind", "seeds", "runs", "mean wall time (h)", "range (h)"], out)


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
    {"stage2": stage2, "stage2_summary": stage2_summary, "stage3": stage3,
     "times_summary": lambda: print(times_summary(sys.argv[2:]))}.get(sys.argv[1], lambda: print(times(sys.argv[2:])))()
