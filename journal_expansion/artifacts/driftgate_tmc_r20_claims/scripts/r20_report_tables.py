"""Round 20: markdown tables for the final report, read only from tables/R20_*.csv. Writes tables/R20_report_tables.md."""
import csv
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "tables"
rd = lambda n: list(csv.DictReader(open(T / n)))
out = []


def m(v):
    return v.split(" (")[0] if v else ""


def table(hdr, rows):
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")


A = rd("R20_T1_accuracy.csv")
ROWS = ["Confidence-based offloading", "Device only", "Raw edge only", "Probability average", "Logit sum", "No correction + adaptive w",
        "Correction + w 0.5", "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only", "Correction + product", "DriftGate"]


def acc_table(title, cond_sets, block="beta 1"):
    out.append(f"**{title}** (accuracy %, in parentheses DriftGate minus it in pp; same seeds)\n")
    hdr = ["rule"] + [f"{c[:4] if c != 'first day' else ''} {s}".strip() for c, s in cond_sets]
    rows = []
    for ru in ROWS + ["strongest main-table rule"]:
        line = [ru]
        for c, s in cond_sets:
            if ru == "strongest main-table rule":
                x = [r for r in A if r["condition"] == c and r["setting"] == s and r["block"] == block and "[strongest" in r["rule"]]
                line.append(f"{x[0]['rule'].split(' [')[0]}" if x else "")
                continue
            x = [r for r in A if r["condition"] == c and r["setting"] == s and r["block"] == block and r["rule"].split(" [")[0] == ru]
            if not x:
                line.append("NA")
                continue
            d = m(x[0]["minus DriftGate pp mean (SD)"])
            line.append(m(x[0]["accuracy mean (SD)"]) + (f" ({-float(d):+.2f})" if d else ""))
        rows.append(line)
    table(hdr, rows)


acc_table("First day, beta 1", [("first day", s) for s in ["S1", "S2", "S1-fast", "partial participation", "stepwise change", "random mobility", "CIFAR-100", "ResNet-18", "K=200", "K=500"]])
acc_table("Trained models, beta 1", [("round > 30", "S1"), ("round > 30", "S2"), ("round > 30", "random mobility"), ("round > 30", "K=500"),
                                      ("frozen replay", "S1 replay"), ("frozen replay", "S2 replay"), ("three-day frozen replay", "S1 three-day"),
                                      ("three-day frozen replay", "S2 three-day")])
# same-budget gap
G = rd("R20_T2_same_budget_gap.csv")
rows = []
for r in G:
    v = list(r.values())
    rows.append([v[0]] + [f"{m(v[2 + 4 * i])} ({v[1 + 4 * i].replace('Correction + ', 'C+').replace('Corrected edge only', 'CE')})" for i in range(4)]
                + [f"{m(v[4 + 4 * i])}" for i in range(4)] + [v[-2], v[-1]])
out.append("**Same mask: DriftGate minus the best corrected fixed rule / minus the best rule without the prior correction (pp)**\n")
table(["setting", "beta 0.25", "beta 0.5", "beta 0.75", "beta 1", "uncorr. 0.25", "uncorr. 0.5", "uncorr. 0.75", "uncorr. 1", "edge calls per request (0.25/0.5/0.75/1)",
       "two-cell share of offloaded"], rows)
# targets
TG = rd("R20_T2_targets.csv")
CORR = ["Correction + w 0.5", "Correction + w 0.2", "Correction + w 0.45", "Corrected edge only", "Correction + product"]
UNC = ["Probability average", "Logit sum", "No correction + adaptive w", "Raw edge only"]
for tn in "ABC":
    rows = []
    for st in dict.fromkeys(r["setting"] for r in TG):
        X = [r for r in TG if r["setting"] == st and r["target"] == tn]
        if not X:
            continue

        def best(group):
            c = [r for r in X if r["rule"] in group and not r["first grid beta reaching it"].startswith("NA")]
            if not c:
                return "NA"
            b = min(c, key=lambda r: float(r["offloading ratio interpolated between grid points"].split()[0]))
            return f"{b['first grid beta reaching it']} ({b['offloading ratio interpolated between grid points'].split()[0]}, {b['rule'].replace('Correction + ', 'C+').replace('Corrected edge only', 'CE')})"
        dg = [r for r in X if r["rule"] == "DriftGate"][0]
        dgs = "NA" if dg["first grid beta reaching it"].startswith("NA") else f"{dg['first grid beta reaching it']} ({dg['offloading ratio interpolated between grid points'].split()[0]})"
        fx = [r for r in X if r["rule"].startswith("DriftGate-P, fixed")]
        tr = X[0]["target rule"]
        tr = "lowest best of the corrected group: " + tr.split("group (")[1].rstrip(")") if "group (" in tr else tr
        rows.append([st, f"{X[0]['target accuracy']} ({tr})", dgs, best(CORR), best(UNC), fx[0]["offloading ratio there"] if fx else ""])
    out.append(f"**Target {tn}: first grid beta reaching the target (interpolated offloading ratio, rule)**\n")
    table(["setting", "target", "DriftGate", "best corrected fixed rule", "best rule without prior correction", "DriftGate-P fixed-threshold sweep (Round 18 A3)"], rows)
# dev operating points
D = rd("R20_T2_dev_operating_points.csv")
rows = [[r["target"], r["development target accuracy (rule)"].replace("lowest best-over-grid accuracy among the corrected group", "lowest best of the corrected group:"), r["rule"], r["beta chosen on development seeds 5-7"], m(r["evaluation accuracy at that beta, seeds 0-4"]),
         r["offloading ratio"], r["edge calls per request"], r["evaluation target (same definition)"].split(" (")[0], r["evaluation target reached"]] for r in D]
out.append("**S1 operating points chosen on development seeds 5-7, evaluated on seeds 0-4**\n")
table(["target", "development target", "rule", "beta", "evaluation accuracy", "offloading", "edge calls", "evaluation target", "reached"], rows)
# Task 3
C3 = rd("R20_T3_correction.csv")
rows = [[r["setting"], r["beta"], r["rule"], m(r["b = 0 (no correction)"]), m(r["constant a = 0.25"]), m(list(r.values())[5]), m(r["current a_k (DriftGate)"]),
         f"{m(r['current a_k minus b = 0 pp'])} [{r['seeds positive (vs b = 0)']}]", r["primary constant"],
         f"{m(r['current a_k minus primary constant pp'])} [{r['seeds positive (vs primary constant)']}]"] for r in C3]
out.append("**Correction ablation (same mask, same weight series)**\n")
table(["setting", "beta", "rule", "b = 0", "a = 0.25", "a0 = 0.2326", "current a_k", "a_k minus b = 0 [seeds +]", "primary constant", "a_k minus primary constant [seeds +]"], rows)
S3 = rd("R20_T3_a_stats.csv")
rows = [[r["setting"], f"{r['a_k p50']} ({r['a_k p5']}-{r['a_k p95']})", f"{r['b_k p50']} ({r['b_k p5']}-{r['b_k p95']})",
         r["within-device SD of a_k over rounds (mean over devices)"], r["within-device SD (max over devices)"], r["between-device SD of device-mean a_k"],
         r["median a_k at home"], r["median a_k away"]] for r in S3]
out.append("**a_k and b_k**\n")
table(["setting", "a_k median (p5-p95)", "b_k median (p5-p95)", "within-device SD (mean)", "within-device SD (max)", "between-device SD", "median a_k home", "median a_k away"], rows)
(T / "R20_report_tables.md").write_text("\n".join(out))
print("R20_report_tables.md", len(out), "lines")
