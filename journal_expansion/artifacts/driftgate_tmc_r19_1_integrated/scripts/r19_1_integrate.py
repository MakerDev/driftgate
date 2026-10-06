"""Round 19.1: integration of the Round 17, 18 and 19 results (no new experiment).

Reads only the committed tables and decision files of
  artifacts/driftgate_tmc_r17_featgate   (feature-distance detectors and gate rules, frozen replays)
  artifacts/driftgate_tmc_r18_compare    (audit, same-budget comparison, offloading claims, utility, conditional combination, BTFL)
  artifacts/driftgate_tmc_r19_weights    (fixed device weights, oracles, three-day frozen replay)
and writes topic tables to tables/R19_1_*.csv, the claim table and the source manifest. Every value is copied or
recomputed by simple arithmetic from those files; nothing is recomputed from the records.
"""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ART = HERE.parent
R17, R18, R19 = ART / "driftgate_tmc_r17_featgate", ART / "driftgate_tmc_r18_compare", ART / "driftgate_tmc_r19_weights"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)
USED = []


def rd(path):
    USED.append(path)
    with open(path) as f:
        return list(csv.DictReader(f))


def rj(path):
    USED.append(path)
    return json.loads(path.read_text())


def rows_raw(path):
    USED.append(path)
    with open(path) as f:
        return list(csv.reader(f))


def mean_of(s):
    """'71.95 (1.56)' -> 71.95 ; '' -> None"""
    if s is None or s.strip() == "":
        return None
    m = re.match(r"\s*([+-]?[0-9.]+|nan)", s)
    return float(m.group(1)) if m else None


def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")


# ------------------------------------------------------------------------------------------------ T1 detectors
def t1_detectors():
    au = {(r["setting"], r["detector"]): r for r in rd(R17 / "tables" / "R17_S0_auroc.csv")}
    sup = {(r["setting"], r["score"]): r for r in rd(R18 / "tables" / "R18_detector_matched_support.csv")}
    ut = rd(R18 / "tables" / "R18_action_utility.csv")
    marg = {(r["setting"], r["Round 17 rule"]): r for r in rd(R18 / "tables" / "R18_R17_corrected_margins.csv")}
    where = {"D1": "device (before offloading)", "D2": "device (before offloading)", "D3": "device, head trained at the edge (before offloading)",
             "D4": "edge only (after offloading)", "entropy": "device (before offloading)"}
    rbkey = {"D1": "R-b D1@0.1", "D2": "R-b D2@0.1", "D3": "R-b D3", "D4": "R-b D4"}
    rows = []
    for st, st18 in (("S1", "S1 replay"), ("S2", "S2 replay")):
        for d in ("D1", "D2", "D3", "D4", "entropy"):
            a = au[(st, d)]
            s = sup.get((st18, d))
            sig = "device entropy" if d == "entropy" else d
            u = next((r for r in ut if r["setting"] == st18 and r["signal"] == sig), None)
            m = re.search(r"AUROC \+1 vs -1 ([0-9.]+ \([0-9.]+\))", u["AUROC or disagreement structure"]) if u else None
            mg = marg.get((st18, rbkey.get(d, "")))
            rows.append([st18, d, where[d], a["AUROC all mean (SD)"], a["home"], a["away"], a["score coverage"],
                         s["AUROC Main vs non-Main (D4-defined requests)"] if s else "", m.group(1) if m else "",
                         mg["minus it pp mean (SD)"] if mg else "", mg["strongest of the 17 reference rules"] if mg else ""])
    wcsv("R19_1_T1_detectors.csv",
         ["setting", "score", "where the score can be computed", "AUROC Main vs non-Main, all requests (R17, evaluation seeds)", "home", "away",
          "score coverage", "AUROC on D4-defined requests (R18, all seeds)", "AUROC rescued vs harmed requests (R18, all seeds)",
          "R-b(score) minus strongest of 17 rules pp (R18 audit)", "strongest of 17 rules"], rows)
    return rows


# ------------------------------------------------------------------------------------------------ T2 same budget
def t2_rules():
    rows = []
    raw = rows_raw(R18 / "tables" / "R18_online_same_budget_summary.csv")
    # columns: setting, beta, DriftGate-P, best uncorrected rule, its accuracy, diff, best corrected rule, its accuracy, diff,
    # realized offloading, edge calls per request (the two diff columns share one name, so read by position)
    assert raw[0][3] == "best uncorrected rule" and raw[0][6] == "best corrected rule"
    for v in raw[1:]:
        if v[1] in ("0.5", "1.0"):
            rows.append([v[0], v[1], v[2], v[3], v[5], v[6], v[8], v[9], v[10]])
    wcsv("R19_1_T2_same_budget.csv", ["setting", "beta", "DriftGate-P", "best uncorrected rule", "DriftGate-P minus it pp",
                                      "best corrected fixed-weight rule", "DriftGate-P minus it pp", "realized offloading", "edge calls per request"], rows)
    return rows


# ------------------------------------------------------------------------------------------------ T3 offloading claims
def t3_offload():
    src = rd(R18 / "tables" / "R18_A3_offload_claims.csv")
    by = {}
    for r in src:
        by.setdefault(r["setting"], {})[r["full-offload comparison set"].split(" (")[0]] = r
    rows = []
    for st, d in by.items():
        o, e = d["Round 15 set"], d["with corrected variants"]
        rows.append([st, o["best full-offload rule"], o["its accuracy"], o["DriftGate-P offloading at first grid point reaching it"],
                     o["reduction of offloaded requests vs confidence-based (0.8 nats)"], e["best full-offload rule"], e["its accuracy"],
                     e["DriftGate-P offloading at first grid point reaching it"], e["reduction of offloaded requests vs confidence-based (0.8 nats)"],
                     e["reduction of edge calls vs confidence-based"], o["confidence-based offloading"]])
    wcsv("R19_1_T3_offloading_claims.csv",
         ["setting", "best full-offload rule (Round 15 set, 7 rules)", "its accuracy", "DriftGate-P offloading reaching it",
          "reduction of offloaded requests vs confidence-based", "best full-offload rule (13 rules incl. corrected variants)", "its accuracy",
          "DriftGate-P offloading reaching it", "reduction of offloaded requests vs confidence-based", "reduction of edge calls vs confidence-based",
          "confidence-based offloading at 0.8 nats"], rows)
    return rows


# ------------------------------------------------------------------------------------------------ T4 weights
def t4_weights():
    summ = rd(R19 / "tables" / "R19_summary.csv")
    rows = []
    for r in summ:
        if r["beta"] == "1.0" and r["window"] in ("full", "round>30"):
            rows.append([r["setting"], r["model"], r["window"], r["DriftGate"], r["DriftGate w mean (SD)"], r["best fixed w (seed mean)"],
                         r["w within 0.1 pp"], r["DriftGate minus best fixed pp"], r["DriftGate minus development w 0.45 pp"],
                         r["DriftGate minus initial fixed pp"], r["DriftGate minus corrected edge only pp"],
                         r["device constant minus DriftGate pp (post hoc)"], r["time-bin oracle minus device oracle pp"]])
    wcsv("R19_1_T4_weights.csv", ["setting", "model", "window", "DriftGate", "DriftGate w mean (SD)", "best fixed w (post hoc)",
                                  "w within 0.1 pp of the best", "DriftGate minus best fixed pp", "DriftGate minus development w 0.45 pp",
                                  "DriftGate minus label-free w fixed after 128 requests pp", "DriftGate minus corrected edge only (w = 0) pp",
                                  "device constant minus DriftGate pp (post hoc)", "time-bin oracle minus device oracle pp (post hoc)"], rows)
    tb = [[r["setting"], r["time bin"], r["evaluation rounds"], f"{r['w within 0.1 pp: min']}-{r['max']}", r["DriftGate w mean"],
           r["DriftGate minus best fixed in the bin pp"]]
          for r in rd(R19 / "tables" / "R19_time_bins.csv") if r["beta"] == "1.0" and r["setting"] in ("S1", "S2", "S1 replay", "S2 replay")]
    wcsv("R19_1_T4b_time_bins.csv", ["setting", "time bin", "evaluation rounds", "w within 0.1 pp of the best in the bin (post hoc)",
                                     "DriftGate w mean", "DriftGate minus best fixed in the bin pp (post hoc)"], tb)
    raw = rows_raw(R19 / "tables" / "R19_multiday_table.csv")
    h = raw[0]
    ix = {k: h.index(k) for k in h}
    md = []
    for v in raw[1:]:
        if v[ix["window"]] in ("three days", "days 2-3", "transition", "home", "away"):
            md.append([v[ix["scenario"]], v[ix["beta"]], v[ix["window"]], v[ix["DriftGate"]],
                       v[ix["DriftGate minus development w 0.45 pp"]], v[ix["DriftGate minus initial fixed pp"]],
                       v[ix["seeds with DriftGate higher than initial fixed"]], v[ix["DriftGate minus corrected edge only pp"]],
                       v[ix["best fixed w (post hoc, seed mean)"]]])
    wcsv("R19_1_T5_three_day_replay.csv", ["scenario", "beta", "rows", "DriftGate", "DriftGate minus development w 0.45 pp",
                                           "DriftGate minus label-free fixed w pp", "seeds with DriftGate higher than label-free fixed w",
                                           "DriftGate minus corrected edge only pp", "best fixed w (post hoc)"], md)
    return rows, md


# ------------------------------------------------------------------------------------------------ T6 candidates
def t6_candidates():
    d17 = rj(R17 / "decision_stage0.json")
    marg = {(r["setting"], r["Round 17 rule"]): r for r in rd(R18 / "tables" / "R18_R17_corrected_margins.csv")}
    c18 = rj(R18 / "tables" / "R18_conditional_decision.json")
    bt = {(r["setting"], r["split"]): r for r in rd(R18 / "tables" / "R18_BTFL_adaptation.csv")}
    s19 = rj(R19 / "tables" / "R19_state.json")
    md = rj(R19 / "tables" / "R19_multiday_verdict.json")
    rows = [
        ["R17", "feature-distance gate R-b(D4) (primary candidate chosen on S1 seed 0)",
         f"AUROC max(D1, D3) S1 {d17['per_setting']['S1']['A']:.3f}, S2 {d17['per_setting']['S2']['A']:.3f} (needed 0.80); "
         f"minus strongest of 17 rules S1 {marg[('S1 replay', 'R-b D4')]['minus it pp mean (SD)']}, S2 {marg[('S2 replay', 'R-b D4')]['minus it pp mean (SD)']} (needed +0.5)",
         d17["decision"], "R17 decision_stage0.json; R18_R17_corrected_margins.csv"],
        ["R18", "conditional combination (own-class mass kept at E), adaptive w",
         f"strict rows passing: beta 1 {c18['strict']['1.0']['n_rows'] - c18['strict']['1.0']['n_fail']}/{c18['strict']['1.0']['n_rows']}, "
         f"beta 0.5 {c18['strict']['0.5']['n_rows'] - c18['strict']['0.5']['n_fail']}/{c18['strict']['0.5']['n_rows']}; "
         f"S2 day 1 minus DriftGate {c18['core']['S2 | full @ 1.0']['minus_driftgate']:+.3f}",
         "FAIL (strict) / not a verification candidate", "R18_conditional_decision.json"],
        ["R18", "BTFL inference adaptation (non-shared device blocks)",
         f"minus DriftGate S1 replay {bt[('S1 replay', 'all')]['BTFL - DriftGate pp']}, S2 replay {bt[('S2 replay', 'all')]['BTFL - DriftGate pp']}",
         "below DriftGate", "R18_BTFL_adaptation.csv"],
        ["R19", "time adaptation of the DriftGate weight (state C on day 1)",
         f"state (beta 1 / 0.5): A {'yes' if s19['A']['1.0'] else 'no'}/{'yes' if s19['A']['0.5'] else 'no'}, "
         f"B {'yes' if s19['B']['1.0'] else 'no'}/{'yes' if s19['B']['0.5'] else 'no'}, C {'yes' if s19['C']['1.0'] else 'no'}/{'yes' if s19['C']['0.5'] else 'no'}; three-day frozen replay DriftGate minus label-free fixed w: "
         f"S1 {md['S1 | 1.0 | three days | initial fixed']['mean']:+.3f} ({md['S1 | 1.0 | three days | initial fixed']['positive_seeds']}/"
         f"{md['S1 | 1.0 | three days | initial fixed']['n']} seeds positive), S2 {md['S2 | 1.0 | three days | initial fixed']['mean']:+.3f} "
         f"({md['S2 | 1.0 | three days | initial fixed']['positive_seeds']}/{md['S2 | 1.0 | three days | initial fixed']['n']})",
         "no improvement in frozen multi-day replay; no time-adaptation claim", "R19_state.json; R19_multiday_verdict.json"],
    ]
    wcsv("R19_1_T6_candidates.csv", ["round", "candidate", "key numbers", "verdict", "source"], rows)
    return rows


# ------------------------------------------------------------------------------------------------ T7 Main share
def t7_main_share():
    rows = []
    for r in rd(R18 / "tables" / "R18_main_share_sensitivity.csv"):
        if r["setting"] in ("S1", "S2", "S1 replay", "S2 replay", "ResNet-18") and r["rule"] in ("DriftGate-P", "Corrected edge only", "Probability average"):
            rows.append([r["setting"], r["rule"]] + [r[f"s={s}"] for s in ("0.3", "0.5", "0.65", "0.8", "0.9")] +
                        [r[f"minus DriftGate s={s}"] for s in ("0.3", "0.5", "0.65", "0.8", "0.9")])
    wcsv("R19_1_T7_main_share.csv", ["setting", "rule"] + [f"accuracy s={s}" for s in ("0.3", "0.5", "0.65", "0.8", "0.9")] +
         [f"minus DriftGate s={s}" for s in ("0.3", "0.5", "0.65", "0.8", "0.9")], rows)


# ------------------------------------------------------------------------------------------------ claims
def claims(t2, t3, t4, md):
    t2v = {(r[0], r[1]): r for r in t2}
    unc = [mean_of(t2v[(s, b)][4]) for s in ("S1", "S2", "S1-fast", "partial participation", "stepwise change", "random mobility", "CIFAR-100")
           for b in ("0.5", "1.0")]
    cor = [mean_of(r[6]) for r in t2]
    t3v = {r[0]: r for r in t3}
    day1 = [r for r in t3 if "replay" not in r[0]]
    orig = [r for r in day1 if r[3] not in ("not reached", "")]
    ext = [r for r in day1 if r[7] not in ("not reached", "")]
    orig53 = [r for r in orig if float(r[3]) <= 0.53]
    t4v = {(r[0], r[2]): r for r in t4}
    late = [mean_of(t4v[(s, "round>30")][10]) for s in ("S1", "S1-fast", "stepwise change", "random mobility", "K=200", "K=500")]
    mdv = {(r[0], r[1], r[2]): r for r in md}
    rows = [
        ["C1", "Correction carries the gain: DriftGate-P minus best uncorrected rule at the same budget (beta 0.5 and 1), seven first-day CNN settings",
         f"{min(unc):+.2f} to {max(unc):+.2f} pp", "R19_1_T2_same_budget.csv (R18)"],
        ["C2", "Adaptive weight adds little: DriftGate-P minus best corrected fixed-weight rule, all settings, beta 0.5 and 1",
         f"{min(cor):+.2f} to {max(cor):+.2f} pp", "R19_1_T2_same_budget.csv (R18)"],
        ["C3", "First-day offloading claim against the Round 15 set (7 full-offload rules): settings reaching the best full-offload accuracy with at most 0.53 offloading",
         f"{len(orig53)} settings at {min(float(r[3]) for r in orig53):.3f}-{max(float(r[3]) for r in orig53):.3f} offloading "
         f"({sum(float(r[3]) <= 0.5 for r in orig53)} at most 0.5); reduction of offloaded requests vs confidence-based "
         f"{min(float(r[4]) for r in orig53):.2f}-{max(float(r[4]) for r in orig53):.2f}; other settings: "
         + "; ".join(f"{r[0]} {r[3]}" for r in day1 if r not in orig53), "R19_1_T3_offloading_claims.csv (R18)"],
        ["C4", "First-day offloading claim with corrected variants (13 rules)",
         f"{len(ext)} settings reach at {min(float(r[7]) for r in ext):.3f}-{max(float(r[7]) for r in ext):.3f}; "
         f"reduction {min(float(r[8]) for r in ext):.2f}-{max(float(r[8]) for r in ext):.2f}", "R19_1_T3_offloading_claims.csv (R18)"],
        ["C5", "After round 30 corrected edge only is best in six CNN mobility settings: DriftGate minus it",
         f"{min(late):+.2f} to {max(late):+.2f} pp", "R19_1_T4_weights.csv (R19)"],
        ["C6", "S1 day 1 full: DriftGate minus corrected edge only (decomposition in R19_decomposition.csv)",
         f"{t4v[('S1', 'full')][10]} pp", "R19_1_T4_weights.csv (R19)"],
        ["C7", "Frozen three-day replay: DriftGate minus label-free fixed w (beta 1)",
         f"S1 {mdv[('S1', '1.0', 'three days')][5]} ({mdv[('S1', '1.0', 'three days')][6]} of 5 seeds higher), "
         f"S2 {mdv[('S2', '1.0', 'three days')][5]} ({mdv[('S2', '1.0', 'three days')][6]} of 3)", "R19_1_T5_three_day_replay.csv (R19)"],
        ["C8", "Frozen replay (trained models): DriftGate minus corrected edge only",
         f"S1 {t4v[('S1 replay', 'full')][10]}, S2 {t4v[('S2 replay', 'full')][10]} pp", "R19_1_T4_weights.csv (R19)"],
    ]
    wcsv("R19_1_claims.csv", ["id", "statement", "value", "source"], rows)


def manifest():
    rows = []
    for p in sorted(set(USED)):
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(p)], capture_output=True, text=True, cwd=ART).stdout.strip()
        rows.append([str(p.relative_to(ART)), sha, commit])
    wcsv("R19_1_sources.csv", ["source file (under journal_expansion/artifacts)", "sha256", "last commit"], rows)


def main():
    t1_detectors()
    t2 = t2_rules()
    t3 = t3_offload()
    t4, md = t4_weights()
    t6_candidates()
    t7_main_share()
    claims(t2, t3, t4, md)
    manifest()


if __name__ == "__main__":
    main()
