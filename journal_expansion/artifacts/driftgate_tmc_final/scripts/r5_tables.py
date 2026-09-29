"""Round 5 — every table of the final package, from RAW run JSON only.
raw JSON -> this script -> tables/*.csv (-> figures, report).
DriftGate = the relonly controller (Round-4 B1 flags, no absolute branch).
integrated accuracy = mean of acc_total over the run's evaluation rounds.
Paired differences use matched seeds only (Student-t 95% CI, n_pos). SD uses ddof=1.
Re-runnable at any time; arms whose runs are missing are left empty (never imputed).
"""
import csv, glob, json, math, re, sys
from pathlib import Path
import numpy as np
from scipy import stats

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
RUNS = JR / "runs"
HERE = Path(__file__).resolve().parent.parent
TAB = HERE / "tables"; TAB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR)); sys.path.insert(0, str(JR.parent))
from src.evaluation.signal_metrics import evaluate_signal

# ----------------------------------------------------------------------------- helpers
_cache = {}
def load(p):
    if p not in _cache: _cache[p] = json.load(open(p))
    return _cache[p]
def famf(pat):
    d = {}
    for f in sorted(glob.glob(str(RUNS / pat))):
        m = re.search(r"_s(\d+)\.json$", f)
        if m: d[int(m.group(1))] = f
    return d
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def ifam(pat): return {s: integ(load(f)) for s, f in famf(pat).items()}
def names(pat): return [Path(f).stem for f in famf(pat).values()]
def ci95(d):
    d = np.asarray(d, float)
    if len(d) < 2: return (float("nan"), float("nan"))
    h = stats.sem(d) * stats.t.ppf(0.975, len(d) - 1); return (d.mean() - h, d.mean() + h)
def paired(a, b):
    """a - b on matched seeds. Returns dict(mean, lo, hi, n_pos, n, seeds) in pp."""
    s = sorted(set(a) & set(b))
    if not s: return None
    d = np.array([a[x] - b[x] for x in s]) * 100; lo, hi = ci95(d)
    return dict(mean=d.mean(), lo=lo, hi=hi, n_pos=int((d > 0).sum()), n=len(s), seeds=s, per_seed=d)
def msd(v):
    v = np.array(list(v.values()), float) * 100
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan"), len(v))
def f4(x): return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
def fci(p): return "" if p is None or math.isnan(p["lo"]) else f"[{p['lo']:+.2f}, {p['hi']:+.2f}]"
def wcsv(name, hdr, rows):
    with open(TAB / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")

# ----------------------------------------------------------------------------- registry
SNAME = {"A": "stepwise composition change", "mob": "client mobility", "c10gsig": "CIFAR-10 gradual",
         "c100gsig": "CIFAR-100 gradual", "tiny": "Tiny-ImageNet", "svhn": "SVHN temporal",
         "resA": "ResNet-18 middle split (stepwise)"}
MAIN = {
 "A": {
  "DriftGate": "phaseT4_B1_views/b1_relonly_A_s*.json",
  "fixed 0.2": "phaseT2r2_signal/r2_A_fx20_s*.json",
  "fixed 0.3": "phaseT1_disjoint/t1_A_fx30_s*.json",
  "fixed 0.4": "phaseT1_disjoint/t1_A_fx40_s*.json",
  "fixed 0.5": "phaseT1_disjoint/t1_A_fx50_s*.json",
  "entropy": "phaseT1_disjoint/t1_A_ent_s*.json",
  "APFL η=0.01": "phaseT4_B4_apfl/b4_apfl_eta001_A_s*.json",
  "APFL η=0.1": "phaseT4_B4_apfl/b4_apfl_eta010_A_s*.json",
  "absonly": "phaseT4_B1_views/b1_absonly_A_s*.json",
 },
 "mob": {
  "DriftGate": "phaseT4_B1_views/b1_relonly_mob_s*.json",
  "fixed 0.2": "phaseT2r2_signal/r2_mob_fx20_s*.json",
  "fixed 0.4": "phaseT1_disjoint/t1_mob_fx40_s*.json",
  "fixed 0.5": "phaseT1_disjoint/t1_mob_fx50_s*.json",   # Round-1 run (a later duplicate b3_mob_fx50 is not used)
  "fixed 0.6": "phaseT4_B3_fixedgrid/b3_mob_fx60_s*.json",
  "entropy": "phaseT1_disjoint/t1_mob_ent_s*.json",
  "APFL η=0.01": "phaseT4_B4_apfl/b4_apfl_eta001_mob_s*.json",
  "APFL η=0.1": "phaseT4_B4_apfl/b4_apfl_eta010_mob_s*.json",
  "absonly": "phaseT4_B1_views/b1_absonly_mob_s*.json",
 },
}
TRANSFER = {
 "c10gsig": {"DriftGate": "phaseT5_E1_relonly_transfer/e1_relonly_c10gsig_s*.json",
             "fixed 0.15": "phaseT4_B3_fixedgrid/b3_c10gsig_fx15_s*.json",
             "fixed 0.2": "phaseT3_fixedref/t3_c10gsig_fx20_s*.json",
             "fixed 0.4": "phaseT3_fixedref/t3_c10gsig_fx40_s*.json"},
 "c100gsig": {"DriftGate": "phaseT4_B1_views/b1_relonly_c100gsig_s*.json",
              "fixed 0.15": "phaseT4_B3_fixedgrid/b3_c100gsig_fx15_s*.json",
              "fixed 0.2": "phaseT3_fixedref/t3_c100gsig_fx20_s*.json",
              "fixed 0.4": "phaseT3_fixedref/t3_c100gsig_fx40_s*.json"},
 "tiny": {"DriftGate": "phaseT5_E1_relonly_transfer/e1_relonly_tiny_s*.json",
          "fixed 0.15": "phaseT4_B3_fixedgrid/b3_tiny_fx15_s*.json",
          "fixed 0.2": "phaseT3_fixedref/t3_tiny_fx20_s*.json",
          "fixed 0.4": "phaseT3_fixedref/t3_tiny_fx40_s*.json"},
 "svhn": {"DriftGate": "phaseT5_E1_relonly_transfer/e1_relonly_svhn_s*.json",
          "fixed 0.15": "phaseT4_B3_fixedgrid/b3_svhn_fx15_s*.json",
          "fixed 0.2": "phaseT2_signal/t2_svhn_fx20_s*.json",
          "fixed 0.4": "phaseT4_B3_fixedgrid/b3_svhn_fx40_s*.json",
          "entropy": "phaseT2_signal/t2_svhn_ent_s*.json"},
}
RESNET = {"DriftGate": "phaseT5_E2_resnet/e2_res_relonly_A_s*.json",
          "fixed 0.2": "phaseT5_E2_resnet/e2_res_fx20_A_s*.json",
          "fixed 0.4": "phaseT5_E2_resnet/e2_res_fx40_A_s*.json"}
E3 = {"A": {"guard 0.25": "phaseT5_E3_const/e3_guard025_A_s*.json", "guard 1.0": "phaseT5_E3_const/e3_guard100_A_s*.json",
            "λ_max 0.60": "phaseT5_E3_const/e3_lmax060_A_s*.json", "λ_max 0.80": "phaseT5_E3_const/e3_lmax080_A_s*.json"},
      "mob": {"λ_max 0.60": "phaseT5_E3_const/e3_lmax060_mob_s*.json", "λ_max 0.80": "phaseT5_E3_const/e3_lmax080_mob_s*.json"}}
SEG = {"A": [("ρ=0 early", [1, 10, 20, 30]), ("ρ=0.4 rising", [40, 50, 60]), ("ρ=0.8", [70, 80, 90]),
             ("ρ=0.4 falling", [100, 110, 120]), ("ρ=0 late", [130, 140, 150])],
       "mob": [("ρ=0 (R1–60)", [1, 10, 20, 30, 40, 50, 60]), ("ρ=0.8 (R70–120)", [70, 80, 90, 100, 110, 120])]}
FIXED_KEYS = lambda d: [k for k in d if k.startswith("fixed")]
PN = []   # paper_numbers rows: item, value, ci, n_seeds, source runs
def pn(item, value, ci="", n="", src=""): PN.append([item, value, ci, n, src])
def src_of(*pats): return "; ".join(f"{Path(p).name.replace('_s*.json', '')}_s{{{','.join(str(s) for s in sorted(famf(p)))}}}" for p in pats if famf(p))

print("Round-5 tables")
# ============================================================ T1 main results (A, mobility)
rows = []
for st, arms in MAIN.items():
    ref = ifam(arms["DriftGate"])
    fk = FIXED_KEYS(arms); fv = {k: ifam(arms[k]) for k in fk}
    best = max((k for k in fk if fv[k]), key=lambda k: np.mean(list(fv[k].values())), default=None)
    for m, pat in arms.items():
        v = ifam(pat)
        if not v:
            rows.append([SNAME[st], m, 0, "", "", "", "", "", "", "missing"]); continue
        mean, sd, n = msd(v)
        p = None if m == "DriftGate" else paired(ref, v)
        rows.append([SNAME[st], m + (" (best fixed)" if m == best else ""), n, f4(mean), f4(sd),
                     "" if p is None else f"{p['mean']:+.4f}", fci(p), "" if p is None else p["n_pos"],
                     "" if p is None else p["n"], src_of(pat)])
        if p is not None:
            pn(f"{SNAME[st]}: DriftGate − {m}", f"{p['mean']:+.3f} pp", fci(p), p["n"], src_of(arms["DriftGate"], pat))
    mean, sd, n = msd(ref) if ref else (float("nan"),) * 3
    pn(f"{SNAME[st]}: DriftGate integrated accuracy", f"{mean:.2f} % (SD {sd:.2f})", "", n, src_of(arms["DriftGate"]))
wcsv("T1_main_results.csv", ["setting", "method", "n_seeds", "integrated_acc_pct", "sd_pct",
     "DriftGate_minus_method_pp", "ci95", "n_pos", "n_matched", "source_runs"], rows)

# absonly on CIFAR-100 gradual too (controller check table)
rows = []
for st, dg, ab in (("A", MAIN["A"]["DriftGate"], MAIN["A"]["absonly"]), ("mob", MAIN["mob"]["DriftGate"], MAIN["mob"]["absonly"]),
                   ("c100gsig", TRANSFER["c100gsig"]["DriftGate"], "phaseT4_B1_views/b1_absonly_c100gsig_s*.json")):
    a, b = ifam(dg), ifam(ab); p = paired(a, b)
    if p is None: continue
    ma, _, _ = msd({s: a[s] for s in p["seeds"]}); mb, _, _ = msd({s: b[s] for s in p["seeds"]})
    rows.append([SNAME[st], "DriftGate vs absonly (λ from TV level)", f4(ma), f4(mb), f"{p['mean']:+.4f}", fci(p), p["n_pos"], p["n"]])
    pn(f"{SNAME[st]}: DriftGate − absonly", f"{p['mean']:+.3f} pp", fci(p), p["n"], src_of(dg, ab))
wcsv("T4a_controller_relonly_vs_absonly.csv", ["setting", "comparison", "DriftGate_pct", "absonly_pct",
     "DriftGate_minus_absonly_pp", "ci95", "n_pos", "n"], rows)

# ============================================================ T4b constant sensitivity (E3)
rows = []
for st, arms in E3.items():
    base = ifam(MAIN[st]["DriftGate"])
    fk = FIXED_KEYS(MAIN[st]); fv = {k: ifam(MAIN[st][k]) for k in fk}
    for lbl, pat in arms.items():
        v = ifam(pat)
        if not v: rows.append([SNAME[st], lbl, 0] + [""] * 9); continue
        p = paired(v, base); seeds = p["seeds"]
        bmean = {k: np.mean([fv[k][s] for s in seeds]) for k in fk if all(s in fv[k] for s in seeds)}
        bk = max(bmean, key=bmean.get); pb = paired(v, fv[bk])
        mv, sdv, n = msd(v)
        rows.append([SNAME[st], lbl, n, f4(mv), f4(sdv), f4(np.mean([base[s] for s in seeds]) * 100),
                     f"{p['mean']:+.4f}", fci(p), p["n_pos"], bk, f"{pb['mean']:+.4f}", fci(pb)])
        pn(f"{SNAME[st]}: DriftGate[{lbl}] − DriftGate[default]", f"{p['mean']:+.3f} pp", fci(p), p["n"], src_of(pat, MAIN[st]["DriftGate"]))
wcsv("T4b_constant_sensitivity.csv", ["setting", "variant", "n_seeds", "variant_acc_pct", "sd_pct",
     "default_DriftGate_same_seeds_pct", "variant_minus_default_pp", "ci95", "n_pos",
     "best_fixed_same_seeds", "variant_minus_best_fixed_pp", "ci95_vs_best_fixed"], rows)

# ============================================================ T2 segment accuracy + contribution
rows_seg, rows_con, rows_best = [], [], []
segstore = {}
for st in ("A", "mob"):
    arms = {k: v for k, v in MAIN[st].items() if k == "DriftGate" or k.startswith("fixed") or k == "entropy"}
    for m, pat in arms.items():
        fs = famf(pat)
        if not fs: continue
        per = {}
        for s, f in fs.items():
            ev = {e["round"]: e for e in load(f)["eval"]}
            per[s] = {sg: {k: np.mean([ev[r][k] for r in rr if r in ev]) for k in ("acc_total", "acc_main", "acc_oop", "acc_oor")}
                      for sg, rr in SEG[st]}
        segstore[(st, m)] = per
        for sg, rr in SEG[st]:
            v = {k: np.array([per[s][sg][k] for s in per]) * 100 for k in ("acc_total", "acc_main", "acc_oop", "acc_oor")}
            rows_seg.append([SNAME[st], m, sg, " ".join(map(str, rr)), f4(v["acc_total"].mean()),
                             f4(v["acc_total"].std(ddof=1)) if len(per) > 1 else "", f4(v["acc_main"].mean()),
                             f4(v["acc_oop"].mean()), f4(v["acc_oor"].mean()), len(per)])
    N = sum(len(rr) for _, rr in SEG[st])
    dg = segstore.get((st, "DriftGate"))
    if not dg: continue
    for m in [k for k in arms if k != "DriftGate"]:
        ref = segstore.get((st, m))
        if not ref: continue
        seeds = sorted(set(dg) & set(ref)); tot = []
        for sg, rr in SEG[st]:
            d = np.array([dg[s][sg]["acc_total"] - ref[s][sg]["acc_total"] for s in seeds]) * 100
            lo, hi = ci95(d); w = len(rr) / N; tot.append(w * d.mean())
            rows_con.append([SNAME[st], f"DriftGate − {m}", sg, f"{d.mean():+.4f}", f"[{lo:+.2f}, {hi:+.2f}]",
                             int((d > 0).sum()), len(d), f"{w:.4f}", f"{w * d.mean():+.4f}"])
        full = paired(ifam(MAIN[st]["DriftGate"]), ifam(MAIN[st][m]))
        rows_con.append([SNAME[st], f"DriftGate − {m}", "SUM of contributions", f"{sum(tot):+.4f}", "", "", len(seeds),
                         "1.0000", f"{sum(tot):+.4f}  (integrated diff {full['mean']:+.4f})"])
    # best / worst fixed per segment and DriftGate vs worst fixed
    fk = [k for k in arms if k.startswith("fixed") and (st, k) in segstore]
    for sg, rr in SEG[st]:
        means = {k: np.mean([segstore[(st, k)][s][sg]["acc_total"] for s in segstore[(st, k)]]) * 100 for k in fk}
        bk, wk = max(means, key=means.get), min(means, key=means.get)
        wref = segstore[(st, wk)]; seeds = sorted(set(dg) & set(wref))
        d = np.array([dg[s][sg]["acc_total"] - wref[s][sg]["acc_total"] for s in seeds]) * 100; lo, hi = ci95(d)
        dgm = np.mean([dg[s][sg]["acc_total"] for s in dg]) * 100
        dbest = np.array([dg[s][sg]["acc_total"] - segstore[(st, bk)][s][sg]["acc_total"] for s in seeds]) * 100
        lo2, hi2 = ci95(dbest)
        rows_best.append([SNAME[st], sg, bk, f4(means[bk]), wk, f4(means[wk]), f4(dgm), f"{d.mean():+.4f}",
                          f"[{lo:+.2f}, {hi:+.2f}]", int((d > 0).sum()), f"{dbest.mean():+.4f}", f"[{lo2:+.2f}, {hi2:+.2f}]", len(seeds)])
wcsv("T2a_segment_accuracy.csv", ["setting", "method", "segment", "eval_rounds", "acc_total_pct", "sd_pct",
     "acc_main_pct", "acc_oop_pct", "acc_oor_pct", "n_seeds"], rows_seg)
wcsv("T2b_segment_contribution.csv", ["setting", "comparison", "segment", "segment_diff_pp", "ci95", "n_pos", "n",
     "weight(n_eval_rounds/N)", "contribution_pp"], rows_con)
wcsv("T2c_segment_best_worst_fixed.csv", ["setting", "segment", "best_fixed", "best_fixed_pct", "worst_fixed", "worst_fixed_pct",
     "DriftGate_pct", "DriftGate_minus_worst_pp", "ci95", "n_pos", "DriftGate_minus_best_pp", "ci95_vs_best", "n"], rows_best)

# ============================================================ T3 transfer
rows = []
for st, arms in TRANSFER.items():
    dg = ifam(arms["DriftGate"]); fk = FIXED_KEYS(arms); fv = {k: ifam(arms[k]) for k in fk}
    seeds = sorted(dg)
    complete = [k for k in fk if fv[k] and all(s in fv[k] for s in seeds)]
    best = max(complete, key=lambda k: np.mean([fv[k][s] for s in seeds])) if complete and seeds else None
    for m, pat in arms.items():
        v = ifam(pat)
        if not v: rows.append([SNAME[st], m, 0] + [""] * 6 + [src_of(pat)]); continue
        mean, sd, n = msd(v); p = None if m == "DriftGate" else paired(dg, v)
        rows.append([SNAME[st], m + (" (best fixed)" if m == best else ""), n, f4(mean), f4(sd),
                     "" if p is None else f"{p['mean']:+.4f}", fci(p), "" if p is None else p["n_pos"],
                     "" if p is None else p["n"], src_of(pat)])
    if best:
        p = paired(dg, fv[best])
        pn(f"{SNAME[st]}: DriftGate − best fixed ({best})", f"{p['mean']:+.3f} pp", fci(p), p["n"], src_of(arms["DriftGate"], arms[best]))
wcsv("T3_transfer.csv", ["setting", "method", "n_seeds", "integrated_acc_pct", "sd_pct", "DriftGate_minus_method_pp",
     "ci95", "n_pos", "n_matched", "source_runs"], rows)

# ============================================================ T5 ResNet-18 middle split
rows = []
dg = ifam(RESNET["DriftGate"]); fv = {k: ifam(p) for k, p in RESNET.items() if k != "DriftGate"}
best = max((k for k in fv if fv[k]), key=lambda k: np.mean(list(fv[k].values())), default=None)
for m, pat in RESNET.items():
    v = ifam(pat)
    if not v: rows.append([SNAME["resA"], m, 0] + [""] * 6 + [src_of(pat)]); continue
    mean, sd, n = msd(v); p = None if m == "DriftGate" else paired(dg, v)
    rows.append([SNAME["resA"], m + (" (best fixed)" if m == best else ""), n, f4(mean), f4(sd),
                 "" if p is None else f"{p['mean']:+.4f}", fci(p), "" if p is None else p["n_pos"], "" if p is None else p["n"], src_of(pat)])
    if p is not None: pn(f"ResNet-18 middle split: DriftGate − {m}", f"{p['mean']:+.3f} pp", fci(p), p["n"], src_of(RESNET["DriftGate"], pat))
wcsv("T5_resnet18_middle.csv", ["setting", "method", "n_seeds", "integrated_acc_pct", "sd_pct", "DriftGate_minus_method_pp",
     "ci95", "n_pos", "n_matched", "source_runs"], rows)

# ============================================================ T10 last evaluation round
rows = []
for st, arms in list(MAIN.items()) + list(TRANSFER.items()) + [("resA", RESNET)]:
    for m, pat in arms.items():
        fs = famf(pat)
        if not fs: continue
        last = [load(f)["eval"][-1] for f in fs.values()]
        v = np.array([e["acc_total"] for e in last]) * 100
        rows.append([SNAME[st], m, last[0]["round"], len(v), f4(v.mean()), f4(v.std(ddof=1)) if len(v) > 1 else ""])
wcsv("T10_last_round_accuracy.csv", ["setting", "method", "last_eval_round", "n_seeds", "acc_total_pct", "sd_pct"], rows)

# ============================================================ λ trajectories (A, mobility): relonly & entropy
rows = []
for st in ("A", "mob"):
    for m in ("DriftGate", "entropy"):
        fs = famf(MAIN[st][m])
        if not fs: continue
        L = np.array([[np.mean([float(x) for x in d.values()]) for d in load(f)["lamdas"]] for f in fs.values()])
        rho = [r if not isinstance(r, dict) else np.mean(list(r.values())) for r in load(list(fs.values())[0])["rho_trace"]]
        for r in range(L.shape[1]):
            rows.append([SNAME[st], m, r + 1, f"{rho[r]:.2f}", f"{L[:, r].mean():.5f}", f"{L[:, r].std(ddof=1):.5f}", L.shape[0]])
wcsv("lambda_trajectories.csv", ["setting", "method", "round", "rho", "lambda_mean", "lambda_sd_over_seeds", "n_seeds"], rows)

# ============================================================ T6 signal levels (passive fixed-λ runs)
LNC = math.log(10)
def ser(h, nm): return np.array([np.mean([float(v) for v in d.values()]) for d in h["signals_per_es"][nm]])
rows_lvl, rows_shift, rows_met = [], [], []
for sched, label in (("A", "100-round stepwise variant (ρ changes at R21, R41, R61, R81)"),
                     ("abrupt", "late abrupt change (ρ 0→0.8 at R51)")):
    fs = famf(f"signal_benchmark/rec_{sched}_s*.json")
    TV = np.array([ser(load(f), "tv_dist") for f in fs.values()])
    EN = np.array([ser(load(f), "ent_client") / LNC for f in fs.values()])
    rho = np.array([r if not isinstance(r, dict) else np.mean(list(r.values())) for r in load(list(fs.values())[0])["rho_trace"]])
    segs = ([("ρ=0", 16, 20), ("ρ=0.4", 21, 40), ("ρ=0.8", 41, 60), ("ρ=0.4", 61, 80), ("ρ=0", 81, 100)] if sched == "A"
            else [("ρ=0", 16, 50), ("ρ=0.8", 51, 100)])
    for sg, a, b in segs:
        t = TV[:, a - 1:b].mean(axis=1); e = EN[:, a - 1:b].mean(axis=1)
        rows_lvl.append([label, sg, f"R{a}–R{b}", f4(t.mean()), f4(t.std(ddof=1)), f4(e.mean()), f4(e.std(ddof=1)), len(t)])
    if sched == "abrupt":
        for nm, X in (("TV", TV), ("H/lnC", EN)):
            pre5, post5 = X[:, 45:50].mean(1), X[:, 50:55].mean(1)          # R46–50, R51–55
            start5 = X[:, 15:20].mean(1)                                    # R16–20
            shift = post5 - pre5; trend = pre5 - start5                     # 35 pre-shift rounds R16–R50
            rows_shift.append([nm, f4(pre5.mean()), f4(post5.mean()), f"{shift.mean():+.4f}", f4(shift.std(ddof=1)),
                               f"{trend.mean():+.4f}", f4(trend.std(ddof=1)), f"{(X[:, 95:100].mean(1) - start5).mean():+.4f}", len(shift)])
    for nm, key in (("TV", "tv_dist"), ("entropy", "ent_client")):
        sp, au = [], []
        for f in fs.values():
            h = load(f); x = ser(h, key); r_ = np.array([r if not isinstance(r, dict) else np.mean(list(r.values())) for r in h["rho_trace"]])
            m = evaluate_signal(x, r_, warmup=15, drift_level=0.5); sp.append(m["spearman_rho"]); au.append(m["auroc_raw"])
        sp, au = np.array(sp), np.array(au)
        rows_met.append([label, nm, f"{sp.mean():+.4f}", f4(sp.std(ddof=1)), f4(au.mean()), f4(max(au.mean(), 1 - au.mean())), len(sp)])
wcsv("T6a_signal_levels.csv", ["schedule", "segment", "rounds", "TV_mean", "TV_sd", "H_over_lnC_mean", "H_over_lnC_sd", "n_seeds"], rows_lvl)
wcsv("T6b_shift_response_late_abrupt.csv", ["signal", "mean_R46_50", "mean_R51_55", "shift_change(R51–55 − R46–50)", "sd",
     "pre_shift_change_35_rounds(R46–50 − R16–20)", "sd", "whole_run_change(R96–100 − R16–20)", "n_seeds"], rows_shift)
wcsv("T6c_passive_signal_metrics.csv", ["schedule", "signal", "spearman_vs_rho", "sd", "raw_direction_auroc",
     "direction_free_auroc", "n_seeds"], rows_met)
for r in rows_shift:
    pn(f"late abrupt change: {r[0]} change at the shift (R51–55 − R46–50)", r[3], "", r[8], "rec_abrupt_s{0,1,2}")
    pn(f"late abrupt change: {r[0]} change over the 35 pre-shift rounds (R46–50 − R16–20)", r[5], "", r[8], "rec_abrupt_s{0,1,2}")
for r in rows_met:
    pn(f"{r[0]}: {r[1]} Spearman vs ρ / raw AUROC", f"{r[2]} / {r[4]}", "", r[6], "rec_*_s{0,1,2}")

# ============================================================ T7 role experiment (existing 3-seed runs)
rows = []
ROLE = {"standard": "phaseC_signals/dvsig_tv_A_s*.json", "same-role exits": "phaseR_role/role_same_role_s*.json",
        "same-role independent init": "phaseR_role/role_same_role_indep_s*.json", "weakened server": "phaseR_role/role_weak_server_s*.json"}
for cond, pat in ROLE.items():
    cs = []
    for f in famf(pat).values():
        h = load(f); tv = ser(h, "tv_dist")
        rho = np.array([r if not isinstance(r, dict) else np.mean(list(r.values())) for r in h["rho_trace"]])
        cs.append(stats.spearmanr(tv[15:], rho[15:]).statistic)
    cs = np.array(cs)
    rows.append([cond, f"{cs.mean():+.4f}", f4(cs.std(ddof=1)), " ".join(f"{c:+.3f}" for c in cs), len(cs),
                 "same-pool; runs used the earlier controller with the absolute branch", src_of(pat)])
    pn(f"role experiment: TV–ρ Spearman, {cond}", f"{cs.mean():+.3f} (SD {cs.std(ddof=1):.3f})", "", len(cs), src_of(pat))
wcsv("T7_role_experiment.csv", ["condition", "tv_rho_spearman_mean", "sd", "per_seed", "n_seeds", "protocol", "source_runs"], rows)

with open(HERE / "paper_numbers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item", "value", "ci95", "n_seeds", "source_runs"]); w.writerows(PN)
print(f"  [paper_numbers.csv] {len(PN)} rows")
print("DONE ->", TAB)
