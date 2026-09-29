"""Round-4 Part B analysis (B1–B6) + run manifest. Reads new runs under runs/phaseT4_*/
and the existing disjoint reference runs. Paired diffs are same-seed only (Student-t 95% CI,
n_pos). Re-runnable; missing arms are simply absent from the CSVs (never imputed).
Writes ../data/b*.csv, ../data/run_manifest.csv
"""
import json, glob, re, csv, math, sys
import numpy as np
from pathlib import Path
from scipy import stats

JR = Path("/disk2/Yujin/adaptive_splitomc_tmc/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT = Path(__file__).resolve().parent.parent / "data"; OUT.mkdir(parents=True, exist_ok=True)
R = JR / "runs"
def load(p): return json.load(open(p))
def famf(pat):
    d = {}
    for f in sorted(glob.glob(str(R / pat))):
        m = re.search(r"_s(\d+)\.json$", f)
        if m: d[int(m.group(1))] = f
    return d
def ifam(pat): return {s: integ(load(f)) for s, f in famf(pat).items()}
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def worst(h): return float(np.mean([e["worst_cell_acc"] for e in h["eval"] if e.get("worst_cell_acc") is not None]))
def ci95(d):
    d = np.asarray(d, float)
    if len(d) < 2: return (float("nan"), float("nan"))
    se = stats.sem(d); h = se * stats.t.ppf(0.975, len(d) - 1); return (d.mean() - h, d.mean() + h)
def paired(a, b):
    s = sorted(set(a) & set(b)); d = np.array([a[x] - b[x] for x in s]); lo, hi = ci95(d)
    return (f"{d.mean()*100:+.4f}", f"[{lo*100:+.2f},{hi*100:+.2f}]", int((d > 0).sum()), len(d)) if len(s) else ("", "", "", 0)
def mean_sd(v): v = np.array(list(v.values())) * 100; return f"{v.mean():.4f}", (f"{v.std(ddof=1):.4f}" if len(v) > 1 else "")
def wcsv(name, hdr, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")
def lam_series(h): return np.array([np.mean([float(v) for v in d.values()]) for d in h["lamdas"]])

ENVN = {"A": "stepwise composition change", "mob": "client mobility", "svhn": "SVHN temporal",
        "c10gsig": "CIFAR-10 gradual", "c100gsig": "CIFAR-100 gradual", "c100sp": "CIFAR-100 spatial", "tiny": "Tiny-ImageNet"}
REF = {  # env -> full TV disjoint run, rate run, a1, entropy
 "A":   dict(tv="phaseT1_disjoint/t1_A_dg_s*.json", rate="phaseT2_signal/t2_A_hard_s*.json", a1="phaseT1_disjoint/t1_A_a1_s*.json", ent="phaseT1_disjoint/t1_A_ent_s*.json"),
 "mob": dict(tv="phaseT1_disjoint/t1_mob_dg_s*.json", rate="phaseT2_signal/t2_mob_hard_s*.json", a1="phaseT1_disjoint/t1_mob_a1_s*.json", ent="phaseT1_disjoint/t1_mob_ent_s*.json"),
 "svhn": dict(tv="phaseT2_signal/t2_svhn_tv_s*.json", rate="phaseT2_signal/t2_svhn_hard_s*.json", ent="phaseT2_signal/t2_svhn_ent_s*.json"),
 "c10gsig": dict(tv="phaseT2r2_signal/r2_c10gsig_tv_s*.json", rate="phaseT2r2_signal/r2_c10gsig_hard_s*.json"),
 "c100gsig": dict(tv="phaseT2r2_signal/r2_c100gsig_tv_s*.json", rate="phaseT2r2_signal/r2_c100gsig_hard_s*.json"),
 "c100sp": dict(tv="phaseT2r2_signal/r2_c100sp_tv_s*.json", rate="phaseT2r2_signal/r2_c100sp_hard_s*.json"),
 "tiny": dict(tv="phaseT2r2_signal/r2_tiny_tv_s*.json", rate="phaseT2r2_signal/r2_tiny_hard_s*.json"),
}
# fixed grid (disjoint) per env: lambda -> glob
FIXED = {
 "A": {0.2: "phaseT2r2_signal/r2_A_fx20_s*.json", 0.3: "phaseT1_disjoint/t1_A_fx30_s*.json", 0.4: "phaseT1_disjoint/t1_A_fx40_s*.json", 0.5: "phaseT1_disjoint/t1_A_fx50_s*.json"},
 "mob": {0.2: "phaseT2r2_signal/r2_mob_fx20_s*.json", 0.3: "phaseT1_disjoint/t1_mob_fx30_s*.json", 0.4: "phaseT1_disjoint/t1_mob_fx40_s*.json",
         0.5: "phaseT1_disjoint/t1_mob_fx50_s*.json", "0.5(B3)": "phaseT4_B3_fixedgrid/b3_mob_fx50_s*.json", 0.6: "phaseT4_B3_fixedgrid/b3_mob_fx60_s*.json"},
 "svhn": {0.15: "phaseT4_B3_fixedgrid/b3_svhn_fx15_s*.json", 0.2: "phaseT2_signal/t2_svhn_fx20_s*.json", 0.4: "phaseT4_B3_fixedgrid/b3_svhn_fx40_s*.json"},
 "c10gsig": {0.15: "phaseT4_B3_fixedgrid/b3_c10gsig_fx15_s*.json", 0.2: "phaseT3_fixedref/t3_c10gsig_fx20_s*.json", 0.4: "phaseT3_fixedref/t3_c10gsig_fx40_s*.json"},
 "c100gsig": {0.15: "phaseT4_B3_fixedgrid/b3_c100gsig_fx15_s*.json", 0.2: "phaseT3_fixedref/t3_c100gsig_fx20_s*.json", 0.4: "phaseT3_fixedref/t3_c100gsig_fx40_s*.json"},
 "c100sp": {0.15: "phaseT4_B3_fixedgrid/b3_c100sp_fx15_s*.json", 0.2: "phaseT3_fixedref/t3_c100sp_fx20_s*.json", 0.4: "phaseT3_fixedref/t3_c100sp_fx40_s*.json"},
 "tiny": {0.15: "phaseT4_B3_fixedgrid/b3_tiny_fx15_s*.json", 0.2: "phaseT3_fixedref/t3_tiny_fx20_s*.json", 0.4: "phaseT3_fixedref/t3_tiny_fx40_s*.json"},
}

# ============================================================ B1 view removal
print("B1 view removal")
rows = []; rows_w = []
for env, arms in (("A", ["absonly", "relonly"]), ("mob", ["absonly", "relonly"]), ("c100gsig", ["absonly", "relonly"]), ("c100sp", ["nospatial"])):
    full = ifam(REF[env]["tv"])
    for arm in arms:
        v = ifam(f"phaseT4_B1_views/b1_{arm}_{env}_s*.json")
        if not v: continue
        m, sd = mean_sd(v); fm, _ = mean_sd({s: full[s] for s in v if s in full})
        d, c, npos, n = paired(full, v)
        rows.append([ENVN[env], arm, m, sd, fm, d, c, npos, n])
        if env == "c100sp":
            fw = {s: worst(load(f)) for s, f in famf(REF[env]["tv"]).items()}; aw = {s: worst(load(f)) for s, f in famf(f"phaseT4_B1_views/b1_{arm}_{env}_s*.json").items()}
            d2, c2, np2, n2 = paired(fw, aw)
            rows_w.append([ENVN[env], arm, f"{np.mean(list(fw.values()))*100:.4f}", f"{np.mean(list(aw.values()))*100:.4f}", d2, c2, np2, n2])
wcsv("b1_view_removal.csv", ["setting", "arm", "arm_mean_pct", "arm_sd", "full_mean_pct", "full_minus_arm_pp", "ci95", "n_pos", "n_seeds"], rows)
wcsv("b1_worst_cluster_c100sp.csv", ["setting", "arm", "full_worst_pct", "arm_worst_pct", "full_minus_arm_pp", "ci95", "n_pos", "n_seeds"], rows_w)

# ============================================================ B2 entnorm
print("B2 entnorm")
rows = []; traj = []
for env in ("A", "mob"):
    tv = ifam(REF[env]["tv"]); ent = ifam(REF[env]["ent"]); en = ifam(f"phaseT4_B2_entnorm/b2_entnorm_{env}_s*.json")
    for lbl, v in (("entropy (relative-only, as run)", ent), ("entnorm (H/lnC, full controller)", en)):
        if not v: continue
        m, sd = mean_sd(v); d, c, npos, n = paired(tv, v)
        rows.append([ENVN[env], lbl, m, sd, d, c, npos, n])
    for s, f in famf(f"phaseT4_B2_entnorm/b2_entnorm_{env}_s*.json").items():
        ls = lam_series(load(f))
        for r, v in enumerate(ls): traj.append([env, s, r + 1, f"{v:.4f}"])
wcsv("b2_entnorm.csv", ["setting", "arm", "mean_pct", "sd", "TV_minus_arm_pp", "ci95", "n_pos", "n_seeds"], rows)
wcsv("b2_entnorm_lambda_traj.csv", ["setting", "seed", "round", "lambda_mean_over_clusters"], traj)

# ============================================================ B3 fixed grid
print("B3 fixed grid")
rows_grid, rows_best, rows_diff_common, rows_diff_full = [], [], [], []
for env, grid in FIXED.items():
    tv = ifam(REF[env]["tv"]); rate = ifam(REF[env]["rate"])
    vals = {}
    for lam, pat in grid.items():
        v = ifam(pat)
        if not v: continue
        vals[lam] = v; m, sd = mean_sd(v)
        rows_grid.append([ENVN[env], lam, m, sd, len(v)])
    if not vals: continue
    seeds = sorted(set(tv))
    means = {lam: np.mean([v[s] for s in seeds if s in v]) for lam, v in vals.items() if all(s in v for s in seeds)}
    if not means: continue
    best = max(means, key=means.get)
    d, c, npos, n = paired(tv, vals[best]); dr, cr, npr, nr = paired(rate, vals[best])
    rows_best.append([ENVN[env], best, f"{means[best]*100:.4f}", "|".join(str(k) for k in sorted(means, key=str)), d, c, npos, n, dr, cr, npr, nr])
    # difference tables: best-of-evaluated minus {TV, rate, each fixed}
    def difftab(subset, tag):
        sub = {k: means[k] for k in subset if k in means}
        if not sub: return
        b = max(sub, key=sub.get)
        for lbl, v in (("TV", tv), ("rate", rate)):
            dd = np.array([vals[b][s] - v[s] for s in seeds if s in v and s in vals[b]])
            (rows_diff_common if tag == "common" else rows_diff_full).append([ENVN[env], tag, b, f"best - {lbl}", f"{dd.mean()*100:+.4f}", f"{dd.max()*100:+.4f}"])
        for k in sub:
            dd = np.array([vals[b][s] - vals[k][s] for s in seeds if s in vals[k] and s in vals[b]])
            (rows_diff_common if tag == "common" else rows_diff_full).append([ENVN[env], tag, b, f"best - fixed {k}", f"{dd.mean()*100:+.4f}", f"{dd.max()*100:+.4f}"])
    difftab([0.2, 0.4], "common"); difftab(list(means.keys()), "full")
wcsv("b3_fixed_grid.csv", ["setting", "lambda", "mean_pct", "sd", "n_seeds"], rows_grid)
wcsv("b3_best_fixed.csv", ["setting", "best_lambda", "best_mean_pct", "evaluated_lambdas", "TV_minus_best_pp", "ci95", "n_pos", "n",
                           "rate_minus_best_pp", "ci95_rate", "n_pos_rate", "n_rate"], rows_best)
wcsv("b3_diff_common.csv", ["setting", "grid", "best_lambda", "comparison", "mean_pp", "worst_seed_pp"], rows_diff_common)
wcsv("b3_diff_full.csv", ["setting", "grid", "best_lambda", "comparison", "mean_pp", "worst_seed_pp"], rows_diff_full)
# cross-setting summary: for each single fixed lambda, mean & worst (over settings) of best_of_grid - fixed(lambda)
summ = {}
for row in rows_diff_full:
    if row[3].startswith("best - fixed "):
        lam = row[3].split("best - fixed ")[1]; summ.setdefault(lam, []).append(float(row[4]))
for lbl in ("TV", "rate"):
    summ[lbl] = [float(r[4]) for r in rows_diff_full if r[3] == f"best - {lbl}"]
rows = [[k, len(v), f"{np.mean(v):+.4f}", f"{np.max(v):+.4f}"] for k, v in summ.items()]
wcsv("b3_single_lambda_gap_summary.csv", ["policy", "n_settings", "mean_gap_to_best_fixed_pp", "worst_gap_pp"], rows)

# ============================================================ B4 APFL
print("B4 APFL")
rows = []; traj = []
for env in ("A", "mob"):
    tv = ifam(REF[env]["tv"]); a1 = ifam(REF[env]["a1"])
    for tag, lbl in (("eta001", "APFL η=lr(0.01)"), ("eta010", "APFL η=10lr(0.1)")):
        fs = famf(f"phaseT4_B4_apfl/b4_apfl_{tag}_{env}_s*.json")
        if not fs: continue
        v = {s: integ(load(f)) for s, f in fs.items()}; m, sd = mean_sd(v)
        d1, c1, p1, n1 = paired(a1, v); d2, c2, p2, n2 = paired(tv, v)
        rows.append([ENVN[env], lbl, m, sd, d1, c1, p1, n1, d2, c2, p2, n2])
        for s, f in fs.items():
            h = load(f)
            for r, cl in enumerate(h.get("apfl_client_lams", [])):
                vals = np.array(list(cl.values()), float)
                traj.append([env, lbl, s, r + 1, f"{vals.mean():.4f}", f"{vals.std():.4f}", f"{vals.min():.4f}", f"{vals.max():.4f}"])
wcsv("b4_apfl.csv", ["setting", "arm", "mean_pct", "sd", "a1_minus_apfl_pp", "ci95", "n_pos", "n", "TV_minus_apfl_pp", "ci95_tv", "n_pos_tv", "n_tv"], rows)
wcsv("b4_apfl_lambda_traj.csv", ["setting", "arm", "seed", "round", "lam_mean", "lam_std_clients", "lam_min", "lam_max"], traj)

# ============================================================ B5 probe count
print("B5 probe count")
rows = []
for env in ("A", "mob"):
    tv_f = famf(REF[env]["tv"]); tv = {s: integ(load(f)) for s, f in tv_f.items()}
    for p in (16, 32):
        fs = famf(f"phaseT4_B5_probe/b5_probe{p}_{env}_s*.json")
        if not fs: continue
        v = {s: integ(load(f)) for s, f in fs.items()}; m, sd = mean_sd(v); d, c, npos, n = paired(tv, v)
        lamdiff, sig_sd, sig_sd64 = [], [], []
        for s, f in fs.items():
            if s not in tv_f: continue
            h = load(f); h64 = load(tv_f[s])
            lamdiff.append(np.mean(np.abs(lam_series(h) - lam_series(h64))))
            for hh, store in ((h, sig_sd), (h64, sig_sd64)):
                ser = hh["signals_per_es"]["tv_dist"]; per_es = {}
                for d_ in ser:
                    for k, vv in d_.items(): per_es.setdefault(k, []).append(float(vv))
                store.append(np.mean([np.std(np.diff(x)) for x in per_es.values()]))
        rows.append([ENVN[env], p, m, sd, d, c, npos, n, f"{np.mean(lamdiff):.4f}", f"{np.mean(sig_sd):.4f}", f"{np.mean(sig_sd64):.4f}"])
wcsv("b5_probe.csv", ["setting", "probe_n", "mean_pct", "sd", "probe64_minus_arm_pp", "ci95", "n_pos", "n",
                      "mean_abs_lambda_diff_vs_64", "signal_round_to_round_sd", "signal_round_to_round_sd_probe64"], rows)

# ============================================================ B6 constants
print("B6 constant sensitivity")
rows = []
for env in ("A", "mob"):
    tv = ifam(REF[env]["tv"]); best_lam = None
    grid = {lam: ifam(p) for lam, p in FIXED[env].items() if isinstance(lam, float)}
    grid = {k: v for k, v in grid.items() if v}
    for tag, lbl in (("guard025", "guard 0.25"), ("guard100", "guard 1.0"), ("lmax060", "λmax 0.60"), ("lmax080", "λmax 0.80")):
        fs = famf(f"phaseT4_B6_const/b6_{tag}_{env}_s*.json")
        if not fs: continue
        v = {s: integ(load(f)) for s, f in fs.items()}; m, sd = mean_sd(v)
        seeds = sorted(v); d, c, npos, n = paired({s: tv[s] for s in seeds if s in tv}, v)
        means = {k: np.mean([g[s] for s in seeds if s in g]) for k, g in grid.items() if all(s in g for s in seeds)}
        bl = max(means, key=means.get) if means else None
        db, cb, pb, nb = paired(v, grid[bl]) if bl is not None else ("", "", "", 0)
        rows.append([ENVN[env], lbl, m, sd, d, c, npos, n, bl, db, cb, pb, nb])
wcsv("b6_constants.csv", ["setting", "arm", "mean_pct", "sd", "default_minus_arm_pp", "ci95", "n_pos", "n",
                          "best_fixed_lambda(same seeds)", "arm_minus_bestfixed_pp", "ci95_bf", "n_pos_bf", "n_bf"], rows)

# ============================================================ run manifest
print("run manifest")
rows = []
for d in sorted(glob.glob(str(R / "phaseT4_*"))):
    for f in sorted(glob.glob(f"{d}/*.json")):
        h = load(f); nm = Path(f).stem; c = h.get("config", {})
        m = re.search(r"_s(\d+)$", nm)
        rows.append([c.get("run_id", nm), nm, Path(d).name, m.group(1) if m else "", c.get("controller_mode"), c.get("controller_signal"),
                     len(h["round"]), len(h["eval"]), h.get("probe_eval_overlap"), "yes"])
wcsv("run_manifest.csv", ["run_id", "run_name", "phase", "seed", "controller_mode", "signal", "rounds", "eval_rounds", "overlap", "complete"], rows)
print("DONE ->", OUT)
