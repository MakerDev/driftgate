"""Round-4 Part A — analyses from existing (and, when present, new Part-B) run JSON.
A1 segment accuracy + contribution decomposition + last-round table
A2 absolute/relative candidate activity (exact from lam_rel/lam_abs when logged,
   else reconstructed from controller_z + raw signal via the frozen mapping)
A3 transfer-setting paired CIs (TV/rate vs fixed 0.2/0.4/0.15)
A4 F1 statistics from the passive rec_* runs (relabelled schedules)
Re-runnable; Part-B arms are picked up automatically once their JSON exists.
Writes ../data/*.csv
"""
import json, glob, re, csv, math, sys
import numpy as np
from pathlib import Path
from scipy import stats

JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
OUT = Path(__file__).resolve().parent.parent / "data"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR)); sys.path.insert(0, str(JR.parent))
from src.controllers.self_calibrating import SIGNAL_RANGE, Z0, TAU_Z, _sigmoid, _consensus

R = JR / "runs"
def load(p): return json.load(open(p))
def famf(pat):
    d = {}
    for f in sorted(glob.glob(str(R / pat))):
        m = re.search(r"_s(\d+)\.json$", f)
        if m: d[int(m.group(1))] = f
    return d
def integ(h): return float(np.mean([e["acc_total"] for e in h["eval"]]))
def ci95(d):
    d = np.asarray(d, float)
    if len(d) < 2: return (float("nan"), float("nan"))
    se = stats.sem(d); h = se * stats.t.ppf(0.975, len(d) - 1); return (d.mean() - h, d.mean() + h)
def wcsv(name, hdr, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"  [{name}] {len(rows)} rows")

# ---------------------------------------------------------------- arm registry
# label -> (glob, kind, signal_name_for_A2)   kind: ctrl | fixed
ARMS = {
 "A": {
  "TV full":           ("phaseT1_disjoint/t1_A_dg_s*.json", "ctrl", "tv_dist"),
  "a1 (adaptive λ, Λ=0.5)": ("phaseT1_disjoint/t1_A_a1_s*.json", "ctrl", "tv_dist"),
  "entropy":           ("phaseT1_disjoint/t1_A_ent_s*.json", "ctrl", "ent_client"),
  "server rate":       ("phaseT2_signal/t2_A_hard_s*.json", "ctrl", "server_nonmain_hard"),
  "fixed 0.2":         ("phaseT2r2_signal/r2_A_fx20_s*.json", "fixed", None),
  "fixed 0.3":         ("phaseT1_disjoint/t1_A_fx30_s*.json", "fixed", None),
  "fixed 0.4":         ("phaseT1_disjoint/t1_A_fx40_s*.json", "fixed", None),
  "fixed 0.5":         ("phaseT1_disjoint/t1_A_fx50_s*.json", "fixed", None),
  # Part B (present once run)
  "B1 absonly":        ("phaseT4_B1_views/b1_absonly_A_s*.json", "ctrl", "tv_dist"),
  "B1 relonly":        ("phaseT4_B1_views/b1_relonly_A_s*.json", "ctrl", "tv_dist"),
  "B2 entnorm":        ("phaseT4_B2_entnorm/b2_entnorm_A_s*.json", "ctrl", "ent_client_norm"),
  "B4 apfl η=lr":      ("phaseT4_B4_apfl/b4_apfl_eta001_A_s*.json", "fixed", None),
  "B4 apfl η=10lr":    ("phaseT4_B4_apfl/b4_apfl_eta010_A_s*.json", "fixed", None),
  "B5 probe16":        ("phaseT4_B5_probe/b5_probe16_A_s*.json", "ctrl", "tv_dist"),
  "B5 probe32":        ("phaseT4_B5_probe/b5_probe32_A_s*.json", "ctrl", "tv_dist"),
  "B6 guard0.25":      ("phaseT4_B6_const/b6_guard025_A_s*.json", "ctrl", "tv_dist"),
  "B6 guard1.0":       ("phaseT4_B6_const/b6_guard100_A_s*.json", "ctrl", "tv_dist"),
  "B6 λmax0.60":       ("phaseT4_B6_const/b6_lmax060_A_s*.json", "ctrl", "tv_dist"),
  "B6 λmax0.80":       ("phaseT4_B6_const/b6_lmax080_A_s*.json", "ctrl", "tv_dist"),
 },
 "mob": {
  "TV full":           ("phaseT1_disjoint/t1_mob_dg_s*.json", "ctrl", "tv_dist"),
  "a1 (adaptive λ, Λ=0.5)": ("phaseT1_disjoint/t1_mob_a1_s*.json", "ctrl", "tv_dist"),
  "entropy":           ("phaseT1_disjoint/t1_mob_ent_s*.json", "ctrl", "ent_client"),
  "server rate":       ("phaseT2_signal/t2_mob_hard_s*.json", "ctrl", "server_nonmain_hard"),
  "fixed 0.2":         ("phaseT2r2_signal/r2_mob_fx20_s*.json", "fixed", None),
  "fixed 0.3":         ("phaseT1_disjoint/t1_mob_fx30_s*.json", "fixed", None),
  "fixed 0.4":         ("phaseT1_disjoint/t1_mob_fx40_s*.json", "fixed", None),
  "fixed 0.5":         ("phaseT1_disjoint/t1_mob_fx50_s*.json", "fixed", None),
  "B3 fixed 0.5":      ("phaseT4_B3_fixedgrid/b3_mob_fx50_s*.json", "fixed", None),
  "B3 fixed 0.6":      ("phaseT4_B3_fixedgrid/b3_mob_fx60_s*.json", "fixed", None),
  "B1 absonly":        ("phaseT4_B1_views/b1_absonly_mob_s*.json", "ctrl", "tv_dist"),
  "B1 relonly":        ("phaseT4_B1_views/b1_relonly_mob_s*.json", "ctrl", "tv_dist"),
  "B2 entnorm":        ("phaseT4_B2_entnorm/b2_entnorm_mob_s*.json", "ctrl", "ent_client_norm"),
  "B4 apfl η=lr":      ("phaseT4_B4_apfl/b4_apfl_eta001_mob_s*.json", "fixed", None),
  "B4 apfl η=10lr":    ("phaseT4_B4_apfl/b4_apfl_eta010_mob_s*.json", "fixed", None),
  "B5 probe16":        ("phaseT4_B5_probe/b5_probe16_mob_s*.json", "ctrl", "tv_dist"),
  "B5 probe32":        ("phaseT4_B5_probe/b5_probe32_mob_s*.json", "ctrl", "tv_dist"),
  "B6 λmax0.60":       ("phaseT4_B6_const/b6_lmax060_mob_s*.json", "ctrl", "tv_dist"),
  "B6 λmax0.80":       ("phaseT4_B6_const/b6_lmax080_mob_s*.json", "ctrl", "tv_dist"),
 },
}
SEG = {
 "A":   [("rho0 early", [1, 10, 20, 30]), ("rho0.4 up", [40, 50, 60]), ("rho0.8", [70, 80, 90]),
         ("rho0.4 down", [100, 110, 120]), ("rho0 late", [130, 140, 150])],
 "mob": [("pre-move", [1, 10, 20, 30, 40, 50, 60]), ("post-move", [70, 80, 90, 100, 110, 120])],
}
# round-index segments for A2 (controller rounds, 1-indexed): boundaries R31/61/91/121 ; mob move at R60
SEG_ROUNDS = {"A": [("rho0 early", 1, 30), ("rho0.4 up", 31, 60), ("rho0.8", 61, 90),
                    ("rho0.4 down", 91, 120), ("rho0 late", 121, 150)],
              "mob": [("pre-move", 1, 60), ("post-move", 61, 120)]}

# ================================================================= A1
print("A1 segment accuracy")
rows_seg, rows_task, rows_last, rows_contrib = [], [], [], []
seg_store = {}   # (setting, arm) -> {seed: {seg: acc}}
for setting, arms in ARMS.items():
    for label, (pat, kind, sig) in arms.items():
        fs = famf(pat)
        if not fs: continue
        per_seed = {}
        for s, f in fs.items():
            h = load(f); ev = {e["round"]: e for e in h["eval"]}
            per_seed[s] = {}
            for segname, rounds in SEG[setting]:
                es = [ev[r] for r in rounds if r in ev]
                per_seed[s][segname] = dict(
                    acc=np.mean([e["acc_total"] for e in es]),
                    main=np.mean([e["acc_main"] for e in es]),
                    oop=np.mean([e["acc_oop"] for e in es]),
                    oor=np.mean([e["acc_oor"] for e in es]),
                    n=len(es))
            last = h["eval"][-1]
            rows_last.append([setting, label, s, last["round"], f"{last['acc_total']*100:.4f}"])
        seg_store[(setting, label)] = per_seed
        for segname, _ in SEG[setting]:
            a = [per_seed[s][segname]["acc"] for s in per_seed]
            rows_seg.append([setting, label, segname, f"{np.mean(a)*100:.4f}",
                             f"{np.std(a, ddof=1)*100:.4f}" if len(a) > 1 else "", len(a)])
            for t in ("main", "oop", "oor"):
                v = [per_seed[s][segname][t] for s in per_seed]
                rows_task.append([setting, label, segname, t, f"{np.mean(v)*100:.4f}", len(v)])
wcsv("a1_segment_acc.csv", ["setting", "arm", "segment", "acc_mean_pct", "acc_sd_pct", "n_seeds"], rows_seg)
wcsv("a1_segment_maintask.csv", ["setting", "arm", "segment", "task", "acc_mean_pct", "n_seeds"], rows_task)
wcsv("a1_last_round.csv", ["setting", "arm", "seed", "last_round", "acc_total_pct"], rows_last)

# contribution decomposition: TV full - ref, integrated diff = sum_seg (n_seg/N) * seg_diff
for setting in ARMS:
    tv = seg_store.get((setting, "TV full"))
    if not tv: continue
    N = sum(len(r) for _, r in SEG[setting])
    for label in ARMS[setting]:
        if label == "TV full" or (setting, label) not in seg_store: continue
        ref = seg_store[(setting, label)]
        seeds = sorted(set(tv) & set(ref))
        if not seeds: continue
        total = []
        for segname, rounds in SEG[setting]:
            d = np.array([tv[s][segname]["acc"] - ref[s][segname]["acc"] for s in seeds])
            w = len(rounds) / N
            lo, hi = ci95(d)
            rows_contrib.append([setting, f"TV full - {label}", segname, f"{d.mean()*100:+.4f}",
                                 f"[{lo*100:+.2f},{hi*100:+.2f}]", int((d > 0).sum()), len(d),
                                 f"{w:.4f}", f"{(w*d.mean())*100:+.4f}"])
            total.append(w * d.mean())
        rows_contrib.append([setting, f"TV full - {label}", "SUM(=integrated diff)", f"{sum(total)*100:+.4f}",
                             "", "", len(seeds), "1.0000", f"{sum(total)*100:+.4f}"])
wcsv("a1_contribution.csv", ["setting", "comparison", "segment", "seg_diff_pp", "ci95", "n_pos", "n_seeds",
                             "weight", "contribution_pp"], rows_contrib)

# ================================================================= A2
print("A2 absolute/relative candidate activity")
NB = {i: [j for j in (i - 1, i + 1) if 0 <= j < 5] for i in range(5)}   # line graph, 5 ES
WARM = 25   # warm-up 15 + burn-in 10 (n_obs <= 25 -> midpoint)
def lam_bounds(name):
    lo, hi = 0.15, 0.70
    if "lmax060" in name: hi = 0.60
    if "lmax080" in name: hi = 0.80
    return lo, hi
def reconstruct(h, sig, lo, hi):
    """Return per-round dicts lam_rel, lam_abs (post-warm-up) from controller_z + raw signal."""
    rng = SIGNAL_RANGE.get(sig, 1.0)
    cz = h["controller_z"]; raw = h["signals_per_es"].get(sig)
    T = len(cz); rel, ab = [], []
    state = {}
    for r in range(T):
        d_rel, d_abs = {}, {}
        if raw is not None and r < len(raw):
            cons = _consensus({int(k): float(v) for k, v in raw[r].items()}, NB, 1)
            for es, v in cons.items():
                prev = state.get(es, v); s = 0.7 * prev + 0.3 * v; state[es] = s
                if r + 1 > WARM and rng:
                    d_hat = min(max(s / rng, 0.0), 1.0)
                    d_abs[es] = hi - (hi - lo) * d_hat
        for k, zv in cz[r].items():
            es = int(k)
            if r + 1 > WARM:
                g = _sigmoid((float(zv) - Z0) / TAU_Z); d_rel[es] = hi - (hi - lo) * g
        rel.append(d_rel); ab.append(d_abs)
    return rel, ab
rows_act, rows_chk, rows_traj = [], [], []
for setting, arms in ARMS.items():
    for label, (pat, kind, sig) in arms.items():
        if kind != "ctrl": continue
        fs = famf(pat)
        if not fs: continue
        abs_enabled = not ("relonly" in pat or label == "entropy")   # entropy arm ran without --abs_cap
        agg_all, agg_seg = [], {sn: [] for sn, _, _ in SEG_ROUNDS[setting]}
        traj = []
        for s, f in fs.items():
            h = load(f); name = Path(f).stem; lo, hi = lam_bounds(name)
            exact = "lam_rel" in h and any(h["lam_rel"])
            if exact:
                rel = [{int(k): v for k, v in d.items()} for d in h["lam_rel"]]
                ab = [{int(k): v for k, v in d.items()} for d in h["lam_abs"]]
            else:
                rel, ab = reconstruct(h, sig, lo, hi)
                # validation vs recorded lambda (full-controller arms only)
                if abs_enabled and "absonly" not in pat:
                    err = []
                    for r in range(WARM, len(rel)):
                        for es in rel[r]:
                            if es in ab[r]:
                                pred = min(rel[r][es], ab[r][es]); rec = float(h["lamdas"][r][str(es)])
                                err.append(abs(pred - rec))
                    if err:
                        rows_chk.append([setting, label, s, len(err), f"{max(err):.4f}", f"{np.mean(err):.5f}"])
            n = act = 0; per_seg = {sn: [0, 0] for sn, _, _ in SEG_ROUNDS[setting]}
            for r in range(WARM, len(rel)):
                for es in rel[r]:
                    if not abs_enabled or es not in ab[r]: continue
                    a = ab[r][es] < rel[r][es] - 1e-12
                    n += 1; act += a
                    for sn, r0, r1 in SEG_ROUNDS[setting]:
                        if r0 <= r + 1 <= r1: per_seg[sn][0] += 1; per_seg[sn][1] += a
            agg_all.append(act / n if n else float("nan"))
            for sn in per_seg:
                agg_seg[sn].append(per_seg[sn][1] / per_seg[sn][0] if per_seg[sn][0] else float("nan"))
            if label == "TV full":
                for r in range(len(h["lamdas"])):
                    lr_ = np.mean(list(rel[r].values())) if rel[r] else float("nan")
                    la_ = np.mean(list(ab[r].values())) if ab[r] else float("nan")
                    lf = np.mean([float(v) for v in h["lamdas"][r].values()])
                    traj.append((s, r + 1, lr_, la_, lf))
        rows_act.append([setting, label, "ALL", f"{np.nanmean(agg_all):.4f}" if abs_enabled else "n/a (abs disabled)",
                         len(agg_all), "exact" if exact else "reconstructed"])
        for sn in agg_seg:
            rows_act.append([setting, label, sn, f"{np.nanmean(agg_seg[sn]):.4f}" if abs_enabled else "n/a",
                             len(agg_seg[sn]), "exact" if exact else "reconstructed"])
        if traj:
            byr = {}
            for s, r, a, b, c in traj: byr.setdefault(r, []).append((a, b, c))
            for r in sorted(byr):
                v = np.array(byr[r], float)
                rows_traj.append([setting, r, f"{np.nanmean(v[:,0]):.4f}", f"{np.nanmean(v[:,1]):.4f}",
                                  f"{np.nanmean(v[:,2]):.4f}", len(v)])
wcsv("a2_abs_activity.csv", ["setting", "arm", "segment", "frac_abs_lt_rel", "n_seeds", "source"], rows_act)
wcsv("a2_reconstruction_check.csv", ["setting", "arm", "seed", "n_cluster_rounds", "max_abs_err", "mean_abs_err"], rows_chk)
wcsv("a2_lambda_traj.csv", ["setting", "round", "lam_rel", "lam_abs", "lam_final", "n_seeds"], rows_traj)

# ================================================================= A3
print("A3 transfer paired CIs")
TR = {"CIFAR-10 gradual": ("c10gsig", "phaseT2r2_signal/r2_c10gsig_tv_s*.json", "phaseT2r2_signal/r2_c10gsig_hard_s*.json"),
      "CIFAR-100 gradual": ("c100gsig", "phaseT2r2_signal/r2_c100gsig_tv_s*.json", "phaseT2r2_signal/r2_c100gsig_hard_s*.json"),
      "CIFAR-100 spatial": ("c100sp", "phaseT2r2_signal/r2_c100sp_tv_s*.json", "phaseT2r2_signal/r2_c100sp_hard_s*.json"),
      "Tiny-ImageNet": ("tiny", "phaseT2r2_signal/r2_tiny_tv_s*.json", "phaseT2r2_signal/r2_tiny_hard_s*.json")}
rows = []
for name, (env, tvp, rtp) in TR.items():
    tv = {s: integ(load(f)) for s, f in famf(tvp).items()}; rt = {s: integ(load(f)) for s, f in famf(rtp).items()}
    fx = {"fixed 0.2": f"phaseT3_fixedref/t3_{env}_fx20_s*.json", "fixed 0.4": f"phaseT3_fixedref/t3_{env}_fx40_s*.json",
          "fixed 0.15": f"phaseT4_B3_fixedgrid/b3_{env}_fx15_s*.json"}
    for fl, fp in fx.items():
        fv = {s: integ(load(f)) for s, f in famf(fp).items()}
        if not fv: continue
        for al, a in (("TV", tv), ("rate", rt)):
            seeds = sorted(set(a) & set(fv)); d = np.array([a[s] - fv[s] for s in seeds]); lo, hi = ci95(d)
            rows.append([name, f"{al} - {fl}", f"{d.mean()*100:+.4f}", f"[{lo*100:+.2f},{hi*100:+.2f}]",
                         int((d > 0).sum()), len(d), f"{np.mean(list(fv.values()))*100:.4f}"])
wcsv("a3_transfer_paired.csv", ["setting", "comparison", "mean_pp", "ci95", "n_pos", "n_seeds", "fixed_mean_pct"], rows)

# ================================================================= A4
print("A4 F1 statistics (passive rec_* backbone)")
rows = []
for sched, label, shift_round in (("A", "shortened stepwise schedule (100 rounds)", None),
                                  ("abrupt", "late abrupt change (round 51)", 51)):
    for sigk, nm in (("TV", "tv_dist"), ("entropy_norm", "ent_client")):
        vals = []
        for s, f in famf(f"signal_benchmark/rec_{sched}_s*.json").items():
            h = load(f); x = np.array([np.mean([float(v) for v in d.values()]) for d in h["signals_per_es"][nm]])
            if nm == "ent_client": x = x / math.log(10)
            rho = np.array(h["rho_trace"], float)
            rec = dict(seed=s, total_decline=x[15] - x[-1], first_post_warm=x[15], last=x[-1])
            if shift_round:
                pre = x[shift_round - 6:shift_round - 1].mean(); post = x[shift_round:shift_round + 5].mean()
                rec.update(pre5=pre, post5=post, shift_delta=post - pre)
            else:
                chg = [i + 1 for i in range(1, len(rho)) if abs(rho[i] - rho[i - 1]) > 1e-9]
                rec.update(rho_change_rounds=" ".join(map(str, chg)))
            vals.append(rec)
        mean = {k: np.mean([v[k] for v in vals]) for k in vals[0] if isinstance(vals[0][k], float)}
        rows.append([label, sigk, len(vals), f"{mean.get('first_post_warm', float('nan')):.4f}", f"{mean.get('last', float('nan')):.4f}",
                     f"{mean.get('total_decline', float('nan')):+.4f}",
                     f"{mean.get('pre5', float('nan')):.4f}" if shift_round else "", f"{mean.get('post5', float('nan')):.4f}" if shift_round else "",
                     f"{mean.get('shift_delta', float('nan')):+.4f}" if shift_round else "",
                     vals[0].get("rho_change_rounds", "")])
wcsv("a4_f1_stats.csv", ["schedule", "signal", "n_seeds", "value_R16", "value_last", "decline_R16_to_last",
                         "pre_shift_mean(R46-50)", "post_shift_mean(R51-55)", "shift_delta", "rho_change_rounds"], rows)

# mobility membership-change rounds (from history)
rows = []
for s, f in famf("phaseT1_disjoint/t1_mob_dg_s*.json").items():
    h = load(f); rows.append([s, " ".join(f"{r}:{n}" for r, n in h.get("mobility_changes", []))])
wcsv("a1_mobility_change_rounds.csv", ["seed", "round:n_clients_rewired"], rows)
print("DONE ->", OUT)
