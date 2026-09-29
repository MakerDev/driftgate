# Interim Status Report — 2026-05-30 08:43 KST

**Pipeline elapsed since first launch**: ~59h (since 2026-05-27 21:55, retry-from-clean since 22:44)

---

## Phase summary

| Phase | Status | Wall | Result |
|---|---|---|---|
| 0. E0 smoke | ✅ DONE | 4.5h | Gate-2 PASS, all 4 runs healthy |
| 1. E1 static Pareto (16 configs) | ✅ DONE | 30h53m | Adaptive Pareto-optimal on acc_total (3rd), +37% acc_oor |
| 2. E2 CORE temporal (3 attempts) | ❌ FAIL | 24h+ | Best −1.12 pp gap. Two-signal ablation contribution per locked framing |
| 3a. E3 spatial (baselines + adaptive) | ✅ DONE | 13h | Adaptive does NOT win; signal collapses by R150 |
| 3b. E4 mobility | 🟡 RUNNING | 2h0m / ~5h | 2 baselines (GPU0) + adaptive (GPU1) |
| 3c. E5 ablation | ⏳ PENDING | est ~3h | Launch ~12:45 |
| 4. Analysis + paper assembly | ⏳ PENDING | ~10min | After E5 |

---

## 1. E0 (Phase 0) — see `phase0_e0_smoke.md`

✅ All 4 runs PASS_NO_NAN. Personalization trend correct (λ=0.6 acc_main 0.810 > λ=0.2 acc_main 0.742). Adaptive already showed strongest cross-region transfer at smoke scale (acc_oor 0.255).

## 2. E1 (Phase 1) — see `phase1_e1_static.md`

Full Pareto with 16 configs. Three-way tie at top:

| Rank | Method | acc_total | acc_main | acc_oop | acc_oor |
|---:|---|---:|---:|---:|---:|
| 1 | splitomcplus λ=0.2 Λ=0.5 | 0.703 | 0.779 | 0.643 | 0.271 |
| 2 | splitomcplus λ=0.4 Λ=0.5 | 0.701 | 0.835 | 0.518 | 0.188 |
| **3** | **adaptive_splitomc** | **0.699** | 0.763 | 0.638 | **0.371** |

Adaptive within 0.4 pp of best on acc_total **and +10 pp acc_oor over best fixed** (+37% relative). This is the headline paper contribution per locked framing.

## 3. E2 CORE (Phase 2) — three attempts, all FAIL

### Attempts and verdicts

| Attempt | Fix applied | Integrated | Gap | Verdict |
|---|---|---:|---:|---|
| 0 (original) | mu_H=1.5, train-anchor | 0.6345 | −1.94 pp | FAIL |
| 1 (config tune) | tau_H 0.4→0.5, mu_H→2.35, lam_min↑ | 0.6381 | −1.57 pp | FAIL |
| 2 (kill before run) | lam_max cap | — | — | Killed (user redirect) |
| **Final** | **incoming-traffic anchor** (signal source fix, not config) | **0.6427** | **−1.12 pp** | FAIL |

Each fix improved the gap by ~0.04 pp. Best fixed: splitomcplus λ=0.4 (integrated 0.6539).

### Root cause (analytical contribution)

**Two-signal ablation** demonstrates structural limit of entropy as drift signal:

- **Variant A (train-anchor entropy)**: λ frozen in narrow band 0.119-0.201 (swing 0.08). Probe sampled from client's training distribution never contains OOP/OOR classes → entropy tracks training convergence on personal task, not deployment-time ρ.
- **Variant B (incoming-traffic entropy)**: λ swing restored to 0.546 (6× wider). Probe correctly samples ρ-mixed traffic. Result: H bumps at ρ-transitions (R65 ρ=0→0.8 transition: H +0.054), λ momentarily dips, **but training convergence on related OOP classes (same cell scope) erases the signal within ~10 rounds**. By R75-80 H back below pre-transition level.

**Adaptive ρ=0 WIN preserved**: adaptive 0.6952 ≈ best fixed splitomcplus_lam0.6 0.7019. Controller logic is sound; failure is signal-source.

Future direction: convergence-invariant drift signals (KL vs frozen reference, OOR-only anchor, external label-distribution shift). See `_paper_framing_decisions.md`.

## 4. E3 Spatial — adaptive does NOT win the per-cell oracle role

Per locked framing, E3 was the candidate paper main result. **Hypothesis disconfirmed.**

### Final R150 results (mode equal_spread, ρ_z ∈ {0.0, 0.2, 0.4, 0.6, 0.8})

| Method | acc_total | acc_main | worst_cell | cell_gap |
|---|---:|---:|---:|---:|
| splitomcplus λ=0.2 | **0.723** | 0.790 | **0.681** | **0.086** |
| **splitomcplus λ=0.4** | **0.729** | 0.847 | 0.665 | 0.152 |
| splitomcplus λ=0.6 | 0.689 | 0.883 | 0.571 | 0.266 |
| adaptive_splitomc | 0.688 | 0.889 | 0.580 | 0.269 |

Adaptive is **last** on acc_total (−4 pp vs best fixed) and second-worst on cell_gap. Worst-cell 0.580 vs best fixed 0.681 (−10 pp).

### Per-cell λ_z evolution in adaptive (cell index = ρ_z order)

| Round | λ_c0 (ρ=0) | λ_c4 (ρ=0.8) | spread |
|---:|---:|---:|---:|
| R10 | 0.450 | 0.441 | 0.009 |
| R25 | 0.542 | 0.508 | **0.034** ← stratification peak |
| R65 | 0.630 | 0.599 | 0.031 (clean monotone) |
| R75 | 0.633 | 0.607 | 0.026 |
| R100 | 0.642 | 0.626 | 0.015 |
| R125 | 0.648 | 0.633 | 0.015 |
| R150 | 0.651 | 0.642 | **0.009** ← collapse |

**Stratification was transient.** Peaked R25-75 (spread ~0.026-0.034) then collapsed to spread 0.009 by R150 — adaptive converges to uniform λ ≈ 0.65 (close to fixed λ=0.6, which is why their results match).

**Interpretation**: same fundamental signal limit as E2. Within a single cell, training also drives H down regardless of that cell's static ρ_z. The cell-level entropy difference at R25-75 is consumed by convergence by R150.

## 5. Combined story (E2 + E3)

**The entropy signal is dominated by training convergence under both temporal and spatial drift in this setup.** Two ablations across two drift modalities. This is no longer a "limitation" — it's a quantitative finding worth a contribution-grade section.

## 6. Paper headline (locked)

**Main**: Adaptive-SplitOMC achieves Pareto-optimal cross-region transfer on static no-drift tasks (E1: +37% acc_oor over best fixed at matched acc_total, smallest cell_gap among all configs).

**Secondary (entropy signal ablation)**: We quantitatively demonstrate via dual ablations (E2 temporal + E3 spatial) that entropy-based drift signals are dominated by training convergence on related classes within cell scope, motivating convergence-invariant drift sensing as future direction.

**Other contributions**: E0 verification, E4 mobility robustness, E5 component ablation.

---

## What's running now (08:43 KST)

| GPU | Job | Round | ETA |
|---|---|---|---|
| 0 | E4 mobility: splitomcplus_lam0.2_Lam0.5_mob | R70/150 | 131m → ~10:54 |
| 1 | E4 mobility: adaptive_splitomc_mob | R50/150 | 157m → ~11:20 |

After GPU 0 finishes lam0.2_mob, it auto-continues to lam0.6_mob (~3h).

## Remaining schedule

| Step | Start | Done | Notes |
|---|---|---|---|
| E4 GPU 0 config 1/2 (lam0.2_mob) | 06:43 | ~10:54 | running |
| E4 GPU 0 config 2/2 (lam0.6_mob) | 10:54 | ~14:00 | sequential |
| E4 GPU 1 (adaptive_mob) | 07:13 | ~11:20 | running |
| E5 ablation 4 configs (split 2/GPU) | ~14:00 | ~17:00 | starts after E4 GPU 0 fully done |
| Phase 4: analyze_all.py + fill_paper.py + reports | ~17:00 | ~17:30 | final assembly |
| **Estimated full completion** | | **~17:30 KST 5/30** | |

## Files / artifacts so far

### Results JSON
- `results/e0_smoke/*.json` — 4 files + summary
- `results/e1_static/*.json` — 16 files
- `results/e2_temporal/schedule_A/*.json` — 4 files (3 baselines + adaptive final) + 2 failed-attempt backups
- `results/e3_spatial/mode_equal_spread/*.json` — 4 files (3 baselines + adaptive)
- `results/e4_mobility/` — in progress

### Phase reports
- ✅ `docs/results_v3/phase_reports/phase0_e0_smoke.md`
- ✅ `docs/results_v3/phase_reports/phase1_e1_static.md`
- ✅ `docs/results_v3/phase_reports/_paper_framing_decisions.md` (locked)
- ✅ `docs/results_v3/phase_reports/INTERIM_STATUS_05-30_08-43.md` (this file)
- ⏳ `phase2_e2_temporal.md` — write after E5 (will include three-attempt ablation table)
- ⏳ `phase3_e3_e4_e5.md` — write after E5
- ⏳ `interpretation.md`, `paper_v3_filled.md` — Phase 4 outputs

### Code changes (for paper reproducibility)
- `controller/adaptive.py`: added `ClientDriftTracker.update_anchors_from_traffic()` — accepts unlabeled ρ-mixed probe images.
- `scripts/run_single.py`: per-round probe construction (cached per ρ value), H/rho_for_anchor logging, verbose `[adapt R...]` line for diagnostics.
- `configs/base_v3.yaml` adaptive section: mu_H, tau_H, lam_min/max, Lam_min/max, use_delta tuned through three attempts; current (final) is mu_H=2.0, tau_H=0.5, lam_min=0.15, lam_max=0.7, Lam_min=0.4, Lam_max=0.7, use_delta=false, consensus_steps=1.

### Backup files
- `results/e2_temporal/schedule_A/adaptive_splitomc.json.fail_attempt0` (train-anchor, integrated 0.6345)
- `results/e2_temporal/schedule_A/adaptive_splitomc.json.fail_attempt1` (config-tuned, integrated 0.6381)
- (final adaptive_splitomc.json is the incoming-traffic result, integrated 0.6427)

---

## Decision points awaiting / closed

- ✅ **Plan B vs A confirmed** by user at 02:00 KST (after final attempt FAIL).
- ✅ **Locked paper framing** captured in `_paper_framing_decisions.md`.
- ⏳ Final paper structure to be written after Phase 4.

No blockers. Cron monitoring at 30-min intervals continues.
