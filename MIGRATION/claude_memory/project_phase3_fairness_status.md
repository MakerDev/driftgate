---
name: project_phase3_fairness_status
description: "Phase 3 (PATH-2 fairness-aware cell weighting, SplitOMC-ACF) status in v5"
metadata: 
  node_type: memory
  type: project
  originSessionId: 48dbf36e-bbd5-4de4-b9d1-08ea2a8eb3c4
---

**Phase 3 = PATH-2: fairness-aware cell weighting** in NEW branch `/disk2/Yujin/adaptive_splitomc_v5` (copied from v4; all E1–E5 + entropy/disagreement backups preserved). Goal: lift worst-cell above best fixed λ via a 2nd DOF (drift-aware cell aggregation weight) on top of disagreement λ. Reuses existing δ_z; ONE new hyperparam η (fairness_eta). Else locked (mu_drift=0.31, tau_drift=0.045, lam[0.15,0.7], Lam[0.4,0.7], beta_margin=0, signal_mode=disagreement).

**STEP 1 (impl) — DONE & verified:**
- `train/trainer.py`: `weighted_average()`; `apply_network_aggregation(..., drift_per_cell, eta, Lam_min)`: η>0 → (a) drift-weighted global pool softmax(η·zscore(δ)) + (b) Λ_eff=max(Lam_min, Λ−η·norm01(δ)) so worst/high-δ cells absorb more global. `run_one_global_round` threads drift_per_cell/fairness_eta/Lam_min.
- `scripts/run_single.py`: reads adaptive.fairness_eta + Lam_min; builds cur_drift_per_es (=delta_per_es in disagreement mode) per round; passes through. CLI `--fairness_eta`. η=0 → use_fair False → original path EXACTLY.
- `configs/base_v3.yaml`: `fairness_eta: 0.0`.
- NEW `scripts/run_phase3_fairness.py`: self-contained (--exp e2/e3/e4 --eta --rounds --run_name --output_dir). Avoids the hardcoded-CONFIGS --methods filter bug from Phase 2 E4.
- pytest 12/12. weighted_average≡uniform (equal & zero w)=True. η=0 = unchanged branch.

**η=0 baseline (= v4 Integrated_R150): E3 worst_cell@R150=0.6319; best fixed worst = λ=0.2 0.6812, λ=0.4 0.6752. Headline target: η>0 worst_cell beats ~0.68.**

**STEP 2 gate — DONE, FAILED, STOPPED (user said "지금 STOP"). NO η=2 retry, no full run.**
E3 equal_spread η=1.0 100R vs η=0 baseline (@R100): worst 0.6445→0.6533 (**GF1 +0.88pp, FAIL** need +1; R80 even −2.35pp), best 0.831→0.760 (**GF2 −7.06pp, FAIL** need ≥−3), acc 0.7177→0.7004. GF3 (no NaN) pass. η=1 worst 0.6533 < best fixed worst (λ=0.2 0.6812, λ=0.4 0.6752) → headline NOT met.
**Why:** fairness EQUALIZES cells (best down 7pp, worst up <1pp) at lower overall acc; GF1 & GF2 fail in OPPOSITE directions (η↑ worsens best, η↓ worsens worst) → no single η works. Worst-cell deficit is TASK-INTRINSIC, survives BOTH DOFs (within-cell λ Path1, cross-cell weight Path2). Stronger negative result.
**Path 3 confirmed:** keep C1(disagreement)+C2(entropy) lead; §6.7 limit strengthened ("worst-cell deficit robust to two adaptation mechanisms → structurally hard, not under-expressive controller"). adaptive ≈ fixed without tuning, doesn't surpass worst-cell.
**Report: `docs/PATH2_fairness_weighting_report.md` (v5, the only Phase-3 report; v4 untouched).** Implementation kept in v5 (works, η=0≡v4). No further Phase-3 experiments unless user asks.

**Guardrails:** v5 separate; v4 reports untouched. NEW report → `docs/PATH2_fairness_weighting_report.md`. Compare ALL fixed λ (Phase-2 lesson: λ=0.4 best fixed, don't omit). η only tunable. worst won't rise → honest Path-3. Relates to [[project_disagreement_signal_result]].

⚠ ENV: bash stdout display flaky this session (lag/empty) — write to files & Read. pkill rc=1 cascade-cancels batched calls; run ONE cmd at a time, append `; true`. setsid + PPID=1 for long runs ([[feedback_long_running_jobs]]).
