---
name: project_paired_shadow_experiment
description: "E2 Schedule-A paired shadow-signal experiment (fixed-policy, same-run H vs delta) — status, design, pending analysis"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4bee259a-e7b1-4d93-ab3c-44d3b8dd5bf2
---

Goal: remove the cross-run confound of the earlier respective-run signal overlay by measuring predictive entropy H and exit disagreement δ from the SAME model state + SAME probe + one forward, on ONE fixed-policy trajectory. Also switch all signal figures from single-R25 to R21–R30 (last-10-pre-drift) mean normalization. Do NOT rewrite paper prose yet.

**Built (all in adaptive_splitomc_v4, git e82b96b):**
- `controller/shadow.py` — paired H+δ from one forward, canonical formulas (H=client-exit entropy; δ=client-vs-mean-server argmax disagreement). ES aggregate unweighted + sample-weighted (weighted==unweighted since every probe is size 16).
- `scripts/run_paired_signal_schedule_A.py` — fixed-policy driver (splitomcplus λ=0.2 Λ=0.5 = repo non-oracle default). Shadow block is RNG save/restore + eval+inference_mode. Deterministic mode ON (use_deterministic_algorithms, cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8). Saves history JSON + manifest + probe_manifest + boundary checkpoints (s100 only).
- `configs/e2_temporal/schedule_A_paired_fixed_shadow.yaml`
- `tests/test_shadow_signal_nonperturbation.py` (test_shadow_reproducible_and_bounded)
- `scripts/run_paired_signal_schedule_A.sh` (3-seed launcher)
- `scripts/audit_paired_signals.py`, `scripts/make_fig_signal_decay.py`, `scripts/verify_offline_paired.py`

**Non-perturbation VERDICT = NON_PERTURBING_FOR_PURPOSE** (see `shadow_logger_nonperturbation.md`). Determinism works (A-vs-A′ byte-identical); shadow-ON production config bit-reproducible incl H&δ (B-vs-B′ identical); shadow-off vs shadow-on differ only ≤1.7e-3 loss (reproducible cuDNN algo-selection effect from shadow's GPU allocations, NOT RNG/param leak, NOT nondeterminism). Acceptable: all signals measured on the actual reproducible trajectory; pairing exact.

**DONE 2026-07-15. VERDICT = `PAIRED_SIGNAL_CONFIRMED`** (3 seeds R150, 4.8-4.9h each). See `paired_signal_experiment_report.md` (16 sections) + `paired_signal_audit.json` + 3 CSVs. Offline verify R31 & R90 EXACT (0.00e+00). C1/C2/C3 all pass, all-seed sign-consistent.

**Key paired numbers (R21-R30 norm, 3-seed, USE THESE — supersede respective-run for the same-run claim):**
- δ segment means ordered by ρ: pre 1.00 / drift1(0.4) 1.199 / severe(0.8) **1.347±0.02** / drift2(0.4) 1.164 / post 0.773. δ ≥1.0 in **100%** of active-drift rounds (all seeds).
- H segment means monotone in TIME: 1.00/0.977/0.941/0.887/0.820. H≥1.0 in ~2% of drift rounds.
- Transitions (δ vs H, 5R rel, all sign-consistent 3 seeds): onset +24.7% vs +0.6%; sev↑ +10.7% vs −0.2%(H not even same-sign); sev↓ −13.6% vs −3.0%; removal −31.6% vs −4.4%. Δδ−ΔH>0 magnitude at every transition, all seeds.
- Spearman: H·round=−0.992, H·ρ=**+0.013**(≈0); δ·round=−0.506, δ·ρ=**+0.622**.
- δ drift peak agg 1.445×@R82 but **round UNSTABLE across seeds (82/64/67)** → figure omits round peak annotation; stable summary = severe-seg 1.35×. Do NOT reuse old 1.42×.
- weighted≡unweighted (all probes size 16, max diff 0.0).

**Figures:** paper-facing `figure/signal_decay.pdf` = CLEAN paired fig `figure/signal_decay_paired_R21_R30_clean.pdf` (sha **ff5af218**, regenerated 07-15 from CSV: main axis R21-150 warm-up omitted, legend above 2-col, solid=δ blue #0072B2 / dashed=H orange #D55E00, SD bands alpha0.12 no-edge, R121 line, Nimbus Roman Type-42, pure vector no raster, 3.45×2.17in). Superseded first paired fig b584597c. `make_fig_signal_decay_clean.py` validates CSV vs audit (PASS). `figure/signal_decay_respective_runs_R21_R30.{pdf,png}` (supplementary/diagnostic, needs "respective runs" caveat). Archive `figure/archive/signal_decay_before_paired_experiment.pdf` sha 4bb22dc9.

**Paper prose NOT yet rewritten** (per task). Too-strong claims to avoid: "entropy blind/monotonic", "δ never decays", "δ precisely measures severity", "peak 1.42×/round-82". See report §15.

**Figure 1 rework already done:** `figure/signal_decay_respective_runs_R21_R30.{pdf,png}` (respective-run diagnostic, R21–R30 base, peak 1.42× since R21–R30 mean=0.34088 ≈ R25). See [[project_fig1_signal_decay_numbers]].
