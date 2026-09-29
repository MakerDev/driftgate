# Phase 1 — Calibration-Leakage Audit & Self-Calibrating Controller

**Date**: 2026-07-11 · **Status**: leakage documented (complete); self-calibrating controller
implemented + unit-tested; downstream pilot queued behind Gate-B passive runs.

---

## 1. The leakage finding (RQ3 motivation)

### 1.1 Documented `mu_drift` history

The hand-calibrated disagreement controller's sigmoid center was re-chosen **three times**,
each time using observations made **on the final test schedule (Schedule A)** with knowledge
of which rounds carried which ρ:

| Step | Value | Basis (verbatim from v4 records) | Leakage type |
|---|---|---|---|
| 1 | `mu=0.392` | STEP-3 calibration: `(delta_lo + delta_hi)/2` where `delta_lo` = δ measured on the ρ=0 rounds and `delta_hi` = δ on the ρ=0.8 rounds of a 90-round Schedule-A gate run | uses **true ρ segment labels** of the test schedule to aggregate δ |
| 2 | `mu=0.35` | re-centered after inspecting the full E2 run: "sustained ρ=0.8 δ settles at ~0.37, just below mu=0.392" (phase2b §5.1) | uses **test-run outcome** under known ρ |
| 3 | `mu=0.31` (final) | config comment: "sustained high-ρ δ≈0.38, low-ρ δ≈0.13 … Sharp separation **for worst-cell**" | uses test observations **and** a downstream metric target |

`tau_drift = 0.045` was likewise adjusted (0.028 → 0.045) using λ-trace behavior *within
known constant-ρ segments* of the same schedule.

### 1.2 What this does and does not invalidate

- It does **not** invalidate the signal-level comparison (E5 +2.43 pp): both entropy and
  disagreement controllers received the same kind of hand calibration (entropy: mu_H=2.0,
  tau_H=0.5, also hand-set), and the E5 ablation swaps only the signal.
- It **does** weaken every "adaptive vs fixed" claim: the adaptive controller's operating
  points were partially fitted to the very schedule it is evaluated on, while the "best fixed
  λ" comparison is itself a hindsight selection. Neither side of that table is deployable as-is.
- Consequence: the journal version must lead with a controller whose constants are
  **not functions of the test environment**.

## 2. The fix — three cleanly separated controllers

| # | Controller | Constants come from | Role in paper |
|---|---|---|---|
| 1 | Hand-calibrated (legacy, mu=0.31) | test schedule observations | reported for continuity, labeled leaky; upper reference for "how much the leak bought" |
| 2 | Source-calibrated | ONE dev environment (CIFAR-10 Schedule A, seeds 0-2), then frozen and transferred | measures calibration transfer |
| 3 | **Self-calibrating online** (main candidate) | nothing environment-specific; dimensionless pre-registered constants | RQ3 answer |

### 2.1 Self-calibrating design (implemented: `src/controllers/{normalizers,self_calibrating}.py`)

Per edge server, on the raw per-ES disagreement δ_z:

1. **Warm-up** (W=15 rounds): controller outputs neutral λ=(λ_min+λ_max)/2; baseline
   μ̂=median, σ̂=1.4826·MAD of the warm-up observations. Warm-up uses only the first W
   rounds of whatever traffic arrives — no ρ labels.
2. **Guarded robust z**: z_t=(δ_t−μ̂)/σ̂. Baseline (μ̂,σ̂) EWMA-adapts (β=0.05) **only when
   z_t < 1.5**: it tracks slow convergence-driven decline of δ but freezes under upward
   drift — this is what preserves sustained-drift retention without an absolute threshold.
3. **Noise floor**: σ̂ ≥ 0.10·|μ̂| (dimensionless), preventing warm-up MAD under-estimation
   from amplifying probe noise into λ oscillation (audit test 9).
4. **Smoothing + consensus**: EWMA(α=0.3) on clipped z ∈ [−2,6]; 1-step neighbor consensus
   (same ES graph as legacy).
5. **Mapping in z-space**: λ_z = λ_max − (λ_max−λ_min)·σ((z̃−1.5)/0.75); Λ analogous.
   z̃≈0 → personalize; z̃≥3 → generalize. No dataset units anywhere.

All constants were fixed **before** any Gate-B benchmark run was analyzed (pre-registered in
this document and in code comments; the only post-hoc change was the σ-floor + warm-up
boundary fix, made to satisfy the pre-written audit tests 9/14 — before any GPU benchmark
results existed).

### 2.2 Normalizer variants implemented (S4 family, for the benchmark)

`guarded` (main), `anchored` (warm-up-frozen baseline), `ewma` (unguarded — expected to
decay under sustained drift; kept as an honesty baseline), `rollq` (rolling quantile),
`cusum` (accumulated change). All strictly causal; online ≡ offline replay (tested).

## 3. Calibration protocol going forward (binding)

1. No constant of controller #3 may be changed after Gate-B analysis begins. If it fails,
   the failure is reported and any revision becomes controller #3b with fresh test schedules.
2. Source calibration (#2) may only use: CIFAR-10, Schedule A, seeds {0,1,2}.
   Everything else (other schedules, CIFAR-100, Tiny-ImageNet, mobility, spatial) is unseen.
3. Absolute-threshold and self-normalized variants are always reported side by side.
4. Hindsight-oracle fixed baselines are marked "upper reference (not deployable)" in every table.
