# Phase 2b — Disagreement-Signal Replacement (STEP 0–5 Report)

> **Note (implementation + gate record).** This documents the signal design, the G1/G2/G3 gates, and the **initial mu=0.392** STEP-5 result (+0.7 pp). Calibration was later finalized at **mu=0.31** and all phases re-run. **Canonical final results: `Integrated_R150.md` / `Integrated_R100.md`.** Kept for the mechanism/gate methodology.

**Run window**: 2026-05-30 09:20 → 15:49 KST
**Branch / dir**: `adaptive_splitomc_v4`
**Status**: ✅ STEP 0–4 complete · gates G1/G2/G3 PASS · **E2 STEP 5 = PARTIAL** (+0.7 pp vs best fixed, +1.8 pp vs entropy)

> **One-line result**: The entropy drift signal that collapsed at training convergence (E2 R75–80, E3 R150) was replaced by a **client-vs-server exit-disagreement signal** that does *not* collapse (gate: R85 δ = 1.47× R25, where entropy died). On the full E2 run this lifts integrated accuracy **+1.8 pp over the failed entropy controller** and recovers the ρ=0.8 collapse (+6.3 pp), landing at **+0.7 pp vs best fixed = PARTIAL** (headline PASS needed +2 pp). The residual gap is a calibration-center issue, not signal death.

---

## 0. Motivation (why the signal was replaced)

The entropy-based adaptive controller failed E2 and E3 by the same root cause:

- **Entropy dies at convergence.** As the client head's predictions sharpen during training, Shannon entropy on incoming traffic decays toward zero regardless of how much OOR/OOP drift is actually present. The controller loses its drift signal exactly when it matters.
- Confirmed empirically: E2 stratification collapsed at **R75–80**, E3 per-cell λ stratification collapsed at **R150**.

**Replacement hypothesis (design §4/§6):** The *client head never learns OOR/OOP classes* — they are not its task and it has no labels for them. So the **rate at which the client head and server head disagree** on incoming traffic stays informative even after the client head converges on its own task. This signal should be **convergence-invariant**.

---

## 1. Implementation (signal SOURCE swap only, ~75 lines)

Strict rule honored: **only the signal source changed** (entropy → disagreement). Consensus, per-ES aggregation, and the sigmoid mapping are untouched (they operate on a generic scalar). The entropy path is **fully preserved** for the C2 ablation.

### `controller/adaptive.py`
- `ClientDriftTracker.compute_disagreement(client_model, server_models, probe_x) → (delta_k, margin_k)`
  - Copies the two-head pattern from `eval/evaluator.py:60–69`, label removed:
    - `client_logits, rep = client_model(probe_x)` (eval, `no_grad`)
    - `server_logits = mean_z( server_model_z(rep) )`
    - **S1** `delta_k = (client_pred != server_pred).float().mean()`
    - **S2** `margin_k = 1 − mean(top1 − top2 of softmax(server_logits))` *(computed + logged; unused at β=0)*
- `gather_signals_for_clients(..., signal_mode='entropy'|'disagreement', beta_margin=0.0)`
  - `disagreement` → `drift_k = delta_k + beta_margin·margin_k` in the primary signal slot (S1-only start).
  - `entropy` path kept verbatim.

### `scripts/run_single.py`
- Reads `signal_mode` / `beta_margin` from config; injects `mu_drift`/`tau_drift` into the controller for disagreement mode (vs `mu_H`/`tau_H` for entropy). Mapping code unchanged.
- Per-round verbose log: `[adapt R{r} ρ=..] delta_z=.. margin_z=.. λ=..`
- History gains `delta_per_es`, `margin_per_es` traces (append symmetry verified 3:3:3).
- New CLI override `--signal_mode {entropy,disagreement}` (for E5 3-way ablation / backups).

### `configs/base_v3.yaml` (adaptive section)
- `signal_mode: disagreement`, `beta_margin: 0.0`
- `mu_drift`, `tau_drift` (calibrated in STEP 3 — see below)
- **Unchanged**: `lam_min 0.15 / lam_max 0.7 / Lam_min 0.4 / Lam_max 0.7`; `mu_H/tau_H` retained for the entropy ablation.

### Verification
- `pytest tests/test_all.py` → **12 passed** (existing structure intact).
- 3-round smoke (GPU 1, no OOM): disagreement path logs `delta_z`/`margin_z` correctly.

---

## 2. STEP 2 Gate — signal validation (90-round run, Schedule A)

A short run (no 5 h commitment) measuring `delta_z` across the ρ schedule. Schedule A uses **absolute 30-round segments**:
R1–30 ρ=0.0 · R31–60 ρ=0.4 · **R61–90 ρ=0.8** · (R91–120 ρ=0.4 · R121–150 ρ=0.0 in the full run).

> Note: the true ρ=0 *converged* window is R20–30 (not R55, which is ρ=0.4). Measured accordingly.

### `delta_z` (client–server disagreement rate) per round

| Round | ρ | delta_z avg | per-cell |
|---:|---:|---:|---|
| R20 | 0.0 | 0.366 | 0.38 0.43 0.33 0.35 0.34 |
| R25 | 0.0 | **0.320** | 0.34 0.35 0.28 0.25 0.38 |
| R30 | 0.0 | 0.322 | 0.36 0.37 0.28 0.26 0.34 |
| R40 | 0.4 | 0.409 | 0.39 0.40 0.44 0.41 0.40 |
| R55 | 0.4 | 0.403 | 0.38 0.45 0.48 0.36 0.35 |
| R65 | 0.8 | 0.466 | 0.46 0.49 0.45 0.47 0.46 |
| R75 | 0.8 | 0.394 | 0.38 0.38 0.40 0.36 0.44 |
| R85 | 0.8 | **0.470** | 0.38 0.48 0.53 0.51 0.46 |

Aggregates: `delta_lo` = delta(ρ=0, R20–30) = **0.336** · `delta_hi` = delta(ρ=0.8, R65–85) = **0.448**.

### Gate verdicts

| Gate | Measurement | Criterion | Verdict |
|---|---|---|:--:|
| **G1 — separation** | delta(ρ=0.8) − delta(ρ=0) = 0.448 − 0.336 = **+0.112** | ≥ +0.10 | **PASS** |
| **G2 — convergence-invariance** ★ | delta@R85 / delta@R25 = 0.470 / 0.320 = **1.47** | ≥ 0.8 | **PASS** |
| **G3 — reactivity** | λ(ρ=0)=0.379 → λ(ρ=0.8)=0.254 (drift↑ ⇒ λ↓) | λ decreases | **PASS** |

### ★ The decisive point
Entropy died at R75–85. **Disagreement at R85 is 1.47× its R25 value — it is *higher* after convergence, not lower.** When ρ=0.8 traffic arrives, the client head (own-task-only, no OOR/OOP labels) and the server head split more often, and that split survives convergence. The design premise (§6) is empirically confirmed. β_margin (S2) was not needed — G1 passes with S1 alone.

---

## 3. STEP 3 — Calibration (measured, no guessing)

From the observed range [delta_lo = 0.336, delta_hi = 0.448]:

| Param | Formula | Value |
|---|---|---|
| `mu_drift` | (delta_lo + delta_hi) / 2 | **0.392** |
| `tau_drift` | initial (delta_hi − delta_lo) / 4 = 0.028 → **adjusted to /2.5** | **0.045** |

### tau adjustment (sigmoid-steepness check)
A `tau` of 0.028 risks turning the sigmoid into a near-step function. Re-mapping the **recorded per-round δ** through the sigmoid (the gate log's λ column used the *placeholder* tau=0.1, so it looked smooth and was not a valid test) showed, **within the constant ρ=0.8 segment** (where λ should be stable):

| tau | λ_avg swing (const ρ=0.8) | mean \|Δλ\|/round | max \|Δλ\| | hits 0.15/0.7 rail |
|---|---|---|---|---|
| 0.028 (/4) | 0.19 ↔ 0.42 | 0.109 | 0.227 | yes (per-cell) |
| **0.045 (/2.5)** | 0.24 ↔ 0.42 | 0.086 | 0.178 | no |

At tau=0.028 the 16-sample probe's δ noise (±0.07/round) spans 2.5·tau, so the sigmoid amplifies sampling noise into λ ping-pong (R65=0.19 → R75=0.42 → R80=0.20) at *constant* ρ — the step-function failure mode. **tau_drift = 0.045** halves the within-segment swing with no rail-hits, while preserving G1 separation (ρ=0 λ≈0.58 vs ρ=0.8 λ≈0.27).

**Final calibrated controller**: `mu_drift=0.392, tau_drift=0.045, beta_margin=0.0` (lam/Lam bounds unchanged).

---

## 4. STEP 4 — E2 disagreement full run (✅ complete, 206 min)

- **Backup**: entropy E2 result preserved as
  `results/e2_temporal/schedule_A/adaptive_splitomc_entropy.json` (150R, verified: H 149 entries, delta 0) — the **C2 ablation baseline**.
- **Launched**: `run_e2_temporal.py --schedule A --methods adaptive_splitomc` on **GPU 1**, detached (PPID=1), unbuffered. Reads calibrated config.
- **ETA**: ~4.5–5 h (eval_every=10; gate run measured ~1.75 min/round → 150R).
- Output regenerated canonical `results/e2_temporal/schedule_A/adaptive_splitomc.json` (150R, eval_every=10, PPID=1, no OOM). Wall: 206 min.

---

## 5. STEP 5 — E2 Gate re-evaluation — **VERDICT: PARTIAL**

### 5.1 λ-trace direction (mechanism check)

Per-cell λ from the full run at representative rounds (cells 0→4):

| Round | ρ | δ_z avg | λ (cell 0→4) | λ avg |
|---:|---:|---:|---|---:|
| R30 | 0.0 | 0.351 | 0.343 0.432 0.571 0.630 0.644 | **0.524** |
| R65 | 0.8 | 0.494 | 0.191 0.194 0.194 0.210 0.217 | **0.201** |
| R75 | 0.8 | 0.378 | 0.394 0.409 0.493 0.516 0.550 | 0.472 |
| R90 | 0.8 | 0.371 | 0.369 0.388 0.422 0.558 0.606 | 0.469 |
| R135 | 0.0 | 0.121 | 0.698 0.698 0.699 0.699 0.699 | 0.699 |

**Direction correct but with a mid-segment relaxation:**
- ✓ At ρ=0.8 **onset** (R65) λ dips hard to **0.20** while δ peaks at 0.49 — the intended reflex.
- ⚠ Within the **sustained** ρ=0.8 segment λ relaxes back to ~0.47 (R75/R90). Cause: sustained ρ=0.8 δ settles at ~0.37, *just below* the calibrated center `mu_drift=0.392`, so the sigmoid returns mid-range λ. The center is ~0.02–0.03 too high vs the *converged* ρ=0.8 δ level.
- ✓ ρ=0 segments: λ high (R30 0.52, R135 0.70) — correct personalization.

**★ G2 on the full run — holds vs entropy.** δ at ρ=0.8 stays alive through the segment (R65 0.49 → R90 0.37), never collapsing. Entropy died here. Caveat: δ *does* decline across the run (ρ=0 δ: R25 0.32 → R135 0.12), so disagreement is convergence-*resistant*, not perfectly *invariant* — enough to beat entropy, not enough to fully pin λ low under sustained drift.

### 5.2 ρ=0.8 segment recovery

| Method | ρ=0.8 acc |
|---|---:|
| **adaptive (disagreement)** | **0.6076** |
| adaptive (entropy) | 0.5450 |
| best fixed @ρ=0.8 (λ=0.2) | 0.6294 |

→ **+6.3 pp over entropy** (collapse largely repaired) but **0.608 < target 0.62**, still under best fixed. The mid-segment λ relaxation leaves accuracy on the table.

### 5.3 Integrated comparison (mean over all eval rounds)

| Method | **integrated** | ρ=0 | ρ=0.4 | ρ=0.8 |
|---|---:|---:|---:|---:|
| **adaptive (disagreement)** | **0.6609** | 0.6990 | 0.6430 | 0.6076 |
| adaptive (entropy) | 0.6427 | 0.6952 | 0.6303 | 0.5450 |
| fixed λ=0.2 | 0.6435 | 0.6350 | 0.6606 | 0.6294 |
| **fixed λ=0.4 (best fixed)** | 0.6539 | 0.6687 | 0.6576 | 0.6118 |
| fixed λ=0.6 | 0.6453 | 0.7019 | 0.6273 | 0.5494 |

- Δ(disagreement − best fixed) = **+0.70 pp** → in [−0.5, +2) band ⇒ **PARTIAL**
- Δ(disagreement − entropy) = **+1.82 pp** → signal swap is a clear, measurable win over the failed entropy controller.

**Why PARTIAL not PASS:** the controller now wins at ρ=0 (0.699, best of all) and is 2nd at ρ=0.4, and no longer collapses at ρ=0.8 — but the slightly-high `mu_drift` lets λ drift up under sustained ρ=0.8, so it doesn't fully claw back the high-drift segment. Net integrated beats best-fixed by +0.7 pp (headline threshold was +2 pp).

### Honest framing
The **root-cause fix worked**: disagreement does not die at convergence (gate G1/G2/G3 passed; G2 re-confirmed on the full run), and it lifts E2 by **+1.8 pp over the entropy controller that failed three times**. The remaining gap to a +2 pp headline is a **calibration-center** issue (sustained-drift δ ≈ 0.37 vs mu = 0.392), **not** a signal-death issue — qualitatively milder than the entropy collapse. A single re-centering of `mu_drift` (≈0.36) is the obvious next lever, but per the no-tuning-loop guardrail this is **reported, not auto-applied**.

---

## Guardrails honored
- E4/E5 entropy results preserved (C2 analysis) — not touched at this stage.
- No 5 h run before the STEP 2 gate passed.
- G2 (the entropy death point) **passed** — no 4th attempt needed; Plan B not triggered.
- No config-tuning loop: signal swap + measurement-based calibration only. (The single tau adjustment was a measured sigmoid-steepness correction, not a blind sweep.)

---

## Artifacts
- Gate run JSON: `results/gate/gate_disagree_A.json` (90R, δ/λ/margin traces)
- **E2 disagreement full run**: `results/e2_temporal/schedule_A/adaptive_splitomc.json` (150R, δ/λ/margin/eval)
- Entropy baseline (C2): `results/e2_temporal/schedule_A/adaptive_splitomc_entropy.json`
- Fixed baselines: `splitomcplus_lam0.{2,4,6}_Lam0.5.json`
- Code: `controller/adaptive.py`, `scripts/run_single.py`, `configs/base_v3.yaml`

---

## Open items / next levers — ALL COMPLETED (this section is historical)
1. ~~Re-center `mu_drift`~~ → DONE: 0.392 → 0.35 → **0.31** (final, locked).
2. ~~E3 disagreement run~~ → DONE (stratification held to R150, spread 0.227).
3. ~~E5 3-way signal ablation~~ → DONE (disagreement +2.43 pp over entropy; margin hurts, β=0 confirmed). Plus E4 disagreement DONE.

**→ Canonical final cross-phase results: `Integrated_R150.md` (primary) and `Integrated_R100.md`. This Phase-2b doc is the implementation/gate record only.**
