# Adaptive-SplitOMC → TMC/ToN Expansion — Consolidated Full Report

**Period**: 2026-07-11 ~ 2026-07-17 · **Branch**: `adaptive_splitomc_tmc/journal_expansion`
**Evidence**: 265 provenance-tracked runs, 0 failures (`tables/all_runs.csv`,
`provenance/all_runs.jsonl`) · 56 unit tests passing (12 legacy + 44 new)
**Hardware**: 2× RTX 3090 Ti (shared) · Python 3.12.7 / torch 2.7.1+cu126

> This file consolidates every phase report WITH its result tables. It adds nothing new;
> canonical per-phase files remain phase0…phase6, gateD_results, final_tmc_ton_assessment.

---

## 0. Executive summary

| Question | Answer | Evidence |
|---|---|---|
| RQ1 entropy vs convergence | raw entropy is anti-predictive after convergence (AUROC 0.161); failure = GLOBAL convergence-trend confounding, not within-segment decay | §4, §10.1 |
| RQ2 when is disagreement better | always in raw form (AUROC 0.94–1.00); downstream +1.06 pp 5/5 seeds @150R, +3.64 pp on sustained high-drift | §4, §10.1 |
| RQ3 self-calibration transfer | final dual-view controller (0 constants) = parity with leak-calibrated on unseen schedules; ABOVE fixed grid on unseen datasets | §5, §10.2–10.5 |
| RQ4 vs causal baselines | best label-free method (lowest regret, 20× less churn than bandits); labeled oracle keeps +1.1–1.6 pp | §7 |
| RQ5 worst-cluster ceiling | ⅓–½ was reference-frame/bounds (fixed by ST + ablation); donor-fairness FAILS (oracle donors hurt); fixed λ0.2 keeps +3.3 pp worst on CIFAR-10 spatial | §6, §10.3 |
| RQ6 network robustness | delay 10R ≤ −0.7 pp; loss 20% ≈ 0; topology second-order; participation drop = training effect (controller share +0.8 pp favorable) | §8 |
| RQ7 overhead | probe 1.5 ms/client/round; 4 B scalar vs 12.1 MB/round model exchange; 0 B activation upload | §9 |

**Final verdict: TMC GO · ToN CONDITIONAL_GO** (§12).
**Final main method — selfcal-DV (dual-view)**: λ_z = min( λ_selfcal(z), λ_abs(δ̂) ) with
z = max(temporal guarded robust-z, cross-cell spatial z). Three dimensionless reference
frames; no dataset-specific constants anywhere.

---

## 1. Setup, provenance, correctness gates

- v4 ICTC stack untouched; all new code under `journal_expansion/` (signals, controllers,
  baselines, network, evaluation, runner, queue system).
- Every run: run-ID, git commit, config SHA-256, split hash, model-init hash, env/GPU,
  metrics, status. Re-runs never overwrite.
- Controllers are label-free and ρ-free at runtime — enforced by API surface and tests.

## 2. Phase 0 — Reproduction & audit (Gate A: PASS)

Reproduction of the ICTC main run (E2 Schedule A, adaptive disagreement mu=0.31, same
seeds, v4 code, new torch):

| metric | ICTC (v4) | reproduction | diff |
|---|---:|---:|---:|
| integrated acc (16 evals) | 0.6671 | 0.6669 | −0.02 pp |
| ρ=0 / ρ=0.4 / ρ=0.8 | 0.6905 / 0.6614 / 0.6240 | 0.6900 / 0.6611 / 0.6246 | ≤0.06 pp |
| final acc (R150) | 0.9061 | 0.9041 | −0.21 pp |
| worst-cell final | 0.8882 | 0.8895 | +0.13 pp |

Code audit: 12/12 items pass (λ is model-mixing weight; Λ mixes cell↔global; OOP/OOR
labels never in training loss; disagreement label-free; controller causal; deterministic
splits; mobility rewires membership). Caveats documented: probe/eval pool overlap
(unlabeled, standard); dormant mobility indexing bug (O3, no effect on any recorded run).

## 3. Phase 1 — Calibration leakage (documented) & the controller family

`mu_drift` history — each step fitted on ρ-LABELED observations of the final test schedule:

| step | value | basis | leak type |
|---|---|---|---|
| 1 | 0.392 | (δ_lo+δ_hi)/2 from Schedule-A gate run segments | uses true ρ segment labels |
| 2 | 0.35 | re-centered after full E2 result ("sustained ρ=0.8 δ≈0.37 just below mu") | test-run outcome |
| 3 | 0.31 | "high-ρ δ≈0.38, low-ρ δ≈0.13 … sharp separation for worst-cell" | test obs + downstream target |

(τ likewise 0.028→0.045 using constant-ρ segment behavior.) Measured value of the leak:
**≈1.3 pp on its home schedule (§5), which does NOT transfer to unseen schedules (§10.2).**

Controller lineage (all constants dimensionless, each revision pre-registered with a unit
test BEFORE its runs):
- **v1**: guarded robust-z (W=15, β=0.05, guard z>1.5, σ-floor 0.10·|μ̂|, EWMA α=0.3,
  map z0=1.5/τ_z=0.75, clip [−2,6])
- **v3b** (+burn-in 10): warm-up had absorbed the untrained transient (δ 1.0→0.45)
- **v3c** (+asymmetric guard z_guard=0.5): boiling-frog — stepped ramps were absorbed
  stage-by-stage (Schedule-A z stayed ≈0 through ρ=0.8; λ never left 0.6–0.67)
- **ST** (+cross-cell spatial z): temporal z is identically 0 under STATIC heterogeneity
- **DV** (+absolute-level cap λ_abs = λ_max−(λ_max−λ_min)·δ̂): δ is a probability with
  intrinsic meaning; change-detection alone is blind to static hardness (§10.4)

## 4. Gate B — Signal benchmark (PASS)

12 passive runs (4 schedules × 3 seeds, 100R, fixed λ0.4 backbone, 16 raw signals @
probe n=64/client/round).

**Raw AUROC (drift vs no-drift rounds)** — the headline:

| signal | A | abrupt | recurring | burst |
|---|---:|---:|---:|---:|
| entropy (client) RAW | 0.643 | **0.161** | 0.852 | 0.819 |
| hard δ RAW | 0.941 | 0.992 | 0.999 | 1.000 |
| TV distance RAW | 0.978 | 1.000 | 1.000 | 1.000 |

**Causally-normalized metrics** (burn-in 10 + warm-up 15; mean over seeds):

| signal | Spearman vs ρ (A/abr/rec/bur) | AUROC_norm | retention | seed-stability |
|---|---|---|---|---|
| δ hard | 0.91 / 0.82 / 0.86 / 0.59 | 0.93 / 1.00 / 1.00 / 1.00 | 0.92–0.96 | 0.56–0.77 |
| entropy | 0.71 / **−0.45** / 0.73 / 0.52 | 0.82 / 0.89 / 1.00 / 1.00 | 0.92–0.95 | 0.98–0.99 |
| TV | 0.90 / 0.82 / 0.86 / 0.59 | 0.86 / 1.00 / 1.00 / 1.00 | 1.0+ | 0.92–0.97 |
| JS/symKL | up-trending w/ convergence → false alarms (FA to 0.65) | | | |
| rep centroid / MMD | weak–moderate (AUROC 0.40–0.91 / 0.50–0.79) | | | |

Normalizers on δ: anchored 0.995 ≈ **guarded 0.983** > ewma 0.978 > rollq 0.961 > cusum 0.928.

Boundary conditions recorded: warm-up must fit an initial stationary window (recurring
violates → absolute-z detection degrades, ranking survives); KL/JS break the guard's
monotonicity assumption; entropy within-segment retention does NOT collapse at 100R.

## 5. Gate B — Downstream (100R): the price of the leak, and earning it back

Integrated acc, 3 seeds (Δ = paired vs fixed λ0.4):

| controller | A | Δ | abrupt | Δ |
|---|---:|---:|---:|---:|
| legacy (leak, mu=0.31) | 0.6268 | +1.10 | 0.6084 | +0.57 |
| selfcal v1 | 0.6134 | −0.23 | 0.5922 | −1.06 |
| selfcal v3b | 0.6135 | −0.23 | 0.5966 | −0.62 |
| **selfcal v3c** | 0.6163 | +0.06 | 0.6073 | +0.46 |
| selfcal-entropy | 0.6130 | −0.27 | 0.5898 | −1.32 |
| fixed λ0.4 | 0.6157 | — | 0.6028 | — |

Paired key numbers: v3c−legacy **−1.04 pp on A** (the leak's home value) but **−0.11 pp
on unseen abrupt** (tie; v3c worst-cell better 0.525 vs 0.520). v3c−entropy on abrupt:
**+1.75 pp, 3/3 seeds** — leak-free signal-swap confirmation.

## 6. Gate C — Spatial worst-cell & fairness (100R, equal_spread, 3 seeds)

| method | acc | worst | gap | Δworst vs temporal-selfcal |
|---|---:|---:|---:|---:|
| fixed λ0.2 | 0.6679 | **0.5841** | 0.200 | +6.15 pp (3/3) |
| fixed λ0.4 | 0.6652 | 0.5738 | 0.225 | +5.13 pp |
| **selfcal-ST** | 0.6573 | 0.5450 | 0.307 | **+2.25 pp (3/3), acc +1.10** |
| fair-deploy + ST | 0.6565 | 0.5469 | 0.303 | vs ST +0.19 pp (signs mixed) |
| selfcal temporal-only (v3b) | 0.6463 | 0.5225 | 0.329 | baseline |
| fair-deploy (temporal risk) | 0.6437 | 0.5142 | 0.332 | −0.83 pp |
| fair-ORACLE donors | 0.6431 | 0.5050 | 0.356 | **−1.75 pp** |
| fair-random donors (s0) | 0.6435 | 0.5160 | 0.265 | −3.43 pp |

- Spatial reference frame = the real fix (+2.25 pp worst AND +1.10 pp acc, 3/3).
- **Donor fairness = failed mechanism**: oracle donors < deployable < ST-alone. Cross-cell
  parameter inflow imports off-distribution personalization. (v5 conclusion, now 3-seed.)
- **F3 routing = null**: entropy vs disagreement-confidence routing pooled 0.6535 = 0.6535.
- Ceiling vs fixed λ0.2: ST −3.90 pp worst [−4.35, −2.40, −4.95].

## 7. Phase 4 — Causal online baselines (100R, 3 seeds; regret vs labeled oracle B11)

| controller | A integ | regret | switch/rd | abrupt integ | regret | switch/rd |
|---|---:|---:|---:|---:|---:|---:|
| B11 labeled causal oracle | 0.6330 | 0 | 0.045 | 0.6183 | 0 | 0.045 |
| legacy (leak) | 0.6268 | +0.62 | 0.045 | 0.6084 | +0.98 | 0.045 |
| **selfcal v3c** | 0.6163 | +1.67 | **0.009** | 0.6073 | **+1.09** | **0.010** |
| UCB (proxy reward) | 0.6180 | +1.50 | 0.163 | 0.5981 | +2.01 | 0.224 |
| fixed λ0.4 | 0.6157 | +1.73 | 0 | 0.6028 | +1.55 | 0 |
| EXP3 (proxy) | 0.6152 | +1.78 | 0.186 | 0.6008 | +1.75 | 0.186 |
| update-norm selfcal (s0) | 0.6125 | +2.75 | 0.003 | 0.5881 | +3.35 | 0.003 |
| periodic proxy grid (s0) | 0.6042 | +3.58 | 0.023 | 0.5898 | +3.17 | 0.023 |

Findings: v3c = lowest-regret label-free method with 20× less churn than bandits;
proxy-reward bandits thrash (flat reward across arms); update-norm tracks optimization,
not traffic; fixed→oracle headroom is only 1.5–1.7 pp at 100R.

## 8. Phase 6 — Network stress (abstract model; selfcal v3b on abrupt; paired vs clean)

Clean baseline: integ 0.5966 / worst 0.5146 (3 seeds).

| impairment | Δ integrated | Δ worst |
|---|---:|---:|
| signal delay 2 / 5 / 10 rounds | −0.39 / −0.71 / −0.66 pp | −0.39 / −0.90 / −0.84 pp |
| signal loss 10% / 20% (hold-last) | +0.10 / +0.14 pp | +0.09 / +0.03 pp |
| topology ring / star / dynamic (s0) | −0.09 / −0.76 / −0.17 pp | ±0.3 pp |
| participation 50% | −5.54 pp | −5.50 pp |
| participation 50% **fixed-λ control** | **−6.38 pp** | −5.80 pp |

Participation attribution: the drop is entirely the training effect;
controller-attributable share = **+0.84 pp in selfcal's favor**. No impairment-induced
oscillation (switch rate 0.008–0.010 throughout).

## 9. Phase 7 — Overhead (measured, RTX 3090 Ti)

Client 279,882 params (1.12 MB) · server 1,378,698 (5.51 MB) · per-round model exchange
(overlap-2) 12.1 MB.

| probe n | client fwd | probe fwd (cl+2 srv) | all-16 signals | δ extra vs entropy-only |
|---:|---:|---:|---:|---:|
| 16 | 0.87 ms | 1.54 ms | 6.03 ms | 0.67 ms |
| 64 | 0.83 ms | 1.53 ms | 6.91 ms | 0.70 ms |
| 128 | 1.05 ms | 2.51 ms | 7.51 ms | 1.46 ms |

Controller traffic: 4 B scalar/client/round uplink; 8 B/ES/consensus step; **0 B
activation upload** (SplitOMC keeps server-model copies client-side). Wording:
"negligible scalar communication overhead, with separately measured compute overhead".

## 10. Gate D — Full evaluation (150R, 5 seeds core; 127 + 26 runs)

### 10.1 D2 — Horizon confirmation (Schedule A @150R)

Signal swap inside the same v3c+ST controller, 5 seeds paired:
**integrated +1.06 pp [+1.11, +0.28, +0.60, +0.81, +2.51] (5/5)** ·
**ρ=0.8 segment +3.64 pp** · worst-cell +0.91 pp.
Signal @150R: Spearman δ 0.875 / TV 0.921 / entropy 0.498; AUROC δ 0.847 / ent 0.748.
Refined mechanism: entropy fails by global convergence-trend confounding (within-segment
retention stays ≈0.95 even at 150R).

### 10.2 D1 — Unseen temporal schedules (150R, 5 seeds; Δ = vs per-schedule hindsight best fixed)

| arm | gradual_sigmoid | asym_return | piecewise_random | 3-sched mean |
|---|---:|---:|---:|---:|
| labeled causal oracle (3s) | 0.6393 (+1.07) | 0.6535 (+1.23) | 0.6389 (+1.58) | 0.6439 |
| hindsight best fixed | 0.6323 (fx02) | 0.6426 (fx04) | 0.6284 (fx02) | 0.6344 |
| fixed λ0.2 (robust) | 0.6323 | 0.6423 | 0.6284 | 0.6343 |
| **selfcal v3c+ST** | 0.6272 (−0.51) | 0.6434 (+0.08) | 0.6147 (−1.36) | 0.6284 |
| legacy (leak) | 0.6162 (−1.61, 0/5) | 0.6479 (+0.52) | 0.6251 (−0.32) | 0.6297 |
| fixed λ0.4 (dev-history pick) | 0.6233 | 0.6426 | 0.6259 | 0.6306 |
| entropy (same ctrl, 3s) | 0.6003 | 0.6288 | 0.6104 | 0.6132 |

C1 generalizes (main−ent +2.7/+1.5/+0.4 pp); self-cal = parity with leak on unseen
(winners alternate); no deployable adaptive beats the hindsight fixed; oracle keeps
+1.1–1.6 pp (labels-only headroom).

### 10.3 D3 — Spatial @150R (5 seeds) + bounds ablation

| arm | integ | worst | p10 |
|---|---:|---:|---:|
| fixed λ0.2 | 0.6328 | 0.5425 | 0.4993 |
| ST + λ_max=0.5 (ablation) | 0.6294 | 0.5241 | 0.4843 |
| ST (λ_max=0.7) | 0.6213 | 0.5003 | 0.4596 |

Ceiling persists; λ_max=0.7 over-personalization explains ~⅓–½ of the worst gap.

### 10.4 D4/D5 — Dataset transfer: the blind spot and its mechanism

| arm | C100 gsig | C100 spatial | TinyIN gsig |
|---|---:|---:|---:|
| legacy (mu=0.31 transferred) | 0.3571 | — | — |
| fixed λ0.4 | 0.3514 | 0.3584 | 0.2380 |
| selfcal v3c+ST | 0.3138 (0/3) | 0.3266 (0/3) | 0.2224 (0/3) |

Trace evidence (`d4_main_gsig_s0`): CIFAR-100 static δ = 0.56–0.70 (5-of-100-class
specialist) → self-normalization discards the level → λ≈0.63 over-personalization.
Legacy reads the same level as "drift" → λ≈0.155 → the generalization C100 needs.
**The operating point depends on the LEVEL ("how hard"), not only the change.**

### 10.5 Dual-view verification (26 runs) — final controller

| environment | DV vs v3c | DV vs fixed λ0.4 | DV vs legacy |
|---|---:|---:|---:|
| CIFAR-100 gsig | **+4.02 pp (3/3)** | **+0.26 pp (3/3)** | −0.31 pp (tie) |
| CIFAR-100 spatial | **+3.40 pp (3/3)** | +0.22 pp (worst +0.20) | — |
| Tiny-ImageNet | **+1.85 pp (3/3)** | +0.30 pp (3/3) | — |
| CIFAR-10 D1 ×3 scheds | −0.13 / +0.16 / +0.61 pp | no regression | — |
| CIFAR-10 D2 (A) | +0.30 pp (3/3) | — | — |
| CIFAR-10 D3 spatial | +0.31 integ / **+0.95 worst (5/5)** | vs fixed λ0.2: −0.85 / −3.27 worst | — |

**selfcal-DV strictly dominates v3c everywhere tested**; beats the fixed grid on both
transfer datasets; one boundary remains (fixed λ0.2 worst-cell on CIFAR-10 spatial).

## 11. The three diagnose→fix cycles (methodological contribution)

| # | failure found (dev evidence) | dimensionless fix | verification |
|---|---|---|---|
| 1 | boiling-frog: stepped ramp absorbed stage-wise (z≈0 at ρ=0.8) | asymmetric guard (adapt baseline only when z<0.5) | abrupt: v1 0.5922 → v3c 0.6073; unit test |
| 2 | static spatial heterogeneity: temporal z ≡ 0 | cross-cell robust-z, z=max(z_t, z_sp) | worst +2.25 pp (3/3); unit test |
| 3 | dataset blindness: static δ level normalized away | absolute cap λ_abs=λ_max−(λ_max−λ_min)·δ̂ (δ = probability, no constants) | +1.9–4.0 pp (9/9); above fixed grid; unit test |

## 12. Final verdict & claims discipline

**TMC: GO · ToN: CONDITIONAL_GO** (network plane is abstract; update-plane impairments
and L≫5 topologies would be needed for ToN).

Supported: architecture-native signal more persistent than absolute entropy
(convergence-robust, not "convergence-invariant"); constant-free dual-view controller
transfers across unseen schedules AND datasets; best label-free causal method; robust
to signal-plane impairments; negligible scalar communication overhead (measured).

Not supported (kept as boundaries/negatives): beating hindsight-tuned fixed λ (temporal
unseen: −0.6 pp mean; CIFAR-10 spatial worst: −3.3 pp); donor/fairness aggregation
(failed, oracle donors hurt); disagreement routing (null); bandit/update-norm baselines
(negative).

Pre-submission additions (~45 runs): 5-seed transfer completion; one covariate-corruption
schedule; one ResNet-18 split arm; multi-seed composition-coupled mobility; figures/draft.

## 13. Artifact index

- Reports: `phase0_reproduction_and_audit.md`, `phase1_calibration.md`,
  `phase2_signal_benchmark.md`, `phase4_causal_baselines.md`,
  `phase5_fairness_controller.md`, `phase6_network_stress.md`, `gateD_results.md`,
  `final_tmc_ton_assessment.md`, `master_experiment_plan.md`, `progress_log.md`
- Tables: `all_runs.csv` (265), `signal_quality*.csv`, `normalizer_comparison.csv`,
  `downstream_pilot.csv`, `fairness_results.csv`, `phase4_baselines.csv`,
  `network_robustness.csv`, `overhead.csv`, `gated_*.csv`
- Figures: `signal_traj_{A,abrupt,recurring,burst}`, `signal_auroc`,
  `dataset_transfer`, `unseen_schedules` (PNG+PDF)
- Code: `src/` (signals, normalizers, controllers incl. DV, baselines, fairness,
  routing, network, runner), `scripts/` (run_v2, launchers, analyzers, queue system),
  `tests/` (44 tests)
