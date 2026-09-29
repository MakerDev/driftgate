# Master Experiment Plan — Adaptive-SplitOMC → TMC/ToN Expansion

**Created**: 2026-07-11 · **Branch**: `adaptive_splitomc_tmc` (copy of v4 final state + v5 lessons)
**Base**: ICTC results in `docs/results_v3/phase_reports/Integrated_R150.md` (canonical)

---

## 1. Research questions (final; all reports answer these)

| RQ | Question | Phase |
|---|---|---|
| RQ1 | How much does absolute entropy confuse deployment drift with training convergence? | P2 |
| RQ2 | Under which conditions is dual-exit disagreement a more persistent/useful drift signal? | P2 |
| RQ3 | Does a self-calibrating disagreement controller work on unseen datasets & unseen schedules? | P1/P3 |
| RQ4 | What do adaptive controllers actually buy vs causal online baselines? | P4 |
| RQ5 | Can fairness-aware cross-cluster transfer break the worst-cluster ceiling that λ/Λ adaptation could not? | P5 |
| RQ6 | Is the controller stable under delay, loss, mobility, topology, and communication budgets? | P6 |
| RQ7 | Do the accuracy/fairness gains justify compute/latency/energy/communication costs? | P7 |

## 2. Core claim under test (and what we will NOT claim)

**Claim**: In split learning with role-separated dual exits, client–server disagreement is an
*architecture-native* drift signal that is *more persistent than absolute predictive entropy*
under continued training (convergence-robust, not convergence-invariant).

Banned phrasings (no evidence level justifies them): "disagreement cannot decay",
"convergence-invariant", "zero overhead", "first drift-aware split learning",
"cell-free 6G validated", "real-time", "theoretically optimal".

## 3. Known problems this plan must fix (from the ICTC record)

1. **Calibration leakage**: `mu_drift` history 0.392 → 0.35 → 0.31; every step used
   observed δ under KNOWN ρ segments of the test schedule (Schedule A), the final step
   explicitly citing worst-cell separation. Documented in `reports/phase1_calibration.md`.
2. **Hindsight-only fixed baselines**: "best fixed λ" was selected after seeing full test traces.
3. **Single seed** (partition_seed=0/model_seed=100) for every headline number.
4. **CIFAR-10-only**, artificial Main/OOP/OOR mixture, one schedule family.
5. **Worst-cluster ceiling**: adaptive λ (v4) and fairness weighting η (v5 gate) both failed
   to lift worst-cell above hand-tuned fixed λ.

## 4. Experiment matrix

### Phase 0 — Reproduction & audit (Gate A)
- Re-run E2/Schedule-A adaptive (disagreement, mu=0.31) with the untouched v4 stack;
  compare to `results/e2_temporal/schedule_A/adaptive_splitomc.json` (single-seed determinism check).
- 17 audit unit tests (`journal_expansion/tests/test_journal.py`) + 12 legacy tests.
- Output: `reports/phase0_reproduction_and_audit.md`.

### Phase 1 — Calibration-leakage fix (self-calibrating controller)
- Controllers: (a) hand-calibrated legacy (mu=0.31 — kept as-is, labeled leaky),
  (b) source-calibrated (fit once on dev env, transfer), (c) **self-calibrating online**
  (GuardedRobustNormalizer z-space; all constants dimensionless & pre-registered:
  W=15, β=0.05, z_guard=1.5, rel-floor=0.10, EWMA α=0.3, z0=1.5, τ_z=0.75, clip [-2,6]).
- Dev/tuning environment: CIFAR-10 schedule A seeds {0,1,2}. Final-test schedules
  (never used for any constant selection): gradual_sigmoid, asym_return, piecewise_random + CIFAR-100/Tiny-IN.

### Phase 2 — Signal benchmark (Gate B)
- **Passive recording runs** (fixed λ=0.4 backbone, controller inactive): all 16 raw signals
  recorded per round per client (probe n=64, causal, label-free).
- Pilot matrix: schedules {A, abrupt, recurring, burst} × seeds {0,1,2} × CIFAR-10 @100R = 12 runs.
- Metrics per signal: Spearman/Pearson vs ρ, AUROC/AUPRC (raw + causally normalized),
  detection delay, false alarms, sustained retention, recovery delay, noise, seed stability.
- Downstream: selfcal(δ) vs selfcal(entropy) vs legacy(mu=0.31) vs fixed λ on schedule A.
- Gate B pass: normalized disagreement ≥ entropy on ≥3/4 schedules (AUROC + retention), and
  selfcal(δ) within 1 pp of legacy hand-calibrated on schedule A (it must not need the leak).

### Phase 3 — Generalization
- Datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet (+FEMNIST if feasible). Per-dataset ND partition generators.
- Partition severity: 1/2/5 classes-per-client, Dirichlet α ∈ {0.1,0.3,1.0}, overlap {0,25,50,75}%.
- Models: current CNN, ResNet-18-lite, MobileNetV2-lite; split points early/middle/late.
- Main results: ≥5 seeds, paired stats (Wilcoxon + t, effect size, 95% CI). Pilots: 3 seeds.

### Phase 4 — Causal baselines
- B0 naive fixed {0.2,0.4,0.6}; B1 hindsight-oracle fixed (upper ref only, marked);
  B2 historical fixed (dev trace → unseen trace); B3 robust fixed; B4 per-cluster historical;
  B5 periodic proxy grid; B6 bandit (UCB / Thompson; proxy + delayed-label rewards);
  B7 online mirror-descent λ; B8 normalized-entropy controller; B9 representation-drift controller;
  B10 update-norm controller; B11 causal dynamic oracle (labeled val per round; upper bound).
- Metrics: integrated acc, Main/OOP/OOR, worst-cluster, p10 client, dynamic regret vs B11,
  gap to B1, switching frequency, adaptation/recovery delay.

### Phase 5 — Worst-cluster ceiling (new controller; Gate C)
- F1 disagreement-aware donor selection (representation-distance + competence + complementarity);
  F2 fairness-aware aggregation (v5 η mechanism re-tested WITH donor selection, not instead of);
  F3 adaptive exit routing (train-time vs inference-time separated); F4 joint (λ, Λ, a, q).
- 14 pre-registered ablations (§7.3 of the task spec). Gate C on CIFAR-10 + CIFAR-100, 3 seeds.
- Success gate (pre-registered, not tuned to): worst-cluster +2 pp vs disagreement controller,
  avg-acc loss ≤ 0.5 pp, majority-direction across datasets/schedules.
- v5 negative result (η-weighting equalizes DOWN) is a boundary condition to report, not hide.

### Phase 6 — Network stress (ToN)
- Topologies: line/ring/grid/random-geometric/star/dynamic (record degree, spectral gap).
- Impairments: signal+model delay {0,1,2,5,10} rounds, loss {0,1,5,10,20}%, stale δ,
  partial participation, asymmetric links, disconnection, heterogeneous fronthaul.
- Mobility must change BOTH connectivity and Main/OOP/OOR composition.
- Abstract reproducible model (documented as abstraction); ns-3/Sionna optional track only if present.

### Phase 7 — Overhead
- Bytes (activation/scalar/model), latency (client/server/95p/99p), probe strategies
  (every 1/5/10, adaptive, drift-triggered), consensus steps, energy proxy; accuracy–overhead Pareto.

### Phase 8 — Theory-adjacent empirics
- Monotonicity of E[δ] vs ρ (+server-competence dependence), false-agreement rate,
  controller stability (noise/delay/slope sensitivity), empirical dynamic regret decomposition.

## 5. Gates

- **Gate A (correctness)**: repro matches, 29/29 tests, leakage tests pass → else stop.
- **Gate B (signal viability)**: as Phase 2 above.
- **Gate C (fairness mechanism)**: as Phase 5 above.
- **Gate D (full evaluation)**: winning method + key baselines only → 5 seeds × all datasets/schedules.

## 6. Compute & seeds

- 2× RTX 3090 Ti (shared machine; ~1.6-1.8 min/round for 50-client ND1 CIFAR-10 runs).
- Pilots 100R/3 seeds; headline runs 150R/5 seeds. Seeds vary partition, model init,
  schedule randomness, mobility trajectory, batch order: (p,m) ∈ {(0,100),(1,101),(2,102),(3,103),(4,104)}.
- All runs get provenance records (`provenance/all_runs.jsonl`); re-runs get new run IDs.

## 7. Artifacts

Machine-readable: `tables/{all_runs,main_results,signal_quality,fairness_results,network_robustness,overhead}.csv`.
Reports: `reports/phase{0..7}_*.md`, `reports/final_tmc_ton_assessment.md`.
Figures: 12 required (see §12.5 of the task spec), PDF+PNG, no small fonts, marker+linestyle coded.
