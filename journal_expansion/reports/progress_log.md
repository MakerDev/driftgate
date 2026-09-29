# Progress Log — TMC/ToN Expansion

Cumulative, newest at the bottom. Every entry: what was implemented / what ran /
results / surprises / go-no-go / claim impact.

---

## 2026-07-11 (session 1)

### Implemented
- `journal_expansion/` scaffolding + provenance system (`src/provenance.py`): run IDs,
  git commit, config SHA-256, split hash, model-init hash, env/GPU, status; append-only
  `provenance/all_runs.jsonl`; re-runs never overwrite.
- Drift-schedule registry (`src/schedules.py`): A (fraction-based), A150_legacy, B_legacy,
  abrupt, recurring, burst, gradual_linear, gradual_sigmoid, asym_return, piecewise_random,
  staggered (per-cell). Terminology guard: these are traffic-composition drift, not concept drift.
- Signal library (`src/signals/library.py`): 16 raw signals per probe — S0 entropy (client/
  server/avg/max), S1 hard δ, S2 soft (symKL, JS, TV, logit-cosine, margin-diff), S3
  representation (centroid/cov/cosine/MMD vs warm-up reference), confidences for the S5 gate.
- S4 causal normalizers (`src/controllers/normalizers.py`): guarded-robust-z (main),
  anchored, EWMA, rolling-quantile, CUSUM. Strictly causal, online≡offline tested.
- Self-calibrating controller (`src/controllers/self_calibrating.py`) — dimensionless
  pre-registered constants; + SourceCalibratedController (legacy mapping, transferred constants).
- Extended runner (`src/runner.py`): passive full-signal recording, pluggable controllers,
  per-client eval traces, per-cell schedules, provenance integration. CLI `scripts/run_v2.py`.
- Audit test suite: 17 new tests; 12 legacy tests still pass (29/29 total).

### Ran / running
- Legacy pytest 12/12; new pytest 17/17 (two pre-registration fixes made to satisfy
  pre-written tests BEFORE any benchmark: warm-up boundary off-by-one; σ noise-floor
  σ̂≥0.10·|μ̂| — without it warm-up MAD underestimation amplified probe noise into λ swings).
- 2-round GPU smokes: passive + selfcal paths OK (~1.8 min/round, 50 clients).
- **Launched (detached, PPID=1)**:
  - Reproduction: E2 Schedule-A adaptive disagreement (v4 code, mu=0.31, 150R, cuda:0)
    → `runs/reproduction/e2_temporal/schedule_A/adaptive_splitomc.json` (~3.5-4.5h).
  - Gate-B pilot matrix: 12 passive recording runs (schedules A/abrupt/recurring/burst ×
    seeds 0/1/2, 100R, probe 64), 3 slots (cuda:0 ×1 shared, cuda:1 ×2) (~10-12h).

### Findings so far
- **Calibration leakage CONFIRMED and documented** (`reports/phase1_calibration.md`):
  mu_drift 0.392→0.35→0.31, every step fitted on ρ-labeled test-schedule observations,
  final step citing worst-cell separation. tau likewise. → journal main method must be the
  self-calibrating controller; hand-calibrated becomes a labeled reference.
- Audit: no correctness-invalidating bug. One dormant indexing bug (mobility rewire with
  empty clients, O3), probe/eval pool overlap documented as caveat B.
- ICTC repro target numbers frozen from `Integrated_R150.md` (E2 0.6671 / E5 swap +2.43 pp /
  E3 worst-cell −4.93 pp vs fixed λ=0.2).

### Claim impact
- "adaptive beats/approaches best fixed" claims must be re-established WITHOUT the leak
  (Gate B downstream) — until then treat E2 +1.33 pp as unverified at journal standard.
- Signal-level claim (δ more persistent than entropy) unaffected in design; will be
  re-measured with proper multi-schedule multi-seed benchmark (Gate B).

### Also implemented this session (while GPU runs proceed)
- **Phase-4 causal baselines** (`src/baselines/online.py` + runner modes): B5 periodic
  proxy grid, B6 UCB bandit, B6b/B7 EXP3 (with advantage baseline — plain EXP3 locked onto
  suboptimal arms under near-1 rewards), B10 update-norm signal hook, B11 greedy labeled
  causal oracle (per-round eval). Pre-registered label-free proxy reward = routed confidence.
- **Phase-5 mechanisms**: F1/F2 donor-selected risk-directed inter-cell aggregation
  (`src/controllers/fairness.py`) — per-RECEIVER donor pools (complementarity × competence ×
  rep-proximity) and absorption only for risk-elevated cells; designed against the v5 failure
  mode (uniform pool equalized downward). Deployable/oracle/random donor variants.
  F3 disagreement-confidence exit routing (`src/evaluation/routing.py`) — evaluated
  alongside the legacy entropy routing in EVERY eval round of every new run (near-zero cost);
  unit test shows the pure mechanism: overconfident client + OOP probe → entropy routing 0%,
  disagreement routing 100%.
- **Phase-3 generalization infra**: datasets cifar100/tinyimagenet/gtsrb/stl10
  (`src/datasets_ext.py`; per-dataset main-classes-per-client, Dirichlet α partitions with
  90%-cover main-class definition), model families resnet18/mobilenetv2 with
  early/middle/late split points + split_stats (`src/models_ext.py`). Tiny-ImageNet
  downloading in background.
- **Phase-6 network abstractions**: topologies line/ring/grid/star/rgg/dynamic with
  spectral-gap stats; SignalChannel (delay/loss/stale-hold with age tracking);
  partial participation. Integrated into the runner (`network_cfg`).
- Test suite now **41 journal tests + 12 legacy = 53, all passing**.
- GPU smokes: fairness+selfcal+routing path and UCB path verified end-to-end (2 rounds).

### Next
- (monitor) repro completion → `scripts/compare_repro.py` → fill phase0 report → Gate A verdict.
- (monitor) 12 passive runs → `scripts/analyze_signals.py` → Gate-B signal tables/figures →
  launch downstream pilot (`scripts/launch_downstream.py`) → `analyze_downstream.py` →
  `reports/phase2_signal_benchmark.md` Gate-B verdict.
- Then: Gate-C fairness pilots (F1/F2/F3 × CIFAR-10/100), Phase-4 baseline pilots,
  Phase-6 delay/loss sweeps.

---

## 2026-07-11 (session 1, later) — Gate A PASS · Gate B signal half PASS

### Gate A (Phase 0) — COMPLETE, PASS
- Reproduction of E2/Schedule-A adaptive (v4 code, same seeds): integrated **0.6669 vs
  0.6671 (−0.02 pp)**; all ρ segments within ±0.06 pp. Full table in phase0 report.

### Gate B signal half — COMPLETE, PASS (report: phase2_signal_benchmark.md)
- 12/12 passive runs done (all exit=0). 192 signal-metric rows.
- **RQ1 evidence**: raw entropy AUROC on `abrupt` = **0.161** (anti-predictive; convergence
  trend swamps drift); raw δ = 0.992, raw TV = 1.000. Normalization rescues entropy for
  onset detection (0.89–1.00) but never beats δ.
- Normalizers on δ: anchored 0.995 ≈ guarded 0.983 > ewma > rollq > cusum (FA≈0 all).
- **Two pre-registration defects found & fixed** → controller v3b: (1) warm-up window
  contaminated by untrained-model transient (δ 1.0→0.45 in R1–10) compressed drift jumps
  to z≈1.5 → burn-in 10 added; (2) rep_* zero-prefix analysis artifact → offset 15.
  v1 AND v3b both reported per protocol.
- Boundary conditions recorded: warm-up must fit inside an initial stationary window
  (recurring violates it → absolute-z detection degrades, ranking survives); KL/JS have an
  UPWARD convergence trend that breaks the guard's assumption; entropy within-segment
  retention did NOT collapse at 100R (ICTC decay was 150R) — long-horizon claim deferred
  to Gate D.
- matplotlib/numpy-2.5 ABI breakage fixed by upgrading matplotlib (3.11.0).

### Running
- 18 downstream v1 runs (sc_delta / sc_ent / legacy_delta × A/abrupt × 3 seeds, 100R,
  routing-eval on) across 4 slots; monitor set. v3b launcher ready (`launch_v3b.py`).

---

## 2026-07-11 (session 1, evening) — downstream v1 interim: selfcal under-reacts; v3c + ST designed

### Findings
- Downstream v1 (A complete): legacy leaky controller +1.10 pp over fixed λ0.4;
  selfcal v1/v3b −0.23 pp — **the leak bought ≈1.3 pp** on its home schedule.
  Gate-B downstream criterion (within 1 pp of legacy on A) FAILS for v1/v3b.
  Selfcal already leads legacy on worst-cell under abrupt (0.537 vs 0.511).
- λ/z-trace diagnosis: **boiling-frog** — guarded baseline (freeze at z>1.5) absorbs a
  stepped ρ ramp stage by stage; z≈0 through the ρ=0.8 segment, λ never generalizes.
- **Structural insight**: temporal self-calibration is inherently blind to STATIC spatial
  heterogeneity (nothing ever changes). Legacy's absolute threshold measures "is my
  traffic hard", selfcal measures "did it change" — E3-type stratification needs a
  spatial reference frame, not a temporal one.

### Pre-registered revisions (before their runs; dev-schedule A analysis only)
- **v3c**: asymmetric guard z_guard=0.5 — baseline adapts only in stationary/declining
  regimes (convergence=down, drift=up prior). Boiling-frog unit test added (43/43).
- **selfcal-ST**: cross-cell spatial robust-z, effective z = max(z_temporal, z_spatial);
  causal, label-free, dimensionless. Static-pattern stratification unit test added.
  Fairness risk input upgraded to max(z_t, z_sp) as well.

### Queue
- 6 v3c downstream jobs prepended; 6 ST Gate-C jobs (±fairness × 3 seeds) prepended;
  queue at 69 jobs, 4 workers (2/GPU). Monitor set for 30 downstream JSONs.

---

## 2026-07-12 (early AM) — Gate B COMPLETE: PASS, v3c adopted

- 30/30 downstream runs done, 0 failures. Final paired results in phase2 report §7.3:
  - v3c ties leak-calibrated legacy on the UNSEEN schedule (−0.11 pp, better worst-cell);
    −1.04 pp residual on legacy's home schedule = measured value of the calibration leak.
  - δ beats entropy **3/3 seeds (+1.75 pp)** inside the same self-calibrated controller on
    the unseen schedule — leak-free confirmation of the ICTC signal-swap result.
  - v1 → v3b → v3c progression on abrupt: 0.5922 → 0.5966 → 0.6073 (each fix earned its keep).
  - Honesty note: at 100R, adaptive ≈ fixed λ0.4 on integrated acc (+0.06/+0.46 pp);
    the adaptive case rests on no-hindsight-tuning + tail metrics + longer horizons (Gate D).
- **Gate B: PASS** (signal half 4/4 AUROC; downstream v3c adopted as main candidate).
- Gate-C early partial (seeds 0-1): selfcal-ST worst +3.2 pp AND acc +1.6 pp over
  temporal-only — possible resolution of the v4/v5 worst-cluster ceiling via the spatial
  reference frame. fair_deploy_sp_s0 was old-code (temporal-risk) → renamed to
  fair_deploy_tRisk_sp_s0 (kept as ablation), new-code re-run queued.
- Monitors: Gate C (23 JSONs) armed. Queue: Phase 4 (22) + Phase 6 (21) follow.

---

## 2026-07-13 — PILOT STAGE COMPLETE (Gate C + Phase 4 + Phase 6), 0 failures

Worker scale-up (8 workers after user freed resources) finished the entire pilot queue.
109 runs in `tables/all_runs.csv`. Reports written: phase4_causal_baselines.md,
phase5_fairness_controller.md, phase6_network_stress.md.

### Gate C (spatial worst-cell) — VERDICT: selfcal-ST selected; fairness arms dropped
- **selfcal-ST: worst +2.25 pp AND acc +1.10 pp over temporal-only, 3/3 seeds** — passes
  the pre-registered bar with an avg GAIN. The spatial reference frame was the missing
  piece; explains why legacy's absolute threshold could stratify E3.
- **F1/F2 donor fairness: NEGATIVE at 3-seed rigor** — deploy+ST adds +0.19 pp (mixed
  signs); ORACLE donors −1.75 pp (worse than deployable!); random worst. Cross-cell
  parameter inflow imports off-distribution personalization: not a knowledge-routing
  problem. v5 conclusion reproduced and extended.
- **Fixed-λ ceiling stands**: fixed λ0.2 worst 0.5841 vs best adaptive 0.5450 (−3.9 pp,
  3/3). Boundary condition, documented not hidden. Bounds ablation queued for Gate D.
- F3 routing: exact tie (0.6535 = 0.6535) on converged models — honest null.

### Phase 4 (RQ4) — v3c is the best label-free causal controller
- Regret vs labeled causal oracle (abrupt): v3c +1.09 pp, UCB +2.01, EXP3 +1.75,
  fixed +1.55, update-norm +3.35, proxy-grid +3.17. v3c churn 0.01 vs bandits 0.16-0.22.
- Fixed→oracle headroom only 1.5-1.7 pp at 100R — adaptivity upside intrinsically modest
  at this horizon (Gate-D question for 150R + harsher drift).
- Negatives: proxy-reward bandits thrash; update-norm tracks optimization not traffic.

### Phase 6 (RQ6) — controller input is delay/loss robust
- Delay 2/5/10 rounds: −0.4/−0.7/−0.7 pp. Loss 10/20%: ≈0 (hold-last absorbs).
  Topology shape: second-order at 5 cells. No impairment-induced oscillation.
- Participation 50%: −5.5 pp but confounded with the training effect — fixed-λ controls
  queued (3 runs) for attribution.

### Next
- Participation controls (~2 h) → finalize phase6 report.
- Then Gate-D matrix design: selfcal-ST + v3c vs {fixed grid incl. hindsight ref, legacy,
  UCB ref, B11} × 150R × 5 seeds × {CIFAR-10, CIFAR-100, Tiny-ImageNet} × unseen
  schedules (gradual_sigmoid, asym_return, piecewise_random) + λ-bounds ablation +
  entropy-death confirmation at 150R.

---

## 2026-07-13 (later) — participation attributed; GATE D LAUNCHED (121 jobs)

- Participation 50% attribution: fixed λ0.4 drops −6.38 pp under the same sampling vs
  selfcal −5.54 pp → drop is entirely the training effect; **controller-attributable
  +0.84 pp in selfcal's favor**. phase6 report finalized.
- run_v2 gained --lam_min/--lam_max (D3 bounds ablation). CIFAR-100 main-method smoke OK;
  Tiny-ImageNet smoke running (D5 enqueued only if it passes).
- **Gate D enqueued (priority order), all pre-registered in `enqueue_gated.py`:**
  D2 entropy-death @A/150R (main/entswap ×5 seeds + passive ×3) = 13;
  D1 unseen schedules ×{main,legacy,fixed02,fixed04 ×5; entswap,oracle ×3} = 78;
  D3 spatial 150R {ST, ST+lam_max0.5, fixed02} ×5 = 15;
  D4 CIFAR-100 {gradual_sigmoid: main/legacy/fixed04; spatial: main/fixed04} ×3 = 15.
  Total 121 jobs ≈ 2.5-3 days at 8 workers. D5 Tiny-IN (6 @100R) pending smoke.
- Tiny-IN smoke passed (~8 min/round) → D5 enqueued (127 total Gate-D jobs).

### Gate-D D2 COMPLETE — C1 confirmed at 150R, 5/5 seeds (phase2 report §8)
- Same v3c+ST controller, signal swap only: δ − entropy = **+1.06 pp integrated
  (5/5 seeds)**, **+3.64 pp in the sustained ρ=0.8 segment**, +0.91 pp worst-cell.
- Signal @150R: Spearman vs ρ — δ 0.875 / TV 0.921 / entropy 0.498.
- Mechanism refined: entropy fails via GLOBAL convergence-trend confounding (correlation
  halved), not within-segment decay (retention ≈0.95 even at 150R). More precise and more
  defensible than the ICTC framing.
- D1 monitor armed (78 runs, ~1.5 days).

---

## 2026-07-15 — Gate-D D1 COMPLETE (unseen schedules; RQ3/RQ4 core numbers)

Full table in `reports/gateD_results.md`. Headlines:
- **C1 generalizes**: main − entswap = +2.7/+1.5/+0.4 pp across the three unseen
  schedules (mean +1.5 pp) — the signal advantage is the robust effect.
- **Self-calibration = parity with leak-calibration on unseen drift** (0.6284 vs 0.6297
  3-schedule mean; winners alternate per schedule). The leak's home advantage does not
  transfer — as predicted by the Phase-1 critique.
- **No deployable adaptive beats a well-chosen fixed λ on unseen temporal drift @150R**
  (robust fixed λ0.2 +0.6 pp over main; dev-history fixed λ0.4 ties main). Positioning
  locked: "approaches hindsight-tuned fixed", never "beats fixed".
- Labeled causal oracle stays +1.1-1.6 pp above hindsight fixed = the labels-only headroom.
- Monitor armed for D3+D4+D5 (queue-empty condition).

---

## 2026-07-16/17 — D3/D4/D5 done; dataset blind spot found & FIXED (dual-view); FINAL ASSESSMENT WRITTEN

- D3 @150R: spatial ceiling persists (−4.2 pp worst vs fixed λ0.2); bounds ablation
  attributes ~⅓-½ to λ_max=0.7.
- **D4/D5 discovery**: change-only selfcal collapses on CIFAR-100/Tiny-IN (0/9 seeds,
  −1.6 to −3.8 pp vs fixed): the STATIC δ level (0.56-0.70) is normalized away →
  over-personalization. Legacy's absolute threshold transferred fine — the operating
  point depends on the LEVEL ("how hard"), not only the change.
- **Dual-view controller (v3d, constant-free)**: λ = min(λ_selfcal(z), λ_abs(δ̂)) using
  δ's intrinsic probability meaning. Verification (26 runs): **+4.0/+3.4/+1.9 pp over
  v3c on C100-gsig/C100-sp/TinyIN (9/9 seeds), ABOVE fixed λ0.4 on all three (+0.2-0.3),
  zero regression on CIFAR-10** (D1 ±0.1-0.6, D2 +0.3, D3 worst +0.95 5/5).
- **selfcal-DV = final main method** (three dimensionless reference frames).
- `reports/final_tmc_ton_assessment.md` written: **TMC GO / ToN CONDITIONAL_GO**, claims
  supported/not-supported enumerated, 5 pre-submission experiment items listed.
- Figures: dataset_transfer, unseen_schedules (+ signal set). all_runs.csv @ 265 runs.

---

## 2026-07-18 — TMC closure directive: claim corrections, DV freeze, Gate-1/2 launched

- **Claim-correction ledger** written (`reports/tmc_remaining_issues_and_claim_corrections.md`):
  12 items, banned→replacement table (no "strictly dominates", "zero-constant",
  "beats the fixed grid", "unseen transfer" for C100/TinyIN, etc.).
- **DV FROZEN** (`provenance/dv_frozen_manifest.json` + `reports/dv_frozen_specification.md`):
  source hashes, all constants, design-contaminated datasets/schedules enumerated;
  holdout candidates: SVHN (selected; downloading), EMNIST, PACS, GTSRB, STL-10.
  One amendment: SIGNAL_RANGE table for Gate-2 (delta behavior identical, hash updated).
- **Mobility bug O3 FIXED** (`scripts/run_single.py` cid-map) + regression test +
  dormancy evidence test (no empty clients in seeds 0-4 partitions). 52 journal tests.
- **Infra built**: per-cell fixed λ/Λ; trajectory replay controller (identity/shuffle/
  shift10/shift20/reverse/cluster_shuffle/lowpass — value-preserving interventions);
  mean-matched job generator from frozen DV traces; mathematical-range normalization
  for non-δ signals' absolute view (TV/1, JS/ln2, cos/2; symKL undefined→documented).
- **Phase J complete** (code, no fake numbers): export (eager/TorchScript/ONNX),
  benchmark_ondevice.py + benchmark_edge_server.py, schema/protocol/table templates,
  build/plot tools that REFUSE unfilled metadata. Host-CPU reference: probe-path 81 ms
  (64 samples, 4 threads), signal 0.94 ms, controller update 0.05 ms.
- **Phase G machinery**: 5 tensor-space corruptions + severity schedules + CorruptedView
  (probe+eval coupled) + exit-agreement decomposition (both-correct/…/false-agreement)
  now recorded in every routing eval.
- **306 jobs queued** (priority): Gate-1 core 27 → Gate-2 signals 36 → full fixed grid
  ~171 → Λ pilot 18 → Gate-1 extended 54. Est. ~5 days at 8 workers.
- Monitors: Gate-1 core armed.

---

## 2026-07-23 — Phase D final baselines done · ResNet OOM handled · G running

- **Phase D final causal baselines (DV-2)** (`reports/phaseD_final_causal_baselines.md`):
  on unseen temporal schedules DV-2 matches the deployable fixed grid to within ±0.5 pp
  (ahead of legacy/fixed0.4 on 2/3, behind fixed0.2 on 2/3); labeled oracle leads all by
  +1.1–1.6 pp; DV-2 vs entropy-DV +2.18 pp (3/3) reconfirms C1 in-loop under TV. Verdict:
  best accuracy–STABILITY trade-off (churn 0.006–0.012 vs bandits 0.16–0.22, no tuning) —
  NOT lowest regret, NOT beats-fixed. C-4 fallback wording adopted.
- **ResNet Phase F OOM (32 fails) — diagnosed + isolated**: shared machine loaded by
  OTHER users (7 GB×2 jobs) + my 10 CNN workers → GPUs 96%/85% full; ResNet's per-client
  server copies don't fit under 10-way contention (solo smoke had passed). Fix: pulled
  ResNet from the shared queue; dedicated SELF-HEALING worker (`resnet_worker.sh`:
  expandable_segments + re-queue-on-OOM + 300 s backoff), 16 clients / 100R, 1 worker on
  the lighter GPU. Progresses opportunistically as memory frees; no jobs lost. Added
  --num_clients override. Main pool healthy (0 non-ResNet failures after the fix).
- Main queue order now: G corruption 54 (running) → H mobility 36 → I network 30 →
  grid 180 → g1ext 54. Monitors: G armed.

## 2026-07-24 — resource cleanup + CPU-thread fix (GPU0 0%→37%)

- **Diagnosis of "GPU0 memory-full, 0% util"**: NOT idle workload — our 4 GPU-0 corruption
  jobs were progressing (R50–R80). Root cause = **CPU saturation** (load avg 111 on 48
  cores): a dozen concurrent CPU-bound federated-sim jobs, each with PyTorch's default
  full thread pool, oversubscribed the cores and starved the GPU.
- **Killed 10 stale Jupyter kernels** (`ipykernel_launcher`, 7–22 days old, 0% util,
  340 MB–1.9 GB each) — user-authorized non-task processes → freed ~7 GB GPU (GPU0 12→5.8 GB).
  Only ipykernel PIDs killed (verified each before kill); no task process touched.
- **CPU-thread cap in run_v2.py** (`OMP/MKL/OPENBLAS_NUM_THREADS` + `torch.set_num_threads`
  = 4, env-overridable via JX_THREADS). Applies to every NEW job immediately (workers
  re-exec python each iteration) — no worker restart, no in-progress job lost. 12×(default
  threads) → capped so total ≈ core count.
- **Added 2 ResNet workers on the freed GPU0** (ResNet is GPU-heavier/CPU-lighter): GPU0
  util **0%→37%**, draining the Phase-F backlog. Pool now 10 CNN + 4 ResNet workers.
- load still ~115 momentarily (pre-cap jobs draining); settles toward ~48 as capped jobs
  replace them. No jobs lost.

## 2026-07-28 — session resume: workers restarted; Phase H + I COMPLETE

- On resume both GPUs were idle and ALL workers gone (session teardown ended them; GPU0 10 h
  window had also expired at ~11:09). No other users on the machine. Restarted GPU1-only
  workers (8 queue + 2 resnet) per standing policy; queue intact (218 main + 8 resnet), 0
  corrupt lines, 0 non-ResNet failures across the whole run.
- **Phase H mobility COMPLETE (36/36)** → `reports/phaseH_mobility_revalidation.md`:
  composition-coupled mobility (O3 bug fixed+tested). **DV-2 beats best fixed λ0.4 by
  +0.7–0.8 pp (3/3) AND entropy by +2.0 pp (3/3)** across slow/med/fast. This is the KEY
  reconciling result: adaptive wins where drift is genuinely DYNAMIC (mobility, temporal),
  loses on STATIC optima (SVHN/spatial) — consistent with Gate 1 + C-13, not contradictory.
- **Phase I final-DV network stress COMPLETE (30/30)** → `reports/phaseI_final_dv_network_stress.md`:
  DV-2 delay10 −0.8 pp, loss20 ≈0, topology second-order; participation 50% −7.1 pp is a
  training effect (fixed-λ drops ~−6.4). Confirms the abstract-model supplement for the
  FINAL method (was v3b).
- Phase F ResNet 26/36, G corruption 50/54, grid 9/180 + g1ext 45 — continuing on GPU1.

## 2026-07-24 (later) — consolidated to GPU1 only (per user request)

- Per-job remaining time computed from recent per-round pace (ETA display was
  starvation-inflated): all 6 GPU-0 jobs had >1 h left (corruption 20–50 rounds remaining
  at a starved pace; 2 ResNet ~132 m / just-started). None <1 h → all killed + requeued.
- **Killed all cuda:0 workers** (4 queue + 2 resnet) and their 6 in-flight jobs; captured
  each command first and **requeued** (4 corruption → main-queue tail, 2 ResNet →
  resnet_queue). No job lost from the manifest.
- **GPU0 now idle** (41 MiB, 0%); **GPU1 only** — 8 queue + 2 ResNet workers, GPU1 at 100%
  util, 12 GB. **load 111 → 43.6** (thread cap + GPU0 drain). All future work is cuda:1.
- Watch item: corruption pace was 42 m/round UNDER the load-111 starvation; expected far
  faster now (load 43, GPU1 100%). If still pathological, optimize CorruptedView
  (per-image Generator) before relying on Phase-G throughput — observing first, not
  changing experiment code preemptively.

## 2026-07-28 — session resume: workers restarted, both GPUs re-enabled

- All workers had stopped (previous Claude process teardown; the setsid workers did not
  survive). Both GPUs idle, no foreign procs. Queue intact (204 main + 4 resnet).
- Restarted 8 cuda:1 queue + 2 cuda:1 resnet workers; then (per user) re-enabled GPU0:
  added 6 cuda:0 queue + 2 cuda:0 resnet workers. **Both GPUs at 100% util**
  (GPU0 8 GB, GPU1 15 GB), 30 concurrent run_v2 jobs, load 12 (thread-capped, healthy).
- Phase completion at resume: **H mobility 36/36 ✓, I network 30/30 ✓**, F ResNet 26/36,
  G corruption 50/54, g1ext/decomp 45, grid 9. Remaining queue is g1ext tail + full grid
  (confirmatory) + 4 slow corruption + ResNet middle/late.
- GPU0 10 h window (01:09–11:09) had already been used + expired before resume; that batch
  of g1ext/grid jobs completed. Now GPU0 is freely usable again per user.

## 2026-07-28 (17:46) — workers kept dying across sessions → durable cron supervisor

- **Root cause found**: setsid workers launched WITHOUT `< /dev/null` stdin detach did not
  survive session teardown (the 14:39 relaunch died within ~3 h, 0 progress). Memory note
  "verify PPID=1" applied.
- **Fix 1**: relaunched all 16 workers (8 cuda:1 queue + 6 cuda:0 queue + 2+2 resnet) with
  `setsid nohup bash ... >/dev/null 2>&1 < /dev/null &`; **verified PPID=1** (fully
  detached) this time. Both GPUs back to ~100 % util, 19 jobs.
- **Fix 2 (durable)**: `scripts/supervisor.sh` (idempotent: restarts the pool only if
  workers are dead AND queue non-empty) installed in **system crontab** `*/10 * * * *`
  + `@reboot`. Survives Claude-session teardown AND machine reboot. This ends the
  repeated-death problem structurally.
- No progress had been lost (queue never corrupted; killed jobs re-queue). Remaining:
  204 main (g1ext tail + full grid + 4 slow corruption) + 4 resnet (F middle/late).

## 2026-07-28 (21:22) — H/I reports written; FIRST comprehensive report issued

- **Phase H mobility report** (`phaseH_mobility_revalidation.md`): DV-2 beats best fixed λ
  +0.6–1.4 pp AND entropy +2.0 pp, 3/3 seeds every speed (t up to 47), O3 bug fixed. The
  positive result that unifies the story (adaptive wins on DYNAMIC drift; fixed wins on
  static optima — no contradiction with the holdout).
- **Phase I network report** (`phaseI_final_dv_network_stress.md`): final-DV robust —
  delay/loss/topology spread 0.72 pp, participation is a training effect; `tables/
  final_dv_network_stress.csv`.
- **Verified remaining jobs are confirmatory only** (user's judgment correct): full grid
  confirms per-env DV-vs-fixed (on Schedule A, completed grid fx00/10/20 = 0.630/0.638/
  0.645 already TRAIL DV-2 0.6595 → reinforces §9); ResNet/g1ext/corruption tails set-in-
  direction. None can reverse C1/C2/boundary law.
- **FIRST comprehensive report** written: `reports/FIRST_REPORT_2026-07-28.md` — full
  synthesis (Gates A–D, Gate 1/2/3, holdout, D/G/H/I, overhead, 13 claim corrections,
  unified non-stationarity boundary law, provisional TMC GO/READY_WITH_LIMITATIONS).
  FINAL report to regenerate once the queue drains (tighter CIs + completed grid tables only).
- Workers healthy (cron supervisor active); both GPUs ~90–100 % util.

- **SVHN frozen holdout (74 runs, both shifts, 5 seeds) — the honesty test**:
  signal PASS (dual-exit divergence beats entropy +2.40 pp temporal / +0.76 pp spatial,
  10/10 seed-shifts); **adaptive-beats-fixed FAIL both shifts** (deployable robust fixed
  λ0.2 beats DV-2 by −1.05/−1.01 pp, worst −2.32 pp). Method NOT revised (freeze held).
  Claim C-13: retract all "beats/above fixed grid"; positioning = calibration-free
  controller (no per-deployment tuning; ~1 pp below oracle-tuned fixed on extreme-λ
  datasets; signal reliably beats entropy; temporal adaptation beats its OWN mean but not
  the best tunable fixed). Lead = signal (C1). `reports/phaseE_frozen_holdout.md`.
- **Phase F/G/H/I implemented + queued** (priority: D remainder → F ResNet-18 splits 36
  → G covariate corruption 54 → H composition-coupled mobility 36 → I final-DV network 30
  → grid 180 → g1ext 54). Built: ResNet split arms; 5 tensor corruptions + severity
  schedules + false-agreement decomposition (aggregation bug fixed before any G run);
  composition-coupled mobility in the v2 runner (moving changes membership + scope +
  Main/OOP/OOR together; smoke: 48/50 rewired at R5) with the O3 cid-index fix. 52 tests.
- DV-2 manifest amended (harness/eval changes only; controller math identical, tests confirm).

## 2026-07-19/20 — Gate 1 VERDICT (Case A temporal / Case B spatial) · Gate 2 → TV · DV-2 FROZEN · holdout preregistered & queued

- **Gate 1 (adaptation value)**: DV beats global/per-cluster mean-matched fixed AND its
  own time-shuffled trajectory on BOTH temporal schedules (18/18 paired seeds, +0.4–0.8 pp,
  t 2.9–5.0) → **Case A on temporal drift**. On static spatial: ties per-cluster
  mean-matched (+0.01 pp), beats global (+0.33 pp, t=18) → **Case B**. Framing locked
  per-regime. `reports/phaseB_adaptation_value.md`.
- **Gate 2 (signal)**: TV env-mean +0.37 pp vs δ (abrupt +0.68 3/3, C100 +0.85 3/3;
  worst-case −0.16 = noise); JS/KL/cos negative → excluded. **Final signal = TV
  ("dual-exit divergence")**; δ = hard-variant ablation. `reports/phaseC_*.md`.
- **Gate 3 freeze → DV-2** (`provenance/dv2_frozen_manifest.json`).
- **SVHN holdout preregistered** (before any run; binding criteria) and 74 runs queued
  at top priority + 6 TV-trace Gate-1 confirmation runs + 18 Phase-D DV-2/entropy arms.
  Queue 332; monitors armed (holdout).

### Infrastructure added for autonomous continuation
- **File-queue worker system** (`scripts/queue_worker.sh` + `enqueue_phase456.py`):
  flock-atomic job queue at `runs/queue/queue.txt`; workers survive session end
  (setsid), poll for new jobs, stop via `runs/queue/STOP`. 2 workers live (1/GPU).
- **65 jobs enqueued** (priority order): v3b downstream ×6 → Gate-C fairness ×16
  (spatial equal_spread: selfcal-v3b baseline / F1F2-deploy / oracle / random donors /
  fixed λ0.2/0.4, mostly 3 seeds) → Phase-4 causal baselines ×22 (UCB/EXP3/greedy-labeled-
  oracle ×A,abrupt×3 seeds + proxy-grid, update-norm ×1 seed) → Phase-6 network ×21
  (signal delay 2/5/10, loss 10/20%, topology ring/star/dynamic, participation 50%).
- Phase-7 overhead microbenchmark DONE (`tables/overhead.csv`): δ probe n=64 =
  1.5 ms/client/round (+0.7 ms vs entropy-only), full 16-signal library 6.9 ms;
  controller uplink 4 B/client/round vs 12.1 MB/round model exchange (overlap-2);
  probe activation upload 0 B (server copies live on-client in SplitOMC).
- Analyzers ready: `analyze_downstream.py`, `analyze_gatec.py`, `gen_all_runs.py`
  (tables/all_runs.csv, 17 runs so far). Datasets verified: CIFAR-100, Tiny-ImageNet
  (downloaded, 481 MB). matplotlib fixed (3.11.0).
