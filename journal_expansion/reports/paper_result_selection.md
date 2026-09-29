# §13 — Paper Result Selection (main vs supplementary vs exclude)

**Date**: 2026-08-02 · Maps every result to a placement, figure/table, claim, and the
sentence it supports. Internal-process artifacts are excluded per §1.2.

## Main paper

| result | section | fig/table | claim / sentence |
|---|---|---|---|
| entropy vs TV trajectory; raw AUROC; 150R same-controller signal swap (+1.06 pp, 5/5; ρ=0.8 +3.64 pp) | Signal motivation | Fig: signal_traj + AUROC bar; Tab: 150R swap | "Dual-exit divergence is a convergence-robust drift signal that outperforms absolute entropy." |
| three reference views (temporal/spatial/absolute) + DriftGate mapping | Method | Fig: system/controller | method definition |
| component ablation (temporal-only / +spatial / +absolute = DriftGate) | Method ablation | Tab: view ablation (NO dev-version names) | each view is necessary |
| Gate-1 mean-matched + time-shuffled decomposition | Adaptation value | Fig: dv_vs_mean_matched | "temporal adaptation adds value beyond a fixed operating point" |
| preferred-operating-point variation vs gain (Spearman +0.82) | When it helps | Fig: adaptive_gain_vs_operating_point_variation | "helps when the preferred level moves trackably" |
| DV-2 vs FULL fixed grid (A +0.93, asym +0.55, gsig tie; static/transfer fixed wins) | Main eval | Tab: full-grid representative | "beats/matches the full fixed grid on temporal drift; fixed preferable on static optima" |
| frozen SVHN holdout (signal +2.40/+0.76; fixed non-inferiority fails) | Main eval | Tab: holdout | "signal generalizes to a held-out dataset; adaptive loses to tuned fixed on its static optimum" |
| composition-coupled mobility (DV-2 > fixed +0.6–1.4 pp; > entropy +2.0 pp) | Main eval | Tab: mobility | "under mobility the controller beats fixed and entropy" |
| role-separation ablation (R1 vs R2/R3/R4) | Main eval | Fig: role signal quality + prediction decomposition | "the signal comes from complementary roles, not arbitrary diversity" (conditioned on §7) |
| ResNet representative splits (early/middle[/late]) | Architecture | Tab: split depth | "signal benefit depends on role separation" (with §7) |
| network delay/loss representative | Robustness | Tab: delay/loss | "controller input robust to stale/lossy signal" |
| communication accounting (SplitOMC baseline + DriftGate +4 B/client/round) | System | Tab: comm accounting | "one scalar per client over the SplitOMC baseline" |
| on-device latency (user-supplied) | System | Tab: on-device | pending user measurement |

## Supplementary
Full fixed grid (all λ, all envs); all causal bandits (UCB/EXP3/proxy-grid/update-norm);
all topologies; all corruption severities + full false-agreement decomposition; failed
signal families (JS/symKL/cosine); detailed calibration-leakage audit; ResNet late-split;
donor/fairness negative; routing null; source-calibrated controller; per-seed tables.

## Excluded (neither main nor supplementary)
Worker/queue logs; run/provenance counts; cron supervisor; OOM/session-teardown recovery;
v1/v3b/v3c development-version names; tuning-iteration history; internal bug-discovery
narrative; artifact index.

## Statistical presentation rules (§12) applied
Main text: mean ± sd, paired mean difference, 95% CI, #improved seeds, effect size. Drop
raw t=43/t=47 emphasis. 3-seed results labeled "pilot/ablation"; 5-seed for main claims.
Full-grid best-per-test-env is a HINDSIGHT reference (selection bias) — never called
deployable; the deployable fixed baseline is the development-selected robust λ0.2.
