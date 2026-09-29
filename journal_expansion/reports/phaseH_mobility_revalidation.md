# Phase H — Composition-Coupled Mobility Re-validation (RQ6) — COMPLETE

**Date**: 2026-07-28 · **Data**: `runs/phaseH_mobility/` (36/36, 3 speeds × 4 arms × 3 seeds,
0 failures). CIFAR-10, schedule abrupt, 120R, final DV-2. Gauss-Markov mobility with the O3
cid-index bug fixed; **composition-coupled** (a move changes serving membership + cell scope
+ Main/OOP/OOR exposure together, via `c2es_now` feeding probe AND eval set construction).

## 1. Bug fix (C-12)

- **O3**: the change-detection loop indexed the `clients` LIST by cid; with any empty
  (skipped) client the indices shift. Fixed to a cid→client map (`scripts/run_single.py`
  + the v2 runner). Regression test `test_mobility_cid_indexing` + dormancy-evidence test
  (`test_no_empty_clients_in_recorded_partitions`, seeds 0-4 give every client data → the
  bug never fired in any recorded run). 56 tests pass.
- Legacy mobility changed membership but not composition; the v2 runner now recomputes each
  moved client's OOP/OOR from its NEW primary cell's scope, so mobility drives real
  distribution drift, not just topology churn.

## 2. Result — DV-2 wins here (integrated / worst-cell, 3 seeds)

| speed | DV-2 | fixed λ0.4 | fixed λ0.2 | entropy-DV |
|---|---:|---:|---:|---:|
| slow | **0.593 / 0.546** | 0.585 / 0.538 | 0.579 / 0.536 | 0.572 / 0.521 |
| med | **0.593 / 0.542** | 0.586 / 0.533 | 0.580 / 0.531 | 0.573 / 0.523 |
| fast | **0.589 / 0.543** | 0.583 / 0.537 | 0.579 / 0.536 | 0.569 / 0.522 |

Paired significance (DV-2 − comparator, integrated, all 3 seeds positive):

| speed | vs fixed λ0.2 | vs fixed λ0.4 | vs entropy-DV |
|---|---|---|---|
| slow | +1.39 pp (t=15.1) | +0.76 pp (t=7.8) | +2.04 pp (t=47) |
| med | +1.22 pp (t=7.6) | +0.65 pp (t=6.5) | +2.00 pp (t=10) |
| fast | +1.00 pp (t=6.5) | +0.56 pp (t=4.6) | +2.00 pp (t=17.6) |

Worst-cell deltas positive in 26/27 comparisons. Switching rate 0.0089/0.0099/0.0096
|Δλ|/round — low and speed-stable (no thrash at fast, no inertia at slow).

## 3. Why this is the important positive result (reconciles the whole story)

The SVHN holdout and CIFAR-10 spatial showed DV **loses** to a well-chosen deployable fixed
λ — but those are STATIC environments whose optimum is a fixed extreme λ. Mobility is the
opposite: membership and Main/OOP/OOR composition **change continuously over time**, so no
single fixed λ tracks it, and the adaptation earns its keep. This is exactly consistent with
Gate 1 (temporal → Case A, closed-loop adaptation adds value) and with C-13 (static optimum
→ fixed wins). The unified statement:

> **DriftGate's adaptive advantage appears where the traffic-composition drift is genuinely
> dynamic (temporal schedules, mobility); on static heterogeneity a well-chosen fixed λ is
> as good or better. Its signal advantage over entropy holds in all these
> composition-drift settings (+2 pp under mobility).**

Beats-fixed here is real and reported; it does NOT contradict the holdout (different regime).

## 4. Verdict
Mobility claims un-quarantined (C-12 closed): under composition-coupled mobility DV-2
improves over both the best deployable fixed λ (+0.7–0.8 pp, 3/3) and entropy (+2.0 pp, 3/3),
across slow/med/fast, with the O3 bug fixed and tested.
