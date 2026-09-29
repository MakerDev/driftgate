# Phase B — Adaptation-Value Decomposition (Gate 1)

**Date**: 2026-07-19 · **Data**: `runs/phaseB_decomp/` (27 core runs, 0 failures) vs the
frozen DV runs (d2_dual_A, d1_dual_gradual_sigmoid, d3_dual_sp). 150R, 3 seeds, paired.
Table: `tables/adaptation_value_decomposition.csv` · Figure: `figures/dv_vs_mean_matched.*`
Extended trajectory interventions (shift10/20, reverse, cluster-shuffle, lowpass,
identity) are queued (g1ext) and will be appended; the core verdict does not depend on them.

## 1. Design

Each control preserves part of the DV trajectory and destroys the rest, per seed:
- **gmm** — global mean-matched fixed: λ = mean over rounds×cells of that seed's DV run
  (Λ likewise). Removes ALL adaptation; keeps the operating point.
- **pcm** — per-cluster mean-matched fixed: per-cell time-means. Keeps spatial
  configuration; removes temporal adaptation.
- **shuf** — time-shuffled replay: the exact DV λ/Λ sequences in random time order.
  Keeps the value distribution AND per-cell assignment; destroys temporal alignment.

## 2. Results (paired DV − control, integrated accuracy)

| env | vs gmm | vs pcm | vs shuffled |
|---|---|---|---|
| Schedule A (temporal) | **+0.41 pp** (3/3, t=3.5) | **+0.44 pp** (3/3, t=5.0) | **+0.42 pp** (3/3, t=3.7) |
| gradual sigmoid (temporal) | **+0.46 pp** (3/3, t=3.5) | **+0.49 pp** (3/3, t=3.3) | **+0.81 pp** (3/3, t=2.9; worst-cell +1.1 pp) |
| equal-spread (spatial, static) | **+0.33 pp** (3/3, t=18.0) | +0.01 pp (1/3, t=0.4) | +0.05 pp (2/3, t=0.5) |

## 3. Verdict (per the pre-registered §16 decision rule)

**Temporal drift environments → Case A: genuine temporal adaptation supported.**
DV consistently improves over every value-preserving control on both temporal schedules,
including the strongest one — its own λ values re-ordered in time (+0.42/+0.81 pp,
6/6 seeds). The same operating points applied at the wrong times lose accuracy; the
alignment itself carries value. The effect size is modest (+0.4–0.8 pp) and is reported
as *"consistently improves"* (all 18/18 paired comparisons positive on temporal envs;
paired t 2.9–5.0 at n=3), not as a large-margin superiority.

**Static spatial heterogeneity → Case B: automatic cluster-specific configuration.**
DV matches per-cluster mean-matched fixed exactly (+0.01 pp) — on a static environment
there is nothing temporal to track, and shuffling a near-constant trajectory changes
little (+0.05 pp), both exactly as theory predicts. Its value here is choosing the
per-cluster operating points without calibration: it beats the global mean-matched
control decisively (+0.33 pp, t=18).

## 4. Paper framing (locked by this result)

> On temporal drift, DriftGate performs **causal closed-loop adaptation**: its accuracy
> exceeds every fixed or time-scrambled control that preserves its own λ value
> distribution. On static spatial heterogeneity, its value reduces — as expected — to
> **calibration-free automatic configuration of per-cluster operating points.**

## 5. Confirmation under the FINAL signal (TV-DV, Gate-3)

Re-run of the Schedule-A decomposition with the frozen DV-2 (signal = TV), 3 seeds:

| control | paired Δ | seeds | t |
|---|---:|---:|---:|
| TV-DV − global mean-matched | **+0.90 pp** | 3/3 | 15.3 |
| TV-DV − time-shuffled | **+1.24 pp** | 3/3 | 3.5 |

The Case-A verdict is **stronger** under the final signal than under δ (+0.42→+1.24 pp
vs the time-shuffled control): the lower-variance TV estimator makes the temporal-alignment
benefit cleaner. Verdict unchanged; the numbers used in the paper come from this final-signal
run.

## 6. Corollaries

Corollaries for §RQ1 writing:
- "automatic operating-point selection only" (Case C) is REJECTED for temporal drift and
  ACCEPTED as the correct description for static spatial settings — stated per-regime.
- The mean-matched fixed baselines (gmm/pcm) are themselves non-deployable (they need
  DV's own run to compute their λ) — noted wherever they appear.
- Segment-mean oracle fixed (future-label analysis baseline) is subsumed by the g1ext
  lowpass/identity arms and will be appended for completeness.
