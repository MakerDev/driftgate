# Task 0 — Headline Recomputation Audit (raw run JSON)

## Schedule A
adaptive DriftGate per-seed: s0=0.6597, s1=0.6570, s2=0.6617, s3=0.6294, s4=0.6893
  adaptive mean (n=5) = 0.659425
  adaptive eval-rounds set(s): {(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150)}
fixed grid (Λ=0.5) per-λ mean and seeds:
  fx00: mean 0.630151 seeds [0, 1, 2] s0=0.6286 s1=0.6295 s2=0.6323
  fx10: mean 0.637678 seeds [0, 1, 2] s0=0.6368 s1=0.6364 s2=0.6398
  fx20: mean 0.644543 seeds [0, 1, 2] s0=0.6451 s1=0.6426 s2=0.6460
  fx40: mean 0.652089 seeds [0, 1, 2, 3, 4] s0=0.6529 s1=0.6475 s2=0.6526 s3=0.6252 s4=0.6822
  fx50: mean 0.649013 seeds [0, 1, 2, 3, 4] s0=0.6505 s1=0.6455 s2=0.6517 s3=0.6189 s4=0.6785
  fx60: mean 0.644946 seeds [0, 1, 2] s0=0.6453 s1=0.6419 s2=0.6476
  fx70: mean 0.642254 seeds [0, 1, 2] s0=0.6415 s1=0.6410 s2=0.6443
  fx80: mean 0.645111 seeds [0, 1, 2] s0=0.6427 s1=0.6467 s2=0.6459

Best fixed on the adaptive-shared seed set: fx40 mean 0.652089 seeds [0, 1, 2, 3, 4]
adaptive − fx40 paired (seeds [0, 1, 2, 3, 4], n=5): mean +0.734pp, per-seed [0.0068, 0.0096, 0.0091, 0.0041, 0.0071]
  [2-seed subset s1,s2]: fx40 mean 0.6500, adaptive−fixed +0.935pp

## Task C Schedule-A cell recomputation
  Task-C DriftGate (3-seed dvsig only) = 0.659498 (seeds [0, 1, 2])
  robust fixed λ0.2 (grid) = 0.644543 (seeds [0, 1, 2])
  hindsight best fixed (Λ0.5) = fx40 but on 3-seed dvsig subset: 0.651007

## asym-return
  DriftGate: mean 0.647231 seeds [0, 1, 2]
  fixed λ0.4: mean 0.642634 seeds [0, 1, 2, 3, 4]
  fixed λ0.2(robust): mean 0.642335 seeds [0, 1, 2, 3, 4]
  DriftGate − fixed0.4 (best): +0.605pp seeds [0, 1, 2]
  DriftGate − fixed0.2 (robust/TaskC): +0.844pp seeds [0, 1, 2]

## CIFAR-100 spatial fixed grid
  fx00: 0.362860 seeds [0, 1, 2]
  fx10: 0.364962 seeds [0, 1, 2]
  fx20: 0.365684 seeds [0, 1, 2]
  fx30: 0.364139 seeds [0, 1, 2]
  fx50: 0.346700 seeds [0, 1, 2]
  fx60: 0.327864 seeds [0, 1, 2]
  fx70: 0.300455 seeds [0, 1, 2]
  fx80: 0.266248 seeds [0, 1, 2]
  >>> grid MAX = fx20 0.365684; fx20 = 0.3656840555399488

---

## Discrepancy resolutions (raw-run classification)

| # | discrepancy | resolved value(s) | cause class |
|---|---|---|---|
| 1 | Schedule A 0.6594 vs 0.6595 | 0.6594 = 5-seed mean; 0.6595 = 3-seed (Task C used dvsig s0–s2) | different seed set (both correct) |
| 2 | adaptive−fixed +0.73 vs +0.93 | +0.734pp = 5-seed paired; +0.935pp = 2-seed subset (s1,s2) | different seed set (both correct) |
| 3 | "s4 within normal range" | s4=0.6893 is a HIGH outlier; s3=0.6294 low; 5-seed spread 0.629–0.689 | stale/wrong characterization (prior text) |
| 4 | Task C hindsight 0.6500 | 0.6500 = fx40 on 2-seed (s1,s2); Task-C 3-seed hindsight fx40 = 0.6510 | different seed set |
| 5 | asym +0.60 vs +0.49 | +0.605pp = DG−fixedλ0.4 (best) PAIRED 3-seed; +0.49 = 3-seed DG mean − 5-seed fixedλ0.2 mean (MIXED-n). Correct paired DG−fixedλ0.2 = +0.844pp | mixed 3-seed/5-seed subtraction (Task C error) |
| 6 | CIFAR-100 spatial: grid best 0.3629 < fx20 0.3657 | grid MAX = fx20 = 0.3657 (fx00=0.3629 is NOT the max). No contradiction: fx20 is both robust λ0.2 and grid max | wrong argmax / wrong run ID (prior text) |
| 7 | A2 DG−best-fixed(λ,Λ) +0.78 | recomputed PAIRED on the 3 common seeds = +0.784pp (prior used 5-seed DG mean − 3-seed fixed mean = MIXED-n, coincidentally ≈ same) | mixed 3-seed/5-seed subtraction |
| — | Task C gradual-sigmoid −0.51 | paired 3-seed = −0.14pp (prior 3-seed DG − 5-seed fixed = −0.51, MIXED-n) | mixed 3-seed/5-seed subtraction |

**Net corrections for the final tables**: (a) label every Schedule-A number by seed count
(5-seed adaptive 0.6594 / paired +0.734; the +0.93 was a 2-seed subset). (b) Task C rows
asym and gradual-sigmoid must be recomputed PAIRED on common seeds (asym +0.49→+0.84;
gsig −0.51→−0.14); other rows already had equal n. (c) CIFAR-100 spatial hindsight best is
fx20 (0.3657), not fx00. (d) A2 and all mixed-n comparisons re-stated as paired-on-common-seed.
All 7 automated asserts (§3.3) PASS after these corrections.
