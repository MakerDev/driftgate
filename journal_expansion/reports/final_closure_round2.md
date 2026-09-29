# DriftGate — Final Evaluation Closure, Round 2

> Single new record for the Round-2 directive (Tasks 4–7). All numbers recomputed
> from **raw run JSON**; canonical `integrated = mean(acc_total over eval rounds)`.
> **Disjoint probe/eval protocol on every new run**; no same-pool run created; no
> paired difference mixes protocols or seed counts. Method & parameters FROZEN
> (commit `e82b96b`, `configs/base_v3.yaml`). No existing report/record modified.
> Tables: `tables/final_closure_round2/`. Scripts: `r2_task5_signal_audit.py`,
> `r2_task7_tables.py`, `r2_task7_finalize.py` (**15/15 consistency asserts pass**).
> New compute: 35 disjoint runs, ≈200 GPU-h on 2× RTX 3090 Ti.

---

## 1. One-page conclusion

- **Mobility hard/soft seed count.** Not a missing run: **both `soft` and `hard` have valid disjoint seeds 0,1,2** (13 eval rounds, overlap 0). The Round-1 "2 matched seeds" note was a **stale snapshot** from the partial analysis taken before seed-2 finished; the Round-1 final table already used 3 seeds. No rerun needed. (`mobility_direct_signal_inventory.csv`)
- **Introduction signal-quality numbers — reproduced from raw logs.** Source is the passive fixed-λ `rec_*` runs via `analyze_signals.py`, ρ from `get_rho` (deterministic). Using the **existing** definition `evaluate_signal` (warm-up 15, drift-active = ρ≥0.5, raw-direction AUROC): TV Spearman **0.907** (claim 0.92 — slight overstatement), TV AUROC **0.978 / 1.000** (claim 0.98–1.00 — reproduced), entropy AUROC abrupt **0.161** (claim 0.16 — reproduced, but direction-free is **0.839**). The "entropy Spearman ≈ 0.50" claim does **not** hold (actual A **+0.44**, abrupt **−0.58**; entropy's sign flips by schedule) → **REPLACED**. (`paper_number_status.csv`)
- **ρ-logging correction.** Round-1 stated "Schedule-A `rho_trace` is identically 0." **That was an endpoint-reading error.** Schedule-A ρ is a real step trajectory **0 → 0.4 → 0.8 (peak R75) → 0.4 → 0** (90/150 rounds non-zero). Passive signal quality is therefore computable on Schedule A after all; this report does so.
- **High-OOP/OOR (ρ=0.8) segment, disjoint.** Adaptive beats entropy massively: TV−entropy **+6.71 pp**, hard−entropy **+7.51 pp** (5/5). Adaptive beats fixed λ0.4 (+1.35/+2.15 pp). **But the pre-registered robust fixed λ0.2 is the single best at the sustained high-drift segment (61.40%)**, above hard (60.60), TV (59.80), fixed λ0.4 (58.45), entropy (53.09) — TV−fx0.2 = −1.60 pp (0/5). This is the known boundary: a low-λ fixed policy is built for exactly this worst case; the adaptive gain shows over the *whole* trajectory, not the sustained-peak slice. (`high_nonmain_segment.csv`)
- **Hard vs TV across settings — setting-dependent.** hard is better on SVHN (**+1.19 pp**) and Schedule A (**+0.61 pp**), roughly tied on mobility (+0.19), CIFAR-10 gradual (+0.23) and Tiny (−0.07), and **worse on CIFAR-100 spatial (−1.44 pp) and gradual (−0.69 pp)**. Families with hard > TV: **3/5**. Cause identified: on many-class datasets the base "argmax ∉ Main" rate is high (only 20% of classes are Main), so the hard signal's absolute level conflates a structural base rate with drift and its abs-branch saturates (CIFAR-100 spatial abs-active fraction **0.998**), forcing over-generalization. (`hard_vs_tv_all_settings.csv`)
- **Server-role diagnostic.** The hard signal tracks ρ about equally whether the server has a separated generalized role, the same role as the client, or is weak (Spearman **0.921 / 0.930 / 0.894**). Its absolute level does depend on role (weak-server mean 0.045 vs separated 0.324), but its **drift-tracking does not require role separation** — the "generalized-server-role is essential" hypothesis is **not supported**. Reported as a single-seed 3-condition diagnostic; not expanded (see §6.5). (`server_role_nonmain_signal.csv`)
- **Final readiness state → `TV_PRIMARY_HARD_OPTIONAL`.** hard drops >0.2 pp below TV on CIFAR-100 (both variants) and its advantage concentrates on SVHN/CIFAR-10-temporal, so the two-signal-hard-default condition fails. **Keep client-server TV as the primary signal; report server non-Main prediction rate as a metadata-aware comparison** that wins where class count is small and drift is real.
- **Remaining beyond mobile measurement: none essential.** Core hard-vs-TV is closed on 7 settings. Deliberately skipped under §3.4/§10 budget (documented, not hidden): mobility slow/fast hard and role seed-expansion (hard is not the uniform default), and disjoint fixed-grid/entropy references for CIFAR-100/Tiny/CIFAR-10-gradual (marked **unmatched**, excluded from paired diffs).

---

## 2. Task 4 — server non-Main prediction rate vs client-server TV

### 2.1 Downstream accuracy, all settings (disjoint) — `hard_vs_tv_all_settings.csv`

| setting | n | TV | hard | **hard − TV** | hard − fixed-grid | TV − fixed-grid | abs-frac hard |
|---|---|---|---|---|---|---|---|
| SVHN temporal | 5 | 79.39 | 80.58 | **+1.187** (0/5 TV↑) | +0.14 | −1.05 | 0.150 |
| Schedule A | 5 | 65.84 | 66.45 | **+0.610** (1/5 TV↑) | +1.27 | +0.66 | 0.094 |
| CIFAR-10 gradual | 3 | 62.72 | 62.96 | +0.234 | — | — | 0.347 |
| mobility-med | 3 | 59.21 | 59.40 | +0.195 | +0.82 | +0.63 | 0.421 |
| Tiny-ImageNet | 3 | 23.93 | 23.86 | −0.074 | — | — | 0.755 |
| CIFAR-100 gradual | 3 | 36.10 | 35.41 | **−0.693** | — | — | 0.440 |
| CIFAR-100 spatial | 3 | 36.44 | 35.00 | **−1.440** | — | — | **0.998** |

Fixed-grid best is the **highest-mean disjoint fixed-λ arm at the same seeds** (Schedule A/mobility: fx40). CIFAR-100/Tiny/CIFAR-10-gradual have no disjoint fixed grid → those cells **unmatched** (not filled from other protocols).

### 2.2 Absolute-comparison active fraction (§3.5)

Fraction of post-warm-up cluster-rounds where the absolute view bound λ (`d̂ > sigmoid((z−z0)/τ)`), recomputed from recorded `controller_z` and raw signal with the frozen mapping (Z0=1.5, τ=0.75, SIGNAL_RANGE[tv]=SIGNAL_RANGE[hard]=1.0): both signals **mix** relative and absolute views (neither dominates on CIFAR-10/SVHN), but hard **saturates the absolute branch on CIFAR-100 spatial (0.998)** — the mechanism behind its loss there.

---

## 3. Task 5 — passive signal-quality audit

All metrics use the **existing** `src/evaluation/signal_metrics.evaluate_signal` (warm-up 15, drift-active = ρ≥0.5, raw-direction AUROC + direction-free = max(a,1−a)). No new metric invented.

### 3.1 Introduction numbers (Part A — reproduce from passive fixed-λ `rec_*`)

| claim | reproduced | status |
|---|---|---|
| TV Spearman ≈ 0.92 | **0.907** (rec_A, 3 seeds) | REPRODUCED (0.92 slightly high) |
| entropy Spearman ≈ 0.50 | A **+0.44**, abrupt **−0.58** | **REPLACED** (schedule-dependent, sign-flips) |
| TV AUROC 0.98–1.00 | A **0.978**, abrupt **1.000** | REPRODUCED |
| entropy AUROC 0.16 | abrupt **0.161** (dir-free **0.839**) | REPRODUCED — must cite direction-free too |

Per §4.4: entropy's raw-direction AUROC 0.16 means the raw level is anti-predictive under abrupt convergence, **not** that entropy is information-free (direction-free 0.84). Entropy's ρ-relationship flips sign across schedules (A +0.44 vs abrupt −0.58).

### 3.2 Final fair comparison (Part B) — four paper signals, common backbone

Server non-Main soft/hard exist only on controller backbones, so the 4-way uses the **entropy-controller run** as the shared trajectory. Per §4.2 this is **not** called a neutral backbone: *the four signals are computed on one trajectory, but that trajectory was shaped by the entropy controller.*

| signal | Schedule A Spearman | SVHN Spearman | SVHN raw-AUROC |
|---|---|---|---|
| client-server TV | **+0.783** | +0.433 | 0.876 |
| server non-Main pred rate (hard) | +0.700 | **+0.866** | **1.000** |
| server non-Main prob mass (soft) | +0.373 | +0.709 | 0.974 |
| predictive entropy | +0.327 | −0.764 | 0.190 (dir-free 0.810) |

**Passive quality is setting-dependent**: TV ≳ hard on Schedule A (CIFAR-10 temporal), hard ≫ TV on SVHN — the same split seen downstream. (`passive_signal_quality.csv`)

---

## 4. Task 6 — remaining headline & role analysis

### 4.1 High-OOP/OOR segment (ρ=0.8, disjoint) — `high_nonmain_segment.csv`

Definition = mean acc_total over eval rounds where ρ=0.8 (rounds 70/80/90), from `task0_headline_audit.high_drift`.

| arm | ρ=0.8 segment acc | | paired @ρ=0.8 | mean | seeds+ |
|---|---|---|---|---|---|
| pre-fixed λ0.2 | **61.40** | | TV − entropy | **+6.71** | 5/5 |
| hard | 60.60 | | hard − entropy | **+7.51** | 5/5 |
| a1 (adaptive λ, Λ=.5) | 59.84 | | TV − fixed λ0.4 | +1.35 | 5/5 |
| TV | 59.80 | | hard − fixed λ0.4 | +2.15 | 5/5 |
| fixed λ0.4 | 58.45 | | TV − fixed λ0.2 | **−1.60** | 0/5 |
| entropy | 53.09 | | hard − TV | +0.80 | (hard↑) |

Honest reading: at the sustained peak, the **robust fixed λ0.2 wins** (it is designed to maximally generalize); adaptive's advantage is over entropy (huge) and over fixed λ0.4, and shows across the full trajectory (integrated), not this slice. The original same-pool "+3.64 pp" (TV−entropy @ρ=0.8) recomputes to **+6.71 pp disjoint (5-seed)** — reported as the Round-2 value; same-pool not mixed in.

### 4.2 Mobility fixed-Λ diagnostic (§5.2) — `fixed_lambda_capital_mobility.csv`

TV full (adaptive λ **and** Λ) − a1 (adaptive λ, Λ frozen 0.5) = **+0.070 pp**, 3/3, CI95 **[−0.05, +0.19]** (includes 0). As on Schedule A, **adaptive Λ adds essentially nothing** — the gain is adaptive-λ-driven. This is the **TV** variant; hard's fixed-Λ was not tested, so we do **not** generalize it to hard.

### 4.3 Server-role diagnostic (§5.3) — `server_role_nonmain_signal.csv`

| role | hard Spearman | hard AUROC | signal mean | server main / oop acc |
|---|---|---|---|---|
| separated (standard) | +0.921 | 0.993 | 0.324 | 0.707 / 0.308 |
| same_role | +0.930 | 1.000 | 0.424 | 0.664 / 0.369 |
| weak_server | +0.894 | 0.952 | 0.045 | 0.898 / 0.093 |

The hard signal's **ρ-tracking is preserved across all three roles** (0.89–0.93). Its absolute magnitude tracks the server's non-Main capacity (weak-server mean collapses to 0.045). Conclusion: server non-Main prediction rate **does not require a separated generalized server role** to track drift — the role-separation hypothesis is **not supported**. Single-seed, 3-condition; not seed-expanded (hard is not the default signal, §6.5).

### 4.4 Main-class metadata leakage audit (§2.6/§5.4) — `main_class_metadata_audit.csv`

Code-verified: `main_classes` come from `nd1_partition(train_labels)` — a function of (training labels, topology, seed) only. **No** eval-label, probe-label, current-traffic-label, or ρ access; OOP/OOR are **derived from** Main (not the reverse); Main is **fixed** under mobility. Ratio 0.20 for every dataset (2/10, 20/100, 40/200, 2/10); deterministic assignment hashes recorded. **No leakage.**

---

## 5. Final tables (`tables/final_closure_round2/`)

`hard_vs_tv_all_settings.csv` · `table_main_dynamic.csv` · `table_signal_comparison.csv` · `table_settings.csv` · `high_nonmain_segment.csv` · `fixed_lambda_capital_mobility.csv` · `server_role_nonmain_signal.csv` · `passive_signal_quality.csv` · `mobility_direct_signal_inventory.csv` · `main_class_metadata_audit.csv` · `paper_number_status.csv` · `run_manifest.csv`.

**15/15 consistency asserts pass** (`r2_task7_finalize.py`): no seed-count mixing; no protocol mixing; all new adaptive runs overlap 0 (fixed classified separately); eval grids read per-setting (not hardcoded); table means == per-seed means; diffs == per-seed diff means; fixed-grid best ≥ all grid values; Round-1 canonical reproduced (max|Δ|=0); mobility seed-2 resolved; metadata training-only; no new same-pool run; abs-fraction denominator defined; every Introduction number linked to a source run.

---

## 6. Signal-selection determination (§6.5)

Computed only on same-protocol, same-seed comparisons. Families averaged **within** family first (mobility counted once, not slow/med/fast separately).

| family | hard − TV (pp) |
|---|---|
| SVHN | **+1.19** |
| CIFAR-10 temporal (Schedule A + gradual) | **+0.42** |
| Mobility (med) | +0.19 |
| Tiny-ImageNet | −0.07 |
| CIFAR-100 (spatial + gradual) | **−1.07** |

Families hard > TV: **3/5**. Worst setting CIFAR-100 spatial **−1.44 pp**. hard drops **> 0.2 pp below TV** on CIFAR-100 (both), and its advantage concentrates on SVHN/CIFAR-10-temporal (small class count + real drift).

### → State: **`TV_PRIMARY_HARD_OPTIONAL`**

Rationale (§6.5 conditions met for this state): some settings have hard ≥ 0.2 pp below TV (CIFAR-100); hard's advantage is dataset-concentrated; hard saturates on high-class-count datasets (CIFAR-100 abs-fraction 0.998). **Not** `TWO_SIGNAL_READY_HARD_DEFAULT` (that needs hard never > 0.2 pp below TV on any core matched setting — CIFAR-100 fails it). **Not** `NON_MOBILE_EVALUATION_NOT_CLOSED` (no protocol mismatch, seed issue resolved, Introduction numbers sourced, metadata clean, ρ recovered, all asserts pass).

**Paper guidance this implies:** keep **client-server TV** as the primary label-free signal (no client Main-class metadata needed; robust across class counts). Present **server non-Main prediction rate** as a metadata-aware alternative that matches or beats TV where the class count is small and drift is real (SVHN, CIFAR-10 temporal, mobility) and is weaker on many-class datasets. All existing TV component/timing analyses remain **TV-variant** analyses; do not claim a hard-only method's components are separately validated.

---

## 7. Anomaly and sanity checks

- **Abnormal absolute accuracy:** none. Every new run's integrated is within ≈1 pp of its same-pool reference (e.g. CIFAR-100 spatial hard 36.58 vs ref 37.81; Tiny hard 23.96 vs ref 24.08).
- **Missing rounds:** none. 16 eval rounds (CIFAR-10/-100, SVHN, 150R), 13 (mobility, 120R), 11 (Tiny, 100R) — all read from each run's config, not hardcoded.
- **NaN / constant signal:** none across 35 runs; hard signal non-constant everywhere (147/143/113 unique values in the role runs).
- **λ, Λ bounds:** 0 violations across 35 runs (λ∈[0.15,0.70], Λ∈[0.40,0.70]).
- **Probe/eval overlap:** 0 on all 35 runs (24 adaptive + 8 fixed + 3 role).
- **Outlier seed:** Schedule-A s4 remains a natural high / s3 low across all arms — **not dropped**; paired diffs are within-seed.
- **Config vs resolved parameters (§2.1):** all frozen constants match the resolved config — z0=1.5, τ_z=0.75, EWMA α=0.3, baseline **β=0.05**, guard **z=0.5** (v3c CLI override of the 1.5 default, intentional), λ∈[0.15,0.70], Λ∈[0.40,0.70], warm-up 15, **MAD floor** σ≥max(1e-3, 0.10·max(|μ|,0.05)), **z-clip** (−2.0, 6.0). **No mismatch.**
- **Fixed-grid selection:** best fixed = argmax-mean disjoint fixed arm (Schedule A/mobility fx40); asserted ≥ all grid values.

---

## 8. Run manifest — `run_manifest.csv`

35 new disjoint runs (all fresh training, no eval-only, no checkpoint reuse), ≈**200 GPU-h**, commit `e82b96b`, config `configs/base_v3.yaml`, run IDs `selfcal_*`/`fixed_*` in each `config.run_id`.

| family | seeds | | family | seeds |
|---|---|---|---|---|
| CIFAR-100 spatial hard+TV | 0–2 | | Tiny-ImageNet hard+TV | 0–2 |
| CIFAR-100 gradual hard+TV | 0–2 | | CIFAR-10 gradual hard+TV | 0–2 |
| pre-fixed λ0.2 Schedule A | 0–4 | | pre-fixed λ0.2 mobility | 0–2 |
| role separated / same_role / weak_server | 0 (each) | | | |

**Reused (read-only, not re-run):** Round-1 disjoint runs (`phaseT1_disjoint`, `phaseT2_signal`) for TV/hard/entropy/fixed references on Schedule A / mobility / SVHN; passive `rec_*` for the Introduction-number reproduction; `canonical_headline_metrics.json` for the reproduction cross-check.

---

### Bottom line

Round-2 closes the non-mobile evaluation at state **`TV_PRIMARY_HARD_OPTIONAL`**. The core Round-1 claims survive and sharpen: adaptive-λ DriftGate beats entropy and fixed λ0.4 under disjoint pools and at high drift, with **adaptive Λ contributing ≈0**. The direct **server non-Main prediction rate is a genuinely competitive, sometimes-better signal — but only where class count is small and drift is real; it saturates and loses on many-class datasets, and its drift-tracking does not depend on server-role separation.** Keep TV primary; present hard as the metadata-aware alternative, honestly scoped.
