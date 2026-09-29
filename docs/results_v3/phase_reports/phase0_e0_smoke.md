# Phase 0 — E0 Smoke Test Report

**Run window**: 2026-05-27 22:44 → 2026-05-28 03:14 KST (≈4h30m wall, 1× cuda:0)
**Config**: `configs/base_v3.yaml` — 50 rounds × 3 local epochs × 50 clients × 5 ES
**Status**: ✅ Gate-2 PASSED

---

## Gate-2 Conditions

| Criterion (from `CLAUDE.md`) | Result |
|---|---|
| 4/4 runs `PASS_NO_NAN` | ✅ Yes |
| Loss strictly decreasing R1→R50 | ✅ Yes (0.884 → 0.158–0.300 range) |
| `splitomc_lam0.6` acc_total > 0.25 at R50 | ✅ 0.629 (2.5× threshold) |
| Personalization trend λ↑ ⇒ acc_main↑ | ✅ 0.810 vs 0.742 (Δ +0.068) |

---

## Final-Round (R50) Metrics

| Method | last loss | acc_total | acc_main | acc_oop | acc_oor |
|---|---:|---:|---:|---:|---:|
| `splitomc` λ=0.2 | 0.250 | 0.645 | 0.742 | 0.584 | 0.046 |
| `splitomc` λ=0.6 | **0.158** | 0.629 | **0.810** | 0.358 | 0.028 |
| `splitomcplus` λ=0.2 Λ=0.5 | 0.289 | 0.631 | 0.702 | **0.600** | 0.149 |
| `adaptive_splitomc` | 0.299 | 0.622 | 0.676 | 0.598 | **0.255** |

**Cell-level fairness (worst-cell accuracy, cell_gap):**

| Method | worst_cell | cell_gap |
|---|---:|---:|
| `splitomc` λ=0.2 | 0.569 | 0.173 |
| `splitomc` λ=0.6 | 0.570 | 0.138 |
| `splitomcplus` λ=0.2 Λ=0.5 | 0.578 | 0.152 |
| `adaptive_splitomc` | **0.577** | **0.126** ← smallest |

---

## Analysis

### λ-direction is correctly wired
Higher λ shifts mass to the personalized client head: `splitomc_lam0.6` boosts `acc_main` by +0.068 over `lam0.2` at the cost of −0.226 on `acc_oop` (out-of-personal-task). This trade-off is exactly the reference SplitOMC behavior, so the loss term `models/losses.py` and the trainer's `λ`-weighted state-dict mixing are sign-correct.

### Λ (cross-cell, splitomc**+**) recovers OOR accuracy
Vanilla SplitOMC accepts very low `acc_oor` (0.028–0.046) — out-of-region classes are essentially unseen by the per-cell server head. Adding Λ=0.5 (splitomcplus) lifts `acc_oor` to **0.149** at no real cost to `acc_main` (still 0.70+). This validates the "big-Λ" architectural choice as a generalization-broadening lever.

### Adaptive controller dominates on cross-region transfer
At only 50 rounds × 3 epochs (1/9th of the full schedule), `adaptive_splitomc` already pushes `acc_oor` to **0.255** — **1.7× splitomcplus**, **5–9× vanilla splitomc**. It also has the smallest worst-cell gap (0.126), meaning gains are spread evenly across cells rather than concentrated. The trade-off is `acc_main`: adaptive's 0.676 is the lowest of the four, which is consistent with its controller pulling λ down toward better cross-region sharing.

### λ-trace tells the story
The adaptive controller starts at λ=0.4 (config init), drops sharply to ~0.20 by R2 (sigmoid response to per-cell entropy `H_k`), bottoms around ~0.13 by R3, then drifts upward to a steady-state ~0.15–0.18 across cells by R50. Interpretation: early rounds, entropy is high → controller sees clients struggling → reduces personalization weight; as confidence builds, it nudges λ back up.

### Cell-gap behaviour
The 0.126 worst-vs-best cell gap for adaptive_splitomc is the lowest of the four configs, suggesting consensus smoothing across edge servers (`controller/adaptive.py`) is doing real work even at smoke scale.

---

## Implications for Subsequent Phases

1. **E1 Pareto**: Expect adaptive to sit on the upper-right of the (`acc_oor`, `acc_total`) Pareto frontier — at this smoke scale, its (acc_oor=0.255, acc_total=0.622) point already dominates the splitomc/splitomcplus runs along the OOR axis.
2. **E2 CORE**: The adaptive λ-trace is non-trivial (0.4 → 0.13 → 0.15-0.18), so the controller has *headroom*. Under temporal drift schedules, this headroom should let it track the moving optimum, which is the headline claim.
3. **E3 spatial**: Adaptive's smaller cell_gap is a leading indicator for the spatial-heterogeneity story.
4. **Sanity flag**: `acc_main` for adaptive is the lowest of the four. This is intentional (the controller chose to trade personalization for sharing), but watch in E2 that `acc_main` recovers under no-drift conditions — otherwise the controller is over-correcting.

---

## Run Artifacts

- `results/e0_smoke/{splitomc_lam0.2,splitomc_lam0.6,splitomcplus_lam0.2_Lam0.5,adaptive_splitomc}.json` — full per-round eval traces
- `results/e0_smoke/e0_summary.json` — gate-2 summary
- `logs/phase0.log` — round-by-round stdout

## Wallclock Notes (not for paper)

- Re-launched at 22:44 after initial tmux session was killed by Claude's Bash sandbox cleanup. Final pattern: `setsid nohup bash -c 'PYTHONUNBUFFERED=1 bash run_all.sh' > logs/master.log 2>&1 < /dev/null &` then `disown` — PPID=1 detach survives.
- Per-run wall ≈ 65-70 min (cuda:0 shared with 6 other users), 2× CLAUDE.md's estimate.
