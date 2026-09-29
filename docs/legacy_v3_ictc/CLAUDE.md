# CLAUDE.md — Adaptive-SplitOMC v3 Autonomous Execution Instructions

> **Purpose**: This file is read by Claude Code at session start. It contains everything Claude Code needs to run the full experimental pipeline autonomously and produce paper-ready results.

> **Researcher**: 유진 (Yujin Shin) — Yonsei + UMich visiting. Target venue: ICTC 2026.

---

## 0. Mission

Run the complete v3 experimental pipeline (E0 → E5) and produce paper-ready outputs:
- `docs/results_v3/interpretation.md` — auto-narrated findings
- `docs/results_v3/tables/*.md` — paper-ready tables
- `docs/results_v3/figures/*.png` — paper-ready figures
- `docs/paper_v3_filled.md` — paper draft with real numbers filled in

**Total wall-time estimate**: 30-40 hours on 2× RTX 3090.

---

## 1. Hardware assumptions

- 2 GPUs (default `cuda:0`, `cuda:1`)
- ~30-50 GB free disk space
- Python 3.10+, PyTorch 2.0+, torchvision, numpy, matplotlib, pyyaml, pytest

If only 1 GPU is available, set `DEVICE1=cuda:0` (sequential) — wall-time doubles.

---

## 2. Project layout

```
adaptive_splitomc_v3/
├── configs/base_v3.yaml         # All hyperparameters
├── data/partition.py            # ND1 partition (reference-aligned)
├── models/architectures.py      # Client(280K) + Server(1.38M) split arch
├── models/losses.py             # Multi-exit loss γ=0.5 (reference exactly)
├── train/trainer.py             # SplitOMCClient (|Z_k| server models) + EdgeServer
├── eval/evaluator.py            # Entropy-routed per-task per-exit accuracy
├── controller/adaptive.py       # H_k, Δ_k signals + consensus + sigmoid → (λ_z, Λ_z)
├── network/mobility.py          # Gauss-Markov mobility + rewire_clients
├── scripts/run_single.py        # Central experiment runner (CLI)
├── scripts/run_e0_smoke.py      # Verification gate
├── scripts/run_e1_static.py     # Pareto sweep
├── scripts/run_e2_temporal.py   # CORE: temporal ρ drift
├── scripts/run_e3_spatial.py    # Spatial heterogeneity
├── scripts/run_e4_mobility.py   # Mobility scenario
├── scripts/run_e5_ablation.py   # Component ablation
├── scripts/analyze_all.py       # Tables + figures + interpretation
├── scripts/fill_paper.py        # Paper draft filling
├── run_all.sh                   # Master pipeline (this is what to invoke)
├── tests/test_all.py            # 12 unit tests (all must pass)
├── EXECUTE_ALL.md               # Step-by-step execution checklist
└── CLAUDE.md                    # This file
```

---

## 3. Single-command autonomous execution

```bash
cd adaptive_splitomc_v3
bash run_all.sh
```

This runs **everything**:
1. Unit tests (gate 1, ~1 minute)
2. Phase 0: E0 smoke (gate 2, ~3 hours)
3. Phase 1: E1 static Pareto (~10-13 hours)
4. Phase 2: E2 temporal drift CORE (~7 hours)
5. Phase 3: E3 spatial + E4 mobility + E5 ablation (~12 hours)
6. Phase 4: analysis + paper filling (~5 minutes)

**Total**: ~32-35 hours on 2 GPUs.

### Running in background with tmux (recommended)

```bash
tmux new-session -s adaptive_v3
cd adaptive_splitomc_v3
bash run_all.sh 2>&1 | tee logs/master.log
# Ctrl+B then D to detach
# Later: tmux attach -t adaptive_v3
```

---

## 4. Gates and decision points

### Gate 1: Unit tests
- **Pass condition**: `pytest tests/test_all.py` returns 12/12 passed.
- **If fail**: Stop. Read error. Likely a Python version or torch incompatibility.

### Gate 2: E0 smoke (after Phase 0)
- **Pass condition**:
  - All 4 runs (`splitomc_lam0.2`, `splitomc_lam0.6`, `splitomcplus_lam0.2_Lam0.5`, `adaptive_splitomc`) complete without NaN.
  - `splitomc_lam0.6` has acc_total > 0.25 at round 50 (must beat random 0.10 by margin).
  - Final loss > 0 and < 5.
- **If fail**: Inspect `results/e0_smoke/e0_summary.json`. Possible issues:
  - Loss not decreasing → check `models/losses.py` and `train/trainer.py` for sign/direction.
  - All accuracies stuck near 0.10 → partition issue; re-verify `data/partition.py`.
  - NaN → reduce learning rate to 0.005 in `configs/base_v3.yaml`.

### Gate 3: E2 CORE (after Phase 2)
- **Headline condition**: `adaptive_splitomc` integrated accuracy on Schedule A ≥ best fixed λ + 2 pp.
- **If fail (gap < 2 pp)**:
  - Inspect `results/e2_temporal/schedule_A/adaptive_splitomc.json` → look at `lamdas` trace.
  - If λ values are constant (no adaptation): controller frozen — check `controller/adaptive.py` sigmoid bounds.
  - If λ values swing but acc doesn't improve: increase `tau_H` to 0.6 in config; reduce `consensus_steps` to 0 (no consensus).

---

## 5. Resuming after interruption

The runner has built-in `--skip_existing` behavior: any completed run (saved JSON in `results/`) is skipped on re-invocation.

```bash
# To resume from where it stopped:
bash run_all.sh
# Or to re-run a specific experiment:
python scripts/run_e2_temporal.py --schedule A --device cuda:0
```

To force a re-run, delete the corresponding JSON:
```bash
rm results/e2_temporal/schedule_A/adaptive_splitomc.json
```

---

## 6. Output artifacts after completion

```
docs/results_v3/
├── interpretation.md            # Auto-narrative summary
├── paper_v3_filled.md           # Paper with real numbers
├── tables/
│   ├── e1_table.md              # Static Pareto table
│   ├── e2_table.md              # Temporal integrated table (CORE)
│   ├── e3_table.md              # Spatial per-cell table
│   ├── e4_table.md              # Mobility summary
│   └── e5_table.md              # Ablation table
└── figures/
    ├── e1_pareto.png            # ρ vs Acc curves
    ├── e2_temporal_A.png        # Acc vs round, schedule A
    ├── e2_temporal_B.png        # Schedule B
    ├── e2_lambda_trace.png      # λ_z(t) over time (proves adaptation)
    └── e3_cells.png             # Per-cell bar chart
```

The user can directly read `docs/results_v3/paper_v3_filled.md` and have a writable draft.

---

## 7. Common issues during long-running execution

### Issue: GPU OOM
- **Cause**: Too many clients training simultaneously holding model copies in GPU memory.
- **Fix**: Reduce `num_clients` to 30 in `configs/base_v3.yaml`, or use sequential client training (already the default — only one client trains at a time).

### Issue: 'cannot pickle' error during dataloader
- **Cause**: `num_workers > 0` with the closure-based dataset.
- **Fix**: Already set to `num_workers=0`. If still occurs, kill and restart.

### Issue: Loss explodes / NaN after some rounds
- **Cause**: lr too high for ND1 with small per-client batches.
- **Fix**: In `configs/base_v3.yaml`, lower `learning_rate: 0.005`. Re-run.

### Issue: Adaptive λ stuck at one value
- **Cause**: μ_H calibration off — entropy values are far from μ_H so sigmoid saturates.
- **Fix**: After E0 completes, look at log entropy values reported during eval. Then update `mu_H` in config (e.g., to actual mean entropy at round 50).

---

## 8. Compute budget contingencies

If wall-time pressure (deadline approaching):
- **Skip E4 (mobility)**: removes 4-5h. Justification: mobility is one contribution but not central.
- **Reduce E1 from 16 to 10 configs**: drop `fedavg`, `fedprox`, `splitfed`, `fedmes`. Saves ~6h.
- **Single seed**: already default; multi-seed is optional and not done by `run_all.sh`.

To launch reduced budget:
```bash
BUDGET=minimal bash run_all.sh  # Not yet supported; manual flag adjustments instead
```

---

## 9. After completion: what to send to user

1. **Send first**: `docs/results_v3/interpretation.md` — high-level findings.
2. **Send second**: `docs/results_v3/paper_v3_filled.md` — paper draft.
3. **Send third**: `docs/results_v3/tables/*.md` — all tables.
4. **Send last**: `docs/results_v3/figures/*.png` — all figures.

Suggest the user:
- Manually review the `interpretation.md` to verify the auto-narrative matches the actual numbers.
- Fill in author names, affiliations, references.
- Tighten language in the `paper_v3_filled.md` Introduction and Conclusion.

---

## 10. Quick sanity command (without running full pipeline)

```bash
# 1-2 minute sanity check that everything imports + tests pass
python -m pytest tests/test_all.py -v

# 5-minute sanity check that one config can run 2 rounds on GPU
python scripts/run_single.py --method splitomc --rounds 2 \
    --device cuda:0 --run_name sanity_check --output_dir /tmp/sanity
```

If both pass, the full pipeline is safe to launch.

---

## 11. Key code conventions

- **ALL hyperparameters live in `configs/base_v3.yaml`**. Don't hard-code.
- **All results saved as JSON** under `results/{experiment}/{run_name}.json`.
- **Logs** go to `logs/{experiment}.log` for debugging.
- **No use of `bash` for-loops** — use Python `scripts/run_*` for batching.
- **Seed determinism**: `partition_seed` controls topology/data; `model_seed` controls weights. Both fixed in config.

---

## 12. Reference paper coordinates

- **SplitGP**: Han et al., "Splitting and Placing Large Deep Learning Models for Effective Trainable Edge-Cloud Computing" (INFOCOM 2023). arxiv: 2212.08343.
- **SplitOMC**: Rizwan et al., "Efficient Split Learning with Overlapping Areas: Handling Distribution Shift in Multi-Cell Networks" (IEEE/ACM ToN 2026).
- **SplitOMC reference code**: github.com/atifrizwan91/SplitOMC

The v3 implementation matches reference verbatim except for:
- 150 global rounds × 3 local epochs (reference: 300 × 5) — compute budget
- Per-round eval (reference: only final 3 rounds) — for analysis richness
- Plus our additions: drift signals, consensus, mobility module.

---

End of CLAUDE.md.
