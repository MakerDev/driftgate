# Adaptive-SplitOMC v3 — README

Per-cell drift-aware hyperparameter control for SplitOMC. ICTC 2026 target.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/test_all.py -v   # Should show 12 passed
bash run_all.sh                          # Full pipeline (~32h on 2 GPUs)
```

## What this is

This is a complete experimental framework that:
1. Implements SplitOMC / SplitOMC+ matching the reference code at github.com/atifrizwan91/SplitOMC
2. Extends it with **Adaptive-SplitOMC**: per-cell drift-aware (λ_z, Λ_z) via entropy + anchor signals + consensus
3. Runs 5 experiments (E1-E5) testing static, temporal, spatial, mobility, and ablation scenarios
4. Auto-generates paper-ready tables, figures, and a filled-in paper draft

## Reading order

1. **CLAUDE.md** — autonomous execution instructions (for Claude Code or operators)
2. **EXECUTE_ALL.md** — step-by-step execution checklist
3. **configs/base_v3.yaml** — all hyperparameters
4. **paper_draft_v3.md** (in companion `deliverables_v3/`) — paper structure

## Output after running

```
docs/results_v3/
├── paper_v3_filled.md          # Paper draft with REAL numbers — read first
├── interpretation.md           # Auto narrative summary
├── tables/                     # Paper-ready Markdown tables
│   ├── e1_table.md             # Static Pareto
│   ├── e2_table.md             # Temporal CORE
│   ├── e3_table.md             # Spatial fairness
│   ├── e4_table.md             # Mobility
│   └── e5_table.md             # Ablation
└── figures/                    # Paper-ready PNG figures
    ├── e1_pareto.png
    ├── e2_temporal_A.png
    ├── e2_temporal_B.png
    └── e3_cells.png
```

## Architecture overview

```
                                    ┌──────────────────────┐
                                    │  configs/base_v3.yaml│
                                    └──────────┬───────────┘
                                               │
   ┌────────────┐    ┌───────────┐    ┌───────▼─────────┐
   │ data/      │    │ models/   │    │ scripts/        │
   │ partition.py│   │ archi-    │    │ run_single.py   │ ◄── core runner
   │            │    │ tectures  │    │                 │
   │ ND1 + ES   │    │ .py       │    │ scripts/        │
   │ topology   │    │           │    │ run_e0..e5.py   │ ◄── experiment scripts
   └─────┬──────┘    └─────┬─────┘    │                 │
         │                 │          │ scripts/        │
         ▼                 ▼          │ analyze_all.py  │ ◄── tables/figures
   ┌─────────────────────────────┐    │                 │
   │ train/trainer.py            │    │ scripts/        │
   │ - SplitOMCClient            │◄───┤ fill_paper.py   │ ◄── paper draft
   │ - EdgeServer                │    └─────────────────┘
   │ - run_one_global_round      │             ▲
   └────────────┬────────────────┘             │
                │                              │
        ┌───────▼──────────┐                   │
        │ eval/evaluator.py│                   │
        │ entropy routing  │                   │
        └───────┬──────────┘                   │
                │                              │
        ┌───────▼──────────┐                   │
        │ controller/      │                   │
        │ adaptive.py      │                   │
        │ H + Δ + consensus│                   │
        │ → (λ_z, Λ_z)     │                   │
        └─────────┬────────┘                   │
                  │                            │
                  └────────────────────────────┘

network/mobility.py: Gauss-Markov mobility for E4 only.
```

## Key implementation facts (vs reference)

| Component | Reference | v3 | Reason for diff |
|---|---|---|---|
| Loss | γ=0.5 multi-exit on D_k | Same | exact match |
| Per-client server models | One per associated ES | Same | exact match |
| ES aggregation | Uniform average | Same | exact match (paper claims κ_n/κ_o weighting but code doesn't) |
| λ aggregation rule | eq. (9) on client-side | Same | exact match |
| SplitOMC+ Λ | Global all-ES average | Same | exact match (paper says "neighboring" but code is global) |
| Test set construction | Per-client via ρ filter | Same | exact match |
| Entropy routing | E_th sample-wise | Same | exact match |
| Training schedule | 300R × 5e × batch 32 | **150R × 3e × batch 32** | compute budget |
| Evaluation frequency | Final 3 rounds only | **Every 10 rounds** | richer analysis |
| Drift signals (H, Δ) | None | **Added** | our contribution |
| Cell consensus | None | **Added** | our contribution |
| Mobility | None | **Added** | our contribution |

## Re-running individual experiments

```bash
# Just E2 (CORE)
python scripts/run_e2_temporal.py --schedule A --device cuda:0

# Just analysis (no re-training)
python scripts/analyze_all.py
python scripts/fill_paper.py

# Single configuration sanity
python scripts/run_single.py --method adaptive_splitomc --rounds 10 --device cuda:0
```

## Troubleshooting

See `CLAUDE.md` § 7 for the troubleshooting table.

## Credits

- SplitOMC formulation: Rizwan et al., IEEE/ACM ToN 2026
- SplitGP foundation: Han et al., INFOCOM 2023
- Reference code: github.com/atifrizwan91/SplitOMC
- Adaptive control + mobility scenario: this work

Built for ICTC 2026 by 유진 (Yujin Shin), with autonomous-execution support.
