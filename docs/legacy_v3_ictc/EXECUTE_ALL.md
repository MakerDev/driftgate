# EXECUTE_ALL.md — Step-by-step execution checklist

> Tells Claude Code (or any operator) exactly what to do, in what order, with what verifications.

---

## Step 0: Setup verification (5 minutes)

```bash
cd adaptive_splitomc_v3

# 0.1 Verify Python and dependencies
python --version  # Should be 3.10+
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 0.2 Verify GPUs visible
nvidia-smi

# 0.3 Run unit tests
python -m pytest tests/test_all.py -v
# Expected: 12 passed
```

**Decision**: If any unit test fails, STOP. Open `tests/test_all.py` and the failing module to diagnose.

---

## Step 1: Phase 0 — E0 smoke test (~3 hours)

```bash
python scripts/run_e0_smoke.py \
    --config configs/base_v3.yaml \
    --device cuda:0 \
    --rounds 50 \
    --local_epochs 3
```

**Expected output**:
- 4 runs complete: `splitomc_lam0.2`, `splitomc_lam0.6`, `splitomcplus_lam0.2_Lam0.5`, `adaptive_splitomc`
- Each writes JSON to `results/e0_smoke/`
- Summary written to `results/e0_smoke/e0_summary.json`

**Verification commands**:
```bash
cat results/e0_smoke/e0_summary.json | python -m json.tool
```

**Pass conditions** (auto-checked in run_all.sh):
1. All 4 runs have status `PASS_NO_NAN`
2. `splitomc_lam0.2`: `acc_total > 0.25` at final round
3. `splitomc_lam0.6`: `acc_total > 0.25` at final round
4. Loss strictly decreasing (compare R1 and R50)

**If E0 fails**:
- Inspect `results/e0_smoke/{run_name}.json` → look at `mean_loss` trace.
- Common fixes:
  - NaN: lower `learning_rate` in config from 0.01 to 0.005
  - All accuracy = 0.1 (random): partition has bug; re-run `python -m pytest tests/test_all.py::test_nd1_partition_cifar10 -v`
  - Loss not decreasing: model factory bug; re-run `python models/architectures.py` to verify

---

## Step 2: Phase 1 — E1 static Pareto (~10-13 hours)

```bash
# Split 16 configs across 2 GPUs
HALF1="fedavg fedprox splitfed fedmes splitgp_lam0.2 splitgp_lam0.5 splitgp_lam0.8 splitomc_lam0.0"
HALF2="splitomc_lam0.2 splitomc_lam0.4 splitomc_lam0.6 splitomc_lam0.8 splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 splitomcplus_lam0.6_Lam0.5 adaptive_splitomc"

python scripts/run_e1_static.py --device cuda:0 --methods $HALF1 \
    > logs/e1_gpu0.log 2>&1 &
python scripts/run_e1_static.py --device cuda:1 --methods $HALF2 \
    > logs/e1_gpu1.log 2>&1 &
wait
```

**Expected output**: 16 JSON files in `results/e1_static/`, each with `final_rho_sweep` containing 5 ρ values.

**Quick mid-run check**:
```bash
ls results/e1_static/ | wc -l  # Should grow as runs complete
tail -20 logs/e1_gpu0.log       # Should show "R150" eventually
```

---

## Step 3: Phase 2 — E2 temporal (CORE) (~7 hours)

This is the **headline experiment**. The paper's main claim rests on Schedule A results.

```bash
# Schedule A (the key one)
python scripts/run_e2_temporal.py --schedule A --device cuda:0 \
    --methods splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 \
    > logs/e2_A_gpu0.log 2>&1 &
python scripts/run_e2_temporal.py --schedule A --device cuda:1 \
    --methods splitomcplus_lam0.6_Lam0.5 adaptive_splitomc \
    > logs/e2_A_gpu1.log 2>&1 &
wait

# Schedule B (smaller)
python scripts/run_e2_temporal.py --schedule B --device cuda:0 \
    --methods splitomcplus_lam0.4_Lam0.5 \
    > logs/e2_B_gpu0.log 2>&1 &
python scripts/run_e2_temporal.py --schedule B --device cuda:1 \
    --methods adaptive_splitomc \
    > logs/e2_B_gpu1.log 2>&1 &
wait
```

**Mid-run verification** (after Schedule A completes):
```bash
python -c "
import json
runs = ['splitomcplus_lam0.2_Lam0.5', 'splitomcplus_lam0.4_Lam0.5',
        'splitomcplus_lam0.6_Lam0.5', 'adaptive_splitomc']
for r in runs:
    d = json.load(open(f'results/e2_temporal/schedule_A/{r}.json'))
    accs = [e['acc_total'] for e in d['eval']]
    print(f'{r}: mean={sum(accs)/len(accs):.3f}, final={accs[-1]:.3f}')
"
```

**Pass condition**: `adaptive_splitomc` mean > best fixed mean + 0.02 (2 pp).

---

## Step 4: Phase 3 — E3, E4, E5 (~12 hours)

```bash
# E3 spatial
python scripts/run_e3_spatial.py --device cuda:0 \
    --methods splitomcplus_lam0.2_Lam0.5 splitomcplus_lam0.4_Lam0.5 \
    > logs/e3_gpu0.log 2>&1 &
python scripts/run_e3_spatial.py --device cuda:1 \
    --methods splitomcplus_lam0.6_Lam0.5 adaptive_splitomc \
    > logs/e3_gpu1.log 2>&1 &
wait

# E4 mobility
python scripts/run_e4_mobility.py --device cuda:0 \
    --methods splitomcplus_lam0.2_Lam0.5_mob splitomcplus_lam0.6_Lam0.5_mob \
    > logs/e4_gpu0.log 2>&1 &
python scripts/run_e4_mobility.py --device cuda:1 \
    --methods adaptive_splitomc_mob \
    > logs/e4_gpu1.log 2>&1 &
wait

# E5 ablation
python scripts/run_e5_ablation.py --device cuda:0 --rounds 100 \
    --methods A_full_adaptive B_no_delta \
    > logs/e5_gpu0.log 2>&1 &
python scripts/run_e5_ablation.py --device cuda:1 --rounds 100 \
    --methods C_no_consensus D_static_mean \
    > logs/e5_gpu1.log 2>&1 &
wait
```

---

## Step 5: Phase 4 — Analysis + paper assembly (~5 minutes)

```bash
python scripts/analyze_all.py
python scripts/fill_paper.py  # NEW: fills paper template with real numbers
```

**Outputs**:
- `docs/results_v3/interpretation.md` — narrative
- `docs/results_v3/tables/*.md` — tables
- `docs/results_v3/figures/*.png` — figures
- `docs/results_v3/paper_v3_filled.md` — paper draft with real numbers

---

## Step 6: User-facing summary

After all the above, send the user:

1. **`docs/results_v3/interpretation.md`** — read first
2. **`docs/results_v3/paper_v3_filled.md`** — for paper drafting
3. Key tables (paste inline in chat):
   ```bash
   echo "## E2 (CORE)"
   cat docs/results_v3/tables/e2_table.md
   echo "## E1 Pareto"
   cat docs/results_v3/tables/e1_table.md
   echo "## E3 Spatial"
   cat docs/results_v3/tables/e3_table.md
   ```

---

## Failure recovery cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `pytest` fails | Python/torch version | `pip install -r requirements.txt` |
| GPU OOM | Too many concurrent models | reduce `num_clients` in config |
| Training loss = NaN | LR too high | `learning_rate: 0.005` in config |
| All accuracies ≈ 10% | Partition broken | Re-run partition test |
| E0 trends wrong (λ direction inverted) | Loss sign bug | Inspect `models/losses.py` |
| E2 adaptive < best fixed | Controller stuck | Inspect λ trace; adjust `tau_H` |
| E3 worst-cell unchanged | Consensus over-smoothing | Set `consensus_steps: 0` in config |
| Some E1 configs missing | Partial completion | Re-run script (skip_existing=True) |
| Analysis script crashes | Missing JSON files | Check `results/*/` for empties |

---

## Quick re-run commands (after fixes)

```bash
# Force re-run a specific experiment by deleting old JSON
rm results/e2_temporal/schedule_A/adaptive_splitomc.json
python scripts/run_e2_temporal.py --schedule A --device cuda:0 --methods adaptive_splitomc

# Re-run only analysis (no re-training)
python scripts/analyze_all.py
python scripts/fill_paper.py
```
