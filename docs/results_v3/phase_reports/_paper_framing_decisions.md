# Paper Framing Decisions for v3 (locked 2026-05-30 ~02:00 KST)

> Locked by user after E2 retry-1 FAIL (−1.57pp) and incoming-traffic anchor STEP 2 partial.
> All `phase_reports/*.md`, the final `interpretation.md`, and `paper_v3_filled.md` MUST follow these.

---

## 1. E2 is NOT a limitation footnote — it is a two-signal ablation contribution

Structure the E2 section as a controlled ablation of the controller's drift signal, not as a "thing that didn't work":

### Signal variant A: train-anchor entropy (attempt 0/1)
- **Result**: completely fails to track ρ-drift. λ stays in a narrow band (swing 0.081 across 150 rounds) regardless of ρ schedule.
- **Why**: anchor sampled from client's own training distribution. Probe inputs never include OOP/OOR classes, so entropy reflects training convergence on personal task, not deployed traffic mix.
- **Outcome**: integrated 0.6345, gap −1.94 pp vs best fixed.

### Signal variant B: incoming-traffic entropy (final fix)
- **Result**: catches transient drift events (R65 ρ-transition shows H bump +0.054, λ momentarily dips), but **training convergence erases the signal within ~10 rounds** (by R75-80 H back below pre-transition level).
- **Why**: even at ρ=0.8, 20% of probe is still client's main classes (model confident on them) plus the OOP classes share semantic structure with main via cell scope, so model rapidly learns them too. Entropy is dominated by overall confidence, not by *which* classes are present.
- **Outcome**: marginal integrated improvement, still below 2-pp gate.

### Contribution
> "We quantitatively demonstrate a structural limitation of entropy-as-drift-signal in federated split learning: even when the controller measures entropy on the deployment-time traffic distribution (eliminating the obvious train-vs-deploy mismatch), the signal is dominated by overall model confidence rather than by class-mix changes. This holds because client models specialize fast enough that out-of-personal classes within the cell scope become low-entropy under continued training. Two signal variants were ablated; both fail in distinct ways, motivating a convergence-invariant drift signal as the necessary next step."

### Future work direction (explicit)
- KL divergence of current logits against a **frozen reference** model snapshot (drift = distance from a fixed point, not from "high entropy").
- Anchor restricted to **OOR (out-of-region) classes only** — these are never trained on by the client model, so entropy on them stays high regardless of convergence.
- External signal: per-round measurement of probe-set class distribution shift directly (Wasserstein on label histograms), bypassing model-based entropy entirely.

---

## 2. E2 ρ=0 segment WIN — explicit, not buried

The retry-1 result (despite overall FAIL) showed adaptive wins at ρ=0:

| Method | ρ=0 mean |
|---|---:|
| splitomcplus λ=0.2 | 0.6350 |
| splitomcplus λ=0.4 | 0.6687 |
| splitomcplus λ=0.6 | 0.7019 |
| **adaptive (retry-1)** | **0.7061** ← highest of all 4 |

Paper text:
> "Adaptive correctly selects the optimal operating point when drift is absent or mild — exceeding every fixed-λ configuration in the no-drift segment (acc@ρ=0 0.706 vs best fixed 0.702). Its limitation is confined to tracking *rapid* explicit drift, where the entropy signal limits identified above apply."

Implication: the controller IS finding meaningful per-state operating points — the failure is signal-source, not controller logic.

---

## 3. E3 (spatial) is the potential MAIN RESULT — check per-cell λ_z carefully

**Key insight to verify** when adaptive runs on E3:
- E3 has **per-cell fixed ρ_z** (steady-state) — equal_spread mode gives ρ_z ∈ {0.0, 0.2, 0.4, 0.6, 0.8} across 5 cells.
- This sidesteps E2's convergence-vs-drift confound: each cell trains under its own static ρ, the controller sees stable H per cell.
- **Hypothesis**: cells with low ρ_z (e.g., 0.0) should converge on high λ_z (e.g., 0.6+), cells with high ρ_z (e.g., 0.8) should converge on low λ_z (e.g., 0.2).

**What to check in E3 adaptive output:**
1. Plot λ_z trajectory per cell over training. By R150, do they stratify?
2. Compute per-cell final λ vs per-cell ρ. Is there monotone (negative) relationship?
3. Compare per-cell acc_total to the best fixed for that cell's ρ. Does adaptive match or beat the per-cell oracle?

**If YES on points 1-3**: this is the headline. Paper main result becomes:
> "Adaptive control finds per-cell optimal operating points under spatially heterogeneous ρ_z, where each cell's controller correctly identifies its local ρ regime via entropy without coordination. (Fig: λ_z vs ρ_z scatter shows monotone fit; Table: adaptive matches per-cell oracle within 1 pp.)"

**If NO**: report honestly. Same signal limit applies in spatial form. Fall back to E1 acc_oor +37% as headline.

---

## 4. Paper-level ordering of contributions

1. **(E1)** Adaptive-SplitOMC achieves Pareto frontier on static no-drift setup, with **+37% cross-region transfer (acc_oor 0.371 vs 0.271 best fixed)** while matching best acc_total within 0.4 pp. Smallest cell_gap of all evaluated configs.
2. **(E3 — pending verification)** Under spatial heterogeneity, adaptive finds per-cell optimal λ_z without coordination. (Conditional on E3 verifying per-cell stratification.)
3. **(E2 — ablation contribution)** We isolate a structural limitation of entropy-based drift signals via a two-variant ablation. Future work targets convergence-invariant drift sensing.
4. **(E0, E4, E5)** Validation, mobility robustness, component ablation.

Headline depends on E3 outcome:
- **E3 stratification confirmed** → main message "per-cell adaptive control with provable signal limits under temporal drift"
- **E3 also fails** → main message "Pareto-optimal cross-region transfer with quantitative drift-signal analysis"

---

## 5. What NOT to write

- ❌ "Adaptive-SplitOMC tracks temporal drift" (E2 data refutes for rapid drift)
- ❌ "Future work: tune hyperparameters" (we already tried; not the issue)
- ❌ Burying E2 in appendix — it's a contribution, frame it that way
- ❌ Hiding the −1.57 pp gap; report it directly with the ablation that explains it
