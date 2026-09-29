# DriftGate Weight Implementation Audit (λ, Λ, γ)

**Date**: 2026-08-03 · Audited against source + configs + run JSONs (no new training). Method
frozen. Evidence cited inline.

## Verdict
- **lambda: CORRECT** — per-cell adaptive model-mixing weight; direction and update location match the design.
- **Lambda: CORRECT (and genuinely ADAPTIVE)** — but its *independent* effect is NOT isolated (λ and Λ share one z-mapping; no fixed-Λ ablation). Paper must say "adaptive λ **and** Λ" and not attribute a separate Λ effect.
- **gamma: CORRECT** — 0.5, weights the client loss; fixed across all families.
- **Main reported results affected: NO.** Implementation matches design; no silent bug. Two clarifications required (below), both wording — no numbers change.

## Exact Training Loss
`models/losses.py:31` `return gamma * l_c + (1 - gamma) * l_s`, with
`l_c = CE(client_logits, y)`, `l_s = (1/|Z_k|) Σ_z CE(server_logits_z, y)`.

$$\mathcal L_k=\gamma\,\mathrm{CE}\!\left(f^{\text{client}}_k(x),y\right)+(1-\gamma)\,\frac{1}{|Z_k|}\sum_{z\in Z_k}\mathrm{CE}\!\left(f^{\text{server}}_z(h_k(x)),y\right),\qquad \gamma=0.5$$

`y` are **Main-class local training labels only**. OOP/OOR classes and probe samples never
enter the loss (probe is a separate `no_grad` forward; training loaders hold only Main-class
data — `test_train_data_only_main_classes`).

## Exact Client Mixing Update (λ)
`train/trainer.py:271` `new_w = mix_state_dicts(local_w, avg_client, lam)`, and
`mix_state_dicts(a,b,w)=w·a+(1−w)·b` (`trainer.py:34-39`):

$$\theta^{\text{client},t+1}_{k}=\lambda_z^{t}\,\theta^{\text{client,local},t+1}_{k}+(1-\lambda_z^{t})\,\bar\theta^{\text{client},t+1}_{z}$$

where $\bar\theta^{\text{client}}_z$ is the cell-average of member clients' post-training
client blocks. The server block is **replaced** by the cell average (no λ):
$\theta^{\text{server},t+1}_{k,z}=\bar\theta^{\text{server},t+1}_z$ (`trainer.py:276`).
λ mixes the **entire client block** (4 conv layers + auxiliary client-exit head + BN running
stats). Optimizer state is not mixed (fresh SGD, momentum 0, each round).

## Exact Inter-Cluster Sharing Update (Λ)
`train/trainer.py:201-204`, run for `splitomcplus`/`adaptive_splitomc` (`trainer.py:301-302`):

$$\bar\theta_z\leftarrow \Lambda_z^{t}\,\bar\theta_z+(1-\Lambda_z^{t})\,\bar\theta_{\text{global}},\qquad \bar\theta_{\text{global}}=\frac{1}{L}\sum_{z'}\bar\theta_{z'}$$

applied to **both** the client-block and server-block cell aggregates. **Λ is adaptive**:
`self_calibrating.py:175,179,187` emit `big_lamdas[es]` every round from the *same* z as λ
(bounds $[\Lambda_{\min},\Lambda_{\max}]=[0.4,0.7]$, same abs-cap). Verified in run traces:
DriftGate Λ varies (e.g. Schedule A range [0.41,0.64], var 0.0055), whereas fixed-grid runs
have Λ≡0.5 (var 0). Applied at cell aggregation, **before** the λ client update.

## Meaning of Larger/Smaller Values (from the tensor update, not names)
- **λ**: mapping `lam = lam_max − (lam_max−lam_min)·σ((z−z0)/τ)` (`self_calibrating.py:174`).
  High drift → high z → **low λ** → `new≈cell-average` → **generalize**. No drift → **high λ**
  → `new≈local` → **personalize**. (Confirmed: `test_lambda_one_preserves_local`,
  `test_lambda_zero_gives_cluster_avg`.) **Larger λ ⇒ more local personalization.**
- **Λ**: `Λ·cell + (1−Λ)·global`. **Larger Λ ⇒ more cell-local; smaller Λ ⇒ more global
  sharing.** (`test_Lambda_zero_gives_global`, `test_Lambda_one_keeps_cell_local`.) With the
  same mapping, high drift → low Λ → more global sharing.
- **γ**: weights the **client** loss. Larger γ ⇒ client exit dominates training; γ=0.5 ⇒ equal.
  (`test_gamma_convention_client_weight`.)

## Final Frozen Parameters

| Parameter | Final value | Source | Scope | Retuned per dataset/schedule |
|---|---:|---|---|---:|
| Probe batch size | 64 | `--probe_n 64` / manifest | all | no |
| Probe frequency | every round | manifest | all | no |
| Warm-up | 15 | manifest | all | no |
| Burn-in | 10 | manifest | all | no |
| Temporal normalizer | GuardedRobustNormalizer (median/1.4826·MAD, EWMA β=0.05 only when z<guard) | `normalizers.py` | all | no |
| EWMA smoothing (z, signal) | α=0.3 | `self_calibrating.py:EWMA_ALPHA` | all | no |
| Asymmetric guard | z_guard_hi = 0.5 | manifest | all | no |
| MAD floor | max(1e-3, 0.10·max(\|μ\|,0.05)) | `normalizers.py` | all | no |
| z clipping | [−2, 6] | manifest | all | no |
| Mapping center / slope | z0=1.5, τ_z=0.75, σ-form | manifest | all | no |
| λ_min, λ_max | 0.15, 0.7 | config `adaptive` | all | no (inherited, not re-tuned) |
| Absolute-level cap | λ_abs=λ_max−(λ_max−λ_min)·clip(EWMA(consensus(TV)),0,1) | `self_calibrating.py:177-180` | all DriftGate | no |
| Consensus steps | 1 (ES line graph) | manifest | all | no |
| **Λ policy** | **adaptive**, same z-mapping+abs-cap, bounds [0.4,0.7] | `self_calibrating.py:175,179` | all DriftGate | no |
| **γ** | **0.5**, L=γ·client+(1−γ)·server | `losses.py`, config | all | no |

All rows: **frozen before holdout** (bounds/γ inherited from ICTC v4 and not re-tuned; the
dimensionless controller constants were fixed at Gate 3 before the SVHN holdout). No
dead/default value overrides a frozen setting (fixed-mode runs set λ via `--lambda_val` and
leave Λ at `big_lambda_default=0.5`; DriftGate sets both adaptively — verified in traces).

## Run-Family Consistency

| Run family | adaptive signal | λ policy | Λ policy | γ | exceptions |
|---|---|---|---|---:|---|
| CIFAR-10 main temporal | TV divergence | adaptive | **adaptive** | 0.5 | — |
| CIFAR-10 spatial | TV | adaptive | adaptive | 0.5 | — |
| full fixed grid | none | fixed (0.0–0.8) | **fixed 0.5** | 0.5 | Λ held at 0.5 (see clarification 1) |
| CIFAR-100 / Tiny-ImageNet | TV | adaptive | adaptive | 0.5 | — |
| frozen SVHN | TV | adaptive | adaptive | 0.5 | — |
| role-separation ablation | TV | adaptive | adaptive | 0.5 | aggregation role structure perturbed (§7) |
| ResNet early/middle/late | TV | adaptive | adaptive | 0.5 | 16 clients / 100R pilot |
| composition-coupled mobility | TV | adaptive | adaptive | 0.5 | — |
| causal baselines | proxy/bandit | per-baseline | fixed 0.5 | 0.5 | λ set by the baseline, not the divergence |
| network stress | TV | adaptive | adaptive | 0.5 | signal delayed/lossy (input only) |
| covariate corruption | TV | adaptive | adaptive | 0.5 | p(x) perturbed |

γ=0.5 and the (λ,Λ) update location are identical across every family; only the fixed-grid
and causal-baseline families use non-adaptive/externally-set λ (and fixed Λ=0.5).

### Algorithm steps (per round t)
1. **Probe**: each client computes the TV divergence between its client-exit and mean
   server-exit softmax on 64 unlabeled ρ-mixed probe samples (end-of-round-(t−1) models).
2. **Signal→control**: per-ES mean TV → temporal guarded-z and cross-cell spatial-z →
   EWMA → 1-step consensus → per-cell (λ_z, Λ_z) via the shared sigmoid mapping + abs-cap
   (warm-up rounds output the neutral midpoints).
3. **Local training**: each client trains its client + server block(s) on Main-class data
   with the multi-exit loss (γ=0.5).
4. **Cell aggregation**: each ES uniform-averages member clients' client and server blocks.
5. **Inter-cluster (Λ)**: each cell mixes its aggregate with the global all-cell average,
   weight Λ_z (both client and server aggregates).
6. **Client update (λ)**: client block ← λ_z·local + (1−λ_z)·cell-average; server block ←
   cell-average. Multi-cluster client: λ_eff = mean of its cells' λ_z, and the cluster model
   is the uniform average of those cells' client aggregates.

## Required Corrections to the Current Design Document
1. **Full-grid comparison is adaptive(λ,Λ) vs fixed-λ-at-Λ=0.5.** The fixed grid varies λ
   only and holds Λ≡0.5; DriftGate adapts BOTH. State the comparison as "adaptive λ and Λ
   vs the best fixed λ (Λ=0.5)", and note the λ/Λ effects are not separately isolated (no
   fixed-Λ DriftGate ablation exists). Do NOT claim an independent Λ contribution.
2. **Write "adaptive λ and Λ" everywhere** (the reports that emphasized only λ are
   incomplete): Λ is genuinely adaptive with bounds [0.4,0.7] and the same z-mapping; add
   the exact Λ equation above. Also: γ is the *client* weight (L=γ·client+(1−γ)·server),
   fixed 0.5 — write it explicitly so it is not confused with the EXP3 baseline's
   exploration parameter (also named `gamma`, value 0.1) or the controller EWMA/mobility
   `alpha` (0.3 vs Gauss-Markov 0.9); use distinct symbols in the paper.

Neither correction changes any number — both are notation/wording. No result is retracted;
"beats the full fixed grid on temporal drift" stands, now precisely stated as adaptive(λ,Λ)
vs best fixed-λ (Λ=0.5).

## Evidence
- Loss: `models/losses.py:8-31`. Client update: `train/trainer.py:257-276`. Inter-cluster:
  `train/trainer.py:191-204`, invoked `trainer.py:301-302`. Controller (λ,Λ emit + mapping +
  abs-cap): `journal_expansion/src/controllers/self_calibrating.py:167-188`.
- Config: `configs/base_v3.yaml:23` (gamma 0.5), `:49-52` (lam/Lam bounds), manifest
  `provenance/dv2_frozen_manifest.json` (controller block).
- Representative runs (traces show adaptive λ AND Λ for DriftGate; fixed both for grid):
  `dvsig_tv_A_s0`, `mob_med_dv2_s0`, `ho_dv2_gsig_s0`, `d4_dual_gsig_s0`, `fx40_A_s0` (grid),
  `all_runs.csv`.
- Tests (all pass, 68 total): `test_lambda_one_preserves_local`,
  `test_lambda_zero_gives_cluster_avg`, `test_Lambda_zero_gives_global`,
  `test_Lambda_one_keeps_cell_local`, `test_gamma_convention_client_weight`,
  `test_train_data_only_main_classes`, `test_role_ablation_*`.
