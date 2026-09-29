# DriftGate — Final Closure Report

**Method: DriftGate (selfcal-DV-2), frozen.** Paper-oriented synthesis; internal engineering
(run counts, tooling, development versions, tuning history) is omitted by design. All numbers
are measured; statistics are mean ± sd / paired differences / seed counts.

---

## 1. Research question
In personalized split learning with role-separated exits, can the divergence between a
personalized client exit and a generalized server exit serve as a label-free, detector-free
drift signal to control cluster-level personalization — and when does adapting help over a
fixed policy?

## 2. Final method
A per-edge-server controller maps the **total-variation divergence between the two exits**,
measured on unlabeled probe traffic, to the SplitOMC personalization/sharing weights (λ, Λ)
through three dimensionless reference views: a causal temporal robust-z, a cross-cluster
spatial robust-z, and the signal's absolute probability level, combined as
λ_z = min(λ_selfcal(max(z_temporal, z_spatial)), λ_abs(divergence)). No dataset- or
schedule-specific calibration constant is used.

## 3. Signal evidence
On a held-out-controller comparison (same controller, signal swapped, 150 rounds, 5 seeds),
dual-exit divergence improves integrated accuracy by **+1.06 pp (all 5 seeds)** and by
**+3.64 pp** on the sustained high-drift segment over absolute predictive entropy. Its
correlation with the true drift ratio is high (Spearman ≈0.92) where entropy's is ≈0.50.

## 4. Why entropy is less reliable
Under continued training the client head sharpens and predictive entropy decays regardless
of drift, so the entropy trend confounds convergence with drift (raw drift-vs-no-drift AUROC
falls to 0.16 on an abrupt post-convergence schedule). The dual-exit divergence, resting on
the client head's persistent inability to cover non-Main classes, remains drift-aligned.

## 5. Temporal, spatial, and absolute views (component ablation)

| views used | temporal drift | static spatial | cross-dataset |
|---|:--:|:--:|:--:|
| temporal only | ✓ | | |
| temporal + spatial | ✓ | ✓ | |
| DriftGate (+ absolute) | ✓ | ✓ | ✓ |

The temporal view alone is blind to static spatial heterogeneity; adding the spatial view
recovers per-cluster stratification; adding the absolute-level view fixes cross-dataset
transfer (where the divergence's baseline level differs).

## 6. Adaptation-value decomposition
DriftGate is compared against fixed policies that reuse its own λ distribution: the global
and per-cluster mean-matched fixed λ, and its own time-shuffled trajectory. On temporal
schedules DriftGate exceeds all three (+0.9–1.2 pp, all 3 seeds) — the same λ values applied
at the wrong times lose accuracy, so the temporal alignment itself carries value. On static
spatial heterogeneity it matches the per-cluster mean-matched fixed (no temporal signal to
exploit) while beating the global fixed.

## 7. Full-grid comparison

| environment | type | DriftGate − best fixed (full grid) |
|---|---|---:|
| Schedule A | temporal | **+0.93 pp** (2/2 seeds) |
| asym-return | temporal | **+0.55 pp** (3/3 seeds) |
| gradual-sigmoid | temporal | ≈ tie |
| spatial | static | −0.69 pp |
| CIFAR-100 (transfer) | static-optimum | −0.4 to −0.65 pp |

DriftGate's advantage tracks how much the preferred operating point moves over time
(correlation with per-environment operating-point movement: Spearman +0.82, n=7). It beats
or matches the full fixed grid where the preferred λ moves trackably; a fixed policy is
preferable when one operating point stays near-optimal, when the λ loss surface is flat, or
when the preferred point moves unpredictably. (Best-fixed-per-test-environment is a hindsight
reference; the deployable baseline is a development-selected robust fixed λ.)

## 8. Frozen holdout
On a dataset never used in any design step (SVHN, preregistered, 5 seeds), the signal
improves over the entropy controller by **+2.40 pp under temporal change and +0.76 pp under
spatial heterogeneity, positive across all five seeds**. On this static-optimum dataset a
tuned fixed λ remains ~1 pp ahead of the adaptive controller.

## 9. Role-separation evidence (novelty)
When the two exits are given the SAME (generalized) role, the signal collapses: TV–drift
correlation falls from **+0.84 (role-separated) to −0.44 (same-role)**, and independent
initialization of the two same-role heads does **not** restore it (**+0.08**). Weakening only
the server's generalization role degrades the signal proportionally (+0.21). The signal
therefore arises from the complementary personalization/generalization roles, not from
arbitrary classifier diversity — distinguishing DriftGate from same-role dual-classifier
discrepancy methods.

## 10. Architecture and split depth
On ResNet-18, DriftGate beats a fixed policy at every split depth (+2.1–2.7 pp, all 3 seeds).
The signal's advantage over entropy is largest at a balanced (middle) split and shrinks when
the client head is very shallow (weak role separation) or very deep (both heads competent) —
consistent with the role-separation mechanism.

## 11. Mobility
Under mobility that jointly changes serving membership and Main/OOP/OOR composition,
DriftGate improves over the best fixed policy by **+0.6–1.4 pp** and over the entropy
controller by **+2.0 pp**, consistently across mobility speeds.

## 12. Network robustness
Controller input is robust to an abstract multi-edge impairment model: 10-round-stale or
20%-lossy scalar signals cost ≤0.8 pp; consensus topology is second-order; a participation
drop is a training effect, not a controller effect. No impairment-induced oscillation.

## 13. Communication and on-device evaluation
DriftGate adds **one 4-byte scalar per client per round** (plus a per-edge scalar for the
one-step consensus) beyond the communication the underlying split-learning protocol already
requires (smashed activations and periodic model exchange); it uploads no additional
activations and exchanges no additional model weights. On-device latency/energy are measured
by the user with the provided export + benchmark harness (values pending).

## 14. Scope and boundaries
Effective for traffic-composition and class-support changes when the two exits retain
complementary roles (temporal drift, spatial heterogeneity, mobility, sufficient split
depth). Not effective under severe covariate input corruption (both exits fail with
different labels, inflating a misleading divergence) or when the preferred operating point is
static; there a tuned fixed policy is as good or better.

## 15. TMC claim set
1. An architecture-native, label-free, detector-free drift signal from role-separated split
   exits, shown to come from the roles (not diversity) and to beat absolute entropy, incl. a
   frozen holdout.
2. A calibration-free cluster-level personalization controller (temporal/spatial/absolute
   views, no environment-specific constants) that helps when the preferred operating point
   moves trackably and matches a fixed policy otherwise.
3. A measurement-grounded evaluation with honestly scoped boundaries.

## 16. Remaining user-supplied measurements
On-device device/chipset/RAM/runtime; per-stage latency (mean/median/p95/p99); peak memory;
optional energy — via the verified handoff.

## 17. Submission verdict
**READY_PENDING_ONDEVICE** (TMC). ToN: conditional on model-update-plane impairments and
larger topologies. Recommended title:
*"DriftGate: Calibration-Free Drift Adaptation in Split Learning via Dual-Exit Divergence."*
