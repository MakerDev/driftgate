# §2 — Final Consistency Audit

**Date**: 2026-08-02 · Reconciles report contradictions and artifact counts. These numbers
are for INTERNAL reliability only; the paper/closure report does not cite raw counts.

## 2.1 Experiment-completion contradiction ("all complete" vs "late split partial")

**Resolution: Case B (complete the late split), in progress.** DV-2 late split was complete
(3 seeds); the fixed-λ and entropy late arms were not (fx20 1/3, fx40 0/3, dve 2/3 = 6
missing). Because a clean DV-vs-fixed comparison at the late split needs the fixed
baselines, the 6 runs were queued (ResNet late ≈ 13.6 h each, dedicated low-concurrency
worker) rather than dropping the split. Until they finish, the accurate status line is:

> "All scheduled core experiments complete; the ResNet late-split fixed/entropy baselines
> and the role-separation ablation are in final completion."

The header "ALL experiments complete" in `FINAL_REPORT_2026-08-02` is corrected to the above
in the closure report. (If a resource cap had made Case B unreasonable, Case C — drop late
from main, keep early+middle — was the fallback; early+middle already establish the trend.)

## 2.2 Artifact count reconciliation (four distinct counting bases)

| basis | count | definition |
|---|---:|---|
| **physical run JSONs** | 798 | completed result files under `runs/**` (excl. manifest/summary); grows as closure runs land |
| **analysis rows** (`all_runs.csv`) | 797 | physical runs that carry an eval trace (1 smoke/partial without eval excluded) |
| **provenance events** | 891 | every RunRecord ever opened, incl. 78 with status "running" = jobs killed/re-queued during the OOM-isolation and session-teardown episodes that never reached `finish()` |
| **logical runs** | ≈ physical | distinct (config, seed) experiments intended; re-queued duplicates collapse to one |
| replay/control rows | 69 | phaseB_decomp trajectory interventions (analysis-only, future-info) — not learning runs of the method |
| full grid | 185 | fixed-λ backbone points |
| unit tests | 66 | 54 journal + 12 legacy (earlier "56" was stale; current after role-ablation tests) |

**Why provenance (891) > physical (798):** provenance logs each launch attempt; the 78
"running"-status records are killed/re-queued attempts (OOM under shared-GPU contention;
session teardown) whose successful re-run IS counted in the 798. No result was lost or
double-counted in `all_runs.csv` (keyed by run_name; re-runs overwrite the same result file).

**Corrections applied**: report headers that cited "646 runs / 880 provenance / 56 tests"
were mid-run snapshots; the reconciled current figures are 798 / 891 / 66. The paper and
closure report cite NONE of these (per §1.2); they live here for internal reliability.

## 2.3 Failure accounting
Non-ResNet training failures: **0**. ResNet OOMs under shared-GPU contention were isolated
to a self-healing re-queue worker (no data loss); those show as extra provenance "running"
events, not as lost runs.
