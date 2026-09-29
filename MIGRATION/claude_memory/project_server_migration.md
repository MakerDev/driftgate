---
name: project-server-migration
description: "2026-09-29 the DriftGate repo moved from the old server (ubuntu20, GPU 0 only) to GitHub MakerDev/driftgate for a new multi-GPU server; MIGRATION/ holds the procedure"
metadata:
  type: project
---

On 2026-09-29 the whole `adaptive_splitomc_tmc` folder (code, all run JSON, provenance, reports,
artifacts; no datasets) was pushed to github.com/MakerDev/driftgate because the old server
(`ubuntu20`, 2× RTX 3090 Ti) allowed only GPU 0 for this project.

- Server-dependent lines carry `# [SERVER-PATH:REPO_ROOT|DATA_ROOT|GIT_ROOT|EXTERNAL]` / `# [SERVER-GPU]`
  markers; `MIGRATION/tools/rewrite_server_paths.py` rewrites them, `verify_code_identity.py` proves
  the training code equals the Round-5 code (sha256 list in driftgate_tmc_final/precheck/code_identity.txt).
- Round 5 was 39/64 runs done at upload; the old server kept running the rest. Who finishes the
  remaining runs (old server + `sync_results_from_old_server.sh`, new server, or split) is the user's call.
- `r5_worker.py` / `supervisor.sh` were parametrized (R5_GPU, R5_GPUS, per-GPU worker counts) in the
  pushed copy only; the old live checkout still runs the GPU-0-fixed versions.
- Old-server paths inside this memory (`/disk2/Yujin/adaptive_splitomc_tmc/...`) map to the new repo root.

**Why:** the user wants more GPUs for the remaining Round-5 work and the final report.
**How to apply:** on the new server start from `MIGRATION/START_HERE.md`; log decisions in
`MIGRATION/MIGRATION_LOG.md`. Related: [[project-tmc-expansion-state]], [[feedback-long-running-jobs]].
