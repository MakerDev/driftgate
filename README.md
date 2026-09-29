# DriftGate

Research code for **DriftGate**, a label-free controller that sets the personalization weight λ of each
serving cluster in personalized split learning (SplitOMC setting). The controller compares the
client-server total variation distance (TV) of each cluster with its own recent values and with the
other clusters, and lowers λ when traffic moves away from the clients' main classes.
Target venue: IEEE Transactions on Mobile Computing.

This repository was moved from the original server on 2026-09-29.

- New server setup and the remaining work: [`MIGRATION/START_HERE.md`](MIGRATION/START_HERE.md) (Korean)
- Project context and rules: [`MIGRATION/PROJECT_CONTEXT.md`](MIGRATION/PROJECT_CONTEXT.md)
- Code map: [`MIGRATION/CODEBASE_MAP.md`](MIGRATION/CODEBASE_MAP.md)
- Datasets (CIFAR-10/100, SVHN, Tiny-ImageNet-200; all re-downloadable, none stored here): [`MIGRATION/DATASETS.md`](MIGRATION/DATASETS.md)

## Layout

```
configs/ data/ models/ train/ eval/ network/   SplitOMC training stack (ICTC 2026 version)
journal_expansion/                             DriftGate (TMC) code, runs, reports, artifacts
  scripts/run_v2.py                            single-run entry point
  artifacts/driftgate_tmc_final/               Round-5 final tables, figures, scripts
MIGRATION/                                     server migration docs and tools
docs/legacy_v3_ictc/                           documents of the earlier ICTC pipeline
```

## Quick check

```bash
pip install -r MIGRATION/requirements-lock.txt
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py journal_expansion/tests -q
```

Paths that depended on the original server are marked with `# [SERVER-PATH:...]` / `# [SERVER-GPU]`
and can be rewritten with `MIGRATION/tools/rewrite_server_paths.py`.
