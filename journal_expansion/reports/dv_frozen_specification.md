# selfcal-DV — Frozen Specification

**Frozen**: 2026-07-18 · manifest: `provenance/dv_frozen_manifest.json` (source SHA-256
prefixes: self_calibrating 16ccf6c4…, normalizers 47161d58…, signals 29a0992c…,
runner fcd0e985…). After this point no DV equation or constant changes in response to
holdout results; a Gate-2 signal change (δ→TV) would re-freeze as **DV-2** with a new
manifest BEFORE any holdout run.

## Controller (per edge server z, per round t)

```
inputs : s_z(t) = per-ES mean of the dual-exit signal on n=64 unlabeled probe
         samples/client drawn from the current deployment traffic (label-free, rho-free)

temporal view : z_t = GuardedRobustNormalizer(s_z)
                burn-in 10 rounds (discarded), warm-up 15 (median/1.4826*MAD baseline),
                baseline EWMA beta=0.05 adapts ONLY while z<0.5 (asymmetric guard),
                sigma >= max(1e-3, 0.10*|mu|), z clipped to [-2, 6]
spatial view  : z_s = (s_z - median_k s_k) / max(1.4826*MAD_k, 0.10*max(|median|,0.05)),
                current round only, clipped [-2, 6]
combination   : z~ = max( EWMA_0.3(z_t), EWMA_0.3(z_s) ) ; 1-step neighbor consensus
selfcal map   : lam_sc = lam_max - (lam_max-lam_min) * sigmoid((z~ - 1.5)/0.75)
absolute view : d^ = EWMA_0.3( consensus(s_z) ) clipped to [0,1]
                lam_abs = lam_max - (lam_max-lam_min) * d^
output        : lam_z = min(lam_sc, lam_abs)     (Lambda_z analogous)
warm-up       : lam = (lam_min+lam_max)/2 for the first 25 rounds
bounds        : lam in [0.15, 0.7], Lambda in [0.4, 0.7]  (inherited, un-retuned; the
                D3 bounds ablation quantifies their effect)
```

All constants are dimensionless and shared across every dataset/schedule/architecture.
No dataset- or schedule-specific calibration constants are used (claim C-2 wording).

## Design provenance (what may NOT be called holdout)

- Datasets used to design/diagnose DV: CIFAR-10 (all controller revisions),
  CIFAR-100 + Tiny-ImageNet (absolute-view diagnosis) → *development transfer datasets*.
- Schedules used: A, abrupt, recurring, burst (Gate B), gradual_sigmoid, asym_return,
  piecewise_random (Gate D), spatial equal_spread (Gate C/D3).
- Eligible frozen holdouts: SVHN, EMNIST/FEMNIST, PACS, GTSRB, STL-10 (none ever run
  through any controller).

## Invocation

```
python journal_expansion/scripts/run_v2.py --mode selfcal --signal delta_hard \
    --burn_in 10 --z_guard 0.5 --spatial_norm --abs_cap ...
```
