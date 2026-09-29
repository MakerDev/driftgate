"""
E1: Static Pareto frontier across ρ.

Trains each (method, λ) combination ONCE for global_rounds.
Final evaluation sweeps ρ ∈ {0.0, 0.2, 0.4, 0.6, 0.8} on each trained model.

Single GPU usage (sequential). For parallelization across 2 GPUs, run two
instances with different --device and --methods filter, then merge with summary script.
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_single import run_experiment, load_config


E1_CONFIGS = [
    # name, method, lambda, big_lambda
    ('fedavg', 'fedavg', None, None),
    ('fedprox', 'fedprox', None, None),
    ('splitfed', 'splitfed', None, None),
    ('fedmes', 'fedmes', None, None),
    ('splitgp_lam0.2', 'splitgp', 0.2, None),
    ('splitgp_lam0.5', 'splitgp', 0.5, None),
    ('splitgp_lam0.8', 'splitgp', 0.8, None),
    ('splitomc_lam0.0', 'splitomc', 0.0, 0.5),
    ('splitomc_lam0.2', 'splitomc', 0.2, 0.5),
    ('splitomc_lam0.4', 'splitomc', 0.4, 0.5),
    ('splitomc_lam0.6', 'splitomc', 0.6, 0.5),
    ('splitomc_lam0.8', 'splitomc', 0.8, 0.5),
    ('splitomcplus_lam0.2_Lam0.5', 'splitomcplus', 0.2, 0.5),
    ('splitomcplus_lam0.4_Lam0.5', 'splitomcplus', 0.4, 0.5),
    ('splitomcplus_lam0.6_Lam0.5', 'splitomcplus', 0.6, 0.5),
    ('adaptive_splitomc', 'adaptive_splitomc', None, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_v3.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default='./results/e1_static')
    p.add_argument('--methods', nargs='+', default=None,
                  help='Filter: names to include')
    p.add_argument('--skip_existing', action='store_true', default=True)
    p.add_argument('--rounds', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = args.device
    if args.rounds is not None:
        cfg['global_rounds'] = args.rounds
    cfg['partition_seed'] = args.seed

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter
    runs = E1_CONFIGS
    if args.methods is not None:
        runs = [r for r in runs if r[0] in args.methods]

    t0 = time.time()
    DEFAULT_RHO_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]
    for i, (name, method, lam, big_lam) in enumerate(runs):
        out_path = os.path.join(args.output_dir, f"{name}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(runs)}] SKIP {name} (exists)")
            continue
        elapsed = (time.time() - t0) / 60
        print(f"\n[{i+1}/{len(runs)}] {name}  elapsed={elapsed:.0f}m")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            run_name=name, output_dir=args.output_dir,
            eval_every=cfg.get('eval_every_n_rounds', 10),
            eval_rho_sweep=DEFAULT_RHO_SWEEP,
            verbose=True,
        )

    # Post-process: rho sweep on each saved model
    # Currently we only evaluate at the default ρ during training.
    # For the final rho sweep, we'd need to load each model and re-eval.
    # For simplicity in this v3, we report final-round accuracy at the
    # default ρ; for full rho sweep one can launch separate eval-only runs.
    print(f"\nE1 done. Total time: {(time.time()-t0)/60:.1f}m")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
