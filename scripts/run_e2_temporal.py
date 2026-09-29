"""
E2: Temporal ρ drift. CORE EXPERIMENT.

Schedule A: piecewise constant ρ 0.0→0.4→0.8→0.4→0.0, 30 rounds each.
Schedule B: random ρ ∈ U(0, 0.8) every 30 rounds (one seed).

Compares fixed-λ SplitOMC vs Adaptive-SplitOMC. Integrated accuracy is
the headline metric.
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_single import run_experiment, load_config


E2_CONFIGS = [
    # name, method, lambda, big_lambda
    ('splitomcplus_lam0.2_Lam0.5', 'splitomcplus', 0.2, 0.5),
    ('splitomcplus_lam0.4_Lam0.5', 'splitomcplus', 0.4, 0.5),
    ('splitomcplus_lam0.6_Lam0.5', 'splitomcplus', 0.6, 0.5),
    ('adaptive_splitomc', 'adaptive_splitomc', None, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_v3.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--schedule', default='A', choices=['A', 'B'])
    p.add_argument('--output_dir', default='./results/e2_temporal')
    p.add_argument('--skip_existing', action='store_true', default=True)
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--rounds', type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = args.device
    if args.rounds is not None:
        cfg['global_rounds'] = args.rounds

    out_dir = os.path.join(args.output_dir, f'schedule_{args.schedule}')
    os.makedirs(out_dir, exist_ok=True)

    runs = E2_CONFIGS
    if args.methods is not None:
        runs = [r for r in runs if r[0] in args.methods]

    t0 = time.time()
    for i, (name, method, lam, big_lam) in enumerate(runs):
        out_path = os.path.join(out_dir, f"{name}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(runs)}] SKIP {name} (exists)")
            continue
        print(f"\n[{i+1}/{len(runs)}] {name} schedule={args.schedule}")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            temporal_schedule=args.schedule,
            run_name=name, output_dir=out_dir,
            eval_every=10,  # finer eval to capture transitions
            verbose=True,
        )

    print(f"\nE2 done. Total: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
