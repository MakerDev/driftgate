"""
E3: Spatial heterogeneity. Each cell has different ρ_z simultaneously.

Tests whether Adaptive-SplitOMC reduces worst-cell accuracy gap.
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_single import run_experiment, load_config


E3_CONFIGS = [
    ('splitomcplus_lam0.2_Lam0.5', 'splitomcplus', 0.2, 0.5),
    ('splitomcplus_lam0.4_Lam0.5', 'splitomcplus', 0.4, 0.5),
    ('splitomcplus_lam0.6_Lam0.5', 'splitomcplus', 0.6, 0.5),
    ('adaptive_splitomc', 'adaptive_splitomc', None, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_v3.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--spatial_mode', default='equal_spread',
                  choices=['equal_spread', 'extreme'])
    p.add_argument('--output_dir', default='./results/e3_spatial')
    p.add_argument('--skip_existing', action='store_true', default=True)
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--rounds', type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = args.device
    if args.rounds is not None:
        cfg['global_rounds'] = args.rounds

    out_dir = os.path.join(args.output_dir, f'mode_{args.spatial_mode}')
    os.makedirs(out_dir, exist_ok=True)

    runs = E3_CONFIGS
    if args.methods is not None:
        runs = [r for r in runs if r[0] in args.methods]

    t0 = time.time()
    for i, (name, method, lam, big_lam) in enumerate(runs):
        out_path = os.path.join(out_dir, f"{name}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(runs)}] SKIP {name} (exists)")
            continue
        print(f"\n[{i+1}/{len(runs)}] {name} spatial={args.spatial_mode}")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            spatial_mode=args.spatial_mode,
            run_name=name, output_dir=out_dir,
            eval_every=20,
            verbose=True,
        )

    print(f"\nE3 done. Total: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
