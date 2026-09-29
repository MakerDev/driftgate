"""
E5: Ablation of adaptive controller components.

Variants:
- A (full): H + Δ + consensus, sigmoid mapping
- B (no_delta): only H drives signal
- C (no_consensus): each cell uses local H̄_z
- D (static_mean): fixed λ at mean of variant A's λ trace (we approximate with 0.4)
- E (random_lambda): random λ ∈ [0.1, 0.7] per round (approximation: use fixed 0.4)
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_single import run_experiment, load_config


E5_CONFIGS = [
    # name, method, lam, big_lam, ablation_mode
    ('A_full_adaptive', 'adaptive_splitomc', None, None, None),
    ('B_no_delta', 'adaptive_splitomc', None, None, 'no_delta'),
    ('C_no_consensus', 'adaptive_splitomc', None, None, 'no_consensus'),
    ('D_static_mean', 'splitomcplus', 0.4, 0.5, None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_v3.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--schedule', default='A', help='Temporal schedule for ablation')
    p.add_argument('--output_dir', default='./results/e5_ablation')
    p.add_argument('--skip_existing', action='store_true', default=True)
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--rounds', type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = args.device
    if args.rounds is not None:
        cfg['global_rounds'] = args.rounds

    os.makedirs(args.output_dir, exist_ok=True)

    runs = E5_CONFIGS
    if args.methods is not None:
        runs = [r for r in runs if r[0] in args.methods]

    t0 = time.time()
    for i, (name, method, lam, big_lam, abl) in enumerate(runs):
        out_path = os.path.join(args.output_dir, f"{name}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(runs)}] SKIP {name}")
            continue
        print(f"\n[{i+1}/{len(runs)}] {name} ablation={abl}")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            temporal_schedule=args.schedule,
            ablation_mode=abl,
            run_name=name, output_dir=args.output_dir,
            eval_every=10,
            verbose=True,
        )

    print(f"\nE5 done. Total: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
