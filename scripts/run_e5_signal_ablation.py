"""
E5 (3-way SIGNAL ablation) — isolate the effect of the drift-signal SOURCE,
holding everything else fixed. Same temporal Schedule A as E2, 150 rounds.

Variants:
    entropy            - signal_mode=entropy (mu_H/tau_H)
    disagreement       - signal_mode=disagreement, beta_margin=0.0 (S1 only)   mu_drift/tau_drift
    disagreement_margin- signal_mode=disagreement, beta_margin=0.5 (S1+S2)     mu_drift/tau_drift

This is the direct C1 evidence: same controller, only the signal changes.
"""
import sys, os, argparse, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_single import run_experiment, load_config

VARIANTS = [
    ('entropy',             dict(signal_mode='entropy')),
    ('disagreement',        dict(signal_mode='disagreement', beta_margin=0.0)),
    ('disagreement_margin', dict(signal_mode='disagreement', beta_margin=0.5)),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_v3.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default='./results/e5_signal')
    p.add_argument('--temporal', default='A', choices=['A','B','C'])
    p.add_argument('--rounds', type=int, default=150)
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--skip_existing', action='store_true', default=True)
    args = p.parse_args()

    base = load_config(args.config)
    base['device'] = args.device
    base['global_rounds'] = args.rounds
    os.makedirs(args.output_dir, exist_ok=True)

    runs = VARIANTS
    if args.methods is not None:
        runs = [v for v in runs if v[0] in args.methods]

    t0 = time.time()
    for i, (name, overrides) in enumerate(runs):
        out_path = os.path.join(args.output_dir, f"{name}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(runs)}] SKIP {name} (exists)")
            continue
        cfg = copy.deepcopy(base)
        cfg.setdefault('adaptive', {})
        cfg['adaptive'].update(overrides)
        print(f"\n[{i+1}/{len(runs)}] E5-signal {name}  adaptive-overrides={overrides}")
        print(f"   mu_drift={cfg['adaptive'].get('mu_drift')} tau_drift={cfg['adaptive'].get('tau_drift')} "
              f"mu_H={cfg['adaptive'].get('mu_H')} tau_H={cfg['adaptive'].get('tau_H')}")
        run_experiment(
            cfg=cfg, method='adaptive_splitomc',
            temporal_schedule=args.temporal,
            run_name=name, output_dir=args.output_dir,
            eval_every=10, verbose=True,
        )
    print(f"\nE5-signal done. Total: {(time.time()-t0)/60:.1f}m")

if __name__ == "__main__":
    main()
