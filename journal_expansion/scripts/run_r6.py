"""Round 6 CLI: one process = one run = one provenance record.

DriftGate (relonly):
  python -u journal_expansion/scripts/run_r6.py --scenario S1 --arm driftgate \
      --mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --disjoint_pools \
      --seed 0 --model_seed 100 --run_name s1_driftgate_s0 --output_dir journal_expansion/runs/phaseT6_S1
Fixed lambda:   --mode fixed --lambda_val 0.4 --big_lambda_val 0.5 --disjoint_pools
Entropy:        DriftGate flags with --signal ent_client
APFL:           --mode apfl --apfl_eta 0.01 --big_lambda_val 0.5 --disjoint_pools
The environment file defaults to runs/phaseT6_env/<env>_seed<seed>.npz (--env; S4 delay arms use S1).
An existing result JSON is never overwritten (the run is skipped).
"""
import os

_THREADS = os.environ.get("JX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(int(_THREADS))
JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.runner_r6 import run_r6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/base_v3.yaml"))
    ap.add_argument("--scenario", required=True, help="label, e.g. S1, S1fast, S2, S3_K200, S4_delay3")
    ap.add_argument("--env", default=None, help="environment file prefix (default: --scenario)")
    ap.add_argument("--arm", required=True, help="label, e.g. driftgate, fixed0.4, entropy, apfl0.01")
    ap.add_argument("--mode", required=True, choices=["selfcal", "fixed", "apfl"])
    ap.add_argument("--signal", default="tv_dist", choices=["tv_dist", "ent_client"])
    ap.add_argument("--burn_in", type=int, default=0)
    ap.add_argument("--z_guard", type=float, default=None)
    ap.add_argument("--spatial_norm", action="store_true")
    ap.add_argument("--disjoint_pools", action="store_true")
    ap.add_argument("--lambda_val", type=float, default=0.4)
    ap.add_argument("--big_lambda_val", type=float, default=0.5)
    ap.add_argument("--apfl_eta", type=float, default=None)
    ap.add_argument("--no_neighbor_avg", action="store_true",
                    help="q from the edge's own score only (the R6 final DriftGate definition)")
    ap.add_argument("--signal_delay", type=int, default=0)
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model_seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    if not args.disjoint_pools:
        raise SystemExit("Round 6 runs must use --disjoint_pools")
    if args.mode == "selfcal" and not (args.burn_in == 10 and args.z_guard == 0.5 and args.spatial_norm):
        raise SystemExit("DriftGate/entropy arms must use --burn_in 10 --z_guard 0.5 --spatial_norm")
    out_json = Path(args.output_dir) / f"{args.run_name}.json"
    if out_json.exists():
        print(f"{out_json} exists; not re-running (results are never overwritten)")
        return
    env_path = JOURNAL_ROOT / "runs" / "phaseT6_env" / f"{args.env or args.scenario}_seed{args.seed}.npz"
    if not env_path.exists():
        raise SystemExit(f"missing environment file {env_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(device=args.device, partition_seed=args.seed, model_seed=args.model_seed,
               global_rounds=args.rounds)
    ckw = {}
    if args.mode == "selfcal":
        ckw = dict(burn_in=args.burn_in, z_guard=args.z_guard, spatial_norm=args.spatial_norm,
                   no_neighbor_avg=args.no_neighbor_avg)
    run_r6(cfg, env_path, args.mode, signal=args.signal, lambda_val=args.lambda_val,
           big_lambda_val=args.big_lambda_val, apfl_eta=args.apfl_eta,
           signal_delay=args.signal_delay, probe_n=args.probe_n, controller_kwargs=ckw,
           run_name=args.run_name, output_dir=args.output_dir, eval_every=args.eval_every,
           save_models=args.save_models, scenario=args.scenario, arm=args.arm)


if __name__ == "__main__":
    main()
