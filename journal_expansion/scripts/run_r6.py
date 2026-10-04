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
    # Round 7 gate (all off by default -> Round 6 behaviour)
    ap.add_argument("--record_device_signals", action="store_true",
                    help="every client computes x_TV and x_SR on its probe in every arm (no grad)")
    ap.add_argument("--device_lambda", action="store_true",
                    help="one lambda per client from its own signal (needs --mode selfcal and the relonly flags)")
    ap.add_argument("--device_signal", default=None, choices=["tv", "sr"])
    ap.add_argument("--oracle_home_away", action="store_true",
                    help="lambda_k = 0.70 at home, 0.15 away from the environment (needs --mode fixed)")
    ap.add_argument("--fixed_Lambda", type=float, default=None, help="fix Lambda for every cell")
    # Round 8 check (off by default -> Round 6/7 behaviour)
    ap.add_argument("--eval_infer_lambdas", default=None,
                    help="comma-separated lambda_inf values; every evaluation round also evaluates "
                         "lambda_inf * theta_local + (1 - lambda_inf) * cell average (no effect on training)")
    ap.add_argument("--eval_mainaware_route", action="store_true",
                    help="also count the Main-aware routing rule on the installed model (reference only)")
    # Round 8 v2 (off by default)
    ap.add_argument("--record_eval_requests", action="store_true",
                    help="per evaluated request and lambda_inf block: client-exit prediction and entropy, server-exit "
                         "prediction, label, kind, client, round, at_home -> {run}_requests.npz")
    ap.add_argument("--record_eval_probs", action="store_true",
                    help="Round 10: per evaluated request, softmax probabilities of both exits (float16) and the "
                         "request items, with an arrival order -> {run}_evalprobs.npz")
    ap.add_argument("--record_train_label_hist", action="store_true",
                    help="Round 10: per training round, label counts trained by each cell's server block and by all clients")
    ap.add_argument("--record_probe_values", action="store_true",
                    help="per probe request TV and server-non-Main indicator (needs --record_device_signals)")
    # Round 13b (off by default -> CIFAR-10 and the default split CNN)
    ap.add_argument("--dataset", default=None, choices=["cifar100"], help="data set other than CIFAR-10")
    ap.add_argument("--model_family", default=None, choices=["resnet20", "vgg11"])
    ap.add_argument("--split_point", default=None, choices=["shallow", "middle"])
    ap.add_argument("--learning_rate", type=float, default=None,
                    help="override of the config learning rate (Round 13b: only after a diverged run, halved once)")
    args = ap.parse_args()
    if (args.model_family is None) != (args.split_point is None):
        raise SystemExit("--model_family and --split_point go together")

    if not args.disjoint_pools:
        raise SystemExit("Round 6 runs must use --disjoint_pools")
    if args.mode == "selfcal" and not (args.burn_in == 10 and args.z_guard == 0.5 and args.spatial_norm):
        raise SystemExit("DriftGate/entropy arms must use --burn_in 10 --z_guard 0.5 --spatial_norm")
    mode = args.mode
    if args.device_lambda:
        if args.mode != "selfcal" or args.device_signal is None:
            raise SystemExit("--device_lambda needs --mode selfcal (relonly flags) and --device_signal")
        mode = "device"
    if args.oracle_home_away:
        if args.mode != "fixed":
            raise SystemExit("--oracle_home_away needs --mode fixed")
        mode = "oracle"
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
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    ckw = {}
    if args.mode == "selfcal":
        ckw = dict(burn_in=args.burn_in, z_guard=args.z_guard, spatial_norm=args.spatial_norm,
                   no_neighbor_avg=args.no_neighbor_avg)
    run_r6(cfg, env_path, mode, signal=args.signal, lambda_val=args.lambda_val,
           big_lambda_val=args.big_lambda_val, apfl_eta=args.apfl_eta,
           signal_delay=args.signal_delay, probe_n=args.probe_n, controller_kwargs=ckw,
           run_name=args.run_name, output_dir=args.output_dir, eval_every=args.eval_every,
           save_models=args.save_models, scenario=args.scenario, arm=args.arm,
           record_device_signals=args.record_device_signals, device_signal=args.device_signal,
           oracle_home_away=args.oracle_home_away, fixed_Lambda=args.fixed_Lambda,
           eval_infer_lambdas=([float(v) for v in args.eval_infer_lambdas.split(",")]
                               if args.eval_infer_lambdas else None),
           eval_mainaware_route=args.eval_mainaware_route,
           record_eval_requests=args.record_eval_requests, record_probe_values=args.record_probe_values,
           record_eval_probs=args.record_eval_probs, record_train_label_hist=args.record_train_label_hist,
           dataset=args.dataset, model_family=args.model_family, split_point=args.split_point)


if __name__ == "__main__":
    main()
