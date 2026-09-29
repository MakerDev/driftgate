"""Generic CLI for journal-expansion runs (passive signal recording or
controller-in-the-loop). One process = one run = one provenance record.

Examples:
  # passive Gate-B recording run (fixed lambda backbone, all signals recorded)
  python journal_expansion/scripts/run_v2.py --mode passive --schedule abrupt \
      --seed 0 --model_seed 100 --rounds 100 --device cuda:1 \
      --run_name rec_abrupt_s0 --output_dir journal_expansion/runs/signal_benchmark

  # downstream self-calibrating controller run
  python journal_expansion/scripts/run_v2.py --mode selfcal --signal delta_hard \
      --schedule A --seed 0 --rounds 100 --device cuda:0 ...
"""
import os
# CPU-thread cap (set BEFORE torch/numpy import). This is a CPU-bound
# federated simulation (50-client sequential training + per-image corruption);
# with a dozen concurrent jobs, PyTorch's default per-process thread pool
# oversubscribes the 48 cores (load avg hit 111) and STARVES the GPU. Capping
# each job to a few compute threads keeps total threads ≈ core count, which
# speeds every job AND raises GPU utilisation. Overridable via env for tuning.
_THREADS = os.environ.get("JX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _THREADS)

import sys
import argparse
import yaml
from pathlib import Path

import torch
torch.set_num_threads(int(_THREADS))

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.runner import run_experiment_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/base_v3.yaml"))
    ap.add_argument("--mode", default="passive",
                    choices=["passive", "fixed", "legacy", "selfcal",
                             "proxy_grid", "ucb", "exp3", "oracle_greedy", "replay",
                             "apfl"])
    ap.add_argument("--replay_from", default=None,
                    help="replay mode: source run JSON with lamdas/big_lamdas traces")
    ap.add_argument("--replay_transform", default="identity",
                    choices=["identity", "shuffle", "shift10", "shift20", "reverse",
                             "cluster_shuffle", "lowpass"])
    ap.add_argument("--lambda_per_cell", default=None,
                    help="comma list of per-cell fixed lambdas, e.g. '0.3,0.4,0.5,0.6,0.6'")
    ap.add_argument("--Lambda_per_cell", default=None,
                    help="comma list of per-cell fixed Lambdas")
    ap.add_argument("--method", default="splitomcplus")
    ap.add_argument("--lambda_val", type=float, default=0.4)
    ap.add_argument("--big_lambda_val", type=float, default=0.5)
    ap.add_argument("--schedule", default="A")
    ap.add_argument("--spatial", default=None)
    ap.add_argument("--signal", default="delta_hard")
    ap.add_argument("--normalizer", default="guarded")
    ap.add_argument("--probe_n", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_seed", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mu", type=float, default=None, help="legacy-mode override")
    ap.add_argument("--tau", type=float, default=None, help="legacy-mode override")
    ap.add_argument("--burn_in", type=int, default=0,
                    help="selfcal v3b: discard first N rounds before warm-up")
    ap.add_argument("--z_guard", type=float, default=None,
                    help="selfcal v3c: freeze baseline adaptation when z exceeds this")
    ap.add_argument("--spatial_norm", action="store_true",
                    help="selfcal-ST: add cross-cell robust-z (max with temporal z)")
    ap.add_argument("--abs_cap", action="store_true",
                    help="selfcal dual-view: cap lambda by the absolute delta level "
                         "(only meaningful with --signal delta_hard)")
    ap.add_argument("--lam_min", type=float, default=None,
                    help="override adaptive lambda lower bound (bounds ablation)")
    ap.add_argument("--lam_max", type=float, default=None,
                    help="override adaptive lambda upper bound (bounds ablation)")
    ap.add_argument("--fairness", default=None,
                    choices=["deployable", "oracle", "random"],
                    help="F1/F2 donor-selected inter-cell aggregation")
    ap.add_argument("--partition_mode", default=None,
                    help="nd1 (default) or dirichlet:<alpha>")
    ap.add_argument("--model_family", default=None,
                    choices=["cnn", "resnet18", "mobilenetv2"])
    ap.add_argument("--split_point", default=None,
                    choices=["early", "middle", "late"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--no_routing_eval", action="store_true")
    ap.add_argument("--topology", default=None,
                    choices=["line", "ring", "grid", "star", "rgg", "dynamic"])
    ap.add_argument("--signal_delay", type=int, default=0)
    ap.add_argument("--signal_loss", type=float, default=0.0)
    ap.add_argument("--participation", type=float, default=1.0)
    ap.add_argument("--corruption", default=None,
                    choices=["gaussian_noise", "motion_blur", "brightness",
                             "contrast", "jpeg"])
    ap.add_argument("--corruption_schedule", default="gradual_sev",
                    choices=["abrupt_sev", "gradual_sev", "recurring_sev"])
    ap.add_argument("--mobility", action="store_true")
    ap.add_argument("--mobility_speed_level", default="med",
                    choices=["slow", "med", "fast"])
    ap.add_argument("--mobility_speed", type=float, default=None,
                    help="ignored placeholder for older manifests")
    ap.add_argument("--num_clients", type=int, default=None,
                    help="override client count (Phase F: fewer clients for large models)")
    ap.add_argument("--role_mode", default="separated",
                    choices=["separated", "same_role", "same_role_indep", "weak_server"],
                    help="§7 role-separation ablation (perturbs aggregation role structure only)")
    ap.add_argument("--fixed_Lambda", type=float, default=None,
                    help="Task A1 diagnostic: freeze Lambda at this value while lambda stays adaptive")
    ap.add_argument("--disjoint_pools", action="store_true",
                    help="Task 1: controller probe pool and evaluation pool share no sample ID")
    ap.add_argument("--controller_pool_frac", type=float, default=0.2)
    ap.add_argument("--abs_only", action="store_true",
                    help="R4/B1: lambda=lambda_abs, Lambda=Lambda_abs (absolute view only)")
    ap.add_argument("--apfl_eta", type=float, default=None,
                    help="R4/B4: APFL-style lambda_k learning rate (default = model lr)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = args.device
    cfg["partition_seed"] = args.seed
    cfg["model_seed"] = args.model_seed
    cfg["global_rounds"] = args.rounds
    if args.num_clients:
        cfg["num_clients"] = args.num_clients
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.partition_mode:
        cfg["partition_mode"] = args.partition_mode
    if args.model_family:
        cfg["model_family"] = args.model_family
    if args.split_point:
        cfg["split_point"] = args.split_point

    controller_mode = {"passive": "fixed", "fixed": "fixed"}.get(args.mode, args.mode)
    # R4/B4: apfl uses NO probe/signal -> do not record (keeps probes untouched)
    record = args.mode == "passive" or controller_mode not in ("fixed", "apfl")

    ckw = {}
    if args.replay_from:
        ckw["replay_from"] = args.replay_from
        ckw["transform"] = args.replay_transform
    if args.abs_only:
        ckw["abs_only"] = True
        ckw["abs_cap"] = True   # abs_only needs the absolute-signal smoothing path
    if args.apfl_eta is not None:
        ckw["apfl_eta"] = args.apfl_eta
    lambda_val = args.lambda_val
    big_lambda_val = args.big_lambda_val
    if args.lambda_per_cell:
        lambda_val = {i: float(v) for i, v in enumerate(args.lambda_per_cell.split(","))}
    if args.Lambda_per_cell:
        big_lambda_val = {i: float(v) for i, v in enumerate(args.Lambda_per_cell.split(","))}
    if args.mu is not None:
        ckw["mu"] = args.mu
    if args.tau is not None:
        ckw["tau"] = args.tau
    if args.burn_in:
        ckw["burn_in"] = args.burn_in
    if args.z_guard is not None:
        ckw["z_guard"] = args.z_guard
    if args.spatial_norm:
        ckw["spatial_norm"] = True
    if args.abs_cap:
        ckw["abs_cap"] = True
    if args.lam_min is not None:
        cfg.setdefault("adaptive", {})["lam_min"] = args.lam_min
    if args.lam_max is not None:
        cfg.setdefault("adaptive", {})["lam_max"] = args.lam_max

    method = args.method
    if controller_mode != "fixed":
        method = "splitomcplus"  # adaptive runs train exactly like splitomcplus

    run_experiment_v2(
        cfg=cfg, method=method,
        lambda_val=lambda_val, big_lambda_val=big_lambda_val,
        schedule=args.schedule, spatial_mode=args.spatial,
        controller_mode=controller_mode, controller_signal=args.signal,
        normalizer=args.normalizer, record_signals=record,
        probe_n=args.probe_n, run_name=args.run_name,
        output_dir=args.output_dir, eval_every=args.eval_every,
        provenance_prefix=args.mode, controller_kwargs=ckw,
        fairness_mode=args.fairness, routing_eval=not args.no_routing_eval,
        network_cfg={"topology": args.topology, "signal_delay": args.signal_delay,
                     "signal_loss": args.signal_loss,
                     "participation": args.participation},
        corruption_cfg=({"type": args.corruption,
                         "schedule": args.corruption_schedule}
                        if args.corruption else None),
        mobility_cfg=({"speed_level": args.mobility_speed_level}
                      if args.mobility else None),
        role_mode=args.role_mode, fixed_Lambda=args.fixed_Lambda,
        disjoint_pools=args.disjoint_pools,
        controller_pool_frac=args.controller_pool_frac)


if __name__ == "__main__":
    main()
