"""
Central experiment driver. Handles all methods, ND1 setup, eval grid,
optional mobility/temporal/spatial scenarios.
"""
import os
import sys
import time
import json
import yaml
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.partition import (
    make_es_topology, nd1_partition, compute_all_used_classes,
    es_neighbors_sequential, get_cifar10, make_client_dataloader,
)
from models.architectures import ModelFactory, count_params
from train.trainer import build_clients_and_es, synchronize_initial_models, run_one_global_round
from eval.evaluator import build_per_client_test_sets, evaluate_all_clients
from controller.adaptive import (
    ClientDriftTracker, gather_signals_for_clients,
    aggregate_signals_per_es, AdaptiveController,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(config, overrides):
    """Merge command-line overrides into config dict."""
    out = dict(config)
    for k, v in overrides.items():
        if v is None:
            continue
        out[k] = v
    return out


def get_temporal_rho(round_idx, schedule_name, total_rounds=150):
    """Get current rho for round_idx given a temporal drift schedule."""
    if schedule_name is None or schedule_name == 'static':
        return None  # constant; use cfg.oop_ratio_default

    if schedule_name == 'A':
        # piecewise: 0.0 -> 0.4 -> 0.8 -> 0.4 -> 0.0, 30 rounds each
        seg = (round_idx - 1) // 30
        seq = [0.0, 0.4, 0.8, 0.4, 0.0]
        return seq[min(seg, 4)]

    if schedule_name == 'B':
        # random per 30R
        rng = np.random.default_rng(round_idx // 30 + 1000)
        return float(rng.uniform(0, 0.8))

    if schedule_name == 'C':
        # smooth oscillation
        t_norm = round_idx / total_rounds
        return 0.4 * (1 - np.cos(2 * np.pi * t_norm)) / 2.0

    raise ValueError(f"Unknown schedule: {schedule_name}")


def make_spatial_rho(num_es, mode='equal_spread'):
    """For E3: per-cell rho_z (fixed throughout training)."""
    if mode == 'equal_spread':
        rhos = np.linspace(0.0, 0.8, num_es).tolist()
    elif mode == 'extreme':
        rhos = [0.0] * (num_es - 1) + [0.8]
    else:
        raise ValueError(f"Unknown spatial mode: {mode}")
    return {i: rhos[i] for i in range(num_es)}


def run_experiment(cfg, method='splitomc', lambda_val=None, big_lambda_val=None,
                  temporal_schedule=None, spatial_mode=None, use_mobility=False,
                  ablation_mode=None, run_name=None, output_dir=None,
                  log_every=10, eval_every=10, eval_rho_sweep=None, verbose=True):
    """
    Run a single training experiment with the given configuration.

    eval_rho_sweep: optional list of ρ values to evaluate at the FINAL round
                   (for Pareto curve construction).

    Returns dict with all results.
    """
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.get('partition_seed', 0))

    # ---------- Setup data and topology ----------
    train_ds, test_ds = get_cifar10(data_root=cfg.get('data_root', './data_cache'),
                                    augment=False)
    train_labels = np.array(train_ds.targets)
    test_labels = np.array(test_ds.targets)

    c2es, es2c = make_es_topology(
        cfg['num_clients'], cfg['num_edge_servers'],
        cfg['overlap_percentage'], seed=cfg['partition_seed'])
    indices, main_classes, scope = nd1_partition(
        train_labels, cfg['num_classes'], cfg['num_clients'],
        c2es, es2c,
        classes_per_es_frac=(cfg['classes_per_es_min_frac'], cfg['classes_per_es_max_frac']),
        classes_per_client_frac=cfg['classes_per_client_frac'],
        seed=cfg['partition_seed'])

    # Apply spatial rho if specified
    per_cell_oop = None
    if spatial_mode is not None:
        per_cell_oop = make_spatial_rho(cfg['num_edge_servers'], mode=spatial_mode)

    # ---------- Build models ----------
    set_seed(cfg.get('model_seed', 100))
    mf = ModelFactory(dataset=cfg['dataset'], num_classes=cfg['num_classes'])
    clients, edge_servers = build_clients_and_es(
        num_clients=cfg['num_clients'],
        num_edge_servers=cfg['num_edge_servers'],
        model_factory=mf,
        train_dataset=train_ds,
        client_indices=indices,
        client_to_es=c2es,
        device=device,
        batch_size=cfg['batch_size'],
        local_epochs=cfg['local_epochs'],
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        num_workers=0,
    )
    synchronize_initial_models(clients, edge_servers)
    # Set ES scope (used by Evaluator)
    for es_id, es in edge_servers.items():
        es.scope = scope[es_id]

    # ---------- Adaptive controller setup ----------
    nbrs = es_neighbors_sequential(cfg['num_edge_servers'])
    adaptive_cfg = cfg.get('adaptive', {})

    # Ablation
    abl_use_delta = adaptive_cfg.get('use_delta', True)
    abl_consensus = True
    if ablation_mode == 'no_delta':
        abl_use_delta = False
    elif ablation_mode == 'no_H':
        # Special: only delta drives signal
        pass
    elif ablation_mode == 'no_consensus':
        abl_consensus = False

    # Signal source: 'entropy' (original) or 'disagreement' (client-vs-server head).
    # For disagreement the sigmoid is calibrated on the drift range (mu_drift/tau_drift),
    # not entropy (mu_H/tau_H). Mapping/consensus code is unchanged: it treats the
    # primary signal as a generic drift scalar.
    signal_mode = adaptive_cfg.get('signal_mode', 'entropy')
    beta_margin = adaptive_cfg.get('beta_margin', 0.0)
    if signal_mode == 'disagreement':
        ctrl_mu = adaptive_cfg.get('mu_drift', 0.3)
        ctrl_tau = adaptive_cfg.get('tau_drift', 0.1)
    else:
        ctrl_mu = adaptive_cfg.get('mu_H', 1.5)
        ctrl_tau = adaptive_cfg.get('tau_H', 0.4)

    controller = AdaptiveController(
        neighbors=nbrs if abl_consensus else {i: [] for i in range(cfg['num_edge_servers'])},
        mu_H=ctrl_mu,
        tau_H=ctrl_tau,
        lam_min=adaptive_cfg.get('lam_min', 0.1),
        lam_max=adaptive_cfg.get('lam_max', 0.7),
        Lam_min=adaptive_cfg.get('Lam_min', 0.2),
        Lam_max=adaptive_cfg.get('Lam_max', 0.8),
        consensus_steps=adaptive_cfg.get('consensus_steps', 1),
        use_delta=abl_use_delta,
        delta_weight=adaptive_cfg.get('delta_weight', 0.3),
    )

    # Drift trackers (only initialized for adaptive method or ablation)
    drift_trackers = {}
    if method == 'adaptive_splitomc' or ablation_mode is not None:
        for c in clients:
            t = ClientDriftTracker(
                anchor_buffer_size=adaptive_cfg.get('anchor_buffer_size', 16),
                entropy_buffer_size=adaptive_cfg.get('drift_buffer_size', 64),
                device=device,
            )
            t.register_anchors(c.train_loader)
            drift_trackers[c.cid] = t

    # ---------- Mobility setup ----------
    mobility = None
    if use_mobility:
        from network.mobility import GaussMarkovMobility, rewire_clients, es_to_clients_from_c2es
        mobility = GaussMarkovMobility(
            num_clients=cfg['num_clients'],
            num_es=cfg['num_edge_servers'],
            map_size=1000.0, alpha=0.9, speed=10.0, dt=10.0,
            seed=cfg.get('mobility_seed', 42),
        )

    # ---------- Default per-cell lambdas / big_lambdas ----------
    if lambda_val is None:
        lambda_val = cfg.get('lambda_default', 0.2)
    if big_lambda_val is None:
        big_lambda_val = cfg.get('big_lambda_default', 0.5)

    fixed_lamdas = {es_id: lambda_val for es_id in range(cfg['num_edge_servers'])}
    fixed_big_lamdas = {es_id: big_lambda_val for es_id in range(cfg['num_edge_servers'])}

    # ---------- Logging ----------
    history = {
        'round': [],
        'mean_loss': [],
        'lamdas': [],
        'big_lamdas': [],
        'H_per_es': [],  # per-round primary signal per ES (entropy or drift)
        'delta_per_es': [],  # per-round raw S1 disagreement rate per ES (disagreement mode)
        'margin_per_es': [],  # per-round S2 margin per ES (disagreement mode)
        'rho_for_anchor': [],  # ρ used for probe anchor that round
        'eval': [],  # list of dicts at each eval round
        'rho_used': [],
        'mobility_changes': [],
    }

    # ---------- Training loop ----------
    if verbose:
        print(f"[{run_name}] Starting {method} with lambda={lambda_val}, big_lambda={big_lambda_val}")
        print(f"  Clients: {len(clients)}, ES: {len(edge_servers)}, Device: {device}")
        print(f"  Total rounds: {cfg['global_rounds']}, Local epochs: {cfg['local_epochs']}")

    t0 = time.time()

    # Probe-anchor state: when ρ changes (or first time), rebuild per-client probe indices
    # that simulate the ρ-mixed incoming traffic the deployed model would see.
    _probe_last_rho = None
    _probe_per_client_indices = None
    _probe_rng = np.random.default_rng(int(cfg.get('partition_seed', 0)) * 31 + 17)

    for r in range(1, cfg['global_rounds'] + 1):
        # 1. Mobility (if enabled)
        if mobility is not None:
            mobility.step()
            if r % 5 == 0:  # rewire every 5 rounds to reduce thrash
                new_c2es = mobility.get_topk_es(k=2)
                # Detect changes. NOTE: `clients` is a LIST that skips clients
                # with empty data, so list index != cid in general; index by a
                # cid map (bug O3 — dormant in all recorded runs because the
                # ND1 partitions never produced an empty client, verified in
                # tests/test_journal.py::test_mobility_cid_indexing).
                by_cid = {c.cid: c for c in clients}
                changes = sum(1 for cid, c in by_cid.items()
                              if set(new_c2es[cid]) != set(c.edge_server_ids))
                if changes > 0:
                    from network.mobility import rewire_clients
                    rewire_clients(clients, edge_servers, new_c2es, mf)
                    history['mobility_changes'].append((r, changes))

        # 2. Compute lambdas
        if method == 'adaptive_splitomc' or ablation_mode is not None:
            # Refresh each client's anchor from current ρ-mixed incoming traffic.
            # ρ for this round mirrors what the deployed model would serve at
            # this point in the schedule (or the static default if no schedule).
            if temporal_schedule is not None:
                rho_now = get_temporal_rho(r, temporal_schedule, cfg['global_rounds'])
            else:
                rho_now = cfg.get('oop_ratio_default', 0.4)

            if rho_now != _probe_last_rho:
                _probe_per_client_indices, _, _ = build_per_client_test_sets(
                    test_labels, cfg['num_clients'], main_classes, c2es, scope,
                    oop_ratio=rho_now,
                    oor_ratio_factor=cfg.get('oor_ratio_factor', 0.3),
                    per_cell_oop_ratio=per_cell_oop,
                )
                _probe_last_rho = rho_now

            n_probe = adaptive_cfg.get('anchor_buffer_size', 16)
            for c in clients:
                if c.cid not in drift_trackers:
                    continue
                cid_idxs = _probe_per_client_indices.get(c.cid, [])
                if not cid_idxs:
                    continue
                k = min(n_probe, len(cid_idxs))
                chosen = _probe_rng.choice(cid_idxs, size=k, replace=False)
                imgs = torch.stack([test_ds[int(i)][0] for i in chosen])
                drift_trackers[c.cid].update_anchors_from_traffic(imgs)

        if method == 'adaptive_splitomc':
            # First round: use defaults
            if r == 1:
                cur_lamdas = fixed_lamdas
                cur_big_lamdas = fixed_big_lamdas
                history['H_per_es'].append({})
                history['delta_per_es'].append({})
                history['margin_per_es'].append({})
                history['rho_for_anchor'].append(rho_now)
            else:
                signals = gather_signals_for_clients(
                    clients, drift_trackers,
                    signal_mode=signal_mode, beta_margin=beta_margin)
                H_per_es, D_per_es = aggregate_signals_per_es(signals, clients)
                cur_lamdas, cur_big_lamdas = controller.step(
                    H_per_es, D_per_es if abl_use_delta else None)
                history['H_per_es'].append(dict(H_per_es))
                history['rho_for_anchor'].append(rho_now)

                # Raw S1/S2 per-ES aggregation (disagreement mode) for gates + logging.
                delta_per_es, margin_per_es = {}, {}
                if signal_mode == 'disagreement':
                    raw_d = {c.cid: (drift_trackers[c.cid].last_delta, 0.0)
                             for c in clients if c.cid in drift_trackers}
                    raw_m = {c.cid: (drift_trackers[c.cid].last_margin, 0.0)
                             for c in clients if c.cid in drift_trackers}
                    delta_per_es, _ = aggregate_signals_per_es(raw_d, clients)
                    margin_per_es, _ = aggregate_signals_per_es(raw_m, clients)
                history['delta_per_es'].append(dict(delta_per_es))
                history['margin_per_es'].append(dict(margin_per_es))

                if verbose and (r <= 10 or r % 5 == 0):
                    L_str = " ".join(f"{cur_lamdas[k]:.3f}" for k in sorted(cur_lamdas.keys()))
                    if signal_mode == 'disagreement':
                        d_str = " ".join(f"{delta_per_es[k]:.3f}" for k in sorted(delta_per_es.keys()))
                        m_str = " ".join(f"{margin_per_es[k]:.3f}" for k in sorted(margin_per_es.keys()))
                        print(f"  [adapt R{r:3d} ρ={rho_now:.1f}] delta_z={d_str} margin_z={m_str} λ={L_str}")
                    else:
                        H_str = " ".join(f"{H_per_es[k]:.3f}" for k in sorted(H_per_es.keys()))
                        print(f"  [adapt R{r:3d} ρ={rho_now:.1f}] H={H_str} λ={L_str}")
        else:
            cur_lamdas = fixed_lamdas
            cur_big_lamdas = fixed_big_lamdas
            history['H_per_es'].append({})
            history['delta_per_es'].append({})
            history['margin_per_es'].append({})
            history['rho_for_anchor'].append(None)

        # 3. Run one global round
        mean_loss = run_one_global_round(
            clients=clients,
            edge_servers_dict=edge_servers,
            lamdas=cur_lamdas,
            big_lambdas=cur_big_lamdas,
            method=method if method != 'adaptive_splitomc' else 'splitomcplus',
            gamma=cfg.get('gamma', 0.5),
        )

        # 4. Log
        history['round'].append(r)
        history['mean_loss'].append(mean_loss)
        history['lamdas'].append(dict(cur_lamdas))
        history['big_lamdas'].append(dict(cur_big_lamdas))

        # 5. Eval
        do_eval = (r == cfg['global_rounds']) or (r % eval_every == 0) or (r == 1)
        if do_eval:
            # Determine rho for this eval
            if temporal_schedule is not None:
                rho_eval = get_temporal_rho(r, temporal_schedule, cfg['global_rounds'])
            else:
                rho_eval = cfg.get('oop_ratio_default', 0.4)
            history['rho_used'].append((r, rho_eval))

            # Build test set
            test_indices, oop_per_c, oor_per_c = build_per_client_test_sets(
                test_labels, cfg['num_clients'], main_classes, c2es, scope,
                oop_ratio=rho_eval, oor_ratio_factor=cfg.get('oor_ratio_factor', 0.3),
                per_cell_oop_ratio=per_cell_oop,
            )
            agg, _ = evaluate_all_clients(
                clients=clients, test_dataset=test_ds,
                per_client_test_indices=test_indices,
                client_main_classes=main_classes,
                oop_classes_per_client=oop_per_c,
                oor_classes_per_client=oor_per_c,
                eth=cfg.get('eth_default', 0.8),
            )
            agg['round'] = r
            agg['rho'] = rho_eval
            agg['mean_loss'] = mean_loss
            history['eval'].append(agg)

            if verbose:
                elapsed = time.time() - t0
                eta_min = (cfg['global_rounds'] - r) * elapsed / r / 60.0
                print(f"  R{r:3d} loss={mean_loss:.3f} acc_main={agg.get('acc_main', 0):.3f} "
                      f"acc_oop={agg.get('acc_oop', 0):.3f} acc_oor={agg.get('acc_oor', 0):.3f} "
                      f"acc_total={agg['acc_total']:.3f} elapsed={elapsed/60:.1f}m ETA={eta_min:.0f}m")

    history['total_time_sec'] = time.time() - t0
    history['config'] = {
        'method': method,
        'lambda_val': lambda_val,
        'big_lambda_val': big_lambda_val,
        'temporal_schedule': temporal_schedule,
        'spatial_mode': spatial_mode,
        'use_mobility': use_mobility,
        'ablation_mode': ablation_mode,
    }

    # ---------- Final-round rho sweep ----------
    if eval_rho_sweep is not None and len(eval_rho_sweep) > 0:
        if verbose:
            print(f"  Running final rho sweep: {eval_rho_sweep}")
        rho_sweep_results = {}
        for rho_val in eval_rho_sweep:
            test_indices, oop_per_c, oor_per_c = build_per_client_test_sets(
                test_labels, cfg['num_clients'], main_classes, c2es, scope,
                oop_ratio=rho_val, oor_ratio_factor=cfg.get('oor_ratio_factor', 0.3),
                per_cell_oop_ratio=per_cell_oop,
            )
            agg, _ = evaluate_all_clients(
                clients=clients, test_dataset=test_ds,
                per_client_test_indices=test_indices,
                client_main_classes=main_classes,
                oop_classes_per_client=oop_per_c,
                oor_classes_per_client=oor_per_c,
                eth=cfg.get('eth_default', 0.8),
            )
            rho_sweep_results[f'rho_{rho_val}'] = agg
            if verbose:
                print(f"    ρ={rho_val}: acc_total={agg['acc_total']:.3f}")
        history['final_rho_sweep'] = rho_sweep_results

    # ---------- Save ----------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = run_name if run_name else f"{method}_lam{lambda_val}_Lam{big_lambda_val}"
        path = os.path.join(output_dir, f"{fname}.json")
        with open(path, 'w') as f:
            json.dump(history, f, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        if verbose:
            print(f"  Saved: {path}")

    return history


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_v3.yaml')
    parser.add_argument('--method', type=str, default='splitomc',
                       choices=['fedavg', 'fedprox', 'splitfed', 'fedmes',
                               'splitgp', 'splitomc', 'splitomcplus', 'adaptive_splitomc'])
    parser.add_argument('--lambda_val', type=float, default=None)
    parser.add_argument('--big_lambda_val', type=float, default=None)
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--temporal', type=str, default=None, choices=['A', 'B', 'C'])
    parser.add_argument('--spatial', type=str, default=None,
                       choices=['equal_spread', 'extreme'])
    parser.add_argument('--mobility', action='store_true')
    parser.add_argument('--ablation', type=str, default=None,
                       choices=['no_delta', 'no_H', 'no_consensus'])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results/single_runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--signal_mode', type=str, default=None,
                        choices=['entropy', 'disagreement'],
                        help='Override adaptive.signal_mode in config (for ablation/backups).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = args.device
    cfg['partition_seed'] = args.seed
    cfg['model_seed'] = args.model_seed
    if args.rounds is not None:
        cfg['global_rounds'] = args.rounds
    if args.signal_mode is not None:
        cfg.setdefault('adaptive', {})['signal_mode'] = args.signal_mode

    run_experiment(
        cfg=cfg,
        method=args.method,
        lambda_val=args.lambda_val,
        big_lambda_val=args.big_lambda_val,
        temporal_schedule=args.temporal,
        spatial_mode=args.spatial,
        use_mobility=args.mobility,
        ablation_mode=args.ablation,
        run_name=args.run_name,
        output_dir=args.output_dir,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
