"""Extended experiment runner for the journal expansion.

A wrapper around the v4 training stack (data/, models/, train/, eval/ of the
parent project — untouched) adding:
  - fraction-based drift schedules (src/schedules.py), incl. per-cell drift
  - passive recording of the FULL signal library every round (src/signals)
  - pluggable controllers: fixed / legacy-absolute / self-calibrating
  - per-client eval traces (for p10 / fairness metrics)
  - provenance records with split + model-init hashes

Causality invariant (audited): signals at round r are computed with the
models from the END of round r-1, on probe traffic drawn from the round-r
traffic mixture; the controller sees only those signals — never labels, never
rho, never anything from rounds > r.
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
for p in (str(PROJECT_ROOT), str(JOURNAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.partition import (
    make_es_topology, nd1_partition, get_cifar10, es_neighbors_sequential,
)
from models.architectures import ModelFactory
from train.trainer import (
    build_clients_and_es, synchronize_initial_models, run_one_global_round,
    aggregate_all_es, apply_network_aggregation, update_client_models,
)
from eval.evaluator import build_per_client_test_sets, evaluate_all_clients
from src.evaluation.routing import evaluate_routing
from src.controllers.fairness import apply_fairness_aggregation
from src.controllers.normalizers import GuardedRobustNormalizer

import math
from src.schedules import get_rho
from src.signals.library import (
    RAW_SIGNAL_NAMES, RepReference, compute_client_signals, aggregate_per_es,
)
from src.controllers.self_calibrating import SelfCalController, SourceCalibratedController
from src.baselines.online import (
    PeriodicProxyGrid, UCBBandit, EXP3, GreedyLabeledOracle,
    proxy_reward_from_signals,
)
from src.provenance import RunRecord, split_hash, model_hash


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_eval_requests_r5(clients, X, Y, test_idx, main_classes, oop_c, oor_c, eth, seed, r, e, c2es_now):
    """Round 12 (--record_eval_probs): per evaluated request, both exits' softmax probabilities (float16), the
    client-exit prediction and entropy, the server-exit prediction, label, kind (0 Main, 1 OOP, 2 OOR), client,
    evaluation index, cells, and a random arrival order per client and round from its own RNG. Uses the cached test
    tensor X (no DataLoader, so no global RNG is consumed) and the same computation as the evaluator; returns the
    records and the per-client correct counts for the consistency check."""
    from src.runner_r6 import eval_client
    out = []
    for c in clients:
        idx = list(test_idx.get(c.cid, []))
        if len(idx) == 0:
            continue
        rec = {}
        res = eval_client(c, X, Y, idx, main_classes[c.cid], oop_c[c.cid], oor_c[c.cid], eth=eth, record=rec,
                          record_probs=True)
        lab = Y[torch.as_tensor(np.asarray(idx), device=Y.device)].cpu().numpy()
        kind = np.where(np.isin(lab, sorted(main_classes[c.cid])), 0,
                        np.where(np.isin(lab, sorted(oop_c[c.cid])), 1, np.where(np.isin(lab, sorted(oor_c[c.cid])), 2, 3)))
        arr = np.random.default_rng([int(seed), 10, int(r), int(c.cid)]).permutation(len(idx))
        cells = list(c2es_now[c.cid])[:2] + [-1] * (2 - len(list(c2es_now[c.cid])[:2]))
        out.append(dict(e=e, k=c.cid, n=len(idx), label=lab, kind=kind, arrival=arr, cells=cells,
                        correct=res["correct"], **{x: rec[x] for x in ("cp", "ent", "sp", "pc", "ps")}))
    return out


def _rho_key(rho):
    if isinstance(rho, dict):
        return tuple(sorted(rho.items()))
    return rho


def run_experiment_v2(cfg,
                      method="splitomcplus",
                      lambda_val=0.4,
                      big_lambda_val=0.5,
                      schedule="A",
                      spatial_mode=None,
                      controller_mode="fixed",       # fixed | legacy | selfcal
                      controller_signal="delta_hard",  # any RAW_SIGNAL_NAMES key
                      normalizer="guarded",
                      record_signals=True,
                      probe_n=64,
                      run_name="run",
                      output_dir=None,
                      eval_every=10,
                      verbose=True,
                      provenance_prefix="jx",
                      controller_kwargs=None,
                      fairness_mode=None,       # None | deployable | oracle | random
                      routing_eval=True,
                      network_cfg=None,         # {topology, signal_delay, signal_loss, participation}
                      corruption_cfg=None,      # {type, schedule} — covariate drift (Phase G)
                      mobility_cfg=None,         # {speed_level} — composition-coupled mobility (Phase H)
                      role_mode="separated",     # separated|same_role|same_role_indep|weak_server (§7)
                      fixed_Lambda=None,         # Task A1 diagnostic: freeze Lambda while lambda stays adaptive
                      disjoint_pools=False,      # Task 1: probe pool and eval pool share no sample ID
                      controller_pool_frac=0.2,
                      record_eval_probs=False,   # Round 12: per-request exit probabilities -> {run}_evalprobs.npz
                      record_train_label_hist=False,   # Round 12: per-round training label counts per cell
                      record_device_signals=False):    # Round 12 (this path): clients' cells at evaluation rounds
    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    total_rounds = cfg["global_rounds"]
    num_es = cfg["num_edge_servers"]
    adaptive_cfg = cfg.get("adaptive", {})

    rec = RunRecord(provenance_prefix, config={
        "cfg": cfg, "method": method, "lambda_val": lambda_val,
        "big_lambda_val": big_lambda_val, "schedule": schedule,
        "spatial_mode": spatial_mode, "controller_mode": controller_mode,
        "controller_signal": controller_signal, "normalizer": normalizer,
        "probe_n": probe_n, "run_name": run_name,
    })

    # ---------- data / topology / partition ----------
    set_seed(cfg.get("partition_seed", 0))
    ds_name = cfg.get("dataset", "cifar10")
    in_spatial = 8
    if ds_name == "cifar10":
        train_ds, test_ds = get_cifar10(data_root=cfg.get("data_root", str(PROJECT_ROOT / "data_cache")))
    else:
        from src.datasets_ext import get_dataset
        train_ds, test_ds, ds_meta = get_dataset(ds_name)
        cfg["num_classes"] = ds_meta["num_classes"]
        in_spatial = ds_meta["in_spatial"]
        cfg.setdefault("classes_per_client_frac", ds_meta["classes_per_client_frac"])
        if cfg["classes_per_client_frac"] * ds_meta["num_classes"] < 1:
            cfg["classes_per_client_frac"] = ds_meta["classes_per_client_frac"]
    train_labels = np.array(train_ds.targets)
    test_labels = np.array(test_ds.targets)

    # Task 1: disjoint controller (probe) / evaluation pools via masked labels.
    probe_labels, eval_labels = test_labels, test_labels
    pool_manifest = None
    _probe_used, _eval_used = set(), set()
    if disjoint_pools:
        from src.disjoint_pools import make_pool_masks
        probe_labels, eval_labels, pool_manifest = make_pool_masks(
            test_labels, cfg["num_classes"], controller_pool_frac,
            seed=cfg.get("partition_seed", 0))
        rec.set(disjoint_pools=pool_manifest)

    # covariate drift: probe AND eval batches see the round's corruption severity
    corr_state = None
    if corruption_cfg:
        from src.corruptions import CorruptedView, SeverityState
        corr_state = SeverityState()
        test_ds = CorruptedView(test_ds, corruption_cfg["type"], corr_state,
                                seed=cfg.get("partition_seed", 0))
        rec.set(corruption=corruption_cfg)

    c2es, es2c = make_es_topology(cfg["num_clients"], num_es,
                                  cfg["overlap_percentage"], seed=cfg["partition_seed"])
    partition_mode = cfg.get("partition_mode", "nd1")
    if str(partition_mode).startswith("dirichlet"):
        from src.datasets_ext import dirichlet_partition, scope_from_mains
        alpha = float(str(partition_mode).split(":")[1])
        indices, main_classes = dirichlet_partition(
            train_labels, cfg["num_classes"], cfg["num_clients"], alpha,
            seed=cfg["partition_seed"])
        scope = scope_from_mains(main_classes, c2es, num_es)
    else:
        indices, main_classes, scope = nd1_partition(
            train_labels, cfg["num_classes"], cfg["num_clients"], c2es, es2c,
            classes_per_es_frac=(cfg["classes_per_es_min_frac"], cfg["classes_per_es_max_frac"]),
            classes_per_client_frac=cfg["classes_per_client_frac"],
            seed=cfg["partition_seed"])

    per_cell_static = None
    if spatial_mode is not None:
        from scripts.run_single import make_spatial_rho
        per_cell_static = make_spatial_rho(num_es, mode=spatial_mode)

    # ---------- models ----------
    set_seed(cfg.get("model_seed", 100))
    family = cfg.get("model_family", "cnn")
    if family == "cnn" and ds_name == "cifar10":
        mf = ModelFactory(dataset=cfg["dataset"], num_classes=cfg["num_classes"])
    else:
        from src.models_ext import FlexModelFactory
        mf = FlexModelFactory(family=family, num_classes=cfg["num_classes"],
                              in_spatial=in_spatial,
                              split=cfg.get("split_point", "middle"))
    clients, edge_servers = build_clients_and_es(
        num_clients=cfg["num_clients"], num_edge_servers=num_es,
        model_factory=mf, train_dataset=train_ds, client_indices=indices,
        client_to_es=c2es, device=device, batch_size=cfg["batch_size"],
        local_epochs=cfg["local_epochs"], lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], num_workers=0)
    synchronize_initial_models(clients, edge_servers)
    for es_id, es in edge_servers.items():
        es.scope = scope[es_id]

    # §7 role-separation ablation: R3 re-inits the client aux head for max diversity.
    if role_mode == "same_role_indep":
        from src.role_ablation import reinit_client_aux
        reinit_client_aux(clients, seed_offset=777 + cfg.get("partition_seed", 0))
    if role_mode != "separated":
        rec.set(role_mode=role_mode)

    rec.set(split_hash=split_hash(indices), model_init_hash=model_hash(clients[0].client_model))

    # ---------- Round 12 records (all off by default) ----------
    r12 = dict(probs=[], cells=[], eval_rounds=[], mismatch=0) if (record_eval_probs or record_device_signals) else None
    X_eval = Y_eval = None
    if record_eval_probs:   # cached test tensor: test_ds[i] is deterministic (no augmentation) and uses no RNG
        X_eval = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
        Y_eval = torch.as_tensor(test_labels, device=device)
    TM = dict(train=0.0, eval=0.0, record=0.0) if (record_eval_probs or record_train_label_hist
                                                  or record_device_signals) else None   # Round 12 timing
    HK = None
    if record_train_label_hist:
        HK = {c.cid: int(cfg["local_epochs"]) * np.bincount(train_labels[np.asarray(indices[c.cid], dtype=np.int64)],
                                                            minlength=cfg["num_classes"]) for c in clients}
        hist_cell = np.zeros((total_rounds, num_es, cfg["num_classes"]), np.int32)
        hist_all = np.zeros((total_rounds, cfg["num_classes"]), np.int32)

    # ---------- mobility (Phase H): composition-coupled ----------
    # As a client moves and its PRIMARY es changes, its Main stays (its trained
    # classes) but its OOP/OOR sets are defined relative to the NEW primary
    # es's scope (build_per_client_test_sets uses c2es_now[cid][0]). So moving
    # changes membership, class-support exposure, AND Main/OOP/OOR composition
    # together — not just connectivity.
    mobility = None
    c2es_now = {cid: list(v) for cid, v in c2es.items()}
    if mobility_cfg:
        from network.mobility import GaussMarkovMobility
        speeds = {"slow": 3.0, "med": 12.0, "fast": 30.0}
        mobility = GaussMarkovMobility(
            num_clients=cfg["num_clients"], num_es=num_es, map_size=1000.0,
            alpha=0.9, speed=speeds.get(mobility_cfg.get("speed_level", "med"), 12.0),
            dt=10.0, seed=cfg.get("mobility_seed", 42))
        rec.set(mobility=mobility_cfg)

    # ---------- controller ----------
    net = network_cfg or {}
    if net.get("topology"):
        from src.network.topology import make_topology, graph_stats
        nbrs = make_topology(net["topology"], num_es,
                             seed=cfg.get("partition_seed", 0), round_idx=0)
        rec.set(topology=net["topology"], graph_stats=graph_stats(nbrs))
    else:
        nbrs = es_neighbors_sequential(num_es)
    ckw = controller_kwargs or {}
    lam_min = adaptive_cfg.get("lam_min", 0.15)
    lam_max = adaptive_cfg.get("lam_max", 0.7)
    Lam_min = adaptive_cfg.get("Lam_min", 0.4)
    Lam_max = adaptive_cfg.get("Lam_max", 0.7)
    controller = None
    apfl_lams, apfl_eta = None, None  # R4/B4 state (set only in apfl mode)
    bandit = None  # arm-based baselines (B5/B6/B7/B11) with select/observe API
    es_ids = list(range(num_es))
    if controller_mode == "proxy_grid":
        bandit = PeriodicProxyGrid(es_ids)
    elif controller_mode == "ucb":
        bandit = UCBBandit(es_ids)
    elif controller_mode == "exp3":
        bandit = EXP3(es_ids, seed=cfg.get("partition_seed", 0))
    elif controller_mode == "oracle_greedy":
        bandit = GreedyLabeledOracle(es_ids, lam0=lambda_val,
                                     lo=lam_min, hi=lam_max)
    elif controller_mode == "selfcal":
        controller = SelfCalController(
            neighbors=nbrs, lam_min=lam_min, lam_max=lam_max,
            Lam_min=Lam_min, Lam_max=Lam_max,
            warmup=ckw.get("warmup", 15), normalizer=normalizer,
            consensus_steps=adaptive_cfg.get("consensus_steps", 1),
            max_step=ckw.get("max_step"), burn_in=ckw.get("burn_in", 0),
            z_guard=ckw.get("z_guard"), spatial_norm=ckw.get("spatial_norm", False),
            abs_cap=ckw.get("abs_cap", False),
            abs_only=ckw.get("abs_only", False),   # R4/B1
            neighbor_avg=not ckw.get("no_neighbor_avg", False),   # R6 P0
            signal_range=__import__("src.controllers.self_calibrating",
                                    fromlist=["SIGNAL_RANGE"]).SIGNAL_RANGE.get(
                                        controller_signal, 1.0))
    elif controller_mode == "apfl":
        # R4/B4: per-client lambda learned from LOCAL labeled loss (APFL-style).
        # No probe / no signal; Lambda fixed (big_lambda_val). controller stays None.
        apfl_lams = {c.cid: 0.5 * (lam_min + lam_max) for c in clients}
        apfl_eta = float(ckw.get("apfl_eta", cfg.get("learning_rate", 0.01)))
    elif controller_mode == "legacy":
        if controller_signal == "delta_hard":
            mu, tau = adaptive_cfg.get("mu_drift", 0.31), adaptive_cfg.get("tau_drift", 0.045)
        else:
            mu, tau = adaptive_cfg.get("mu_H", 2.0), adaptive_cfg.get("tau_H", 0.5)
        mu = ckw.get("mu", mu)
        tau = ckw.get("tau", tau)
        controller = SourceCalibratedController(
            neighbors=nbrs, mu=mu, tau=tau, lam_min=lam_min, lam_max=lam_max,
            Lam_min=Lam_min, Lam_max=Lam_max,
            consensus_steps=adaptive_cfg.get("consensus_steps", 1))

    need_signals = record_signals or controller is not None or bandit is not None \
        or fairness_mode is not None
    oracle_eval_every_round = controller_mode == "oracle_greedy"
    track_update_norm = controller_signal == "update_norm" or record_signals
    prev_client_state = {}  # cid -> flat tensor, for update-norm signal
    # fairness state: risk normalizer per ES on delta_hard (independent of the
    # lambda controller, so fairness composes with ANY controller_mode)
    fair_rng = np.random.default_rng(cfg.get("partition_seed", 0) + 999)
    fair_risk_norm = {z: GuardedRobustNormalizer(warmup=15) for z in es_ids} \
        if fairness_mode else {}
    # network impairments (Phase 6)
    sig_channel = None
    participation = None
    if net.get("signal_delay") or net.get("signal_loss"):
        from src.network.impairments import SignalChannel
        sig_channel = SignalChannel(delay=net.get("signal_delay", 0),
                                    loss=net.get("signal_loss", 0.0),
                                    seed=cfg.get("partition_seed", 0))
    if net.get("participation", 1.0) < 1.0:
        from src.network.impairments import ParticipationSampler
        participation = ParticipationSampler(net["participation"],
                                             seed=cfg.get("partition_seed", 0))

    # ---------- probe + reference state ----------
    probe_rng = np.random.default_rng(int(cfg.get("partition_seed", 0)) * 31 + 17)
    rep_refs = {c.cid: RepReference(warmup_rounds=15) for c in clients} if need_signals else {}
    _probe_key = object()
    _probe_pool = None

    # fixed lambdas: scalar or per-cell (mean-matched decomposition baselines)
    if isinstance(lambda_val, dict):
        fixed_lams = {z: lambda_val.get(z, 0.4) for z in range(num_es)}
    else:
        fixed_lams = {z: lambda_val for z in range(num_es)}
    if isinstance(big_lambda_val, dict):
        fixed_Lams = {z: big_lambda_val.get(z, 0.5) for z in range(num_es)}
    else:
        fixed_Lams = {z: big_lambda_val for z in range(num_es)}

    # trajectory replay (analysis-only interventions; may use future info by design)
    replay_lam, replay_Lam = None, None
    if controller_mode == "replay":
        from src.replay import load_transformed_trajectory
        replay_lam, replay_Lam = load_transformed_trajectory(
            ckw["replay_from"], ckw.get("transform", "identity"),
            seed=cfg.get("partition_seed", 0), num_es=num_es)
        rec.set(replay_from=str(ckw["replay_from"]),
                replay_transform=ckw.get("transform", "identity"))

    history = {
        "round": [], "mean_loss": [], "lamdas": [], "big_lamdas": [],
        "rho_trace": [], "eval": [], "controller_z": [],
        "signals_per_es": {name: [] for name in RAW_SIGNAL_NAMES} if need_signals else {},
        "update_norm_per_es": [],
    }
    update_norm_per_es = {}  # from the PREVIOUS round's training (causal)
    per_client_sig = [] if need_signals else None  # [T, n_clients, n_signals]
    client_ids = [c.cid for c in clients]

    if verbose:
        print(f"[{run_name}] method={method} controller={controller_mode} "
              f"signal={controller_signal} schedule={schedule} rounds={total_rounds} device={device}")

    t0 = time.time()
    for r in range(1, total_rounds + 1):
        # -- 1. current traffic mixture (environment side; NOT visible to controller)
        if corr_state is not None:
            from src.corruptions import severity_schedule
            corr_state.severity = severity_schedule(
                corruption_cfg["schedule"], r, total_rounds)
            history.setdefault("corruption_severity", []).append(corr_state.severity)
        if mobility is not None:
            mobility.step()
            if r % 5 == 0:
                new_c2es = mobility.get_topk_es(k=2)
                by_cid = {c.cid: c for c in clients}
                changed = sum(1 for cid, c in by_cid.items()
                              if set(new_c2es[cid]) != set(c.edge_server_ids))
                if changed:
                    from network.mobility import rewire_clients
                    rewire_clients(clients, edge_servers, new_c2es, mf)
                    c2es_now = {cid: list(new_c2es[cid]) for cid in range(cfg["num_clients"])}
                    _probe_key = object()  # force probe/eval set rebuild
                    history.setdefault("mobility_changes", []).append((r, changed))
        rho_now = get_rho(schedule, r, total_rounds,
                          seed=cfg.get("partition_seed", 0), num_cells=num_es)
        if rho_now is None:
            rho_now = cfg.get("oop_ratio_default", 0.4)
        per_cell_now = per_cell_static
        rho_scalar = rho_now
        if isinstance(rho_now, dict):
            per_cell_now, rho_scalar = rho_now, 0.0

        key = (_rho_key(rho_now), _rho_key(per_cell_static))
        if key != _probe_key:
            _probe_pool, _, _ = build_per_client_test_sets(
                probe_labels, cfg["num_clients"], main_classes, c2es_now, scope,
                oop_ratio=rho_scalar,
                oor_ratio_factor=cfg.get("oor_ratio_factor", 0.3),
                per_cell_oop_ratio=per_cell_now)
            _probe_key = key

        # -- 2. signals from END-OF-(r-1) models on round-r traffic probes
        sig_per_es = {}
        rep_centroids = None
        # Task 2: also compute server non-Main signals if the controller uses one
        # (or always record when recording signals). Main classes are TRAINING
        # metadata (not probe/eval labels).
        from src.signals.library import SERVER_NONMAIN_NAMES
        want_nonmain = record_signals or controller_signal in SERVER_NONMAIN_NAMES
        if need_signals:
            per_client = {name: {} for name in RAW_SIGNAL_NAMES}
            for name in SERVER_NONMAIN_NAMES:
                per_client[name] = {}
            per_client["ent_client_norm"] = {}   # R4/B2: H(p)/ln C in [0,1]
            _lnC = math.log(cfg["num_classes"])
            row = np.zeros((len(clients), len(RAW_SIGNAL_NAMES)), dtype=np.float32)
            rep_means = {}
            for ci, c in enumerate(clients):
                pool = _probe_pool.get(c.cid, [])
                if len(pool) == 0:
                    continue
                k = min(probe_n, len(pool))
                chosen = probe_rng.choice(pool, size=k, replace=False)
                if disjoint_pools:
                    _probe_used.update(int(i) for i in chosen)
                probe_x = torch.stack([test_ds[int(i)][0] for i in chosen])
                mc = set(main_classes.get(c.cid, set())) if want_nonmain else None
                sig, rep_mean = compute_client_signals(
                    c.client_model, list(c.server_models.values()),
                    probe_x, device, rep_refs.get(c.cid), return_rep=True,
                    main_classes=mc)
                if rep_mean is not None:
                    rep_means[c.cid] = rep_mean
                for si, name in enumerate(RAW_SIGNAL_NAMES):
                    per_client[name][c.cid] = sig[name]
                    row[ci, si] = sig[name]
                per_client["ent_client_norm"][c.cid] = float(sig["ent_client"]) / _lnC
                if want_nonmain:
                    for name in SERVER_NONMAIN_NAMES:
                        per_client[name][c.cid] = sig[name]
            per_client_sig.append(row)
            for name in RAW_SIGNAL_NAMES:
                sig_per_es[name] = aggregate_per_es(per_client[name], clients)
                history["signals_per_es"][name].append(sig_per_es[name])
            sig_per_es["ent_client_norm"] = aggregate_per_es(per_client["ent_client_norm"], clients)
            history["signals_per_es"].setdefault("ent_client_norm", []).append(sig_per_es["ent_client_norm"])
            if want_nonmain:
                for name in SERVER_NONMAIN_NAMES:
                    sig_per_es[name] = aggregate_per_es(per_client[name], clients)
                    history["signals_per_es"].setdefault(name, []).append(sig_per_es[name])
            if fairness_mode and rep_means:
                rep_centroids = {}
                per_es_lists = {}
                for c in clients:
                    if c.cid in rep_means:
                        for z in c.edge_server_ids:
                            per_es_lists.setdefault(z, []).append(rep_means[c.cid])
                rep_centroids = {z: np.mean(v, axis=0) for z, v in per_es_lists.items()}

        if need_signals and update_norm_per_es:
            sig_per_es["update_norm"] = dict(update_norm_per_es)

        # dynamic topology: refresh the consensus graph
        if net.get("topology") == "dynamic":
            from src.network.topology import make_topology
            new_nbrs = make_topology("dynamic", num_es,
                                     seed=cfg.get("partition_seed", 0), round_idx=r)
            if controller is not None:
                controller.neighbors = new_nbrs
        # signal channel: what the controller RECEIVES (delayed/lossy/stale)
        if sig_channel is not None and sig_per_es.get(controller_signal):
            delivered, ages = sig_channel.send_and_receive(sig_per_es[controller_signal])
            sig_per_es[controller_signal] = delivered
            history.setdefault("signal_age", []).append(ages)

        # -- 3. controller
        if replay_lam is not None:
            idx = min(r - 1, len(replay_lam) - 1)
            cur_lams = dict(replay_lam[idx])
            cur_Lams = dict(replay_Lam[idx])
            history["controller_z"].append({})
        elif bandit is not None:
            # attribute reward for the arm played in round r-1 (observed via
            # this round's pre-training probe), then select the next arm
            if sig_per_es.get("delta_hard") and not isinstance(bandit, GreedyLabeledOracle):
                bandit.observe(proxy_reward_from_signals(sig_per_es))
            cur_lams = bandit.select()
            cur_Lams = dict(fixed_Lams)
            history["controller_z"].append({})
        elif controller is not None and sig_per_es.get(controller_signal):
            cur_lams, cur_Lams = controller.step(sig_per_es[controller_signal])
            if fixed_Lambda is not None:  # Task A1: freeze Lambda, keep lambda adaptive
                cur_Lams = {z: fixed_Lambda for z in cur_Lams}
            history["controller_z"].append(dict(getattr(controller, "last_z", {})))
            # R4/A2: exact relative/absolute candidates (post-warm-up only)
            history.setdefault("lam_rel", []).append(dict(getattr(controller, "last_lam_rel", {})))
            history.setdefault("lam_abs", []).append(dict(getattr(controller, "last_lam_abs", {})))
        else:
            cur_lams, cur_Lams = fixed_lams, fixed_Lams
            history["controller_z"].append({})

        # -- 4. one global round
        _tt0 = time.perf_counter()
        active_clients = clients
        _saved_attach = None
        if participation is not None:
            active_clients = participation.sample(clients)
            active_set = {c.cid for c in active_clients}
            _saved_attach = {z: edge_servers[z].clients for z in edge_servers}
            for z in edge_servers:
                edge_servers[z].clients = [c for c in _saved_attach[z]
                                           if c.cid in active_set] or _saved_attach[z]
        if HK is not None:   # Round 12: labels the server block of each cell trains on in this round
            act_ids = {c.cid for c in active_clients}
            for z in edge_servers:
                for c in edge_servers[z].clients:
                    if c.cid in act_ids:
                        hist_cell[r - 1, z] += HK[c.cid]
            for c in active_clients:
                hist_all[r - 1] += HK[c.cid]
        if fairness_mode is not None:
            # same sequence as run_one_global_round, but inter-ES mixing is the
            # donor-selected, risk-directed aggregation (F1/F2)
            # risk = max(temporal z, cross-cell spatial z): static spatial
            # heterogeneity never "changes", so temporal z alone is blind to it
            risk_z = {}
            d_per_es = sig_per_es.get("delta_hard", {})
            from src.controllers.self_calibrating import spatial_z as _spz
            sp = _spz(d_per_es) if len(d_per_es) >= 3 else {}
            for z in es_ids:
                if z in d_per_es:
                    zt = fair_risk_norm[z].update(d_per_es[z])
                    risk_z[z] = max(zt, sp.get(z, 0.0))
            losses = [c.train_one_round(method="splitomcplus",
                                        gamma=cfg.get("gamma", 0.5))
                      for c in active_clients]
            mean_loss = float(np.mean(losses))
            aggregate_all_es(edge_servers)
            oracle_rho = None
            if fairness_mode == "oracle":
                oracle_rho = (dict(rho_now) if isinstance(rho_now, dict)
                              else ({z: per_cell_now[z] for z in es_ids}
                                    if per_cell_now else {z: rho_scalar for z in es_ids}))
            lam_eff = apply_fairness_aggregation(
                edge_servers, cur_Lams, risk_z,
                rep_centroids=rep_centroids,
                conf_per_es=sig_per_es.get("conf_server"),
                num_classes=cfg["num_classes"], mode=fairness_mode,
                rng=fair_rng, oracle_rho=oracle_rho)
            history.setdefault("fair_lam_eff", []).append(lam_eff)
            history.setdefault("fair_risk_z", []).append(risk_z)
            update_client_models(active_clients, edge_servers, cur_lams,
                                 method="splitomcplus")
        elif controller_mode == "apfl":
            # R4/B4: APFL-style per-client lambda from local labeled loss.
            from src.apfl_baseline import run_apfl_round
            mean_loss, apfl_lams = run_apfl_round(
                active_clients, edge_servers, apfl_lams, eta=apfl_eta,
                big_lambdas=cur_Lams, gamma=cfg.get("gamma", 0.5),
                lam_min=lam_min, lam_max=lam_max)
            history.setdefault("apfl_client_lams", []).append(dict(apfl_lams))
            cur_lams = {z: float(np.mean([apfl_lams[c.cid] for c in edge_servers[z].clients]))
                        for z in edge_servers}
        elif role_mode != "separated":
            from src.role_ablation import run_role_round
            mean_loss = run_role_round(active_clients, edge_servers, cur_lams,
                                       cur_Lams, role_mode, gamma=cfg.get("gamma", 0.5))
        else:
            mean_loss = run_one_global_round(
                clients=active_clients, edge_servers_dict=edge_servers,
                lamdas=cur_lams, big_lambdas=cur_Lams,
                method="splitomcplus" if method in ("adaptive_splitomc",) else method,
                gamma=cfg.get("gamma", 0.5))
        if _saved_attach is not None:
            for z in edge_servers:
                edge_servers[z].clients = _saved_attach[z]

        if TM is not None:
            TM["train"] += time.perf_counter() - _tt0
        history["round"].append(r)
        history["mean_loss"].append(float(mean_loss))
        history["lamdas"].append(dict(cur_lams))
        history["big_lamdas"].append(dict(cur_Lams))
        history["rho_trace"].append(rho_now if not isinstance(rho_now, dict)
                                    else dict(rho_now))

        # -- 4b. update-norm signal (B10): ||w_r - w_{r-1}|| of the client
        # block AFTER this round; consumed by the controller NEXT round.
        if track_update_norm:
            per_client_norm = {}
            for c in clients:
                flat = torch.cat([v.flatten().float()
                                  for _, v in sorted(c.client_model.state_dict().items())]).cpu()
                if c.cid in prev_client_state:
                    per_client_norm[c.cid] = float(
                        torch.norm(flat - prev_client_state[c.cid]).item())
                prev_client_state[c.cid] = flat
            update_norm_per_es = aggregate_per_es(per_client_norm, clients) \
                if per_client_norm else {}
            history["update_norm_per_es"].append(dict(update_norm_per_es))

        # -- 5. eval
        if r == 1 or r % eval_every == 0 or r == total_rounds or oracle_eval_every_round:
            _te0 = time.perf_counter()
            test_idx, oop_c, oor_c = build_per_client_test_sets(
                eval_labels, cfg["num_clients"], main_classes, c2es_now, scope,
                oop_ratio=rho_scalar,
                oor_ratio_factor=cfg.get("oor_ratio_factor", 0.3),
                per_cell_oop_ratio=per_cell_now)
            if disjoint_pools:
                for ids in test_idx.values():
                    _eval_used.update(int(i) for i in ids)
            agg, per_client_res = evaluate_all_clients(
                clients=clients, test_dataset=test_ds,
                per_client_test_indices=test_idx,
                client_main_classes=main_classes,
                oop_classes_per_client=oop_c, oor_classes_per_client=oor_c,
                eth=cfg.get("eth_default", 0.8))
            agg["round"] = r
            agg["rho"] = rho_now if not isinstance(rho_now, dict) else dict(rho_now)
            agg["per_client_acc"] = {p["cid"]: p["acc_total"] for p in per_client_res}
            if routing_eval:
                agg["routing"] = evaluate_routing(clients, test_ds, test_idx,
                                                  eth=cfg.get("eth_default", 0.8))
            history["eval"].append(agg)
            if TM is not None:
                TM["eval"] += time.perf_counter() - _te0
            if r12 is not None:
                e_idx = len(r12["eval_rounds"])
                r12["eval_rounds"].append(r)
                r12["cells"].append([list(c2es_now[c.cid])[:2] + [-1] * (2 - len(list(c2es_now[c.cid])[:2]))
                                     for c in clients])
                if record_eval_probs:
                    _tr0 = time.perf_counter()
                    recs = record_eval_requests_r5(clients, X_eval, Y_eval, test_idx, main_classes, oop_c, oor_c,
                                                   cfg.get("eth_default", 0.8), cfg.get("partition_seed", 0), r, e_idx, c2es_now)
                    acc_ev = {p["cid"]: p["acc_total"] for p in per_client_res}
                    r12["mismatch"] += sum(1 for m in recs if abs(m["correct"] / m["n"] - acc_ev.get(m["k"], -1)) > 1e-12)
                    r12["probs"].extend(recs)
                    TM["record"] += time.perf_counter() - _tr0
            if isinstance(bandit, GreedyLabeledOracle):
                bandit.observe({int(k): v for k, v in agg["cell_means"].items()})
            if verbose:
                el = time.time() - t0
                print(f"  R{r:3d} loss={mean_loss:.3f} acc={agg['acc_total']:.3f} "
                      f"worst={agg.get('worst_cell_acc', 0):.3f} "
                      f"elapsed={el/60:.1f}m ETA={(total_rounds-r)*el/r/60:.0f}m", flush=True)

    history["total_time_sec"] = time.time() - t0
    if TM is not None:   # Round 12: training, evaluation and per-request recording in seconds
        history["r12_timing_sec"] = dict(TM)
    history["client_ids"] = client_ids
    if disjoint_pools:
        overlap = len(_probe_used & _eval_used)
        history["pool_manifest"] = pool_manifest
        history["probe_eval_overlap"] = overlap
        history["probe_used_count"] = len(_probe_used)
        history["eval_used_count"] = len(_eval_used)
        assert overlap == 0, f"probe/eval sample overlap = {overlap} (must be 0)"
    history["config"] = {
        "method": method, "lambda_val": lambda_val, "big_lambda_val": big_lambda_val,
        "schedule": schedule, "spatial_mode": spatial_mode,
        "controller_mode": controller_mode, "controller_signal": controller_signal,
        "normalizer": normalizer, "probe_n": probe_n,
        "partition_seed": cfg.get("partition_seed"), "model_seed": cfg.get("model_seed"),
        "run_id": rec.run_id,
    }
    if ckw.get("no_neighbor_avg"):
        history["config"]["no_neighbor_avg"] = True
    if record_eval_probs or record_train_label_hist or record_device_signals:
        history["config"].update(record_eval_probs=record_eval_probs, record_train_label_hist=record_train_label_hist,
                                 record_device_signals=record_device_signals)
    if record_eval_probs:   # client-rounds whose recorded accuracy differs from the evaluator's
        history["r12_eval_record_mismatch"] = int(r12["mismatch"])

    # ---------- save ----------
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        jpath = out / f"{run_name}.json"
        with open(jpath, "w") as f:
            json.dump(history, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        if need_signals and per_client_sig:
            np.savez_compressed(
                out / f"{run_name}_signals.npz",
                signals=np.stack(per_client_sig),       # [T, n_clients, n_signals]
                signal_names=np.array(RAW_SIGNAL_NAMES),
                client_ids=np.array(client_ids),
                primary_es=np.array([c.edge_server_ids[0] for c in clients]))
        if record_eval_probs and r12["probs"]:
            P = r12["probs"]
            ns = [m["n"] for m in P]
            np.savez_compressed(out / f"{run_name}_evalprobs.npz", eval_rounds=np.array(r12["eval_rounds"]),
                                req_eval_index=np.repeat([m["e"] for m in P], ns).astype(np.uint8),
                                req_client=np.repeat([m["k"] for m in P], ns).astype(np.int16),
                                req_home=np.full(sum(ns), -1, np.int8),   # no home in this scenario family
                                req_label=np.concatenate([m["label"] for m in P]).astype(np.int16),
                                req_kind=np.concatenate([m["kind"] for m in P]).astype(np.int8),
                                req_order=np.concatenate([np.arange(m["n"]) for m in P]).astype(np.int32),
                                req_arrival=np.concatenate([m["arrival"] for m in P]).astype(np.int32),
                                cp=np.concatenate([m["cp"] for m in P]).astype(np.int16),
                                ent=np.concatenate([m["ent"] for m in P]).astype(np.float32),
                                sp=np.concatenate([m["sp"] for m in P]).astype(np.int16),
                                pc=np.concatenate([m["pc"] for m in P]).astype(np.float16),
                                ps=np.concatenate([m["ps"] for m in P]).astype(np.float16),
                                arrival_rng=np.array("numpy default_rng([partition_seed, 10, round, client]).permutation(n)"))
        if r12 is not None or HK is not None:
            extra = {}
            if r12 is not None:
                extra.update(eval_rounds=np.array(r12["eval_rounds"]), eval_cells=np.array(r12["cells"], np.int16),
                             client_ids=np.array(client_ids))
            if HK is not None:
                extra.update(train_hist_cell=hist_cell, train_hist_all=hist_all)
            np.savez_compressed(out / f"{run_name}_rec.npz", **extra)
        if verbose:
            print(f"  Saved: {jpath}")

    final = history["eval"][-1] if history["eval"] else {}
    rec.finish(metrics={
        "acc_total_final": final.get("acc_total"),
        "worst_cell_final": final.get("worst_cell_acc"),
        "integrated_acc": float(np.mean([e["acc_total"] for e in history["eval"]]))
        if history["eval"] else None,
        "wall_min": history["total_time_sec"] / 60.0,
    })
    return history
