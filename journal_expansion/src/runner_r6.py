"""Round 6 runner: environment-driven membership, per-client rho, Poisson request
counts, participation, signal loss/delay (R6 directive §2-§5).

Reuses the existing stack unchanged: nd1_partition, ModelFactory, SplitOMCClient /
EdgeServer, rewire_clients (client mobility code path: a client joining a cell gets
a server block warm-started from that cell's current average, a client leaving a
cell drops that server block), update_client_models (lambda mixing, server block =
cell average), APFL's _grad_wrt_lambda, compute_client_signals, make_pool_masks.

Round r (1-indexed), environment state read from runs/phaseT6_env/*.npz:
  1. membership Z_k^r -> rewire participating clients whose cells changed
     (a non-participating client receives nothing, so it keeps its old blocks)
  2. signals (DriftGate / entropy arms): participating client k with
     n = min(N_k^r, 64) > 0 draws n probe requests from its request pool and
     sends d_k to each of its edges (a message can be lost, §S4); edge d-bar =
     mean of received values; with delay d the controller uses the d-bar of r-d
  3. controller (EdgeDriftGate) -> lambda, Lambda per cell
  4. participating clients train; each cell with >= 1 participating member
     averages them, cells mix with the global average of the active cells (Lambda),
     participating clients take lambda-mixed client blocks and cell server blocks.
     An empty cell does not train and keeps its state.
  5. evaluation rounds {1, 5, 10, ..., 150}: every client, with its current model,
     on the evaluation request set built from its current cells and rho_k^r.
With all clients participating and no empty cell, step 4 is exactly
run_one_global_round (tested in tests/test_r6.py).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.partition import get_cifar10
from eval.evaluator import compute_entropy
from models.architectures import ModelFactory
from network.mobility import rewire_clients
from train.trainer import (build_clients_and_es, mix_state_dicts, synchronize_initial_models,
                           uniform_average, update_client_models)
from src.apfl_baseline import _grad_wrt_lambda
from src.disjoint_pools import make_pool_masks
from src.provenance import RunRecord, model_hash, split_hash
from src.r6_controller import DeviceDriftGate, EdgeDriftGate
from src.r6_env import load_env
from src.r6_requests import RequestBuilder, build_partition
from src.signals.library import compute_client_signals


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.indices[i]


class CachedLoader:
    """Drop-in for the client's DataLoader(Subset(train, idx), batch_size, shuffle=True,
    drop_last=False, num_workers=0): the SAME DataLoader/RandomSampler machinery runs
    over the index list (so batch order and global-RNG consumption are unchanged) and
    each batch is gathered from a cached, already-transformed train tensor (the train
    transform is deterministic: ToTensor + Normalize, no augmentation). Bit-identical
    training is tested in tests/test_r6.py."""

    def __init__(self, indices, X, Y, batch_size):
        self.idx_loader = torch.utils.data.DataLoader(_IndexDataset(indices), batch_size=batch_size,
                                                      shuffle=True, num_workers=0, drop_last=False)
        self.X, self.Y = X, Y

    def __iter__(self):
        for g in self.idx_loader:
            g = g.to(self.X.device)
            yield self.X[g], self.Y[g]

    def __len__(self):
        return len(self.idx_loader)


def use_cached_loaders(clients, indices, train_ds, device, batch_size):
    X = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)
    Y = torch.as_tensor(np.array(train_ds.targets), device=device)
    for c in clients:
        c.train_loader = CachedLoader(indices[c.cid], X, Y, batch_size)
    return X, Y


def client_label_hist(indices, train_labels, num_classes, local_epochs):
    """Round 10: label counts of the samples one client trains on in one round. CachedLoader passes over all of the
    client's indices once per local epoch (shuffle, drop_last=False), and every batch trains the client block and the
    server block of every cell the client belongs to."""
    return local_epochs * np.bincount(np.asarray(train_labels)[np.asarray(indices, dtype=np.int64)], minlength=num_classes)


def eval_rounds_for(total_rounds, every=5):
    return sorted({1} | set(range(every, total_rounds + 1, every)) | {total_rounds})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_client(client, X, Y, idx, main, oop, oor, eth=0.8, batch_size=128, mainaware=False, record=None,
                record_probs=False):
    """Same computation as eval.evaluator.evaluate_one_client (client exit if the
    client-exit entropy <= eth, else the mean of the client's server-exit logits),
    on a cached normalized test tensor, plus offload counts per request kind.
    mainaware (Round 8, reference only): also count the rule "server exit if its prediction is outside
    the client's Main classes, otherwise the entropy rule" (ma_* keys); the other keys are unchanged.
    record (Round 8 v2): a dict that receives, for every request in idx order, the client-exit prediction "cp",
    the client-exit entropy "ent" (float32, the value compared with eth) and the server-exit prediction "sp".
    record_probs (Round 10, with record): also the softmax probabilities of the client exit "pc" and of the server exit
    "ps" (after averaging the server logits of the client's cells), float16, in idx order."""
    client.client_model.eval()
    for sm in client.server_models.values():
        sm.eval()
    out = dict(n_total=0, correct=0, n_off=0,
               n_main=0, c_main=0, off_main=0,
               n_nonmain=0, c_nonmain=0, off_nonmain=0,
               n_oop=0, c_oop=0, n_oor=0, c_oor=0)
    if mainaware:
        out.update(ma_correct=0, ma_n_off=0, ma_c_main=0, ma_c_nonmain=0)
    if len(idx) == 0:
        return out
    idx_t = torch.as_tensor(np.asarray(idx), device=X.device, dtype=torch.long)
    main_t = torch.as_tensor(sorted(main), device=X.device, dtype=torch.long)
    oop_t = torch.as_tensor(sorted(oop), device=X.device, dtype=torch.long)
    oor_t = torch.as_tensor(sorted(oor), device=X.device, dtype=torch.long)
    sms = list(client.server_models.values())
    rec = {"cp": [], "ent": [], "sp": []} if record is not None else None
    if rec is not None and record_probs:
        rec.update(pc=[], ps=[])
    for s in range(0, len(idx_t), batch_size):
        b = idx_t[s:s + batch_size]
        x, y = X[b], Y[b]
        client_logits, rep = client.client_model(x)
        entropy = compute_entropy(client_logits)
        _, client_preds = client_logits.max(dim=1)
        server_logits = sum([sm(rep)[0] for sm in sms]) / len(sms)
        _, server_preds = server_logits.max(dim=1)
        route = entropy > eth
        if rec is not None:
            rec["cp"].append(client_preds.cpu())
            rec["ent"].append(entropy.float().cpu())
            rec["sp"].append(server_preds.cpu())
            if record_probs:
                rec["pc"].append(torch.softmax(client_logits.float(), dim=1).half().cpu())
                rec["ps"].append(torch.softmax(server_logits.float(), dim=1).half().cpu())
        final = torch.where(route, server_preds, client_preds)
        ok = final == y
        is_main = torch.isin(y, main_t)
        is_oop = torch.isin(y, oop_t)
        is_oor = torch.isin(y, oor_t)
        nonmain = is_oop | is_oor
        out["n_total"] += int(y.numel())
        out["correct"] += int(ok.sum())
        out["n_off"] += int(route.sum())
        out["n_main"] += int(is_main.sum())
        out["c_main"] += int((ok & is_main).sum())
        out["off_main"] += int((route & is_main).sum())
        out["n_nonmain"] += int(nonmain.sum())
        out["c_nonmain"] += int((ok & nonmain).sum())
        out["off_nonmain"] += int((route & nonmain).sum())
        out["n_oop"] += int(is_oop.sum())
        out["c_oop"] += int((ok & is_oop).sum())
        out["n_oor"] += int(is_oor.sum())
        out["c_oor"] += int((ok & is_oor).sum())
        if mainaware:
            ma_route = route | ~torch.isin(server_preds, main_t)
            ma_ok = torch.where(ma_route, server_preds, client_preds) == y
            out["ma_correct"] += int(ma_ok.sum())
            out["ma_n_off"] += int(ma_route.sum())
            out["ma_c_main"] += int((ma_ok & is_main).sum())
            out["ma_c_nonmain"] += int((ma_ok & nonmain).sum())
    if rec is not None:
        record.update({k: torch.cat(v).numpy() for k, v in rec.items()})
    return out


def _ratio(a, b):
    return a / b if b > 0 else float("nan")


def r6_training_round(clients, active, ES, lamdas, Lams, mode, gamma, apfl=None, client_lams=None, capture=None):
    """Step 4. active: bool array over client ids. Returns (mean_loss, active_cells).
    client_lams (Round 7 gate): {client: lambda_k} -> each participating client mixes with its own lambda_k
    instead of the mean of its cells' lambdas (same mixing and server-block update otherwise).
    capture (Round 8): a dict that receives {client: (theta_local, cell average used for mixing)} for every
    participating client; theta_local is the client-side state right after local training (CPU copy)."""
    act = [c for c in clients if active[c.cid]]
    losses = [c.train_one_round(method="splitomcplus", gamma=gamma) for c in act]
    active_cells = []
    for z, es in ES.items():
        mem = [c for c in es.clients if active[c.cid]]
        if not mem:
            continue
        es.clients_avg_weights = uniform_average([c.get_client_state() for c in mem])
        es.server_avg_weights = uniform_average([c.get_server_state(z) for c in mem])
        active_cells.append(z)
    if active_cells:
        global_c = uniform_average([ES[z].clients_avg_weights for z in active_cells])
        global_s = uniform_average([ES[z].server_avg_weights for z in active_cells])
        for z in active_cells:
            Lam = Lams.get(z, 0.5)
            ES[z].clients_avg_weights = mix_state_dicts(ES[z].clients_avg_weights, global_c, Lam)
            ES[z].server_avg_weights = mix_state_dicts(ES[z].server_avg_weights, global_s, Lam)
    if capture is not None:   # same average as train/trainer.update_client_models; nothing is modified
        for c in act:
            if len(c.edge_server_ids) == 1:
                avg = ES[c.edge_server_ids[0]].clients_avg_weights
            else:
                avg = uniform_average([ES[e].clients_avg_weights for e in c.edge_server_ids])
            capture[c.cid] = (c.get_client_state(), avg)
    if mode == "apfl":
        lams, eta, lo, hi = apfl["lams"], apfl["eta"], apfl["lam_min"], apfl["lam_max"]
        for c in act:   # same update as src/apfl_baseline.run_apfl_round
            if len(c.edge_server_ids) == 1:
                avg = ES[c.edge_server_ids[0]].clients_avg_weights
            else:
                avg = uniform_average([ES[e].clients_avg_weights for e in c.edge_server_ids])
            local = c.get_client_state()
            lam = float(lams[c.cid])
            g, _ = _grad_wrt_lambda(c, local, avg, lam, gamma)
            lam_new = float(min(max(lam - eta * g, lo), hi))
            lams[c.cid] = lam_new
            c.set_client_state(mix_state_dicts(local, avg, lam_new))
            for e in c.edge_server_ids:
                c.set_server_state(e, ES[e].server_avg_weights)
    elif client_lams is not None:
        for c in act:   # same as train/trainer.update_client_models, lambda = the client's own lambda_k
            if len(c.edge_server_ids) == 1:
                avg = ES[c.edge_server_ids[0]].clients_avg_weights
            else:
                avg = uniform_average([ES[e].clients_avg_weights for e in c.edge_server_ids])
            c.set_client_state(mix_state_dicts(c.get_client_state(), avg, float(client_lams[c.cid])))
            for e in c.edge_server_ids:
                c.set_server_state(e, ES[e].server_avg_weights)
    else:
        update_client_models(act, ES, lamdas, method="splitomcplus")
    return (float(np.mean(losses)) if losses else float("nan")), active_cells


def run_r6(cfg, env_path, mode, signal="tv_dist", lambda_val=0.4, big_lambda_val=0.5,
           apfl_eta=None, signal_delay=0, probe_n=64, controller_kwargs=None,
           run_name="run", output_dir=None, eval_every=5, save_models=False, verbose=True,
           scenario=None, arm=None, record_device_signals=False, device_signal=None,
           oracle_home_away=False, fixed_Lambda=None, eval_infer_lambdas=None, eval_mainaware_route=False,
           record_eval_requests=False, record_probe_values=False, record_eval_probs=False,
           record_train_label_hist=False):
    """Round 7 gate options (all off by default -> Round 6 behaviour):
    record_device_signals: every participating client computes x_TV and x_SR on its probe in every arm
        (no grad; the probe uses the numpy probe RNG only, so training is unchanged);
    mode="device" + device_signal in {"tv", "sr"}: one lambda per client (DeviceDriftGate);
    mode="oracle" (oracle_home_away): lambda_k = 0.70 at home, 0.15 away (env at_home), every round;
    fixed_Lambda: Lambda fixed for every cell (also overrides the cell controller's Lambda).
    Round 8 options (off by default):
    eval_infer_lambdas: at every evaluation round each participating client is also evaluated with the
        client-side model lambda_inf * theta_local + (1 - lambda_inf) * cell average, for every lambda_inf in
        the list (same requests, server blocks, exits and routing); the installed model is restored after;
    eval_mainaware_route: also count the Main-aware routing rule on the installed model (reference only).
    Round 8 v2 options (off by default):
    record_eval_requests (needs eval_infer_lambdas): for every evaluated request of every participating client and
        evaluation round, keep the client-exit prediction and entropy and the server-exit prediction for every
        lambda_inf block, plus the installed model when lambda_t is not one of the lambda_inf values; with the
        request's client, evaluation round, label, kind (Main / OOP / OOR) and at_home -> {run}_requests.npz;
    record_probe_values (needs record_device_signals): per probe request TV and server-non-Main indicator
        (probe_tv, probe_sr in the trace; in probe draw order).
    Round 10 options (off by default):
    record_eval_probs: for every evaluated request of every client, the softmax probabilities of both exits (float16)
        with the items of record_eval_requests for the installed model, the position in the evaluation order and a
        random arrival order per client and evaluation round (own RNG, no effect on training) -> {run}_evalprobs.npz;
    record_train_label_hist: per training round, the label counts of the samples each cell's server block trained on
        and of the samples all participating clients trained on (train_hist_cell [T, L, C], train_hist_all [T, C])."""
    device = cfg["device"]
    seed, total_rounds = int(cfg["partition_seed"]), int(cfg["global_rounds"])
    ckw = controller_kwargs or {}
    env, meta = load_env(env_path)
    K, T_env = env["member"].shape[:2]
    L = len(env["cell_xy"])
    assert total_rounds <= T_env, (total_rounds, T_env)
    ad = cfg.get("adaptive", {})
    lam_min, lam_max = ad.get("lam_min", 0.15), ad.get("lam_max", 0.7)
    Lam_min, Lam_max = ad.get("Lam_min", 0.4), ad.get("Lam_max", 0.7)
    gamma, eth = cfg.get("gamma", 0.5), cfg.get("eth_default", 0.8)
    lossy = float(meta.get("loss", 0.0)) > 0
    config = dict(scenario=scenario, arm=arm, env_file=str(env_path), env_meta=meta, mode=mode,
                  signal=signal if mode == "selfcal" else None, lambda_val=lambda_val,
                  big_lambda_val=big_lambda_val, apfl_eta=apfl_eta, signal_delay=signal_delay,
                  probe_n=probe_n, eval_every=eval_every, controller_kwargs=ckw,
                  partition_seed=seed, model_seed=cfg["model_seed"], run_name=run_name,
                  cfg=cfg)
    if record_device_signals or mode in ("device", "oracle") or fixed_Lambda is not None:
        config.update(record_device_signals=record_device_signals, device_signal=device_signal,
                      oracle_home_away=oracle_home_away, fixed_Lambda=fixed_Lambda)
    if eval_infer_lambdas or eval_mainaware_route:
        config.update(eval_infer_lambdas=list(eval_infer_lambdas or []), eval_mainaware_route=eval_mainaware_route)
    if record_eval_requests or record_probe_values:
        assert not record_eval_requests or eval_infer_lambdas, "--record_eval_requests needs --eval_infer_lambdas"
        assert not record_probe_values or record_device_signals, "--record_probe_values needs --record_device_signals"
        config.update(record_eval_requests=record_eval_requests, record_probe_values=record_probe_values)
    if record_eval_probs or record_train_label_hist:
        config.update(record_eval_probs=record_eval_probs, record_train_label_hist=record_train_label_hist)
    rec = RunRecord("r6_" + mode, config=config)

    # ---------- data, partition, pools ----------
    set_seed(seed)
    train_ds, test_ds = get_cifar10(data_root=cfg.get("data_root", "./data_cache"))
    train_labels, test_labels = np.array(train_ds.targets), np.array(test_ds.targets)
    probe_labels, eval_labels, pool_manifest = make_pool_masks(test_labels, cfg["num_classes"], 0.2, seed=seed)
    indices, mains, cell_groups, all_used = build_partition(train_labels, env, seed, cfg["num_classes"], cfg)
    probe_rb = RequestBuilder(probe_labels, cfg["num_classes"], cell_groups, all_used)
    eval_rb = RequestBuilder(eval_labels, cfg["num_classes"], cell_groups, all_used)
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
    Y_test = torch.as_tensor(test_labels, device=device)

    member = env["member"]
    cells_of = lambda k, t: sorted(int(z) for z in member[k, t] if z >= 0)
    slots_of = lambda k, t: [int(z) for z in member[k, t]]

    # ---------- models ----------
    set_seed(cfg["model_seed"])
    mf = ModelFactory(dataset="cifar10", num_classes=cfg["num_classes"])
    clients, ES = build_clients_and_es(
        num_clients=K, num_edge_servers=L, model_factory=mf, train_dataset=train_ds,
        client_indices=indices, client_to_es={k: cells_of(k, 0) for k in range(K)},
        device=device, batch_size=cfg["batch_size"], local_epochs=cfg["local_epochs"],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], num_workers=0)
    assert len(clients) == K and [c.cid for c in clients] == list(range(K))
    use_cached_loaders(clients, indices, train_ds, device, cfg["batch_size"])
    synchronize_initial_models(clients, ES)
    init_c = clients[0].get_client_state()
    init_s = clients[0].get_server_state(clients[0].edge_server_ids[0])
    for es in ES.values():   # a cell nobody has joined yet hands out the common initial blocks
        es.clients_avg_weights, es.server_avg_weights = init_c, init_s
    rec.set(split_hash=split_hash(indices), model_init_hash=model_hash(clients[0].client_model),
            disjoint_pools=pool_manifest,
            cell_groups={str(z): sorted(v) for z, v in cell_groups.items()},
            client_main={str(k): sorted(v) for k, v in mains.items()},
            client_n_train={str(k): len(v) for k, v in indices.items()})

    # ---------- controller ----------
    nbmat = env["neighbors"]
    neighbors = {z: [int(w) for w in np.flatnonzero(nbmat[z])] for z in range(L)}
    ctrl, apfl = None, None
    if mode == "selfcal":
        ctrl = EdgeDriftGate(L, neighbors, lam_min, lam_max, Lam_min, Lam_max,
                             warmup=ckw.get("warmup", 15), burn_in=ckw["burn_in"],
                             z_guard=ckw["z_guard"], spatial_norm=ckw["spatial_norm"],
                             neighbor_avg=not ckw.get("no_neighbor_avg", False))
    elif mode == "apfl":
        apfl = dict(lams={c.cid: 0.5 * (lam_min + lam_max) for c in clients},
                    eta=float(apfl_eta if apfl_eta is not None else cfg["learning_rate"]),
                    lam_min=lam_min, lam_max=lam_max)
    elif mode == "device":
        assert device_signal in ("tv", "sr"), device_signal
        dctrl = DeviceDriftGate(K, lam_min, lam_max, warmup=ckw.get("warmup", 15), burn_in=ckw["burn_in"],
                                z_guard=ckw["z_guard"], spatial_norm=ckw["spatial_norm"])
    elif mode == "oracle":
        assert oracle_home_away
    elif mode != "fixed":
        raise ValueError(mode)
    fixed_lam = {z: float(lambda_val) for z in range(L)}
    fixed_Lam = {z: float(big_lambda_val) for z in range(L)}
    probe_rng = np.random.default_rng(seed * 31 + 17)

    E_ROUNDS = eval_rounds_for(total_rounds, eval_every)
    E = len(E_ROUNDS)
    nan = np.nan
    R = dict(dk=np.full((total_rounds, K), nan, np.float32), nprobe=np.zeros((total_rounds, K), np.int16),
             delivered=np.zeros((total_rounds, K, 2), bool), participated=np.zeros((total_rounds, K), bool),
             probe_us=np.full((total_rounds, K), nan, np.float32),
             apfl_lam=np.full((total_rounds, K), nan, np.float32),
             client_lam=np.full((total_rounds, K), nan, np.float32),
             ctrl_ns=np.zeros((total_rounds, L), np.int64))
    r7 = record_device_signals or mode in ("device", "oracle")
    if r7:
        R.update(x_tv=np.full((total_rounds, K), nan, np.float32), x_sr=np.full((total_rounds, K), nan, np.float32),
                 q_k=np.full((total_rounds, K), nan, np.float32),
                 at_home=env["at_home"][:, :total_rounds].T.copy(),
                 cells=np.transpose(member[:, :total_rounds, :], (1, 0, 2)).copy())
    if record_train_label_hist:
        C_ = cfg["num_classes"]
        HK = {k: client_label_hist(indices[k], train_labels, C_, int(cfg["local_epochs"])) for k in range(K)}
        R.update(train_hist_cell=np.zeros((total_rounds, L, C_), np.int32), train_hist_all=np.zeros((total_rounds, C_), np.int32))
        if "cells" not in R:
            R["cells"] = np.transpose(member[:, :total_rounds, :], (1, 0, 2)).copy()
    PRB = [] if record_eval_probs else None
    TM = dict(train=0.0, eval=0.0) if (record_eval_probs or record_train_label_hist) else None   # Round 12 timing
    EV = {k: np.zeros((E, K), np.int32) for k in
          ("n_total", "correct", "n_off", "n_main", "c_main", "off_main", "n_nonmain",
           "c_nonmain", "off_nonmain", "n_oop", "c_oop", "n_oor", "c_oor")}
    if eval_mainaware_route:
        EV.update({k: np.zeros((E, K), np.int32) for k in ("ma_correct", "ma_n_off", "ma_c_main", "ma_c_nonmain")})
    REQ = None
    if record_eval_requests:   # blocks: one per lambda_inf, plus the installed model if lambda_t is not among them
        inst_block = (eval_infer_lambdas.index(float(lambda_val))
                      if mode == "fixed" and float(lambda_val) in eval_infer_lambdas else len(eval_infer_lambdas))
        n_blocks = max(len(eval_infer_lambdas), inst_block + 1)
        REQ = dict(meta=[], blocks=[[] for _ in range(n_blocks)])
    if record_probe_values:
        R.update(probe_tv=np.full((total_rounds, K, probe_n), nan, np.float32),
                 probe_sr=np.full((total_rounds, K, probe_n), -1, np.int8))
    EVI = None
    if eval_infer_lambdas:   # -1 = client not participating in that evaluation round (not evaluated)
        EVI = {k: np.full((E, len(eval_infer_lambdas), K), -1, np.int32) for k in
               ("correct", "n_off", "c_main", "c_nonmain", "off_main", "off_nonmain")}
    hist = dict(round=[], mean_loss=[], lamdas=[], big_lamdas=[], q=[], s_temp=[], s_spat=[], m=[],
                dbar=[], dbar_used=[], n_members=[], n_signals=[], rho_cell=[], n_active=[],
                n_rewired=[], active_cells=[], probe_fallback=[], eval=[])
    probe_used, eval_used = set(), set()
    dbar_hist = []
    cur_lam, cur_Lam = dict(fixed_lam), dict(fixed_Lam)
    if mode != "fixed":
        cur_lam = {z: 0.5 * (lam_min + lam_max) for z in range(L)}
        cur_Lam = ({z: 0.5 * (Lam_min + Lam_max) for z in range(L)} if mode == "selfcal"
                   else dict(fixed_Lam))
    if fixed_Lambda is not None:
        cur_Lam = {z: float(fixed_Lambda) for z in range(L)}
    client_lams = None
    if verbose:
        print(f"[{run_name}] scenario={scenario} arm={arm} mode={mode} K={K} L={L} rounds={total_rounds} "
              f"delay={signal_delay} device={device}", flush=True)
    t_start = time.time()
    by_id = {c.cid: c for c in clients}

    for r in range(1, total_rounds + 1):
        t = r - 1
        active = env["avail"][:, t].astype(bool)
        R["participated"][t] = active
        # 1. membership (participating clients follow their position; others keep blocks)
        target = {k: (cells_of(k, t) if active[k] else list(by_id[k].edge_server_ids)) for k in range(K)}
        n_rew = sum(1 for k in range(K) if set(target[k]) != set(by_id[k].edge_server_ids))
        if n_rew:
            rewire_clients(clients, ES, target, mf)
        # 2. signals
        dbar_now, n_sig = {}, {z: 0 for z in range(L)}
        fb = {"": 0, "oor_empty": 0, "oop_empty": 0, "both_empty": 0}
        xdev = {}
        if mode == "selfcal" or mode == "device" or record_device_signals:
            vals = {z: [] for z in range(L)}
            for c in clients:
                k = c.cid
                if not active[k]:
                    continue
                n = min(int(env["n_req"][k, t]), probe_n)
                if n <= 0:
                    continue
                pool, _, _, flag = probe_rb.build(mains[k], cells_of(k, t), float(env["rho"][k, t]))
                fb[flag] += 1
                n = min(n, len(pool))
                if n == 0:
                    continue
                chosen = probe_rng.choice(pool, size=n, replace=False)
                probe_used.update(int(i) for i in chosen)
                t0 = time.perf_counter_ns()
                want_sr = record_device_signals or device_signal == "sr"
                sig = compute_client_signals(c.client_model, list(c.server_models.values()),
                                             X_test[torch.as_tensor(chosen, device=device)], device, None,
                                             main_classes=set(mains[k]) if want_sr else None,
                                             per_request=record_probe_values)
                if record_probe_values:
                    sig, per = sig
                    R["probe_tv"][t, k, :n] = per["tv"]
                    R["probe_sr"][t, k, :n] = per["sr"]
                R["probe_us"][t, k] = (time.perf_counter_ns() - t0) / 1e3
                if r7:
                    R["x_tv"][t, k] = float(sig["tv_dist"])
                    if want_sr:
                        R["x_sr"][t, k] = float(sig["server_nonmain_hard"])
                R["nprobe"][t, k] = n
                if mode == "device":
                    xdev[k] = float(sig["tv_dist"] if device_signal == "tv" else sig["server_nonmain_hard"])
                    R["dk"][t, k] = xdev[k]
                if mode != "selfcal":
                    continue
                d = float(sig[signal])
                R["dk"][t, k] = d
                for j, z in enumerate(slots_of(k, t)):
                    if z < 0:
                        continue
                    if lossy and env["lost_c2e"][k, t, j]:
                        continue
                    vals[z].append(d)
                    R["delivered"][t, k, j] = True
            for z in range(L):
                n_sig[z] = len(vals[z])
                if vals[z]:
                    dbar_now[z] = float(np.mean(vals[z]))
        dbar_hist.append(dbar_now)
        dbar_used = dbar_hist[t - signal_delay] if t - signal_delay >= 0 else {}
        # 3. controller
        if mode == "selfcal":
            cur_lam, cur_Lam = ctrl.step(dbar_used,
                                         lost_dbar=env["lost_dbar"][t] if lossy else None,
                                         lost_nbr=env["lost_nbr"][t] if (lossy and ctrl.neighbor_avg) else None)
            R["ctrl_ns"][t] = [ctrl.last["t_ns"][z] for z in range(L)]
            if fixed_Lambda is not None:
                cur_Lam = {z: float(fixed_Lambda) for z in range(L)}
        elif mode == "device":
            mt_ = member[:, t, :]
            mem_t = {z: set(np.flatnonzero((mt_ == z).any(axis=1)).tolist()) for z in range(L)}
            reference = {k: sorted(set().union(*[mem_t[z] for z in cells_of(k, t)])) for k in xdev}
            client_lams = dctrl.step(xdev, reference)
            R["q_k"][t] = [dctrl.q[k] for k in range(K)]
        elif mode == "oracle":
            client_lams = {k: (0.70 if env["at_home"][k, t] else 0.15) for k in range(K)}
        if client_lams is not None:
            cur_lam = {z: float(np.mean([client_lams[k] for k in range(K) if z in member[k, t]]))
                       if (member[:, t, :] == z).any() else float("nan") for z in range(L)}
        # 4. training
        if record_train_label_hist:   # membership used by this round's training and cell averages
            for z in range(L):
                for c in ES[z].clients:
                    if active[c.cid]:
                        R["train_hist_cell"][t, z] += HK[c.cid]
            R["train_hist_all"][t] = sum(HK[k] for k in range(K) if active[k]) if active.any() else 0
        capture = {} if (EVI is not None and r in E_ROUNDS) else None
        _tt0 = time.perf_counter()
        mean_loss, act_cells = r6_training_round(clients, active, ES, cur_lam, cur_Lam, mode, gamma, apfl,
                                                 client_lams=client_lams, capture=capture)
        if TM is not None:
            TM["train"] += time.perf_counter() - _tt0
        if mode == "apfl":
            R["apfl_lam"][t] = [apfl["lams"][k] for k in range(K)]
            cur_lam = {z: float(np.mean([apfl["lams"][c.cid] for c in ES[z].clients]))
                       if ES[z].clients else nan for z in range(L)}
        for k in range(K):
            zs = by_id[k].edge_server_ids
            R["client_lam"][t, k] = (apfl["lams"][k] if mode == "apfl"
                                     else client_lams[k] if client_lams is not None
                                     else float(np.mean([cur_lam[z] for z in zs])))
        mt = member[:, t, :]
        members = {z: np.flatnonzero((mt == z).any(axis=1)).tolist() for z in range(L)}
        hist["round"].append(r)
        hist["mean_loss"].append(mean_loss)
        hist["lamdas"].append({str(z): cur_lam[z] for z in range(L)})
        hist["big_lamdas"].append({str(z): cur_Lam[z] for z in range(L)})
        if ctrl is not None:
            hist["q"].append({str(z): ctrl.q[z] for z in range(L)})
            hist["s_temp"].append({str(z): v for z, v in ctrl.last["s_temp"].items()})
            hist["s_spat"].append({str(z): v for z, v in ctrl.last["s_spat"].items()})
            hist["m"].append({str(z): v for z, v in ctrl.last["m"].items()})
        hist["dbar"].append({str(z): v for z, v in dbar_now.items()})
        hist["dbar_used"].append({str(z): v for z, v in dbar_used.items()})
        hist["n_members"].append({str(z): len(v) for z, v in members.items()})
        hist["n_signals"].append({str(z): v for z, v in n_sig.items()})
        hist["rho_cell"].append({str(z): (float(np.mean(env["rho"][v, t])) if v else None)
                                 for z, v in members.items()})
        hist["n_active"].append(int(active.sum()))
        hist["n_rewired"].append(n_rew)
        hist["active_cells"].append(act_cells)
        hist["probe_fallback"].append(fb)
        # 5. evaluation
        if r in E_ROUNDS:
            _te0 = time.perf_counter()
            e = E_ROUNDS.index(r)
            efb = {"": 0, "oor_empty": 0, "oop_empty": 0, "both_empty": 0}
            for c in clients:
                k = c.cid
                idx, oop, oor, flag = eval_rb.build(mains[k], cells_of(k, t), float(env["rho"][k, t]))
                efb[flag] += 1
                eval_used.update(int(i) for i in idx)
                rq = REQ is not None and capture is not None and k in capture
                rec_inst = {} if (rq or PRB is not None) else None
                res = eval_client(c, X_test, Y_test, idx, mains[k], oop, oor, eth=eth, mainaware=eval_mainaware_route,
                                  record=rec_inst, record_probs=PRB is not None)
                if PRB is not None:   # Round 10: both exits' probabilities, with a random arrival order
                    lab_p = np.asarray(Y_test[torch.as_tensor(np.asarray(idx), device=Y_test.device)].cpu().numpy())
                    kind_p = np.where(np.isin(lab_p, sorted(mains[k])), 0, np.where(np.isin(lab_p, sorted(oop)), 1,
                                      np.where(np.isin(lab_p, sorted(oor)), 2, 3)))
                    arr = np.random.default_rng([int(seed), 10, int(r), int(k)]).permutation(len(idx))
                    PRB.append(dict(e=e, k=k, n=len(idx), label=lab_p, kind=kind_p, home=bool(env["at_home"][k, t]),
                                    arrival=arr, **{x: rec_inst[x] for x in ("cp", "ent", "sp", "pc", "ps")}))
                for key in EV:
                    EV[key][e, k] = res[key]
                if capture is not None and k in capture:   # Round 8: inference-only mixing ratios
                    installed = c.get_client_state()
                    local, avg = capture[k]
                    for i, li in enumerate(eval_infer_lambdas):
                        c.set_client_state(mix_state_dicts(local, avg, float(li)))
                        rec_i = {} if rq else None
                        ri = eval_client(c, X_test, Y_test, idx, mains[k], oop, oor, eth=eth, record=rec_i)
                        for key in EVI:
                            EVI[key][e, i, k] = ri[key]
                        if rq:
                            REQ["blocks"][i].append(rec_i)
                    c.set_client_state(installed)
                if rq:
                    if inst_block == len(eval_infer_lambdas):
                        REQ["blocks"][inst_block].append(rec_inst)
                    elif not all(np.array_equal(rec_inst[x], REQ["blocks"][inst_block][-1][x]) for x in ("cp", "ent", "sp")):
                        REQ["installed_mismatch"] = REQ.get("installed_mismatch", 0) + 1
                    lab = np.asarray(Y_test[torch.as_tensor(np.asarray(idx), device=Y_test.device)].cpu().numpy())
                    kind = np.where(np.isin(lab, sorted(mains[k])), 0, np.where(np.isin(lab, sorted(oop)), 1,
                                    np.where(np.isin(lab, sorted(oor)), 2, 3)))
                    REQ["meta"].append(dict(e=e, k=k, n=len(idx), label=lab, kind=kind,
                                            home=bool(env["at_home"][k, t])))
            acc = EV["correct"][e] / np.maximum(EV["n_total"][e], 1)
            acc_main = [_ratio(a, b) for a, b in zip(EV["c_main"][e], EV["n_main"][e])]
            acc_nm = [_ratio(a, b) for a, b in zip(EV["c_nonmain"][e], EV["n_nonmain"][e])]
            ev = dict(round=r, acc_total=float(np.mean(acc)),
                      acc_main=float(np.nanmean(acc_main)), acc_nonmain=float(np.nanmean(acc_nm)),
                      offload_rate=float(EV["n_off"][e].sum() / max(EV["n_total"][e].sum(), 1)),
                      n_eval=int(EV["n_total"][e].sum()), fallback=efb,
                      per_client_acc={str(k): float(acc[k]) for k in range(K)})
            if EVI is not None:
                ok_k = EVI["correct"][e, 0] >= 0
                ev["acc_infer"] = {str(li): float(np.mean(EVI["correct"][e, i][ok_k] / np.maximum(EV["n_total"][e][ok_k], 1)))
                                   for i, li in enumerate(eval_infer_lambdas)}
            if eval_mainaware_route:
                ev["acc_mainaware"] = float(np.mean(EV["ma_correct"][e] / np.maximum(EV["n_total"][e], 1)))
            if TM is not None:
                TM["eval"] += time.perf_counter() - _te0
            hist["eval"].append(ev)
            if verbose:
                el = time.time() - t_start
                print(f"  R{r:3d} loss={mean_loss:.3f} acc={ev['acc_total']:.4f} off={ev['offload_rate']:.3f} "
                      f"active={int(active.sum())} elapsed={el/60:.1f}m ETA={(total_rounds-r)*el/r/60:.0f}m",
                      flush=True)

    hist["total_time_sec"] = time.time() - t_start
    if TM is not None:   # Round 12: training calls and evaluation (incl. the per-request recording) in seconds
        hist["r12_timing_sec"] = dict(TM)
    overlap = len(probe_used & eval_used)
    hist.update(pool_manifest=pool_manifest, probe_eval_overlap=overlap,
                probe_used_count=len(probe_used), eval_used_count=len(eval_used),
                eval_rounds=E_ROUNDS, K=K, L=L, env_meta=meta,
                config={k: v for k, v in config.items() if k != "cfg"} | {"run_id": rec.run_id})
    if REQ is not None:   # client-rounds whose installed-model requests differ from the lambda_inf = lambda_t block
        hist["installed_block_mismatch"] = int(REQ.get("installed_mismatch", 0))
    assert overlap == 0, f"probe/eval sample overlap = {overlap} (must be 0)"
    integrated = float(np.mean([e["acc_total"] for e in hist["eval"]]))
    hist["integrated_acc"] = integrated
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{run_name}.json", "w") as f:
            json.dump(hist, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        np.savez_compressed(out / f"{run_name}_trace.npz", eval_rounds=np.array(E_ROUNDS),
                            **R, **{f"eval_{k}": v for k, v in EV.items()},
                            **({"infer_lambdas": np.array(eval_infer_lambdas, np.float64)} if EVI is not None else {}),
                            **({f"evinf_{k}": v for k, v in EVI.items()} if EVI is not None else {}))
        if PRB is not None:
            ns = [m["n"] for m in PRB]
            np.savez_compressed(out / f"{run_name}_evalprobs.npz", eval_rounds=np.array(E_ROUNDS),
                                req_eval_index=np.repeat([m["e"] for m in PRB], ns).astype(np.uint8),
                                req_client=np.repeat([m["k"] for m in PRB], ns).astype(np.int16),   # uint8 wrapped K > 255
                                req_home=np.repeat([m["home"] for m in PRB], ns).astype(bool),
                                req_label=np.concatenate([m["label"] for m in PRB]).astype(np.uint8),
                                req_kind=np.concatenate([m["kind"] for m in PRB]).astype(np.int8),
                                req_order=np.concatenate([np.arange(m["n"]) for m in PRB]).astype(np.int32),
                                req_arrival=np.concatenate([m["arrival"] for m in PRB]).astype(np.int32),
                                cp=np.concatenate([m["cp"] for m in PRB]).astype(np.uint8),
                                ent=np.concatenate([m["ent"] for m in PRB]).astype(np.float32),
                                sp=np.concatenate([m["sp"] for m in PRB]).astype(np.uint8),
                                pc=np.concatenate([m["pc"] for m in PRB]).astype(np.float16),
                                ps=np.concatenate([m["ps"] for m in PRB]).astype(np.float16),
                                arrival_rng=np.array("numpy default_rng([partition_seed, 10, round, client]).permutation(n)"))
        if REQ is not None:
            ns = [m["n"] for m in REQ["meta"]]
            blk_lam = list(eval_infer_lambdas) + ([float(lambda_val) if mode == "fixed" else float("nan")]
                                                  if inst_block == len(eval_infer_lambdas) else [])
            np.savez_compressed(out / f"{run_name}_requests.npz", eval_rounds=np.array(E_ROUNDS),
                                block_lambda=np.array(blk_lam, np.float64), installed_block=np.int64(inst_block),
                                req_eval_index=np.repeat([m["e"] for m in REQ["meta"]], ns).astype(np.uint8),
                                req_client=np.repeat([m["k"] for m in REQ["meta"]], ns).astype(np.int16),   # uint8 wrapped K > 255
                                req_home=np.repeat([m["home"] for m in REQ["meta"]], ns).astype(bool),
                                req_label=np.concatenate([m["label"] for m in REQ["meta"]]).astype(np.uint8),
                                req_kind=np.concatenate([m["kind"] for m in REQ["meta"]]).astype(np.int8),
                                cp=np.stack([np.concatenate([r["cp"] for r in b]) for b in REQ["blocks"]]).astype(np.uint8),
                                ent=np.stack([np.concatenate([r["ent"] for r in b]) for b in REQ["blocks"]]).astype(np.float32),
                                sp=np.stack([np.concatenate([r["sp"] for r in b]) for b in REQ["blocks"]]).astype(np.uint8))
        if save_models:
            home = int(env["home_cell"][0])
            hub = int(np.flatnonzero(env["cell_is_hub"])[0]) if env["cell_is_hub"].any() else home
            torch.save({"client_block_and_exit": {k: v.cpu() for k, v in clients[0].client_model.state_dict().items()},
                        "server_block_and_exit_home_cell": {k: v.cpu() for k, v in ES[home].server_avg_weights.items()},
                        "server_block_and_exit_hub_cell": {k: v.cpu() for k, v in ES[hub].server_avg_weights.items()},
                        "client": 0, "home_cell": home, "hub_cell": hub, "run_id": rec.run_id,
                        "round": total_rounds}, out / f"{run_name}_final_models.pt")
        if verbose:
            print(f"  Saved: {out / (run_name + '.json')}  integrated={integrated:.4f}", flush=True)
    rec.finish(metrics={"integrated_acc": integrated, "acc_total_final": hist["eval"][-1]["acc_total"],
                        "wall_min": hist["total_time_sec"] / 60.0, "probe_eval_overlap": overlap})
    return hist
