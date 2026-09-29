"""R4/B4 — APFL-style per-client mixing-weight learning (existing-method baseline).

lambda_k is learned ONLY from the client's local LABELED (Main-class) training loss.
No probe, no drift signal, no traffic information.

Per round, after local training and cluster/global aggregation:
    theta_k(lam) = lam * theta_local_k + (1 - lam) * theta_bar_k
    dL/dlam      = < grad_theta L(theta_k(lam)), theta_local_k - theta_bar_k >   (one local minibatch)
    lam_k       <- clip(lam_k - eta * dL/dlam, lam_min, lam_max)
then the client block is set to theta_k(lam_k_new); server blocks take the cell average
(same as SplitOMC+ eq. 8). Lambda (cluster<->global) is fixed by the caller.
The gradient probe uses eval() mode so it does not perturb BN running statistics.
"""
import numpy as np
import torch

from train.trainer import (aggregate_all_es, apply_network_aggregation,
                           mix_state_dicts, uniform_average)
from models.losses import multi_exit_loss


def _grad_wrt_lambda(client, local_sd, avg_sd, lam, gamma):
    """dL/dlam at the mixed model theta(lam), from ONE local minibatch."""
    client.set_client_state(mix_state_dicts(local_sd, avg_sd, lam))
    client.client_model.eval()
    for sm in client.server_models.values():
        sm.eval()
    images, labels = next(iter(client.train_loader))
    images = images.to(client.device, non_blocking=True)
    labels = labels.to(client.device, non_blocking=True)
    client.client_model.zero_grad(set_to_none=True)
    for sm in client.server_models.values():
        sm.zero_grad(set_to_none=True)
    with torch.enable_grad():
        client_logits, rep = client.client_model(images)
        s_list = [client.server_models[e](rep)[0] for e in client.edge_server_ids]
        loss = multi_exit_loss(client_logits, s_list, labels, gamma=gamma)
        loss.backward()
    g = 0.0
    for name, p in client.client_model.named_parameters():
        if p.grad is None or name not in local_sd:
            continue
        diff = (local_sd[name].float() - avg_sd[name].float()).to(p.grad.device)
        g += float((p.grad.float() * diff).sum().item())
    client.client_model.zero_grad(set_to_none=True)
    for sm in client.server_models.values():
        sm.zero_grad(set_to_none=True)
    return g, float(loss.item())


def run_apfl_round(clients, edge_servers_dict, client_lams, eta, big_lambdas,
                   gamma=0.5, lam_min=0.15, lam_max=0.70):
    """One global round with APFL-style lambda_k. Returns (mean_loss, new_client_lams)."""
    losses = [c.train_one_round(method="splitomcplus", gamma=gamma) for c in clients]
    aggregate_all_es(edge_servers_dict)
    apply_network_aggregation(edge_servers_dict, big_lambdas)

    new_lams = dict(client_lams)
    for client in clients:
        if len(client.edge_server_ids) == 1:
            avg_client = edge_servers_dict[client.edge_server_ids[0]].clients_avg_weights
        else:
            avg_client = uniform_average(
                [edge_servers_dict[e].clients_avg_weights for e in client.edge_server_ids])
        local_w = client.get_client_state()
        lam = float(client_lams[client.cid])
        g, _ = _grad_wrt_lambda(client, local_w, avg_client, lam, gamma)
        lam_new = float(min(max(lam - eta * g, lam_min), lam_max))
        new_lams[client.cid] = lam_new
        client.set_client_state(mix_state_dicts(local_w, avg_client, lam_new))
        for e in client.edge_server_ids:
            client.set_server_state(e, edge_servers_dict[e].server_avg_weights)
    return float(np.mean(losses)), new_lams
