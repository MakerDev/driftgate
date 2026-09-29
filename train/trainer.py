"""
SplitOMC training engine matching reference (Client.py, EdgeServer.py, Network.py).

Key structural fact: each Client owns one client_model and |Z_k| server_models
(one per associated ES). Training computes multi-exit loss with all of them.
"""
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

from models.losses import multi_exit_loss, single_ce_loss, fedprox_term


# --------------------------------------------------------------------------
# State dict utilities
# --------------------------------------------------------------------------

def uniform_average(state_dicts: List[Dict]) -> Dict:
    """Uniform average of a list of state dicts."""
    if len(state_dicts) == 0:
        raise ValueError("Cannot average empty list")
    out = {k: torch.zeros_like(v).float() for k, v in state_dicts[0].items()}
    n = len(state_dicts)
    for sd in state_dicts:
        for k in out:
            out[k] = out[k] + sd[k].float()
    for k in out:
        out[k] = out[k] / n
    return out


def mix_state_dicts(sd_a: Dict, sd_b: Dict, weight_a: float) -> Dict:
    """Return weight_a * sd_a + (1 - weight_a) * sd_b, key-wise."""
    out = {}
    for k in sd_a:
        out[k] = weight_a * sd_a[k].float() + (1.0 - weight_a) * sd_b[k].float()
    return out


def detach_state_dict(state_dict):
    return {k: v.detach().clone().cpu() for k, v in state_dict.items()}


# --------------------------------------------------------------------------
# Client and Edge Server
# --------------------------------------------------------------------------

class SplitOMCClient:
    """One client; owns one client_model and |Z_k| server_models (per ES)."""

    def __init__(self, cid, train_loader, model_factory, edge_server_ids, device,
                 lr=0.01, weight_decay=1e-4, local_epochs=3, momentum=0.0):
        self.cid = cid
        self.train_loader = train_loader
        self.device = device
        self.lr = lr
        self.wd = weight_decay
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.edge_server_ids = list(edge_server_ids)
        self.client_model = model_factory.make_client().to(device)
        self.server_models = {
            es_id: model_factory.make_server().to(device)
            for es_id in self.edge_server_ids
        }
        # For FedProx
        self.global_client_params = None
        self.global_server_params = None

    def train_one_round(self, method='splitomc', gamma=0.5, fedprox_mu=0.01):
        """
        method:
            'splitomc' / 'splitomcplus' / 'adaptive_splitomc' / 'splitgp':
                multi-exit loss on (client + server forward)
            'fedavg': single CE through full forward (client -> server[0])
            'fedprox': fedavg + prox term
        Returns:
            mean_loss over local epochs
        """
        self.client_model.train()
        for sm in self.server_models.values():
            sm.train()

        opt_c = torch.optim.SGD(self.client_model.parameters(),
                               lr=self.lr, momentum=self.momentum,
                               weight_decay=self.wd)
        opt_s = {
            es_id: torch.optim.SGD(sm.parameters(), lr=self.lr,
                                  momentum=self.momentum, weight_decay=self.wd)
            for es_id, sm in self.server_models.items()
        }

        epoch_losses = []
        for epoch in range(self.local_epochs):
            batch_losses = []
            for batch in self.train_loader:
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                opt_c.zero_grad()
                for o in opt_s.values():
                    o.zero_grad()

                client_logits, representation = self.client_model(images)

                if method in ('splitomc', 'splitomcplus', 'adaptive_splitomc', 'splitgp'):
                    server_logits_list = []
                    for es_id in self.edge_server_ids:
                        s_logits, _ = self.server_models[es_id](representation)
                        server_logits_list.append(s_logits)
                    loss = multi_exit_loss(client_logits, server_logits_list,
                                          labels, gamma=gamma)
                elif method in ('fedavg', 'splitfed', 'fedmes'):
                    es_id = self.edge_server_ids[0]
                    s_logits, _ = self.server_models[es_id](representation)
                    loss = single_ce_loss(s_logits, labels)
                elif method == 'fedprox':
                    es_id = self.edge_server_ids[0]
                    s_logits, _ = self.server_models[es_id](representation)
                    ce = single_ce_loss(s_logits, labels)
                    prox = 0.0
                    if self.global_client_params is not None:
                        cur = list(self.client_model.parameters()) + list(self.server_models[es_id].parameters())
                        glob = self.global_client_params + self.global_server_params
                        prox = fedprox_term(cur, glob, mu=fedprox_mu)
                    loss = ce + prox
                else:
                    raise ValueError(f"Unknown method: {method}")

                loss.backward()
                opt_c.step()
                for o in opt_s.values():
                    o.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses) if batch_losses else 0.0)

        return np.mean(epoch_losses) if epoch_losses else 0.0

    def get_client_state(self):
        return detach_state_dict(self.client_model.state_dict())

    def get_server_state(self, es_id):
        return detach_state_dict(self.server_models[es_id].state_dict())

    def set_client_state(self, state):
        self.client_model.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def set_server_state(self, es_id, state):
        self.server_models[es_id].load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def snapshot_global_params(self):
        """For FedProx: snapshot current params to penalize drift from."""
        self.global_client_params = [p.detach().clone() for p in self.client_model.parameters()]
        es_id = self.edge_server_ids[0]
        self.global_server_params = [p.detach().clone() for p in self.server_models[es_id].parameters()]


class EdgeServer:
    """Edge server holding cell-level aggregated state."""

    def __init__(self, es_id):
        self.es_id = es_id
        self.clients_avg_weights = None  # client-side block average
        self.server_avg_weights = None   # server-side block average
        self.clients = []  # list of SplitOMCClient
        self.scope = None  # set of class IDs

    def attach(self, client):
        self.clients.append(client)

    def aggregate(self):
        cw_list = [c.get_client_state() for c in self.clients]
        sw_list = [c.get_server_state(self.es_id) for c in self.clients]
        self.clients_avg_weights = uniform_average(cw_list)
        self.server_avg_weights = uniform_average(sw_list)


# --------------------------------------------------------------------------
# Aggregation orchestration
# --------------------------------------------------------------------------

def aggregate_all_es(edge_servers_dict: Dict[int, EdgeServer]):
    """Each ES aggregates its clients."""
    for es in edge_servers_dict.values():
        es.aggregate()


def apply_network_aggregation(edge_servers_dict: Dict[int, EdgeServer],
                              big_lambdas_per_cell: Dict[int, float]):
    """
    SplitOMC+ inter-ES aggregation (eq. 11-12).
    Each ES mixes its own aggregate with the GLOBAL all-ES average.
    """
    es_list = list(edge_servers_dict.values())
    global_c = uniform_average([es.clients_avg_weights for es in es_list])
    global_s = uniform_average([es.server_avg_weights for es in es_list])

    for es_id, es in edge_servers_dict.items():
        Lam = big_lambdas_per_cell.get(es_id, 0.5)
        es.clients_avg_weights = mix_state_dicts(es.clients_avg_weights, global_c, Lam)
        es.server_avg_weights = mix_state_dicts(es.server_avg_weights, global_s, Lam)


def update_client_models(clients: List[SplitOMCClient],
                        edge_servers_dict: Dict[int, EdgeServer],
                        lamdas_per_cell: Dict[int, float],
                        method='splitomc'):
    """
    Apply eq.(9) for client-side; eq.(8) for server-side.

    For FedAvg/FedProx: client-side is just replaced by avg (no lambda mixing).
    For SplitGP: ignores cell structure; uses global average across all ES.
    """
    if method == 'splitgp':
        # SplitGP: global average across all clients (no cell structure)
        all_cw = uniform_average([es.clients_avg_weights for es in edge_servers_dict.values()])
        for client in clients:
            es_id = client.edge_server_ids[0]
            lam = lamdas_per_cell.get(es_id, 0.2)
            local_w = client.get_client_state()
            new_w = mix_state_dicts(local_w, all_cw, lam)
            client.set_client_state(new_w)
            # Server side: replace with global average
            all_sw = uniform_average([es.server_avg_weights for es in edge_servers_dict.values()])
            for es_id in client.edge_server_ids:
                client.set_server_state(es_id, all_sw)
        return

    if method in ('fedavg', 'splitfed', 'fedprox'):
        # No lambda; just replace with average
        all_cw = uniform_average([es.clients_avg_weights for es in edge_servers_dict.values()])
        all_sw = uniform_average([es.server_avg_weights for es in edge_servers_dict.values()])
        for client in clients:
            client.set_client_state(all_cw)
            for es_id in client.edge_server_ids:
                client.set_server_state(es_id, all_sw)
        return

    if method == 'fedmes':
        # No cloud aggregation; clients update from their ES's average directly
        for client in clients:
            if len(client.edge_server_ids) == 1:
                es_id = client.edge_server_ids[0]
                avg_c = edge_servers_dict[es_id].clients_avg_weights
                avg_s = edge_servers_dict[es_id].server_avg_weights
            else:
                avg_c = uniform_average([edge_servers_dict[es].clients_avg_weights for es in client.edge_server_ids])
                avg_s = uniform_average([edge_servers_dict[es].server_avg_weights for es in client.edge_server_ids])
            client.set_client_state(avg_c)
            for es_id in client.edge_server_ids:
                client.set_server_state(es_id, edge_servers_dict[es_id].server_avg_weights)
        return

    # SplitOMC / SplitOMC+ / Adaptive: cell-based with lambda
    for client in clients:
        if len(client.edge_server_ids) == 1:
            es_id = client.edge_server_ids[0]
            avg_client = edge_servers_dict[es_id].clients_avg_weights
            lam = lamdas_per_cell.get(es_id, 0.2)
        else:
            avg_list = [edge_servers_dict[es_id].clients_avg_weights
                       for es_id in client.edge_server_ids]
            avg_client = uniform_average(avg_list)
            lam = float(np.mean([lamdas_per_cell.get(es_id, 0.2)
                                for es_id in client.edge_server_ids]))

        local_w = client.get_client_state()
        new_w = mix_state_dicts(local_w, avg_client, lam)
        client.set_client_state(new_w)

        # Server-side: replace with ES's cell average (eq. 8, no lambda)
        for es_id in client.edge_server_ids:
            client.set_server_state(es_id, edge_servers_dict[es_id].server_avg_weights)


def run_one_global_round(clients, edge_servers_dict, lamdas, big_lambdas,
                        method='splitomc', gamma=0.5, fedprox_mu=0.01):
    """One full SplitOMC global round.

    Sequence:
    1. Each client trains locally for E epochs
    2. Each ES aggregates its clients
    3. (SplitOMC+ only) Inter-ES global mixing with Lambda
    4. Client update: eq.(9) for client-side, eq.(8) for server-side
    """
    if method == 'fedprox':
        for c in clients:
            c.snapshot_global_params()

    losses = []
    for client in clients:
        loss = client.train_one_round(method=method, gamma=gamma,
                                     fedprox_mu=fedprox_mu)
        losses.append(loss)

    aggregate_all_es(edge_servers_dict)

    if method in ('splitomcplus', 'adaptive_splitomc'):
        apply_network_aggregation(edge_servers_dict, big_lambdas)

    update_client_models(clients, edge_servers_dict, lamdas, method=method)

    return float(np.mean(losses))


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def build_clients_and_es(num_clients, num_edge_servers, model_factory,
                       train_dataset, client_indices, client_to_es,
                       device='cuda:0', batch_size=32, local_epochs=3,
                       lr=0.01, weight_decay=1e-4, num_workers=0):
    """Build all clients and edge servers and wire them up."""
    from data.partition import make_client_dataloader

    clients = []
    for cid in range(num_clients):
        if len(client_indices[cid]) == 0:
            # Empty client - skip
            continue
        loader = make_client_dataloader(train_dataset, client_indices[cid],
                                       batch_size=batch_size, num_workers=num_workers)
        c = SplitOMCClient(
            cid=cid,
            train_loader=loader,
            model_factory=model_factory,
            edge_server_ids=client_to_es[cid],
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
        )
        clients.append(c)

    edge_servers_dict = {es_id: EdgeServer(es_id) for es_id in range(num_edge_servers)}
    for c in clients:
        for es_id in c.edge_server_ids:
            edge_servers_dict[es_id].attach(c)

    return clients, edge_servers_dict


def synchronize_initial_models(clients, edge_servers_dict):
    """Make all clients start from the same client_model and server_model weights.
    
    This is the 'distributed at start of training' step from paper §IV-A.
    """
    if not clients:
        return
    init_client_state = clients[0].get_client_state()
    init_server_state = clients[0].get_server_state(clients[0].edge_server_ids[0])

    for c in clients:
        c.set_client_state(init_client_state)
        for es_id in c.edge_server_ids:
            c.set_server_state(es_id, init_server_state)
