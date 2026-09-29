"""Role-separation ablation (§7) — changes ONLY the aggregation role structure of
the underlying split learning; DV-2 (signal + controller + mapping) is untouched.

Role separation in DriftGate comes from the client update step, NOT the loss:
  - client aux head: lambda-mixed toward its LOCAL Main-class model  -> personalized
  - server head    : replaced by the cluster average                 -> generalized
The two exits disagree on OOP/OOR because one is a Main specialist and the other a
cluster generalist. These variants perturb exactly that structure.

Variants (role_mode):
  separated       R1 — current DriftGate (client personalized, server generalized).
                       Use the stock run_one_global_round; this module not needed.
  same_role       R2 — client head ALSO replaced by the cluster average. Both exits are
                       cluster generalists (different modules, independent inits) → tests
                       whether the signal needs role separation or just two classifiers.
  same_role_indep R3 — R2 + the client aux head re-initialised from a distinct seed at
                       build time (handled in the runner) → tests whether ensemble
                       diversity alone (independent inits, same role) creates the signal.
  weak_server     R4 — client stays personalized; the server head is mixed 50/50 with its
                       OWN pre-aggregation weights, shrinking its generalization advantage
                       (a graded reduction of role separation, not random damage).

lambda from the controller is still emitted every round; in same_role/same_role_indep it
is structurally bypassed for the client head (both generalized), which is the intended
manipulation. All variants reuse the identical DV-2 probe/signal/mapping.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from train.trainer import (
    aggregate_all_es, apply_network_aggregation, uniform_average, mix_state_dicts,
)

WEAK_SERVER_MIX = 0.5  # R4: fraction of the server head kept as own (less generalized)


def _cell_avg_client(client, edge_servers):
    if len(client.edge_server_ids) == 1:
        return edge_servers[client.edge_server_ids[0]].clients_avg_weights
    return uniform_average([edge_servers[z].clients_avg_weights
                            for z in client.edge_server_ids])


def role_aware_update(clients, edge_servers, lamdas_per_cell, role_mode):
    """Role-structure-perturbed client update. Server-side cell aggregation
    (aggregate_all_es) and the SplitOMC+ inter-cell mixing must already be done."""
    for client in clients:
        avg_client = _cell_avg_client(client, edge_servers)
        if role_mode in ("same_role", "same_role_indep"):
            # both exits generalized: client head = cluster average (no lambda)
            client.set_client_state(avg_client)
            for es_id in client.edge_server_ids:
                client.set_server_state(es_id, edge_servers[es_id].server_avg_weights)
        elif role_mode == "weak_server":
            if len(client.edge_server_ids) == 1:
                lam = lamdas_per_cell.get(client.edge_server_ids[0], 0.2)
            else:
                lam = float(np.mean([lamdas_per_cell.get(z, 0.2)
                                     for z in client.edge_server_ids]))
            local_w = client.get_client_state()
            client.set_client_state(mix_state_dicts(local_w, avg_client, lam))
            # weaken server generalization: keep half of its own (pre-agg) weights
            for es_id in client.edge_server_ids:
                own_s = client.get_server_state(es_id)
                cell_s = edge_servers[es_id].server_avg_weights
                client.set_server_state(es_id, mix_state_dicts(own_s, cell_s,
                                                               WEAK_SERVER_MIX))
        else:
            raise ValueError(f"role_aware_update called with role_mode={role_mode}")


def run_role_round(clients, edge_servers, lamdas, big_lambdas, role_mode, gamma=0.5):
    """One global round with a perturbed role structure (mirrors
    run_one_global_round for method='splitomcplus' but swaps the client update)."""
    losses = [c.train_one_round(method="splitomcplus", gamma=gamma) for c in clients]
    aggregate_all_es(edge_servers)
    apply_network_aggregation(edge_servers, big_lambdas)
    role_aware_update(clients, edge_servers, lamdas, role_mode)
    return float(np.mean(losses))


def reinit_client_aux(clients, seed_offset=777):
    """R3: re-initialize each client's auxiliary (client-exit) head from a distinct
    seed so the two exits start maximally diverse (same-role diversity control)."""
    import torch
    for i, c in enumerate(clients):
        dev = next(c.client_model.parameters()).device
        g = torch.Generator(device=dev).manual_seed(seed_offset + i)
        with torch.no_grad():
            for name, p in c.client_model.named_parameters():
                if name.startswith("aux_"):
                    if p.dim() >= 2:
                        torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5, generator=g)
                    else:
                        p.uniform_(-0.1, 0.1, generator=g)
