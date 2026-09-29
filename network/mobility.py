"""
Gauss-Markov mobility + user-centric AP cluster (top-k nearest).
For E4 mobility experiment.
"""
import numpy as np
from typing import Dict, List


class GaussMarkovMobility:
    """
    Gauss-Markov mobility model.
    p_k(t+1) = alpha * p_k(t) + (1-alpha) * mean_p + sqrt(1-alpha^2) * noise
    Velocity p_k(t+1) - p_k(t) is exponentially correlated.
    """

    def __init__(self, num_clients, num_es, map_size=1000.0, alpha=0.9, speed=10.0,
                 dt=10.0, seed=42):
        """
        map_size: side of square map in meters
        alpha: temporal correlation (0.9 = smooth)
        speed: mean speed in m/s
        dt: seconds per round
        """
        self.num_clients = num_clients
        self.num_es = num_es
        self.map_size = map_size
        self.alpha = alpha
        self.speed = speed
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Initialize client positions
        self.client_pos = self.rng.uniform(0, map_size, size=(num_clients, 2))
        # Initial velocity
        angles = self.rng.uniform(0, 2 * np.pi, num_clients)
        self.client_vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speed

        # ES positions - grid-like
        n_grid = int(np.ceil(np.sqrt(num_es)))
        es_pos_list = []
        for i in range(num_es):
            r = i // n_grid
            c = i % n_grid
            x = (c + 0.5) / n_grid * map_size
            y = (r + 0.5) / n_grid * map_size
            es_pos_list.append([x, y])
        self.es_pos = np.array(es_pos_list)

    def step(self):
        """Advance one round."""
        # Update velocity (Gauss-Markov)
        angles_new = self.rng.uniform(0, 2 * np.pi, self.num_clients)
        vel_new = np.stack([np.cos(angles_new), np.sin(angles_new)], axis=1) * self.speed
        self.client_vel = self.alpha * self.client_vel + (1 - self.alpha) * vel_new

        # Update position
        self.client_pos = self.client_pos + self.client_vel * self.dt

        # Bounce at boundaries
        self.client_pos = np.clip(self.client_pos, 0, self.map_size)

    def get_topk_es(self, k=2):
        """For each client, return list of k nearest ES IDs (primary first)."""
        client_to_es = {}
        for cid in range(self.num_clients):
            dists = np.linalg.norm(self.es_pos - self.client_pos[cid], axis=1)
            order = np.argsort(dists)
            client_to_es[cid] = order[:k].tolist()
        return client_to_es


def es_to_clients_from_c2es(client_to_es, num_es):
    """Invert client_to_es mapping."""
    out = {es_id: [] for es_id in range(num_es)}
    for cid, es_list in client_to_es.items():
        for es_id in es_list:
            out[es_id].append(cid)
    return out


def rewire_clients(clients, edge_servers_dict, new_client_to_es, model_factory):
    """
    Rewire clients' ES associations after mobility step.

    For each client:
    - For each NEW ES not previously associated: create a fresh server_model
      initialized from that ES's current server_avg_weights (if available),
      else from a fresh model_factory.make_server().
    - For each OLD ES no longer associated: drop the server_model
    - Update client.edge_server_ids
    Then re-attach to edge_servers_dict.
    """
    # Clear attachments
    for es in edge_servers_dict.values():
        es.clients = []

    for c in clients:
        old_es = set(c.edge_server_ids)
        new_es = set(new_client_to_es[c.cid])

        # Drop departed
        for es_id in old_es - new_es:
            if es_id in c.server_models:
                del c.server_models[es_id]

        # Add joined
        for es_id in new_es - old_es:
            new_sm = model_factory.make_server().to(c.device)
            # Warm-start from ES's current state if available
            if edge_servers_dict[es_id].server_avg_weights is not None:
                state = edge_servers_dict[es_id].server_avg_weights
                new_sm.load_state_dict({k: v.to(c.device) for k, v in state.items()})
            c.server_models[es_id] = new_sm

        c.edge_server_ids = sorted(new_es)

    # Re-attach
    for c in clients:
        for es_id in c.edge_server_ids:
            edge_servers_dict[es_id].attach(c)
