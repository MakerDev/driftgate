"""
Per-client drift signal tracking + per-cell aggregation + consensus + sigmoid mapping.
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from collections import deque


class ClientDriftTracker:
    """Maintain rolling buffer of recent entropy values + anchor logit history."""

    def __init__(self, anchor_buffer_size=16, entropy_buffer_size=64, device='cuda'):
        self.anchor_buffer_size = anchor_buffer_size
        self.entropy_buffer_size = entropy_buffer_size
        self.device = device
        self.anchor_images = None
        self.anchor_labels = None
        self.prev_anchor_logits = None
        self.entropy_buffer = deque(maxlen=entropy_buffer_size)
        # Last disagreement-signal values (for per-ES logging / history)
        self.last_delta = 0.0   # S1: client-vs-server prediction disagreement rate
        self.last_margin = 0.0  # S2: 1 - server top-1/top-2 prob margin

    def register_anchors(self, train_loader):
        """Sample a fixed anchor set from the client's training data."""
        anchors_x, anchors_y = [], []
        n_needed = self.anchor_buffer_size
        for x, y in train_loader:
            anchors_x.append(x)
            anchors_y.append(y)
            if sum(a.shape[0] for a in anchors_x) >= n_needed:
                break
        if not anchors_x:
            self.anchor_images = None
            return
        all_x = torch.cat(anchors_x, dim=0)[:n_needed]
        all_y = torch.cat(anchors_y, dim=0)[:n_needed]
        self.anchor_images = all_x.detach().clone()
        self.anchor_labels = all_y.detach().clone()
        self.prev_anchor_logits = None

    def update_anchors_from_traffic(self, traffic_images):
        """Replace anchor with a batch of unlabeled incoming-traffic images.
        Simulates real deployment: controller measures entropy on the
        distribution it actually serves (ρ-mixed traffic), not on its own
        training data. Labels are not needed for entropy."""
        if traffic_images is None or traffic_images.shape[0] == 0:
            return
        n = min(traffic_images.shape[0], self.anchor_buffer_size)
        self.anchor_images = traffic_images[:n].detach().clone()
        self.anchor_labels = None  # entropy is label-free
        self.prev_anchor_logits = None  # Δ reset when anchors change

    def update_entropy_from_batch(self, client_logits_batch):
        """Append per-sample entropies from a forward pass."""
        with torch.no_grad():
            p = F.softmax(client_logits_batch, dim=1)
            log_p = F.log_softmax(client_logits_batch, dim=1)
            ent = -(p * log_p).sum(dim=1).detach().cpu().numpy()
        for e in ent:
            self.entropy_buffer.append(float(e))

    def mean_entropy(self):
        if len(self.entropy_buffer) == 0:
            return 0.0
        return float(np.mean(self.entropy_buffer))

    def compute_delta(self, client_model):
        """L1 logit variation on anchor set vs previous round."""
        if self.anchor_images is None:
            return 0.0
        client_model.eval()
        with torch.no_grad():
            cur, _ = client_model(self.anchor_images.to(self.device))
        cur = cur.detach().cpu()
        if self.prev_anchor_logits is None:
            self.prev_anchor_logits = cur
            return 0.0
        delta = (cur - self.prev_anchor_logits).abs().sum(dim=1).mean().item()
        self.prev_anchor_logits = cur
        return float(delta)

    def compute_disagreement(self, client_model, server_models, probe_x):
        """Exit-disagreement signal between client head and server head(s).

        Mirrors the two-head pattern in eval/evaluator.py (lines 60-69) but
        is label-free: the client head never learns OOR/OOP classes (not its
        task, no labels), so disagreement does NOT collapse at training
        convergence the way entropy does.

        Returns (delta_k, margin_k):
            delta_k  (S1): fraction of probe samples where client_pred != server_pred
            margin_k (S2): 1 - mean(server top-1 prob - top-2 prob)
        """
        if probe_x is None or probe_x.shape[0] == 0:
            return 0.0, 0.0
        server_models = list(server_models)
        if len(server_models) == 0:
            return 0.0, 0.0
        client_model.eval()
        for sm in server_models:
            sm.eval()
        with torch.no_grad():
            x = probe_x.to(self.device)
            client_logits, rep = client_model(x)
            server_logits = sum(sm(rep)[0] for sm in server_models) / len(server_models)
            client_pred = client_logits.argmax(dim=1)
            server_pred = server_logits.argmax(dim=1)
            delta_k = (client_pred != server_pred).float().mean().item()       # S1
            sm_prob = F.softmax(server_logits, dim=1)
            top2 = sm_prob.topk(2, dim=1).values
            margin_k = 1.0 - (top2[:, 0] - top2[:, 1]).mean().item()           # S2
        self.last_delta = float(delta_k)
        self.last_margin = float(margin_k)
        return float(delta_k), float(margin_k)


def gather_signals_for_clients(clients, drift_trackers, recompute_entropy=True,
                               signal_mode='entropy', beta_margin=0.0):
    """
    For each client produce a (primary_scalar, secondary_scalar) signal tuple.

    signal_mode='entropy' (original, kept verbatim for the C2 ablation):
        primary = H_k (mean entropy on probe), secondary = Delta_k (anchor logit variation).

    signal_mode='disagreement':
        primary = drift_k = delta_k + beta_margin * margin_k  (S1 [+ S2]),
        secondary = 0.0 (unused; consensus/sigmoid treat primary as the generic
        drift scalar). High drift => OOP/OOR traffic present => favor
        generalization => low lambda, same monotone direction as high entropy.
    """
    signals = {}
    for c in clients:
        cid = c.cid
        tr = drift_trackers[cid]

        if signal_mode == 'disagreement':
            server_models = list(c.server_models.values())
            delta_k, margin_k = tr.compute_disagreement(
                c.client_model, server_models, tr.anchor_images)
            drift_k = delta_k + beta_margin * margin_k
            signals[cid] = (drift_k, 0.0)
            continue

        if recompute_entropy and tr.anchor_images is not None:
            c.client_model.eval()
            with torch.no_grad():
                cl, _ = c.client_model(tr.anchor_images.to(c.device))
                p = F.softmax(cl, dim=1)
                log_p = F.log_softmax(cl, dim=1)
                ent = -(p * log_p).sum(dim=1).detach().cpu().numpy()
            H = float(np.mean(ent))
        else:
            H = tr.mean_entropy()

        delta = tr.compute_delta(c.client_model)
        signals[cid] = (H, delta)
    return signals


def aggregate_signals_per_es(client_signals, clients):
    """Average H and Delta over clients in each ES."""
    per_es = {}
    for c in clients:
        for es_id in c.edge_server_ids:
            per_es.setdefault(es_id, []).append(client_signals[c.cid])
    out_H, out_D = {}, {}
    for es_id, sig_list in per_es.items():
        Hs = [s[0] for s in sig_list]
        Ds = [s[1] for s in sig_list]
        out_H[es_id] = float(np.mean(Hs)) if Hs else 0.0
        out_D[es_id] = float(np.mean(Ds)) if Ds else 0.0
    return out_H, out_D


# --------------------------------------------------------------------------
# Consensus and mapping
# --------------------------------------------------------------------------

def consensus_step(signals_per_es, neighbors, steps=1):
    """One-step average consensus over the ES graph."""
    current = dict(signals_per_es)
    for _ in range(steps):
        new = {}
        for es_id, val in current.items():
            nbr_vals = [current[n] for n in neighbors.get(es_id, []) if n in current]
            all_vals = [val] + nbr_vals
            new[es_id] = float(np.mean(all_vals)) if all_vals else val
        current = new
    return current


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def map_H_to_lambda(H_tilde, mu_H, tau_H, lam_min=0.1, lam_max=0.7):
    """High H => OOP/OOR present => favor generalization => low lambda."""
    z = -(H_tilde - mu_H) / max(tau_H, 1e-6)
    return lam_min + (lam_max - lam_min) * sigmoid(z)


def map_H_to_big_lambda(H_tilde, mu_H, tau_H, Lam_min=0.2, Lam_max=0.8):
    """Same shape mapping."""
    z = -(H_tilde - mu_H) / max(tau_H, 1e-6)
    return Lam_min + (Lam_max - Lam_min) * sigmoid(z)


class AdaptiveController:
    """Per-cell lambda/Lambda controller from drift signals."""

    def __init__(self, neighbors, mu_H=1.5, tau_H=0.4,
                lam_min=0.1, lam_max=0.7, Lam_min=0.2, Lam_max=0.8,
                consensus_steps=1, use_delta=True, delta_weight=0.3):
        self.neighbors = neighbors
        self.mu_H = mu_H
        self.tau_H = tau_H
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.Lam_min = Lam_min
        self.Lam_max = Lam_max
        self.consensus_steps = consensus_steps
        self.use_delta = use_delta
        self.delta_weight = delta_weight

    def step(self, H_per_es, D_per_es=None):
        """Take per-ES signal averages, do consensus, map to lambdas."""
        H_consensed = consensus_step(H_per_es, self.neighbors, steps=self.consensus_steps)

        D_consensed = {}
        if D_per_es is not None and self.use_delta:
            D_consensed = consensus_step(D_per_es, self.neighbors, steps=self.consensus_steps)

        lamdas, big_lamdas = {}, {}
        for es_id, H in H_consensed.items():
            # Combine H and Delta signal
            effective_H = H
            if D_per_es is not None and self.use_delta:
                # Bias: high delta => model unstable => prefer generalization => boost effective H
                D = D_consensed.get(es_id, 0.0)
                effective_H = H + self.delta_weight * D

            lamdas[es_id] = map_H_to_lambda(effective_H, self.mu_H, self.tau_H,
                                           self.lam_min, self.lam_max)
            big_lamdas[es_id] = map_H_to_big_lambda(effective_H, self.mu_H, self.tau_H,
                                                   self.Lam_min, self.Lam_max)
        return lamdas, big_lamdas
