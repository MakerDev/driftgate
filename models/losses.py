"""
Losses for SplitOMC and baselines. Matches reference Client.py exactly.
"""
import torch
import torch.nn as nn


def multi_exit_loss(client_logits, server_logits_list, labels, gamma=0.5):
    """
    SplitGP / SplitOMC multi-exit loss (eq.5).

    F_k = gamma * CE(client) + (1-gamma) * mean_z CE(server_z)

    All on D_k (main classes). No OOP/OOR term.

    Args:
        client_logits: [B, num_classes] from client auxiliary head
        server_logits_list: list of [B, num_classes], one per associated ES
        labels: [B] integer labels
        gamma: client weight, fixed 0.5 per paper

    Returns:
        scalar loss
    """
    ce = nn.CrossEntropyLoss()
    l_c = ce(client_logits, labels)
    if len(server_logits_list) == 0:
        return l_c
    l_s_list = [ce(sl, labels) for sl in server_logits_list]
    l_s = sum(l_s_list) / len(l_s_list)
    return gamma * l_c + (1 - gamma) * l_s


def single_ce_loss(logits, labels):
    """For baselines that use only one output (FedAvg through full path)."""
    return nn.CrossEntropyLoss()(logits, labels)


def fedprox_term(current_params, global_params, mu=0.01):
    """Compute FedProx regularization term."""
    prox = 0.0
    for cp, gp in zip(current_params, global_params):
        prox = prox + (cp - gp).pow(2).sum()
    return (mu / 2.0) * prox
