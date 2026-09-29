"""
Evaluator matching reference Evaluator.evaluate.

For each test sample:
1. Forward through client model -> get logits + representation
2. Compute Shannon entropy on client logits
3. If entropy > E_th: route to server (average across all ES for overlap clients)
   Else: use client prediction
4. Track per-task (main/oop/oor) per-exit (client/server) accuracy
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Subset


def compute_entropy(logits):
    """Shannon entropy per sample (in nats)."""
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    return -(p * log_p).sum(dim=1)


@torch.no_grad()
def evaluate_one_client(client, test_dataset, test_indices,
                       client_main_classes, oop_classes, oor_classes,
                       eth=0.8, batch_size=128):
    """
    Returns dict with:
        acc_total, acc_main, acc_oop, acc_oor,
        client_main, server_main, client_oop, server_oop, client_oor, server_oor,
        n_main, n_oop, n_oor, n_total
    """
    device = client.device
    client.client_model.eval()
    for sm in client.server_models.values():
        sm.eval()

    if len(test_indices) == 0:
        return {f'acc_{k}': 0.0 for k in ['total', 'main', 'oop', 'oor']}

    sub = Subset(test_dataset, test_indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)

    counts = {t: {'cc': 0, 'ct': 0, 'sc': 0, 'st': 0}
             for t in ('main', 'oop', 'oor')}
    total_correct = 0
    total_samples = 0

    main_set = set(client_main_classes)
    oop_set = set(oop_classes)
    oor_set = set(oor_classes)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        client_logits, rep = client.client_model(images)
        entropy = compute_entropy(client_logits)
        _, client_preds = client_logits.max(dim=1)

        server_logits_list = []
        for sm in client.server_models.values():
            s_logits, _ = sm(rep)
            server_logits_list.append(s_logits)
        server_logits = sum(server_logits_list) / len(server_logits_list)
        _, server_preds = server_logits.max(dim=1)

        route_to_server = entropy > eth

        # Total acc
        final = torch.where(route_to_server, server_preds, client_preds)
        total_correct += (final == labels_dev).sum().item()
        total_samples += labels.size(0)

        # Per-task per-exit
        cp_cpu = client_preds.cpu().numpy()
        sp_cpu = server_preds.cpu().numpy()
        rs_cpu = route_to_server.cpu().numpy()
        lb_cpu = labels.numpy()

        for i in range(len(lb_cpu)):
            y = int(lb_cpu[i])
            if y in main_set:
                t = 'main'
            elif y in oop_set:
                t = 'oop'
            elif y in oor_set:
                t = 'oor'
            else:
                continue

            if rs_cpu[i]:
                counts[t]['st'] += 1
                if sp_cpu[i] == y:
                    counts[t]['sc'] += 1
            else:
                counts[t]['ct'] += 1
                if cp_cpu[i] == y:
                    counts[t]['cc'] += 1

    def sdiv(a, b):
        return a / b if b > 0 else 0.0

    result = {'acc_total': sdiv(total_correct, total_samples),
             'n_total': total_samples}
    for t in ('main', 'oop', 'oor'):
        c = counts[t]
        result[f'acc_{t}'] = sdiv(c['cc'] + c['sc'], c['ct'] + c['st'])
        result[f'client_{t}'] = sdiv(c['cc'], c['ct'])
        result[f'server_{t}'] = sdiv(c['sc'], c['st'])
        result[f'n_{t}'] = c['ct'] + c['st']

    return result


def evaluate_all_clients(clients, test_dataset, per_client_test_indices,
                        client_main_classes, oop_classes_per_client,
                        oor_classes_per_client, eth=0.8, batch_size=128):
    """Mean of per-client results."""
    per_client_results = []
    for client in clients:
        cid = client.cid
        result = evaluate_one_client(
            client, test_dataset,
            per_client_test_indices[cid],
            client_main_classes[cid],
            oop_classes_per_client[cid],
            oor_classes_per_client[cid],
            eth=eth,
            batch_size=batch_size,
        )
        result['cid'] = cid
        result['primary_es'] = client.edge_server_ids[0]
        per_client_results.append(result)

    keys = [k for k in per_client_results[0].keys() if k.startswith('acc_') or k.startswith('client_') or k.startswith('server_')]
    aggregate = {k: float(np.mean([r[k] for r in per_client_results])) for k in keys}

    # Per-cell aggregates for fairness/spatial analysis
    per_cell = {}
    for r in per_client_results:
        es = r['primary_es']
        per_cell.setdefault(es, []).append(r['acc_total'])
    cell_means = {es: float(np.mean(v)) for es, v in per_cell.items()}
    aggregate['cell_means'] = cell_means
    if cell_means:
        aggregate['worst_cell_acc'] = float(min(cell_means.values()))
        aggregate['best_cell_acc'] = float(max(cell_means.values()))
        aggregate['cell_gap'] = aggregate['best_cell_acc'] - aggregate['worst_cell_acc']

    return aggregate, per_client_results


def build_per_client_test_sets(test_labels, num_clients, client_main_classes,
                              client_to_es, es_scope, oop_ratio,
                              oor_ratio_factor=0.3,
                              per_cell_oop_ratio=None):
    """
    Build per-client test indices, OOP class sets, OOR class sets.
    
    per_cell_oop_ratio: optional dict {es_id -> ρ_z} for spatial heterogeneity.
                       If None, use global oop_ratio.
    """
    from data.partition import build_per_client_test_set, compute_all_used_classes

    all_used = compute_all_used_classes(client_main_classes)
    per_client_test_indices = {}
    oop_per_client = {}
    oor_per_client = {}

    for cid in range(num_clients):
        if cid not in client_main_classes:
            per_client_test_indices[cid] = []
            oop_per_client[cid] = set()
            oor_per_client[cid] = set()
            continue
        primary = client_to_es[cid][0]
        rho = (per_cell_oop_ratio[primary]
              if per_cell_oop_ratio is not None
              else oop_ratio)
        idxs, oop_classes, oor_classes = build_per_client_test_set(
            test_labels, client_main_classes[cid], es_scope, primary,
            all_used, oop_ratio=rho, oor_ratio_factor=oor_ratio_factor)
        per_client_test_indices[cid] = idxs
        oop_per_client[cid] = oop_classes
        oor_per_client[cid] = oor_classes

    return per_client_test_indices, oop_per_client, oor_per_client
