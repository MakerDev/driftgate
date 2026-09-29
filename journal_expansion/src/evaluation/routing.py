"""F3 — exit-routing variants (inference-time controller).

The legacy evaluator routes to the server exit when CLIENT ENTROPY > eth.
A converged (overconfident) client head under-routes OOP/OOR traffic — the
same convergence confound the entropy *training* controller had. These
routing rules are per-sample, label-free at decision time:

  entropy       — legacy: server iff client entropy > eth (reference)
  disagree_conf — client if exits agree; on disagreement, the more confident
                  exit answers (confidence-arbitrated disagreement routing)
  client_always / server_always — bounds
  oracle        — correct if EITHER exit is correct (routing upper bound;
                  uses labels, analysis only)

evaluate_routing() computes ALL variants from the same forward passes, so
adding it to an eval round costs almost nothing extra.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

ROUTING_MODES = ["entropy", "disagree_conf", "client_always", "server_always", "oracle"]


@torch.no_grad()
def evaluate_routing_one_client(client, test_dataset, test_indices, eth=0.8,
                                batch_size=128):
    if len(test_indices) == 0:
        return {m: 0.0 for m in ROUTING_MODES}, 0
    device = client.device
    client.client_model.eval()
    for sm in client.server_models.values():
        sm.eval()
    loader = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    correct = {m: 0 for m in ROUTING_MODES}
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        cl, rep = client.client_model(images)
        sl = sum(sm(rep)[0] for sm in client.server_models.values()) / len(client.server_models)
        pc = F.softmax(cl, dim=1)
        ps = F.softmax(sl, dim=1)
        ent = -(pc * torch.log(pc + 1e-12)).sum(dim=1)
        cpred, spred = cl.argmax(1), sl.argmax(1)
        cconf, sconf = pc.max(1).values, ps.max(1).values

        preds = {
            "entropy": torch.where(ent > eth, spred, cpred),
            "disagree_conf": torch.where(cpred == spred, cpred,
                                         torch.where(sconf >= cconf, spred, cpred)),
            "client_always": cpred,
            "server_always": spred,
        }
        for m, p in preds.items():
            correct[m] += (p == labels).sum().item()
        correct["oracle"] += ((cpred == labels) | (spred == labels)).sum().item()
        # exit-agreement decomposition (Phase G false-agreement analysis)
        cok, sok = cpred == labels, spred == labels
        agree = cpred == spred
        for k, mask in (("both_correct", cok & sok),
                        ("client_only_correct", cok & ~sok),
                        ("server_only_correct", ~cok & sok),
                        ("both_wrong_same", ~cok & ~sok & agree),
                        ("both_wrong_diff", ~cok & ~sok & ~agree)):
            correct[k] = correct.get(k, 0) + mask.sum().item()
        n += labels.numel()
    out = {m: correct[m] / n for m in ROUTING_MODES}
    for k in ("both_correct", "client_only_correct", "server_only_correct",
              "both_wrong_same", "both_wrong_diff"):
        out[k] = correct.get(k, 0) / n
    return out, n


def evaluate_routing(clients, test_dataset, per_client_test_indices, eth=0.8):
    """Client-mean accuracy per routing mode + per-cell worst under each mode."""
    per_client = {}
    for c in clients:
        accs, n = evaluate_routing_one_client(c, test_dataset,
                                              per_client_test_indices[c.cid], eth=eth)
        if n > 0:
            per_client[c.cid] = (accs, c.edge_server_ids[0])
    out = {}
    for m in ROUTING_MODES:
        vals = [a[m] for a, _ in per_client.values()]
        out[m] = float(np.mean(vals)) if vals else 0.0
        cells = {}
        for a, es in per_client.values():
            cells.setdefault(es, []).append(a[m])
        if cells:
            out[f"{m}_worst_cell"] = float(min(np.mean(v) for v in cells.values()))
    # exit-agreement decomposition (Phase G false-agreement analysis)
    for k in ("both_correct", "client_only_correct", "server_only_correct",
              "both_wrong_same", "both_wrong_diff"):
        vals = [a[k] for a, _ in per_client.values() if k in a]
        if vals:
            out[k] = float(np.mean(vals))
    return out
