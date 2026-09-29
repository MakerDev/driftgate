"""Signal library: all drift-signal candidates computed from ONE probe forward.

Label-free by construction: nothing in this module ever receives labels or the
true rho. All signals are computed from (client_model, server_models, probe_x).

S0  predictive entropy         (client / server / avg; mean and max)
S1  hard exit disagreement     delta = P[argmax f_c != argmax f_s]
S2  soft distribution disagreement (sym-KL, JS, TV, logit-cosine, margin diff)
S3  representation-level drift vs a warm-up reference (centroid / cov / cosine / MMD)
aux confidence terms for the S5 server-competence gate

S4 (temporal normalization) lives in src/controllers/normalizers.py and
src/evaluation/signal_metrics.py — it is a causal transform of these raw series.
"""
import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-8

# Ordered list of raw signal names produced per probe (keep stable: npz columns)
RAW_SIGNAL_NAMES = [
    "ent_client", "ent_client_max", "ent_server", "ent_avg",
    "delta_hard",
    "kl_sym", "js_div", "tv_dist", "cos_logit_dist", "margin_diff",
    "conf_client", "conf_server",
    "rep_centroid", "rep_cov", "rep_cos", "rep_mmd",
]


@torch.no_grad()
def probe_forward(client_model, server_models, probe_x, device):
    """One forward pass; returns tensors needed by every signal."""
    client_model.eval()
    sms = list(server_models)
    for sm in sms:
        sm.eval()
    x = probe_x.to(device)
    client_logits, rep = client_model(x)
    if len(sms) > 0:
        server_logits = sum(sm(rep)[0] for sm in sms) / len(sms)
    else:
        server_logits = client_logits.clone()
    rep_pooled = F.adaptive_avg_pool2d(rep, 1).flatten(1)  # [B, C]
    return client_logits, server_logits, rep_pooled


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    return -(p * logp).sum(dim=1)


@torch.no_grad()
def raw_signals(client_logits, server_logits, rep_pooled, rep_ref=None):
    """Compute the full raw-signal dict from probe outputs. Label-free."""
    out = {}
    ent_c = _entropy(client_logits)
    ent_s = _entropy(server_logits)
    out["ent_client"] = ent_c.mean().item()
    out["ent_client_max"] = ent_c.max().item()
    out["ent_server"] = ent_s.mean().item()
    out["ent_avg"] = 0.5 * (out["ent_client"] + out["ent_server"])

    cp = client_logits.argmax(dim=1)
    sp = server_logits.argmax(dim=1)
    out["delta_hard"] = (cp != sp).float().mean().item()

    p = F.softmax(client_logits, dim=1)
    q = F.softmax(server_logits, dim=1)
    logp = torch.log(p + EPS)
    logq = torch.log(q + EPS)
    kl_pq = (p * (logp - logq)).sum(dim=1)
    kl_qp = (q * (logq - logp)).sum(dim=1)
    out["kl_sym"] = (0.5 * (kl_pq + kl_qp)).mean().item()
    m = 0.5 * (p + q)
    logm = torch.log(m + EPS)
    js = 0.5 * (p * (logp - logm)).sum(dim=1) + 0.5 * (q * (logq - logm)).sum(dim=1)
    out["js_div"] = js.mean().item()
    out["tv_dist"] = (0.5 * (p - q).abs().sum(dim=1)).mean().item()
    out["cos_logit_dist"] = (1.0 - F.cosine_similarity(client_logits, server_logits, dim=1)).mean().item()

    top2c = p.topk(2, dim=1).values
    top2s = q.topk(2, dim=1).values
    marg_c = top2c[:, 0] - top2c[:, 1]
    marg_s = top2s[:, 0] - top2s[:, 1]
    out["margin_diff"] = (marg_c - marg_s).abs().mean().item()
    out["conf_client"] = p.max(dim=1).values.mean().item()
    out["conf_server"] = q.max(dim=1).values.mean().item()

    if rep_ref is not None and rep_ref.frozen:
        r = rep_ref.compare(rep_pooled)
        out.update(r)
    else:
        out.update({"rep_centroid": 0.0, "rep_cov": 0.0, "rep_cos": 0.0, "rep_mmd": 0.0})
    return out


class RepReference:
    """Warm-up reference statistics of the client representation.

    Accumulates pooled representations during the first W rounds (no rho
    labels used — only the raw traffic seen in warm-up), then freezes:
    mean vector, per-dim variance, a sample bank for MMD, and an RBF
    bandwidth from the median heuristic on the bank.
    """

    def __init__(self, max_bank=256, warmup_rounds=15):
        self.max_bank = max_bank
        self.warmup_rounds = warmup_rounds
        self._seen_rounds = 0
        self._bank = []
        self.frozen = False
        self.mean = None
        self.var = None
        self.bw2 = None  # squared RBF bandwidth

    def accumulate(self, rep_pooled):
        if self.frozen:
            return
        self._bank.append(rep_pooled.detach().cpu())
        self._seen_rounds += 1
        if self._seen_rounds >= self.warmup_rounds:
            self.freeze()

    def freeze(self):
        if self.frozen or not self._bank:
            return
        bank = torch.cat(self._bank, dim=0)
        if bank.shape[0] > self.max_bank:
            idx = torch.linspace(0, bank.shape[0] - 1, self.max_bank).long()
            bank = bank[idx]
        self.bank = bank
        self.mean = bank.mean(dim=0)
        self.var = bank.var(dim=0) + EPS
        with torch.no_grad():
            d = torch.cdist(bank, bank).flatten()
            med = d[d > 0].median().item() if (d > 0).any() else 1.0
        self.bw2 = max(med ** 2, EPS)
        self._bank = None
        self.frozen = True

    @torch.no_grad()
    def compare(self, rep_pooled):
        x = rep_pooled.detach().cpu()
        mu_t = x.mean(dim=0)
        var_t = x.var(dim=0) + EPS
        centroid = torch.norm(mu_t - self.mean).item() / (torch.norm(self.mean).item() + EPS)
        cov = (var_t - self.var).abs().sum().item() / (self.var.sum().item() + EPS)
        cos = 1.0 - F.cosine_similarity(mu_t.unsqueeze(0), self.mean.unsqueeze(0)).item()
        mmd = self._mmd(x)
        return {"rep_centroid": centroid, "rep_cov": cov, "rep_cos": cos, "rep_mmd": mmd}

    def _mmd(self, x):
        y = self.bank
        def k(a, b):
            return torch.exp(-torch.cdist(a, b) ** 2 / (2 * self.bw2))
        kxx = k(x, x)
        n = x.shape[0]
        kxx = (kxx.sum() - kxx.diag().sum()) / max(n * (n - 1), 1)
        kyy = k(y, y)
        m = y.shape[0]
        kyy = (kyy.sum() - kyy.diag().sum()) / max(m * (m - 1), 1)
        kxy = k(x, y).mean()
        return max((kxx + kyy - 2 * kxy).item(), 0.0)


# Task-2 direct server non-Main signals (NOT in the npz RAW_SIGNAL_NAMES; added
# to the returned dict only when main_classes is supplied). Both in [0,1].
SERVER_NONMAIN_NAMES = ["server_nonmain_soft", "server_nonmain_hard"]


@torch.no_grad()
def server_nonmain_signals(server_logits, main_classes):
    """B: soft = mean(1 - sum_{j in Main} p_server(j)); C: hard = mean 1[argmax not in Main].
    main_classes = iterable of Main class IDs for this client (training metadata,
    NOT probe/eval labels). Multi-server logits are already averaged upstream."""
    q = F.softmax(server_logits, dim=1)
    idx = torch.tensor(sorted(int(c) for c in main_classes), device=q.device, dtype=torch.long) \
        if len(main_classes) else torch.tensor([], device=q.device, dtype=torch.long)
    if idx.numel() == 0:
        soft = 1.0
        hard = 1.0
    else:
        main_mass = q[:, idx].sum(dim=1)
        soft = (1.0 - main_mass).mean().item()
        top1 = q.argmax(dim=1)
        in_main = (top1.unsqueeze(1) == idx.unsqueeze(0)).any(dim=1)
        hard = (~in_main).float().mean().item()
    return {"server_nonmain_soft": float(min(max(soft, 0.0), 1.0)),
            "server_nonmain_hard": float(min(max(hard, 0.0), 1.0))}


def compute_client_signals(client_model, server_models, probe_x, device, rep_ref=None,
                           return_rep=False, main_classes=None):
    """Full pipeline for one client. Returns dict of RAW_SIGNAL_NAMES values
    (and the mean pooled representation vector if return_rep). If main_classes is
    given, also adds SERVER_NONMAIN_NAMES (Task 2)."""
    if probe_x is None or probe_x.shape[0] == 0:
        empty = {k: 0.0 for k in RAW_SIGNAL_NAMES}
        if main_classes is not None:
            empty.update({k: 0.0 for k in SERVER_NONMAIN_NAMES})
        return (empty, None) if return_rep else empty
    cl, sl, rp = probe_forward(client_model, server_models, probe_x, device)
    if rep_ref is not None and not rep_ref.frozen:
        rep_ref.accumulate(rp)
    sig = raw_signals(cl, sl, rp, rep_ref=rep_ref)
    if main_classes is not None:
        sig.update(server_nonmain_signals(sl, main_classes))
    if return_rep:
        return sig, rp.mean(dim=0).cpu().numpy()
    return sig


def aggregate_per_es(per_client_vals, clients):
    """Average a {cid: scalar} map over the clients of each ES."""
    per_es = {}
    for c in clients:
        if c.cid not in per_client_vals:
            continue
        for es_id in c.edge_server_ids:
            per_es.setdefault(es_id, []).append(per_client_vals[c.cid])
    return {es: float(np.mean(v)) for es, v in per_es.items()}
