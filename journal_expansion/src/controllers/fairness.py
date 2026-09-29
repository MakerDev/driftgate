"""F1/F2 — donor-selected, risk-directed inter-cell aggregation (Phase 5).

Why v5's fairness weighting failed (PATH2 report): it re-weighted ONE shared
global pool for everyone, so boosting the worst cell's inflow necessarily
polluted the best cells' inflow — best fell 7 pp, worst gained 0.9 pp.

This mechanism differs in two structural ways:
  1. PER-RECEIVER pools (F1): each cell z absorbs its own donor mixture
     pool_z = sum_k w_{z<-k} cell_k, with donor scores favoring
        - class-support complementarity  c_{z,k}: how much of what z LACKS
          (classes outside its scope) donor k covers — infrastructure
          knowledge (ES scopes), not runtime labels;
        - donor competence q_k: mean server confidence of donor cell;
        - representation proximity: exp(-alpha * d_rep(z,k)).
  2. RISK-DIRECTED absorption (F2): only high-risk (drift-elevated z-score)
     cells absorb more (Lambda_eff lowered); low-risk cells keep their own
     personalization untouched — the best cell is never dragged toward the pool.

Deployable version uses scopes + confidence + representation centroids (no
labels, no rho). Oracle-donor and random-donor variants exist for analysis.

Pre-registered constants: ALPHA_REP=1.0, BETA_Q=1.0, GAMMA_C=2.0,
ETA_ABSORB=0.15 (Lambda offset per unit of normalized risk), RISK_Z0=1.5.
"""
import numpy as np
import torch

ALPHA_REP = 1.0
BETA_Q = 1.0
GAMMA_C = 2.0
ETA_ABSORB = 0.15
RISK_Z0 = 1.5


def complementarity(scopes, num_classes):
    """c[z][k] = |scope_k covers of (all \\ scope_z)| / |all \\ scope_z|."""
    cells = sorted(scopes)
    allc = set(range(num_classes))
    c = {}
    for z in cells:
        lack = allc - set(scopes[z])
        c[z] = {}
        for k in cells:
            c[z][k] = (len(set(scopes[k]) & lack) / len(lack)) if lack else 0.0
    return c


def donor_weights(cells, rep_centroids, conf_per_es, comp, mode="deployable",
                  rng=None, oracle_rho=None):
    """w[z][k] over donors k != z (normalized)."""
    w = {}
    for z in cells:
        scores = {}
        for k in cells:
            if k == z:
                continue
            if mode == "random":
                scores[k] = rng.random()
                continue
            if mode == "oracle" and oracle_rho is not None:
                # oracle: donors with LOW true drift and high complementarity
                scores[k] = np.exp(GAMMA_C * comp[z][k] - 2.0 * oracle_rho.get(k, 0.0))
                continue
            d_rep = 0.0
            if rep_centroids and z in rep_centroids and k in rep_centroids:
                a, b = rep_centroids[z], rep_centroids[k]
                d_rep = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8))
            q = conf_per_es.get(k, 0.5) if conf_per_es else 0.5
            scores[k] = np.exp(-ALPHA_REP * d_rep + BETA_Q * q + GAMMA_C * comp[z][k])
        tot = sum(scores.values()) or 1.0
        w[z] = {k: v / tot for k, v in scores.items()}
    return w


def _mix(weighted_sds):
    """weighted_sds: list of (weight, state_dict); weights assumed normalized."""
    out = {k: torch.zeros_like(v).float() for k, v in weighted_sds[0][1].items()}
    for wgt, sd in weighted_sds:
        for k in out:
            out[k] += wgt * sd[k].float()
    return out


def apply_fairness_aggregation(edge_servers_dict, big_lambdas, risk_z_per_es,
                               rep_centroids=None, conf_per_es=None,
                               num_classes=10, mode="deployable", rng=None,
                               oracle_rho=None):
    """Replaces trainer.apply_network_aggregation for fairness runs.

    Lambda semantics preserved: new_cell = Lambda_eff * own + (1-Lambda_eff) * pool_z,
    but pool_z is the per-receiver donor mixture and Lambda_eff drops (absorbs
    more) only where risk z-score is elevated.
    Returns per-cell effective Lambda (for logging).
    """
    cells = sorted(edge_servers_dict)
    scopes = {z: edge_servers_dict[z].scope or set() for z in cells}
    comp = complementarity(scopes, num_classes)
    w = donor_weights(cells, rep_centroids, conf_per_es, comp, mode=mode,
                      rng=rng, oracle_rho=oracle_rho)

    own_c = {z: edge_servers_dict[z].clients_avg_weights for z in cells}
    own_s = {z: edge_servers_dict[z].server_avg_weights for z in cells}

    lam_eff_log = {}
    new_c, new_s = {}, {}
    for z in cells:
        Lam = big_lambdas.get(z, 0.5)
        # F2: risk-directed absorption — only drift-elevated cells absorb more
        risk = max(0.0, (risk_z_per_es.get(z, 0.0) - RISK_Z0))
        Lam_eff = max(0.0, Lam - ETA_ABSORB * min(risk, 3.0))
        lam_eff_log[z] = Lam_eff
        pool_c = _mix([(w[z][k], own_c[k]) for k in w[z]]) if w[z] else own_c[z]
        pool_s = _mix([(w[z][k], own_s[k]) for k in w[z]]) if w[z] else own_s[z]
        new_c[z] = {k: Lam_eff * own_c[z][k].float() + (1 - Lam_eff) * pool_c[k]
                    for k in own_c[z]}
        new_s[z] = {k: Lam_eff * own_s[z][k].float() + (1 - Lam_eff) * pool_s[k]
                    for k in own_s[z]}
    for z in cells:
        edge_servers_dict[z].clients_avg_weights = new_c[z]
        edge_servers_dict[z].server_avg_weights = new_s[z]
    return lam_eff_log
