"""Round 18 (own-class detector): detectors trained at the edge, validation statistics and request scores.

Directive 3.1-3.2 and 4.1. One DetectorBank per run; the runner calls
  begin_round()                     before the training step
  post_local_train(client)          right after a device's local training (before any aggregation)
  end_round(r, ...)                 after aggregation and mixing (installed models)
  begin_eval(k, model) / batch(client_logits, rep) / end_eval()   while a device is evaluated
Every random draw uses the bank's own torch.Generator (seeded from the run seed, round and device), so training,
evaluation requests and arrival orders are the same as without the bank. Evaluation labels never enter the bank: the
evaluation hook receives only the device logits and the block output of each batch.

Detector g_k(x) = P(x comes from a class of M_k), computed on the device from a vector z_k(x):
  inputs   block   device-block output averaged over space (from the device's own block)
           exit    input of the last linear layer of the device exit (device k's installed exit applied to a block output)
           concat  [block, exit]
  models   logreg  logistic regression;  mlp64  one hidden layer of 64 ReLU units
  negatives cell   training samples of the other participating devices in the device's current cells whose label is not
                   in M_k (block outputs of those devices' own blocks); at most NEG_CAP, a seeded random subset beyond it
           global  N_VIRTUAL virtual vectors per class absent from the device's cells (and not in M_k), drawn from the
                   class mean and diagonal variance over all participating devices' training vectors of the round
           both    cell + global
Positives are the device's own training samples (block output of the device's model at the end of its local training,
eval mode). Positive and negative weights are balanced (0.5 each in total). Inputs are standardised with the mean and SD
of all training rows of the device (positives, cell and virtual negatives). Training: full-batch Adam, STEPS steps,
learning rate ADAM_LR, L2 penalty L2 on the weights, fresh initialisation every round (logistic regression: zeros; MLP:
PyTorch's default uniform ranges, seeded). A negative set that is empty in a round keeps the device's previous detector
for that variant. A device without any detector gets g = -1 (offloads everything).

Validation (3.2): after the round's models are installed, every participating device forwards its held-out samples; own =
g_k on device k's held-out vectors, other = g_k on the other participating devices' held-out vectors whose label is not
in M_k; each a 64-bin histogram on [0, 1].
Schedules for the main variant (directive 6.3 ablation): frozen-at-home = the main detector (with its statistics) last
trained while the device was at home; every-10 = the main detector trained at rounds 1, 11, 21, ...
Request scores (4.1): device energy -logsumexp(logits); own-mahalanobis and own-knn on the block vector of the installed
model (Round 17 D1 and D2: class means of M_k, shared class-centred covariance / n plus 1e-3 trace / d, minimum distance;
cosine distance to the 10th nearest own training vector).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.r18oc_models import block_vector, exit_vector

INPUTS = ("block", "exit", "concat")
MODELS = ("logreg", "mlp64")
NEGS = ("cell", "global", "both")
VARIANTS = [f"{i}|{m}|{n}" for i in INPUTS for m in MODELS for n in NEGS]
SCHEDULES = ("frozen-at-home", "every-10")
NBINS = 64
N_VIRTUAL = 64
NEG_CAP = 8192
STEPS = 300
ADAM_LR = 0.01
L2 = 1e-3
HIDDEN = 64
KNN_K = 10
EPS_FACTOR = 1e-3
CHUNK_ROWS = 160_000      # devices per training chunk: sum of padded rows <= CHUNK_ROWS
FWD_BATCH = 512


def _hist(g):
    b = torch.clamp((g * NBINS).long(), 0, NBINS - 1)
    return torch.bincount(b, minlength=NBINS)


class DetectorBank:
    def __init__(self, K, C, mains, train_idx, holdout_idx, X, Y, seed, device, main_variant=None):
        self.K, self.C, self.seed, self.device = K, C, int(seed), device
        self.M = torch.zeros(K, C, dtype=torch.bool, device=device)
        for k, v in mains.items():
            self.M[int(k), sorted(v)] = True
        self.X, self.Y = X, Y
        self.tr = {k: torch.as_tensor(np.asarray(v, np.int64), device=device) for k, v in train_idx.items()}
        self.ho = {k: torch.as_tensor(np.asarray(v, np.int64), device=device) for k, v in holdout_idx.items()}
        self.main = main_variant
        assert main_variant is None or main_variant in VARIANTS, main_variant
        self.V = len(VARIANTS) + (len(SCHEDULES) if main_variant else 0)
        self.names = VARIANTS + ([f"{main_variant}@{s}" for s in SCHEDULES] if main_variant else [])
        # det[k][(input, model)] = dict(mu [3, d], sd [3, d], params..., valid [3] bool, round [3] int)
        self.det = {k: {} for k in range(K)}
        self.hist_own = torch.zeros(K, len(VARIANTS), NBINS, dtype=torch.int32)
        self.hist_other = torch.zeros(K, len(VARIANTS), NBINS, dtype=torch.int32)
        self.snap = {s: {k: None for k in range(K)} for s in SCHEDULES}   # dict(input, model, slot params, hist_own, hist_other, round)
        self.round_log = []        # per training round: (r, k, n_pos, n_cell_all, n_cell_used, n_virtual, n_ho_own, n_ho_other)
        self.tsec = dict(features=0.0, train=0.0, validation=0.0, eval=0.0)
        self.dims = None
        self.buf = None
        self._eval = None
        self._own_stats = {}       # replay cache: k -> own-class statistics of the frozen model
        self.other = False         # attempt 1 (exit-with-other-class): set by the runner; P(other) is appended to every input
        self.cls_stats = None      # attempt 1: (mu, sd, have) of the exit vectors per class from the last training round
        self.rounds_begun = 0

    # ------------------------------------------------------------------ helpers
    def _gen(self, *parts):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(np.random.SeedSequence([self.seed, 18, *[int(p) for p in parts]]).generate_state(1)[0]))
        return g

    @torch.no_grad()
    def _reps(self, model, idx):
        model.eval()   # every forward of the bank is in eval mode (batch-norm statistics are never updated here)
        out = []
        for s in range(0, len(idx), FWD_BATCH):
            _, rep = model(self.X[idx[s:s + FWD_BATCH]])
            out.append(rep.half())
        return torch.cat(out) if out else None

    @torch.no_grad()
    def _vectors(self, model, reps):
        """block and exit vectors (float32) of block outputs `reps` with `model`'s exit."""
        model.eval()
        vb, ve = [], []
        if reps is None or len(reps) == 0:
            db, de = self.dims
            z = torch.zeros(0, db, device=self.device)
            return z, torch.zeros(0, de, device=self.device)
        for s in range(0, len(reps), FWD_BATCH):
            r_ = reps[s:s + FWD_BATCH].float()
            vb.append(block_vector(r_))
            e_ = exit_vector(model, r_)
            ve.append(torch.cat([e_, self._p_other(model, e_)], 1) if self.other else e_)
        return torch.cat(vb), torch.cat(ve)

    def _p_other(self, model, e_):
        """attempt 1: probability of the "other" output of the device exit for exit vectors e_ (column vector)."""
        return torch.softmax(model.exit_fc(e_), 1)[:, -1:]

    def _cols(self, inp):
        db, de = self.dims
        if self.other:   # de includes the P(other) column, which every input gets
            return {"block": list(range(db)) + [db + de - 1], "exit": slice(db, db + de), "concat": slice(0, db + de)}[inp]
        return {"block": slice(0, db), "exit": slice(db, db + de), "concat": slice(0, db + de)}[inp]

    # ------------------------------------------------------------------ training-round hooks
    def begin_round(self):
        self.buf = {}
        self.rounds_begun += 1

    @torch.no_grad()
    def post_local_train(self, c):
        t0 = time.perf_counter()
        m = c.client_model
        was = m.training
        m.eval()
        self.buf[c.cid] = self._reps(m, self.tr[c.cid])
        m.train(was)
        self.tsec["features"] += time.perf_counter() - t0

    @torch.no_grad()
    def end_round(self, r, by_id, at_home):
        """r: 1-indexed round; by_id: {k: client} with installed models; at_home: bool array [K] of this round."""
        P = sorted(self.buf)
        if not P:
            return
        t0 = time.perf_counter()
        models = {k: by_id[k].client_model for k in P}
        for m in models.values():
            m.eval()
        lab = {k: self.Y[self.tr[k]] for k in P}
        own = {}
        for k in P:
            vb, ve = self._vectors(models[k], self.buf[k])
            own[k] = torch.cat([vb, ve], 1)
        if self.dims is None:
            self.dims = (vb.shape[1], ve.shape[1])
        d = sum(self.dims)
        # global class statistics (mean, diagonal variance) of the own vectors of all participating devices
        S1 = torch.zeros(self.C, d, device=self.device, dtype=torch.float64)
        S2 = torch.zeros_like(S1)
        N = torch.zeros(self.C, device=self.device, dtype=torch.float64)
        for k in P:
            S1.index_add_(0, lab[k], own[k].double())
            S2.index_add_(0, lab[k], own[k].double() ** 2)
            N += torch.bincount(lab[k], minlength=self.C).double()
        have = N > 1
        mu_c = (S1 / N.clamp(min=1)[:, None])
        var_c = (S2 / N.clamp(min=1)[:, None] - mu_c ** 2).clamp(min=0)
        mu_c, sd_c = mu_c.float(), var_c.sqrt().float()
        if self.other:   # attempt 1: the edge sends the exit-vector statistics with the round's models
            db_, de_ = self.dims
            self.cls_stats = (mu_c[:, db_:db_ + de_ - 1].clone(), sd_c[:, db_:db_ + de_ - 1].clone(), have.clone())
        # cells of each participating device and its participating cell mates
        cells = {k: list(by_id[k].edge_server_ids) for k in P}
        members = {}
        for k in P:
            for z in cells[k]:
                members.setdefault(z, []).append(k)
        rows = {}
        for k in P:
            mates = sorted({j for z in cells[k] for j in members[z] if j != k})
            present = torch.zeros(self.C, dtype=torch.bool, device=self.device)
            present[lab[k]] = True
            pos_j, owner_j = [], []
            for j in mates:
                present[lab[j]] = True
                sel = torch.nonzero(~self.M[k][lab[j]]).flatten()
                pos_j.append(sel)
                owner_j.append(torch.full((len(sel),), j, device=self.device))
            neg_n = int(sum(len(p) for p in pos_j))
            if neg_n:
                pos_all, own_all = torch.cat(pos_j), torch.cat(owner_j)
                if neg_n > NEG_CAP:   # seeded random subset; the order of the kept rows follows the mates and samples
                    keep = torch.randperm(neg_n, generator=self._gen(r, k, 1), device=self.device)[:NEG_CAP].sort().values
                    pos_all, own_all = pos_all[keep], own_all[keep]
                neg_rep = torch.cat([self.buf[j][pos_all[own_all == j]] for j in mates])
                vb, ve = self._vectors(models[k], neg_rep)
                cellv = torch.cat([vb, ve], 1)
                del neg_rep
            else:
                cellv = torch.zeros(0, d, device=self.device)
            absent = torch.nonzero(~present & ~self.M[k] & have).flatten()
            if len(absent):
                gz = torch.randn(len(absent), N_VIRTUAL, d, generator=self._gen(r, k, 2), device=self.device)
                virt = (mu_c[absent][:, None, :] + sd_c[absent][:, None, :] * gz).reshape(-1, d)
                if self.other:   # P(other) of a virtual vector from device k's exit, not drawn from the statistics
                    db_ = self.dims[0]
                    virt[:, -1:] = self._p_other(models[k], virt[:, db_:d - 1])
            else:
                virt = torch.zeros(0, d, device=self.device)
            rows[k] = (own[k], cellv, virt)
            self.round_log.append((r, k, len(own[k]), neg_n, len(cellv), len(virt), len(self.ho[k]), 0))
        self.tsec["features"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        self._train(r, P, rows)
        self.tsec["train"] += time.perf_counter() - t1
        t2 = time.perf_counter()
        self._validate(r, P, models, at_home)
        self.tsec["validation"] += time.perf_counter() - t2
        self.buf = None

    def _train(self, r, P, rows):
        # chunks of devices by padded size
        sizes = {k: sum(len(x) for x in rows[k]) for k in P}
        order = sorted(P, key=lambda k: sizes[k])
        chunks, cur, cur_max = [], [], 0
        for k in order:
            nm = max(cur_max, sizes[k])
            if cur and nm * (len(cur) + 1) > CHUNK_ROWS:
                chunks.append(cur)
                cur, nm = [], sizes[k]
            cur.append(k)
            cur_max = nm
        if cur:
            chunks.append(cur)
        for ch in chunks:
            B, Nmax = len(ch), max(sizes[k] for k in ch)
            d = sum(self.dims)
            X = torch.zeros(B, Nmax, d, device=self.device)
            y = torch.zeros(B, Nmax, device=self.device)
            Wt = torch.zeros(B, 3, Nmax, device=self.device)
            ok = torch.zeros(B, 3, dtype=torch.bool)
            for b, k in enumerate(ch):
                o, cv, vv = rows[k]
                no, nc, nv = len(o), len(cv), len(vv)
                X[b, :no + nc + nv] = torch.cat([o, cv, vv])
                y[b, :no] = 1.0
                Wt[b, :, :no] = 0.5 / no
                for s, (use_c, use_v) in enumerate(((True, False), (False, True), (True, True))):
                    nn_ = (nc if use_c else 0) + (nv if use_v else 0)
                    if nn_ == 0:
                        continue
                    ok[b, s] = True
                    if use_c:
                        Wt[b, s, no:no + nc] = 0.5 / nn_
                    if use_v:
                        Wt[b, s, no + nc:no + nc + nv] = 0.5 / nn_
            valid_rows = torch.zeros(B, Nmax, device=self.device)
            for b, k in enumerate(ch):
                valid_rows[b, :sizes[k]] = 1.0
            for inp in INPUTS:
                cs = self._cols(inp)
                Xi = X[:, :, cs]
                cnt = valid_rows.sum(1, keepdim=True)
                mu = (Xi * valid_rows[..., None]).sum(1) / cnt
                sd = (((Xi - mu[:, None]) ** 2 * valid_rows[..., None]).sum(1) / cnt).sqrt().clamp(min=1e-6)
                Xs = (Xi - mu[:, None]) / sd[:, None] * valid_rows[..., None]
                params = self._fit(r, ch, inp, Xs, y, Wt)
                for b, k in enumerate(ch):
                    for mdl in MODELS:
                        ent = self.det[k].setdefault((inp, mdl), None)
                        new = {nm: v[b].detach().clone() for nm, v in params[mdl].items()}
                        dd = Xi.shape[2]
                        if ent is None:
                            ent = dict(mu=torch.zeros(3, dd, device=self.device), sd=torch.ones(3, dd, device=self.device),
                                       valid=torch.zeros(3, dtype=torch.bool), round=torch.full((3,), -1, dtype=torch.int32),
                                       **{nm: torch.zeros_like(v) for nm, v in new.items()})
                            self.det[k][(inp, mdl)] = ent
                        for s in range(3):
                            if ok[b, s]:
                                ent["mu"][s], ent["sd"][s] = mu[b], sd[b]
                                for nm, v in new.items():
                                    ent[nm][s] = v[s]
                                ent["valid"][s] = True
                                ent["round"][s] = r
            del X, Wt

    def _fit(self, r, ch, inp, Xs, y, Wt):
        B, Nmax, d = Xs.shape
        dev = self.device
        lr_w = torch.zeros(B, 3, d, device=dev, requires_grad=True)
        lr_b = torch.zeros(B, 3, device=dev, requires_grad=True)
        W1 = torch.empty(B, 3, d, HIDDEN, device=dev)
        b1 = torch.empty(B, 3, HIDDEN, device=dev)
        w2 = torch.empty(B, 3, HIDDEN, device=dev)
        b2 = torch.empty(B, 3, device=dev)
        ii = INPUTS.index(inp)
        for b, k in enumerate(ch):
            g = self._gen(r, k, 10 + ii)
            a1, a2 = 1.0 / np.sqrt(d), 1.0 / np.sqrt(HIDDEN)
            W1[b].uniform_(-a1, a1, generator=g)
            b1[b].uniform_(-a1, a1, generator=g)
            w2[b].uniform_(-a2, a2, generator=g)
            b2[b].uniform_(-a2, a2, generator=g)
        mlp = [t.requires_grad_(True) for t in (W1, b1, w2, b2)]
        opt = torch.optim.Adam([lr_w, lr_b] + mlp, lr=ADAM_LR)
        yb = y[:, None, :].expand(B, 3, Nmax)
        with torch.enable_grad():
            for _ in range(STEPS):
                opt.zero_grad(set_to_none=True)
                z_lr = torch.einsum("bnd,bsd->bsn", Xs, lr_w) + lr_b[..., None]
                h = torch.relu(torch.bmm(Xs, W1.permute(0, 2, 1, 3).reshape(B, d, 3 * HIDDEN)).view(B, Nmax, 3, HIDDEN)
                               + b1[:, None])
                z_mlp = torch.einsum("bnsh,bsh->bsn", h, w2) + b2[..., None]
                loss = (Wt * F.binary_cross_entropy_with_logits(z_lr, yb, reduction="none")).sum() \
                    + (Wt * F.binary_cross_entropy_with_logits(z_mlp, yb, reduction="none")).sum() \
                    + L2 * ((lr_w ** 2).sum() + (W1 ** 2).sum() + (w2 ** 2).sum())
                loss.backward()
                opt.step()
        return {"logreg": dict(w=lr_w.detach(), b=lr_b.detach()),
                "mlp64": dict(W1=W1.detach(), b1=b1.detach(), w2=w2.detach(), b2=b2.detach())}

    # ------------------------------------------------------------------ scoring
    def _score_group(self, ent, mdl, x):
        """g [n, 3] of one (input, model) group for vectors x [n, d]."""
        xs = (x[:, None, :] - ent["mu"][None]) / ent["sd"][None]          # [n, 3, d]
        if mdl == "logreg":
            z = (xs * ent["w"][None]).sum(-1) + ent["b"][None]
        else:
            h = torch.relu(torch.einsum("nsd,sdh->nsh", xs, ent["W1"]) + ent["b1"][None])
            z = (h * ent["w2"][None]).sum(-1) + ent["b2"][None]
        g = torch.sigmoid(z)
        return torch.where(ent["valid"].to(g.device)[None], g, torch.full_like(g, -1.0))

    def score(self, k, v):
        """g [n, 18] of device k's current detectors for vectors v [n, db + de] (VARIANTS order)."""
        out = torch.full((len(v), len(VARIANTS)), -1.0, device=self.device)
        for ii, inp in enumerate(INPUTS):
            x = v[:, self._cols(inp)]
            for mi, mdl in enumerate(MODELS):
                ent = self.det[k].get((inp, mdl))
                if ent is None:
                    continue
                j = (ii * len(MODELS) + mi) * 3
                out[:, j:j + 3] = self._score_group(ent, mdl, x)
        return out

    def _score_snap(self, sn, v):
        inp, mdl, s = sn["inp"], sn["mdl"], sn["slot"]
        ent = {kk: (vv[s:s + 1] if torch.is_tensor(vv) and vv.dim() >= 1 else vv) for kk, vv in sn["ent"].items()}
        return self._score_group(ent, mdl, v[:, self._cols(inp)])[:, 0]

    # ------------------------------------------------------------------ validation statistics
    @torch.no_grad()
    def _validate(self, r, P, models, at_home):
        ho_rep = {k: self._reps(models[k], self.ho[k]) for k in P}
        ho_lab = {k: self.Y[self.ho[k]] for k in P}
        allrep = torch.cat([ho_rep[k] for k in P])
        allab = torch.cat([ho_lab[k] for k in P])
        owner = torch.cat([torch.full((len(self.ho[k]),), k, device=self.device) for k in P])
        for k in P:
            vb, ve = self._vectors(models[k], ho_rep[k])
            g_own = self.score(k, torch.cat([vb, ve], 1))
            sel = (owner != k) & ~self.M[k][allab]
            vb, ve = self._vectors(models[k], allrep[sel])
            g_oth = self.score(k, torch.cat([vb, ve], 1))
            for j in range(len(VARIANTS)):
                if g_own[0, j] >= 0:
                    self.hist_own[k, j] = _hist(g_own[:, j]).cpu().int()
                    self.hist_other[k, j] = _hist(g_oth[:, j]).cpu().int()
            self.round_log[-len(P) + P.index(k)] = self.round_log[-len(P) + P.index(k)][:7] + (int(sel.sum()),)
            if self.main is not None:
                inp, mdl, neg = self.main.split("|")
                s = NEGS.index(neg)
                ent = self.det[k].get((inp, mdl))
                if ent is not None and bool(ent["valid"][s]) and int(ent["round"][s]) == r:
                    j = VARIANTS.index(self.main)
                    sn = dict(inp=inp, mdl=mdl, slot=s, ent={kk: vv.clone() for kk, vv in ent.items()},
                              hist_own=self.hist_own[k, j].clone(), hist_other=self.hist_other[k, j].clone(), round=r)
                    if bool(at_home[k]):
                        self.snap["frozen-at-home"][k] = sn
                    if (r - 1) % 10 == 0:
                        self.snap["every-10"][k] = sn

    # ------------------------------------------------------------------ evaluation hooks
    @torch.no_grad()
    def _own_class_stats(self, k, model):
        """Round 17 D1/D2 statistics on the block vectors of device k's training samples (installed model)."""
        vb, _ = self._vectors(model, self._reps(model, self.tr[k]))
        vb = vb.double()
        lab = self.Y[self.tr[k]]
        assert bool(self.M[k][lab].all()), "training labels outside M_k"
        cls = torch.nonzero(self.M[k]).flatten()
        mus = torch.stack([vb[lab == c].mean(0) for c in cls])
        cen = vb - mus[torch.searchsorted(cls, lab)]
        S = cen.T @ cen / len(vb)
        S = S + EPS_FACTOR * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        Pm = torch.linalg.inv(S)
        T = F.normalize(vb.float(), dim=1)
        return dict(mus=mus, P=Pm, T=T)

    def begin_eval(self, k, model, replay=False):
        t0 = time.perf_counter()
        model.eval()
        if replay and k in self._own_stats:
            st = self._own_stats[k]
        else:
            st = self._own_class_stats(k, model)
            if replay:
                self._own_stats[k] = st
        self._eval = dict(k=k, model=model, st=st, g=[], en=[], ma=[], kn=[])
        self.tsec["eval"] += time.perf_counter() - t0

    @torch.no_grad()
    def batch(self, client_logits, rep):
        t0 = time.perf_counter()
        ev = self._eval
        k, model, st = ev["k"], ev["model"], ev["st"]
        vb, ve = block_vector(rep), exit_vector(model, rep.float())
        if self.other:
            ve = torch.cat([ve, self._p_other(model, ve)], 1)
        if self.dims is None:
            self.dims = (vb.shape[1], ve.shape[1])
        v = torch.cat([vb, ve], 1)
        g = self.score(k, v)
        if self.main is not None:
            j = VARIANTS.index(self.main)
            extra = []
            for s in SCHEDULES:
                sn = self.snap[s][k]
                extra.append(g[:, j] if sn is None else self._score_snap(sn, v))
            g = torch.cat([g, torch.stack(extra, 1)], 1)
        ev["g"].append(g.half().cpu())
        ev["en"].append((-torch.logsumexp(client_logits.float(), 1)).cpu())
        x = vb.double()
        dif = x[:, None, :] - st["mus"][None]
        ev["ma"].append(torch.einsum("ncd,de,nce->nc", dif, st["P"], dif).min(1).values.float().cpu())
        sim = F.normalize(vb.float(), dim=1) @ st["T"].T
        kk = min(KNN_K, sim.shape[1])
        ev["kn"].append((1.0 - sim.topk(kk, dim=1).values[:, kk - 1]).cpu())
        self.tsec["eval"] += time.perf_counter() - t0

    def end_eval(self):
        ev = self._eval
        self._eval = None
        out = dict(g=torch.cat(ev["g"]).numpy(), energy=torch.cat(ev["en"]).numpy().astype(np.float32),
                   maha=torch.cat(ev["ma"]).numpy().astype(np.float32), knn=torch.cat(ev["kn"]).numpy().astype(np.float32))
        return out

    def eval_state(self, k):
        """hist_own, hist_other [V, 64] and detector round [V] in use for device k (VARIANTS + schedules)."""
        ho, hx = [self.hist_own[k]], [self.hist_other[k]]
        rd = []
        for inp in INPUTS:
            for mdl in MODELS:
                ent = self.det[k].get((inp, mdl))
                rd += ([-1] * 3 if ent is None else [int(x) if bool(v) else -1 for x, v in zip(ent["round"], ent["valid"])])
        ho, hx = ho[0], hx[0]
        if self.main is not None:
            j = VARIANTS.index(self.main)
            eho, ehx = [], []
            for s in SCHEDULES:
                sn = self.snap[s][k]
                eho.append(self.hist_own[k, j] if sn is None else sn["hist_own"])
                ehx.append(self.hist_other[k, j] if sn is None else sn["hist_other"])
                rd.append(rd[j] if sn is None else sn["round"])
            ho, hx = torch.cat([ho, torch.stack(eho)]), torch.cat([hx, torch.stack(ehx)])
        return ho.numpy(), hx.numpy(), np.array(rd, np.int16)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self):
        cpu = lambda e: {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv) for kk, vv in e.items()}
        return dict(det={k: {key: cpu(e) for key, e in v.items()} for k, v in self.det.items()},
                    hist_own=self.hist_own.clone(), hist_other=self.hist_other.clone(), dims=self.dims, main=self.main,
                    snap={s: {k: (None if sn is None else dict(sn, ent=cpu(sn["ent"]))) for k, sn in v.items()}
                          for s, v in self.snap.items()})

    def load_state_dict(self, st):
        dev = lambda e: {kk: (vv.to(self.device) if torch.is_tensor(vv) and vv.dtype != torch.bool and kk not in ("round",)
                              else vv) for kk, vv in e.items()}
        self.det = {int(k): {key: dev(e) for key, e in v.items()} for k, v in st["det"].items()}
        self.hist_own, self.hist_other, self.dims = st["hist_own"].clone(), st["hist_other"].clone(), st["dims"]
        assert st["main"] == self.main, (st["main"], self.main)
        self.snap = {s: {int(k): (None if sn is None else dict(sn, ent=dev(sn["ent"]))) for k, sn in v.items()}
                     for s, v in st["snap"].items()}


# ------------------------------------------------------------------------------------------------ attempt 1
def train_one_round_other(self, method="splitomcplus", gamma=0.5, fedprox_mu=0.01, bank=None):
    """attempt 1 (`exit-with-other-class`), bound to a SplitOMCClient: the splitomcplus local round of
    train/trainer.SplitOMCClient.train_one_round with the device exit's (C+1)-way logits in the multi-exit loss, plus
    gamma x CE(exit_fc(v), "other") on as many virtual exit vectors per batch as real samples, drawn uniformly over the
    classes outside M_k (class mean and diagonal SD of the exit vectors from the edge, previous round; none in round 1)."""
    from models.losses import multi_exit_loss
    assert method == "splitomcplus"
    m = self.client_model
    m.train()
    for sm in self.server_models.values():
        sm.train()
    opt_c = torch.optim.SGD(m.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
    opt_s = {z: torch.optim.SGD(sm.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
             for z, sm in self.server_models.items()}
    st = bank.cls_stats
    cls = None
    if st is not None:
        mu, sd, have = st
        cls = torch.nonzero(have & ~bank.M[self.cid]).flatten()
        if len(cls) == 0:
            cls = None
    gen = bank._gen(bank.rounds_begun, self.cid, 30)
    epoch_losses = []
    for _ in range(self.local_epochs):
        bl = []
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            opt_c.zero_grad()
            for o in opt_s.values():
                o.zero_grad()
            rep = m.block_forward(images)
            full = m.exit_fc(m.exit_vec(rep))
            loss = multi_exit_loss(full, [self.server_models[z](rep)[0] for z in self.edge_server_ids], labels, gamma=gamma)
            if cls is not None:
                pick = cls[torch.randint(len(cls), (len(labels),), generator=gen, device=self.device)]
                v = mu[pick] + sd[pick] * torch.randn(len(labels), mu.shape[1], generator=gen, device=self.device)
                loss = loss + gamma * F.cross_entropy(m.exit_fc(v), torch.full((len(labels),), m.C, device=self.device))
            loss.backward()
            opt_c.step()
            for o in opt_s.values():
                o.step()
            bl.append(loss.item())
        epoch_losses.append(np.mean(bl) if bl else 0.0)
    return np.mean(epoch_losses) if epoch_losses else 0.0
