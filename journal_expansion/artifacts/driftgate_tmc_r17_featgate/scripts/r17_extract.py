"""Round 17 stage 0: frozen-model replay of the Round 15 end-of-day checkpoints with feature records.

One process = one replay. The Round 15 replay command is re-run unchanged (src/runner_r6.run_r6 with --replay_from,
same flags) into runs/phaseT17_featgate, with three read-only hooks installed from outside (no change to the runner):
  - eval_client is wrapped: after the unchanged call, for every request of a recorded evaluation the device-block
    output rep [128, 8, 8] is pooled over space (f_d, 128 values) and, for every cell server model of the device,
    the server-block output in front of the server exit (relu(fc2), 128 values; f_e) is kept, in request order;
  - build_partition, use_cached_loaders, build_clients_and_es and load_env are wrapped to keep their return values
    (training indices, Main classes, training images, clients, cells, environment).
After the replay (models are frozen), the own training samples are passed through each device's block (f_d of every
training sample of every device, with labels) and, for every cell, the training samples of its residents (home cell)
through each resident's own block and the cell's server block (f_e). Outputs (gitignored):
  {run}.json / _trace.npz / _evalprobs.npz   the replay record (same format as Round 15)
  {run}_features.npz                         fd [N, 128] f16, fe [N, 2, 128] f16, fe_cell [N, 2] (-1 = none)
  {run}_own.npz                              own_fd/own_k/own_y (devices), cell_fe/cell_z/cell_k/cell_y (cells),
                                             home_cell, main [K, C]
  {run}_extract.json                         timings and shapes
Usage: python r17_extract.py --scenario S1 --seed 0 --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
ROOT = JR.parent
for p in (str(ROOT), str(JR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src.runner_r6 as R6  # noqa: E402

OUT = JR / "runs" / "phaseT17_featgate"
CKPT = JR / "runs" / "phaseT15_replay"
CAP, FEAT = {}, []


def server_feature(sm, rep):
    """CIFARServer.forward up to the input of fc3 (the server exit)."""
    x = F.relu(sm.bn(sm.conv(rep)))
    x = F.max_pool2d(x, 2).flatten(1)
    x = F.relu(sm.fc1(x))
    return F.relu(sm.fc2(x))


def install_hooks():
    orig_eval = R6.eval_client

    def eval_client(client, X, Y, idx, main, oop, oor, eth=0.8, batch_size=128, mainaware=False, record=None,
                    record_probs=False, record_logprobs=False):
        out = orig_eval(client, X, Y, idx, main, oop, oor, eth=eth, batch_size=batch_size, mainaware=mainaware,
                        record=record, record_probs=record_probs, record_logprobs=record_logprobs)
        if record is None:
            return out
        cells = list(client.server_models.keys())
        sms = list(client.server_models.values())
        assert len(sms) <= 2
        fd, fe = [], [[] for _ in sms]
        with torch.no_grad():
            idx_t = torch.as_tensor(np.asarray(idx), device=X.device, dtype=torch.long)
            for s in range(0, len(idx_t), batch_size):
                logits, rep = client.client_model(X[idx_t[s:s + batch_size]])
                fd.append(rep.float().mean((2, 3)).half().cpu())
                for j, sm in enumerate(sms):
                    h = server_feature(sm, rep)
                    if s == 0 and "fc3_check" not in CAP:     # the hook reproduces the server exit
                        CAP["fc3_check"] = float((sm.fc3(h) - sm(rep)[0]).abs().max())
                    fe[j].append(h.half().cpu())
        n = len(idx)
        FEAT.append(dict(n=n, fd=torch.cat(fd).numpy() if n else np.zeros((0, 128), np.float16),
                         fe=[torch.cat(f).numpy() for f in fe] if n else [], cells=cells))
        return out

    R6.eval_client = eval_client
    for name in ("build_partition", "use_cached_loaders", "build_clients_and_es", "load_env"):
        orig = getattr(R6, name)

        def wrap(*a, _orig=orig, _name=name, **kw):
            r = _orig(*a, **kw)
            CAP[_name] = r
            if _name == "build_clients_and_es":
                CAP["model_factory"] = kw.get("model_factory")
            return r
        setattr(R6, name, wrap)


def own_features(device):
    indices, mains, cell_groups, all_used = CAP["build_partition"]
    Xtr, Ytr = CAP["use_cached_loaders"]
    clients, ES = CAP["build_clients_and_es"]
    env, _ = CAP["load_env"]
    mf = CAP["model_factory"]
    K = len(clients)
    C = int(Ytr.max().item()) + 1
    home = np.asarray(env["home_cell"]).astype(np.int64)
    main = np.zeros((K, C), bool)
    for k in range(K):
        main[k, sorted(mains[k])] = True
    own_fd, own_k, own_y = [], [], []
    reps = {}
    with torch.no_grad():
        for c in clients:
            c.client_model.eval()
            idx = torch.as_tensor(np.asarray(indices[c.cid], dtype=np.int64), device=device)
            fs, rs = [], []
            for s in range(0, len(idx), 512):
                _, rep = c.client_model(Xtr[idx[s:s + 512]])
                fs.append(rep.float().mean((2, 3)).cpu())
                rs.append(rep)
            reps[c.cid] = torch.cat(rs)
            own_fd.append(torch.cat(fs).numpy())
            own_k.append(np.full(len(idx), c.cid, np.int16))
            own_y.append(Ytr[idx].cpu().numpy().astype(np.int16))
        cell_fe, cell_z, cell_k, cell_y = [], [], [], []
        for z in sorted(ES):
            sm = mf.make_server().to(device)
            sm.load_state_dict(ES[z].server_avg_weights)
            sm.eval()
            for k in np.flatnonzero(home == z):
                rep = reps[int(k)]
                fe = torch.cat([server_feature(sm, rep[s:s + 512]).cpu() for s in range(0, len(rep), 512)]).numpy()
                cell_fe.append(fe)
                cell_z.append(np.full(len(fe), z, np.int16))
                cell_k.append(np.full(len(fe), k, np.int16))
                cell_y.append(own_y[int(k)])
    return dict(own_fd=np.concatenate(own_fd).astype(np.float32), own_k=np.concatenate(own_k), own_y=np.concatenate(own_y),
                cell_fe=np.concatenate(cell_fe).astype(np.float32), cell_z=np.concatenate(cell_z),
                cell_k=np.concatenate(cell_k), cell_y=np.concatenate(cell_y), home_cell=home, main=main)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["S1", "S2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    name = f"r17_{a.scenario.lower()}_replay_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{name}_extract.json").exists():
        print(f"{name} done; not re-running")
        return
    ckpt = CKPT / f"r15_{a.scenario.lower()}_fixed040_s{a.seed}_ckpt.pt"
    cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))
    cfg.update(device=a.device, partition_seed=a.seed, model_seed=100 + a.seed, global_rounds=150)
    env_path = JR / "runs" / "phaseT6_env" / f"{a.scenario}_seed{a.seed}.npz"
    install_hooks()
    t0 = time.time()
    R6.run_r6(cfg, env_path, "fixed", signal="tv_dist", lambda_val=0.4, big_lambda_val=0.5, apfl_eta=None,
              signal_delay=0, probe_n=64, controller_kwargs={}, run_name=name, output_dir=str(OUT), eval_every=5,
              save_models=False, scenario=a.scenario, arm="fixed040", record_device_signals=True,
              record_eval_probs=True, record_train_label_hist=True, record_eval_logprobs=True, replay_from=str(ckpt))
    t_replay = time.time() - t0
    q = np.load(OUT / f"{name}_evalprobs.npz")
    N = len(q["req_label"])
    assert sum(f["n"] for f in FEAT) == N, (sum(f["n"] for f in FEAT), N)
    fd = np.concatenate([f["fd"] for f in FEAT])
    fe = np.zeros((N, 2, 128), np.float16)
    fe_cell = np.full((N, 2), -1, np.int16)
    pos = 0
    for f in FEAT:
        for j, (c, v) in enumerate(zip(f["cells"], f["fe"])):
            fe[pos:pos + f["n"], j] = v
            fe_cell[pos:pos + f["n"], j] = c
        pos += f["n"]
    np.savez(OUT / f"{name}_features.npz", fd=fd, fe=fe, fe_cell=fe_cell)
    t1 = time.time()
    own = own_features(a.device)
    np.savez(OUT / f"{name}_own.npz", **own)
    t_own = time.time() - t1
    meta = dict(run=name, checkpoint=str(ckpt), n_requests=int(N), replay_seconds=t_replay, own_feature_seconds=t_own,
                fc3_check_max_abs=CAP.get("fc3_check"), own_samples=int(len(own["own_k"])), cell_samples=int(len(own["cell_z"])),
                device=a.device, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / f"{name}_extract.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
