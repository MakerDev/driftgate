"""Round 18 (own-class detector): stage-3 tables (directive 6.3, 6.4, 7.5), written before any stage-3 result is read.

  python .../r18oc_stage3.py tables <main cfg> <other pretrained cfg>   # every stage-3 table and decision_stage3.json
  python .../r18oc_stage3.py check                                       # cell_of() equals r18oc_tables.cell() on stage 2

Cells (7.1, 7.2) use the stage-2 definitions unchanged (r18oc_tables: ratio grid, envelope B over the 25 comparators,
A* = B(1.0) - 0.5, S, G, room). Seeds: S1 and S2 0-4, every other setting 0-2; cnn-scratch in S1, S2 and CIFAR-100.
Tables (tables/R18oc_stage3_*.csv):
  cells       setting x model configuration x condition: ratio-at-target, S, accuracy at 0.25/0.5/0.75, G(0.25), G(0.5),
              the strongest comparator at each ratio, oracle S and G(0.5), room, judgement, border
  curves      seed-mean curves of every rule
  auroc       non-Main detection (all, home, away) of every score
  breakdown   S1 and S2: accuracy at ratio 0.5 by request kind, by location, by time of day (three-hour blocks of the day)
              and of the bottom 10% of devices, for the new method, the v32 method, the strongest comparator at 0.5 and
              the oracle
  controller  budget controller beta 0.25 / 0.5: detector rule (g quantile) and the entropy controller with the five
              answer rules; accuracy and realised offloading ratio
  scale       200 and 500 devices: accuracy at 0.25/0.5/0.75 and ratio-at-target of every rule
  ablation    S1, S2, CIFAR-100, iNaturalist (S1 movement): negatives, update schedule, answer rule, inputs and models,
              held-out fraction 0.05/0.10/0.20, finetuned vs frozen block; S and G against the envelope of the same runs,
              and the same-seed difference from the main variant at 0.5
  cost        metric 9 per setting (definitions in cost_rows())
decision_stage3.json: the 7.5 judgement from the second-day and frozen-replay cells of the main configuration.
"""
import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tb", HERE / "scripts" / "r18oc_tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
an = T.an
GRID, SCORES, ANSWERS, NEW, NEW_AVG, ORACLE = an.GRID, an.SCORES, an.ANSWERS, an.NEW, an.NEW_AVG, an.ORACLE
CACHE, TABLES, RUNS = T.CACHE, T.TABLES, an.RUNS
TAG, fmt, sd, judge, ratio_at = T.TAG, T.fmt, T.sd, T.judge, T.ratio_at
V32 = ("device-entropy", "average-corrected-even")
J5 = list(GRID).index(0.5)

SETTINGS = ["s1", "s2", "s1fast", "part", "stepwise", "randmob", "c100", "k200", "k500", "inat1", "inat2"]
NAME = {"s1": "S1", "s2": "S2", "s1fast": "S1-fast", "part": "partial participation", "stepwise": "stepwise change",
        "randmob": "random mobility", "c100": "CIFAR-100", "k200": "200 devices", "k500": "500 devices",
        "inat1": "iNaturalist (S1 movement)", "inat2": "iNaturalist (S2 movement)"}
SEEDS = {st: ([0, 1, 2, 3, 4] if st in ("s1", "s2") else [0, 1, 2]) for st in SETTINGS}
CNN_SETTINGS = ("s1", "s2", "c100")
ABL_SETTINGS = ("s1", "s2", "c100", "inat1")
DECISION_SETTINGS = [st for st in SETTINGS if st not in ("inat1", "inat2")]   # 7.5: "3단계 설정"; iNaturalist judged apart


def run_name(cfg, st, seed, ho=None):
    h = f"_h{round(100 * ho):02d}" if ho is not None and abs(ho - 0.1) > 1e-9 else ""
    return f"r18oc_{TAG[cfg]}_{st}{h}_s{seed}"


def caches(cfg, st, cond, seeds, ho=None):
    O = []
    for s in seeds:
        f = CACHE / f"{run_name(cfg, st, s, ho)}__{cond}.pkl"
        if not f.exists():
            return None
        O.append(pickle.load(open(f, "rb")))
    return O


def write(name, header, rows):
    T.write(name, header, rows)


# ------------------------------------------------------------------------------------------------ cells
def cell_of(O, main=None):
    """r18oc_tables.cell() on given caches (same arithmetic; check() compares the two on the stage-2 caches)."""
    main = main or O[0]["main"]
    comp = [(s, a) for s in SCORES for a in ANSWERS]
    mean = lambda key: 100 * np.mean([o["curves"][key] for o in O], axis=0)
    per = lambda key: 100 * np.array([o["curves"][key] for o in O])
    C = {key: mean(key) for key in comp}
    env = np.max(np.stack([C[k] for k in comp]), axis=0)
    env_arg = [comp[int(i)] for i in np.argmax(np.stack([C[k] for k in comp]), axis=0)]
    A = env[-1] - 0.5
    r_env = ratio_at(env, A)
    out = dict(main=main, A=A, B1=env[-1], B1_rule=env_arg[-1], r_env=r_env, env=env, env_arg=env_arg, O=O)

    def rule(key):
        cur = mean(key)
        r = ratio_at(cur, A)
        S = 0.0 if r is None else 1.0 - r / r_env
        G, Gsd = {}, {}
        for rho in (0.25, 0.5, 0.75):
            j = list(GRID).index(rho)
            G[rho] = float(cur[j] - env[j])
            Gsd[rho] = sd(per(key)[:, j] - per(env_arg[j])[:, j])
        return dict(curve=cur, ratio=r, S=S, G=G, G_sd=Gsd, acc={rho: float(cur[list(GRID).index(rho)]) for rho in (0.25, 0.5, 0.75)},
                    judgement=judge(S, G[0.5]), per=per(key))

    out["rule"] = rule
    out["new"] = rule((main, NEW))
    out["oracle"] = rule((ORACLE, "edge-raw"))
    out["B"] = {rho: (float(env[list(GRID).index(rho)]), env_arg[list(GRID).index(rho)]) for rho in (0.25, 0.5, 0.75, 1.0)}
    out["room"] = out["oracle"]["judgement"] == "pass"
    n = out["new"]
    out["border"] = bool(abs(n["S"] - T.PASS_S) < 0.01 or abs(n["S"] - T.PART_S) < 0.01
                         or abs(n["G"][0.5] - T.PASS_G) < 0.05 or abs(n["G"][0.5] - T.PART_G) < 0.05)
    return out


def check():
    bad = 0
    for cfg in TAG:
        for st in ("s1", "s2", "c100"):
            for cond in an.CONDITIONS:
                c0 = T.cell(cfg, st, cond, T.SEEDS[st])
                if c0 is None:
                    continue
                c1 = cell_of(caches(cfg, st, cond, T.SEEDS[st]))
                same = (c0["A"] == c1["A"] and c0["r_env"] == c1["r_env"] and c0["new"]["S"] == c1["new"]["S"]
                        and c0["new"]["G"] == c1["new"]["G"] and c0["oracle"]["S"] == c1["oracle"]["S"]
                        and c0["new"]["judgement"] == c1["new"]["judgement"] and c0["room"] == c1["room"])
                bad += not same
                print(cfg, st, cond, "same" if same else "DIFFERENT")
    print("differences:", bad)


def cell_row(st, cfg, cond, seeds, c):
    n, o = c["new"], c["oracle"]
    return [NAME[st], cfg, cond, c["main"], " ".join(map(str, seeds)), fmt(c["A"]), "/".join(c["B1_rule"]),
            fmt(n["ratio"], d=3), fmt(c["r_env"], d=3), fmt(n["S"], d=3),
            fmt(n["G"][0.25], n["G_sd"][0.25]), fmt(n["G"][0.5], n["G_sd"][0.5]),
            fmt(n["acc"][0.25]), fmt(n["acc"][0.5]), fmt(n["acc"][0.75]),
            fmt(c["B"][0.25][0]), "/".join(c["B"][0.25][1]), fmt(c["B"][0.5][0]), "/".join(c["B"][0.5][1]),
            fmt(c["B"][0.75][0]), "/".join(c["B"][0.75][1]),
            fmt(o["ratio"], d=3), fmt(o["S"], d=3), fmt(o["G"][0.5]), "yes" if c["room"] else "no", n["judgement"],
            "yes" if c["border"] else "no"]


CELL_HEADER = ["setting", "model config", "condition", "detector variant", "evaluation seeds", "A* (%)", "B(1.0) rule",
               "ratio-at-target new", "ratio-at-target comparator envelope", "S", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)",
               "new accuracy at 0.25", "new accuracy at 0.5", "new accuracy at 0.75", "B(0.25)", "B(0.25) rule", "B(0.5)",
               "B(0.5) rule", "B(0.75)", "B(0.75) rule", "oracle ratio-at-target", "oracle S", "oracle G(0.5) pp", "room",
               "judgement", "border"]


# ------------------------------------------------------------------------------------------------ breakdowns
def n_kind(name, cond):
    """requests per (kind, evaluation round, device) of a run and condition (cached; the per-run caches keep only the
    correct counts per kind)."""
    f = CACHE / f"{name}__{cond}__nkind.npy"
    if f.exists():
        return np.load(f)
    tr, rp = an.run_names(name)
    D, X = an.load(rp if cond == "frozen-replay" else tr)
    if cond == "first-day":
        D, X = an.subset(D, X, D["er"] <= an.DAY)
    elif cond == "second-day":
        D, X = an.subset(D, X, D["er"] > an.DAY)
    out = np.stack([np.bincount(D["g"], weights=(D["kind"] == kk), minlength=D["E"] * D["K"]).reshape(D["E"], D["K"])
                    for kk in range(3)]).astype(np.int32)
    np.save(f, out)
    return out


def macro_mask(cnt, n, present, mask):
    """mean over evaluation rounds (with any device in mask) of the mean over devices in mask of cnt / n."""
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(present & mask & (n > 0), cnt / n, np.nan)
    per_round = np.nanmean(A, axis=1) if A.size else np.array([])
    return float(np.nanmean(per_round)) if np.isfinite(per_round).any() else float("nan")


def breakdown_rows(cfg, st, cond, c, seeds):
    rows = []
    rules = [("new method", (c["main"], NEW)), ("v32 method", V32), ("strongest comparator at 0.5", c["B"][0.5][1]),
             ("request-kind oracle", (ORACLE, "edge-raw"))]
    names = [run_name(cfg, st, s) for s in seeds]
    nk = [n_kind(nm, cond) for nm in names]
    for label, key in rules:
        res = {}
        for o, nks in zip(c["O"], nk):
            cnt = o["cnt"][key][J5].astype(float)
            n, pres, home = o["n_ek"], o["present"], o["home"] == 1
            r = {}
            for kk, kn in enumerate(("Main", "OOP", "OOR")):
                r[f"kind {kn}"] = macro_mask(o["cnt_kind"][key][0.5][kk].astype(float), nks[kk], pres, np.ones_like(pres, bool))
            r["location home"] = macro_mask(cnt, n, pres, home)
            r["location away"] = macro_mask(cnt, n, pres, ~home)
            day_r = (np.asarray(o["er"]) - 1) % an.DAY + 1
            clock = 5 * 60 + 6 * (day_r - 1)
            for h0 in (5, 8, 11, 14, 17):
                sel = (clock >= h0 * 60) & (clock < (h0 + 3) * 60)
                m = np.zeros_like(pres, bool)
                m[sel] = True
                r[f"time {h0:02d}:00-{h0 + 3:02d}:00"] = macro_mask(cnt, n, pres, m)
            with np.errstate(invalid="ignore", divide="ignore"):
                dev = np.nanmean(np.where(pres & (n > 0), cnt / n, np.nan), axis=0)
            dev = np.sort(dev[np.isfinite(dev)])
            r["bottom 10% devices"] = float(dev[:max(1, int(np.ceil(0.1 * len(dev))))].mean())
            for k_, v_ in r.items():
                res.setdefault(k_, []).append(100 * v_)
        for k_, vals in res.items():
            vals = np.asarray(vals, float)
            rows.append([NAME[st], cfg, cond, label, "/".join(key), k_, fmt(float(np.nanmean(vals)), sd(vals[np.isfinite(vals)]))])
    return rows


# ------------------------------------------------------------------------------------------------ cost (metric 9)
FLOAT = 4
LOCAL_EPOCHS = 3
_SIZES = {}


def model_sizes(cfg, dataset):
    """float32 bytes of one block output, of the device block and of the device exit (from the model code)."""
    key = (cfg, dataset)
    if key not in _SIZES:
        import torch
        sys.path.insert(0, str(an.JR.parent))
        sys.path.insert(0, str(an.JR))
        from src.r18oc_models import device_param_bytes, make_factory
        C = 100 if dataset in ("cifar100", "inat") else 10
        m = make_factory(cfg, C, dataset=dataset or "cifar10").make_client().eval()
        x = torch.zeros(2, 3, 96, 96, dtype=torch.uint8) if dataset == "inat" else torch.zeros(2, 3, 32, 32)
        with torch.no_grad():
            _, rep = m(x)
        _SIZES[key] = dict(block_out=FLOAT * int(np.prod(rep.shape[1:])), block=device_param_bytes(m, "block"),
                           exit=device_param_bytes(m, "exit"), C=C)
    return _SIZES[key]


def cost_rows(cfg, st, seeds, main):
    """Per setting (seed mean). Definitions (implementation choices, report_stage3.md):
      detector bytes            float32 parameters (logreg d + 1; mlp64 64 d + 64 + 64 + 1) + input standardisation (2 d)
      updates per device, day   detector trainings of a device in day 1 (rounds 1-150), mean over devices
      held-out uplink           held-out samples x d x 4 bytes per training round of a device
      device-exit uplink        device-exit parameter bytes per training round (only when the input uses the exit vector)
      training uplink           block output bytes x training samples x 3 local epochs + device block and exit bytes, per
                                training round of a device (split learning with model averaging)
      class statistics          classes x 2 (mean, diagonal variance) x d x 4 bytes per cell and training round
                                (only with global negatives)
      edge training time        bank training seconds per training round / number of trained variants (18, +2 schedules)
      device FLOPs              multiply-adds x 2 of the detector on one request vector
    """
    inp, mdl, neg = main.split("|")
    vals = {}
    for s in seeds:
        nm = run_name(cfg, st, s)
        Z = np.load(RUNS / f"{nm}_r18oc.npz")
        h = json.load(open(RUNS / f"{nm}.json"))
        dataset = {"c100": "cifar100", "inat1": "inat", "inat2": "inat"}.get(st, "cifar10")
        ms = model_sizes(cfg, dataset)
        db, de = (int(x) for x in Z["dims"])
        d = {"block": db, "exit": de, "concat": db + de}[inp]
        n_par = d + 1 if mdl == "logreg" else 64 * d + 64 + 64 + 1
        rl = Z["round_log"]
        K = int(h["K"])
        n_rounds = len(np.unique(rl[:, 0]))
        # round_log columns: round, client, n_pos (training samples of the device), ..., n_holdout_own, n_holdout_other
        tr_up = rl[:, 2].astype(float) * LOCAL_EPOCHS * ms["block_out"] + ms["block"] + ms["exit"]
        ho_up = rl[:, 6].astype(float) * d * FLOAT
        ex_up = np.full(len(rl), float(ms["exit"]) if inp in ("exit", "concat") else 0.0)
        ts = dict(zip([str(x) for x in Z["timing_cols"]], [float(x) for x in Z["timing_sec"]]))
        n_var = len(Z["variants"])
        add = lambda k, v: vals.setdefault(k, []).append(float(v))
        add("detector bytes", FLOAT * (n_par + 2 * d))
        add("detector input dimension", d)
        add("detector FLOPs per request", 2 * d if mdl == "logreg" else 2 * 64 * d + 2 * 64 + 64)
        add("detector updates per device and day", (rl[:, 0] <= an.DAY).sum() / K)
        add("held-out uplink bytes per device and training round", ho_up.mean())
        add("device-exit uplink bytes per device and training round", ex_up.mean())
        add("training uplink bytes per device and training round", tr_up.mean())
        add("held-out and device-exit uplink / training uplink", ((ho_up + ex_up) / tr_up).mean())
        add("class statistics bytes per cell and training round", FLOAT * 2 * d * ms["C"] if neg in ("global", "both") else 0)
        add("edge detector training seconds per training round and variant", ts["train"] / max(n_rounds, 1) / n_var)
        add("edge validation seconds per training round", ts["validation"] / max(n_rounds, 1))
        add("training run wall time (h)", h["total_time_sec"] / 3600)
    return [[NAME[st], cfg, main, k, fmt(float(np.mean(v)), d=4 if np.mean(v) < 1 else (1 if np.mean(v) < 1e4 else 0))]
            for k, v in vals.items()]


# ------------------------------------------------------------------------------------------------ tables
def tables(cfg_main, cfg_other):
    v_main = T.chosen_variant(cfg_main)
    cells = {}
    rows, rows_curve, rows_au, rows_bd, rows_ct, rows_sc, rows_ab, rows_cost = [], [], [], [], [], [], [], []
    for st in SETTINGS:
        cfgs = [cfg_main] + (["cnn-scratch"] if st in CNN_SETTINGS else []) + ([cfg_other] if st in ("s1", "s2", "c100", "inat1") else [])
        for cfg in cfgs:
            seeds = SEEDS[st] if cfg != cfg_other or st in ("inat1",) else T.SEEDS.get(st, [0, 1, 2])
            if cfg == cfg_other and st in ("s1", "s2"):
                seeds = [0, 1, 2]   # the other configuration has the stage-2 evaluation seeds only
            for cond in an.CONDITIONS:
                O = caches(cfg, st, cond, seeds)
                if O is None:
                    print(f"missing: {cfg} {st} {cond} seeds {seeds}")
                    continue
                c = cell_of(O)
                cells[(cfg, st, cond)] = (c, seeds)
                rows.append(cell_row(st, cfg, cond, seeds, c))
                for key in sorted({k for o in O for k in o["curves"]}, key=str):
                    if all(key in o["curves"] for o in O):
                        cur = 100 * np.mean([o["curves"][key] for o in O], axis=0)
                        rows_curve.append([NAME[st], cfg, cond, key[0], key[1]] + [f"{v:.3f}" for v in cur])
                for score in [c["main"], f"{c['main']}@frozen-at-home", f"{c['main']}@every-10"] + SCORES:
                    vals = [o["auroc"].get(score) for o in O]
                    if any(v is None for v in vals):
                        continue
                    rows_au.append([NAME[st], cfg, cond, score] + [fmt(float(np.mean([v[s] for v in vals])), sd([v[s] for v in vals]), 3)
                                                                   for s in ("all", "home", "away")])
                if cfg != cfg_main:
                    continue
                if st in ("s1", "s2"):
                    rows_bd += breakdown_rows(cfg, st, cond, c, seeds)
                for beta in an.BETAS:
                    ex = [o["extra"][c["main"]][f"controller {beta}"] for o in O]
                    acc, rat = [100 * e["acc"] for e in ex], [e["ratio"] for e in ex]
                    rows_ct.append([NAME[st], cfg, cond, beta, "detector g quantile", NEW, fmt(float(np.mean(acc)), sd(acc)),
                                    fmt(float(np.mean(rat)), d=3)])
                    ref = [o["reference"][f"entropy controller beta {beta}"] for o in O]
                    for ans in ANSWERS:
                        acc = [100 * r["acc"][ans] for r in ref]
                        rows_ct.append([NAME[st], cfg, cond, beta, "entropy controller", ans, fmt(float(np.mean(acc)), sd(acc)),
                                        fmt(float(np.mean([r["ratio"] for r in ref])), d=3)])
                if st in ("k200", "k500"):
                    keys = [(c["main"], NEW)] + [(s, a) for s in SCORES for a in ANSWERS] + [(ORACLE, "edge-raw")]
                    for key in keys:
                        r = c["rule"](key)
                        rows_sc.append([NAME[st], cfg, cond, "/".join(key), fmt(r["acc"][0.25]), fmt(r["acc"][0.5]), fmt(r["acc"][0.75]),
                                        fmt(r["ratio"], d=3), fmt(r["S"], d=3), fmt(r["G"][0.5], r["G_sd"][0.5])])
                if st in ABL_SETTINGS:
                    rows_ab += ablation_rows(cfg, st, cond, c, seeds, cfg_other)
        if (cfg_main, st, "first-day") in cells:
            try:
                rows_cost += cost_rows(cfg_main, st, SEEDS[st], v_main)
            except (FileNotFoundError, KeyError) as e:
                print(f"cost {st}: {e}")
    write("R18oc_stage3_cells.csv", CELL_HEADER, rows)
    write("R18oc_stage3_curves.csv", ["setting", "model config", "condition", "score or detector", "answer rule"]
          + [f"ratio {r:.2f}" for r in GRID], rows_curve)
    write("R18oc_stage3_auroc.csv", ["setting", "model config", "condition", "score", "AUROC all", "AUROC home", "AUROC away"], rows_au)
    write("R18oc_stage3_breakdown.csv", ["setting", "model config", "condition", "rule", "score/answer", "group",
                                         "accuracy at ratio 0.5 mean (SD)"], rows_bd)
    write("R18oc_stage3_controller.csv", ["setting", "model config", "condition", "beta", "controller", "answer rule",
                                          "accuracy mean (SD)", "offloading ratio"], rows_ct)
    write("R18oc_stage3_scale.csv", ["setting", "model config", "condition", "rule", "accuracy at 0.25", "accuracy at 0.5",
                                     "accuracy at 0.75", "ratio-at-target", "S", "G(0.5) pp mean (SD)"], rows_sc)
    write("R18oc_stage3_ablation.csv", ["setting", "model config", "condition", "ablation", "item", "accuracy at 0.25",
                                        "accuracy at 0.5", "accuracy at 0.75", "S", "G(0.5) pp", "difference from main at 0.5 pp mean (SD)",
                                        "envelope of"], rows_ab)
    write("R18oc_stage3_cost.csv", ["setting", "model config", "detector variant", "quantity", "seed mean"], rows_cost)
    dec = final_decision(cells, cfg_main)
    (HERE / "decision_stage3.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1))


def ablation_rows(cfg, st, cond, c, seeds, cfg_other):
    main = c["main"]
    inp, mdl, neg = main.split("|")
    rows = []
    base = c["new"]["per"][:, J5]

    def add(abl, item, key, cc=None, env_of="same runs"):
        cc = cc or c
        if not all(key in o["curves"] for o in cc["O"]):
            return
        r = cc["rule"](key)
        diff = r["per"][:, J5] - base if cc is c else None
        rows.append([NAME[st], cfg, cond, abl, item, fmt(r["acc"][0.25]), fmt(r["acc"][0.5]), fmt(r["acc"][0.75]), fmt(r["S"], d=3),
                     fmt(r["G"][0.5]), fmt(float(np.mean(diff)), sd(diff)) if diff is not None else "", env_of])

    for n in T.NEGS:
        add("negatives", {"cell": "detector-cell-negatives", "global": "detector-global-negatives", "both": "both"}[n],
            (f"{inp}|{mdl}|{n}", NEW))
    add("update", "detector-frozen-at-home", (f"{main}@frozen-at-home", NEW))
    add("update", "detector-update-every-10-rounds", (f"{main}@every-10", NEW))
    add("update", "every round", (main, NEW))
    for ans in ("edge-raw", "edge-corrected-even", NEW, NEW_AVG):
        add("answer rule", ans, (main, ans))
    for v in T.VARIANTS:
        add("input and model", v, (v, NEW))
    for ho in (0.05, 0.2):
        O = caches(cfg, st, cond, seeds if st not in ("s1", "s2") else [0, 1, 2], ho)
        if O is not None:
            cc = cell_of(O, main)
            add("held-out fraction", f"{ho:.2f}", (main, NEW), cc, env_of=f"held-out {ho:.2f} runs")
    O10 = caches(cfg, st, cond, [0, 1, 2])
    if O10 is not None:
        add("held-out fraction", "0.10 (seeds 0-2)", (main, NEW), cell_of(O10, main), env_of="held-out 0.10 runs, seeds 0-2")
    Oo = caches(cfg_other, st, cond, [0, 1, 2])
    if Oo is not None:
        co = cell_of(Oo)
        add("finetuned vs frozen block", f"{cfg_other} ({co['main']})", (co["main"], NEW), co, env_of=f"{cfg_other} runs, seeds 0-2")
        add("finetuned vs frozen block", f"{cfg} ({main}, seeds 0-2)", (main, NEW), cell_of(O10, main) if O10 else None,
            env_of=f"{cfg} runs, seeds 0-2")
    return rows


# ------------------------------------------------------------------------------------------------ 7.5
def final_decision(cells, cfg):
    J = {}
    for st in DECISION_SETTINGS:
        for cond in ("second-day", "frozen-replay"):
            if (cfg, st, cond) in cells:
                c = cells[(cfg, st, cond)][0]
                J[f"{NAME[st]} {cond}"] = dict(judgement=c["new"]["judgement"], room=c["room"], border=c["border"],
                                               S=c["new"]["S"], G05=c["new"]["G"][0.5])
    inat = {}
    for st in ("inat1", "inat2"):
        for cond in ("second-day", "frozen-replay"):
            if (cfg, st, cond) in cells:
                c = cells[(cfg, st, cond)][0]
                inat[f"{NAME[st]} {cond}"] = dict(judgement=c["new"]["judgement"], room=c["room"], border=c["border"],
                                                  S=c["new"]["S"], G05=c["new"]["G"][0.5])
    js = [x["judgement"] for x in J.values()]
    n = len(js)
    n_pass, n_part = sum(j == "pass" for j in js), sum(j in ("pass", "partial") for j in js)
    k200 = [x["judgement"] for k, x in J.items() if k.startswith(NAME["k200"])]
    inat_ok = bool(inat) and all(x["judgement"] in ("pass", "partial") for x in inat.values())
    recommend = n > 0 and n_pass >= 2 * n / 3 and n_part == n and inat_ok and k200 and all(j != "fail" for j in k200)
    conditional = n > 0 and n_part >= n / 2
    verdict = "TMC 진행 권고" if recommend else ("조건부 권고" if conditional else "중단 권고")
    return dict(model_config=cfg, cells=J, inat=inat, n_cells=n, n_pass=n_pass, n_partial_or_better=n_part,
                inat_partial_or_better=inat_ok, k200=k200, verdict=verdict,
                border_cells=[k for k, x in {**J, **inat}.items() if x["border"]],
                cells_without_room=[k for k, x in {**J, **inat}.items() if not x["room"]],
                below_strongest_comparator=[k for k, x in {**J, **inat}.items() if x["G05"] < 0])


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "tables":
        tables(sys.argv[2], sys.argv[3])
