"""Round 18 (own-class detector): tables and judgements from the per-run caches (decision_rule.md sections 1-3).

  python .../r18oc_tables.py select <model config>   # design selection on the S1 development seeds 8-10 (section 3)
  python .../r18oc_tables.py stage2                  # stage-2 cells, judgements (7.1-7.3), AUROC, extras, references
  python .../r18oc_tables.py holdout                 # cnn-scratch first day vs the v32 records (effect of the held-out 10%)
  python .../r18oc_tables.py attempt_dev <n> <cfg>   # 7.4 gate on the development seeds: frozen-replay, ratio 0.5, chosen variant
  python .../r18oc_tables.py attempt_cells <n> <cfg> # the four stage-2 decision cells of attempt n on the evaluation seeds
"""
import csv
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ART = HERE.parent
CACHE = HERE / "cache"
TABLES = HERE / "tables"
_spec = importlib.util.spec_from_file_location("an", HERE / "scripts" / "r18oc_analysis.py")
an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(an)
GRID, SCORES, ANSWERS, NEW, ORACLE = an.GRID, an.SCORES, an.ANSWERS, an.NEW, an.ORACLE
TAG = {"cnn-scratch": "cnn", "pretrained-finetuned-block": "ft", "pretrained-frozen-block": "fz"}
PRETRAINED = ["pretrained-finetuned-block", "pretrained-frozen-block"]
INPUTS, MODELS, NEGS = ("block", "exit", "concat"), ("logreg", "mlp64"), ("cell", "global", "both")
VARIANTS = [f"{i}|{m}|{n}" for i in INPUTS for m in MODELS for n in NEGS]
SEEDS = {"s1": [0, 1, 2], "s2": [0, 1, 2], "c100": [0, 1, 2]}
DEV = [8, 9, 10]
TIE_PP = 0.10
PASS_S, PASS_G, PART_S, PART_G = 0.25, 2.0, 0.15, 1.0
SETTING_NAME = {"s1": "S1", "s2": "S2", "c100": "CIFAR-100"}


def cache(cfg, st, seed, cond, att=None):
    f = CACHE / f"r18oc_{TAG[cfg]}{f'a{att}' if att else ''}_{st}_s{seed}__{cond}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def fmt(m, s=None, d=2):
    if m is None or (isinstance(m, float) and not np.isfinite(m)):
        return "none"
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ({s:.{d}f})"


def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def write(name, header, rows):
    TABLES.mkdir(exist_ok=True)
    assert len(set(header)) == len(header), header
    with open(TABLES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


# ------------------------------------------------------------------------------------------------ design selection
def complexity(v):
    i, m, n = v.split("|")
    return (MODELS.index(m), INPUTS.index(i), NEGS.index(n))


def select(cfg):
    rows, acc = [], {}
    for cond in ("frozen-replay", "first-day", "second-day"):
        O = [cache(cfg, "s1", s, cond) for s in DEV]
        if any(o is None for o in O):
            print(f"missing caches for {cfg} {cond}")
            if cond == "frozen-replay":
                return None
            continue
        for v in VARIANTS:
            j = list(GRID).index(0.5)
            vals = [100 * o["curves"][(v, NEW)][j] for o in O]
            if cond == "frozen-replay":
                acc[v] = float(np.mean(vals))
            rows.append([cfg, cond, v, fmt(float(np.mean(vals)), sd(vals)), " ".join(f"{x:.2f}" for x in vals),
                         fmt(float(np.mean([o["auroc"][v]["all"] for o in O])), d=3)])
    best = max(acc.values())
    cands = [v for v in VARIANTS if acc[v] >= best - TIE_PP - 1e-12]
    chosen = min(cands, key=complexity)
    write(f"R18oc_design_selection_{TAG[cfg]}.csv",
          ["model config", "condition", "variant", "accuracy at ratio 0.5 (detector rule) mean (SD)", "per seed (8, 9, 10)",
           "AUROC (all requests)"], rows)
    res = dict(model_config=cfg, chosen=chosen, best=max(acc, key=acc.get), best_acc=best, chosen_acc=acc[chosen],
               candidates_within_tie=cands, tie_pp=TIE_PP, acc_frozen_replay_ratio_0_5=acc, seeds=DEV)
    (TABLES / f"R18oc_design_selection_{TAG[cfg]}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(model_config=cfg, chosen=chosen, chosen_acc=round(acc[chosen], 3), best=res["best"],
                          best_acc=round(best, 3)), indent=1))
    return res


# ------------------------------------------------------------------------------------------------ cells
def ratio_at(acc, A):
    idx = np.flatnonzero(np.asarray(acc) >= A - 1e-12)
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(GRID[0])
    x0, x1, y0, y1 = GRID[i - 1], GRID[i], acc[i - 1], acc[i]
    return float(x0 + (A - y0) * (x1 - x0) / (y1 - y0))


def judge(S, G5):
    if S >= PASS_S or G5 >= PASS_G:
        return "pass"
    if S >= PART_S or G5 >= PART_G:
        return "partial"
    return "fail"


def cell(cfg, st, cond, seeds, main_override=None, att=None):
    O = [cache(cfg, st, s, cond, att) for s in seeds]
    if any(o is None for o in O):
        return None
    main = main_override or O[0]["main"]
    comp = [(s, a) for s in SCORES for a in ANSWERS]
    mean = lambda key: 100 * np.mean([o["curves"][key] for o in O], axis=0)
    per = lambda key: 100 * np.array([o["curves"][key] for o in O])
    C = {key: mean(key) for key in comp}
    env = np.max(np.stack([C[k] for k in comp]), axis=0)
    env_arg = [comp[int(i)] for i in np.argmax(np.stack([C[k] for k in comp]), axis=0)]
    B1 = env[-1]
    A = B1 - 0.5
    r_env = ratio_at(env, A)
    out = dict(cfg=cfg, setting=st, cond=cond, seeds=seeds, main=main, A=A, B1=B1, B1_rule=env_arg[-1], r_env=r_env)
    for nm, key in (("new", (main, NEW)), ("oracle", (ORACLE, "edge-raw"))):
        cur = mean(key)
        r = ratio_at(cur, A)
        S = 0.0 if r is None else 1.0 - r / r_env
        G = {rho: float(cur[list(GRID).index(rho)] - env[list(GRID).index(rho)]) for rho in (0.25, 0.5, 0.75)}
        Gsd = {}
        for rho in (0.25, 0.5, 0.75):
            j = list(GRID).index(rho)
            Gsd[rho] = sd(per(key)[:, j] - per(env_arg[j])[:, j])
        out[nm] = dict(curve=cur, ratio=r, S=S, G=G, G_sd=Gsd, acc={rho: float(cur[list(GRID).index(rho)]) for rho in (0.25, 0.5, 0.75)},
                       judgement=judge(S, G[0.5]))
    out["B"] = {rho: (float(env[list(GRID).index(rho)]), env_arg[list(GRID).index(rho)]) for rho in (0.25, 0.5, 0.75, 1.0)}
    out["room"] = out["oracle"]["judgement"] == "pass"
    out["border"] = bool(abs(out["new"]["S"] - PASS_S) < 0.01 or abs(out["new"]["S"] - PART_S) < 0.01
                         or abs(out["new"]["G"][0.5] - PASS_G) < 0.05 or abs(out["new"]["G"][0.5] - PART_G) < 0.05)
    out["O"] = O
    return out


def stage2_decision(cells):
    """decision_rule.md 2.3 (rows checked from the top)."""
    res = dict(per_config={}, rows=[])
    for cfg in PRETRAINED:
        four = [cells.get((cfg, st, cond)) for st in ("s1", "s2") for cond in ("second-day", "frozen-replay")]
        if any(c is None for c in four):
            res["per_config"][cfg] = dict(complete=False)
            continue
        roomy = [c for c in four if c["room"]]
        J = [c["new"]["judgement"] for c in roomy]
        c100 = [cells.get((cfg, "c100", cond)) for cond in ("second-day", "frozen-replay")]
        c100_pass = all(c is not None and c["new"]["judgement"] == "pass" for c in c100)
        if roomy and all(j == "pass" for j in J):
            row = 1
        elif roomy and all(j in ("pass", "partial") for j in J):
            row = 2
        elif roomy:
            row = 3
        elif c100_pass:
            row = 4
        else:
            row = 5
        res["per_config"][cfg] = dict(complete=True, judgements={f"{c['setting']} {c['cond']}": c["new"]["judgement"] for c in four},
                                      room={f"{c['setting']} {c['cond']}": c["room"] for c in four}, n_no_room=4 - len(roomy),
                                      c100_second_and_replay_pass=c100_pass, row=row)
    done = [cfg for cfg in PRETRAINED if res["per_config"][cfg].get("complete")]
    if len(done) < len(PRETRAINED):
        res["decision"] = "incomplete"
        return res
    best_row = min(res["per_config"][cfg]["row"] for cfg in PRETRAINED)
    follow = [cfg for cfg in PRETRAINED if res["per_config"][cfg]["row"] == best_row]
    res["row"] = best_row
    res["config"] = follow[0]   # finetuned first when both configurations are in the same row
    res["decision"] = {1: "stage 3 with this configuration as the main configuration",
                       2: "improvement attempt 1 once, then stage 3 with the better one on the development seeds",
                       3: "improvement attempts 1-3 in order until all four cells are partial or better, then stage 3",
                       4: "stage 3 with CIFAR-100 and iNaturalist as the main settings (S1, S2 as reference)",
                       5: "skip stage 3; stage 4 with the judgement 'stop'"}[best_row]
    res["some_cells_without_room"] = bool(res["per_config"][res["config"]]["n_no_room"] > 0)
    return res


def stage2():
    cells = {}
    cfgs = list(TAG)
    for cfg in cfgs:
        for st in ("s1", "s2", "c100"):
            if st == "c100" and cfg == "cnn-scratch":
                continue
            for cond in an.CONDITIONS:
                c = cell(cfg, st, cond, SEEDS[st])
                if c is not None:
                    cells[(cfg, st, cond)] = c
    rows, rows_au, rows_ref, rows_ex, rows_curve = [], [], [], [], []
    for (cfg, st, cond), c in cells.items():
        n, o = c["new"], c["oracle"]
        rows.append([SETTING_NAME[st], cfg, cond, c["main"], " ".join(map(str, c["seeds"])), fmt(c["A"]), "/".join(c["B1_rule"]),
                     fmt(n["ratio"], d=3), fmt(c["r_env"], d=3), fmt(n["S"], d=3),
                     fmt(n["G"][0.25], n["G_sd"][0.25]), fmt(n["G"][0.5], n["G_sd"][0.5]),
                     fmt(n["acc"][0.25]), fmt(n["acc"][0.5]), fmt(n["acc"][0.75]),
                     fmt(c["B"][0.25][0]), "/".join(c["B"][0.25][1]), fmt(c["B"][0.5][0]), "/".join(c["B"][0.5][1]),
                     fmt(c["B"][0.75][0]), "/".join(c["B"][0.75][1]),
                     fmt(o["ratio"], d=3), fmt(o["S"], d=3), fmt(o["G"][0.5]), "yes" if c["room"] else "no",
                     n["judgement"], "yes" if c["border"] else "no"])
        for score in [c["main"]] + SCORES + [ORACLE]:
            vals = [x["auroc"].get(score) for x in c["O"]]
            if any(v is None for v in vals):
                continue
            rows_au.append([SETTING_NAME[st], cfg, cond, score] + [fmt(float(np.mean([v[s] for v in vals])), sd([v[s] for v in vals]), 3)
                                                                  if s in vals[0] else "" for s in ("all", "home", "away")])
        for nm in ("Device only", "Edge only"):
            vv = [100 * x["reference"][nm] for x in c["O"]]
            rows_ref.append([SETTING_NAME[st], cfg, cond, nm, "", fmt(float(np.mean(vv)), sd(vv)), ""])
        vv = [100 * x["reference"]["Confidence-based offloading"]["acc"] for x in c["O"]]
        rr = [x["reference"]["Confidence-based offloading"]["ratio"] for x in c["O"]]
        rows_ref.append([SETTING_NAME[st], cfg, cond, "Confidence-based offloading", "", fmt(float(np.mean(vv)), sd(vv)), fmt(float(np.mean(rr)), d=3)])
        for beta in an.BETAS:
            for ans in ANSWERS:
                vv = [100 * x["reference"][f"entropy controller beta {beta}"]["acc"][ans] for x in c["O"]]
                rr = [x["reference"][f"entropy controller beta {beta}"]["ratio"] for x in c["O"]]
                rows_ref.append([SETTING_NAME[st], cfg, cond, f"entropy controller beta {beta}", ans, fmt(float(np.mean(vv)), sd(vv)),
                                 fmt(float(np.mean(rr)), d=3)])
        for v, ex in c["O"][0]["extra"].items():
            for k_ in ex:
                X_ = [x["extra"][v][k_] for x in c["O"]]
                acc = [100 * e["acc"] for e in X_] if "acc" in X_[0] else None
                ratio = [e["ratio"] for e in X_] if "ratio" in X_[0] else None
                se = [e["s_err"] for e in X_]
                rows_ex.append([SETTING_NAME[st], cfg, cond, v, k_, fmt(float(np.mean(acc)), sd(acc)) if acc else "",
                                fmt(float(np.mean(ratio)), d=3) if ratio else "",
                                fmt(float(np.nanmean([e["all"] for e in se])), d=3), fmt(float(np.nanmean([e["home"] for e in se])), d=3),
                                fmt(float(np.nanmean([e["away"] for e in se])), d=3), fmt(float(np.mean([e["defined_share"] for e in se])), d=3)])
        keys = sorted({k for x in c["O"] for k in x["curves"]}, key=str)
        for key in keys:
            if all(key in x["curves"] for x in c["O"]):
                cur = 100 * np.mean([x["curves"][key] for x in c["O"]], axis=0)
                rows_curve.append([SETTING_NAME[st], cfg, cond, key[0], key[1]] + [f"{v:.3f}" for v in cur])
    write("R18oc_stage2_cells.csv",
          ["setting", "model config", "condition", "detector variant", "evaluation seeds", "A* (%)", "B(1.0) rule",
           "ratio-at-target new", "ratio-at-target comparator envelope", "S", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)",
           "new accuracy at 0.25", "new accuracy at 0.5", "new accuracy at 0.75", "B(0.25)", "B(0.25) rule", "B(0.5)", "B(0.5) rule",
           "B(0.75)", "B(0.75) rule", "oracle ratio-at-target", "oracle S", "oracle G(0.5) pp", "room", "judgement", "border"], rows)
    write("R18oc_stage2_auroc.csv", ["setting", "model config", "condition", "score", "AUROC all", "AUROC home", "AUROC away"], rows_au)
    write("R18oc_stage2_reference.csv", ["setting", "model config", "condition", "rule", "answer rule", "accuracy mean (SD)",
                                         "offloading ratio"], rows_ref)
    write("R18oc_stage2_extra.csv", ["setting", "model config", "condition", "detector column", "operating point", "accuracy mean (SD)",
                                     "offloading ratio", "s-hat error all", "s-hat error home", "s-hat error away", "s-hat defined share"], rows_ex)
    write("R18oc_stage2_curves.csv", ["setting", "model config", "condition", "score or detector", "answer rule"]
          + [f"ratio {r:.2f}" for r in GRID], rows_curve)
    dec = stage2_decision(cells)
    dec["first_day_G05_below_minus_0_5"] = [f"{SETTING_NAME[st]} {cfg}" for (cfg, st, cond), c in cells.items()
                                            if cond == "first-day" and c["new"]["G"][0.5] < -0.5]
    (HERE / "decision_stage2.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1))


# ------------------------------------------------------------------------------------------------ held-out effect
def holdout():
    """cnn-scratch first day (held-out 10%, seeds 0-2) against the v32 records r12_s1_fixed040_s{0,1,2} (Round 20 cache)."""
    r20c = ART / "driftgate_tmc_r20_claims" / "cache"
    rows = []
    for st, pat in (("s1", "r12_s1_fixed040_s{}"), ("s2", "r12_s2_fixed040_s{}")):
        new = [cache("cnn-scratch", st, s, "first-day") for s in SEEDS[st]]
        if any(o is None for o in new):
            continue
        old = [pickle.load(open(r20c / f"{pat.format(s)}.pkl", "rb")) for s in SEEDS[st]]
        mac = lambda o, key: 100 * float(np.mean(an.macro_rounds(o["cnt"][key], o["n_ek"], o["present"])))
        pairs = [("Device only", lambda o: 100 * o["reference"]["Device only"], "Device only"),
                 ("Confidence-based offloading", lambda o: 100 * o["reference"]["Confidence-based offloading"]["acc"], "Confidence-based offloading"),
                 ("Raw edge only (ratio 1)", lambda o: 100 * o["curves"][("device-entropy", "edge-raw")][-1], "Raw edge only @ 1.0"),
                 ("Probability average (ratio 1)", lambda o: 100 * o["curves"][("device-entropy", "probability-average")][-1], "Probability average @ 1.0"),
                 ("Logit sum (ratio 1)", lambda o: 100 * o["curves"][("device-entropy", "logit-sum")][-1], "Logit sum @ 1.0"),
                 ("Corrected edge only (ratio 1)", lambda o: 100 * o["curves"][("device-entropy", "edge-corrected-even")][-1], "Corrected edge only @ 1.0"),
                 ("v32 method (ratio 1)", lambda o: 100 * o["curves"][("device-entropy", "average-corrected-even")][-1], "DriftGate @ 1.0")]
        for nm, fn, key in pairs:
            a = [fn(o) for o in new]
            b = [mac(o, key) for o in old]
            rows.append([SETTING_NAME[st], nm, fmt(float(np.mean(a)), sd(a)), fmt(float(np.mean(b)), sd(b)),
                         fmt(float(np.mean(np.subtract(a, b))), sd(np.subtract(a, b)))])
    write("R18oc_holdout_effect.csv", ["setting", "rule", "held-out 10% (this round, seeds 0-2) mean (SD)",
                                       "v32 records (seeds 0-2) mean (SD)", "difference pp mean (SD)"], rows)


# ------------------------------------------------------------------------------------------------ improvement attempts (7.4)
ATT_GAIN_PP = 0.5


def chosen_variant(cfg):
    return json.loads((TABLES / f"R18oc_design_selection_{TAG[cfg]}.json").read_text())["chosen"]


def attempt_dev(n, cfg):
    """decision_rule.md 2.4: the attempt passes to the evaluation seeds when the new method's seed-mean accuracy at ratio 0.5
    in frozen-replay (S1 development seeds 8-10, the configuration's chosen variant) is at least 0.5 pp above the run without
    the change (decisions.md D12)."""
    v = chosen_variant(cfg)
    j = list(GRID).index(0.5)
    rows, a, b = [], [], []
    for s in DEV:
        o0, o1 = cache(cfg, "s1", s, "frozen-replay"), cache(cfg, "s1", s, "frozen-replay", n)
        if o0 is None or o1 is None:
            print(f"missing caches for seed {s}")
            return None
        a.append(100 * o0["curves"][(v, NEW)][j])
        b.append(100 * o1["curves"][(v, NEW)][j])
        rows.append([cfg, n, v, s, f"{a[-1]:.2f}", f"{b[-1]:.2f}", f"{b[-1] - a[-1]:.2f}",
                     f"{o0['auroc'][v]['all']:.3f}", f"{o1['auroc'][v]['all']:.3f}"])
    d = np.subtract(b, a)
    rows.append([cfg, n, v, "mean (SD)", fmt(float(np.mean(a)), sd(a)), fmt(float(np.mean(b)), sd(b)), fmt(float(np.mean(d)), sd(d)), "", ""])
    write(f"R18oc_attempt{n}_{TAG[cfg]}_dev.csv", ["model config", "attempt", "detector variant", "seed",
                                                    "without the change: accuracy at 0.5 (%)", "attempt: accuracy at 0.5 (%)",
                                                    "difference pp", "AUROC without", "AUROC attempt"], rows)
    res = dict(model_config=cfg, attempt=n, variant=v, base=float(np.mean(a)), attempt_acc=float(np.mean(b)),
               gain_pp=float(np.mean(d)), gain_sd=sd(d), threshold_pp=ATT_GAIN_PP, go_to_evaluation=bool(np.mean(d) >= ATT_GAIN_PP))
    (TABLES / f"R18oc_attempt{n}_{TAG[cfg]}_dev.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def attempt_cells(n, cfg):
    rows, J = [], {}
    for st in ("s1", "s2"):
        for cond in an.CONDITIONS:
            c = cell(cfg, st, cond, SEEDS[st], att=n)
            if c is None:
                print(f"missing caches for {st} {cond}")
                continue
            nw, o = c["new"], c["oracle"]
            rows.append([SETTING_NAME[st], cfg, n, cond, c["main"], fmt(c["A"]), fmt(nw["ratio"], d=3), fmt(c["r_env"], d=3),
                         fmt(nw["S"], d=3), fmt(nw["G"][0.25], nw["G_sd"][0.25]), fmt(nw["G"][0.5], nw["G_sd"][0.5]),
                         fmt(nw["acc"][0.5]), fmt(c["B"][0.5][0]), "/".join(c["B"][0.5][1]), fmt(o["S"], d=3), fmt(o["G"][0.5]),
                         "yes" if c["room"] else "no", nw["judgement"], "yes" if c["border"] else "no"])
            if cond != "first-day":
                J[f"{st} {cond}"] = dict(judgement=nw["judgement"], room=c["room"])
    write(f"R18oc_attempt{n}_{TAG[cfg]}_cells.csv",
          ["setting", "model config", "attempt", "condition", "detector variant", "A* (%)", "ratio-at-target new",
           "ratio-at-target comparator envelope", "S", "G(0.25) pp mean (SD)", "G(0.5) pp mean (SD)", "new accuracy at 0.5",
           "B(0.5)", "B(0.5) rule", "oracle S", "oracle G(0.5) pp", "room", "judgement", "border"], rows)
    roomy = [x for x in J.values() if x["room"]]
    res = dict(model_config=cfg, attempt=n, cells=J, complete=len(J) == 4,
               all_partial_or_better=bool(len(J) == 4 and roomy and all(x["judgement"] in ("pass", "partial") for x in roomy)))
    (TABLES / f"R18oc_attempt{n}_{TAG[cfg]}_cells.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(sys.argv[2])
    elif sys.argv[1] == "stage2":
        stage2()
    elif sys.argv[1] == "holdout":
        holdout()
    elif sys.argv[1] == "attempt_dev":
        attempt_dev(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "attempt_cells":
        attempt_cells(int(sys.argv[2]), sys.argv[3])
