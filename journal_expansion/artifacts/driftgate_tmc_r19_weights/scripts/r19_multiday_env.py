"""Round 19 section 5: environment files of three new days for the multi-day frozen replay (protocol: R19_multiday_protocol.md).

S1 (synthetic): r6_env.build_synthetic("S1", env_seed) with env_seed = 9000 + 10 * seed + day (day = 1, 2, 3; not used
before). Homes follow the fixed topology, so devices, home cells and (with the run's partition seed) Main classes are
those of the checkpoint; plans, speeds, request counts and participation are new.
S2 (GeoLife): the same 50 users as S2_seed*.npz. For day d the d-th best other weekday of the user (same candidate rule
and score as scripts/r6_trace_env.py, the original day excluded; ties: earlier date). A user with fewer other days cycles
through them; a user with none repeats the original day (counted). Positions use the original planar origin and edge
positions; membership uses the original rule; home cell and class group are kept from the original file. Request counts
use r6_env.request_counts(env_seed, "trace", K, "normal") with env_seed = 9000 + 10 * seed + day.
Output: runs/phaseT19_multiday/env/{S1,S2}_s{seed}_day{d}.npz and S2_day_selection.csv. Existing files are never overwritten.
"""
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
JR = HERE.parents[1]
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

_spec = importlib.util.spec_from_file_location("tr", JR / "scripts" / "r6_trace_env.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)
OUT = JR / "runs" / "phaseT19_multiday" / "env"
DAYS = (1, 2, 3)


def env_seed(seed, day):
    return 9000 + 10 * seed + day


def save(path, env, meta):
    if path.exists():
        print(f"{path.name} exists; not overwritten")
        return
    r6_env.save_env(path, env, meta)


def build_s1(seeds=range(5)):
    for s in seeds:
        for d in DAYS:
            env, meta, plans, trip = r6_env.build_synthetic("S1", env_seed(s, d))
            meta = dict(meta, multiday=dict(base="S1", run_seed=s, day=d))
            orig, _ = r6_env.load_env(JR / "runs" / "phaseT6_env" / f"S1_seed{s}.npz")
            assert np.array_equal(orig["home_cell"], env["home_cell"]) and np.array_equal(orig["client_district"], env["client_district"])
            save(OUT / f"S1_s{s}_day{d}.npz", env, meta)
            print(f"S1 s{s} day{d}: at_home {env['at_home'].mean():.3f} two cells {(env['member'][..., 1] >= 0).mean():.3f}")


def build_s2(seeds=range(3)):
    sel = list(csv.DictReader(open(JR / "runs" / "phaseT6_env" / "S2_selection.csv")))
    orig, ometa = r6_env.load_env(JR / "runs" / "phaseT6_env" / "S2_seed0.npz")
    lat0, lon0 = ometa["lat0"], ometa["lon0"]
    centers = np.asarray(ometa["edge_xy_km"])
    kx = math.cos(math.radians(lat0)) * math.pi / 180 * tr.EARTH_KM
    ky = math.pi / 180 * tr.EARTH_KM
    import datetime
    rows, latlon_days = [], {d: [] for d in DAYS}
    reuse = {d: 0 for d in DAYS}
    for row in sel:
        uid = int(row["user"])
        orig_day = (datetime.date.fromisoformat(row["date"]) - datetime.date(1899, 12, 30)).days
        a = tr.clean(tr.load_user(tr.GEOLIFE / f"{uid:03d}"))
        day, minute, wd = tr.day_parts(a[:, 2])
        cands = []
        for dd in np.unique(day):
            m = day == dd
            if wd[m][0] >= 5 or int(dd) == orig_day:
                continue
            mins = minute[m]
            if not ((mins < 600).any() and (mins > 960).any()):
                continue
            cands.append((tr.score(mins), int(dd), a[m], minute[m]))
        cands.sort(key=lambda c: (-c[0], c[1]))
        for d in DAYS:
            if cands:
                c = cands[(d - 1) % len(cands)]
                if d > len(cands):
                    reuse[d] += 1
            else:
                m = day == orig_day
                c = (tr.score(minute[m]), orig_day, a[m], minute[m])
                reuse[d] += 1
            latlon_days[d].append(tr.positions(c[3], c[2][:, 0], c[2][:, 1]))
            rows.append(dict(client=row["client"], user=row["user"], day=d, date=(datetime.date(1899, 12, 30) + datetime.timedelta(days=c[1])).isoformat(),
                             score=c[0], other_weekdays_available=len(cands)))
    K, T, L = len(sel), r6_env.T_ROUNDS, len(centers)
    for d in DAYS:
        latlon = np.stack(latlon_days[d])
        pos = np.stack([(latlon[..., 1] - lon0) * kx, (latlon[..., 0] - lat0) * ky], axis=-1)
        dist = np.linalg.norm(pos[:, :, None, :] - centers[None, None], axis=3)
        order = np.argsort(dist, axis=2, kind="stable")
        d1 = np.take_along_axis(dist, order[..., :1], 2)[..., 0]
        d2 = np.take_along_axis(dist, order[..., 1:2], 2)[..., 0]
        member = np.stack([order[..., 0], np.where(d2 <= 1.2 * d1, order[..., 1], -1)], axis=2).astype(np.int16)
        home = orig["home_cell"]
        at_home = (member == home[:, None, None]).any(axis=2)
        rho = np.where(at_home, r6_env.RHO_HOME, r6_env.RHO_AWAY).astype(np.float32)
        for s in seeds:
            o_s, m_s = r6_env.load_env(JR / "runs" / "phaseT6_env" / f"S2_seed{s}.npz")
            mu, n_req = r6_env.request_counts(env_seed(s, d), "trace", K, "normal")
            env = {k: v for k, v in o_s.items()}
            env.update(pos=pos.astype(np.float32), member=member, at_home=at_home, rho=rho, n_req=n_req, mu=mu.astype(np.float32),
                       latlon=latlon.astype(np.float64))
            meta = dict(m_s, env_seed=env_seed(s, d), multiday=dict(base="S2", run_seed=s, day=d, users_reusing_a_day=reuse[d]))
            save(OUT / f"S2_s{s}_day{d}.npz", env, json.loads(json.dumps(meta)))
            print(f"S2 s{s} day{d}: at_home {at_home.mean():.3f} two cells {(member[..., 1] >= 0).mean():.3f} reuse {reuse[d]}")
    with open(OUT / "S2_day_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_s1()
    build_s2()
