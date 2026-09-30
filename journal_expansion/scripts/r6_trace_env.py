"""Round 6 S2: real mobility traces -> environment files (R6 directive §S2).

GeoLife Trajectories 1.3 (expected under data/traces/, i.e. <DATA_ROOT>/traces/):
  1. GMT -> Beijing time (UTC+8)
  2. keep lon in [116.0, 116.8], lat in [39.6, 40.2]; then, per user in time order,
     drop a point whose speed from the previously KEPT point exceeds 50 m/s
     (points with a duplicate timestamp are dropped)
  3. weekdays only; a user-day is a candidate if it has records before 10:00 and after 16:00
  4. score = number of the 150 round starts (05:00 + 6(r-1) min) with a record within +-15 min;
     best day per user (ties: earlier date), then the top 50 (ties: smaller user id, earlier date)
  5. the 50 user-days are overlaid on one day, one client each
  6. round-start position: linear interpolation if the enclosing records are <= 30 min apart,
     otherwise the last record's position; before the first / after the last record: that record
  7. planar km around the mean latitude; k-means (L=5, n_init=10, random_state=0) on all
     client-round positions gives the edge positions
  8. member = nearest edge (+ the second nearest if its distance <= 1.2 x the nearest); all edge
     pairs are neighbours
  9. home cell = nearest edge at round 1; cells ranked by resident count get class groups 0..4
 10. requests / rho as in S1; paths are the same for every seed, env_seed drives request counts only
If fewer than 50 GeoLife candidates exist, the directive's T-Drive branch applies (not implemented
until needed; the script stops and says so).

  python journal_expansion/scripts/r6_trace_env.py            # writes S2_seed{0,1,2}.npz + selection CSV
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

JR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JR.parent))
sys.path.insert(0, str(JR))
from src import r6_env  # noqa: E402

TRACE_ROOT = JR.parent / "data" / "traces"
GEOLIFE = TRACE_ROOT / "Geolife Trajectories 1.3" / "Data"
OUT = JR / "runs" / "phaseT6_env"
LON, LAT = (116.0, 116.8), (39.6, 40.2)
VMAX = 50.0          # m/s
N_SELECT = 50
L_EDGES = 5
EARTH_KM = 6371.0088
ROUND_STARTS = np.array([r6_env.round_start_min(r) for r in range(1, r6_env.T_ROUNDS + 1)])


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * 1000 * math.asin(min(1.0, math.sqrt(a)))


def load_user(udir):
    arrs = []
    for f in sorted((udir / "Trajectory").glob("*.plt")):
        a = np.loadtxt(f, delimiter=",", skiprows=6, usecols=(0, 1, 4), ndmin=2)
        if a.size:
            arrs.append(a)
    if not arrs:
        return np.zeros((0, 3))
    a = np.concatenate(arrs)
    a[:, 2] += 8.0 / 24.0                     # GMT -> Beijing (days since 1899-12-30)
    return a[np.argsort(a[:, 2], kind="stable")]


def clean(a):
    """bbox, then sequential speed filter against the last kept point."""
    m = (a[:, 1] >= LON[0]) & (a[:, 1] <= LON[1]) & (a[:, 0] >= LAT[0]) & (a[:, 0] <= LAT[1])
    a = a[m]
    keep = []
    last = None
    for i, (lat, lon, d) in enumerate(a.tolist()):
        if last is None:
            keep.append(i)
            last = (lat, lon, d)
            continue
        dt = (d - last[2]) * 86400.0
        if dt <= 0:
            continue
        if haversine_m(last[0], last[1], lat, lon) / dt > VMAX:
            continue
        keep.append(i)
        last = (lat, lon, d)
    return a[keep]


def day_parts(days):
    day = np.floor(days).astype(np.int64)
    minute = (days - day) * 1440.0
    weekday = (day + 5) % 7                  # 1899-12-30 (day 0) was a Saturday; Mon = 0
    return day, minute, weekday


def score(minutes):
    """rounds whose start has a record within +-15 min."""
    ms = np.sort(minutes)
    pos = np.searchsorted(ms, ROUND_STARTS)
    ok = np.zeros(len(ROUND_STARTS), bool)
    for side in (pos - 1, pos):
        valid = (side >= 0) & (side < len(ms))
        ok |= valid & (np.abs(ms[np.clip(side, 0, len(ms) - 1)] - ROUND_STARTS) <= 15.0)
    return int(ok.sum())


def positions(minutes, lat, lon):
    out = np.zeros((len(ROUND_STARTS), 2))
    for r, s in enumerate(ROUND_STARTS):
        j = np.searchsorted(minutes, s, side="right")      # records[:j] have minute <= s
        if j == 0:
            out[r] = (lat[0], lon[0])
        elif j == len(minutes):
            out[r] = (lat[-1], lon[-1])
        else:
            t0, t1 = minutes[j - 1], minutes[j]
            if t1 - t0 <= 30.0:
                f = (s - t0) / (t1 - t0) if t1 > t0 else 0.0
                out[r] = (lat[j - 1] + f * (lat[j] - lat[j - 1]), lon[j - 1] + f * (lon[j] - lon[j - 1]))
            else:
                out[r] = (lat[j - 1], lon[j - 1])
    return out


def main():
    if not GEOLIFE.exists():
        raise SystemExit(f"GeoLife not found at {GEOLIFE}. Put 'Geolife Trajectories 1.3' (unzipped) under data/traces/.")
    cands = []            # (score, user, day, record array)
    n_points = n_kept = 0
    for udir in sorted(GEOLIFE.iterdir()):
        uid = int(udir.name)
        a = load_user(udir)
        n_points += len(a)
        a = clean(a)
        n_kept += len(a)
        if not len(a):
            continue
        day, minute, wd = day_parts(a[:, 2])
        best = None
        for d in np.unique(day):
            m = day == d
            if wd[m][0] >= 5:
                continue
            mins = minute[m]
            if not ((mins < 600).any() and (mins > 960).any()):
                continue
            sc = score(mins)
            if best is None or sc > best[0]:
                best = (sc, uid, int(d), a[m], minute[m])
        if best is not None:
            cands.append(best)
    print(f"GeoLife points {n_points:,}, kept after filters {n_kept:,}, candidate users {len(cands)}")
    if len(cands) < N_SELECT:
        raise SystemExit(f"only {len(cands)} GeoLife candidates (< {N_SELECT}): the T-Drive branch of §S2 applies")
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    sel = cands[:N_SELECT]
    latlon = np.stack([positions(c[4], c[3][:, 0], c[3][:, 1]) for c in sel])    # [K, T, 2] (lat, lon)
    lat0, lon0 = float(latlon[..., 0].mean()), float(latlon[..., 1].mean())
    kx = math.cos(math.radians(lat0)) * math.pi / 180 * EARTH_KM
    ky = math.pi / 180 * EARTH_KM
    pos = np.stack([(latlon[..., 1] - lon0) * kx, (latlon[..., 0] - lat0) * ky], axis=-1)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=L_EDGES, n_init=10, random_state=0).fit(pos.reshape(-1, 2))
    centers = km.cluster_centers_
    d = np.linalg.norm(pos[:, :, None, :] - centers[None, None], axis=3)       # [K, T, L]
    order = np.argsort(d, axis=2, kind="stable")
    d1 = np.take_along_axis(d, order[..., :1], 2)[..., 0]
    d2 = np.take_along_axis(d, order[..., 1:2], 2)[..., 0]
    member = np.stack([order[..., 0], np.where(d2 <= 1.2 * d1, order[..., 1], -1)], axis=2).astype(np.int16)
    home = member[:, 0, 0].astype(int)
    residents = np.bincount(home, minlength=L_EDGES)
    rank = sorted(range(L_EDGES), key=lambda z: (-residents[z], z))
    group_of_cell = {z: g for g, z in enumerate(rank)}
    client_group = np.array([group_of_cell[h] for h in home])
    at_home = (member == home[:, None, None]).any(axis=2)
    rho = np.where(at_home, r6_env.RHO_HOME, r6_env.RHO_AWAY).astype(np.float32)
    K = N_SELECT
    OUT.mkdir(parents=True, exist_ok=True)
    sel_rows = []
    for k, c in enumerate(sel):
        import datetime
        date = (datetime.date(1899, 12, 30) + datetime.timedelta(days=c[2])).isoformat()
        sel_rows.append(dict(client=k, user=f"{c[1]:03d}", date=date, score=c[0], n_records=len(c[3]),
                             home_cell=int(home[k]), class_group=int(client_group[k])))
    with open(OUT / "S2_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0]))
        w.writeheader()
        w.writerows(sel_rows)
    for s in (0, 1, 2):
        mu, n_req = r6_env.request_counts(s, "trace", K, "normal")
        env = dict(pos=pos.astype(np.float32), member=member, at_home=at_home, rho=rho, n_req=n_req,
                   avail=np.ones((K, r6_env.T_ROUNDS), bool), mu=mu.astype(np.float32),
                   lost_c2e=np.zeros((K, r6_env.T_ROUNDS, 2), bool),
                   lost_dbar=np.zeros((r6_env.T_ROUNDS, L_EDGES, L_EDGES), bool),
                   lost_nbr=np.zeros((r6_env.T_ROUNDS, L_EDGES, L_EDGES), bool),
                   home_cell=home, home_pos=pos[:, 0].astype(np.float32),
                   plan_type=np.array(["trace"] * K), cell_xy=centers.astype(np.float32),
                   cell_local=np.arange(L_EDGES), cell_district=np.zeros(L_EDGES, int),
                   cell_is_hub=np.zeros(L_EDGES, bool),
                   neighbors=~np.eye(L_EDGES, dtype=bool),
                   client_district=np.zeros(K, int), client_local=np.arange(K),
                   client_group=client_group, trace_layout=np.array(True),
                   latlon=latlon.astype(np.float64),
                   round_start_min=ROUND_STARTS)
        meta = dict(scenario="S2", dataset="GeoLife Trajectories 1.3", env_seed=s, K=K, L=L_EDGES,
                    T=r6_env.T_ROUNDS, layout="trace", mu_range=r6_env.MU_RANGE["normal"], avail=1.0,
                    loss=0.0, n_points=int(n_points), n_points_kept=int(n_kept),
                    n_candidates=len(cands), n_selected=K, lat0=lat0, lon0=lon0,
                    edge_xy_km=centers.round(4).tolist(),
                    edge_latlon=[[lat0 + c[1] / ky, lon0 + c[0] / kx] for c in centers],
                    residents_per_cell=residents.tolist(), class_group_of_cell=group_of_cell,
                    rho_home=r6_env.RHO_HOME, rho_away=r6_env.RHO_AWAY,
                    partition="one nd1_partition over the 50 clients, client_to_es = class group of "
                              "the home cell, seed = run seed")
        f = OUT / f"S2_seed{s}.npz"
        if f.exists():
            old, old_meta = r6_env.load_env(f)
            same = old_meta == json.loads(json.dumps(meta)) and all(np.array_equal(old[k], env[k]) for k in env)
            print(f"{f.name}: exists, identical rebuild = {same}")
            if not same:
                raise SystemExit(f"{f} differs from a fresh build; refusing to overwrite")
            continue
        r6_env.save_env(f, env, meta)
        print(f"{f.name}: at_home={at_home.mean():.3f} two_cells={(member[..., 1] >= 0).mean():.3f} "
              f"residents={residents.tolist()} n0={(n_req == 0).mean():.4f}")
    print("selected scores", [c[0] for c in sel][:10], "...", sel[-1][0], "| edge km", centers.round(2).tolist())


if __name__ == "__main__":
    main()
