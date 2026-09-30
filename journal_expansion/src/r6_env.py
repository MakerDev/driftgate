"""Round 6 environment module (R6 directive §3): positions, commute mobility,
cluster membership, per-client rho, Poisson request counts, participation and
signal-loss masks.

The training code only READS the files written here
(runs/phaseT6_env/<scenario>_seed<s>.npz); every arm of the same seed sees the
same file. Four independent random streams per (layout, env_seed) (§3.8):

  stream 1  home positions + daily plans
  stream 2  per-trip speed quantiles u in [0,1)   -> speed = lo + (hi-lo)*u
  stream 3  request-rate quantiles + Poisson counts
  stream 4  participation + signal loss (uniforms; lost = u < p, avail = u < a)

Using quantiles/uniforms makes the conditions common-random-number coupled:
S1 and S1-fast share the plans AND the speed quantile of every trip; S4
conditions share S1's paths; a=0.5 participants are a subset of a=0.7
participants; p=0.1 losses are a subset of p=0.3 losses.

Time: round r (1..T) starts at 05:00 + 6*(r-1) min; the environment state of
round r is its value at that instant (§3.1). Distances are km, times minutes.
"""
import json
import math
import numpy as np

T_ROUNDS = 150
ROUND_MIN = 6.0
DAY_START_MIN = 5 * 60.0
R_KM = 1.0              # edge coverage radius (§3.2)
POS_SIGMA_KM = 0.35     # "random position inside a cell" = centre + N(0, 0.35^2 I)
RHO_HOME, RHO_AWAY = 0.1, 0.8
MAX_CELLS_PER_CLIENT = 2
NEIGHBOR_DIST_KM = 2.0 * R_KM
DISTRICT_PITCH_KM = 4.5

# base district: cell local id -> (name, offset from hub); E=0, N=1, W=2, S=3, H=4 (§3.3)
CELL_NAMES = ["E", "N", "W", "S", "H"]
CELL_OFFSETS = np.array([[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5], [0.0, 0.0]])
HUB_LOCAL = 4
RES_LOCAL = [0, 1, 2, 3]

SPEED_RANGE = {"walk": (1.0, 3.0), "vehicle": (8.0, 15.0)}      # m/s (§3.5)
MU_RANGE = {"normal": (32.0, 128.0), "low": (8.0, 24.0)}         # requests / round (§3.6)

# scenario -> layout + condition. S4 delay uses the S1 file (delay is a run flag).
SCENARIOS = {
    "S1":         dict(layout="base", grid=(1, 1), speed="walk", mu="normal", avail=1.0, loss=0.0),
    "S1fast":     dict(layout="base", grid=(1, 1), speed="vehicle", mu="normal", avail=1.0, loss=0.0),
    "S4_lowreq":  dict(layout="base", grid=(1, 1), speed="walk", mu="low", avail=1.0, loss=0.0),
    "S4_part07":  dict(layout="base", grid=(1, 1), speed="walk", mu="normal", avail=0.7, loss=0.0),
    "S4_part05":  dict(layout="base", grid=(1, 1), speed="walk", mu="normal", avail=0.5, loss=0.0),
    "S4_loss01":  dict(layout="base", grid=(1, 1), speed="walk", mu="normal", avail=1.0, loss=0.1),
    "S4_loss03":  dict(layout="base", grid=(1, 1), speed="walk", mu="normal", avail=1.0, loss=0.3),
    "S3_K200":    dict(layout="grid2x2", grid=(2, 2), speed="walk", mu="normal", avail=1.0, loss=0.0),
    "S3_K500":    dict(layout="grid2x5", grid=(2, 5), speed="walk", mu="normal", avail=1.0, loss=0.0),
}
LAYOUT_ID = {"base": 0, "grid2x2": 1, "grid2x5": 2, "trace": 3}


def rng_for(env_seed, stream, layout):
    return np.random.default_rng(np.random.SeedSequence([int(env_seed), int(stream), LAYOUT_ID[layout]]))


def round_start_min(r):
    """Clock minute (since 00:00) at which round r (1-indexed) starts."""
    return DAY_START_MIN + ROUND_MIN * (r - 1)


def hhmm(m):
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


# ----------------------------------------------------------------------------- map

def district_layout(grid):
    """Returns cell_xy [L,2], cell_local [L], cell_district [L], district_xy [D,2].
    District d = row*ncols + col; its hub sits at (col*4.5, row*4.5) km."""
    nrows, ncols = grid
    dxy, cxy, cloc, cdist = [], [], [], []
    for row in range(nrows):
        for col in range(ncols):
            d = row * ncols + col
            hub = np.array([col * DISTRICT_PITCH_KM, row * DISTRICT_PITCH_KM])
            dxy.append(hub)
            for loc in range(5):
                cxy.append(hub + CELL_OFFSETS[loc])
                cloc.append(loc)
                cdist.append(d)
    return np.array(cxy), np.array(cloc), np.array(cdist), np.array(dxy)


def neighbor_matrix(cell_xy, max_dist=NEIGHBOR_DIST_KM):
    d = np.linalg.norm(cell_xy[:, None, :] - cell_xy[None, :, :], axis=2)
    nb = (d <= max_dist + 1e-9)
    np.fill_diagonal(nb, False)
    return nb


def membership(pos, cell_xy, R=R_KM, kmax=MAX_CELLS_PER_CLIENT):
    """Cells within R (nearest first, at most kmax); if none, the nearest cell.
    pos [..., 2] -> member [..., kmax] (-1 padded)."""
    d = np.linalg.norm(pos[..., None, :] - cell_xy, axis=-1)          # [..., L]
    order = np.argsort(d, axis=-1, kind="stable")
    dsorted = np.take_along_axis(d, order, axis=-1)
    out = np.full(pos.shape[:-1] + (kmax,), -1, dtype=np.int16)
    for j in range(kmax):
        inside = dsorted[..., j] <= R
        out[..., j] = np.where(inside, order[..., j], -1)
    none = out[..., 0] < 0
    out[..., 0] = np.where(none, order[..., 0], out[..., 0])
    return out


# ------------------------------------------------------------------- daily plans

def _truncnorm(rng, mean, sd, lo, hi):
    while True:
        v = rng.normal(mean, sd)
        if lo <= v <= hi:
            return v


def _pos_in_cell(rng, cell_xy, c):
    return cell_xy[c] + rng.normal(0.0, POS_SIGMA_KM, size=2)


def draw_plans(env_seed, layout, grid, homes_local):
    """Stream 1. homes_local[k] = (district, local cell) of client k's home.
    Returns list of per-client dicts with home position and planned trips."""
    rng = rng_for(env_seed, 1, layout)
    cell_xy, cell_local, cell_district, dxy = district_layout(grid)
    D = len(dxy)
    gid = lambda d, loc: 5 * d + loc
    s3 = D > 1
    plans = []
    for k, (d, loc) in enumerate(homes_local):
        home_cell = gid(d, loc)
        home_pos = _pos_in_cell(rng, cell_xy, home_cell)
        u = rng.random()
        work_cell = None
        if loc == HUB_LOCAL:
            if u >= 0.8:
                work_cell = gid(d, RES_LOCAL[rng.integers(4)])
        else:
            others = [x for x in RES_LOCAL if x != loc]
            if not s3:
                if u < 0.6:
                    work_cell = gid(d, HUB_LOCAL)
                elif u < 0.8:
                    work_cell = gid(d, others[rng.integers(3)])
            else:
                if u < 0.45:
                    work_cell = gid(d, HUB_LOCAL)
                elif u < 0.60:
                    dd = np.linalg.norm(dxy - dxy[d], axis=1)
                    dd[d] = np.inf
                    near = np.flatnonzero(np.isclose(dd, dd.min()))
                    work_cell = gid(int(near[rng.integers(len(near))]), HUB_LOCAL)
                elif u < 0.80:
                    work_cell = gid(d, others[rng.integers(3)])
        p = dict(client=k, home_cell=home_cell, home_pos=home_pos, trips=[])
        if work_cell is not None:
            p["type"] = "work"
            p["work_cell"] = work_cell
            work_pos = _pos_in_cell(rng, cell_xy, work_cell)
            p["work_pos"] = work_pos
            t_out = _truncnorm(rng, 7 * 60 + 45, 30, 6 * 60 + 30, 9 * 60 + 30)
            t_back = _truncnorm(rng, 17 * 60 + 45, 45, 16 * 60, 19 * 60 + 30)
            p["trips"].append(dict(kind="commute_out", depart=t_out, dest=work_pos, dest_cell=work_cell))
            if rng.random() < 0.3:
                t_l = rng.normal(12 * 60 + 15, 20)
                wd = cell_district[work_cell]
                cand = [gid(wd, x) for x in range(5) if gid(wd, x) != work_cell]
                lc = cand[rng.integers(len(cand))]
                lpos = _pos_in_cell(rng, cell_xy, lc)
                stay = max(15.0, rng.exponential(45.0))
                p["trips"].append(dict(kind="lunch_out", depart=t_l, dest=lpos, dest_cell=lc, stay=stay))
                p["trips"].append(dict(kind="lunch_back", depart=None, dest=work_pos, dest_cell=work_cell))
            p["trips"].append(dict(kind="commute_back", depart=t_back, dest=home_pos, dest_cell=home_cell))
        else:
            p["type"] = "stay"
            if rng.random() < 0.3:
                t_e = rng.uniform(10 * 60, 16 * 60)
                cand = [gid(d, x) for x in range(5) if gid(d, x) != home_cell]
                ec = cand[rng.integers(len(cand))]
                epos = _pos_in_cell(rng, cell_xy, ec)
                stay = max(20.0, rng.exponential(60.0))
                p["trips"].append(dict(kind="errand_out", depart=t_e, dest=epos, dest_cell=ec, stay=stay))
                p["trips"].append(dict(kind="errand_back", depart=None, dest=home_pos, dest_cell=home_cell))
        plans.append(p)
    return plans


def simulate_positions(plans, speed_q, speed_range, T=T_ROUNDS):
    """Piecewise-linear motion (§3.5). speed_q[k, j] = quantile of trip j of client k.
    A return trip (lunch_back / errand_back) departs `stay` minutes after the
    outbound trip ARRIVES; it is dropped if the outbound trip was interrupted or if
    it would depart at/after the next scheduled trip. Returns pos [K,T,2] and
    per-trip execution records (actual depart time, speed, status)."""
    lo, hi = speed_range
    K = len(plans)
    t_round = np.array([round_start_min(r) for r in range(1, T + 1)])
    pos = np.zeros((K, T, 2))
    records = []
    for k, p in enumerate(plans):
        # build the executed segment list: (t0, p0, t1, p1)
        segs = []
        cur_t, cur_p = -1e9, p["home_pos"].copy()
        trips = p["trips"]
        sched = [dict(tr) for tr in trips]
        for j, tr in enumerate(sched):
            tr["speed_mps"] = lo + (hi - lo) * float(speed_q[k, j])
        i = 0
        while i < len(sched):
            tr = sched[i]
            if tr["depart"] is None:            # return trip whose outbound never arrived
                tr["status"] = "dropped"
                i += 1
                continue
            nxt = next((s["depart"] for s in sched[i + 1:] if s["depart"] is not None), None)
            t0 = tr["depart"]
            # position at t0 given current motion segment
            p0 = _pos_at(segs, t0, p["home_pos"])
            dist = float(np.linalg.norm(tr["dest"] - p0))
            v_km_min = tr["speed_mps"] * 60.0 / 1000.0
            t_arr = t0 + dist / v_km_min
            if nxt is not None and nxt < t_arr:
                # interrupted: cut segment at nxt
                frac = (nxt - t0) / (t_arr - t0) if t_arr > t0 else 1.0
                segs.append((t0, p0, nxt, p0 + frac * (tr["dest"] - p0)))
                tr["status"] = "interrupted"
                tr["arrive"] = None
            else:
                segs.append((t0, p0, t_arr, tr["dest"].copy()))
                tr["status"] = "done"
                tr["arrive"] = t_arr
            # schedule the matching return trip
            if tr["kind"] in ("lunch_out", "errand_out") and i + 1 < len(sched):
                back = sched[i + 1]
                if tr["status"] == "done":
                    t_back = tr["arrive"] + tr["stay"]
                    nxt2 = next((s["depart"] for s in sched[i + 2:] if s["depart"] is not None), None)
                    if nxt2 is None or t_back < nxt2:
                        back["depart"] = t_back
            i += 1
        for r_i, t in enumerate(t_round):
            pos[k, r_i] = _pos_at(segs, t, p["home_pos"])
        for j, tr in enumerate(sched):
            records.append(dict(client=k, trip=j, kind=tr["kind"], depart=tr["depart"],
                                dest_cell=tr["dest_cell"], dest_x=float(tr["dest"][0]),
                                dest_y=float(tr["dest"][1]), speed_mps=tr["speed_mps"],
                                status=tr.get("status", "dropped"),
                                arrive=tr.get("arrive")))
    return pos, records


def _pos_at(segs, t, start):
    """Position at time t along executed segments (hold last point between segments)."""
    p = start.copy()
    for (t0, p0, t1, p1) in segs:
        if t < t0:
            break
        if t >= t1:
            p = p1.copy()
        else:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            p = p0 + frac * (p1 - p0)
            break
    return p


# ----------------------------------------------------------------- scenario build

def _synthetic_homes(grid):
    """Per district: the static topology's cluster g basic members (local ids
    10g..10g+9) live in cell g (E=0,N=1,W=2,S=3,H=4) (§3.3)."""
    D = grid[0] * grid[1]
    homes = []
    for d in range(D):
        for local in range(50):
            homes.append((d, local // 10))
    return homes


def request_counts(env_seed, layout, K, mu_kind, T=T_ROUNDS):
    rng = rng_for(env_seed, 3, layout)
    u = rng.random(K)
    lo, hi = MU_RANGE[mu_kind]
    mu = lo + (hi - lo) * u
    n = rng.poisson(mu[:, None], size=(K, T)).astype(np.int32)
    return mu, n


def participation_and_loss(env_seed, layout, K, L, a, p, T=T_ROUNDS):
    rng = rng_for(env_seed, 4, layout)
    u_av = rng.random((K, T))
    u_c2e = rng.random((K, T, MAX_CELLS_PER_CLIENT))
    u_dbar = rng.random((T, L, L))
    u_nbr = rng.random((T, L, L))
    avail = u_av < a
    return avail, (u_c2e < p), (u_dbar < p), (u_nbr < p)


def build_synthetic(scenario, env_seed):
    sc = SCENARIOS[scenario]
    grid, layout = sc["grid"], sc["layout"]
    cell_xy, cell_local, cell_district, dxy = district_layout(grid)
    L = len(cell_xy)
    homes = _synthetic_homes(grid)
    K = len(homes)
    plans = draw_plans(env_seed, layout, grid, homes)
    speed_q = rng_for(env_seed, 2, layout).random((K, 4))
    pos, trip_rec = simulate_positions(plans, speed_q, SPEED_RANGE[sc["speed"]])
    member = membership(pos, cell_xy)
    home_cell = np.array([p["home_cell"] for p in plans])
    at_home = (member == home_cell[:, None, None]).any(axis=2)
    rho = np.where(at_home, RHO_HOME, RHO_AWAY).astype(np.float32)
    mu, n_req = request_counts(env_seed, layout, K, sc["mu"])
    avail, lost_c2e, lost_dbar, lost_nbr = participation_and_loss(
        env_seed, layout, K, L, sc["avail"], sc["loss"])
    client_district = np.array([h[0] for h in homes])
    client_local = np.array([i % 50 for i in range(K)])
    env = dict(
        pos=pos.astype(np.float32), member=member, at_home=at_home, rho=rho,
        n_req=n_req, avail=avail, mu=mu.astype(np.float32),
        lost_c2e=lost_c2e, lost_dbar=lost_dbar, lost_nbr=lost_nbr,
        home_cell=home_cell, home_pos=np.array([p["home_pos"] for p in plans], dtype=np.float32),
        plan_type=np.array([p["type"] for p in plans]),
        cell_xy=cell_xy.astype(np.float32), cell_local=cell_local, cell_district=cell_district,
        cell_is_hub=(cell_local == HUB_LOCAL), neighbors=neighbor_matrix(cell_xy),
        client_district=client_district, client_local=client_local,
        client_group=np.array([h[1] for h in homes]),
        round_start_min=np.array([round_start_min(r) for r in range(1, T_ROUNDS + 1)]),
    )
    meta = dict(scenario=scenario, env_seed=int(env_seed), K=int(K), L=int(L), T=T_ROUNDS,
                grid=list(grid), layout=layout, speed_mps=SPEED_RANGE[sc["speed"]],
                mu_range=MU_RANGE[sc["mu"]], avail=sc["avail"], loss=sc["loss"],
                R_km=R_KM, pos_sigma_km=POS_SIGMA_KM, rho_home=RHO_HOME, rho_away=RHO_AWAY,
                partition="district d: nd1_partition on the static 50-client topology, "
                          "seed = env_seed if d == 0 else env_seed*1000 + d; "
                          "cluster g basic members live in cell g")
    return env, meta, plans, trip_rec


def save_env(path_npz, env, meta):
    np.savez_compressed(path_npz, meta=np.array(json.dumps(meta)), **env)


def load_env(path_npz):
    z = np.load(path_npz, allow_pickle=False)
    env = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    return env, meta


def write_plan_csv(path_csv, plans, trip_rec):
    import csv
    by_client = {}
    for rec in trip_rec:
        by_client.setdefault(rec["client"], []).append(rec)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "home_cell", "plan_type", "work_cell", "trip", "kind",
                    "depart_min", "depart_hhmm", "dest_cell", "dest_x_km", "dest_y_km",
                    "speed_mps", "status", "arrive_hhmm"])
        for p in plans:
            recs = by_client.get(p["client"], [])
            if not recs:
                w.writerow([p["client"], p["home_cell"], p["type"], p.get("work_cell", ""),
                            "", "", "", "", "", "", "", "", "", ""])
            for rec in recs:
                dep = rec["depart"]
                w.writerow([p["client"], p["home_cell"], p["type"], p.get("work_cell", ""),
                            rec["trip"], rec["kind"],
                            "" if dep is None else f"{dep:.2f}", "" if dep is None else hhmm(dep),
                            rec["dest_cell"], f"{rec['dest_x']:.4f}", f"{rec['dest_y']:.4f}",
                            f"{rec['speed_mps']:.3f}", rec["status"],
                            "" if rec.get("arrive") is None else hhmm(rec["arrive"])])
