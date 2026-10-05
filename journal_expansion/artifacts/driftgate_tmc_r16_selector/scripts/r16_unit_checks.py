"""Round 16 stage 1: unit checks of the new helpers in r16_stage1.py against direct computations (synthetic data)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("r16", Path(__file__).resolve().parent / "r16_stage1.py")
r16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r16)
rng = np.random.default_rng(0)

# 1. correctness intervals of argmax(w p_d + (1 - w) p_e) against the float32 mix at many w
N, C = 20000, 10
pd_ = rng.dirichlet(np.full(C, 0.3), N).astype(np.float16).astype(np.float32)
pe_ = rng.dirichlet(np.full(C, 0.3), N).astype(np.float16).astype(np.float32)
y = rng.integers(0, C, N)
lo, hi = r16.intervals(pd_, pe_, y)
ws = np.round(rng.uniform(0, 1, 50), 6)
agree = np.mean([(((lo < w) & (w < hi)) == (r16.mix_scalar(w, pd_, pe_) == y)).mean() for w in ws])
print("1 interval vs mix agreement", agree)
assert agree > 0.9999
# count_in against a direct count
for _ in range(20):
    idx = rng.choice(N, 300, replace=False)
    w = rng.uniform(0, 1, 40)
    direct = ((lo[idx][None, :] < w[:, None]) & (w[:, None] < hi[idx][None, :])).sum(1)
    assert np.array_equal(r16.count_in(np.sort(lo[idx]), np.sort(hi[idx]), w), direct)
print("1b count_in ok")

# 2. mix_scalar equals r15 mix_answer
for w in r16.W_GRID:
    assert np.array_equal(r16.mix_scalar(w, pd_, pe_), r16.m15.mix_answer(np.full(N, w), pd_, pe_)), w
print("2 mix_scalar == mix_answer")

# 3. period window against a direct loop (prev_ok random)
E, K = 6, 7
e = rng.integers(0, E, 3000)
k = rng.integers(0, K, 3000)
D = dict(K=K, e=e, k=k, g=e * K + k, arrival=rng.permutation(3000))
ix = r16.m14.window_index(D)
v = rng.integers(0, 2, 3000).astype(float)
prev_ok = rng.random(len(ix["starts"])) < 0.5
s_, n_ = r16.period_window(ix, v, prev_ok)
gid = D["g"][ix["order"][ix["starts"]]]
for gi in range(len(gid)):
    rows = ix["order"][ix["starts"][gi]:ix["starts"][gi] + ix["sizes"][gi]]
    pg = ix["prev"][gi]
    prow = ix["order"][ix["starts"][pg]:ix["starts"][pg] + ix["sizes"][pg]] if pg >= 0 else None
    for j, r in enumerate(rows):
        if j >= 8:
            ref = (v[rows[:j]].sum(), j)
        elif pg >= 0 and prev_ok[gi]:
            ref = (v[prow].sum(), len(prow))
        else:
            ref = (v[rows[:j]].sum(), j)
        assert (s_[r], n_[r]) == ref
# with prev_ok all True it equals the Round 14 window wherever the Round 14 window exists
s14, n14 = r16.m14.window_sum(ix, v)
s2, n2 = r16.period_window(ix, v, np.ones(len(gid), bool))
m = n14 > 0
assert np.array_equal(s14[m], s2[m]) and np.array_equal(n14[m], n2[m])
print("3 period window ok")

# 4. nearest r with ties
sh = np.array([0.75, 0.65, 0.5, 0.9, 0.1, 0.55, 0.45, 0.35, 0.25, 0.123])
print("4 r(s):", dict(zip(sh.tolist(), r16.R_NUM[r16.r_index(sh)].tolist())))
assert r16.R_NUM[r16.r_index(np.array([0.75]))][0] == 0.3 and r16.R_NUM[r16.r_index(np.array([0.25]))][0] == 0.7
assert r16.R_NUM[r16.r_index(np.array([0.45]))][0] == 0.5 and r16.R_NUM[r16.r_index(np.array([0.55]))][0] == 0.5

# 5. pick: priority order on ties
J = np.array([[0.5, 0.5, 0.4], [0.3, 0.5, 0.5 + 1e-13], [0.1, 0.2, 0.3]])
assert list(r16.pick(J)) == [0, 1, 2]
print("5 pick ok")

# 6. grid order and column maps
assert r16.GRID[0] == (0.0, 0.5) and r16.GRID[9] == (0.0, None) and r16.GRID[-1] == (1.0, None)
for ri in range(9):
    assert r16.GRID[r16.COL_B0[ri] - 1] == (0.0, r16.R_NUM[ri]) and r16.GRID[r16.COL_B5[ri] - 1] == (0.5, r16.R_NUM[ri])
    assert [r16.GRID[c - 1] for c in r16.COL_BG[ri][1:]] == [(w, r16.R_NUM[ri]) for w in r16.W_GRID] and r16.COL_BG[ri][0] == 0
print("6 grid columns ok")

# 7. image keys: same (label, rank) within contiguous device-round blocks
Dk = dict(g=np.repeat([0, 1, 2], [5, 4, 3]), y=np.array([3, 3, 1, 3, 1, 1, 1, 3, 2, 3, 3, 2]))
print("7 keys", (r16.image_keys(Dk) % (1 << 24)).tolist())
assert (r16.image_keys(Dk) % (1 << 24)).tolist() == [0, 1, 0, 2, 1, 0, 1, 0, 0, 0, 1, 0]
print("all unit checks passed")
