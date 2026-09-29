"""Task 1 — disjoint controller (probe) vs evaluation sample pools.

Per class, split the test-set indices into a controller pool (default 20%) and an
evaluation pool (80%), stratified, disjoint, deterministic from the run seed. We
then hand the existing `build_per_client_test_sets` a MASKED label array for each
pool (indices outside the pool set to -1), so `np.where(labels==c)` returns only
that pool's indices — no change to the parent eval/partition code, and the two
pools share no sample ID.
"""
import numpy as np


def make_pool_masks(test_labels, num_classes, controller_frac=0.2, seed=0):
    """Returns (controller_labels, eval_labels, manifest).
    controller_labels / eval_labels are int arrays == test_labels on their own
    pool indices and -1 elsewhere. manifest records the split per class."""
    test_labels = np.asarray(test_labels)
    rng = np.random.default_rng(seed * 2654435761 % (2**32))
    controller_labels = np.full_like(test_labels, -1)
    eval_labels = np.full_like(test_labels, -1)
    manifest = {"controller_frac": controller_frac, "seed": seed, "per_class": {}}
    ctrl_ids, eval_ids = set(), set()
    for c in range(num_classes):
        idx = np.where(test_labels == c)[0]
        if len(idx) == 0:
            continue
        perm = rng.permutation(idx)
        n_ctrl = max(1, int(round(len(perm) * controller_frac)))
        c_ids, e_ids = perm[:n_ctrl], perm[n_ctrl:]
        controller_labels[c_ids] = c
        eval_labels[e_ids] = c
        ctrl_ids.update(int(i) for i in c_ids)
        eval_ids.update(int(i) for i in e_ids)
        manifest["per_class"][int(c)] = {"n_total": int(len(idx)),
                                         "n_controller": int(len(c_ids)),
                                         "n_eval": int(len(e_ids))}
    assert ctrl_ids.isdisjoint(eval_ids), "controller and eval pools overlap!"
    manifest["controller_pool_size"] = len(ctrl_ids)
    manifest["eval_pool_size"] = len(eval_ids)
    manifest["global_overlap"] = len(ctrl_ids & eval_ids)
    return controller_labels, eval_labels, manifest
