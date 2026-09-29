"""
CPU end-to-end smoke test: runs 1-2 round versions of all experiments
on synthetic data, then runs analyze + fill_paper.

Validates the entire pipeline before launching the multi-hour GPU run.
"""
import sys
import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Monkey-patch CIFAR-10 loader to use synthetic data (no download)
class FakeCIFAR(Dataset):
    def __init__(self, n=500, num_classes=10):
        torch.manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32)
        self.targets = np.array([i % num_classes for i in range(n)])
    def __len__(self): return len(self.targets)
    def __getitem__(self, i): return self.images[i], int(self.targets[i])


import data.partition as dp
dp.get_cifar10 = lambda data_root='./data_cache', augment=False: (FakeCIFAR(1000), FakeCIFAR(300))


from scripts.run_single import run_experiment, load_config


def make_tiny_cfg():
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs/base_v3.yaml'))
    cfg['device'] = 'cpu'
    cfg['num_clients'] = 6
    cfg['num_edge_servers'] = 2
    cfg['overlap_percentage'] = 30
    cfg['global_rounds'] = 3
    cfg['local_epochs'] = 1
    cfg['batch_size'] = 16
    cfg['eval_every_n_rounds'] = 3
    return cfg


def main():
    test_root = '/tmp/v3_cpu_smoke'
    shutil.rmtree(test_root, ignore_errors=True)
    os.makedirs(test_root, exist_ok=True)
    os.chdir(test_root)
    os.makedirs('results/e1_static', exist_ok=True)
    os.makedirs('results/e2_temporal/schedule_A', exist_ok=True)
    os.makedirs('docs/results_v3', exist_ok=True)

    cfg = make_tiny_cfg()

    print("E1-style mini runs (2 configs)")
    for name, method, lam, big_lam in [
        ('splitomc_lam0.2', 'splitomc', 0.2, 0.5),
        ('adaptive_splitomc', 'adaptive_splitomc', None, None),
    ]:
        print(f"  [{name}]")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            run_name=name, output_dir='results/e1_static',
            eval_every=3, eval_rho_sweep=[0.0, 0.4],
            verbose=False,
        )

    print("E2 mini run (1 config + adaptive)")
    for name, method, lam, big_lam in [
        ('splitomcplus_lam0.4_Lam0.5', 'splitomcplus', 0.4, 0.5),
        ('adaptive_splitomc', 'adaptive_splitomc', None, None),
    ]:
        print(f"  [{name}]")
        run_experiment(
            cfg=cfg, method=method, lambda_val=lam, big_lambda_val=big_lam,
            temporal_schedule='A',
            run_name=name, output_dir='results/e2_temporal/schedule_A',
            eval_every=3, verbose=False,
        )

    print("Running analyze + fill_paper")
    os.system(f"cd {test_root} && python {PROJECT_ROOT}/scripts/analyze_all.py "
              f"--results_root ./results --output_dir docs/results_v3 > /tmp/analyze.log 2>&1")
    os.system(f"cd {test_root} && python {PROJECT_ROOT}/scripts/fill_paper.py "
              f"--output docs/results_v3/paper_v3_filled.md > /tmp/fill.log 2>&1")

    must_exist = [
        'docs/results_v3/paper_v3_filled.md',
        'docs/results_v3/interpretation.md',
        'docs/results_v3/tables/e1_table.md',
        'docs/results_v3/tables/e2_table.md',
    ]
    all_ok = True
    for path in must_exist:
        full = os.path.join(test_root, path)
        if os.path.exists(full):
            size = os.path.getsize(full)
            print(f"  ✓ {path} ({size:,} bytes)")
        else:
            print(f"  ✗ MISSING: {path}")
            all_ok = False

    if all_ok:
        print("\n[SMOKE PASSED] All artifacts produced.\n")
        with open(os.path.join(test_root, 'docs/results_v3/paper_v3_filled.md')) as f:
            content = f.read()
        # Look for actual numbers being filled in (e.g., E1 table)
        import re
        nums = re.findall(r'\d+\.\d+', content)
        print(f"Paper draft length: {len(content)} chars, {len(nums)} numbers")
        # Print E2 table section
        e2_start = content.find('### 5.2 Temporal')
        e2_end = content.find('### 5.3')
        if e2_start >= 0 and e2_end >= 0:
            print("\nE2 (CORE) section preview:")
            print(content[e2_start:e2_end])
    else:
        print("\n[SMOKE FAILED]\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
