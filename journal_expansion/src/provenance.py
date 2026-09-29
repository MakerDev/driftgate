"""Run provenance recording for journal-expansion experiments.

Every run gets a unique run ID (re-running the same config creates a NEW ID,
never overwrites). Records: git commit, config + SHA-256, command, times,
seeds, dataset split hash, model init hash, software env, GPU, final metrics,
status + failure reason.
"""
import os
import sys
import json
import time
import uuid
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_DIR = JOURNAL_ROOT / "provenance"
GIT_REPO_DIR = "/disk2/Yujin"  # repo root containing this project  # [SERVER-PATH:GIT_ROOT]


def new_run_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


def sha256_of_obj(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", GIT_REPO_DIR, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def split_hash(client_indices: dict) -> str:
    canon = {str(k): sorted(int(i) for i in v) for k, v in client_indices.items()}
    return sha256_of_obj(canon)[:16]


def model_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


class RunRecord:
    """Create at run start, finalize at end. Persists to provenance/<run_id>.json
    and appends one line to provenance/all_runs.jsonl."""

    def __init__(self, prefix: str, config: dict, command: str = None, extra: dict = None):
        self.run_id = new_run_id(prefix)
        self.rec = {
            "run_id": self.run_id,
            "git_commit": git_commit(),
            "config": config,
            "config_sha256": sha256_of_obj(config),
            "command": command or " ".join(sys.argv),
            "start_time": datetime.now().isoformat(),
            "env": env_info(),
            "status": "running",
        }
        if extra:
            self.rec.update(extra)
        self._t0 = time.time()
        PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
        self._flush()

    def set(self, **kwargs):
        self.rec.update(kwargs)
        self._flush()

    def finish(self, metrics: dict = None, status: str = "completed", failure_reason: str = None):
        self.rec["end_time"] = datetime.now().isoformat()
        self.rec["wall_time_sec"] = time.time() - self._t0
        self.rec["status"] = status
        if failure_reason:
            self.rec["failure_reason"] = failure_reason
        if metrics:
            self.rec["final_metrics"] = metrics
        self._flush()
        with open(PROVENANCE_DIR / "all_runs.jsonl", "a") as f:
            f.write(json.dumps(self.rec, default=str) + "\n")

    def _flush(self):
        with open(PROVENANCE_DIR / f"{self.run_id}.json", "w") as f:
            json.dump(self.rec, f, indent=1, default=str)
