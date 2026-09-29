#!/bin/bash
# Run ON THE OLD SERVER (ubuntu20), inside a clone of github.com/MakerDev/driftgate
# (e.g. /disk2/Yujin/driftgate_github, the folder this repository was first pushed from).
#
# Copies run RECORDS that the old server produced after the first push from the live checkout
# into this clone, then commits and pushes them. Code is never copied.
#   copied : journal_expansion/runs/**  (result JSON, signal npz, per-run logs, worker logs)
#            journal_expansion/provenance/<run_id>.json
#            journal_expansion/provenance/all_runs.jsonl (merged as a union of lines)
#   skipped: queue files (runs/queue_r5/{heavy,light}.txt, enqueued_*_snapshot.txt, requeued.txt,
#            queue.lock, running_heavy*/), because the new server rebuilds its own queue.
#
# Usage:  bash MIGRATION/tools/sync_results_from_old_server.sh            # copy + commit + push
#         NO_GIT=1 bash MIGRATION/tools/sync_results_from_old_server.sh   # copy only
set -euo pipefail
LIVE="${LIVE:-/disk2/Yujin/adaptive_splitomc_tmc}"  # [SERVER-PATH:REPO_ROOT]
CLONE="$(cd "$(dirname "$0")/../.." && pwd)"
[ "$CLONE" = "$LIVE" ] && { echo "run this inside the GitHub clone, not the live checkout"; exit 1; }
cd "$CLONE"
if [ -z "${NO_GIT:-}" ]; then git pull --no-rebase --no-edit origin main; fi

rsync -a \
  --exclude 'queue.lock' --exclude 'running_heavy*/' \
  --exclude '/queue_r5/heavy.txt' --exclude '/queue_r5/light.txt' \
  --exclude '/queue_r5/enqueued_*_snapshot.txt' --exclude '/queue_r5/requeued.txt' \
  --exclude '__pycache__/' \
  "$LIVE/journal_expansion/runs/" "$CLONE/journal_expansion/runs/"
rsync -a --exclude 'all_runs.jsonl' "$LIVE/journal_expansion/provenance/" "$CLONE/journal_expansion/provenance/"

# all_runs.jsonl is appended on both servers: keep every line from both, in order, no duplicates
python3 - "$LIVE/journal_expansion/provenance/all_runs.jsonl" "$CLONE/journal_expansion/provenance/all_runs.jsonl" <<'EOF'
import sys
live, clone = sys.argv[1], sys.argv[2]
have = open(clone).read().splitlines()
seen = set(have)
new = [l for l in open(live).read().splitlines() if l.strip() and l not in seen]
if new:
    with open(clone, "a") as f:
        f.write("".join(l + "\n" for l in new))
print(f"all_runs.jsonl: +{len(new)} line(s)")
EOF

n_json=$(git status --porcelain -- journal_expansion/runs | grep -c '\.json"\?$' || true)
echo "new or changed run JSON files: $n_json"
if [ -z "${NO_GIT:-}" ]; then
  git add journal_expansion/runs journal_expansion/provenance
  if git diff --cached --quiet; then echo "nothing new to push"; exit 0; fi
  git commit -q -m "Sync run records from old server ($(hostname), $(date +%F\ %H:%M))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  git push origin HEAD:main
  git log --oneline -1
fi
