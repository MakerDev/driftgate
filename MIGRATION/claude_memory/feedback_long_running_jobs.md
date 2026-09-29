---
name: feedback-long-running-jobs
description: "How to launch multi-hour/day pipelines so they survive Claude Code's Bash subprocess cleanup"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 54509bec-9fc6-4918-afff-3e323d5c6c31
---

For long-running jobs (hours+) on this server, do NOT use `tmux new-session -d` from the Bash tool. The tmux server gets killed when the Bash tool call ends (observed: tmux session created at 21:55 was alive at 22:14 but gone by 22:42, ~30 min later, despite no OOM or crash).

**Why:** Claude Code's Bash sandbox appears to clean up child processes (possibly including detached daemons started in the same call). The tmux server inherits the sandbox's process group somehow.

**How to apply:** Launch via `setsid nohup bash -c '...' > log 2>&1 < /dev/null &` then `disown`. This pattern reparents the process to PID 1 (init/systemd), so it's fully decoupled from any shell session, the Bash tool call, or even the entire Claude session ending. Verify with `ps -p $PID -o ppid` — PPID should be 1.

**Concrete launch idiom for `/disk2/Yujin/*`-style projects:**
```bash
cd /path/to/project
setsid nohup bash -c 'PYTHONUNBUFFERED=1 bash run_all.sh' > logs/master.log 2>&1 < /dev/null &
echo $! > logs/pipeline.pid
disown
```
The actual run_all.sh PID is the child of $! (because setsid forks once); find it with `pgrep -P $! -f run_all` or look at `ps -ef | grep run_all`.

**Monitoring:** Save PID to `logs/pipeline.pid` for cross-session reference. Check with `ps -p $(cat logs/pipeline.pid)` — fresh shells can still see it because PPID=1.
