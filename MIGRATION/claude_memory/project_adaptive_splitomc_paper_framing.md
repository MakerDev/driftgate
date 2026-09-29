---
name: project-adaptive-splitomc-paper-framing
description: "Locked paper framing decisions for Adaptive-SplitOMC v3 — E2 is a two-signal ablation contribution (not a limitation), E3 spatial is potential main result, all phase reports must follow these"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54509bec-9fc6-4918-afff-3e323d5c6c31
---

For Adaptive-SplitOMC v3 / ICTC 2026 target / Yujin Shin paper, the framing is **locked** as of 2026-05-30 02:00 KST after E2 retry-1 (-1.94pp) and retry incoming-traffic anchor (transient signal, training convergence erases) both failed the +2pp CORE gate.

**Why:** User explicitly committed to "Plan B (honest reporting)" but specified that E2 must NOT be buried as a limitation. The two failed signal variants together form a contribution: a quantitative ablation showing entropy-as-drift-signal has structural limits in federated split learning.

**How to apply:**
- E2 section in paper draft frames train-anchor (attempt 1) vs incoming-traffic (final fix) as a controlled two-signal ablation. Both fail differently; together they motivate convergence-invariant drift sensing (frozen-ref KL, OOR-only anchor, external label-distribution shift) as future work.
- E2 ρ=0 segment WIN (adaptive 0.706 > best fixed 0.702) must be explicitly preserved — proves controller logic is sound; failure is signal-source.
- E3 spatial is the potential main result. Each cell has fixed ρ_z (steady-state), so the convergence-vs-drift confound from E2 doesn't apply. Hypothesis: cells with low ρ_z converge to high λ_z, cells with high ρ_z converge to low λ_z. When adaptive runs on E3, check (a) per-cell λ_z stratification, (b) monotone λ_z vs ρ_z fit, (c) per-cell accuracy vs per-cell oracle. If yes → paper headline becomes per-cell adaptive control. If no → headline falls back to E1 acc_oor +37%.
- All `docs/results_v3/phase_reports/*.md` and the final `interpretation.md` + `paper_v3_filled.md` follow this. Detailed framing at `/disk2/Yujin/adaptive_splitomc_v3/docs/results_v3/phase_reports/_paper_framing_decisions.md`.

See also: [[feedback-long-running-jobs]] (the long pipeline this lives in).
