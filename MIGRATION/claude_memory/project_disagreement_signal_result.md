---
name: project_disagreement_signal_result
description: Outcome of replacing entropy drift signal with exit-disagreement in Adaptive-SplitOMC (v4)
metadata: 
  node_type: memory
  type: project
  originSessionId: 48dbf36e-bbd5-4de4-b9d1-08ea2a8eb3c4
---

Adaptive-SplitOMC v4 (`/disk2/Yujin/adaptive_splitomc_v4`): replaced the entropy drift signal (which collapsed at training convergence) with a **client-vs-server exit-disagreement signal** (δ = fraction of probe samples where client_pred ≠ server_pred). Implemented as signal source swap only; entropy path preserved for ablation.

**Result (updated 2026-05-31, mu=0.31 final + E4/E5 disagreement re-run):**
- Gate G1/G2/G3 all passed; δ at ρ=0.8 R85 = 1.47× R25 (entropy died here).
- E3 spatial: per-cell λ stratification HELD to R150 (spread 0.227 @mu0.31) vs entropy collapse (0.009). Clean paper figure.
- E2 temporal: PARTIAL, integrated +1.33pp vs best fixed (target +2), +2.44pp vs entropy.
- E3: adaptive worst_cell 0.632 < best fixed 0.681 (R150) — criterion NOT met at full horizon (but at R100 acc BEATS fixed +1.97pp, worst tie).
- **★ E4 mobility (CORRECTED — earlier draft wrongly said −3.24pp): at R150 adaptive disagreement BEATS best fixed (λ=0.2 mob, 0.6737) on BOTH acc (+0.76pp, 0.6813) AND worst-cell (+0.30pp, 0.6505). Only experiment winning both. +3.68pp vs entropy. Caveat: R100 it's −0.49pp below fixed (full-horizon win only); λ=0.4 mob baseline re-running for grid robustness.**
- **E5 3-way SIGNAL ablation (NEW, 150R, same controller): disagreement +2.43pp(R150)/+2.32pp(R100) over entropy — horizon-invariant, the cleanest C1 isolation. margin(β=0.5): −0.23pp(R150)/−0.28pp(R100) vs S1-only → margin HURTS, S1 alone best, β=0 confirmed.**
- Root cause of E2/E3 residual gap: sustained high-ρ δ ≈ calibration center mu_drift, so worst-cell λ relaxes instead of approaching lam_min. Calibration/convergence-fade, not signal death.

**Calibration locked:** mu_drift=0.31, tau_drift=0.045 (THREE measured re-centerings: 0.392→0.35→0.31; user authorized the last as data-driven not blind tuning, then locked it — no 4th).

**R100 vs R150 horizon finding (key):** disagreement signal δ fades with convergence, so adaptive peaks mid-training. E2 and E3 move OPPOSITELY with horizon: E2 adaptive advantage grows (R100 +0.28pp → R150 +1.33pp); E3 adaptive advantage shrinks (R100 acc +1.97pp WIN, worst_cell tie → R150 acc -0.32pp, worst_cell -4.93pp). At R100 E3 spatial is a near-headline (beats fixed on acc, ties worst_cell); at R150 fixed overtakes. Integrated reports: `docs/results_v3/phase_reports/Integrated_R100.md` and `Integrated_R150.md`. Only E2/E3 use disagreement; E1/E4/E5 still entropy.

**Paper implication:** lead with C1 (disagreement signal, isolated by E5 3-way +2.43pp) + C2 (entropy failure); **feature E4 mobility as the accuracy headline** (adaptive beats best fixed on both acc & worst-cell at R150, the most dynamic setting). Report E2/E3 honestly (PARTIAL / horizon-dependent). See `docs/results_v3/phase_reports/STEP6_e2rerun_e3_results.md` and `phase2b_disagreement_signal.md`. Relates to [[project_adaptive_splitomc_paper_framing]].
