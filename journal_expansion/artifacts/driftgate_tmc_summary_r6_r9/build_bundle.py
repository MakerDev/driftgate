"""Build DriftGate_R6_R9_bundle.zip from files already in the repository (no computation)."""
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parent.parent          # journal_expansion/artifacts
HERE = Path(__file__).resolve().parent
R6, R7, R8, R9 = (A / "driftgate_tmc_r6", A / "driftgate_tmc_r7_gate", A / "driftgate_tmc_r8_check",
                  A / "driftgate_tmc_r9_routing")
FILES = {
    "DriftGate_R6_R9_summary_ko.md": HERE / "DriftGate_R6_R9_summary_ko.md",
    "README_ko.md": HERE / "README_ko.md",
    "reports/DriftGate_R6_report_ko.md": R6 / "DriftGate_R6_report_ko.md",
    "reports/DriftGate_R7_gate_report_ko.md": R7 / "DriftGate_R7_gate_report_ko.md",
    "reports/DriftGate_R8_check_report_ko.md": R8 / "DriftGate_R8_check_report_ko.md",
    "reports/DriftGate_R9_routing_report_ko.md": R9 / "DriftGate_R9_routing_report_ko.md",
    "decisions/R7_decision.json": R7 / "decision.json",
    "decisions/R8_decision_v2.json": R8 / "decision_v2.json",
    "decisions/R9_decision.json": R9 / "decision.json",
}
for f in ("fig1_scenario_commute", "fig2_lambda_trajectories", "fig3_accuracy_by_time", "fig4_scale_robustness", "fig5_latency"):
    FILES[f"figures/R6_{f}.png"] = R6 / "figures" / f"{f}.png"
FILES["figures/R6_figure_captions.md"] = R6 / "figures" / "figure_captions.md"
for f in ("figA_lambda_home_away", "figB_accuracy_by_time"):
    FILES[f"figures/R7_{f}.png"] = R7 / "figures" / f"{f}.png"
FILES["figures/R7_figure_captions.md"] = R7 / "figures" / "figure_captions.md"
for f in ("v2_figA_accuracy_vs_server_use", "v2_figB_probes"):
    FILES[f"figures/R8_{f}.png"] = R8 / "figures" / f"{f}.png"
FILES["figures/R8_figure_captions.md"] = R8 / "figures" / "v2_figure_captions.md"
for f in ("R9_figA_placementA", "R9_figB_placementB"):
    FILES[f"figures/{f}.png"] = R9 / "figures" / f"{f}.png"
FILES["figures/R9_figure_captions.md"] = R9 / "figures" / "R9_figure_captions.md"
for f in ("T2_main_results", "T3b_extra_metrics_paired", "T4b_time_of_day_contribution", "T5c_lambda_by_time_and_cell",
          "T6_speed", "T7_scale", "T8_robustness", "T9_latency", "T11b_best_fixed_by_cell_and_time"):
    FILES[f"tables/R6_{f}.csv"] = R6 / "tables" / f"{f}.csv"
FILES["tables/R6_paper_numbers_r6.csv"] = R6 / "paper_numbers_r6.csv"
FILES["tables/R0_neighbor_avg_record.csv"] = R6 / "r0" / "tables" / "R0_neighbor_avg_record.csv"
FILES["tables/R0_paper_numbers.csv"] = R6 / "r0" / "paper_numbers.csv"
for f in ("T1_runs_and_means", "T2_paired_differences", "T3_lambda_home_away", "T4_signal_auroc_away", "T7_slot_decomposition"):
    FILES[f"tables/R7_{f}.csv"] = R7 / "tables" / f"{f}.csv"
for f in ("v2_T2_tau08", "v2_T4_points_mean", "v2_T5_G", "v2_T6_splits_at_0.9", "v2_T7_probe_auroc", "v2_T8_probe_G",
          "v2_S1_exit_accuracy_mean", "v2_S2_server_use_at_reference", "v2_S3_controls", "v2_S4_G_by_time_slot"):
    FILES[f"tables/R8_{f}.csv"] = R8 / "tables" / f"{f}.csv"
for f in ("R9_T2_placementA", "R9_T3_placementB", "R9_T4_splits", "R9_T6_B_kind_oracle_points", "R9_S1_detection"):
    FILES[f"tables/{f}.csv"] = R9 / "tables" / f"{f}.csv"

out = HERE / "DriftGate_R6_R9_bundle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, src in FILES.items():
        assert src.exists(), src
        z.write(src, arc)
print(f"{out} : {len(FILES)} files, {out.stat().st_size / 1e6:.2f} MB")
