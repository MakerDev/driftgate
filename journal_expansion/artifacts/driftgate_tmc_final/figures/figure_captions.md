# Figure captions (English)

Numbers in the captions come from `tables/T6a_signal_levels.csv`, `tables/T6b_shift_response_late_abrupt.csv`,
`tables/T2a_segment_accuracy.csv`, `tables/T2c_segment_best_worst_fixed.csv`, `tables/lambda_trajectories.csv`
and `tables/T7_role_experiment.csv`.

**Figure 1** (`fig1_signal_dynamics.pdf`, double column).
Signal values on one model trained with a fixed weight (λ = 0.4, Λ = 0.5, 3 seeds, bands show the seed SD).
The top panels show the share ρ of non-Main traffic. The bottom panels show the client-server TV (blue) and
the predictive entropy divided by ln C (orange). (a) A 100-round version of the stepwise composition change,
with ρ changing at rounds 21, 41, 61 and 81. TV rises from 0.244 at ρ = 0 to 0.303 at ρ = 0.8 and returns to
0.248 when ρ goes back to 0. Entropy falls in every segment, from 0.786 to 0.590, whatever ρ does.
(b) Late abrupt change, with ρ going from 0 to 0.8 at round 51. At the shift, TV rises by 0.053 and entropy
rises by 0.039 (mean of rounds 51 to 55 minus rounds 46 to 50). Over the 35 rounds before the shift, entropy
falls by 0.100 while TV changes by 0.008. Shaded rounds are the first 15 rounds, which the signal statistics skip.

**Figure 2** (`fig2_system_overview.pdf`, double column).
Overview of DriftGate. (a) Each client runs its client block and two exits on the same request. The client exit
gives p_c and the server block with its exit gives p_s. Their total variation distance, averaged over 64 recent
unlabeled requests, is the only signal. (b) Each serving cluster averages the TV values of its clients and runs its
own controller. The controller compares the cluster signal with its own recent values (temporal comparison) and
with the other clusters in the same round (spatial comparison), and sets the cluster weights λ_z and Λ_z. Edges
exchange only scalars. No labels and no ρ enter any controller. Two clusters are drawn as an example.

**Figure 3** (`fig3_lambda_trajectories.pdf`, double column).
Personalization weight λ chosen by DriftGate and by the same controller fed with entropy, averaged over clusters
and then over seeds (bands show the seed SD). (a) Stepwise composition change, 5 seeds. (b) Client mobility,
3 seeds, where ρ goes from 0 to 0.8 at round 61 while clients keep moving between clusters. Dotted lines mark
the fixed weights λ = 0.4 and λ = 0.2. The first 25 rounds (burn-in and warm-up, shaded) use λ = 0.425.
On the stepwise schedule DriftGate lowers λ to 0.23 on average while ρ = 0.8 and raises it again when ρ returns
to 0. Under client mobility it uses 0.55 before ρ changes and 0.37 after. The entropy controller stays between
0.58 and 0.64 after warm-up in both settings.

**Figure 4** (`fig4_segment_accuracy.pdf`, double column).
Accuracy in each segment of the evaluation rounds (mean over the rounds of a segment, error bars show the seed
SD). (a) Stepwise composition change, 5 seeds. (b) Client mobility, 3 seeds. The best fixed λ changes with the
segment. On the stepwise schedule λ = 0.5 is best when ρ = 0 and λ = 0.2 is best when ρ > 0. Under client
mobility λ = 0.6 is best before ρ changes and λ = 0.2 is best after. DriftGate stays above the worst fixed λ in
every segment.

**Figure 5** (`fig5_role_experiment.pdf`, single column).
Spearman correlation between TV and ρ when the two exits lose their different roles (stepwise composition
change, 3 seeds, open circles show each seed and filled circles show the mean with the SD). With the standard
roles the correlation is 0.87. When both exits play the same role, start from independent weights, or the server
sees less non-Main data, the correlation drops to between -0.21 and 0.15. These runs used the earlier controller
and a shared probe and evaluation pool.
