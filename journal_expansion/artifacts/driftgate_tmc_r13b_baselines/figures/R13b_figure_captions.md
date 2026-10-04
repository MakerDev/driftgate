# Round 13b figure captions

## R13b_fig1_home_away

Accuracy of clients at home against accuracy of clients away for each inference rule in the commute scenario S1 (seeds 0 to 4) and the GeoLife scenario S2 (seeds 0 to 2), training ratio 0.4, evaluation rounds from training round 30 on. Related-work rules are computed from the recorded probabilities of both exits; the confidence-exit rule is read at the server use of entropy routing.

## R13b_fig2_curves

Accuracy against the share of requests that need the server exit in S1 and S2: entropy routing, selective fusion (client exit below the entropy threshold, otherwise the fusion F), selective mean of both exits, and the confidence exit with a product of both exits beyond it. Evaluation rounds from training round 30 on, seed means. The dotted line marks the server use of entropy threshold 0.8.

## R13b_fig3_structures

Accuracy of entropy routing, either exit alone, the mean of both exits and the fusion F for each model and split point, evaluation rounds from training round 30 on, seed means. The default CNN (S1, seeds 0 to 4) and ResNet-18 (middle split, stepwise schedule A, seeds 0 to 2) come from Round 12; ResNet-20 and VGG-11 run S1 with seeds 0 to 2.
