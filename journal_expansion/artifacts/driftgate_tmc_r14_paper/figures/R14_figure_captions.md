# Round 14 figure captions

## R14_fig1_motivation
Accuracy of the device-only and edge-only exits in the commute scenario S1 (training ratio 0.4, seeds 0 to 4, whole day): (a) clients in their home cell and clients in another cell; (b) requests of the device's own training classes (Main), of classes trained only by other clients of the current cell (OOP), and of other classes (OOR). Error bars: standard deviation over seeds.

## R14_fig2_device_oracle
Accuracy gain of a device-level oracle that sets the personalization ratio from the true home/away state (0.70 at home, 0.15 away) over a fixed ratio of 0.4, for the whole day, clients at home and clients away (Round 7, S1, seeds 5 to 7). Bars: mean; error bars: standard deviation; dots: seeds.

## R14_fig3_main
Accuracy of DriftGate minus SplitGP (entropy routing with threshold 0.8) over the whole day in the ten settings. Error bars: standard deviation of the per-seed difference.

## R14_fig4_home_away
Accuracy of clients at home against accuracy of clients away for each inference rule in S1 (seeds 0 to 4) and S2 (GeoLife traces, seeds 0 to 2), whole day.

## R14_fig5_offload
Accuracy against the share of requests sent to the edge when only requests whose device-exit entropy exceeds a threshold are sent: SplitGP answers them with the edge exit, ZTW with the product of both exits (device confidence threshold), and DriftGate with its fusion computed from the requests the edge has received. Whole day, seed means. The dotted line marks the server use of SplitGP at threshold 0.8.

## R14_fig6_time_of_day
Accuracy over the day in S1 for SplitGP, the device-only exit, the edge-only exit and DriftGate (seed means per evaluation round). The shaded background is the share of clients in their home cell (right axis); dashed lines separate pre-commute, commute, daytime, return and evening.
