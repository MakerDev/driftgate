# Round 15 paper figures (captions as used in manuscript/v28)

## motivation_exits (Fig. 1)
Accuracy of the device exit alone (Device only) and the edge exit alone (Edge only) in the commute scenario S1 over the first day. (a) Devices in their home cell and devices in another cell. (b) Requests from the device's own classes (Main), from classes learned only by other devices of the current cell (OOP), and from other classes (OOR). Bars show the mean over five seeds and error bars the standard deviation.

## device_oracle (Fig. 2)
Accuracy gain of a location-aware training policy that sets each device's personalization ratio from its true location (0.70 at home, 0.15 away) over a fixed ratio of 0.4 in S1, for all devices, devices at home, and devices away. Both use confidence-based offloading at inference. Bars show the mean over three seeds, error bars the standard deviation, and dots the seeds.

## replay_time_of_day (Fig. 4)
Accuracy over the day in S1 (left) and S2 (right) on the first day (dotted) and in the frozen replay of the same day with the end-of-day models (solid), for confidence-based offloading, the device exit alone, the edge exit alone, and DriftGate. Seed means.

## home_away (Fig. 5)
Accuracy of devices in their home cell against accuracy of devices in another cell for six inference rules in S1 (left, five seeds) and S2 (right, three seeds), over the first day.

## time_of_day (Fig. 6)
Accuracy over the first day in S1 for confidence-based offloading, the device exit alone, the edge exit alone, and DriftGate (seed means of five seeds). The shaded background shows the share of devices in their home cell (right axis). Dashed lines separate the pre-commute, commute, daytime, return, and evening periods.

## offload_curves (Fig. 7)
Accuracy over the first day against the offloading ratio when only requests on which the device exit is uncertain are sent to the edge, in S1 (left) and S2 (right). Confidence-based offloading and DriftGate offload requests whose device-exit entropy exceeds a threshold, and the geometric ensemble offloads requests whose largest device-exit probability is below a threshold. Confidence-based offloading answers the offloaded requests with the edge exit, the geometric ensemble with the geometric mean of both exits, and DriftGate with its combination, whose weight uses only offloaded requests. Curves interpolate the seed means between thresholds. The dotted line marks the offloading ratio of confidence-based offloading at 0.8 nats.

## demo/mobile_cost_demo (not for submission)
DEMO only: accuracy against modelled latency per request with the assumed per-branch costs of v27 Table 9; not a measurement.
