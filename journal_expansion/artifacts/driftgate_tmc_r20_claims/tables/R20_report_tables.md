**First day, beta 1** (accuracy %, in parentheses DriftGate minus it in pp; same seeds)

| rule | S1 | S2 | S1-fast | partial participation | stepwise change | random mobility | CIFAR-100 | ResNet-18 | K=200 | K=500 |
|---|---|---|---|---|---|---|---|---|---|---|
| Confidence-based offloading | 62.87 (+5.45) | 65.18 (+6.52) | 63.33 (+5.63) | 56.45 (+8.37) | 65.17 (+7.18) | 58.56 (+12.53) | 35.84 (+5.45) | 59.21 (+7.43) | 65.26 (+3.73) | 67.03 (+3.08) |
| Device only | 64.16 (+4.16) | 69.04 (+2.66) | 64.39 (+4.57) | 61.75 (+3.06) | 68.93 (+3.43) | 67.71 (+3.39) | 33.93 (+7.36) | 66.39 (+0.25) | 62.63 (+6.37) | 63.12 (+7.00) |
| Raw edge only | 62.91 (+5.41) | 65.24 (+6.46) | 63.38 (+5.59) | 56.44 (+8.38) | 65.20 (+7.15) | 58.61 (+12.48) | 35.84 (+5.45) | 57.04 (+9.60) | 65.30 (+3.69) | 67.07 (+3.04) |
| Probability average | 67.32 (+0.99) | 69.88 (+1.81) | 68.06 (+0.90) | 62.64 (+2.18) | 71.44 (+0.92) | 69.58 (+1.52) | 38.15 (+3.14) | 66.39 (+0.25) | 68.90 (+0.10) | 70.38 (-0.27) |
| Logit sum | 67.37 (+0.95) | 70.07 (+1.63) | 68.12 (+0.84) | 62.94 (+1.88) | 71.39 (+0.97) | 69.23 (+1.87) | 39.83 (+1.46) | 66.00 (+0.64) | 68.86 (+0.13) | 70.29 (-0.18) |
| No correction + adaptive w | 67.11 (+1.20) | 69.29 (+2.41) | 67.84 (+1.12) | 62.06 (+2.76) | 71.25 (+1.10) | 69.33 (+1.77) | 37.50 (+3.79) | 66.46 (+0.17) | 69.00 (-0.00) | 70.60 (-0.49) |
| Correction + w 0.5 | 68.13 (+0.19) | 71.77 (-0.07) | 68.74 (+0.23) | 64.85 (-0.03) | 72.21 (+0.14) | 70.96 (+0.14) | 41.45 (-0.16) | 66.71 (-0.07) | 68.58 (+0.41) | 69.62 (+0.49) |
| Correction + w 0.2 | 68.16 (+0.15) | 71.03 (+0.67) | 68.93 (+0.03) | 64.22 (+0.60) | 71.78 (+0.58) | 70.51 (+0.58) | 40.95 (+0.34) | 64.02 (+2.62) | 69.23 (-0.23) | 70.53 (-0.41) |
| Correction + w 0.45 | 68.21 (+0.11) | 71.68 (+0.02) | 68.86 (+0.10) | 64.80 (+0.02) | 72.22 (+0.14) | 70.98 (+0.12) | 41.36 (-0.07) | 66.49 (+0.15) | 68.80 (+0.19) | 69.90 (+0.21) |
| Corrected edge only | 67.87 (+0.45) | 70.49 (+1.21) | 68.66 (+0.31) | 63.64 (+1.18) | 71.24 (+1.12) | 69.89 (+1.21) | 40.69 (+0.59) | 61.72 (+4.92) | 69.15 (-0.16) | 70.53 (-0.42) |
| Correction + product | 67.90 (+0.41) | 71.76 (-0.06) | 68.48 (+0.49) | 64.83 (-0.01) | 72.08 (+0.27) | 70.78 (+0.31) | 41.73 (-0.45) | 66.59 (+0.05) | 68.15 (+0.84) | 69.11 (+1.00) |
| DriftGate | 68.31 | 71.70 | 68.96 | 64.82 | 72.36 | 71.09 | 41.29 | 66.64 | 69.00 | 70.11 |
| strongest main-table rule | Logit sum | Logit sum | Logit sum | Logit sum | Probability average | Probability average | Label-shift EM | Logit-entropy weighting | Probability average | Probability average |

**Trained models, beta 1** (accuracy %, in parentheses DriftGate minus it in pp; same seeds)

| rule | roun S1 | roun S2 | roun random mobility | roun K=500 | froz S1 replay | froz S2 replay | thre S1 three-day | thre S2 three-day |
|---|---|---|---|---|---|---|---|---|
| Confidence-based offloading | 64.80 (+2.31) | 68.04 (+3.92) | 61.70 (+3.69) | 69.62 (-0.34) | 71.47 (+0.48) | 74.76 (+1.77) | 71.35 (+0.58) | 74.58 (+1.74) |
| Device only | 61.77 (+5.33) | 68.33 (+3.62) | 60.50 (+4.90) | 60.90 (+8.38) | 66.38 (+5.57) | 71.12 (+5.41) | 66.44 (+5.49) | 70.62 (+5.71) |
| Raw edge only | 64.85 (+2.25) | 68.11 (+3.85) | 61.77 (+3.63) | 69.67 (-0.39) | 71.74 (+0.20) | 74.94 (+1.60) | 71.64 (+0.29) | 74.76 (+1.57) |
| Probability average | 66.74 (+0.36) | 70.72 (+1.24) | 65.29 (+0.11) | 70.29 (-1.02) | 72.27 (-0.33) | 75.90 (+0.64) | 72.19 (-0.27) | 75.69 (+0.64) |
| Logit sum | 66.68 (+0.42) | 70.82 (+1.14) | 65.06 (+0.33) | 70.06 (-0.78) | 72.08 (-0.13) | 75.86 (+0.68) | 72.04 (-0.12) | 75.64 (+0.69) |
| No correction + adaptive w | 66.71 (+0.40) | 70.21 (+1.75) | 65.28 (+0.12) | 70.71 (-1.44) | 72.46 (-0.52) | 75.88 (+0.65) | 72.36 (-0.44) | 75.66 (+0.67) |
| Correction + w 0.5 | 66.81 (+0.29) | 71.97 (-0.01) | 65.15 (+0.25) | 68.63 (+0.65) | 71.53 (+0.41) | 76.23 (+0.31) | 71.52 (+0.40) | 75.99 (+0.34) |
| Correction + w 0.2 | 67.55 (-0.45) | 71.70 (+0.26) | 65.93 (-0.53) | 70.19 (-0.92) | 72.58 (-0.63) | 76.66 (-0.13) | 72.52 (-0.60) | 76.50 (-0.17) |
| Correction + w 0.45 | 67.02 (+0.08) | 71.97 (-0.02) | 65.37 (+0.03) | 69.03 (+0.25) | 71.83 (+0.12) | 76.40 (+0.14) | 71.81 (+0.12) | 76.18 (+0.15) |
| Corrected edge only | 67.66 (-0.55) | 71.42 (+0.53) | 66.05 (-0.65) | 70.59 (-1.31) | 72.74 (-0.79) | 76.59 (-0.05) | 72.67 (-0.74) | 76.43 (-0.10) |
| Correction + product | 66.48 (+0.63) | 71.82 (+0.14) | 64.78 (+0.61) | 67.95 (+1.32) | 71.12 (+0.83) | 76.00 (+0.54) | 71.13 (+0.79) | 75.71 (+0.62) |
| DriftGate | 67.10 | 71.96 | 65.40 | 69.28 | 71.95 | 76.54 | 71.93 | 76.33 |
| strongest main-table rule | Probability average | Logit sum | Probability average | Probability average | Probability average | Probability average | Probability average | Probability average |

**Same mask: DriftGate minus the best corrected fixed rule / minus the best rule without the prior correction (pp)**

| setting | beta 0.25 | beta 0.5 | beta 0.75 | beta 1 | uncorr. 0.25 | uncorr. 0.5 | uncorr. 0.75 | uncorr. 1 | edge calls per request (0.25/0.5/0.75/1) | two-cell share of offloaded |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | +0.027 (C+w 0.2) | +0.050 (C+w 0.2) | +0.088 (C+w 0.2) | +0.107 (C+w 0.45) | +0.380 | +0.664 | +0.840 | +0.946 | 0.287 / 0.565 / 0.844 / 1.132 | 0.132 |
| S2 | +0.010 (C+w 0.5) | +0.006 (C+w 0.5) | -0.032 (C+w 0.5) | -0.073 (C+w 0.5) | +0.635 | +1.175 | +1.502 | +1.632 | 0.268 / 0.528 / 0.788 / 1.058 | 0.058 |
| S1-fast | -0.011 (C+w 0.2) | -0.028 (C+w 0.2) | -0.017 (C+w 0.2) | +0.033 (C+w 0.2) | +0.352 | +0.596 | +0.740 | +0.843 | 0.279 / 0.550 / 0.821 / 1.101 | 0.101 |
| partial participation | +0.017 (C+w 0.5) | +0.019 (C+w 0.5) | +0.001 (C+w 0.5) | -0.026 (C+w 0.5) | +0.574 | +1.114 | +1.539 | +1.879 | 0.281 / 0.552 / 0.825 / 1.107 | 0.107 |
| stepwise change | +0.042 (C+w 0.45) | +0.090 (C+w 0.45) | +0.120 (C+w 0.45) | +0.137 (C+w 0.45) | +0.438 | +0.692 | +0.840 | +0.920 | 0.354 / 0.697 / 1.042 / 1.400 | 0.400 |
| random mobility | +0.043 (C+w 0.45) | +0.091 (C+w 0.45) | +0.112 (C+w 0.45) | +0.119 (C+w 0.45) | +0.687 | +1.062 | +1.326 | +1.517 | 0.498 / 0.981 / 1.465 / 1.969 | 0.969 |
| CIFAR-100 | -0.151 (C+product) | -0.268 (C+product) | -0.367 (C+product) | -0.445 (C+product) | +0.508 | +0.926 | +1.258 | +1.456 | 0.279 / 0.551 / 0.824 / 1.107 | 0.107 |
| ResNet-18 | +0.002 (C+w 0.5) | -0.033 (C+w 0.5) | -0.074 (C+w 0.5) | -0.067 (C+w 0.5) | +0.110 | +0.178 | +0.215 | +0.175 | 0.317 / 0.624 / 0.931 / 1.250 | 0.250 |
| K=200 | -0.080 (CE) | -0.205 (CE) | -0.250 (C+w 0.2) | -0.231 (C+w 0.2) | +0.101 | +0.074 | -0.004 | -0.004 | 0.294 / 0.579 / 0.864 / 1.159 | 0.159 |
| K=500 | -0.145 (CE) | -0.362 (CE) | -0.482 (CE) | -0.422 (CE) | -0.034 | -0.245 | -0.437 | -0.487 | 0.292 / 0.575 / 0.859 / 1.153 | 0.153 |
| S1 replay | -0.104 (CE) | -0.347 (CE) | -0.625 (CE) | -0.791 (CE) | -0.153 | -0.312 | -0.440 | -0.517 | 0.288 / 0.566 / 0.845 / 1.132 | 0.132 |
| S2 replay | -0.002 (C+w 0.2) | -0.037 (C+w 0.2) | -0.093 (C+w 0.2) | -0.128 (C+w 0.2) | +0.184 | +0.429 | +0.584 | +0.637 | 0.269 / 0.529 / 0.790 / 1.058 | 0.058 |

**Target A: first grid beta reaching the target (interpolated offloading ratio, rule)**

| setting | target | DriftGate | best corrected fixed rule | best rule without prior correction | DriftGate-P fixed-threshold sweep (Round 18 A3) |
|---|---|---|---|---|---|
| S1 | 67.37 (Logit sum) | 0.5 (0.421) | 0.5 (0.430, C+w 0.2) | 1 (1.000, Logit sum) | 0.517 |
| S2 | 70.07 (Logit sum) | 0.25 (0.122) | 0.25 (0.123, C+w 0.5) | 0.25 (0.174, Logit sum) | 0.242 |
| S1-fast | 68.12 (Logit sum) | 0.5 (0.460) | 0.5 (0.455, C+w 0.2) | 1 (1.000, Logit sum) | 0.524 |
| partial participation | 62.94 (Logit sum) | 0.25 (0.158) | 0.25 (0.159, C+w 0.5) | 0.25 (0.226, Logit sum) | 0.136 |
| stepwise change | 71.44 (Probability average) | 0.5 (0.345) | 0.5 (0.363, C+w 0.45) | 0.75 (0.709, Probability average) | 0.474 |
| random mobility | 69.58 (Probability average) | 0.25 (0.224) | 0.25 (0.228, C+w 0.45) | 0.5 (0.476, Probability average) | 0.372 |
| CIFAR-100 | 40.44 (Label-shift EM) | 1 (0.762) | 0.75 (0.704, C+product) | NA | 0.953 |
| ResNet-18 | 66.54 (Logit-entropy weighting) | 0.25 (0.091) | 0.25 (0.091, C+w 0.5) | 0.25 (0.121, No correction + adaptive w) | 0.225 |
| K=200 | 68.90 (Probability average) | 1 (0.895) | 0.75 (0.714, C+w 0.2) | 1 (0.891, No correction + adaptive w) | 0.850 |
| K=500 | 70.38 (Probability average) | NA | 1 (0.787, CE) | 1 (0.811, No correction + adaptive w) | not reached |
| S1 replay | 72.27 (Probability average) | NA | 0.75 (0.700, CE) | 1 (0.771, No correction + adaptive w) | not reached |
| S2 replay | 75.90 (Probability average) | 0.5 (0.480) | 0.5 (0.474, C+w 0.2) | 1 (1.000, Probability average) | 0.509 |
| S1 development | 66.66 (Probability average) | 0.5 (0.379) | 0.5 (0.390, C+w 0.45) | 0.75 (0.702, Logit sum) |  |

**Target B: first grid beta reaching the target (interpolated offloading ratio, rule)**

| setting | target | DriftGate | best corrected fixed rule | best rule without prior correction | DriftGate-P fixed-threshold sweep (Round 18 A3) |
|---|---|---|---|---|---|
| S1 | 68.16 (Correction + w 0.2) | 0.75 (0.725) | 1 (0.885, C+w 0.45) | NA | 0.762 |
| S2 | 71.77 (Correction + w 0.5) | 0.75 (0.530) | 0.75 (0.535, C+w 0.5) | NA | not reached |
| S1-fast | 68.93 (Correction + w 0.2) | 1 (0.941) | 1 (1.000, C+w 0.2) | NA | 0.870 |
| partial participation | 64.85 (Correction + w 0.5) | NA | 1 (1.000, C+w 0.5) | NA | not reached |
| stepwise change | 72.21 (Correction + w 0.5) | 0.75 (0.676) | 1 (0.970, C+w 0.45) | NA | 0.727 |
| random mobility | 70.96 (Correction + w 0.5) | 0.75 (0.704) | 1 (0.937, C+w 0.45) | NA | 0.781 |
| CIFAR-100 | 41.73 (Correction + product) | NA | 1 (1.000, C+product) | NA | not reached |
| ResNet-18 | 66.71 (Correction + w 0.5) | 0.25 (0.188) | 0.25 (0.188, C+w 0.5) | 0.25 (0.251, No correction + adaptive w) | 0.433 |
| K=200 | 69.23 (Correction + w 0.2) | NA | 1 (1.000, C+w 0.2) | NA | not reached |
| K=500 | 70.60 (No correction + adaptive w) | NA | NA | 1 (1.000, No correction + adaptive w) | not reached |
| S1 replay | 72.74 (Corrected edge only) | NA | 1 (1.000, CE) | NA | not reached |
| S2 replay | 76.66 (Correction + w 0.2) | NA | 1 (1.000, C+w 0.2) | NA | not reached |
| S1 development | 67.50 (Correction + w 0.5) | 0.75 (0.719) | 1 (0.925, C+w 0.45) | NA |  |

**Target C: first grid beta reaching the target (interpolated offloading ratio, rule)**

| setting | target | DriftGate | best corrected fixed rule | best rule without prior correction | DriftGate-P fixed-threshold sweep (Round 18 A3) |
|---|---|---|---|---|---|
| S1 | 67.90 (lowest best of the corrected group: Correction + product) | 0.75 (0.590) | 0.75 (0.627, C+w 0.2) | NA |  |
| S2 | 71.23 (lowest best of the corrected group: Corrected edge only) | 0.5 (0.279) | 0.5 (0.283, C+w 0.5) | NA |  |
| S1-fast | 68.48 (lowest best of the corrected group: Correction + product) | 0.75 (0.580) | 0.75 (0.568, C+w 0.2) | NA |  |
| partial participation | 63.81 (lowest best of the corrected group: Corrected edge only) | 0.5 (0.309) | 0.5 (0.315, C+w 0.5) | NA |  |
| stepwise change | 71.57 (lowest best of the corrected group: Corrected edge only) | 0.5 (0.381) | 0.5 (0.400, C+w 0.45) | NA |  |
| random mobility | 70.23 (lowest best of the corrected group: Corrected edge only) | 0.5 (0.368) | 0.5 (0.388, C+w 0.45) | NA |  |
| CIFAR-100 | 40.69 (lowest best of the corrected group: Corrected edge only) | 1 (0.834) | 0.75 (0.738, C+product) | NA |  |
| ResNet-18 | 66.39 (lowest best of the corrected group: Correction + w 0.2) | 0 (0.000) | 0 (0.000, C+w 0.5) | 0 (0.000, No correction + adaptive w) |  |
| K=200 | 68.15 (lowest best of the corrected group: Correction + product) | 0.75 (0.560) | 0.5 (0.498, CE) | 0.75 (0.575, No correction + adaptive w) |  |
| K=500 | 69.11 (lowest best of the corrected group: Correction + product) | 0.75 (0.529) | 0.5 (0.470, CE) | 0.5 (0.483, No correction + adaptive w) |  |
| S1 replay | 71.12 (lowest best of the corrected group: Correction + product) | 0.5 (0.487) | 0.5 (0.441, CE) | 0.5 (0.443, No correction + adaptive w) |  |
| S2 replay | 76.00 (lowest best of the corrected group: Correction + product) | 0.5 (0.498) | 0.5 (0.492, C+w 0.2) | NA |  |
| S1 development | 66.86 (lowest best of the corrected group: Corrected edge only) | 0.5 (0.426) | 0.5 (0.439, C+w 0.45) | NA |  |

**S1 operating points chosen on development seeds 5-7, evaluated on seeds 0-4**

| target | development target | rule | beta | evaluation accuracy | offloading | edge calls | evaluation target | reached |
|---|---|---|---|---|---|---|---|---|
| A | 66.66 (Probability average) | DriftGate | 0.5 | 67.73 | 0.499 | 0.565 | 67.37 | yes |
| A | 66.66 (Probability average) | No correction + adaptive w | NA (not reached on development) |  |  |  | 67.37 |  |
| A | 66.66 (Probability average) | Probability average | 0.75 | 67.31 | 0.745 | 0.844 | 67.37 | no |
| A | 66.66 (Probability average) | Logit sum | 0.75 | 67.36 | 0.745 | 0.844 | 67.37 | no |
| A | 66.66 (Probability average) | Raw edge only | NA (not reached on development) |  |  |  | 67.37 |  |
| A | 66.66 (Probability average) | Correction + w 0.5 | 0.5 | 67.60 | 0.499 | 0.565 | 67.37 | yes |
| A | 66.66 (Probability average) | Correction + w 0.2 | 0.5 | 67.68 | 0.499 | 0.565 | 67.37 | yes |
| A | 66.66 (Probability average) | Correction + w 0.45 | 0.5 | 67.66 | 0.499 | 0.565 | 67.37 | yes |
| A | 66.66 (Probability average) | Corrected edge only | 0.5 | 67.55 | 0.499 | 0.565 | 67.37 | yes |
| A | 66.66 (Probability average) | Correction + product | 0.5 | 67.41 | 0.499 | 0.565 | 67.37 | yes |
| B | 67.50 (Correction + w 0.5) | DriftGate | 0.75 | 68.20 | 0.745 | 0.844 | 68.16 | yes |
| B | 67.50 (Correction + w 0.5) | No correction + adaptive w | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Probability average | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Logit sum | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Raw edge only | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Correction + w 0.5 | 1 | 68.13 | 1.000 | 1.132 | 68.16 | no |
| B | 67.50 (Correction + w 0.5) | Correction + w 0.2 | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Correction + w 0.45 | 1 | 68.21 | 1.000 | 1.132 | 68.16 | yes |
| B | 67.50 (Correction + w 0.5) | Corrected edge only | NA (not reached on development) |  |  |  | 68.16 |  |
| B | 67.50 (Correction + w 0.5) | Correction + product | NA (not reached on development) |  |  |  | 68.16 |  |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | DriftGate | 0.5 | 67.73 | 0.499 | 0.565 | 67.90 | no |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | No correction + adaptive w | NA (not reached on development) |  |  |  | 67.90 |  |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Probability average | NA (not reached on development) |  |  |  | 67.90 |  |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Logit sum | NA (not reached on development) |  |  |  | 67.90 |  |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Raw edge only | NA (not reached on development) |  |  |  | 67.90 |  |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Correction + w 0.5 | 0.5 | 67.60 | 0.499 | 0.565 | 67.90 | no |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Correction + w 0.2 | 0.5 | 67.68 | 0.499 | 0.565 | 67.90 | no |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Correction + w 0.45 | 0.5 | 67.66 | 0.499 | 0.565 | 67.90 | no |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Corrected edge only | 0.75 | 67.91 | 0.745 | 0.844 | 67.90 | yes |
| C | 66.86 (lowest best of the corrected group: (Corrected edge only)) | Correction + product | 0.5 | 67.41 | 0.499 | 0.565 | 67.90 | no |

**Correction ablation (same mask, same weight series)**

| setting | beta | rule | b = 0 | a = 0.25 | a0 = 0.2326 | current a_k | a_k minus b = 0 [seeds +] | primary constant | a_k minus primary constant [seeds +] |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 1 | DriftGate | 67.11 | 68.45 | 68.40 | 68.31 | +1.201 [5/5] | a0 = 0.2326 | -0.088 [0/5] |
| S1 | 1 | Corrected edge only | 62.91 | 68.19 | 68.26 | 67.87 | +4.959 [5/5] | a0 = 0.2326 | -0.394 [0/5] |
| S1 | 1 | Correction + w 0.5 | 67.32 | 68.24 | 68.19 | 68.13 | +0.803 [5/5] | a0 = 0.2326 | -0.064 [0/5] |
| S1 | 0.5 | DriftGate | 66.90 | 67.83 | 67.80 | 67.73 | +0.825 [5/5] | a0 = 0.2326 | -0.066 [0/5] |
| S1 | 0.5 | Corrected edge only | 64.73 | 67.75 | 67.78 | 67.55 | +2.824 [5/5] | a0 = 0.2326 | -0.224 [0/5] |
| S1 | 0.5 | Correction + w 0.5 | 67.02 | 67.69 | 67.65 | 67.60 | +0.582 [5/5] | a0 = 0.2326 | -0.053 [1/5] |
| S2 | 1 | DriftGate | 69.29 | 71.91 | 71.98 | 71.70 | +2.407 [3/3] | a = 0.25 | -0.213 [1/3] |
| S2 | 1 | Corrected edge only | 65.24 | 70.85 | 71.02 | 70.49 | +5.249 [3/3] | a = 0.25 | -0.362 [1/3] |
| S2 | 1 | Correction + w 0.5 | 69.88 | 71.95 | 72.00 | 71.77 | +1.888 [3/3] | a = 0.25 | -0.177 [1/3] |
| S2 | 0.5 | DriftGate | 70.17 | 71.92 | 71.95 | 71.77 | +1.599 [3/3] | a = 0.25 | -0.145 [1/3] |
| S2 | 0.5 | Corrected edge only | 68.20 | 71.47 | 71.56 | 71.23 | +3.036 [3/3] | a = 0.25 | -0.236 [1/3] |
| S2 | 0.5 | Correction + w 0.5 | 70.44 | 71.89 | 71.92 | 71.77 | +1.325 [3/3] | a = 0.25 | -0.123 [1/3] |
| S1 replay | 1 | DriftGate | 72.46 | 72.44 | 72.35 | 71.95 | -0.517 [1/5] | a0 = 0.2326 | -0.401 [0/5] |
| S1 replay | 1 | Corrected edge only | 71.74 | 73.30 | 73.27 | 72.74 | +0.994 [5/5] | a0 = 0.2326 | -0.534 [0/5] |
| S1 replay | 1 | Correction + w 0.5 | 72.27 | 71.98 | 71.90 | 71.53 | -0.739 [0/5] | a0 = 0.2326 | -0.368 [0/5] |
| S1 replay | 0.5 | DriftGate | 71.52 | 71.59 | 71.53 | 71.21 | -0.312 [1/5] | a0 = 0.2326 | -0.323 [0/5] |
| S1 replay | 0.5 | Corrected edge only | 70.90 | 71.96 | 71.94 | 71.55 | +0.649 [5/5] | a0 = 0.2326 | -0.389 [0/5] |
| S1 replay | 0.5 | Correction + w 0.5 | 71.42 | 71.27 | 71.20 | 70.91 | -0.512 [0/5] | a0 = 0.2326 | -0.296 [0/5] |
| S2 replay | 1 | DriftGate | 75.88 | 76.62 | 76.61 | 76.54 | +0.651 [2/3] | a = 0.25 | -0.087 [1/3] |
| S2 replay | 1 | Corrected edge only | 74.94 | 76.71 | 76.73 | 76.59 | +1.652 [3/3] | a = 0.25 | -0.118 [1/3] |
| S2 replay | 1 | Correction + w 0.5 | 75.90 | 76.34 | 76.30 | 76.23 | +0.331 [2/3] | a = 0.25 | -0.106 [1/3] |
| S2 replay | 0.5 | DriftGate | 75.58 | 76.10 | 76.08 | 76.01 | +0.429 [2/3] | a = 0.25 | -0.091 [1/3] |
| S2 replay | 0.5 | Corrected edge only | 74.91 | 76.10 | 76.10 | 75.99 | +1.078 [3/3] | a = 0.25 | -0.103 [1/3] |
| S2 replay | 0.5 | Correction + w 0.5 | 75.54 | 75.87 | 75.84 | 75.78 | +0.240 [2/3] | a = 0.25 | -0.096 [1/3] |

**a_k and b_k**

| setting | a_k median (p5-p95) | b_k median (p5-p95) | within-device SD (mean) | within-device SD (max) | between-device SD | median a_k home | median a_k away |
|---|---|---|---|---|---|---|---|
| S1 | 0.2403 (0.1898-0.3674) | 1.151 (0.543-1.452) | 0.0401 | 0.0882 | 0.0315 | 0.2601 | 0.2196 |
| S2 | 0.2515 (0.1690-0.3905) | 1.091 (0.445-1.593) | 0.0454 | 0.1380 | 0.0416 | 0.2610 | 0.2150 |
| S1 replay | 0.2463 (0.1168-0.3704) | 1.118 (0.530-2.023) | 0.0423 | 0.1256 | 0.0508 | 0.2942 | 0.1875 |
| S2 replay | 0.2341 (0.1317-0.3979) | 1.185 (0.414-1.886) | 0.0335 | 0.1312 | 0.0503 | 0.2478 | 0.1794 |
| S1 development | 0.2326 (0.1817-0.4068) | 1.194 (0.377-1.505) | 0.0478 | 0.1128 | 0.0368 | 0.2550 | 0.2147 |
| S1-fast | 0.2341 (0.1920-0.3800) | 1.185 (0.490-1.437) | 0.0402 | 0.0939 | 0.0330 | 0.2592 | 0.2182 |
| partial participation | 0.2410 (0.1073-0.4355) | 1.147 (0.259-2.119) | 0.0894 | 0.1461 | 0.0337 | 0.2636 | 0.2225 |
| stepwise change | 0.2603 (0.1985-0.3466) | 1.044 (0.634-1.396) | 0.0000 | 0.0000 | 0.0430 | nan | nan |
| random mobility | 0.2179 (0.1875-0.2657) | 1.278 (1.016-1.467) | 0.0208 | 0.0486 | 0.0112 | nan | nan |
| CIFAR-100 | 0.2402 (0.1994-0.3245) | 1.151 (0.733-1.390) | 0.0328 | 0.0705 | 0.0210 | 0.2630 | 0.2135 |
| ResNet-18 | 0.3211 (0.2452-0.4448) | 0.749 (0.222-1.124) | 0.0000 | 0.0000 | 0.0637 | nan | nan |
| K=200 | 0.2367 (0.1796-0.3603) | 1.171 (0.574-1.519) | 0.0418 | 0.1152 | 0.0349 | 0.2589 | 0.2168 |
| K=500 | 0.2384 (0.1794-0.3665) | 1.161 (0.547-1.520) | 0.0413 | 0.1261 | 0.0368 | 0.2656 | 0.2165 |
