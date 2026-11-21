# Round 18 최종 보고서: own class 판별기로 offloading을 정하는 방법

이 보고서는 지시문 6.5절 2번의 최종 보고서입니다. 작업 기간은 2026-11-20부터 2026-11-22까지이고, branch는 `r18-own-class-detector`입니다. 산출물은 `journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/`에 있습니다. 해석과 결론은 이 보고서와 `decisions.md`에만 적었습니다. 중간 보고서(`report_stage1.md`, `report_stage2.md`)에는 표와 수치만 있습니다.

표의 수치는 모두 다음 경로로 만들었습니다: run 기록(JSON, npz) → 분석 스크립트 → `tables/`의 CSV → 이 보고서의 표. 판정 기준은 결과를 보기 전에 `decision_rule.md`로 commit했습니다(commit `fc1c157`).

| 단계 | 스크립트 | commit |
|---|---|---|
| run마다 조건별 분석 cache | `scripts/r18oc_analysis.py` | `c6f863c` |
| 2단계의 칸, 7.3절의 판정 | `scripts/r18oc_tables.py stage2` | `c6f863c` |
| 실패 원인의 진단 | `scripts/r18oc_diagnose.py stage2` | `b4884cc` |
| 개선 시도 1·2번의 관문과 판정 칸 | `scripts/r18oc_tables.py attempt_dev`, `attempt_cells` | `c6f863c` |
| 개선 시도 3번 | `scripts/r18oc_attempt3.py` | `5b441d3` |
| 시도 2번의 비용 | `scripts/r18oc_cost.py` | `d17b43e` |
| 이 보고서의 표와 요약 그림 | `scripts/r18oc_final.py cells`, `figure`, `md` | `30838ac` |
| 모바일 내보내기와 가정값 그림 | `mobile/r18oc_export_mobile.py`, `r18oc_tflite_check.py`, `r18oc_mobile_plot.py` | `eda5be3`, `b0a1509` |

이 보고서에서 쓰는 약어와 기호는 다음과 같습니다.

- 모델 구성은 세 가지입니다. `cnn-scratch`는 v32와 같은 CNN을 처음부터 학습한 구성입니다. `pretrained-finetuned-block`(이하 ft)은 ImageNet으로 사전학습한 MobileNetV2의 앞부분을 device block으로 쓰고 함께 학습한 구성입니다. `pretrained-frozen-block`(이하 fz)은 같은 device block을 고정한 구성입니다.
- S1, S2는 CIFAR-10으로 만든 두 이동 설정입니다. CIFAR-100은 S1의 이동에 CIFAR-100을 쓴 설정입니다.
- 조건은 세 가지입니다. `first-day`는 학습하는 첫날(라운드 1–150)입니다. `second-day`는 같은 학습을 이어 가는 둘째 날(라운드 151–300)입니다. `frozen-replay`는 라운드 150의 모델을 고정하고 둘째 날을 재생한 조건입니다.
- Main, OOP, OOR은 요청의 종류입니다. Main은 요청한 device가 학습한 class(own class)의 요청입니다. OOP(out-of-preference)는 지금 cell의 다른 device만 학습한 class의 요청이고, OOR(out-of-region)은 그 밖의 요청입니다.
- offloading 비율 ρ는 요청 가운데 edge로 보내는 요청의 비율입니다.
- 비교 방법은 점수 5개(`device-entropy`, `device-max-prob`, `device-energy`, `own-mahalanobis`, `own-knn`)와 답 규칙 5개를 곱한 25개입니다. B(ρ)는 비율 ρ마다 이 25개의 seed 평균 정확도 가운데 가장 높은 값입니다(이하 envelope). A*는 B(1.0) − 0.5 %p입니다.
- S는 1 − (새 방법이 A*에 닿는 비율)/(envelope가 A*에 닿는 비율)이고, 새 방법이 A*에 닿지 못하면 0입니다. G(ρ)는 비율 ρ에서 새 방법의 정확도에서 B(ρ)를 뺀 값(%p, 퍼센트포인트)입니다.
- 칸 하나는 (설정, 모델 구성, 조건)입니다. 칸의 판정은 평가용 seed 0–2의 평균으로 합니다. 통과는 S ≥ 0.25 또는 G(0.5) ≥ 2.0 %p이고, 부분 통과는 S ≥ 0.15 또는 G(0.5) ≥ 1.0 %p이며, 그 밖은 미달입니다.
- 여지는 request-kind oracle이 통과 기준에 닿는지로 정합니다. request-kind oracle은 요청의 종류(label)를 알고 Main이 아닌 요청부터 보내는 규칙입니다. oracle도 통과하지 못하는 칸은 "여지 없음"이고, 판정에서 뺍니다(7.2절).
- 판정 대상 칸은 사전학습 구성마다 S1·S2의 `second-day`·`frozen-replay` 네 칸 가운데 여지가 있는 칸입니다(D17, D19).
- 새 방법은 판별기 순위(own class일 확률 g가 낮은 요청부터 보냄)와 답 규칙 `edge-corrected-by-detector`(판별기로 추정한 비율 r로 edge 출력을 보정)의 조합입니다. 판별기 조합은 세 구성 모두 `exit|mlp64|both`(입력|모델|음성 표본)입니다(D11, D13, D15).
- v32 방법은 `device-entropy` 순위와 `average-corrected-even` 답 규칙(보정한 edge 출력과 device 출력의 평균)의 조합이고, 25개 비교 방법 가운데 하나입니다.
- 괄호 안의 수는 SD입니다. 차이의 SD는 같은 seed끼리 뺀 차이의 표준편차(ddof = 1)입니다.

## 1. 결론: 중단 권고

1. 2단계 평가용 run의 24칸(구성 3개, 설정 2–3개, 조건 3개)이 모두 미달입니다. 모든 칸에서 비율 0.5의 새 방법이 envelope보다 낮습니다(G(0.5) −0.02…−4.25 %p).
2. 7.4절의 개선 시도 세 개 가운데 1번은 두 구성 모두 관문(+0.5 %p)을 넘지 못했습니다. 관문을 넘은 2번과 3번도 판정 대상 칸을 하나도 부분 통과로 올리지 못했습니다. 판정 대상 칸에서 가장 높은 G(0.5)는 시도 3번 ft S2 `frozen-replay`의 −0.36 %p(SD 0.39)입니다.
3. 따라서 7.3절 표의 마지막 행("개선 시도 3번 뒤에도 미달")에 해당합니다. 3단계는 건너뛰었고, 판정은 "중단 권고"입니다. `cnn-scratch`도 통과한 칸이 없습니다.
4. 판별기의 순위는 fz의 CIFAR-10에서 device 출력 점수보다 좋았습니다(AUROC 0.87–0.90 대 0.55–0.76). 그러나 이 차이는 정확도로 이어지지 않았습니다. 2단계 24칸 가운데 23칸에서 비율 0.5의 envelope는 `device-max-prob` 순위에 device 출력을 섞거나 edge 출력을 보정하는 답 규칙을 붙인 조합이었습니다. 시도 2번의 run에서는 12칸 가운데 6칸에서 envelope가 `own-mahalanobis`나 `own-knn` 순위의 규칙에서 나왔습니다.
5. 경계(S 기준값 ± 0.01, G 기준값 ± 0.05 %p)에 걸린 칸은 없습니다.

논문의 규모: 지금까지의 결과로는 letter 규모의 논문이 적당합니다. 주제는 "edge가 학습한 own class 판별기는 순위를 개선하지만, 같은 offloading 비율의 정확도는 답 규칙이 좌우한다"는 부정적 결과와 그 진단입니다.

## 2. 대표 수치

### 2.1 2단계의 칸 (설정 × 조건)

칸마다 "S / G(0.5) %p (SD) / 판정"입니다. "(no room)"은 여지 없는 칸입니다. `cnn-scratch`의 CIFAR-100은 지시문 6.2절의 2단계 구성에 없어서 돌리지 않았습니다.

| setting | condition | cnn | ft | fz |
|---|---|---|---|---|
| S1 | first-day | -0.723 / -0.61 (0.17) / fail | 0.000 / -2.05 (0.41) / fail | 0.000 / -0.31 (0.25) / fail |
| S1 | second-day | -0.389 / -0.69 (0.05) / fail | -0.017 / -0.02 (0.84) / fail (no room) | 0.000 / -1.87 (0.83) / fail |
| S1 | frozen-replay | -0.436 / -0.91 (0.19) / fail | 0.000 / -1.73 (1.13) / fail | 0.000 / -1.96 (0.73) / fail |
| S2 | first-day | 0.000 / -0.20 (0.51) / fail | 0.000 / -1.81 (0.97) / fail | 0.000 / -1.29 (0.67) / fail (no room) |
| S2 | second-day | -0.963 / -0.54 (0.30) / fail | 0.000 / -0.81 (0.95) / fail (no room) | 0.000 / -1.75 (1.05) / fail (no room) |
| S2 | frozen-replay | -0.677 / -0.67 (0.15) / fail | 0.000 / -1.38 (0.75) / fail | 0.000 / -1.77 (1.13) / fail (no room) |
| CIFAR-100 | first-day | not run | -0.063 / -4.25 (0.35) / fail (no room) | -0.019 / -3.07 (0.21) / fail (no room) |
| CIFAR-100 | second-day | not run | 0.000 / -2.63 (0.36) / fail (no room) | -0.067 / -2.59 (0.21) / fail (no room) |
| CIFAR-100 | frozen-replay | not run | -0.146 / -2.38 (0.29) / fail (no room) | -0.051 / -1.98 (0.17) / fail (no room) |

### 2.2 판정 대상 칸과 개선 시도

사전학습 두 구성에서 S1·S2의 `second-day`·`frozen-replay` 칸을 2단계와 시도 2·3번에 걸쳐 모았습니다. 시도 2번은 학습을 새로 하므로 여지를 그 run에서 다시 계산했습니다(D19). 시도 3번은 같은 run에서 답 규칙만 바꾸므로 envelope와 여지가 2단계와 같습니다. 시도 1번은 관문을 넘지 못해 평가용 run이 없습니다.

| source | model config | setting | condition | room | judged cell | S | G(0.5) pp mean (SD) | new at 0.5 | B(0.5) | B(0.5) rule | judgement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 2 | ft | S1 | second-day | no | no | -0.017 | -0.02 (0.84) | 69.11 | 69.13 | device-max-prob/edge-corrected-even | fail |
| stage 2 | ft | S1 | frozen-replay | yes | yes | 0.000 | -1.73 (1.13) | 70.96 | 72.70 | device-max-prob/logit-sum | fail |
| stage 2 | ft | S2 | second-day | no | no | 0.000 | -0.81 (0.95) | 74.91 | 75.72 | device-max-prob/average-corrected-even | fail |
| stage 2 | ft | S2 | frozen-replay | yes | yes | 0.000 | -1.38 (0.75) | 75.93 | 77.31 | device-max-prob/average-corrected-even | fail |
| stage 2 | fz | S1 | second-day | yes | yes | 0.000 | -1.87 (0.83) | 72.91 | 74.78 | device-max-prob/logit-sum | fail |
| stage 2 | fz | S1 | frozen-replay | yes | yes | 0.000 | -1.96 (0.73) | 72.17 | 74.14 | device-max-prob/logit-sum | fail |
| stage 2 | fz | S2 | second-day | no | no | 0.000 | -1.75 (1.05) | 75.67 | 77.42 | device-max-prob/average-corrected-even | fail |
| stage 2 | fz | S2 | frozen-replay | no | no | 0.000 | -1.77 (1.13) | 75.08 | 76.84 | device-max-prob/average-corrected-even | fail |
| attempt 2 | ft | S1 | second-day | yes | yes | -1.257 | -1.41 (0.39) | 71.48 | 72.89 | own-mahalanobis/logit-sum | fail |
| attempt 2 | ft | S1 | frozen-replay | yes | yes | 0.000 | -1.85 (0.09) | 71.86 | 73.71 | own-mahalanobis/logit-sum | fail |
| attempt 2 | ft | S2 | second-day | no | no | -1.112 | -1.28 (1.10) | 76.98 | 78.26 | own-mahalanobis/probability-average | fail |
| attempt 2 | ft | S2 | frozen-replay | no | no | 0.000 | -1.69 (0.59) | 77.31 | 79.00 | own-mahalanobis/logit-sum | fail |
| attempt 2 | fz | S1 | second-day | no | no | 0.000 | -3.10 (1.27) | 74.43 | 77.53 | device-max-prob/logit-sum | fail |
| attempt 2 | fz | S1 | frozen-replay | yes | yes | 0.000 | -2.77 (0.88) | 74.23 | 77.00 | device-max-prob/logit-sum | fail |
| attempt 2 | fz | S2 | second-day | no | no | 0.000 | -1.93 (1.15) | 77.88 | 79.81 | device-max-prob/average-corrected-even | fail |
| attempt 2 | fz | S2 | frozen-replay | yes | yes | 0.000 | -1.82 (0.93) | 77.71 | 79.53 | device-max-prob/average-corrected-even | fail |
| attempt 3 | ft | S1 | second-day | no | no | -0.458 | -0.67 (0.55) | 68.47 | 69.13 | device-max-prob/edge-corrected-even | fail |
| attempt 3 | ft | S1 | frozen-replay | yes | yes | -1.977 | -0.96 (1.41) | 71.74 | 72.70 | device-max-prob/logit-sum | fail |
| attempt 3 | ft | S2 | second-day | no | no | -0.576 | 0.07 (0.80) | 75.79 | 75.72 | device-max-prob/average-corrected-even | fail |
| attempt 3 | ft | S2 | frozen-replay | yes | yes | -0.722 | -0.36 (0.39) | 76.95 | 77.31 | device-max-prob/average-corrected-even | fail |
| attempt 3 | fz | S1 | second-day | yes | yes | 0.000 | -0.97 (0.56) | 73.81 | 74.78 | device-max-prob/logit-sum | fail |
| attempt 3 | fz | S1 | frozen-replay | yes | yes | 0.000 | -1.20 (0.50) | 72.93 | 74.14 | device-max-prob/logit-sum | fail |
| attempt 3 | fz | S2 | second-day | no | no | -0.370 | -0.70 (0.71) | 76.73 | 77.42 | device-max-prob/average-corrected-even | fail |
| attempt 3 | fz | S2 | frozen-replay | no | no | -0.301 | -0.82 (0.84) | 76.02 | 76.84 | device-max-prob/average-corrected-even | fail |

그림 `figures/R18oc_final_judged_cells.png`는 이 표의 G(0.5)를 점으로 그린 것입니다. 속이 찬 점이 판정 대상 칸이고, 점선은 부분 통과(+1.0 %p)와 통과(+2.0 %p)의 기준입니다.

### 2.3 새 방법이 가장 강한 비교 방법보다 낮은 칸

비율 0.5에서 새 방법이 envelope B(0.5)보다 낮은 칸은 44칸 가운데 43칸입니다. 2단계는 24칸 가운데 24칸, 시도 2번은 12칸 가운데 12칸, 시도 3번은 8칸 가운데 7칸입니다. 시도 2번은 두 사전학습 구성의 S1·S2에서 세 조건을 모두 돌렸으므로 12칸입니다. 시도 3번은 `second-day`와 `frozen-replay`만 계산했으므로 8칸입니다.

그 43칸에서 B(0.5)가 나온 비교 방법은 다음과 같습니다. 43칸 가운데 36칸에서 B(0.5)는 `device-max-prob` 순위의 규칙에서 나왔습니다.

| B(0.5) rule | cells |
|---|---|
| device-max-prob/average-corrected-even | 22 |
| device-max-prob/logit-sum | 10 |
| device-max-prob/edge-corrected-even | 4 |
| own-mahalanobis/logit-sum | 4 |
| device-energy/average-corrected-even | 1 |
| own-knn/average-corrected-even | 1 |
| own-mahalanobis/probability-average | 1 |

G(0.5)가 0 이상인 칸은 하나뿐이고, 그 칸은 여지가 없어서 판정 대상이 아닙니다.

| source | setting | model config | condition | room | G(0.5) pp mean (SD) |
|---|---|---|---|---|---|
| attempt 3 | S2 | pretrained-finetuned-block | second-day | no | 0.07 (0.80) |

### 2.4 고정된 비교 방법 하나(v32 방법)와의 비교

envelope는 비율마다 가장 좋은 비교 방법을 고르므로, 실제로 배치할 수 있는 한 규칙보다 강한 기준입니다(5절). 그래서 고정된 규칙 하나인 v32 방법과도 비교했습니다. seed 평균 곡선의 차이이므로 같은 seed끼리 뺀 차이의 평균과 같습니다.

| setting | model config | condition | ratio 0.25 | ratio 0.5 | ratio 0.75 |
|---|---|---|---|---|---|
| S1 | cnn | first-day | 0.25 | -0.27 | -0.61 |
| S1 | cnn | second-day | 0.05 | -0.25 | -0.12 |
| S1 | cnn | frozen-replay | -0.28 | -0.55 | -0.30 |
| S2 | cnn | first-day | 0.55 | 0.07 | -0.75 |
| S2 | cnn | second-day | -0.15 | -0.41 | -0.50 |
| S2 | cnn | frozen-replay | -0.17 | -0.51 | -0.60 |
| S1 | ft | first-day | -1.41 | -1.75 | -1.90 |
| S1 | ft | second-day | 0.41 | 0.85 | 0.80 |
| S1 | ft | frozen-replay | -1.12 | -1.49 | -2.09 |
| S2 | ft | first-day | -1.25 | -1.49 | -1.74 |
| S2 | ft | second-day | -0.86 | -0.79 | -1.75 |
| S2 | ft | frozen-replay | -1.01 | -1.24 | -1.88 |
| CIFAR-100 | ft | first-day | -4.11 | -3.97 | -2.92 |
| CIFAR-100 | ft | second-day | -1.58 | -1.70 | -1.32 |
| CIFAR-100 | ft | frozen-replay | -1.44 | -1.38 | -1.18 |
| S1 | fz | first-day | 1.12 | 0.47 | -1.01 |
| S1 | fz | second-day | -0.14 | -0.63 | -2.16 |
| S1 | fz | frozen-replay | 0.02 | -0.62 | -2.08 |
| S2 | fz | first-day | 0.37 | -0.43 | -1.19 |
| S2 | fz | second-day | -0.75 | -1.20 | -2.13 |
| S2 | fz | frozen-replay | -0.38 | -1.12 | -1.93 |
| CIFAR-100 | fz | first-day | -3.07 | -3.04 | -1.79 |
| CIFAR-100 | fz | second-day | -1.75 | -1.67 | -1.08 |
| CIFAR-100 | fz | frozen-replay | -1.20 | -1.00 | -0.64 |

비율 0.5에서 새 방법이 v32 방법보다 높은 칸은 24칸 가운데 3칸입니다. 그 세 칸은 S2 `cnn-scratch`의 `first-day`(+0.07 %p), S1 ft의 `second-day`(+0.85 %p), S1 fz의 `first-day`(+0.47 %p)입니다.

### 2.5 개선 시도의 관문 (S1 개발용 seed 8–10, `frozen-replay`, 비율 0.5)

| 시도 | 바꾼 것 | 결정 기록 |
|---|---|---|
| 1번 `exit-with-other-class` | device exit에 "other" 출력을 더하고, edge의 class 통계에서 뽑은 가상 음성 벡터로 학습합니다. | D6, D20, D22 |
| 2번 `deeper-split` | device block을 `features[0]`–`features[10]`(64×4×4, stride 16)으로 늘립니다. | D6, D21, D23, D25, D27 |
| 3번 `average-corrected-by-detector` | 보낸 요청의 답에 device 출력을 v32 방법의 가중치로 섞습니다(분석만 바뀜). | D18, D24 |

| attempt | config | without the change (%) | attempt (%) | difference pp mean (SD) | per seed (8, 9, 10) | AUROC mean (without → attempt) | passes the gate |
|---|---|---|---|---|---|---|---|
| 1 | ft | 72.07 (1.95) | 66.64 (0.37) | -5.43 (2.06) | -7.81, -4.13, -4.36 | 0.761 → 0.683 | no |
| 1 | fz | 71.78 (1.82) | 70.94 (2.10) | -0.85 (0.29) | -0.55, -0.85, -1.13 | 0.865 → 0.851 | no |
| 2 | ft | 72.07 (1.95) | 72.74 (2.46) | 0.67 (0.56) | 1.20, 0.70, 0.09 | 0.761 → 0.804 | yes |
| 2 | fz | 71.78 (1.82) | 74.28 (1.80) | 2.50 (0.06) | 2.46, 2.57, 2.48 | 0.865 → 0.894 | yes |
| 3 | ft | 72.07 (1.95) | 73.16 (2.03) | 1.09 (0.45) | 1.19, 0.59, 1.48 |  | yes |
| 3 | fz | 71.78 (1.82) | 72.77 (1.18) | 0.99 (0.78) | 0.41, 0.67, 1.87 |  | yes |

시도 3번은 판별기를 바꾸지 않으므로 AUROC 칸을 비웠습니다. 관문을 넘은 시도만 평가용 run(seed 0–2)으로 판정했습니다(D12).

### 2.6 시도 2번의 비용 (64×64 이미지 한 장)

| device split | used in | features[] stages on the device | smashed data shape (C x H x W) | device block parameters | device exit parameters | device block MMACs | device exit MMACs | edge parameters | edge MMACs | uplink bytes per offloaded request (float32) | downlink bytes per offloaded request (float32) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stride8 | stage 2 | 7 | 32x8x8 | 55488 | 8458 | 9.25 | 0.395 | 2181194 | 15.22 | 8192 | 40 |
| stride16 | attempt 2 (deeper-split) | 11 | 64x4x4 | 239360 | 14602 | 12.39 | 0.199 | 1997322 | 12.07 | 4096 | 40 |

device block을 늘리면 device 쪽 계산량은 9.25 MMAC에서 12.39 MMAC로 1.34배가 되고, offload한 요청 하나의 uplink는 8,192 byte에서 4,096 byte로 줄어듭니다. MMAC은 백만 번의 곱셈-덧셈 연산입니다.

### 2.7 `first-day`에서 G(0.5)가 −0.5 %p보다 낮은 칸 (7.3절의 추가 조건)

7.3절의 추가 조건에 따라, 2단계의 `first-day` 8칸 가운데 G(0.5)가 −0.5 %p보다 낮은 칸을 적습니다. 8칸 가운데 6칸이 해당합니다.

| setting | model config | G(0.5) pp mean (SD) |
|---|---|---|
| S1 | cnn-scratch | -0.61 (0.17) |
| S1 | pretrained-finetuned-block | -2.05 (0.41) |
| S2 | pretrained-finetuned-block | -1.81 (0.97) |
| CIFAR-100 | pretrained-finetuned-block | -4.25 (0.35) |
| S2 | pretrained-frozen-block | -1.29 (0.67) |
| CIFAR-100 | pretrained-frozen-block | -3.07 (0.21) |

## 3. 지시문 1.2절의 가설별 결과

| 가설 | 결과 | 근거 |
|---|---|---|
| 1. 사전학습 모델에서도 "어느 요청을 보낼지"에 여지가 남는다 | 일부만 맞습니다. 2단계에서 사전학습 구성의 CIFAR-10 `second-day`·`frozen-replay` 여덟 칸 가운데 여지가 있는 칸은 ft의 S1·S2 `frozen-replay`와 fz의 S1 `second-day`·`frozen-replay` 네 칸입니다. CIFAR-100의 여섯 칸에는 여지가 없습니다. `first-day`에서는 fz S2를 뺀 사전학습 구성의 CIFAR-10 세 칸에 여지가 있고, `cnn-scratch`는 여섯 칸 모두 여지가 있습니다. `frozen-replay`에서 oracle이 OOR 요청을 모두 edge로 보내도 그 정확도는 7.6–10.6%입니다. OOR 요청은 전체의 5.9–7.5%이므로, 보낼 요청을 잘 골라 얻는 이득은 대부분 OOP 요청에서 나옵니다. | `tables/R18oc_stage2_cells.csv`의 oracle 열, `R18oc_stage2_diagnosis_kind.csv` |
| 2. edge가 학습한 판별기가 출력 기반 신호보다 훨씬 잘 가려낸다(AUROC 0.9 이상) | fz에서는 맞지만 0.9에는 닿지 않았습니다. CIFAR-10에서 판별기는 0.866–0.898이고, device 출력 점수는 0.547–0.758입니다. ft에서는 판별기(0.698–0.740)와 device 출력 점수(0.588–0.748)가 비슷합니다. CIFAR-100에서는 판별기가 0.63–0.72로 높지만 절대값이 낮습니다. | `tables/R18oc_stage2_auroc.csv`, `report_stage2.md` 5절 |
| 3. 판별기의 검증 통계로 요청 구성 비율을 label 없이 추정한다 | fz에서는 추정 오차가 작습니다(비율 0.5에서 0.065–0.093, CIFAR-10). ft에서는 0.125–0.188입니다. 그러나 이 추정으로 보정한 답(`edge-corrected-by-detector`)은 같은 판별기 순위에서 보정 세기를 고정한 `edge-corrected-even`보다 비율 0.5에서 0.15–0.99 %p 높은 데 그쳤고, device 출력을 섞는 규칙보다 낮았습니다. | `tables/R18oc_stage2_extra.csv`, `R18oc_stage2_diagnosis_rules.csv` |

## 4. 논문의 주장 세 개와 근거

지시문 0.1절이 둔 원래 주장("같은 정확도를 더 적은 offloading으로", "같은 비율에서 더 높은 정확도")은 결과가 뒷받침하지 않습니다. S는 모든 판정 대상 칸에서 0 이하이고, G(0.5)는 2단계와 개선 시도의 44칸 가운데 43칸에서 음수입니다. 아래는 결과가 뒷받침하는 주장입니다.

| 주장 | 뒷받침하는 표 | 뒷받침하는 그림 | 한계 |
|---|---|---|---|
| 1. 고정한 사전학습 feature 위에서는, edge가 cell의 음성 표본으로 학습한 판별기가 device 출력 기반 점수보다 own class가 아닌 요청을 더 잘 가려냅니다(CIFAR-10 AUROC 0.87–0.90 대 0.76 이하). | `tables/R18oc_stage2_auroc.csv` | 없음(표로 제시) | ft에서는 차이가 없습니다. 목표 0.9에는 닿지 않았습니다. |
| 2. 이 순위의 이점은 같은 offloading 비율의 정확도로 이어지지 않습니다. 2단계와 개선 시도의 44칸 가운데 43칸에서 새 방법이 envelope보다 낮고, 고정된 v32 방법보다도 2단계 24칸 가운데 21칸에서 낮습니다. | `tables/R18oc_final_cells.csv`, `R18oc_final_new_vs_v32.csv`, `R18oc_stage2_cells.csv` | `figures/R18oc_final_judged_cells.png`, `figures/R18oc_stage2_*.png` | seed가 3개라서 칸마다 G(0.5)의 SD가 0.05–1.41 %p입니다. |
| 3. 같은 비율의 정확도는 순위보다 보낸 요청의 답 규칙이 좌우합니다. 같은 `device-max-prob` 순위에서 답 규칙만 `edge-corrected-even`에서 `average-corrected-even`으로 바꾸면 `frozen-replay`에서 +1.06…+1.92 %p입니다. 반면 같은 답 규칙에서 판별기 순위와 `device-max-prob` 순위의 차이는 `edge-corrected-even`에서 −0.70…+0.07 %p입니다. device 출력을 섞는 시도 3번은 판정 대상 칸의 격차를 0.76–1.02 %p 줄였습니다. | `tables/R18oc_stage2_diagnosis_rules.csv`, `R18oc_attempt3_{ft,fz}_cells.csv` | `figures/R18oc_final_judged_cells.png` | 시도 3번의 가중치는 v32 방법의 것을 그대로 썼고, 다른 가중치는 시험하지 않았습니다. |

## 5. 판단 기준에 대한 의견 (지시문 9절 3번)

기준은 바꾸지 않았습니다. 다음 두 가지는 기준이 엄격한 쪽으로 치우친다고 봅니다. 두 가지 모두 받아들여도 판정은 바뀌지 않습니다.

첫째, envelope의 선택입니다.

- envelope B(ρ)는 25개 비교 방법 가운데 비율마다 가장 높은 것을 평가용 seed의 결과로 고릅니다. 배치할 때는 한 규칙을 미리 정해야 하므로, envelope는 실제로 배치할 수 있는 어떤 비교 방법보다 강합니다. 고정된 비교 방법 하나와 비교하는 쪽이 논문의 비교에 더 가깝습니다.
- 고정된 v32 방법과 비교해도 새 방법은 2단계 24칸 가운데 21칸에서 낮습니다(2.4절). 2단계 24칸 가운데 23칸에서 B(0.5)는 `device-max-prob` 순위의 규칙에서 나온 값입니다(2.3절).

둘째, G를 재는 비율입니다.

- 판별기 순위가 이점을 가장 크게 낼 수 있는 비율은 Main이 아닌 요청의 비중 근처입니다. 이 비중은 S1에서 34.1%, S2에서 25.9%입니다. request-kind oracle의 곡선도 비율 0.35 근처에서 가장 높고, 그 뒤로는 Main 요청까지 보내야 하므로 떨어집니다(`figures/R18oc_stage2_*.png`). 비율 0.5의 G는 이 지점을 지난 뒤의 값입니다.
- 비율 0.25의 G(0.25)로 보아도 판정 대상 칸은 모두 음수입니다. 가장 높은 값은 시도 2번 ft S1 `second-day`의 −0.58 %p(SD 0.16)입니다. G(0.25)가 양수인 칸은 판정 대상이 아닌 `first-day` 두 칸(fz S1 +0.38 %p, `cnn-scratch` S2 +0.12 %p)뿐입니다.

## 6. 가까운 선행 연구와의 차이

서지는 2026-11-21에 웹에서 확인했습니다. 확인하지 못한 항목은 "확인 필요"로 적었습니다.

| 연구 | 서지 | 내용 | 이번 방법과의 관계 |
|---|---|---|---|
| EARLIN | Nimi, Arefeen, Uddin, Lee. ECML-PKDD 2021, LNCS pp. 635–651, doi 10.1007/978-3-030-86486-6_39, arXiv 2106.13842 | 사전학습 CNN의 얕은 층 특징에서 거리를 재어, 단말에서 분포 밖(OOD) 입력을 가려냅니다. OOD로 판단한 입력은 서버로 보내지 않고 거부합니다. 재학습 없이 분포 안의 자료만 씁니다. | 이번 방법은 자기 클래스가 아닌 요청을 거부하지 않고 edge로 보내 답을 받습니다. "자기 클래스가 아님"의 기준은 device마다 다르고, device가 이동하면 바뀝니다. 판별기는 cell의 음성 표본으로 라운드마다 다시 학습합니다. |
| Hierarchical inference | Moothedath, Champati, Gross, arXiv 2304.00891 (학회 확인 필요); Al-Atat et al. 2023, doi 10.1145/3597062.3597278 (학회 확인 필요); Behera et al. 2024, arXiv 2407.11061 | device의 작은 모델이 틀릴 것 같은 입력만 큰 모델로 넘깁니다. 최대 softmax 확률에 문턱을 두고, 문턱은 온라인으로 학습합니다. | 이번 실험의 `device-max-prob` 비교 방법 계열입니다. 2단계 24칸 가운데 23칸에서 비율 0.5의 envelope는 이 계열의 규칙에서 나왔습니다. |
| FOSTER | Yu, Hong, Wang, Wang, Zhou. ICLR 2023 | 연합학습에서 클래스 조건부 생성기로 특징 공간에 가상의 외부 클래스 표본을 만들고, OOD 검출을 학습합니다. 다른 client의 클래스를 OOD로 씁니다. | 시도 1번(클래스 통계에서 뽑은 가상 음성 벡터)과 cell 음성 표본에 가깝습니다. 이번 방법은 검출 결과를 offloading과 edge 답의 보정에 씁니다. |
| SplitGP | Han, Kim, Choi, Brinton, Moon. IEEE INFOCOM 2023, arXiv 2212.08343 | client 쪽 모델은 주 클래스에 맞추고, server 쪽 모델은 일반화를 맡습니다. 다중 exit와 device 엔트로피 문턱으로 offloading합니다. | `device-entropy` 비교 방법 계열입니다. |
| SplitOMC | "Efficient Split Learning with Overlapping Areas: Handling Distribution Shift in Multi-Cell Networks" (저자와 학회 확인 필요) | 겹치는 cell 영역의 client와 다중 exit를 다룹니다. 작업을 client-preferred, out-of-preference, out-of-region으로 나누고, MNIST와 CIFAR-10/100, Jetson Nano에서 실험합니다. | 이번 실험의 학습 방식과 요청 종류(Main, OOP, OOR)의 바탕입니다. |
| FedRoD | Chen, Chao. ICLR 2022 (spotlight), arXiv 2107.00778 | 균형 손실로 학습한 일반 head와 개인화 head를 함께 둡니다. | device exit(개인화)와 edge(일반)를 함께 쓰는 구조가 비슷합니다. |
| FedTHE, FedTHE+ | Jiang, Lin. ICLR 2023, arXiv 2205.10920 | 요청마다 특징 정렬로 가중치를 정해 일반 head와 개인화 head를 섞고, test-time에 조정합니다. | 시도 3번은 요청마다 다른 가중치로 device 출력과 edge 출력을 섞는다는 점에서 가깝습니다. 이번에는 v32 방법의 가중치를 그대로 씁니다. |

## 7. 모바일 측정 준비 (6.5절 1번)

- 내보낸 모델: ft의 2단계 평가용 run `r18oc_ft_s1_s0`의 라운드 150 checkpoint에서 device 0의 device block, device exit, 판별기, 세 단계를 묶은 device 전체, edge block을 ONNX와 TFLite float32로 내보냈습니다(D26). 파일은 8T 디스크의 `driftgate_r18_own_class_261120/mobile_export/stage4_ft_s1_s0_device0/`에 있고, 기록은 `tables/R18oc_mobile_export_manifest_stage4_ft_s1_s0_device0.json`입니다.
- 일치 확인: 다섯 모델 모두 같은 입력에서 PyTorch 출력과의 최대 절대 차이가 ONNX에서 1.53e−5 이하, TFLite에서 2.15e−5 이하입니다(기준 1e−4).

| 모델 | ONNX에 저장된 값의 수 | ONNX 최대 차이 | TFLite 최대 차이 |
|---|---|---|---|
| device block (`device_block.onnx`) | 53,728 | 8.46e-06 | 1.39e-05 |
| device exit (`device_exit.onnx`) | 8,266 | 1.67e-06 | 7.15e-07 |
| 판별기 (`detector.onnx`) | 12,801 | 1.19e-07 | 2.38e-07 |
| device 전체(세 모델을 묶음) (`device_side.onnx`) | 74,795 | 8.46e-06 | 1.39e-05 |
| edge block (`edge_block.onnx`) | 2,165,898 | 1.53e-05 | 2.15e-05 |

- offload한 요청 하나의 응용 계층 데이터는 uplink 8,192 byte(32×8×8 float32)와 downlink 40 byte(logit 10개)입니다. 프로토콜 헤더는 포함하지 않았습니다.
- 측정 도구: `mobile/README_mobile_ko.md`에 순서가 있습니다. ONNX Runtime 지연 측정(`measure_onnx.py`), Jetson의 tegrastats 에너지 측정, Android `benchmark_model` 로그 변환(`parse_benchmark_model.py`), 배터리 표본 수집(`android_power_sample.sh`)을 준비했습니다. 결과는 `measurement_template.json` 하나에 모읍니다.
- 그림: `figures/R18oc_mobile_assumed_{S1,S2}_pretrained-finetuned-block_frozen-replay.png`는 v32 원고의 가정값(device 11.8 ms, uplink 50 Mbps, 왕복 20 ms, edge 0.6 ms)으로 그린 정확도 대 평균 지연 그림이고, "가정값"이라고 표시했습니다. device 시간 11.8 ms는 v32 CNN의 값이며 MobileNetV2의 측정값이 아닙니다. 에너지는 가정값이 없어서 그리지 않았습니다.

## 8. 남은 위험

1. 3단계(S1·S2 seed 0–4, S1-fast, partial participation, stepwise change, random mobility, 200·500대, iNaturalist, ablation)를 하지 않았습니다. 따라서 이 설정들에서의 결과는 없습니다. iNaturalist 자료(북미 다섯 지역, 100종)는 준비만 했습니다(D7, D8).
2. 평가용 seed는 3개입니다. 판정 대상 칸에서 G(0.5)의 SD는 0.09–1.41 %p이므로, 칸 하나의 판정은 seed에 따라 달라질 수 있습니다. 다만 모든 판정 대상 칸의 G(0.5)가 부분 통과 기준(+1.0 %p)보다 1.36 %p 이상 낮아서, seed를 늘려도 판정이 바뀔 가능성은 낮다고 봅니다.
3. 판별기의 학습 설정(D4)과 조합(D11, D13, D15)은 결과를 보기 전에 개발용 seed에서 정했고 바꾸지 않았습니다. 다른 학습 설정이나 더 큰 판별기는 시험하지 않았습니다.
4. 모바일 지연과 에너지는 측정하지 않았습니다. 그림의 지연은 가정값입니다.
5. 선행 연구 가운데 hierarchical inference 두 편의 학회와 SplitOMC의 저자·학회는 "확인 필요"입니다.

## 9. 사용자가 해야 할 일

1. 모바일 실측: `mobile/README_mobile_ko.md`의 순서대로 갤럭시 스마트폰과 Jetson에서 device 쪽 시간, edge 시간, 요청당 에너지, 네트워크 속도를 재고 `measurement_template.json`을 채웁니다. 측정값을 채우고 `measurement_type`을 `measured`로 바꾼 뒤 `mobile/r18oc_mobile_plot.py`로 그림을 다시 그리면, 그림에서 "가정값" 표시가 빠집니다.
2. 논문 방향: 이 방법으로 TMC 원고를 이어 갈지, letter 규모의 부정적 결과로 정리할지, 이 방향을 접을지를 정합니다. 이 보고서의 권고는 "중단 권고"입니다.
3. 선행 연구의 "확인 필요" 항목(6절)을 확인합니다.
4. 8T 디스크의 `driftgate_r18_own_class_261120/`을 계속 보관할지 정합니다. run 출력, 분석 cache, 내보낸 모델은 파일 959개, 합계 83.1 GB이고, 파일마다 sha256을 `tables/R18oc_files_sha256.csv`에 적었습니다. 그 밖에 3단계를 위해 준비한 iNaturalist 자료가 2.4 GB입니다.

## 10. `decisions.md`의 요약

| 결정 | 내용 |
|---|---|
| D1 | 작업 이름과 위치를 지시문대로 두었습니다(기존 Round 18 `driftgate_tmc_r18_compare`와 다른 폴더와 branch). |
| D2 | 개발용 seed를 S1 seed 8–10으로 바꿨습니다(지시문의 5–7은 이전 라운드에서 개발용으로 썼습니다). |
| D3 | `cnn-scratch`는 기존 run을 재사용하지 않고 새로 돌렸습니다(held-out 분리와 판별기 기록이 없음). |
| D4 | 판별기 학습, cell 음성의 상한, MobileNetV2 입력 처리 등 구현 값을 결과를 보기 전에 정했습니다. |
| D5 | 판단 기준과 학습 코드를 commit한 직후 개발용 run을 띄우고, 분석 스크립트를 commit하기 전에는 결과를 읽지 않았습니다. |
| D6 | 개선 시도 1·2번을 기본값이 꺼진 옵션으로 미리 구현하고 시험했습니다. |
| D7, D8 | iNaturalist 2021 `train`에서 북미 다섯 지역과 100종을 골라 3단계에 대비했습니다(3단계를 하지 않아 쓰지 않았습니다). |
| D9 | 사용자 지시에 따라 run 출력, 분석 cache, 데이터, 내보낸 모델을 8T 디스크에 두고 저장소에는 링크를 두었습니다. |
| D10 | 가짜 cache로 만든 시험용 판정 파일을 지웠습니다. |
| D11, D13, D15 | 세 구성 모두 개발용 seed에서 `exit|mlp64|both`를 골랐습니다(ft와 fz는 0.10 %p 동률 범위에서 단순한 쪽). |
| D12 | 개선 시도는 고른 조합을 그대로 쓰고, 같은 seed끼리 비율 0.5의 정확도를 비교해 +0.5 %p 이상이면 평가용 seed로 넘깁니다. |
| D14 | stepwise change와 random mobility를 Round 6 실행기의 환경 파일로 만들었습니다(3단계를 하지 않아 쓰지 않았습니다). |
| D16 | 개선 시도 1·2번의 개발용 run을 2단계 평가용 run 뒤에 미리 넣었습니다. |
| D17 | 2단계 판정: 두 사전학습 구성 모두 7.3절 표의 행 3이고, 판정 대상은 여지 있는 칸 두 개씩입니다. 시도끼리는 쌓지 않습니다. |
| D18 | 시도 3번의 가설(보낸 요청의 답에서 device 출력을 버리는 것이 주된 원인)과 기대 수치를 결과를 보기 전에 적었습니다. |
| D19 | 시도 1·2번의 여지는 시도의 평가용 run에서 다시 계산합니다. |
| D20, D22 | 시도 1번은 두 구성 모두 관문을 넘지 못했습니다(ft −5.43, fz −0.85 %p). |
| D21, D23 | 시도 2번은 두 구성 모두 관문을 넘었습니다(fz +2.50, ft +0.67 %p). |
| D24 | 시도 3번은 두 구성 모두 관문을 넘었지만(+1.09, +0.99 %p), 판정 대상 네 칸이 모두 미달입니다. |
| D25 | 시도 2번 fz의 판정 대상 칸 두 개가 모두 미달입니다(G(0.5) −2.77, −1.82 %p). |
| D26 | 4단계에서는 ft를 내보냈습니다. |
| D27 | 시도 2번 ft의 판정 대상 칸 두 개(S1 `second-day`, S1 `frozen-replay`)가 모두 미달입니다(G(0.5) −1.41, −1.85 %p). |
| D28 | 최종 판정: 7.3절의 마지막 행에 따라 3단계를 건너뛰고, 판정은 "중단 권고"입니다. |

## 11. 그림과 검토용 묶음

- 그림마다 무엇을 보여 주는지는 `figures/README_ko.md`에 한 문장씩 적었습니다.
- 그림 전부와 이 보고서를 `r18_review_bundle.zip`으로 묶었습니다.
- run은 학습 run과 재생 run을 합쳐 114개입니다. 2단계 run마다 걸린 시간은 `report_stage2.md` 1절에 있습니다. 아래 표는 모든 run을 구성, 설정, 종류, seed 묶음별로 모아 평균과 범위를 적은 것입니다.

표의 시간은 run 하나가 시작부터 끝까지 걸린 시간(wall time)입니다. config tag의 `fta1`, `fta2`, `fza1`, `fza2`는 ft와 fz의 개선 시도 1·2번 run입니다. kind의 training은 라운드 1–300을 학습한 run이고, replay는 라운드 150의 모델을 고정하고 둘째 날을 재생한 run입니다.

| config tag | setting | kind | seeds | runs | mean wall time (h) | range (h) |
|---|---|---|---|---|---|---|
| cnn | s1 | replay | development | 3 | 0.14 | 0.13-0.16 |
| cnn | s1 | replay | evaluation | 3 | 0.14 | 0.13-0.15 |
| cnn | s1 | training | development | 3 | 3.11 | 2.59-3.88 |
| cnn | s1 | training | evaluation | 3 | 4.63 | 4.48-4.78 |
| cnn | s2 | replay | evaluation | 3 | 0.20 | 0.08-0.25 |
| cnn | s2 | training | evaluation | 3 | 4.05 | 2.84-4.68 |
| ft | c100 | replay | evaluation | 3 | 0.42 | 0.40-0.44 |
| ft | c100 | training | evaluation | 3 | 13.10 | 12.97-13.23 |
| ft | s1 | replay | development | 3 | 0.08 | 0.08-0.09 |
| ft | s1 | replay | evaluation | 3 | 0.34 | 0.31-0.36 |
| ft | s1 | training | development | 3 | 6.37 | 5.97-6.58 |
| ft | s1 | training | evaluation | 3 | 12.75 | 12.35-13.22 |
| ft | s2 | replay | evaluation | 3 | 0.31 | 0.28-0.34 |
| ft | s2 | training | evaluation | 3 | 12.69 | 12.32-13.19 |
| fta1 | s1 | replay | development | 3 | 0.28 | 0.17-0.35 |
| fta1 | s1 | training | development | 3 | 10.27 | 5.97-12.55 |
| fta2 | s1 | replay | development | 3 | 0.18 | 0.17-0.19 |
| fta2 | s1 | replay | evaluation | 3 | 0.25 | 0.20-0.30 |
| fta2 | s1 | training | development | 3 | 7.87 | 7.68-7.98 |
| fta2 | s1 | training | evaluation | 3 | 7.95 | 7.77-8.11 |
| fta2 | s2 | replay | evaluation | 3 | 0.19 | 0.13-0.23 |
| fta2 | s2 | training | evaluation | 3 | 7.90 | 7.78-8.08 |
| fz | c100 | replay | evaluation | 3 | 0.40 | 0.38-0.43 |
| fz | c100 | training | evaluation | 3 | 10.57 | 10.44-10.68 |
| fz | s1 | replay | development | 3 | 0.08 | 0.08-0.09 |
| fz | s1 | replay | evaluation | 3 | 0.32 | 0.30-0.37 |
| fz | s1 | training | development | 3 | 6.49 | 5.65-7.00 |
| fz | s1 | training | evaluation | 3 | 10.46 | 10.21-10.62 |
| fz | s2 | replay | evaluation | 3 | 0.29 | 0.27-0.34 |
| fz | s2 | training | evaluation | 3 | 10.32 | 10.18-10.49 |
| fza1 | s1 | replay | development | 3 | 0.18 | 0.18-0.19 |
| fza1 | s1 | training | development | 3 | 6.58 | 5.47-7.15 |
| fza2 | s1 | replay | development | 3 | 0.17 | 0.16-0.18 |
| fza2 | s1 | replay | evaluation | 3 | 0.31 | 0.25-0.34 |
| fza2 | s1 | training | development | 3 | 5.30 | 4.81-6.08 |
| fza2 | s1 | training | evaluation | 3 | 6.25 | 6.05-6.35 |
| fza2 | s2 | replay | evaluation | 3 | 0.27 | 0.23-0.30 |
| fza2 | s2 | training | evaluation | 3 | 6.26 | 6.16-6.38 |
