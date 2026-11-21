# Round 18 2단계 보고서: 예비 실험

이 보고서는 지시문 6.2절의 2단계 결과를 적습니다. 중간 보고서이므로 표와 수치, 구현에서 정한 것, 걸린 시간만 적고, 해석은 `decisions.md`(D17, D18)에 적었습니다. 판정 기준은 결과를 보기 전에 `decision_rule.md`로 commit했습니다(commit `fc1c157`).

표의 수치는 모두 다음 경로로 만들었습니다: run 기록(JSON, npz) → 분석 스크립트 → `tables/`의 CSV → 이 보고서의 표.

| 단계 | 스크립트 | commit |
|---|---|---|
| run마다 조건별 분석 cache | `scripts/r18oc_analysis.py` | `c6f863c` |
| 칸의 판정, 7.3절의 판정, 참고 표 | `scripts/r18oc_tables.py stage2`, `holdout` | `c6f863c` |
| 실패 원인의 진단 표 | `scripts/r18oc_diagnose.py stage2` | `b4884cc` |
| 곡선 그림 | `scripts/r18oc_figures.py curves` | `5839a2e` |
| 이 보고서의 표 | `scripts/r18oc_report.py stage2`, `stage2_summary`, `times_summary` | 이 보고서와 같은 commit |

이 보고서에서 쓰는 약어와 기호는 다음과 같습니다.

- S1, S2: CIFAR-10으로 만든 두 이동 설정입니다. CIFAR-100은 S1의 이동에 CIFAR-100을 쓴 설정입니다.
- Main, OOP, OOR: 요청의 종류입니다. Main은 요청한 device가 학습한 class(own class)의 요청이고, OOP(out-of-place)는 지금 cell의 다른 device가 학습한 class의 요청이며, OOR(out-of-region)은 그 밖의 요청입니다.
- offloading 비율 ρ: 요청 가운데 edge로 보내는 요청의 비율입니다.
- B(ρ): 비교 방법 25개(점수 5개 × 답 규칙 5개)의 seed 평균 곡선에서 비율 ρ마다 가장 높은 값입니다(이하 envelope). A*는 B(1.0) − 0.5 %p입니다.
- S와 G(ρ): S는 1 − (새 방법이 A*에 닿는 비율)/(envelope가 A*에 닿는 비율)이고, 새 방법이 A*에 닿지 못하면 0입니다. G(ρ)는 비율 ρ에서 새 방법의 정확도에서 B(ρ)를 뺀 값(%p, 퍼센트포인트)입니다.
- 괄호 안의 수는 SD(같은 seed끼리 뺀 차이의 표준편차, ddof = 1)입니다. 정확도의 괄호는 seed 사이의 표준편차입니다.
- AUROC(ROC 곡선 아래 면적): Main이 아닌 요청을 양성으로 두고, 점수가 그 요청을 얼마나 잘 가려내는지를 나타냅니다.
- 새 방법은 판별기 순위(g가 작은 요청부터 보냄)와 답 규칙 `edge-corrected-by-detector`의 조합입니다. 판별기 조합 이름은 "입력|모델|음성 표본"입니다.

## 1. run과 걸린 시간

- 평가용 run: `cnn-scratch`는 S1·S2 seed 0–2, 두 사전학습 구성(`pretrained-finetuned-block`, `pretrained-frozen-block`)은 S1·S2·CIFAR-100 seed 0–2입니다. 모두 300라운드이고, 라운드 150의 checkpoint에서 둘째 날을 고정된 모델로 다시 돌린 run(`frozen-replay`)이 하나씩 있습니다.
- 개발용 run: 세 구성 모두 S1 seed 8–10입니다. 설계 선택(2절)에만 썼습니다.
- 조건: `first-day`는 라운드 1–150, `second-day`는 라운드 151–300, `frozen-replay`는 고정 모델로 다시 돌린 라운드 151–300의 요청입니다. 세 조건에서 요청 종류별 개수는 같습니다(예: `r18oc_ft_s1_s0`에서 Main 2,480,000, OOP 1,022,489, OOR 283,441개).
- 걸린 시간(run JSON의 `total_time_sec`): 평가용 run은 GPU 하나에 run을 최대 5개까지 함께 돌렸으므로 개발용 run보다 오래 걸렸습니다.

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
| fz | c100 | replay | evaluation | 3 | 0.40 | 0.38-0.43 |
| fz | c100 | training | evaluation | 3 | 10.57 | 10.44-10.68 |
| fz | s1 | replay | development | 3 | 0.08 | 0.08-0.09 |
| fz | s1 | replay | evaluation | 3 | 0.32 | 0.30-0.37 |
| fz | s1 | training | development | 3 | 6.49 | 5.65-7.00 |
| fz | s1 | training | evaluation | 3 | 10.46 | 10.21-10.62 |
| fz | s2 | replay | evaluation | 3 | 0.29 | 0.27-0.34 |
| fz | s2 | training | evaluation | 3 | 10.32 | 10.18-10.49 |

## 2. 설계 선택 (지시문 3.5, 개발용 seed 8–10)

S1의 `frozen-replay`, 비율 0.5에서 새 방법의 seed 평균 정확도로 18개 조합을 비교했습니다. 전체 표는 `tables/R18oc_design_selection_{cnn,ft,fz}.csv`에 있습니다.

| 모델 구성 | 고른 조합 | 정확도 (SD) | 가장 높은 조합 | 가장 낮은 조합 | 기록 |
|---|---|---|---|---|---|
| `cnn-scratch` | `exit\|mlp64\|both` | 70.85 (2.70) | 같은 조합; 2등 `exit\|mlp64\|cell` 70.66 | `exit\|mlp64\|global` 67.82 | D11 |
| `pretrained-finetuned-block` | `exit\|mlp64\|both` | 72.07 (1.95) | `concat\|mlp64\|both` 72.09 (0.10 %p 동률 범위 안) | `block\|logreg\|global` 70.10 | D13 |
| `pretrained-frozen-block` | `exit\|mlp64\|both` | 71.78 (1.82) | `concat\|mlp64\|both` 71.80 (0.10 %p 동률 범위 안) | `block\|mlp64\|global` 69.47 | D15 |

## 3. 칸의 판정 (7.1, 7.2)

`tables/R18oc_stage2_cells.csv`입니다. `room`은 request-kind oracle(요청 종류를 아는 순위와 `edge-raw`)이 통과 기준을 넘는지를 나타냅니다. `border`는 S나 G(0.5)가 기준값에서 0.01 또는 0.05 %p 안에 있는지를 나타냅니다.

| setting | model config | condition | detector variant | A* (%) | ratio-at-target new | ratio-at-target comparator envelope | S | G(0.25) pp mean (SD) | G(0.5) pp mean (SD) | B(0.5) rule | oracle S | oracle G(0.5) pp | room | judgement | border |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | cnn-scratch | first-day | exit\|mlp64\|both | 68.16 | 0.962 | 0.558 | -0.723 | -0.31 (0.09) | -0.61 (0.17) | device-max-prob/average-corrected-even | 0.722 | 2.73 | yes | fail | no |
| S1 | cnn-scratch | second-day | exit\|mlp64\|both | 72.83 | 0.871 | 0.627 | -0.389 | -0.43 (0.04) | -0.69 (0.05) | device-max-prob/edge-corrected-even | 0.616 | 2.39 | yes | fail | no |
| S1 | cnn-scratch | frozen-replay | exit\|mlp64\|both | 71.86 | 0.927 | 0.646 | -0.436 | -0.74 (0.10) | -0.91 (0.19) | device-max-prob/edge-corrected-even | 0.658 | 2.97 | yes | fail | no |
| S2 | cnn-scratch | first-day | exit\|mlp64\|both | 70.77 | none | 0.474 | 0.000 | 0.12 (0.43) | -0.20 (0.51) | device-max-prob/average-corrected-even | 0.719 | -0.36 | yes | fail | no |
| S2 | cnn-scratch | second-day | exit\|mlp64\|both | 75.95 | 0.909 | 0.463 | -0.963 | -0.53 (0.27) | -0.54 (0.30) | device-max-prob/average-corrected-even | 0.497 | 0.01 | yes | fail | no |
| S2 | cnn-scratch | frozen-replay | exit\|mlp64\|both | 75.62 | 0.799 | 0.477 | -0.677 | -0.63 (0.03) | -0.67 (0.15) | device-max-prob/average-corrected-even | 0.593 | 0.54 | yes | fail | no |
| S1 | pretrained-finetuned-block | first-day | exit\|mlp64\|both | 72.36 | none | 0.450 | 0.000 | -1.75 (0.43) | -2.05 (0.41) | device-max-prob/average-corrected-even | 0.549 | -0.46 | yes | fail | no |
| S1 | pretrained-finetuned-block | second-day | exit\|mlp64\|both | 69.01 | 0.484 | 0.476 | -0.017 | -0.36 (0.88) | -0.02 (0.84) | device-max-prob/edge-corrected-even | 0.000 | -2.38 | no | fail | no |
| S1 | pretrained-finetuned-block | frozen-replay | exit\|mlp64\|both | 71.98 | none | 0.228 | 0.000 | -1.50 (0.54) | -1.73 (1.13) | device-max-prob/logit-sum | 0.508 | -0.03 | yes | fail | no |
| S2 | pretrained-finetuned-block | first-day | exit\|mlp64\|both | 74.97 | none | 0.393 | 0.000 | -1.66 (0.57) | -1.81 (0.97) | device-max-prob/average-corrected-even | 0.408 | -3.88 | yes | fail | no |
| S2 | pretrained-finetuned-block | second-day | exit\|mlp64\|both | 75.21 | none | 0.223 | 0.000 | -1.21 (0.43) | -0.81 (0.95) | device-max-prob/average-corrected-even | -0.016 | -4.20 | no | fail | no |
| S2 | pretrained-finetuned-block | frozen-replay | exit\|mlp64\|both | 76.85 | none | 0.272 | 0.000 | -1.41 (0.44) | -1.38 (0.75) | device-max-prob/average-corrected-even | 0.414 | -3.90 | yes | fail | no |
| CIFAR-100 | pretrained-finetuned-block | first-day | exit\|mlp64\|both | 62.85 | 0.991 | 0.932 | -0.063 | -4.18 (0.33) | -4.25 (0.35) | device-max-prob/average-corrected-even | 0.000 | -6.46 | no | fail | no |
| CIFAR-100 | pretrained-finetuned-block | second-day | exit\|mlp64\|both | 66.67 | none | 0.826 | 0.000 | -2.17 (0.29) | -2.63 (0.36) | device-max-prob/average-corrected-even | 0.000 | -3.68 | no | fail | no |
| CIFAR-100 | pretrained-finetuned-block | frozen-replay | exit\|mlp64\|both | 67.04 | 0.996 | 0.869 | -0.146 | -2.00 (0.14) | -2.38 (0.29) | device-max-prob/average-corrected-even | 0.000 | -3.15 | no | fail | no |
| S1 | pretrained-frozen-block | first-day | exit\|mlp64\|both | 72.64 | none | 0.678 | 0.000 | 0.38 (0.34) | -0.31 (0.25) | device-max-prob/logit-sum | 0.552 | -0.48 | yes | fail | no |
| S1 | pretrained-frozen-block | second-day | exit\|mlp64\|both | 74.70 | none | 0.483 | 0.000 | -1.14 (0.57) | -1.87 (0.83) | device-max-prob/logit-sum | 0.411 | -2.62 | yes | fail | no |
| S1 | pretrained-frozen-block | frozen-replay | exit\|mlp64\|both | 74.20 | none | 0.514 | 0.000 | -1.04 (0.33) | -1.96 (0.73) | device-max-prob/logit-sum | 0.485 | -1.96 | yes | fail | no |
| S2 | pretrained-frozen-block | first-day | exit\|mlp64\|both | 74.74 | none | 0.675 | 0.000 | -0.50 (0.56) | -1.29 (0.67) | device-max-prob/average-corrected-even | 0.000 | -3.53 | no | fail | no |
| S2 | pretrained-frozen-block | second-day | exit\|mlp64\|both | 77.26 | none | 0.457 | 0.000 | -1.39 (0.65) | -1.75 (1.05) | device-max-prob/average-corrected-even | 0.000 | -4.84 | no | fail | no |
| S2 | pretrained-frozen-block | frozen-replay | exit\|mlp64\|both | 77.09 | none | 0.564 | 0.000 | -1.12 (0.58) | -1.77 (1.13) | device-max-prob/average-corrected-even | 0.000 | -4.14 | no | fail | no |
| CIFAR-100 | pretrained-frozen-block | first-day | exit\|mlp64\|both | 63.61 | 0.991 | 0.972 | -0.019 | -3.35 (0.27) | -3.07 (0.21) | device-energy/average-corrected-even | 0.000 | -5.35 | no | fail | no |
| CIFAR-100 | pretrained-frozen-block | second-day | exit\|mlp64\|both | 67.52 | 0.988 | 0.925 | -0.067 | -2.17 (0.22) | -2.59 (0.21) | device-max-prob/average-corrected-even | 0.000 | -3.96 | no | fail | no |
| CIFAR-100 | pretrained-frozen-block | frozen-replay | exit\|mlp64\|both | 67.59 | 0.987 | 0.939 | -0.051 | -1.65 (0.24) | -1.98 (0.17) | device-max-prob/average-corrected-even | 0.000 | -3.53 | no | fail | no |

비율 0.25, 0.5, 0.75에서 새 방법과 envelope의 정확도(%)입니다.

| setting | model config | condition | new accuracy at 0.25 | new accuracy at 0.5 | new accuracy at 0.75 | B(0.25) | B(0.5) | B(0.75) | B(1.0) rule |
|---|---|---|---|---|---|---|---|---|---|
| S1 | cnn-scratch | first-day | 66.34 | 67.37 | 67.86 | 66.66 | 67.98 | 68.54 | device-entropy/average-corrected-even |
| S1 | cnn-scratch | second-day | 70.14 | 71.68 | 72.43 | 70.58 | 72.38 | 73.10 | device-entropy/edge-corrected-even |
| S1 | cnn-scratch | frozen-replay | 68.77 | 70.39 | 71.29 | 69.51 | 71.30 | 72.13 | device-entropy/edge-corrected-even |
| S2 | cnn-scratch | first-day | 69.93 | 70.64 | 70.42 | 69.81 | 70.84 | 71.21 | device-entropy/average-corrected-even |
| S2 | cnn-scratch | second-day | 74.31 | 75.52 | 75.87 | 74.84 | 76.06 | 76.38 | device-entropy/edge-corrected-even |
| S2 | cnn-scratch | frozen-replay | 73.76 | 75.03 | 75.45 | 74.39 | 75.69 | 76.06 | device-entropy/average-corrected-even |
| S1 | pretrained-finetuned-block | first-day | 69.19 | 70.50 | 70.93 | 70.94 | 72.55 | 72.89 | device-entropy/average-corrected-even |
| S1 | pretrained-finetuned-block | second-day | 66.60 | 69.11 | 69.91 | 66.96 | 69.13 | 69.68 | device-entropy/edge-corrected-even |
| S1 | pretrained-finetuned-block | frozen-replay | 70.63 | 70.96 | 70.46 | 72.13 | 72.70 | 72.77 | device-entropy/logit-sum |
| S2 | pretrained-finetuned-block | first-day | 72.32 | 73.50 | 73.70 | 73.98 | 75.30 | 75.61 | device-entropy/average-corrected-even |
| S2 | pretrained-finetuned-block | second-day | 74.16 | 74.91 | 73.99 | 75.37 | 75.72 | 75.74 | device-entropy/average-corrected-even |
| S2 | pretrained-finetuned-block | frozen-replay | 75.36 | 75.93 | 75.50 | 76.77 | 77.31 | 77.44 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-finetuned-block | first-day | 36.71 | 46.07 | 55.16 | 40.89 | 50.32 | 58.62 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-finetuned-block | second-day | 53.15 | 58.93 | 64.04 | 55.32 | 61.56 | 65.93 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-finetuned-block | frozen-replay | 49.98 | 57.14 | 63.42 | 51.98 | 59.52 | 65.41 | device-entropy/average-corrected-even |
| S1 | pretrained-frozen-block | first-day | 68.74 | 71.18 | 71.42 | 68.36 | 71.49 | 72.87 | device-entropy/logit-sum |
| S1 | pretrained-frozen-block | second-day | 71.65 | 72.91 | 72.23 | 72.79 | 74.78 | 75.20 | device-entropy/logit-sum |
| S1 | pretrained-frozen-block | frozen-replay | 70.69 | 72.17 | 71.65 | 71.74 | 74.14 | 74.69 | device-entropy/logit-sum |
| S2 | pretrained-frozen-block | first-day | 70.00 | 72.31 | 73.47 | 70.51 | 73.60 | 75.00 | device-entropy/average-corrected-even |
| S2 | pretrained-frozen-block | second-day | 74.33 | 75.67 | 75.53 | 75.72 | 77.42 | 77.75 | device-entropy/average-corrected-even |
| S2 | pretrained-frozen-block | frozen-replay | 73.64 | 75.08 | 75.35 | 74.75 | 76.84 | 77.48 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-frozen-block | first-day | 30.46 | 41.84 | 53.31 | 33.81 | 44.90 | 55.70 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-frozen-block | second-day | 43.89 | 53.04 | 61.80 | 46.05 | 55.63 | 63.74 | device-entropy/average-corrected-even |
| CIFAR-100 | pretrained-frozen-block | frozen-replay | 41.17 | 51.39 | 61.01 | 42.81 | 53.37 | 62.54 | device-entropy/average-corrected-even |

## 4. 7.3절의 판정

`decision_stage2.json`입니다. 판정 칸은 사전학습 구성마다 S1·S2의 `second-day`와 `frozen-replay`입니다.

| 모델 구성 | S1 second-day | S1 frozen-replay | S2 second-day | S2 frozen-replay | 여지 없는 칸 | 해당 행 |
|---|---|---|---|---|---|---|
| `pretrained-finetuned-block` | 미달 (여지 없음) | 미달 | 미달 (여지 없음) | 미달 | 2 | 3 |
| `pretrained-frozen-block` | 미달 | 미달 | 미달 (여지 없음) | 미달 (여지 없음) | 2 | 3 |

- 두 구성 모두 2.3절 표의 행 3("미달인 칸이 있지만 여지는 있음")에 해당합니다. 따라서 7.4절의 개선 시도를 1번부터 차례로 두 구성 모두에 합니다(D12, D17).
- 일부 칸이 여지 없음이므로, 판정 대상은 여지 있는 칸 두 개씩이고 최종 판정은 "조건부 권고" 이하입니다.
- `first-day`에서 G(0.5)가 −0.5 %p보다 낮은 칸은 S1 `cnn-scratch`, S1·S2·CIFAR-100 `pretrained-finetuned-block`, S2·CIFAR-100 `pretrained-frozen-block`의 6칸입니다.
- `cnn-scratch`의 여섯 칸도 모두 미달입니다.

## 5. AUROC (모든 요청)

`tables/R18oc_stage2_auroc.csv`의 seed 평균입니다. 집과 밖으로 나눈 값과 SD는 CSV에 있습니다.

| setting | model config | condition | exit|mlp64|both | device-entropy | device-max-prob | device-energy | own-mahalanobis | own-knn |
|---|---|---|---|---|---|---|---|---|
| S1 | cnn-scratch | first-day | 0.681 | 0.645 | 0.670 | 0.609 | 0.664 | 0.725 |
| S1 | cnn-scratch | second-day | 0.737 | 0.713 | 0.723 | 0.678 | 0.666 | 0.745 |
| S1 | cnn-scratch | frozen-replay | 0.703 | 0.690 | 0.697 | 0.665 | 0.667 | 0.747 |
| S2 | cnn-scratch | first-day | 0.652 | 0.605 | 0.631 | 0.571 | 0.649 | 0.702 |
| S2 | cnn-scratch | second-day | 0.722 | 0.697 | 0.708 | 0.665 | 0.650 | 0.729 |
| S2 | cnn-scratch | frozen-replay | 0.708 | 0.707 | 0.717 | 0.679 | 0.652 | 0.736 |
| S1 | pretrained-finetuned-block | first-day | 0.736 | 0.660 | 0.703 | 0.597 | 0.707 | 0.683 |
| S1 | pretrained-finetuned-block | second-day | 0.698 | 0.672 | 0.688 | 0.630 | 0.733 | 0.697 |
| S1 | pretrained-finetuned-block | frozen-replay | 0.740 | 0.728 | 0.748 | 0.676 | 0.725 | 0.709 |
| S2 | pretrained-finetuned-block | first-day | 0.714 | 0.632 | 0.659 | 0.588 | 0.721 | 0.699 |
| S2 | pretrained-finetuned-block | second-day | 0.733 | 0.698 | 0.712 | 0.657 | 0.746 | 0.728 |
| S2 | pretrained-finetuned-block | frozen-replay | 0.740 | 0.716 | 0.734 | 0.680 | 0.737 | 0.744 |
| CIFAR-100 | pretrained-finetuned-block | first-day | 0.628 | 0.477 | 0.502 | 0.470 | 0.584 | 0.538 |
| CIFAR-100 | pretrained-finetuned-block | second-day | 0.689 | 0.575 | 0.603 | 0.539 | 0.630 | 0.649 |
| CIFAR-100 | pretrained-finetuned-block | frozen-replay | 0.671 | 0.567 | 0.598 | 0.540 | 0.611 | 0.629 |
| S1 | pretrained-frozen-block | first-day | 0.894 | 0.645 | 0.682 | 0.582 | 0.659 | 0.675 |
| S1 | pretrained-frozen-block | second-day | 0.898 | 0.736 | 0.758 | 0.668 | 0.659 | 0.675 |
| S1 | pretrained-frozen-block | frozen-replay | 0.872 | 0.726 | 0.741 | 0.662 | 0.659 | 0.675 |
| S2 | pretrained-frozen-block | first-day | 0.866 | 0.598 | 0.619 | 0.547 | 0.672 | 0.688 |
| S2 | pretrained-frozen-block | second-day | 0.871 | 0.703 | 0.709 | 0.653 | 0.672 | 0.688 |
| S2 | pretrained-frozen-block | frozen-replay | 0.868 | 0.694 | 0.699 | 0.655 | 0.672 | 0.688 |
| CIFAR-100 | pretrained-frozen-block | first-day | 0.709 | 0.486 | 0.515 | 0.481 | 0.560 | 0.559 |
| CIFAR-100 | pretrained-frozen-block | second-day | 0.717 | 0.532 | 0.564 | 0.508 | 0.560 | 0.559 |
| CIFAR-100 | pretrained-frozen-block | frozen-replay | 0.697 | 0.535 | 0.560 | 0.520 | 0.560 | 0.559 |

## 6. 참고 행

`tables/R18oc_stage2_reference.csv`입니다. Confidence-based offloading은 device entropy가 0.8 nats보다 큰 요청을 보내고 `edge-raw`로 답합니다. 마지막 열은 v32의 방법(device entropy의 예산 controller, β = 0.5, `average-corrected-even`)입니다.

| setting | model config | condition | Device only | Edge only | Confidence-based offloading (ratio) | v32 method, entropy controller 0.5 (ratio) |
|---|---|---|---|---|---|---|
| S1 | cnn-scratch | first-day | 63.94 | 62.86 | 62.83 (0.927) | 67.98 (0.499) |
| S1 | cnn-scratch | second-day | 66.72 | 72.93 | 72.58 (0.741) | 71.71 (0.500) |
| S1 | cnn-scratch | frozen-replay | 65.26 | 70.63 | 70.40 (0.785) | 70.68 (0.500) |
| S2 | cnn-scratch | first-day | 68.15 | 64.71 | 64.67 (0.856) | 71.36 (0.499) |
| S2 | cnn-scratch | second-day | 71.36 | 75.23 | 74.97 (0.664) | 75.95 (0.500) |
| S2 | cnn-scratch | frozen-replay | 70.89 | 74.19 | 74.01 (0.749) | 75.61 (0.500) |
| S1 | pretrained-finetuned-block | first-day | 67.36 | 60.73 | 61.46 (0.904) | 71.90 (0.499) |
| S1 | pretrained-finetuned-block | second-day | 62.93 | 59.20 | 61.03 (0.722) | 68.32 (0.500) |
| S1 | pretrained-finetuned-block | frozen-replay | 69.33 | 57.91 | 61.25 (0.760) | 72.35 (0.500) |
| S2 | pretrained-finetuned-block | first-day | 70.87 | 63.08 | 64.27 (0.843) | 74.71 (0.498) |
| S2 | pretrained-finetuned-block | second-day | 72.72 | 63.14 | 65.86 (0.655) | 75.68 (0.500) |
| S2 | pretrained-finetuned-block | frozen-replay | 74.30 | 62.99 | 66.15 (0.715) | 77.21 (0.500) |
| CIFAR-100 | pretrained-finetuned-block | first-day | 27.51 | 57.85 | 57.85 (0.999) | 47.69 (0.498) |
| CIFAR-100 | pretrained-finetuned-block | second-day | 47.53 | 62.65 | 62.69 (0.976) | 60.59 (0.500) |
| CIFAR-100 | pretrained-finetuned-block | frozen-replay | 42.88 | 62.87 | 62.89 (0.994) | 58.58 (0.500) |
| S1 | pretrained-frozen-block | first-day | 63.42 | 62.36 | 62.71 (0.968) | 70.53 (0.499) |
| S1 | pretrained-frozen-block | second-day | 68.76 | 60.57 | 62.26 (0.884) | 73.66 (0.500) |
| S1 | pretrained-frozen-block | frozen-replay | 66.94 | 59.34 | 61.13 (0.903) | 72.86 (0.500) |
| S2 | pretrained-frozen-block | first-day | 65.97 | 65.47 | 65.92 (0.947) | 72.65 (0.498) |
| S2 | pretrained-frozen-block | second-day | 71.82 | 65.08 | 66.69 (0.838) | 76.92 (0.500) |
| S2 | pretrained-frozen-block | frozen-replay | 70.39 | 64.97 | 66.31 (0.890) | 76.40 (0.500) |
| CIFAR-100 | pretrained-frozen-block | first-day | 19.98 | 58.54 | 58.54 (1.000) | 43.39 (0.498) |
| CIFAR-100 | pretrained-frozen-block | second-day | 35.30 | 63.60 | 63.61 (0.997) | 54.58 (0.500) |
| CIFAR-100 | pretrained-frozen-block | frozen-replay | 31.44 | 63.42 | 63.41 (1.000) | 52.45 (0.500) |

## 7. 고른 조합의 추가 지표와 갱신 일정

`tables/R18oc_stage2_extra.csv`와 `R18oc_stage2_curves.csv`입니다. 앞의 세 열은 비율 0.5에서 새 방법의 정확도이며, 판별기를 매 라운드 갱신한 판, 집에 있을 때만 갱신하고 밖에서는 고정한 판(`frozen-at-home`), 10라운드마다 갱신한 판(`every-10`)입니다. 고정 문턱 0.5와 예산 controller(β = 0.25, 0.5)는 정확도 뒤의 괄호에 실제 offloading 비율을 적었습니다. 마지막 열은 비율 0.5에서 ŝ(직전 128개 요청 가운데 Main이 아닌 요청의 비율의 추정)의 평균 절대 오차입니다.

| setting | model config | condition | curve 0.5 (every round) | curve 0.5 (frozen-at-home) | curve 0.5 (every-10) | fixed threshold 0.5 | controller 0.25 | controller 0.5 | s-hat error at curve 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | cnn-scratch | first-day | 67.368 | 66.715 | 63.896 | 67.19 (0.441) | 66.26 (0.254) | 67.30 (0.500) | 0.195 |
| S1 | cnn-scratch | second-day | 71.685 | 71.871 | 70.533 | 71.05 (0.367) | 69.98 (0.254) | 71.48 (0.500) | 0.166 |
| S1 | cnn-scratch | frozen-replay | 70.390 | 70.390 | 68.589 | 69.46 (0.332) | 68.52 (0.254) | 70.19 (0.500) | 0.180 |
| S2 | cnn-scratch | first-day | 70.645 | 70.472 | 66.655 | 70.53 (0.414) | 69.93 (0.254) | 70.77 (0.500) | 0.155 |
| S2 | cnn-scratch | second-day | 75.516 | 75.454 | 74.342 | 74.86 (0.333) | 74.17 (0.254) | 75.41 (0.500) | 0.130 |
| S2 | cnn-scratch | frozen-replay | 75.027 | 74.695 | 73.365 | 74.65 (0.403) | 73.70 (0.254) | 75.11 (0.500) | 0.138 |
| S1 | pretrained-finetuned-block | first-day | 70.503 | 69.272 | 67.707 | 70.63 (0.539) | 68.98 (0.254) | 70.18 (0.500) | 0.154 |
| S1 | pretrained-finetuned-block | second-day | 69.108 | 67.446 | 65.235 | 69.26 (0.508) | 65.73 (0.253) | 68.07 (0.500) | 0.188 |
| S1 | pretrained-finetuned-block | frozen-replay | 70.964 | 70.964 | 68.148 | 70.99 (0.424) | 70.50 (0.254) | 70.80 (0.499) | 0.165 |
| S2 | pretrained-finetuned-block | first-day | 73.495 | 73.228 | 71.133 | 73.54 (0.513) | 72.16 (0.254) | 73.27 (0.499) | 0.129 |
| S2 | pretrained-finetuned-block | second-day | 74.910 | 74.793 | 72.099 | 74.83 (0.417) | 74.16 (0.254) | 74.89 (0.499) | 0.130 |
| S2 | pretrained-finetuned-block | frozen-replay | 75.932 | 75.891 | 73.875 | 75.87 (0.468) | 75.31 (0.253) | 75.92 (0.499) | 0.125 |
| CIFAR-100 | pretrained-finetuned-block | first-day | 46.072 | 43.211 | 43.647 | 51.26 (0.641) | 36.92 (0.254) | 46.18 (0.500) | 0.187 |
| CIFAR-100 | pretrained-finetuned-block | second-day | 58.927 | 58.135 | 57.120 | 60.98 (0.594) | 53.53 (0.254) | 59.15 (0.500) | 0.162 |
| CIFAR-100 | pretrained-finetuned-block | frozen-replay | 57.140 | 57.140 | 55.240 | 59.33 (0.585) | 50.22 (0.254) | 57.22 (0.500) | 0.156 |
| S1 | pretrained-frozen-block | first-day | 71.185 | 70.409 | 71.162 | 71.26 (0.520) | 68.21 (0.254) | 71.32 (0.500) | 0.065 |
| S1 | pretrained-frozen-block | second-day | 72.913 | 72.499 | 72.878 | 72.90 (0.513) | 71.29 (0.254) | 73.00 (0.500) | 0.065 |
| S1 | pretrained-frozen-block | frozen-replay | 72.173 | 72.173 | 72.313 | 72.12 (0.474) | 70.33 (0.254) | 72.27 (0.500) | 0.074 |
| S2 | pretrained-frozen-block | first-day | 72.308 | 71.994 | 72.318 | 72.33 (0.501) | 69.85 (0.254) | 72.54 (0.500) | 0.090 |
| S2 | pretrained-frozen-block | second-day | 75.673 | 75.533 | 75.673 | 75.67 (0.491) | 74.18 (0.254) | 75.62 (0.500) | 0.090 |
| S2 | pretrained-frozen-block | frozen-replay | 75.077 | 74.823 | 75.328 | 75.12 (0.512) | 73.51 (0.254) | 75.33 (0.500) | 0.093 |
| CIFAR-100 | pretrained-frozen-block | first-day | 41.837 | 39.348 | 41.667 | 50.01 (0.678) | 31.26 (0.254) | 42.48 (0.500) | 0.137 |
| CIFAR-100 | pretrained-frozen-block | second-day | 53.044 | 52.127 | 52.930 | 58.39 (0.650) | 44.65 (0.254) | 53.63 (0.500) | 0.130 |
| CIFAR-100 | pretrained-frozen-block | frozen-replay | 51.392 | 51.392 | 51.056 | 56.04 (0.617) | 41.74 (0.254) | 51.67 (0.500) | 0.135 |

## 8. held-out 10%의 영향 (`cnn-scratch`, `first-day`)

`tables/R18oc_holdout_effect.csv`입니다. 이번 run은 device마다 학습 자료의 10%를 판별기의 검증용으로 떼어 두었고, v32의 기록은 떼어 두지 않았습니다.

| setting | rule | held-out 10% (this round, seeds 0-2) mean (SD) | v32 records (seeds 0-2) mean (SD) | difference pp mean (SD) |
|---|---|---|---|---|
| S1 | Device only | 63.94 (0.84) | 64.33 (0.43) | -0.39 (0.57) |
| S1 | Confidence-based offloading | 62.83 (1.05) | 63.23 (0.36) | -0.40 (1.38) |
| S1 | Raw edge only (ratio 1) | 62.86 (1.03) | 63.27 (0.37) | -0.40 (1.38) |
| S1 | Probability average (ratio 1) | 67.71 (0.16) | 67.95 (0.87) | -0.24 (1.02) |
| S1 | Logit sum (ratio 1) | 67.77 (0.21) | 68.02 (0.81) | -0.25 (0.96) |
| S1 | Corrected edge only (ratio 1) | 68.33 (0.28) | 68.54 (0.67) | -0.21 (0.91) |
| S1 | v32 method (ratio 1) | 68.66 (0.12) | 68.86 (0.63) | -0.20 (0.74) |
| S2 | Device only | 68.15 (2.57) | 69.04 (2.80) | -0.89 (0.28) |
| S2 | Confidence-based offloading | 64.67 (4.26) | 65.18 (3.73) | -0.51 (0.53) |
| S2 | Raw edge only (ratio 1) | 64.71 (4.29) | 65.24 (3.76) | -0.53 (0.54) |
| S2 | Probability average (ratio 1) | 69.13 (2.57) | 69.88 (2.02) | -0.76 (0.58) |
| S2 | Logit sum (ratio 1) | 69.32 (2.53) | 70.07 (1.97) | -0.75 (0.60) |
| S2 | Corrected edge only (ratio 1) | 69.98 (1.14) | 70.49 (0.82) | -0.51 (0.35) |
| S2 | v32 method (ratio 1) | 71.27 (1.40) | 71.70 (0.97) | -0.43 (0.44) |

## 9. 실패 원인의 진단 (7.4의 3번을 위한 자료)

`scripts/r18oc_diagnose.py`로 만든 `tables/R18oc_stage2_diagnosis_{rules,kind,r}.csv`입니다. 2단계 평가용 cache만 읽었습니다. 해석과 그에 따른 개선 시도 3번의 가설은 `decisions.md`의 D18에 있습니다.

비율 0.5에서 순위와 답 규칙의 조합별 정확도(%)입니다. `detector`는 고른 조합의 순위이고, `oracle`은 request-kind oracle의 순위입니다. 열 이름의 ft와 fz는 각각 `pretrained-finetuned-block`과 `pretrained-frozen-block`입니다.

| ranking | answer rule | S1 ft second-day | S1 ft frozen-replay | S2 ft second-day | S2 ft frozen-replay | S1 fz second-day | S1 fz frozen-replay | S2 fz second-day | S2 fz frozen-replay |
|---|---|---|---|---|---|---|---|---|---|
| detector | edge-corrected-by-detector | 69.11 | 70.96 | 74.91 | 75.93 | 72.91 | 72.17 | 75.67 | 75.08 |
| detector | edge-raw | 64.66 | 67.03 | 69.74 | 71.44 | 70.72 | 69.92 | 72.48 | 72.18 |
| detector | edge-corrected-even | 68.91 | 70.73 | 73.92 | 75.35 | 72.10 | 71.74 | 75.49 | 74.93 |
| detector | edge-corrected-by-detector + device 0.2 | 69.33 | 71.43 | 75.33 | 76.39 | 73.62 | 72.75 | 76.18 | 75.60 |
| device-max-prob | edge-raw | 64.27 | 67.24 | 68.80 | 70.62 | 69.39 | 68.82 | 71.71 | 71.60 |
| device-max-prob | edge-corrected-even | 69.13 | 71.31 | 73.85 | 75.39 | 72.63 | 72.23 | 75.84 | 75.63 |
| device-max-prob | average-corrected-even | 68.92 | 72.54 | 75.72 | 77.31 | 73.99 | 73.29 | 77.42 | 76.84 |
| device-max-prob | logit-sum | 67.10 | 72.70 | 73.44 | 75.84 | 74.78 | 74.14 | 76.68 | 76.24 |
| oracle | edge-raw | 66.75 | 72.67 | 71.52 | 73.41 | 72.16 | 72.18 | 72.58 | 72.70 |
| oracle | edge-corrected-even | 65.95 | 72.21 | 72.96 | 74.79 | 71.51 | 71.70 | 73.97 | 73.74 |
| envelope B | max over the 25 comparators | 69.13 | 72.70 | 75.72 | 77.31 | 74.78 | 74.14 | 77.42 | 76.84 |

`frozen-replay`에서 요청 종류별 정확도(%, Main / OOP / OOR)입니다. 각 값은 라운드마다 device별 정확도를 평균한 뒤 라운드에 걸쳐 평균한 값입니다.

| setting | model config | ranking | answer rule | ratio 0.25 | ratio 0.5 | ratio 0.75 |
|---|---|---|---|---|---|---|
| S1 | pretrained-finetuned-block | detector | edge-corrected-by-detector | 90.03 / 22.63 / 3.52 | 89.72 / 21.70 / 2.17 | 89.33 / 20.04 / 1.81 |
| S1 | pretrained-finetuned-block | detector | edge-raw | 85.97 / 31.83 / 5.29 | 78.02 / 42.45 / 6.03 | 68.44 / 49.55 / 7.15 |
| S1 | pretrained-finetuned-block | detector | edge-corrected-even | 89.91 / 25.37 / 3.57 | 88.19 / 31.20 / 3.10 | 85.66 / 35.14 / 3.40 |
| S1 | pretrained-finetuned-block | detector | edge-corrected-by-detector + device 0.2 | 90.47 / 22.76 / 3.76 | 90.77 / 21.57 / 2.35 | 91.04 / 19.66 / 1.94 |
| S1 | pretrained-finetuned-block | device-max-prob | edge-raw | 86.74 / 34.18 / 6.40 | 77.72 / 45.39 / 6.80 | 66.35 / 51.48 / 7.48 |
| S1 | pretrained-finetuned-block | device-max-prob | edge-corrected-even | 90.88 / 28.07 / 4.38 | 88.39 / 33.75 / 3.51 | 84.76 / 36.48 / 3.49 |
| S1 | pretrained-finetuned-block | device-max-prob | average-corrected-even | 91.79 / 26.85 / 4.83 | 91.86 / 29.19 / 4.35 | 91.86 / 29.09 / 4.36 |
| S1 | pretrained-finetuned-block | device-max-prob | logit-sum | 89.61 / 32.59 / 7.76 | 87.55 / 39.79 / 8.39 | 86.07 / 41.90 / 8.52 |
| S1 | pretrained-finetuned-block | oracle | edge-raw | 90.22 / 44.48 / 7.68 | 82.67 / 53.32 / 7.63 | 70.85 / 53.32 / 7.63 |
| S1 | pretrained-finetuned-block | oracle | edge-corrected-even | 90.22 / 32.59 / 4.66 | 88.49 / 37.10 / 3.49 | 85.77 / 37.10 / 3.49 |
| S2 | pretrained-finetuned-block | detector | edge-corrected-by-detector | 89.09 / 28.18 / 3.61 | 89.96 / 24.85 / 2.89 | 89.06 / 24.97 / 2.88 |
| S2 | pretrained-finetuned-block | detector | edge-raw | 85.30 / 40.49 / 5.18 | 80.14 / 48.78 / 6.43 | 73.42 / 54.14 / 7.65 |
| S2 | pretrained-finetuned-block | detector | edge-corrected-even | 88.74 / 32.94 / 3.97 | 88.08 / 36.48 / 4.16 | 86.25 / 39.22 / 4.57 |
| S2 | pretrained-finetuned-block | detector | edge-corrected-by-detector + device 0.2 | 89.32 / 29.02 / 3.66 | 90.53 / 25.63 / 2.94 | 90.27 / 25.59 / 2.90 |
| S2 | pretrained-finetuned-block | device-max-prob | edge-raw | 85.41 / 44.70 / 5.65 | 78.40 / 52.46 / 7.19 | 70.73 / 55.74 / 7.82 |
| S2 | pretrained-finetuned-block | device-max-prob | edge-corrected-even | 89.41 / 36.62 / 3.98 | 87.57 / 39.45 / 4.45 | 85.29 / 40.37 / 4.70 |
| S2 | pretrained-finetuned-block | device-max-prob | average-corrected-even | 90.08 / 36.52 / 4.31 | 90.34 / 38.34 / 4.36 | 90.38 / 38.45 / 4.38 |
| S2 | pretrained-finetuned-block | device-max-prob | logit-sum | 87.59 / 43.94 / 6.92 | 85.79 / 49.37 / 7.75 | 85.07 / 50.64 / 7.89 |
| S2 | pretrained-finetuned-block | oracle | edge-raw | 87.73 / 55.57 / 8.02 | 80.99 / 56.42 / 8.11 | 73.99 / 56.42 / 8.11 |
| S2 | pretrained-finetuned-block | oracle | edge-corrected-even | 87.73 / 40.24 / 4.79 | 86.55 / 40.51 / 4.76 | 85.32 / 40.51 / 4.76 |
| S1 | pretrained-frozen-block | detector | edge-corrected-by-detector | 85.73 / 36.77 / 5.85 | 85.25 / 36.19 / 4.85 | 85.26 / 32.34 / 3.88 |
| S1 | pretrained-frozen-block | detector | edge-raw | 85.61 / 40.27 / 6.17 | 79.35 / 53.20 / 8.92 | 70.16 / 57.15 / 10.12 |
| S1 | pretrained-frozen-block | detector | edge-corrected-even | 87.55 / 31.70 / 3.93 | 87.29 / 39.75 / 4.76 | 85.65 / 42.04 / 5.16 |
| S1 | pretrained-frozen-block | detector | edge-corrected-by-detector + device 0.2 | 86.09 / 36.70 / 6.07 | 86.57 / 35.50 / 4.95 | 87.47 / 31.40 / 3.86 |
| S1 | pretrained-frozen-block | device-max-prob | edge-raw | 85.13 / 36.09 / 7.96 | 79.16 / 47.53 / 8.93 | 69.51 / 55.30 / 9.95 |
| S1 | pretrained-frozen-block | device-max-prob | edge-corrected-even | 88.95 / 30.16 / 5.58 | 88.60 / 36.93 / 5.17 | 85.84 / 41.19 / 5.11 |
| S1 | pretrained-frozen-block | device-max-prob | average-corrected-even | 89.78 / 29.02 / 5.59 | 91.36 / 33.64 / 5.09 | 91.63 / 35.00 / 5.11 |
| S1 | pretrained-frozen-block | device-max-prob | logit-sum | 88.49 / 34.14 / 8.34 | 88.78 / 42.50 / 9.01 | 87.87 / 46.41 / 9.41 |
| S1 | pretrained-frozen-block | oracle | edge-raw | 87.04 / 47.87 / 9.63 | 80.40 / 58.31 / 10.59 | 69.99 / 58.31 / 10.59 |
| S1 | pretrained-frozen-block | oracle | edge-corrected-even | 87.04 / 36.33 / 5.79 | 86.07 / 42.61 / 5.33 | 84.53 / 42.61 / 5.33 |
| S2 | pretrained-frozen-block | detector | edge-corrected-by-detector | 84.17 / 39.31 / 5.62 | 85.29 / 37.52 / 4.96 | 85.72 / 36.40 / 4.66 |
| S2 | pretrained-frozen-block | detector | edge-raw | 82.75 / 48.25 / 7.07 | 78.96 / 59.66 / 9.40 | 73.57 / 62.88 / 10.17 |
| S2 | pretrained-frozen-block | detector | edge-corrected-even | 85.02 / 39.01 / 4.80 | 85.73 / 46.06 / 5.92 | 85.10 / 48.06 / 6.40 |
| S2 | pretrained-frozen-block | detector | edge-corrected-by-detector + device 0.2 | 84.45 / 39.69 / 5.56 | 86.06 / 37.68 / 4.91 | 87.04 / 36.34 / 4.54 |
| S2 | pretrained-frozen-block | device-max-prob | edge-raw | 82.78 / 46.01 / 7.19 | 78.63 / 57.02 / 8.52 | 73.07 / 62.13 / 9.77 |
| S2 | pretrained-frozen-block | device-max-prob | edge-corrected-even | 86.40 / 39.12 / 5.38 | 86.56 / 45.87 / 5.88 | 85.25 / 48.13 / 6.29 |
| S2 | pretrained-frozen-block | device-max-prob | average-corrected-even | 87.05 / 38.30 / 4.61 | 88.62 / 43.41 / 4.72 | 89.10 / 44.65 / 4.77 |
| S2 | pretrained-frozen-block | device-max-prob | logit-sum | 85.49 / 44.01 / 6.43 | 85.61 / 52.27 / 7.34 | 85.30 / 55.18 / 7.66 |
| S2 | pretrained-frozen-block | oracle | edge-raw | 83.68 / 62.48 / 10.36 | 78.55 / 63.79 / 10.56 | 73.22 / 63.79 / 10.56 |
| S2 | pretrained-frozen-block | oracle | edge-corrected-even | 83.68 / 47.82 / 6.49 | 83.60 / 48.59 / 6.56 | 83.51 / 48.59 / 6.56 |

요청 종류의 비율과, 비율 0.5에서 보낸 요청에 쓰인 r(다른 class 쪽의 사전 확률)의 평균입니다. 마지막 열은 고정 문턱 0.5에서의 offloading 비율입니다.

| setting | model config | condition | detector variant | Main share (%) | OOP share (%) | OOR share (%) | mean r on offloaded requests at 0.5 | s-hat error at 0.5 | offloading ratio at fixed threshold 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | pretrained-finetuned-block | first-day | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.396 | 0.154 | 0.539 |
| S1 | pretrained-finetuned-block | second-day | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.362 | 0.188 | 0.508 |
| S1 | pretrained-finetuned-block | frozen-replay | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.373 | 0.165 | 0.424 |
| S2 | pretrained-finetuned-block | first-day | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.336 | 0.129 | 0.513 |
| S2 | pretrained-finetuned-block | second-day | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.342 | 0.130 | 0.417 |
| S2 | pretrained-finetuned-block | frozen-replay | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.342 | 0.125 | 0.468 |
| S1 | pretrained-frozen-block | first-day | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.612 | 0.065 | 0.520 |
| S1 | pretrained-frozen-block | second-day | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.616 | 0.065 | 0.513 |
| S1 | pretrained-frozen-block | frozen-replay | exit\|mlp64\|both | 66.0 | 26.6 | 7.5 | 0.586 | 0.074 | 0.474 |
| S2 | pretrained-frozen-block | first-day | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.492 | 0.090 | 0.501 |
| S2 | pretrained-frozen-block | second-day | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.496 | 0.090 | 0.491 |
| S2 | pretrained-frozen-block | frozen-replay | exit\|mlp64\|both | 74.1 | 20.0 | 5.9 | 0.484 | 0.093 | 0.512 |

## 10. 그림

`figures/R18oc_stage2_<설정>_<모델 구성>.png`의 8장입니다. 그림마다 세 조건을 나란히 두고, 새 방법(고른 조합), envelope B, v32의 방법(`device-entropy` + `average-corrected-even`), request-kind oracle(`edge-raw`, 점선)의 seed 평균 곡선을 그렸습니다. 그림은 `tables/R18oc_stage2_curves.csv`만 읽어 그렸습니다.

## 11. 개선 시도의 상태 (이 보고서를 쓴 2026-11-21 16:22 기준)

| 시도 | 내용 | 상태 |
|---|---|---|
| 1번 `exit-with-other-class` | device exit에 "other" 출력을 더하고 가상 음성 벡터로 학습 | 개발용 run 6개(두 구성 × seed 8–10)가 도는 중입니다(D16). |
| 2번 `deeper-split` | device block을 `features[0]`–`features[10]`으로 늘림 | 개발용 run 6개가 도는 중입니다(D16). |
| 3번 `average-corrected-by-detector` | 보낸 요청의 답에 device 출력을 섞음(분석만 바뀜) | 가설과 기대 수치를 D18(commit `54e7429`)에, 코드를 `scripts/r18oc_attempt3.py`(commit `5b441d3`)에 두었습니다. 1번과 2번의 관문을 계산한 뒤에 계산합니다. |

관문은 D12와 같습니다. S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 같은 seed끼리 비교한 평균 정확도가 0.5 %p 이상 오르면 평가용 seed로 넘어갑니다.

## 12. 구현에서 정한 것

- 개발용 run은 주 조합을 고르기 전에 돌았으므로 cache에 주 조합의 표시가 없습니다. 개선 시도의 관문에서는 `tables/R18oc_design_selection_*.json`에 적힌 고른 조합을 씁니다.
- 요청 종류별 정확도의 분모(요청 종류별 개수)는 run 기록에서 한 번 세어 `cache/<run>__<조건>__nkind.npy`에 두었습니다(`r18oc_stage3.n_kind`).
- 분석 cache와 run 출력은 `/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120/`에 있고, 저장소에는 그 폴더를 가리키는 링크만 있습니다(D9).
