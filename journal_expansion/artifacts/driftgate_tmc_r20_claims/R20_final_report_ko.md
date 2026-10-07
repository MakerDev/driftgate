# Round 20 최종 보고서: 기존 결과로 확정한 제출 주장

## 이 보고서의 범위

**수행한 작업과 자료**
- Round 20 지시문의 기본 작업인 Task 0–3을 수행했습니다.
- 새 학습과 새 추론은 하지 않았습니다. 이미 저장된 요청별 출력(device exit 확률, raw edge exit 확률, label, 요청 순서, cell 소속, M_k, a_k)에서 규칙의 답을 다시 계산했습니다.
- 다시 계산한 값은 Round 15, 16, 18, 19의 값과 모두 정확히 같았습니다(`tables/R20_checks.csv`, 53개 run).

**수행하지 않은 작업**
- **Task 4 (offloading의 정답 이득을 예측하는 선택적 탐색):** 기본 실행에서 비활성화되어 있으므로 하지 않았습니다.
- **Task 5 (모바일 실측):** 장비가 없어 하지 않았고, 미확인으로 남깁니다.

**정의를 고정한 시점**
- 계획(`R20_plan.md`)은 계산 전에 commit했습니다(`88d9b4b`).
- 평가에 쓴 seed와 요청은 Round 15–19에서 이미 결과를 본 탐색 자료입니다. 따라서 이 보고서의 수치를 독립 검증 결과로 부르지 않습니다.

**표기**
- **정확도:** 원고의 지표(이하 macro 정확도)입니다. 평가 round마다 요청이 있는 device의 정확도를 평균한 뒤, 그 값을 round에 대해 평균합니다.
- **차이:** 같은 seed와 같은 요청에서 뺀 값의 seed 평균(표준편차)이며, 단위는 pp입니다. 통계적 동등성이나 비열등성은 선언하지 않습니다.
- **β:** offload할 요청 비율의 목표입니다. Round 15의 controller는 device마다 이전 128개 요청의 device exit entropy 분포에서 (1 − β) 분위수를 threshold로 씁니다. 같은 β에서는 모든 규칙이 같은 offload mask를 씁니다.
- **규칙 이름:** 영어 그대로 씁니다.
  - "Correction"은 원고 식 (correct)의 prior correction(r = 0.5)입니다.
  - "w"는 offload된 요청에서 device exit에 주는 가중치입니다.
  - 표의 C+w는 correction + w를, CE는 corrected edge only를 줄여 쓴 것입니다.

## 1. 권장 최종 기여와 근거

권장하는 기여는 세 가지입니다.

### 1.1 문제 측정 (기존 원고 유지)

- mobility에서 device exit와 edge exit는 서로 다른 요청에 강합니다.
- device exit의 confidence는 이 두 종류의 요청을 약하게만 구분합니다. entropy의 AUROC(수신자 조작 특성 곡선 아래 면적)는 0.64입니다.
- Round 20은 이 측정을 바꾸지 않았습니다.

### 1.2 방법

DriftGate는 label 없이 동작하는 추론 규칙입니다. 동작은 세 단계로 이루어집니다.
1. edge exit를 cell의 class 통계로 보정합니다.
2. 두 exit의 확률을 entropy 기반 가중치로 평균합니다.
3. device exit가 불확실한 요청만 offload합니다.

보정은 own class의 logit에 b_k = log((1 − a_k)/a_k)를 더하는 계산과 같습니다(r = 0.5). S1에서 b_k의 중앙값은 1.15이고, a_k가 0.25일 때 b_k는 log 3 ≈ 1.10입니다.

### 1.3 평가에서 확정된 사실 (원고의 주장으로 쓸 수 있는 범위)

1. **보정한 결합의 효과**
   - 같은 offload mask에서 DriftGate는 보정하지 않은 결합 규칙보다 높습니다. 첫날 설정 10개 가운데 7개에서 β = 0.5와 1의 차이는 +0.60 ~ +1.88 pp입니다.
   - 나머지 세 설정에서 이 차이는 다음과 같습니다(`R20_reconciliations.md` 3.1절).
     - ResNet-18: +0.18 pp
     - K=200: +0.07 ~ −0.00 pp
     - K=500: −0.25 ~ −0.49 pp
2. **이득은 주로 보정에서 나옵니다**
   - 같은 보정에 고정 w를 쓴 규칙(w 0.45 포함)과 DriftGate의 차이는, 첫날 β = 1의 주요 8개 설정 가운데 7개에서 ±0.15 pp 이내입니다. 원고 ablation 표의 변형만으로 비교하면 최대 차이는 0.152 pp입니다(S1, correction + w 0.2).
   - 예외는 CIFAR-100입니다. 이 설정에서는 correction + product가 0.45 pp 높습니다.
   - 같은 mask에서 β를 0.25에서 1까지 바꿔도, 첫날 CNN 설정 6개에서 두 곡선의 차이는 −0.07 ~ +0.14 pp입니다. 이 6개 설정은 S1, S2, S1-fast, partial participation, stepwise change, random mobility입니다.
   - 따라서 entropy 가중치는 모델마다 조정하지 않고 쓰는 기본값으로 설명합니다. 독립적인 정확도 우위는 주장하지 않습니다.
3. **offloading에서 생기는 비용 이득**
   - DriftGate는 원고 주표의 full-offload 최고값에 β = 0.25 또는 0.5에서 도달합니다. 도달하는 설정은 S1, S2, S1-fast, partial participation, stepwise change, random mobility, ResNet-18이고, S2 replay에서도 β = 0.5에서 도달합니다.
   - 보정한 고정 w 규칙도 같은 β에서 도달합니다. 따라서 이 비용 이득은 보정에서 나온 것이며, entropy 가중치에서 나온 것이 아닙니다.
   - 보정한 기준 규칙들도 budget을 조절할 수 있게 하면, DriftGate에만 남는 비용 이득은 거의 없습니다(3.3절).
4. **학습된 모델에서는 결과가 달라집니다**
   - round > 30에서는 corrected edge only가 6개 설정에서 DriftGate보다 0.48–1.31 pp 높습니다.
   - frozen replay에서는 S1에서 0.79 pp, S2에서 0.05 pp 높습니다.
   - 세 날 frozen replay에서는 S1에서 0.75 pp, S2에서 0.10 pp 높습니다.
   - 본문이나 표에서 독자가 이 결과를 확인할 수 있어야 합니다.
5. **a_k의 역할:** 보정에 현재 cell의 a_k 대신 상수를 써도 정확도는 낮아지지 않습니다(4절). 따라서 "cell 변화에 맞춰 보정하므로 좋아진다"는 설명은 쓰지 않습니다.
6. **시스템 측면의 기여**
   - 같은 budget에서 보정하지 않은 결합보다 정확합니다.
   - 추가 계산은 서버 CPU에서 요청당 3.2 μs로 측정된 기존 값만 있습니다.
   - 모바일의 지연과 에너지는 측정하지 않았습니다.

**원고 요약 문장 제안 (영문)**

> DriftGate corrects the edge exit for the class mix of its cell, averages the class probabilities of both exits with a weight set from their recent entropies, and offloads only uncertain requests. Over the first day, it is 0.6 to 1.9 points more accurate than combinations without the correction at the same offloading ratio in seven of ten settings. Most of this gain comes from the correction: with the same correction, a fixed weight is within 0.16 points of DriftGate in seven of eight settings, and with trained models the corrected edge exit alone is often more accurate.

## 2. 닫힌 질문과 미확인 질문

| 질문 | 상태 | 근거 |
|---|---|---|
| 원고, Round 18, Round 17–19의 비교 집합이 각각 무엇을 포함하는가 | 닫힘 | `tables/R20_comparator_sets.csv`, `R20_reconciliations.md` 1절 |
| 같은 이름의 수치가 다른 이유(K=500 70.38/70.60, seed 집합, 재학습 run, 7개 설정의 범위, S1 분해, 43–86% 감소의 분모) | 닫힘 | `R20_reconciliations.md` 2–3절 |
| 기존 결과로 만든 대표 정확도 표(첫날, round > 30, frozen replay, 세 날 replay) | 닫힘 | 3.1절 |
| 같은 budget에서 보정한 규칙과 DriftGate의 차이 | 닫힘(β 격자 0.25 간격 안에서) | 3.2절 |
| 각 방법이 budget을 조절할 때 같은 정확도를 얻는 비용 | S1은 development로 정한 운영점으로 닫힘. 나머지 설정은 사후 기술로만 답함 | 3.3절 |
| 보정에 현재 a_k가 필요한가 | S1, S2와 두 replay에서 닫힘. S2의 상수는 development 통계가 없어 a = 0.25를 쓴 진단값 | 4절 |
| 요청 수로 가중한 정확도에서 순위가 바뀌는가와 그 원인 | 닫힘(away 정보가 있는 설정) | 3.1절 |
| 모바일의 지연, 에너지, 실제 전송 byte | **미확인** (Task 5, 장비 없음) | |
| β 격자(0.25 간격)보다 촘촘한 운영점에서의 비용 차이 | **미확인** (보간값만 있음) | 3.3절 |
| S1을 뺀 설정에서 development로 정한 운영점의 비용 | **미확인** (development 기록 없음) | |
| offloading의 정답 이득을 직접 예측하는 gate(Task 4) | **미수행** (선택 작업, 비활성) | |
| 모든 요청별 선택법이 실패하는가 | **결론 내리지 않음**: 시험한 후보(Round 16–18)만 실패했습니다 | |

## 3. 대표 표

### 3.1 정확도 (β = 1, 모든 요청 offload)

아래 두 표의 값은 정확도(%)이고, 괄호 안은 DriftGate − 그 규칙(pp)입니다.
- confidence-based offloading은 0.8 nats threshold에서 자기 offload 비율을 씁니다.
- 마지막 줄은 원고 주표의 11개 규칙 가운데 DriftGate를 뺀 10개 중에서 가장 높은 규칙입니다.
- 원본은 `tables/R20_report_tables.md`, `tables/R20_T1_accuracy.csv`이고, seed별 값은 `tables/R20_T1_accuracy_per_seed.csv`에 있습니다.

**첫날**

| rule | S1 | S2 | S1-fast | partial part. | stepwise | random mob. | CIFAR-100 | ResNet-18 | K=200 | K=500 |
|---|---|---|---|---|---|---|---|---|---|---|
| Confidence-based offloading | 62.87 (+5.45) | 65.18 (+6.52) | 63.33 (+5.63) | 56.45 (+8.37) | 65.17 (+7.18) | 58.56 (+12.53) | 35.84 (+5.45) | 59.21 (+7.43) | 65.26 (+3.73) | 67.03 (+3.08) |
| Device only | 64.16 (+4.16) | 69.04 (+2.66) | 64.39 (+4.57) | 61.75 (+3.06) | 68.93 (+3.43) | 67.71 (+3.39) | 33.93 (+7.36) | 66.39 (+0.25) | 62.63 (+6.37) | 63.12 (+7.00) |
| Raw edge only | 62.91 (+5.41) | 65.24 (+6.46) | 63.38 (+5.59) | 56.44 (+8.38) | 65.20 (+7.15) | 58.61 (+12.48) | 35.84 (+5.45) | 57.04 (+9.60) | 65.30 (+3.69) | 67.07 (+3.04) |
| Probability average | 67.32 (+0.99) | 69.88 (+1.81) | 68.06 (+0.90) | 62.64 (+2.18) | 71.44 (+0.92) | 69.58 (+1.52) | 38.15 (+3.14) | 66.39 (+0.25) | 68.90 (+0.10) | 70.38 (−0.27) |
| Logit sum | 67.37 (+0.95) | 70.07 (+1.63) | 68.12 (+0.84) | 62.94 (+1.88) | 71.39 (+0.97) | 69.23 (+1.87) | 39.83 (+1.46) | 66.00 (+0.64) | 68.86 (+0.13) | 70.29 (−0.18) |
| No correction + adaptive w | 67.11 (+1.20) | 69.29 (+2.41) | 67.84 (+1.12) | 62.06 (+2.76) | 71.25 (+1.10) | 69.33 (+1.77) | 37.50 (+3.79) | 66.46 (+0.17) | 69.00 (−0.00) | 70.60 (−0.49) |
| Correction + w 0.5 | 68.13 (+0.19) | 71.77 (−0.07) | 68.74 (+0.23) | 64.85 (−0.03) | 72.21 (+0.14) | 70.96 (+0.14) | 41.45 (−0.16) | 66.71 (−0.07) | 68.58 (+0.41) | 69.62 (+0.49) |
| Correction + w 0.2 (development, 원고) | 68.16 (+0.15) | 71.03 (+0.67) | 68.93 (+0.03) | 64.22 (+0.60) | 71.78 (+0.58) | 70.51 (+0.58) | 40.95 (+0.34) | 64.02 (+2.62) | 69.23 (−0.23) | 70.53 (−0.41) |
| Correction + w 0.45 (development, Round 19) | 68.21 (+0.11) | 71.68 (+0.02) | 68.86 (+0.10) | 64.80 (+0.02) | 72.22 (+0.14) | 70.98 (+0.12) | 41.36 (−0.07) | 66.49 (+0.15) | 68.80 (+0.19) | 69.90 (+0.21) |
| Corrected edge only | 67.87 (+0.45) | 70.49 (+1.21) | 68.66 (+0.31) | 63.64 (+1.18) | 71.24 (+1.12) | 69.89 (+1.21) | 40.69 (+0.59) | 61.72 (+4.92) | 69.15 (−0.16) | 70.53 (−0.42) |
| Correction + product | 67.90 (+0.41) | 71.76 (−0.06) | 68.48 (+0.49) | 64.83 (−0.01) | 72.08 (+0.27) | 70.78 (+0.31) | 41.73 (−0.45) | 66.59 (+0.05) | 68.15 (+0.84) | 69.11 (+1.00) |
| **DriftGate** | 68.31 | 71.70 | 68.96 | 64.82 | 72.36 | 71.09 | 41.29 | 66.64 | 69.00 | 70.11 |
| 원고 주표의 최고 비교 규칙 | Logit sum | Logit sum | Logit sum | Logit sum | Prob. average | Prob. average | Label-shift EM (40.44, +0.85) | Logit-entropy w. (66.54, +0.10) | Prob. average | Prob. average |

**학습된 모델 (round > 30, frozen replay, 세 날 frozen replay)**

| rule | round>30 S1 | round>30 S2 | round>30 random mob. | round>30 K=500 | replay S1 | replay S2 | 세 날 S1 | 세 날 S2 |
|---|---|---|---|---|---|---|---|---|
| Confidence-based offloading | 64.80 (+2.31) | 68.04 (+3.92) | 61.70 (+3.69) | 69.62 (−0.34) | 71.47 (+0.48) | 74.76 (+1.77) | 71.35 (+0.58) | 74.58 (+1.74) |
| Device only | 61.77 (+5.33) | 68.33 (+3.62) | 60.50 (+4.90) | 60.90 (+8.38) | 66.38 (+5.57) | 71.12 (+5.41) | 66.44 (+5.49) | 70.62 (+5.71) |
| Raw edge only | 64.85 (+2.25) | 68.11 (+3.85) | 61.77 (+3.63) | 69.67 (−0.39) | 71.74 (+0.20) | 74.94 (+1.60) | 71.64 (+0.29) | 74.76 (+1.57) |
| Probability average | 66.74 (+0.36) | 70.72 (+1.24) | 65.29 (+0.11) | 70.29 (−1.02) | 72.27 (−0.33) | 75.90 (+0.64) | 72.19 (−0.27) | 75.69 (+0.64) |
| Logit sum | 66.68 (+0.42) | 70.82 (+1.14) | 65.06 (+0.33) | 70.06 (−0.78) | 72.08 (−0.13) | 75.86 (+0.68) | 72.04 (−0.12) | 75.64 (+0.69) |
| No correction + adaptive w | 66.71 (+0.40) | 70.21 (+1.75) | 65.28 (+0.12) | 70.71 (−1.44) | 72.46 (−0.52) | 75.88 (+0.65) | 72.36 (−0.44) | 75.66 (+0.67) |
| Correction + w 0.5 | 66.81 (+0.29) | 71.97 (−0.01) | 65.15 (+0.25) | 68.63 (+0.65) | 71.53 (+0.41) | 76.23 (+0.31) | 71.52 (+0.40) | 75.99 (+0.34) |
| Correction + w 0.2 | 67.55 (−0.45) | 71.70 (+0.26) | 65.93 (−0.53) | 70.19 (−0.92) | 72.58 (−0.63) | 76.66 (−0.13) | 72.52 (−0.60) | 76.50 (−0.17) |
| Correction + w 0.45 | 67.02 (+0.08) | 71.97 (−0.02) | 65.37 (+0.03) | 69.03 (+0.25) | 71.83 (+0.12) | 76.40 (+0.14) | 71.81 (+0.12) | 76.18 (+0.15) |
| Corrected edge only | 67.66 (−0.55) | 71.42 (+0.53) | 66.05 (−0.65) | 70.59 (−1.31) | 72.74 (−0.79) | 76.59 (−0.05) | 72.67 (−0.74) | 76.43 (−0.10) |
| Correction + product | 66.48 (+0.63) | 71.82 (+0.14) | 64.78 (+0.61) | 67.95 (+1.32) | 71.12 (+0.83) | 76.00 (+0.54) | 71.13 (+0.79) | 75.71 (+0.62) |
| **DriftGate** | 67.10 | 71.96 | 65.40 | 69.28 | 71.95 | 76.54 | 71.93 | 76.33 |

**표를 읽는 방법**
- **첫날:** DriftGate는 원고 주표의 10개 비교 규칙보다 S1부터 K=200까지 높고, K=500에서만 probability average보다 0.27 pp 낮습니다. 그러나 보정한 변형과의 차이는 대부분 ±0.15 pp 이내입니다.
  - DriftGate보다 높은 보정 변형이 있는 설정은 S2(w 0.5, product), partial participation(w 0.5, product), CIFAR-100(w 0.5, w 0.45, product), ResNet-18(w 0.5), K=200, K=500입니다.
- **학습된 모델:** corrected edge only나 correction + w 0.2가 가장 높은 경우가 많습니다.
- **추가 행:** 부록용 행(lower-entropy exit, logit-entropy weighting, geometric ensemble, label-shift EM, learned weight, correction + learned weight)과 사후 참조 행(평가 seed로 고른 최고 고정 w)은 `tables/R20_T1_accuracy.csv`에 있습니다.
- **사후 참조 행의 성격:** 사후 참조 행은 평가 label로 고른 값이므로 배포할 수 있는 규칙이 아닙니다.

**요청 수로 가중한 정확도 (보조 지표, `tables/R20_T1_request_weighted.csv`)**

이 지표로 주 지표를 바꾸지 않습니다.

| 설정 | 첫날 β = 1에서 DriftGate의 순위 (macro → 요청 가중) | DriftGate보다 높아지는 규칙 (요청 가중 차이) |
|---|---|---|
| S1 | 1위 → 3위 (17개 규칙) | correction + w 0.2 (+0.16 pp), corrected edge only (+0.06 pp) |
| S1-fast | 1위 → 3위 | correction + w 0.2 (+0.28 pp), corrected edge only (+0.22 pp) |
| S2 | 3위 → 2위 | |
| partial participation | 3위 → 1위 | |
| K=200 | 4위 → 7위 | |
| S1 replay | 7위 → 12위 | |

**순위가 바뀌는 원인**
- 요청 수로 가중하면 away(집이 아닌 cell에 있는) device-round의 비중이 커집니다.
  - S1 첫날에서 away device-round는 전체 device-round의 41.6%이지만, 요청은 56.3%를 차지합니다.
  - away device-round 하나는 home device-round보다 요청을 1.81배 많이 받습니다(`tables/R20_T1_away_weight.csv`). away에서는 OOP와 OOR 요청이 Main 요청 하나마다 ρ = 0.8과 0.24개씩 더해지기 때문입니다.
- DriftGate는 away 요청에서 edge 쪽 규칙보다 약합니다. 그래서 요청 가중 지표에서는 edge 비중이 큰 규칙(corrected edge only, w 0.2)이 앞섭니다.
- stepwise change, random mobility, ResNet-18에는 home과 away 정보가 없어 원인 진단을 하지 않았습니다.

### 3.2 같은 budget (같은 offload mask)

아래 표의 값은 두 가지 차이(pp)입니다. 원본은 `tables/R20_T2_same_budget_gap.csv`와 `tables/R20_T2_cost_curves.csv`입니다.
- **보정한 고정 w 규칙 대비:** DriftGate − 보정한 고정 w 규칙 중 최고. 비교 후보는 w 0.5, 0.2, 0.45, corrected edge only, product이고, 괄호 안이 그 규칙입니다.
- **보정 없는 규칙 대비:** DriftGate − 보정 없는 규칙 중 최고. 비교 후보는 raw edge only, probability average, logit sum, no correction + adaptive w입니다.

| setting | β 0.25 | β 0.5 | β 0.75 | β 1 | 보정 없는 규칙 대비 (0.25 / 0.5 / 0.75 / 1) | 요청당 edge 호출 (0.25 / 0.5 / 0.75 / 1) | 두 cell 비율 |
|---|---|---|---|---|---|---|---|
| S1 | +0.03 (w 0.2) | +0.05 (w 0.2) | +0.09 (w 0.2) | +0.11 (w 0.45) | +0.38 / +0.66 / +0.84 / +0.95 | 0.287 / 0.565 / 0.844 / 1.132 | 0.132 |
| S2 | +0.01 (w 0.5) | +0.01 (w 0.5) | −0.03 (w 0.5) | −0.07 (w 0.5) | +0.64 / +1.18 / +1.50 / +1.63 | 0.268 / 0.528 / 0.788 / 1.058 | 0.058 |
| S1-fast | −0.01 (w 0.2) | −0.03 (w 0.2) | −0.02 (w 0.2) | +0.03 (w 0.2) | +0.35 / +0.60 / +0.74 / +0.84 | 0.279 / 0.550 / 0.821 / 1.101 | 0.101 |
| partial participation | +0.02 (w 0.5) | +0.02 (w 0.5) | +0.00 (w 0.5) | −0.03 (w 0.5) | +0.57 / +1.11 / +1.54 / +1.88 | 0.281 / 0.552 / 0.825 / 1.107 | 0.107 |
| stepwise change | +0.04 (w 0.45) | +0.09 (w 0.45) | +0.12 (w 0.45) | +0.14 (w 0.45) | +0.44 / +0.69 / +0.84 / +0.92 | 0.354 / 0.697 / 1.042 / 1.400 | 0.400 |
| random mobility | +0.04 (w 0.45) | +0.09 (w 0.45) | +0.11 (w 0.45) | +0.12 (w 0.45) | +0.69 / +1.06 / +1.33 / +1.52 | 0.498 / 0.981 / 1.465 / 1.969 | 0.969 |
| CIFAR-100 | −0.15 (product) | −0.27 (product) | −0.37 (product) | −0.45 (product) | +0.51 / +0.93 / +1.26 / +1.46 | 0.279 / 0.551 / 0.824 / 1.107 | 0.107 |
| ResNet-18 | +0.00 (w 0.5) | −0.03 (w 0.5) | −0.07 (w 0.5) | −0.07 (w 0.5) | +0.11 / +0.18 / +0.22 / +0.18 | 0.317 / 0.624 / 0.931 / 1.250 | 0.250 |
| K=200 | −0.08 (CE) | −0.21 (CE) | −0.25 (w 0.2) | −0.23 (w 0.2) | +0.10 / +0.07 / −0.00 / −0.00 | 0.294 / 0.579 / 0.864 / 1.159 | 0.159 |
| K=500 | −0.15 (CE) | −0.36 (CE) | −0.48 (CE) | −0.42 (CE) | −0.03 / −0.25 / −0.44 / −0.49 | 0.292 / 0.575 / 0.859 / 1.153 | 0.153 |
| S1 replay | −0.10 (CE) | −0.35 (CE) | −0.63 (CE) | −0.79 (CE) | −0.15 / −0.31 / −0.44 / −0.52 | 0.288 / 0.566 / 0.845 / 1.132 | 0.132 |
| S2 replay | −0.00 (w 0.2) | −0.04 (w 0.2) | −0.09 (w 0.2) | −0.13 (w 0.2) | +0.18 / +0.43 / +0.58 / +0.64 | 0.269 / 0.529 / 0.790 / 1.058 | 0.058 |

**표에서 확인되는 사실**
- **비용은 β에 따라 정해집니다.** 같은 β에서 모든 방법은 같은 mask를 쓰므로 offload 비율, edge 호출 수, 두 cell 비율이 같습니다. β = 0.5에서 실제 offload 비율은 모든 설정에서 0.498–0.500입니다.
  - 요청당 edge 호출 수는 β만으로 정해지지 않고 두 cell 비율에 따라 달라집니다. 예를 들어 β = 1에서 S2는 1.058, random mobility는 1.969입니다.
- **보정한 고정 w 규칙과의 차이는 작습니다.** 이 차이는 첫날 CNN 설정 6개(S1, S2, S1-fast, partial participation, stepwise change, random mobility)에서 모든 β에 걸쳐 −0.07 ~ +0.14 pp입니다.
  - CIFAR-100, K=200, K=500, S1 replay, S2 replay에서는 보정한 규칙이 모든 β에서 높거나 같습니다.

### 3.3 같은 정확도에 필요한 비용

세 가지 목표를 비교했습니다(계획 4.2절). 각 칸은 그 목표에 처음 도달하는 β 격자점이고, 괄호 안은 인접 격자점 사이를 선형 보간한 offload 비율입니다.
- **사후 기술 분석:** S1 development 행을 뺀 모든 행은 평가 label로 도달 여부를 판정한 사후 기술 분석입니다. label 없이 동작하는 온라인 정책이 아닙니다.
- **NA:** β = 1까지 도달하지 못했음을 뜻합니다.
- **원본:** `tables/R20_T2_targets.csv`, `tables/R20_report_tables.md`

**목표 A: 원고 주표의 full-offload 비교값 (원고 offloading 문장의 7개 규칙 중 최고)**

| setting | 목표 | DriftGate | 보정한 고정 w 규칙 중 가장 먼저 도달 | 보정 없는 규칙 중 가장 먼저 도달 | 기존 고정 threshold sweep의 DriftGate-P |
|---|---|---|---|---|---|
| S1 | 67.37 (Logit sum) | 0.5 (0.421) | 0.5 (0.430, w 0.2) | 1 (Logit sum) | 0.517 |
| S2 | 70.07 (Logit sum) | 0.25 (0.122) | 0.25 (0.123, w 0.5) | 0.25 (0.174, Logit sum) | 0.242 |
| S1-fast | 68.12 (Logit sum) | 0.5 (0.460) | 0.5 (0.455, w 0.2) | 1 (Logit sum) | 0.524 |
| partial participation | 62.94 (Logit sum) | 0.25 (0.158) | 0.25 (0.159, w 0.5) | 0.25 (0.226, Logit sum) | 0.136 |
| stepwise change | 71.44 (Prob. average) | 0.5 (0.345) | 0.5 (0.363, w 0.45) | 0.75 (0.709, Prob. average) | 0.474 |
| random mobility | 69.58 (Prob. average) | 0.25 (0.224) | 0.25 (0.228, w 0.45) | 0.5 (0.476, Prob. average) | 0.372 |
| CIFAR-100 | 40.44 (Label-shift EM) | 1 (0.762) | 0.75 (0.704, product) | NA | 0.953 |
| ResNet-18 | 66.54 (Logit-entropy w.) | 0.25 (0.091) | 0.25 (0.091, w 0.5) | 0.25 (0.121, no corr. + adaptive w) | 0.225 |
| K=200 | 68.90 (Prob. average) | 1 (0.895) | 0.75 (0.714, w 0.2) | 1 (0.891, no corr. + adaptive w) | 0.850 |
| K=500 | 70.38 (Prob. average) | NA | 1 (0.787, CE) | 1 (0.811, no corr. + adaptive w) | 도달 못 함 |
| S1 replay | 72.27 (Prob. average) | NA | 0.75 (0.700, CE) | 1 (0.771, no corr. + adaptive w) | 도달 못 함 |
| S2 replay | 75.90 (Prob. average) | 0.5 (0.480) | 0.5 (0.474, w 0.2) | 1 (Prob. average) | 0.509 |

**목표 B: 보정 변형까지 넣은 13개 규칙의 full-offload 최고값**
- **DriftGate가 더 먼저 도달하는 설정 (S1, stepwise change, random mobility):**
  - DriftGate는 β = 0.75에서 도달합니다(보간 0.725, 0.676, 0.704).
  - 보정한 고정 w 규칙은 β = 1이 필요합니다(w 0.45의 보간값은 0.885, 0.970, 0.937).
  - 다만 이 차이는 DriftGate의 β = 0.75 정확도가 목표를 0.04, 0.09, 0.05 pp 넘어서 생긴 것입니다. S1의 경우 68.20%가 목표 68.16%를 넘었습니다. 모두 0.10 pp보다 작은 차이입니다.
- **같은 β에서 도달하는 설정:** S2(β = 0.75)와 ResNet-18(β = 0.25)입니다.
- **같은 β이지만 DriftGate의 보간값이 더 작은 설정:** S1-fast에서는 둘 다 β = 1에서 도달하지만, DriftGate는 보간 0.941이고 보정한 규칙은 1.000입니다.
- **DriftGate가 도달하지 못하는 설정:** partial participation, CIFAR-100, K=200, K=500, 두 replay입니다.

**목표 C: 보정한 방법 5개가 공통으로 도달하는 최고 정확도**
- 이 목표는 DriftGate, correction + w 0.5, correction + w 0.2, corrected edge only, correction + product가 β 격자에서 얻는 최고 정확도 가운데 최솟값입니다.
- 첫날 CNN 설정 6개에서 DriftGate와 가장 먼저 도달하는 보정 규칙은 같은 격자 β에서 도달합니다. 보간한 offload 비율의 차이는 −0.04 ~ +0.01입니다(S1 0.590 대 0.627, S2 0.279 대 0.283, stepwise 0.381 대 0.400).
- CIFAR-100, K=200, K=500에서는 보정 규칙이 더 작은 β에서 도달합니다.
- S1 replay에서는 둘 다 β = 0.5에서 도달하지만, 보간값은 보정 규칙(corrected edge only)이 더 작습니다(0.441 대 0.487).

**S1에서 development seed 5–7로 정한 운영점** (평가 seed 0–4, `tables/R20_T2_dev_operating_points.csv`)

| 목표 | 방법 | development에서 정한 β | 평가 정확도 | offload 비율 | 요청당 edge 호출 | 평가 목표 도달 |
|---|---|---|---|---|---|---|
| A (dev 66.66, 평가 67.37) | DriftGate, correction + w 0.5 / 0.2 / 0.45, corrected edge only, product | 0.5 | 67.41–67.73 | 0.499 | 0.565 | 모두 예 |
| A | probability average, logit sum | 0.75 | 67.31, 67.36 | 0.745 | 0.844 | 아니요 (0.06, 0.01 pp 부족) |
| B (dev 67.50, 평가 68.16) | DriftGate | 0.75 | 68.20 | 0.745 | 0.844 | 예 |
| B | correction + w 0.45 / w 0.5 | 1 | 68.21 / 68.13 | 1.000 | 1.132 | 예 / 아니요 |
| B | 그 밖의 방법 | development에서 도달 못 함 | | | | |
| C (dev 66.86, 평가 67.90) | corrected edge only | 0.75 | 67.91 | 0.745 | 0.844 | 예 |
| C | DriftGate와 나머지 보정 규칙 | 0.5 | 67.41–67.73 | 0.499 | 0.565 | 아니요 |

**S1 운영점 표를 읽는 방법**
- **목표 A:** 보정 규칙들은 DriftGate와 같은 비용으로 도달합니다. 보정 없는 규칙은 offload 비율 0.745가 필요하고, 그래도 평가 목표에 조금 못 미칩니다.
- **목표 B:**
  - DriftGate는 offload 비율 0.745로 68.20%를 얻습니다. correction + w 0.45는 모든 요청을 offload해서 68.21%를 얻습니다.
  - 이 S1 결과는 development와 평가에서 모두 같은 방향입니다. 다만 development에서 DriftGate의 β = 0.75 정확도는 67.54%로, 목표 67.50%를 0.04 pp만 넘었습니다.
  - 이 차이는 β 격자의 간격과 0.1 pp보다 작은 정확도 차이에 기대고 있으므로, 원고의 주장으로 쓰기에는 약합니다.
- **목표 C:** development의 정확도 수준이 평가보다 약 0.7 pp 낮습니다. 그래서 development로 정한 운영점이 평가 목표에 도달하지 못합니다.

**세 질문에 대한 답**

| 질문 | 답 |
|---|---|
| 기존 주표의 full-offload 비교값을 넘는 데 필요한 offloading | 고정 threshold sweep에서는 7개 설정, 0.136–0.524입니다(Round 18 A3, 기존 값). controller에서는 β = 0.25–0.5이며, 이 경우 7개 설정(S1, S2, S1-fast, partial participation, stepwise change, random mobility, ResNet-18)과 S2 replay입니다. |
| 보정 변형까지 포함한 full-offload 비교값을 넘는 데 필요한 offloading | 고정 threshold sweep에서는 5개 설정, 0.433–0.870입니다. controller에서는 S1, S2, stepwise change, random mobility에서 β = 0.75, ResNet-18에서 0.25, S1-fast에서 1입니다. 나머지 6개 설정에서는 도달하지 못합니다. |
| 각 방법이 budget을 조절할 때 같은 정확도를 얻는 데 필요한 offloading과 edge 호출 수 | 보정 없는 규칙과 비교하면 DriftGate와 보정한 고정 w 규칙이 더 적은 offloading으로 도달합니다(목표 A). 보정한 고정 w 규칙과 비교하면 대부분 같은 β에서 도달하므로 비용 차이가 없습니다. 목표 B의 세 설정에서 β 한 단계 차이가 있지만, 0.1 pp보다 작은 정확도 차이에서 생긴 것입니다. |

따라서 시스템 기여는 다음 두 가지로 정리합니다.
- 같은 budget에서 보정하지 않은 결합보다 정확합니다.
- 추가 계산이 작습니다.

43–86%와 5–20%는 confidence-based offloading(0.8 nats)의 offload 비율에 대한 감소율입니다. 가장 강한 보정 기준 규칙 대비 지연이나 에너지 절감으로 바꾸어 쓰지 않습니다.

## 4. 현재 a_k와 상수 보정의 비교

| 항목 | 내용 |
|---|---|
| 고정한 조건 | 같은 offload mask와 같은 가중치 시계열. 보정만 바꿉니다 |
| a0 | S1 development seed 5–7에서 요청이 있는 device-round의 a_k 중앙값, 0.2326 |
| S2와 S2 replay의 주 비교 | 원고 값 a = 0.25를 쓴 진단입니다 |
| 원본 | `tables/R20_T3_correction.csv`, `tables/R20_T3_a_stats.csv` |

| 설정 | β | 규칙 | b = 0 | a = 0.25 | a0 | 현재 a_k | 현재 a_k − b = 0 [양수 seed] | 현재 a_k − 주 상수 [양수 seed] |
|---|---|---|---|---|---|---|---|---|
| S1 | 1 | DriftGate | 67.11 | 68.45 | 68.40 | 68.31 | +1.20 [5/5] | −0.09 [0/5] (a0) |
| S1 | 1 | Corrected edge only | 62.91 | 68.19 | 68.26 | 67.87 | +4.96 [5/5] | −0.39 [0/5] |
| S1 | 0.5 | DriftGate | 66.90 | 67.83 | 67.80 | 67.73 | +0.83 [5/5] | −0.07 [0/5] |
| S2 | 1 | DriftGate | 69.29 | 71.91 | 71.98 | 71.70 | +2.41 [3/3] | −0.21 [1/3] (a = 0.25) |
| S2 | 1 | Corrected edge only | 65.24 | 70.85 | 71.02 | 70.49 | +5.25 [3/3] | −0.36 [1/3] |
| S2 | 0.5 | DriftGate | 70.17 | 71.92 | 71.95 | 71.77 | +1.60 [3/3] | −0.15 [1/3] |
| S1 replay | 1 | DriftGate | 72.46 | 72.44 | 72.35 | 71.95 | −0.52 [1/5] | −0.40 [0/5] (a0) |
| S1 replay | 1 | Corrected edge only | 71.74 | 73.30 | 73.27 | 72.74 | +0.99 [5/5] | −0.53 [0/5] |
| S2 replay | 1 | DriftGate | 75.88 | 76.62 | 76.61 | 76.54 | +0.65 [2/3] | −0.09 [1/3] (a = 0.25) |
| S2 replay | 1 | Corrected edge only | 74.94 | 76.71 | 76.73 | 76.59 | +1.65 [3/3] | −0.12 [1/3] |

correction + w 0.5와 β = 0.5의 나머지 행은 CSV에 있으며, 방향이 같습니다.

**보정의 기여 (현재 a_k − 보정 없음)**
- 첫날 S1과 S2에서 세 규칙 모두 양수이고, 모든 seed에서 양수입니다(+0.58 ~ +5.25 pp).
- frozen replay에서는 corrected edge only의 보정이 여전히 도움이 됩니다(S1 +0.99, S2 +1.65 pp).
- 반면 DriftGate와 correction + w 0.5에서는 S1 replay의 보정 효과가 음수입니다(−0.52, −0.74 pp). 학습된 S1 모델에서 device exit와 함께 쓸 때 보정이 과한 것으로 보입니다. 이 부분은 해석이며, 따로 시험하지 않았습니다.

**현재 cell 통계의 기여 (현재 a_k − 상수)**
- 네 설정, 두 β, 세 규칙의 24행 모두에서 평균이 음수입니다(−0.05 ~ −0.53 pp). S1과 S1 replay에서는 거의 모든 seed에서 상수가 더 높습니다.
- 계획 5절의 기준(+0.10 pp 이상이고 모든 seed에서 양수)을 만족한 행은 없습니다. 따라서 "현재 a_k가 정확도에 기여했다"고 쓰지 않습니다.

**a_k와 b_k의 분포 (S1 첫날)**
- **분포:** a_k의 중앙값은 0.240이고, 5–95 백분위수는 0.190–0.367입니다. b_k의 중앙값은 1.15이고, 5–95 백분위수는 0.54–1.45입니다.
- **device 안의 시간 변화:** device마다 round에 대한 a_k의 표준편차를 구하면 평균 0.040, 최대 0.088입니다. device 평균 a_k의 device 간 표준편차는 0.032입니다.
- **home과 away:** a_k의 중앙값은 home에서 0.260, away에서 0.220입니다.
  - 즉 away에서 a_k가 작아지고 b_k가 커집니다. 다시 말해 own class를 더 강하게 끌어올립니다.
  - 그런데 away에서는 Main 요청의 비중이 낮습니다.
  - 상수 보정이 조금 더 나은 결과는 이 방향과 맞지만, 원인으로 확인하지는 않았습니다(해석).
- **stepwise change와 ResNet-18:** cell 소속이 고정되어 있으므로 device 안의 변화가 0입니다.

**원고에 쓸 수 있는 문장의 범위**
- 보정은 학습 class 통계로 own class logit의 상수를 정하는 간단한 방법입니다.
- 그 값이 cell 변화를 따라가서 정확도가 좋아진다는 근거는 없습니다.

## 5. 원고 v28의 유지, 수정, 삭제 제안

원고는 덮어쓰지 않았습니다. 아래는 제안이며, 영문은 원고에 넣을 문장의 초안입니다.

| 위치 | 현재 문장(요지) | 조치 | 제안 | 근거 |
|---|---|---|---|---|
| Abstract, Introduction, 6.2 | "DriftGate has the highest mean accuracy in all eight settings, 0.1 to 1.9 points above the strongest alternative" | 수정 | "Among the eleven inference rules of Table main, DriftGate has the highest mean accuracy in all eight settings over the first day. Variants that keep its correction but use a fixed weight are within 0.16 points of it in seven settings and 0.45 points above it in CIFAR-100 (Table ablation)." | 3.1절 첫날 표 |
| Abstract, Conclusion, 6.5 | "offloading at most half of the requests is more accurate than every rule that offloads all of them in five of the eight settings" | 수정 | 비교 집합을 "every rule of Table main that offloads all requests"로 명시하고, "Rules that use the same correction with a fixed weight reach the same accuracy at the same offloading ratio."를 덧붙입니다 | 3.3절 목표 A |
| 6.5 controller 문단 | "With the same offloading decisions, DriftGate is the most accurate way to answer the offloaded requests ... in S1, S2, S2 replay" | 수정 | "most accurate among rules without the prior correction"으로 바꾸고, "With the correction and a fixed weight, the difference is −0.04 to +0.05 points in these cases."를 덧붙입니다 | `R20_reconciliations.md` 2.5 |
| 5.3 가중치 | "A weight tuned for one model can be far from the best for another" | 유지 + 보완 | "The adaptive weight is a default that needs no tuning per model. Within a run it changes little, and replacing it by its per-device mean changes accuracy by −0.14 to +0.09 points; we do not claim that it adapts to mobility." | Round 19 표 |
| 5.2 보정 | "a_k of about 0.25"와 r = 1/2의 설명 | 유지 + 보완 | "With r = 1/2 the correction adds log((1 − a_k)/a_k), about 1.1, to the logits of the device's own classes. Replacing a_k by a constant (0.23 from development runs, or 0.25) does not lower accuracy (Section ablation)." r은 두 그룹을 균등하게 두는 설계값이며 실제 요청 분포를 복원하지 않는다는 기존 문장은 유지합니다 | 4절 |
| 6.3 학습된 모델 | round > 30, replay 결과를 주표 11개 규칙의 순위로만 제시 | 수정 | "After round 30, the corrected edge exit alone is 0.48 to 1.31 points above DriftGate in S1, S1-fast, stepwise change, random mobility, and both scale settings."를 덧붙이고, 세 날 frozen replay의 결과(S1에서 corrected edge only +0.75 pp, S2에서 +0.10 pp)를 같은 절에 추가합니다 | 3.1절 학습된 모델 표 |
| 6.6 ablation | "The adaptive weight is close to the better fixed weight in each setting rather than better than both." | 유지 | | 3.1절 |
| 6.7 규모 | "with 500 devices, where the probability average is the strongest rule" | 수정 | "the strongest of the eleven rules"로 바꾸고, "The uncorrected adaptive weight and the corrected edge exit alone are 0.49 and 0.42 points above DriftGate."를 덧붙입니다 | `R20_reconciliations.md` 4절 |
| 6.7 하위 10% | "describes the lower tail of the accuracy distribution, not every individual device" | 유지 | | 원고가 이미 한정함 |
| 2.3 학습 쪽 정책 | "trade-off of this policy rather than attribute the loss to residents" | 유지 | 이미 특정 정책의 관찰로 한정되어 있습니다. 2.4의 "came with a loss for devices away"도 "the policy we tested"로 한정되어 있어 유지합니다 | |
| 7 Discussion 첫 문단 | "Distances in the feature space of the device block ... are a natural next candidate." | 삭제 후 교체 | "We tested such distances on the device and edge features. They separate Main requests with an AUROC of 0.66 to 0.74 on the device and 0.84 to 0.86 at the edge, but they separate the requests that switching to the corrected edge exit fixes from those it breaks only with an AUROC of 0.49 to 0.66, and gating on them did not improve accuracy." | Round 17, 18 |
| 7 Discussion 첫 문단 | "the label-free features we combined with a logistic regression did not close the gap" | 수정 | "a logistic regression trained to predict whether a request is Main"으로 학습 목표를 밝히고, 모든 요청별 결합법을 배제하는 증거로 쓰지 않습니다 | Round 18 |
| 7 Discussion, 학습된 모델 | "should therefore check the combination and the weight on its own traffic" | 유지 + 보완 | 세 날 frozen replay에서 처음 128개 요청 뒤 한 번 정한 가중치가 DriftGate보다 S1 0.23 pp, S2 0.05 pp 높았다는 결과를 덧붙입니다 | Round 19 |
| 5.5, 6.8 비용 | "an application payload of 44 bytes", "two edge calls" | 수정 | "44 bytes is the application payload returned by the endpoint that combines the request, not its network traffic. For a device in two cells, one edge receives the other edge's logits, averages them, applies the correction once, and returns the result; we count both edge calls and the transfer between the edges." 실측 전에는 표의 빨간 값을 측정값으로 쓰지 않습니다 | Task 5 미확인 |
| Conclusion | "the benefit of its correction and weighting depends on the deployment" | 수정 | "Most of the gain comes from the prior correction. A fixed weight with the same correction is about as accurate, and with trained models the corrected edge exit alone is often more accurate." | 1.3절 |

## 6. 추가 학습 없이 제출 원고를 마무리할 수 있는가

추가 학습 없이 마무리할 수 있습니다. 근거는 다음 세 가지입니다.

1. 현재 주장을 직접 제한하는 결과가 모두 기존 기록에서 확인되었습니다. 학습 후의 corrected edge only, 보정한 고정 w, 요청 가중 지표가 여기에 해당합니다.
2. 남은 미확인 질문은 원고의 정확도 주장을 바꾸지 않습니다. 촘촘한 β 격자, S1 밖의 development 운영점, Task 4가 여기에 해당합니다.
3. 새 backbone, seed 추가, faithful BTFL 재학습, 분할 위치 추가는 위 결론을 바꿀 구체적인 질문이 없으므로 제안하지 않습니다.

다만 두 조건이 있습니다.
- **원고 수정:** 5절의 수정, 특히 비교 집합의 명시와 보정 변형의 제시를 반영해야 합니다.
- **비용 절:** 모바일 실측(Task 5) 없이는 지연과 에너지를 측정값으로 쓸 수 없습니다. 실측 전에는 비용 절을 "가정에 기반한 모델"로 표시하거나 다음 두 사실로 줄입니다.
  - 요청당 edge 호출 수와 payload
  - 서버 CPU에서 측정한 결합 시간(3.2 μs)

## 7. 계획과 다른 점, 재현 검증, 산출물

### 계획과 다른 점

계획 4.3절은 S1 development의 목표 A와 B를 Round 15 cache의 full-offload 값으로 정한다고 했습니다.

- **발견한 문제:** 계산을 시작한 뒤 Round 15 cache에 development seed 5만 있고 seed 6과 7은 없다는 것을 확인했습니다.
- **조치:** development의 목표 A와 B는 Round 20이 다시 계산할 수 있는 규칙만으로 정했습니다.
  - 목표 A에서 빠진 규칙은 logit-entropy weighting, label-shift EM, learned weight입니다.
  - 목표 B에서는 여기에 correction + learned weight가 더 빠집니다.
- **결과를 보기 전의 결정:** 이 변경은 결과를 보기 전에, 자료가 없다는 이유로 정했습니다.
- **영향:** 평가 seed의 S1에서는 빠진 규칙들이 목표를 정하지 않습니다(평가 seed의 목표 A는 logit sum, 목표 B는 correction + w 0.2). 다만 development에서도 그런지는 확인할 수 없습니다.

### 재현 검증 (`tables/R20_checks.csv`)

| 비교 대상 | 결과 |
|---|---|
| Round 18 (42개 run) | 정확도·비용 차이 0 |
| Round 15 (43개 run) | 차이 0 |
| Round 19 (53개 run) | 정답 수 일치 |
| Round 16 refs (42개 run) | 답 100% 일치 |
| 보정식과 logit 상수 덧셈 | 확률 차이 최대 2.7e-7 |

### 산출물

| 파일 | 내용 |
|---|---|
| `R20_plan.md` | 계산 전에 고정한 정의 (commit `88d9b4b`) |
| `R20_reconciliations.md` | Task 0: 비교 집합, 수치 차이의 원인, 확인 상태 |
| `R20_final_report_ko.md` | 이 보고서 |
| `scripts/r20_recompute.py` | 저장된 출력에서 규칙을 다시 계산하고 재현 검증을 저장합니다 |
| `scripts/r20_tables.py` | Task 0–3의 표를 만듭니다 |
| `scripts/r20_report_tables.py` | 보고서용 markdown 표(`tables/R20_report_tables.md`)를 만듭니다 |
| `scripts/r20_sources.py` | 원천 파일 목록 `tables/R20_sources.csv`를 만듭니다 |
| `tables/R20_comparator_sets.csv` | 비교 집합의 대응표 |
| `tables/R20_T0_reconcile.csv` | 차이 설명에 쓴 값 |
| `tables/R20_T1_*.csv` | 정확도 표, seed별 값, 요약, 요청 가중 지표, away 비중 |
| `tables/R20_T2_*.csv` | 비용 곡선, 같은 budget 차이, 목표별 도달점, S1 development 운영점 |
| `tables/R20_T3_*.csv` | 보정 비교, a_k와 b_k의 통계 |
| `tables/R20_checks.csv`, `tables/R20_sources.csv` | 재현 검증과 원천 파일 목록 |

**재현 방법:** 저장소 루트에서 다음 순서로 실행합니다.
1. `cache/queue_small.sh`와 `cache/queue_big.sh`를 실행합니다. `cache/`는 git에 넣지 않습니다.
2. `python journal_expansion/artifacts/driftgate_tmc_r20_claims/scripts/r20_tables.py`를 실행합니다.
3. `r20_report_tables.py`와 `r20_sources.py`를 실행합니다.
