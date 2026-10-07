# Round 20 계획: 기존 결과로 제출 주장을 확정하기

이 문서는 Round 20 지시문의 Task 0–3을 실행하기 전에 고정한 정의입니다. 이 문서를 commit한 뒤에 계산을 시작합니다. 계산 결과를 본 뒤에는 목표 정확도, 상수 a0, 비교 집합, 서술 규칙을 바꾸지 않습니다.

## 0. 범위

**수행하는 작업**
- **Task 0–3:** 수행합니다.
- **새 학습과 새 추론:** 하지 않습니다. 계산은 이미 저장된 요청별 출력(device exit 확률 p_d, raw edge exit 확률 p_e, 요청 순서, cell 소속, M_k, a_k)에서 규칙의 답을 다시 구하는 데까지만 합니다.

**수행하지 않는 작업**
- **Task 4 (offloading의 정답 이득을 예측하는 선택적 탐색):** 기본 실행에서 비활성화되어 있으므로 수행하지 않습니다.
- **Task 5 (모바일 실측):** 장비가 준비되지 않았으므로 수행하지 않고, 해당 항목은 미확인으로 남깁니다.
- **탐색 금지:** 탐지기, 가중치, BBSE(black-box shift estimation), 조건부 결합의 새 변형은 탐색하지 않습니다.

**자료의 성격:** 평가에 쓰는 seed와 요청은 Round 15–19에서 이미 결과를 본 탐색 자료입니다. 이 계획을 먼저 commit하더라도 이 자료를 독립 검증 자료로 부르지 않습니다.

## 1. 자료

| 구분 | 설정 | 기록 | seed |
|---|---|---|---|
| 첫날 | S1, S2, S1-fast, partial participation, stepwise change, random mobility, CIFAR-100, ResNet-18, K=200, K=500 | Round 12, 13b | S1·stepwise 0–4, 나머지 0–2 |
| frozen replay | S1 replay, S2 replay | Round 15 | 0–4, 0–2 |
| 세 날 frozen replay | S1, S2 | Round 19 (`runs/phaseT19_multiday`) | 0–4, 0–2 |
| development | S1 development | Round 10 (`runs/phaseT10_prior/r10_T40_s{5,6,7}`) | 5–7 |

**S1 development의 사용 범위**
- 이 기록은 S1과 학습 설정이 같습니다. 다른 것은 seed와 run 이름뿐입니다.
- Task 2의 S1 운영점과 Task 3의 a0를 정하는 데에만 씁니다.
- 다른 설정에는 development 기록이 없습니다.

**재사용하는 기존 값**
- **Round 15 cache:** 규칙별 첫날 정확도, round > 30 정확도, frozen replay 정확도, 고정 threshold sweep
- **Round 18 cache:** β = 0.25, 0.5, 0.75, 1에서 같은 offload mask로 구한 규칙별 round 정확도, offload 비율, edge 호출 수
- **Round 19 표와 cache:** 고정 w 격자, 세 날 replay
- **Round 16 cache:** 요청별 기준 답(refs)

## 2. 재계산과 재현 검증 (`scripts/r20_recompute.py`)

각 run의 기록을 읽어 아래 값을 다시 계산하고 `cache/`에 저장합니다.

- **β와 offload mask:**
  - β ∈ {0.25, 0.5, 0.75, 1}에서는 Round 18의 controller로 offload mask를 만듭니다. β = 1은 모든 요청을 offload합니다.
  - β = 0의 점은 offload하지 않는 device only로 정의합니다.
  - 지시문의 β = 0 확인을 위해 controller(β = 0)의 실제 offload 비율도 기록합니다.
- **DriftGate의 가중치:** 각 mask에서 offload된 요청만으로 갱신합니다(DriftGate-P, Round 17의 `dgp_weight`). β = 1에서는 원래 DriftGate의 가중치와 같아야 합니다.
- **규칙:**
  - device only, raw edge only, probability average, logit sum, lower-entropy exit
  - no correction + adaptive w
  - correction + w 0.5, correction + w 0.2, correction + w 0.45, corrected edge only, correction + product
  - DriftGate
  - confidence-based offloading (0.8 nats, 고정 threshold)
- **refs가 있는 run의 β = 1 규칙:** Round 16 refs에서 logit-entropy weighting, label-shift EM, learned weight, correction + learned weight의 답을 읽습니다.
- **집계:** device-round마다 정답 수와 요청 수를 저장합니다. 이 두 값으로 원고의 지표(평가 round마다 device 정확도를 평균한 뒤 round에 대해 평균, 이하 macro 정확도)와 요청 수로 가중한 정확도를 모두 계산할 수 있습니다.
- **비용:** offload 비율, 요청당 edge 호출 수, offload된 요청 중 두 cell에 속한 요청의 비율

**재현 검증 기준**
1. Round 18 cache에 있는 규칙과 β에서, round별 macro 정확도가 재계산 값과 1e-9 이내로 같아야 합니다.
2. β = 1의 첫날 전체와 frozen replay 정확도가 Round 15 cache의 값과 1e-9 이내로 같아야 합니다.
3. 세 날 replay와 S1 development에서는 Round 19 cache의 DriftGate와 고정 w 정답 수가 정확히 같아야 합니다.
4. 같지 않은 항목이 있으면 그 항목을 VERIFIED로 표시하지 않고 원인을 보고합니다.

## 3. Task 1: 최종 정확도 표

- **구간:** 첫날 전체, round > 30(첫날 설정 10개), frozen replay(S1, S2), 세 날 frozen replay(S1, S2)를 씁니다. 새 구간은 만들지 않습니다.
- **budget:** β = 1(모든 요청 offload)과 β = 0.5(같은 mask)를 씁니다. confidence-based offloading은 0.8 nats threshold의 자기 offload 비율로 보고합니다.
- **필수 행:**
  - device only, raw edge only, confidence-based offloading
  - probability average, logit sum
  - 원고 주표의 최고 비교 규칙: 원고 주표 11개 규칙 가운데 DriftGate를 뺀 10개 중, 그 구간에서 seed 평균이 가장 높은 규칙입니다.
  - no correction + adaptive w
  - correction + w 0.5
  - correction + development w: 0.2(Round 11, 원고)와 0.45(Round 19, S1 development)를 둘 다 둡니다.
  - corrected edge only, correction + product, DriftGate
- **부록 행:**
  - lower-entropy exit, logit-entropy weighting, geometric ensemble with early exit, label-shift EM, learned weight, correction + learned weight
  - Round 19의 최고 고정 w: 평가 결과로 고른 값이므로 "사후 참조"로 표시합니다.
- **차이와 통계:**
  - 모든 차이는 같은 seed의 같은 요청에서 DriftGate와 비교해 계산합니다.
  - seed별 값, 평균, 표준편차(ddof = 1)를 보존합니다.
  - 동등성이나 비열등성은 선언하지 않습니다.
- **보조 지표:**
  - 요청 수로 가중한 정확도는 refs나 재계산 답이 있는 규칙에 대해 한 번 계산합니다. geometric ensemble은 요청별 답이 저장되어 있지 않아 제외합니다.
  - 이 지표로 주 지표를 바꾸지 않습니다. DriftGate의 순위가 바뀌는지와 그 원인만 보고합니다.

## 4. Task 2: 같은 정확도를 얻는 데 필요한 비용

### 4.1 비교 방법과 곡선

- **곡선의 점:** 방법마다 β ∈ {0, 0.25, 0.5, 0.75, 1}의 정확도, 실제 offload 비율, 요청당 edge 호출 수, 두 cell 비율을 씁니다.
- **같은 β에서의 mask:** 모든 방법이 같은 mask를 씁니다. 따라서 같은 β에서는 비용도 같습니다. 이 사실은 표에서 수치로 확인합니다.
- **비교 방법 (10개):**
  - DriftGate, no correction + adaptive w
  - probability average, logit sum, raw edge only
  - correction + w 0.5, correction + w 0.2, correction + w 0.45
  - corrected edge only, correction + product

### 4.2 목표 정확도

| 목표 | 정의 | 규칙 집합 |
|---|---|---|
| A | 원고의 offloading 문장이 쓴 full-offload 비교값 | Round 15의 7개 full-offload 규칙(raw edge only, probability average, logit sum, lower-entropy exit, logit-entropy weighting, label-shift EM, learned weight)의 β = 1 정확도 중 최댓값 |
| B | 보정 변형까지 포함한 full-offload 비교값 | A의 7개에 no correction + adaptive w, correction + w 0.5, correction + w 0.2, corrected edge only, correction + learned weight, correction + product를 더한 13개 규칙의 β = 1 최댓값 |
| C | 보정한 방법들이 budget을 조절해 공통으로 도달하는 정확도 | DriftGate, correction + w 0.5, correction + w 0.2, corrected edge only, correction + product 각각이 β 격자에서 얻는 최고 정확도 가운데 최솟값 |

### 4.3 운영점

- **S1 (development로 정한 운영점):**
  - development seed 5–7에서 목표 A, B, C를 같은 정의로 계산합니다(A와 B의 β = 1 값은 Round 15 cache).
  - 방법마다 development seed 평균이 목표에 처음 도달하는 가장 작은 격자 β를 운영점으로 정합니다. 운영점에서는 보간하지 않습니다.
  - 평가 seed 0–4에서는 그 β의 정확도, offload 비율, edge 호출 수와 평가 seed에서 다시 계산한 목표의 도달 여부를 보고합니다.
  - learned weight는 development run으로 학습했으므로 development에서 계산한 그 값은 in-sample입니다. 이 사실은 표에 적습니다.
- **모든 설정 (사후 기술 분석):**
  - 평가 seed 평균 곡선에서 목표에 처음 도달하는 격자 β와 그 비용을 보고합니다.
  - 인접 격자점 사이를 선형 보간한 offload 비율도 함께 보고하되, 보간값이라고 표시합니다.
  - 도달하지 못한 목표는 NA로 둡니다.
  - 이 값은 평가 label로 고른 점이므로 label 없이 동작하는 온라인 정책이라고 부르지 않습니다.
- **원고 문장과의 대응:**
  - 원고의 43–86%와 5–20%는 confidence-based offloading(0.8 nats)의 offload 비율에 대한 감소율입니다.
  - 이 비율은 지금 Round 15의 고정 threshold sweep에서 나온 값이며, 이번 Round에서도 그대로 다시 보고합니다.
  - 목표 C의 비교는 이 비율과 별도로 보고합니다.

### 4.4 최소 확인

- **β = 0:** controller(β = 0)의 실제 offload 비율을 기록합니다. device only의 정확도는 Round 18 cache와 같아야 합니다.
- **β = 1:** 각 규칙의 정확도가 Round 15의 full-offload 값과 같아야 합니다.

## 5. Task 3: 보정에 실제 cell 통계가 필요한가

- **설정과 β:** S1, S2, S1 replay, S2 replay에서 β = 0.5와 1을 씁니다.
- **고정하는 조건:** 각 β에서 offload mask와 DriftGate 가중치 시계열은 원래 계산과 같게 둡니다. 가중치는 raw edge entropy로 계산하므로 보정을 바꿔도 변하지 않습니다.

**보정의 세 형태**
1. **b = 0:** 보정하지 않습니다.
2. **상수 보정:** 모든 요청에 하나의 a를 씁니다.
   - **a0:** S1 development seed 5–7에서 요청이 있는 device-round의 a_k를 모두 모아 구한 중앙값입니다. 평가 label은 보지 않습니다.
   - **a = 0.25:** 원고의 기술값으로 고정한 진단용 비교입니다.
   - **주 비교:** S1과 S1 replay는 a0를 씁니다. S2와 S2 replay는 같은 시나리오의 development 기록이 없으므로 a = 0.25를 씁니다. 두 상수의 결과는 네 설정 모두에 대해 함께 보고합니다.
3. **현재 a_k:** 기존 DriftGate가 쓰는 a_k입니다.

**적용하는 규칙**
- DriftGate(가중치 고정)
- corrected edge only: b = 0이면 raw edge only가 됩니다.
- correction + w 0.5: b = 0이면 probability average가 됩니다.

**기여의 정의**
- **보정의 기여:** (현재 a_k) − (b = 0)
- **현재 cell 통계의 기여:** (현재 a_k) − (상수 보정)

**a_k와 b_k의 기술 통계**
- b_k = log((1 − a_k)/a_k)는 r = 0.5일 때 own class logit에 더하는 상수입니다.
- 분포: 요청이 있는 device-round에서 5, 25, 50, 75, 95 백분위수를 구합니다.
- device 안의 시간 변화: device마다 round에 대한 표준편차를 구한 뒤, device 평균과 최댓값을 보고합니다.
- device 사이의 차이를 보고합니다.
- 보정식과 logit 상수 덧셈의 동치를 수치로 확인합니다. 이 동치는 새 알고리즘으로 주장하지 않습니다.

**서술 규칙**
- 현재 cell 통계의 기여가 한 설정·β에서 평균 +0.10 pp 이상이고 모든 seed에서 양수일 때만, 그 설정에서 현재 a_k가 정확도에 기여했다고 씁니다.
- 그 밖의 경우에는 "cell 변화에 맞춰 보정하므로 성능이 좋아진다"는 설명을 쓰지 않습니다.

## 6. 서술 규칙 (공통)

- **작은 차이:** 평균 차이의 절댓값이 0.10 pp 미만이거나 seed마다 방향이 다르면 "차이가 작거나 방향이 일정하지 않다"고 씁니다. 같다거나 동등하다고 쓰지 않습니다.
- **보정한 변형의 해석:** 보정한 변형은 DriftGate의 구성 요소를 쓰는 변형입니다. 이 변형이 높다고 해서 보정의 가치가 사라졌다고 해석하지 않습니다. 반대로 entropy 가중치의 독립적 우위도 주장하지 않습니다.

## 7. 산출물

**Task 0**
- `tables/R20_sources.csv`: 원천 파일, sha256, 마지막 commit
- `tables/R20_comparator_sets.csv`: 비교 집합의 대응표
- `R20_reconciliations.md`: 같은 이름의 수치가 다른 이유와 확인 상태

**Task 1**
- `tables/R20_T1_accuracy.csv`, `tables/R20_T1_accuracy_per_seed.csv`: 대표 정확도 표
- `tables/R20_T1_request_weighted.csv`: 요청 수로 가중한 보조 지표

**Task 2**
- `tables/R20_T2_cost_curves.csv`: 방법별 곡선과 비용
- `tables/R20_T2_targets.csv`: 목표별로 필요한 offloading
- `tables/R20_T2_dev_operating_points.csv`: S1의 development 운영점

**Task 3**
- `tables/R20_T3_correction.csv`: 세 보정의 비교
- `tables/R20_T3_a_stats.csv`: a_k와 b_k의 통계

**최종**
- `tables/R20_checks.csv`: 재현 검증 결과
- `R20_final_report_ko.md`: 최종 보고서
