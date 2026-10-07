# Round 20 Task 0: 출처, 비교 범위, 수치 차이의 설명

이 문서는 같은 이름으로 불리지만 값이 다른 수치들이 왜 다른지 정리합니다. 각 항목에는 확인 상태를 붙였습니다.

**확인 상태의 정의**
- **VERIFIED:** Round 20에서 원시 요청 기록(요청별 device exit 확률, raw edge exit 확률, label, 요청 순서, cell 소속)으로 다시 계산했고, 이전 값과 같음을 확인했습니다.
- **CONSISTENT:** 이전 Round의 cache나 표에서 읽은 값이 서로 맞는다는 것만 확인했습니다. Round 20에서 원시 기록으로 다시 계산하지는 않았습니다.
- **UNVERIFIED:** 확인할 자료가 없습니다.

**원천 파일과 재현 검증**
- 원천 파일 478개의 크기, sha256, 마지막 commit(또는 git-ignored 여부)은 `tables/R20_sources.csv`에 있습니다.
- 재현 검증 결과는 `tables/R20_checks.csv`에 있습니다(53개 run).

## 0. 재현 검증 요약 (VERIFIED)

`scripts/r20_recompute.py`는 저장된 요청별 출력에서 규칙의 답을 다시 계산합니다. 새 학습이나 새 추론은 하지 않습니다.

| 비교 대상 | 범위 | 결과 |
|---|---|---|
| Round 18 cache | 42개 평가 run, β = 0.25–1의 round별 macro 정확도(10개 규칙과 device only), offload 비율, edge 호출 수 | 최대 절대 차이 0 |
| Round 15 cache | 43개 run(S1 development seed 5 포함), β = 1의 16개 규칙 정확도 | 최대 절대 차이 0 |
| Round 19 cache | 53개 run 전체, DriftGate와 고정 w 0, 0.2, 0.45, 0.5의 device-round별 정답 수(β = 1, 0.5) | 모두 일치 |
| Round 16 refs | 42개 run의 요청별 답 | 고정 규칙, DriftGate, no correction + adaptive w 모두 100% 일치. DriftGate 가중치의 차이도 0 |
| 보정식과 logit 상수 덧셈의 동치 | 모든 run의 처음 200,000개 요청 | 확률 차이 최대 2.7e-7, argmax 100% 일치 |

- 여기서 macro 정확도는 원고의 지표입니다. 평가 round마다 device 정확도를 평균한 뒤 round에 대해 평균합니다.
- 보정식과 logit 상수 덧셈의 동치는 다음과 같습니다. 보정식은 own class의 logit에 b_k = log((1 − r)(1 − a_k)/(r a_k))를 더하는 계산과 같습니다.
- **β = 0의 확인:**
  - Round 18의 controller를 β = 0으로 돌리면 offload 비율이 0.008이 됩니다. 0이 되지 않는 이유는 두 가지입니다. 처음 16개 요청에는 τ = 0.8을 쓰고, 그 뒤에는 이전 128개 entropy의 최댓값을 넘는 요청을 offload하기 때문입니다.
  - 그래서 Round 20 곡선의 β = 0 점은 offload하지 않는 device only로 정의했습니다(계획 2절).
- **β = 1의 확인:** β = 1의 각 규칙은 Round 15의 full-offload 값과 같습니다(위 표).

## 1. 비교 집합의 대응 (`tables/R20_comparator_sets.csv`)

| 비교 집합 | 규칙 수 | DriftGate 포함 | 보정한 고정 w | product | 보정 없는 adaptive w | label-shift EM | budget |
|---|---|---|---|---|---|---|---|
| 원고 주표 | 11 | 예 | 없음 | 보정 없는 logit sum만 | 없음 | 예 | confidence-based는 0.8 nats, geometric ensemble은 confidence-based와 같은 비율, 나머지는 모두 offload |
| 원고 ablation 표 | 8 | 예 | w 0.5, 0.2, corrected edge only | correction + product | 예 | 없음 | 모두 offload |
| 원고 offloading 문장, Round 18 A3의 Round 15 집합 | 7 | 아니요(비교 대상만) | 없음 | logit sum | 없음 | 예 | 모두 offload |
| Round 18 A3의 확장 집합 | 13 | 아니요 | w 0.5, 0.2, corrected edge only, correction + learned weight | logit sum, correction + product | 예 | 예 | 모두 offload |
| Round 17 strongest other | 11 | 아니요 | corrected edge only만 | logit sum | 없음 | 예 | 원고 주표와 같음 |
| Round 16–19 기준 집합 | 17 | 예 | w 0.5, 0.2, corrected edge only, correction + learned weight | 둘 다 | 예 | 예 | 원고 주표와 같음 |
| Round 18 같은 budget 비교의 "보정 없는 규칙" | 4 | 아니요 | 없음 | logit sum | 예 | **없음** | 같은 controller mask |
| Round 18 같은 budget 비교의 "보정한 규칙" | 5 | 아니요 | w 0.5, 0.2, corrected edge only, correction + learned weight | correction + product | 없음 | 없음 | 같은 controller mask |
| Round 19 고정 w 격자 | 21개 w | 예 | w = 0, 0.05, …, 1 | 없음 | 없음 | 없음 | β = 1과 0.5 |
| Round 20 비용 비교 | 10 | 예 | w 0.5, 0.2, 0.45, corrected edge only | 둘 다 | 예 | 없음 | 같은 controller mask, β 격자 |

**주의할 점**
- **label-shift EM:** 이 규칙은 EM 추정으로 자체 보정을 합니다. 따라서 "보정하지 않은 규칙"이라고 부르지 않습니다. Round 18과 Round 19.1이 말한 "보정하지 않은 규칙"은 위 표의 4개 규칙(raw edge only, probability average, logit sum, no correction + adaptive w)을 뜻하며, EM을 포함하지 않습니다.
- **w 0.45:** Round 19가 S1 development seed로 고른 값입니다. 원고에는 없습니다.

## 2. 같은 이름의 수치가 다른 경우

| # | 수치 | 원인 | 상태 |
|---|---|---|---|
| 2.1 | K=500 첫날에서 원고 규모 표의 Avg. 70.38%와, Round 19.1의 "보정 없는 최고 규칙"에서 역산한 70.60% | 규칙이 다릅니다. 70.38%는 probability average이고, 70.60%는 no correction + adaptive w입니다. run(seed 0–2), checkpoint, 구간(첫날 전체), budget(모두 offload), 집계(macro)는 같습니다. 원고의 11개 규칙에는 no correction + adaptive w가 없으므로, 원고에서는 probability average가 가장 강한 비교 규칙입니다. | VERIFIED |
| 2.2 | S1 첫날 DriftGate: 원고 주표 68.31%와 replay 표의 첫날 68.32%. S2 71.70%와 71.74%. S2 probability average 69.88%와 69.93% | run이 다릅니다. 주표는 Round 12의 원래 학습 run이고, replay 표의 첫날은 Round 15에서 같은 설정과 seed로 다시 학습한 run입니다. 원고에도 "within 0.05 points"라고 적혀 있습니다. | Round 12 값은 VERIFIED, Round 15 재학습 run의 값은 CONSISTENT(Round 15 cache) |
| 2.3 | S1 replay DriftGate: Round 17 표의 71.82%와 원고·Round 15의 71.95% | seed 집합이 다릅니다. Round 17은 seed 0을 후보 선택에 쓰고 seed 1–4로 판정했습니다. seed 0–4의 평균은 71.95%, seed 1–4의 평균은 71.82%이며, seed별 값은 같습니다. | VERIFIED |
| 2.4 | ResNet-18 첫날의 비교 대상: 원고 주표의 strongest alternative인 logit-entropy weighting 66.54%(DriftGate +0.10)와, Round 18의 "보정 없는 최고 규칙"인 no correction + adaptive w 66.46%(DriftGate +0.18) | 비교 집합이 다릅니다. Round 18의 4개 규칙 집합에는 logit-entropy weighting이 없습니다. | VERIFIED |
| 2.5 | 원고 β = 0.5 문장의 수치: S1 DriftGate 67.7%(logit sum 67.1%, edge 64.7%), S2 71.8%(70.6%, 68.2%), S2 replay 76.0%(probability average 75.5%), S1 replay에서 probability average보다 0.2 pp 낮음 | Round 20 재계산과 일치합니다. 다만 보정한 고정 w 규칙과 비교하면 차이는 다음과 같습니다. 원고 문장은 보정한 규칙을 비교하지 않았습니다. | VERIFIED |

2.5의 보정한 고정 w 규칙과의 차이(DriftGate − 규칙, `tables/R20_T0_reconcile.csv`)는 다음과 같습니다.
- S1: correction + w 0.2 대비 +0.05 pp
- S2: correction + w 0.5 대비 +0.01 pp
- S2 replay: correction + w 0.2 대비 −0.04 pp
- S1 replay: corrected edge only 대비 −0.35 pp

## 3. Round 19.1 문장의 범위

### 3.1 "첫날 CNN 7개 설정에서 +0.60 ~ +1.88 pp" (VERIFIED, 범위 수정 필요)

이 문장의 값은 DriftGate − (Round 18의 보정 없는 4개 규칙 중 최고)입니다. 14개 값은 다음과 같습니다(`tables/R20_T0_reconcile.csv`).

| 설정 | 최고 규칙 | β = 0.5 | β = 1 |
|---|---|---|---|
| S1 | logit sum | +0.66 | +0.95 |
| S2 | logit sum | +1.18 | +1.63 |
| S1-fast | logit sum | +0.60 | +0.84 |
| partial participation | logit sum | +1.11 | +1.88 |
| stepwise change | probability average | +0.69 | +0.92 |
| random mobility | probability average | +1.06 | +1.52 |
| CIFAR-100 | logit sum | +0.93 | +1.46 |
| ResNet-18 | no correction + adaptive w | +0.18 | +0.18 |
| K=200 | no correction + adaptive w | +0.07 | −0.00 |
| K=500 | no correction + adaptive w | −0.25 | −0.49 |

- **"CNN 7개"라는 표현은 정확하지 않습니다.** K=200과 K=500도 첫날 CNN 설정입니다. 7개는 첫날 설정 10개 가운데 ResNet-18, K=200, K=500을 뺀 것입니다. 빠진 세 설정에서는 no correction + adaptive w가 가장 강한 비교 규칙입니다.
- **고친 문장:** 첫날 설정 10개, β = 0.5와 1에서 DriftGate − 보정 없는 최고 규칙은 −0.49 ~ +1.88 pp입니다. 그중 7개 설정은 +0.60 ~ +1.88 pp이고, ResNet-18은 +0.18 pp, K=200은 +0.07 ~ −0.00 pp, K=500은 −0.25 ~ −0.49 pp입니다.

### 3.2 S1의 corrected edge only 대비 +0.45 pp의 분해 (VERIFIED, 표현 수정 필요)

S1, β = 1에서 DriftGate − corrected edge only를 구간별로 나누면 다음과 같습니다. 평가 round 31개 가운데 7개가 round ≤ 30입니다.

| 값의 종류 | round ≤ 30 | round > 30 | 첫날 전체 |
|---|---|---|---|
| 구간 안의 평균 차이 | +3.87 (0.89) | −0.55 (0.35) | +0.447 (0.190) |
| 첫날 평균에 대한 가중 기여(구간의 round 합 / 31) | +0.874 (0.201) | −0.427 (0.272) | +0.447 |

- 가중 기여의 합은 0.874 − 0.427 = 0.447로, 첫날 평균과 반올림 없이 맞습니다.
- Round 19.1 보고서는 "round ≤ 30 구간의 +0.87 pp와 round > 30 구간의 −0.43 pp"라고 썼습니다. 이 값은 구간 평균이 아니라 가중 기여입니다. 구간 평균은 +3.87 pp와 −0.55 pp입니다.
- **β = 0.5:** 구간 평균은 +1.67과 −0.26이고, 가중 기여는 +0.378과 −0.201입니다.
- **S2:** 구간 평균은 +3.54와 +0.53이고, 가중 기여는 +0.799와 +0.413입니다. S2에서는 후반에도 DriftGate가 높습니다.

### 3.3 offload 요청 43–86% 감소와 5–20% 감소의 분모 (CONSISTENT, Round 18 A3)

**계산 방식(`R18_analysis.a3`)**
- 감소율은 1 − (DriftGate-P가 목표에 처음 도달하는 고정 threshold에서의 offload 비율) / (confidence-based offloading이 0.8 nats에서 offload하는 비율)입니다. 분모는 confidence-based offloading의 offload 비율이며, S1에서 0.920입니다.
- 목표는 비교 집합의 full-offload 최고 정확도입니다. 43–86%는 Round 15의 7개 규칙에 대한 값이고, 5–20%는 13개 규칙에 대한 값입니다.
- DriftGate-P의 곡선은 Round 15의 고정 threshold sweep(0–2.3 nats, 0.05 간격)입니다. "처음 도달하는 threshold"는 평가 seed의 정확도로 고른 점입니다.
- Round 20은 이 고정 threshold sweep을 다시 계산하지 않았습니다. Round 18 A3 표의 값을 그대로 씁니다. 대신 β controller 격자는 다시 계산했습니다.

**이 수치가 말하는 것과 말하지 않는 것**
- 이 감소율은 DriftGate가 비교 규칙의 full-offload 정확도를 넘는 데 필요한 offloading을 기본 confidence threshold와 비교한 값입니다.
- 비교 규칙이 자기 budget을 조절할 때 같은 정확도를 얻는 최소 비용과 비교한 값은 아닙니다. 이 비교는 Round 20 Task 2에서 따로 합니다(`R20_final_report_ko.md` 3.3절).
- 지연이나 에너지의 절감률도 아닙니다.

**설정별 범위**
- Round 15 집합에서 도달하는 첫날 설정은 7개이고, offload 비율은 0.136–0.524입니다. 그중 0.5 이하는 5개입니다. S1(0.517)과 S1-fast(0.524)는 0.5를 조금 넘습니다.
- 13개 집합에서는 5개 설정, 0.433–0.870입니다.

## 4. 원고의 기술값

| 원고 문장 | Round 20 값 | 상태 |
|---|---|---|
| "a_k is about 0.25 on average" (commute) | S1 첫날, 요청이 있는 device-round: 중앙값 0.240, 5–95 백분위수 0.190–0.367. development 중앙값 a0 = 0.2326 | VERIFIED |
| "w_k is 0.42 on average with a standard deviation of 0.04" | 평균 0.419는 Round 19 표와 맞습니다. 실행 안의 표준편차 0.04는 Round 20에서 다시 계산하지 않았습니다. | 평균은 CONSISTENT, 표준편차는 UNVERIFIED |
| "in S1 this adds 0.13 calls per offloaded request" | offload된 요청 중 두 cell 요청의 비율 0.132(β와 무관) | VERIFIED |
| 규모 표 K=500 "Gain −0.27" | 원고의 11개 규칙 기준으로 맞습니다. no correction + adaptive w(70.60)와 corrected edge only(70.53)를 넣으면 DriftGate가 0.49 pp와 0.42 pp 낮습니다. | VERIFIED |
