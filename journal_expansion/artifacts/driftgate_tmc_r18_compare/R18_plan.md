# Round 18 분석 계획과 판정 규칙

이 문서는 Round 18 지시문의 A–C와 E1을 실행하기 전에 고정한 정의입니다. 분석 스크립트(`scripts/r18_analysis.py`)는 이 정의를 그대로 구현합니다. C의 후보는 Round 17 결과를 본 뒤 제안된 가설이므로, 기존 seed에서 얻은 결과는 탐색 결과입니다. 이 계획을 결과 전에 commit하더라도 기존 seed의 결과를 독립 검증이라고 부르지 않습니다. 새 학습은 하지 않습니다.

## 1. 기록

| 이름 | 기록 | seed |
|---|---|---|
| S1, S2 (첫날) | `runs/phaseT12_fusion/r12_s{1,2}_fixed040_s{}` (논문 주표와 같은 Round 12 기록) | 0–4, 0–2 |
| S1-fast, partial participation, stepwise change, random mobility, ResNet-18, K=200, K=500 | Round 12 기록 (Round 15 A1과 같음) | 0–2 또는 0–4 |
| CIFAR-100 | `runs/phaseT13b_arch/r13b_c100_s{}` | 0–2 |
| S1 replay, S2 replay | `runs/phaseT15_replay/r15_s{1,2}_replay_s{}` | 0–4, 0–2 |

- 기존 규칙의 요청별 답은 Round 16 phase A cache(`driftgate_tmc_r16_selector/cache/*__refs-v1.npz`)에서 가져옵니다. 이 cache는 Round 15 정의를 그대로 따른 값이며, Round 16에서 850개 값이 Round 15 cache와 비트 단위로 같음을 확인했습니다. 쓰는 파일의 크기와 수정 시각은 `R18_reference_manifest.csv`에 적습니다.
- Round 15에서 재학습한 S1·S2 첫날 기록은 이 표들에 넣지 않습니다. 한 표 안에서 서로 다른 첫날 기록을 섞지 않기 위해서입니다.
- feature가 필요한 진단(B1의 D1–D4, B2)은 Round 17의 재생 기록(`runs/phaseT17_featgate`)을 씁니다. 이 기록은 Round 15 재생 기록과 비트 단위로 같습니다.

## 2. 같은 budget의 온라인 비교 (A2)

**controller:** Round 15 C2의 controller를 그대로 쓰되, 목표 비율 β만 바꿉니다.
- 문턱은 device의 직전 128개 요청의 device entropy에서 구한 (1 − β) 분위수입니다(선형 보간).
- 직전 요청이 16개 미만이면 τ = 0.8을 씁니다.
- 현재 요청은 문턱 계산에 넣지 않습니다.
- β ∈ {0.25, 0.5, 0.75, 1.0}이고, β = 1은 모든 요청을 offload하는 경우로 정의합니다.
- β = 0.5의 결과는 Round 15의 `online_offload`와 같아야 합니다(스크립트에서 확인).

β마다 offload mask를 한 번 만들고, 아래 모든 규칙에 같은 mask를 적용합니다. offload하지 않은 요청에는 device exit의 답을 씁니다.

| 규칙 | offload된 요청의 답 |
|---|---|
| Raw edge only | 보정하지 않은 edge exit의 argmax |
| Probability average | Round 15 정의 |
| Logit sum | Round 15 정의 |
| DriftGate-P | 보정한 edge와 device의 평균입니다. 가중치는 Round 15 DriftGate-P 가중치이며, 같은 mask의 offload된 요청으로 갱신합니다 |
| Correction + fixed w = 0.5 | Round 15 정의 |
| Correction + development w = 0.2 | Round 15 정의 |
| Corrected edge only | r = 0.5로 보정한 edge exit |
| Correction + product | Round 15 정의 (지수 1의 확률 곱) |
| No correction + adaptive w | DriftGate-P와 같은 가중치로 보정하지 않은 edge와 평균 |
| Correction + learned weight | Round 15 정의. 특징 가운데 window 평균 특징은 전체 요청의 edge 출력으로 계산되어 있으므로, offload되지 않은 요청의 정보를 씁니다(보고서에 적음) |

Device only(offloading 0)는 따로 적습니다. 비교 집합은 위 표의 규칙들입니다. **strongest reference**는 같은 행과 같은 β에서, DriftGate-P를 뺀 나머지 9개 규칙 가운데 seed 평균이 가장 높은 규칙입니다.

**보고 항목:**
- 정확도(전체, home, away, Main, OOP, OOR)
- 같은 seed끼리 뺀 차이
- 실제 offloading 비율
- 요청당 edge 호출 수: offload된 요청은 cell 수만큼 호출합니다.

**같은 정확도에 필요한 offloading:** 목표 정확도는 β = 1에서 9개 비교 규칙 가운데 가장 높은 seed 평균입니다. 규칙마다 (0, Device only)와 네 β의 (실제 offloading, seed 평균 정확도) 점을 offloading 순서로 선형 보간해, 목표에 처음 도달하는 offloading을 구합니다. 도달하지 못하면 미도달로 적습니다.

## 3. 논문 문장의 재계산 (A3)

Round 15 C1과 같은 방법으로 계산하되, full-offload 비교 집합만 넓힙니다.
- 원래 집합은 Edge only, Probability average, Logit sum, Lower-entropy exit, Label-shift EM, Learned weight, Logit-entropy weighting입니다.
- 넓힌 집합은 원래 집합에 보정 변형 6개를 더한 것입니다: no correction + adaptive w, correction + w 0.5, correction + dev w 0.2, corrected edge only, correction + learned weight, correction + product.

DriftGate-P의 고정 threshold 곡선(Round 15 cache)이 각 집합의 최고 seed 평균에 처음 도달하는 격자점의 offloading 비율을 구합니다. 그 비율과 confidence-based offloading의 기본 threshold(0.8 nats) 사이의 상대 감소도 다시 구합니다. 같은 threshold에서 요청당 edge 호출 수(두 cell 포함)도 계산합니다.

## 4. 진단 (B)

모든 요청을 offload한 상태(β = 1)에서 base 답을 DriftGate, alternative 답을 corrected edge only로 둡니다. u = 1[alternative 정답] − 1[base 정답]입니다.

- 비율은 논문 평가와 같은 가중치로 계산합니다. 요청의 가중치는 1/(device-round의 요청 수 × 그 round에서 요청이 있는 device 수 × round 수)입니다. 두 규칙의 답이 다른 요청에서도 따로 계산합니다.
- **신호:** 신호마다 u = +1인 요청과 u = −1인 요청을 얼마나 구별하는지 AUROC(수신자 조작 특성 곡선 아래 면적)로 봅니다. 요청 단위로 모아 계산하며 가중치는 쓰지 않습니다.
  - device entropy
  - 보정한 edge의 Main 확률 합
  - 두 규칙의 예측 margin 차이(corrected edge의 top-1 − top-2에서 DriftGate의 top-1 − top-2를 뺀 값)
  - Q_DG − E = w(D − E)
  - D1–D4(재생만, Round 17의 α = 0.10)
- **전환 규칙:** 다음 문턱에서 alternative로 바꾸는 규칙의 rescued, harmed, net을 전체 요청 분모로 구합니다.
  - entropy > 0.8
  - Main 확률 합 < 0.5
  - margin 차이 > 0
  - Q_DG − E > 0
  - D1–D4는 Round 17의 판정을 씁니다.
- **headroom:** 두 답 가운데 정답을 고르는 oracle의 정확도와 DriftGate의 차이입니다.
- **D4의 유효 범위(B2):** D4가 정의된 요청에서 entropy, D1, D2, D3, D4의 Main 대 non-Main AUROC를 다시 계산합니다. home과 away로 나누고, device별 AUROC의 평균, 한 종류만 있는 device의 수와 요청 비중을 적습니다.

## 5. Conditional combination (C)

요청마다 d = device 확률, e = 보정한 edge 확률(r = 0.5), M = M_k, D = Σ_{c∈M} d_c, E = Σ_{c∈M} e_c로 둡니다. 후보는 다음과 같습니다.

- c ∈ M이면 p(c) = E·w·d_c/D + (1 − w)·e_c
- c ∉ M이면 p(c) = e_c
- 답은 argmax입니다. 계산은 float64로 합니다.
- D 또는 E가 10⁻¹²보다 작으면 corrected edge only로 답하고, 그 횟수를 셉니다.

| 이름 | w | e |
|---|---|---|
| 주 후보 | DriftGate 가중치 (β = 1은 Round 13b F-auto, 일부 offload는 같은 mask의 DriftGate-P 가중치) | 보정한 edge |
| 통제 1 | 0.5 | 보정한 edge |
| 통제 2 | 주 후보와 같음 | 보정하지 않은 edge |

결과를 본 뒤에 r, w, 문턱의 격자를 추가하지 않습니다.

**행:** 행마다 (후보 − DriftGate)와 (후보 − strongest reference)를 seed끼리 구하고, reference 이름을 적습니다.
- 첫날 10개 설정에서 구간 4개: 전체, round ≤ 30, round > 30, round > 50
- S1·S2 첫날의 시간대 5개: 출근 전, 출근, 낮, 귀가, 저녁
- S1·S2 재생의 전체, home, away

여기서 DriftGate는 같은 β의 DriftGate-P입니다. β = 1에서는 Round 15의 DriftGate와 같아야 하며, 이를 스크립트에서 확인합니다. strongest reference는 2절의 9개 규칙에서 고릅니다.

**엄격 판정:** β = 1과 β = 0.5 각각에서 위의 모든 행이 다음 두 조건을 만족하면 통과합니다. 반올림하지 않고 내부 부동소수점 오차 10⁻⁸ pp만 허용합니다.
- 후보 ≥ DriftGate
- 후보 ≥ strongest reference

**독립 검증 후보 판정:** 핵심 행은 S1·S2 첫날 전체와 S1·S2 재생 전체이고, β = 1과 β = 0.5에서 봅니다(모두 8칸). 다음 두 조건을 모두 만족하면 독립 검증 후보로 올립니다. 이 판정은 실험 자원을 배분하기 위한 것이며, 엄격 판정이나 논문 채택 조건을 대신하지 않습니다.
- 8칸 모두에서 후보 ≥ DriftGate
- 재생 4칸 모두에서 후보 − strongest reference ≥ +0.5 pp

**오류 분해(C3):** 같은 가중치로 다음 네 종류의 오류율을 구하고, 네 차이의 합이 전체 오류 차이와 같은지 확인합니다.
- 정답이 Main인데 non-Main을 예측
- 정답이 non-Main인데 Main을 예측
- Main 안에서 다른 Main class를 예측
- non-Main 안에서 다른 class를 예측

## 6. Main 비중 민감도 (E1)

β = 1의 예측을 고정합니다. device-round마다 Main 정확도 a_M과 non-Main 정확도 a_N(관측된 OOP/OOR 구성 그대로)을 구합니다. s ∈ {0.3, 0.5, 0.65, 0.8, 0.9}마다 s·a_M + (1 − s)·a_N을 device-round의 정확도로 둡니다. 그 값을 round마다 device에 대해 평균하고, 다시 round에 대해 평균합니다. Main과 non-Main이 모두 있는 device-round만 쓰며, 그 비율(coverage)을 적습니다.
