# Round 19 0단계 계획: 고정 가중치와 DriftGate 가중치의 비교

이 문서는 Round 19 지시문 2절(0단계)과 3절(상태 판정)을 실행하기 전에 고정한 정의입니다. 분석 스크립트(`scripts/r19_analysis.py`)는 이 정의를 그대로 구현합니다. 평가에 쓰는 seed와 요청 기록은 Round 18까지 이미 결과를 본 탐색 자료입니다. 이 계획을 먼저 commit하더라도 이 자료를 독립 검증 자료로 부르지 않습니다. 새 학습은 하지 않습니다.

여기서 w는 offload된 요청에서 device exit에 주는 가중치입니다. 학습의 personalization ratio λ(0.4)와는 다릅니다. prior correction(r = 0.5), offloading controller, 학습 설정은 바꾸지 않습니다.

## 1. 기록 (사용 가능한 기존 설정 전체)

| 설정 | 기록 | seed | 모델과 분할 |
|---|---|---|---|
| S1, S2, S1-fast, partial participation, stepwise change, random mobility, K=200, K=500 | Round 12 첫날 기록 | S1·stepwise 0–4, 나머지 0–2 | 기본 CNN |
| CIFAR-100 | Round 13b | 0–2 | 기본 CNN (100 class) |
| ResNet-18 | Round 12 | 0–2 | ResNet-18 (device 16개) |
| ResNet-20 shallow, ResNet-20 middle, VGG-11 shallow, VGG-11 middle | Round 13b (S1 시나리오) | 0–2 | 같은 backbone의 두 분할 위치 |
| S1 replay, S2 replay | Round 15 frozen replay | 0–4, 0–2 | 기본 CNN, 학습 종료 모델 |
| S1 development | Round 10 (`runs/phaseT10_prior/r10_T40_s{5,6,7}`) | 5–7 | 기본 CNN. w 선택에만 씀 |

**없는 자료:** 기본 CNN이 아닌 설정의 development 기록이 없습니다. S1·S2를 뺀 설정은 frozen replay와 checkpoint가 없습니다. 하루를 넘는 기록도 없습니다.

## 2. 비교할 가중치

모든 방법은 DriftGate와 같은 보정(r = 0.5)을 쓰고, 같은 run의 같은 출력, 요청 순서, offload mask를 공유합니다.

**β와 offload mask**
- β = 1은 모든 요청을 offload합니다(주 진단).
- β = 0.5는 Round 15 C2의 controller로 offload합니다. offload되지 않은 요청에는 device exit의 답을 씁니다.

**배포할 수 있는 방법**
1. **DriftGate:** β = 1에서는 Round 13b의 F-auto 가중치를 쓰고, β = 0.5에서는 같은 mask의 DriftGate-P 가중치를 씁니다.
2. **공통 고정 w:** w ∈ {0, 0.05, …, 1.00}(21개)이고, w = 0은 corrected edge only입니다.
3. **development 공통 w:** S1 development seed 5–7에서 β = 1, 전체 구간의 seed 평균 정확도가 가장 높은 격자값입니다. 같으면 0.5에 가까운 값, 그다음 작은 값을 고릅니다. 이 값을 모든 설정과 두 β에 씁니다. Round 11에서 고른 w = 0.2도 따로 적습니다.
4. **모델·분할별 development w:** development 자료는 기본 CNN에만 있습니다. 그래서 기본 CNN 설정에서는 3과 같은 값이고, 다른 모델에서는 "자료 없음"으로 둡니다.
5. **초기 관측 후 고정 w (label-free):** 아래 순서로 정합니다.
   - device마다 (round, 도착) 순서로 offload된 요청을 셉니다.
   - 128개를 관측하기 전까지는 DriftGate의 가중치로 답합니다.
   - 128번째 요청에도 직전 history로 계산한 DriftGate 가중치를 씁니다.
   - 처음 128개 offload 요청의 보정하지 않은 entropy 평균으로 w = H̄_e/(H̄_d + H̄_e)를 계산합니다. 이 값을 129번째 요청부터 하루 끝까지 고정하며, 날짜나 이동에 따라 다시 정하지 않습니다.
   - offload 요청이 128개에 미치지 못한 device의 비율을 적습니다.

**사후 진단** (실현할 수 없는 기준이며, 표에서 따로 구분합니다)
6. **device 상수:** device의 전체 기록에서 구한 DriftGate 가중치의 평균을 상수로 씁니다.
7. **공통 고정 oracle:** run마다 전체 기록의 정답으로 고른 가장 좋은 격자값입니다.
8. **device별 고정 oracle:** device마다 전체 기록의 정답으로 고른 격자값입니다. 기준은 그 device가 논문 지표에 기여하는 값, 즉 round마다 acc_ek를 그 round의 device 수로 나눈 값의 합입니다.
9. **device와 시간대별 oracle:** device와 시간대마다 정답으로 고른 격자값입니다. 시간대는 평가 round의 시작 시각(300 + 6(r − 1)분)으로 나눈 다섯 구간(출근 전, 출근, 낮, 귀가, 저녁)이며, 결과와 무관하게 정했습니다. device와 시간대마다 요청 수를 함께 적습니다.
10. **다른 seed로 고른 모델별 w (leave-one-seed-out):** 같은 설정의 다른 평가 seed에서 가장 좋은 격자값을 골라 남은 seed에 씁니다. 독립 development 자료가 아니므로 배포할 수 있는 baseline으로 부르지 않습니다.

oracle(7–9)이 같은 동률이면 0.5에 가까운 값, 그다음 작은 값을 고릅니다.

## 3. 지표와 구간

- **주 지표:** 논문과 같습니다. 평가 round마다 요청이 있는 device의 정확도를 평균하고, 그 값을 round에 대해 평균합니다.
- **보조 지표:** 요청 수로 가중한 정확도(정답 수 ÷ 요청 수)와 오류 수를 따로 적습니다. 두 집계를 섞지 않습니다.
- **구간:** 전체, round > 30, round > 50(첫날 기록), frozen replay 전체입니다.
- **전환 device-round:** device의 cell 집합이 직전 평가 round와 다른 device-round입니다. 정확도 결과와 무관하게 위치 기록으로 정합니다. 전환 device-round와 나머지 device-round의 정확도를 따로 적습니다. 계산은 round마다 해당 device의 평균을 구하고, 그 값을 round에 대해 평균합니다.
- **전체 차이의 분해:** DriftGate − 각 방법의 전체 차이를 round ≤ 30과 round > 30의 기여로 나눕니다. 각 기여는 (그 구간의 round 수 ÷ 전체 round 수) × 구간 평균 차이입니다.
- **seed:** seed별 값을 먼저 계산합니다. 표에는 seed 평균(표준편차)을 적고, 차이는 같은 seed끼리 뺍니다.

## 4. 그림과 표

| 이름 | 내용 |
|---|---|
| 그림 1 | S1, S2, ResNet-18, 두 replay에서 정확도를 w에 따라 그린 곡선. β = 1 패널과 β = 0.5 패널을 따로 둡니다. DriftGate, development w, 초기 관측 후 고정 w를 표시합니다. 같은 형식의 모델·분할 그림도 만듭니다 |
| 그림 2 | 시간대마다 최고 정확도에서 0.1 pp 이내인 w 범위와 DriftGate의 평균 w. 아래 패널에 DriftGate − 그 시간대 최고 고정 w의 정확도 차이를 둡니다 |
| 표 1 | 전체, round > 30, round > 50, replay에서 DriftGate와 배포할 수 있는 고정 방법(development w, 모델별 w, 초기 관측 후 고정 w, corrected edge only)의 paired 차이 |
| 표 2 | 공통 고정 oracle, device별 oracle, device와 시간대별 oracle, device 상수, leave-one-seed-out w와 DriftGate의 차이 |

곡선에서 최고 정확도로부터 0.1 pp 이내인 w 범위를 적습니다. 0.1 pp는 곡선이 얼마나 평탄한지를 요약하는 기술적 기준이며, 통계적 유의성 기준이 아닙니다. DriftGate와 각 고정 방법 사이에서 고친 요청과 망가뜨린 요청의 비율, home/away와 Main/OOP/OOR의 정확도와 비중도 적습니다.

## 5. 상태 판정 (β = 1, seed 평균, 0.1 pp는 위와 같은 기술적 기준)

설정마다 다음 값을 구합니다.
- ΔI = DriftGate − 초기 관측 후 고정 w
- ΔG = DriftGate − 그 설정의 최고 공통 고정 격자값(seed 평균 곡선의 최댓값)
- L_dev = 최고 공통 고정 격자값 − development 공통 w
- H_time = device와 시간대별 oracle − device별 고정 oracle

각 상태는 독립적으로 판정하며, 둘 이상이 함께 성립하면 모두 적습니다.

| 상태 | 조건 |
|---|---|
| **A** (한 번 정한 가중치로 충분) | 모든 설정의 전체 구간에서 \|ΔI\| ≤ 0.1이고 ΔG ≤ 0.1 |
| **B** (모델·분할 사이의 차이가 중요) | (i) 모든 설정의 near-optimal 범위에 공통으로 들어가는 격자값이 없거나, 어떤 설정에서 L_dev > 0.1이고, (ii) 그 설정들에서 초기 관측 후 고정 w 또는 DriftGate가 최고 격자값보다 0.1 pp 넘게 낮지 않음 |
| **C** (같은 device의 시간 적응이 도움) | 어떤 설정에서 H_time > 0.1이고, ΔI > 0.1이 전체 구간과 전환 device-round에서 함께 성립 |

β = 0.5에서 같은 판정을 반복해 재현 여부를 적습니다. 판정에 따라 4절(분할 위치)과 5절(다일 frozen replay)의 후속 작업을 제안하거나, 기존 checkpoint로 할 수 있는 부분을 실행합니다. 새 학습은 제안과 비용 추정까지만 합니다.
