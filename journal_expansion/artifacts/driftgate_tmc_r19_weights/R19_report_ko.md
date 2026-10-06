# Round 19 보고서: 고정 가중치의 한계와 DriftGate 가중치의 실제 역할

## 0. 요약

이 보고서에서 w는 offload된 요청에서 device exit에 주는 가중치입니다. 학습의 personalization ratio λ와는 다릅니다. 모든 방법은 DriftGate와 같은 보정(r = 0.5)과 같은 offload mask를 씁니다.

- **질문 1 (모델·분할에 따라 적절한 w가 달라지는가): 그렇습니다.**
  - 최고 정확도에서 0.1 pp 이내인 w 범위(이하 near-optimal 범위)는 첫날 기록에서 K=500의 0.00–0.25부터 ResNet-20 shallow의 0.75–1.00까지 넓게 퍼져 있습니다. 모든 설정에 공통으로 들어가는 w는 없습니다.
  - development 자료로 고른 공통 w(0.45)는 설정마다 가장 좋은 고정 w보다 최대 1.30 pp 낮습니다.
- **질문 2 (같은 device에서도 시간에 따라 적절한 w가 달라지는가): 그렇습니다. 하지만 DriftGate의 w는 그 변화를 거의 따라가지 않습니다.**
  - 첫날 S1에서 시간대마다 최적 w는 출근 전 0.70–0.85에서 낮 0.00–0.10으로 내려갑니다. round > 30에서는 w = 0(corrected edge only)이 가장 높습니다.
  - DriftGate의 w는 하루 동안 0.46에서 0.36으로만 움직입니다.
  - device마다 DriftGate w의 평균을 상수로 써도 결과는 DriftGate와 −0.14 ~ +0.09 pp만 다릅니다.
- **질문 3 (DriftGate가 한 번 정한 w보다 운영 전체와 후반에서 나은가): 학습이 진행되는 첫날에만 낫습니다.**
  - 첫날에는 초기 관측 후 고정 w보다 0.1–0.5 pp 높습니다. 다만 이 고정값은 거의 학습되지 않은 첫 평가 round의 모델에서 정해져 약 0.50이 됩니다.
  - 모델이 고정된 재생에서는 DriftGate가 초기 관측 후 고정 w보다 낮습니다. 하루짜리 재생에서 −0.06 ~ −0.23 pp이고, 새 경로 세 날의 재생에서는 모든 seed에서 −0.05 ~ −0.23 pp입니다.
- **판정:** 결과 전에 정한 규칙으로는 상태 C만 성립했습니다(β = 1과 0.5 모두). 그러나 C를 확인하기 위해 실행한 세 날 frozen replay에서는 개선이 나타나지 않았습니다. 그래서 후반 이동에서 시간 적응의 이득은 주장하지 않습니다. B는 공통 고정값이 취약하다는 조건(i)은 성립했지만, 라벨 없이 그 차이를 메운다는 조건(ii)은 성립하지 않았습니다.
- **결론:** 추가 학습 실험은 정당화되지 않습니다. DriftGate의 가중치는 "라벨 없이 정해지는 기본값"으로 설명하고, 이득은 보정과 결합에 두는 것이 수치에 맞습니다(Round 18과 같은 방향).

## 1. 실행 전에 저장한 것과 자료 구분

- **`f56f501`:** 0단계의 계획, 정의, A/B/C 판정 규칙, 분석 스크립트(`R19_plan.md`, `scripts/r19_analysis.py`)
- **`2060276`:** 5절 세 날 frozen replay의 프로토콜, 새 날짜의 환경 파일, 실행·분석 스크립트(`R19_multiday_protocol.md`)
- **0단계 자료:** Round 18까지 이미 결과를 본 탐색 자료이며, 독립 검증 자료가 아닙니다.
- **세 날 replay:** 기존 checkpoint에 새 경로와 새 요청 수를 적용한 '새 경로 검증'입니다. 새 학습 seed 검증이 아닙니다.
- **development 자료:** S1 development seed 5–7(기본 CNN)뿐입니다. 이 자료에서 고른 공통 w는 0.45입니다(Round 11에서 고른 0.20도 함께 적었습니다). 다른 모델에는 development 자료가 없습니다. 그래서 모델별 development w는 기본 CNN에서만 정의되고, 다른 모델에서는 "자료 없음"입니다.
- **빠진 자료:** S1·S2를 뺀 설정은 checkpoint가 없어 frozen replay를 하지 못했습니다.

## 2. 0단계 결과 (β = 1, 전체 구간)

숫자는 seed 평균이고, 괄호 안은 같은 seed끼리 뺀 차이의 표준편차입니다. 전체 표는 `tables/R19_summary.csv`, `R19_table1_deployable.csv`, `R19_table2_oracles.csv`에 있습니다.

### 2.1 w에 따른 정확도와 DriftGate

| 설정 | near-optimal 범위 | 최고 고정 w | DriftGate 평균 w | DriftGate − 최고 고정 w | DriftGate − dev w 0.45 | DriftGate − 초기 관측 후 고정 w | DriftGate − corrected edge only |
|---|---|---|---|---|---|---|---|
| S1 | 0.20–0.45 | 0.35 | 0.42 | +0.06 | +0.11 | +0.19 | +0.45 |
| S2 | 0.50–0.60 | 0.55 | 0.38 | −0.11 | +0.02 | −0.12 | +1.21 |
| S1-fast | 0.20–0.40 | 0.30 | 0.42 | −0.02 | +0.10 | +0.23 | +0.31 |
| partial participation | 0.45–0.60 | 0.55 | 0.43 | −0.04 | +0.02 | −0.02 | +1.18 |
| stepwise change | 0.40–0.55 | 0.45 | 0.42 | +0.14 | +0.14 | +0.14 | +1.12 |
| random mobility | 0.40–0.55 | 0.45 | 0.45 | +0.12 | +0.12 | +0.14 | +1.21 |
| K=200 | 0.00–0.30 | 0.20 | 0.41 | −0.23 | +0.19 | +0.44 | −0.16 |
| K=500 | 0.00–0.25 | 0.10 | 0.41 | −0.45 | +0.22 | +0.53 | −0.42 |
| CIFAR-100 | 0.60–0.70 | 0.65 | 0.38 | −0.36 | −0.08 | −0.16 | +0.59 |
| ResNet-18 | 0.55–0.70 | 0.60 | 0.59 | −0.20 | +0.15 | −0.14 | +4.92 |
| ResNet-20 shallow | 0.75–1.00 | 0.90 | 0.55 | −0.80 | +0.50 | +0.07 | +7.11 |
| ResNet-20 middle | 0.70–1.00 | 0.95 | 0.55 | −0.22 | +0.10 | +0.01 | +1.28 |
| VGG-11 shallow | 0.35–0.45 | 0.40 | 0.45 | −0.08 | 0.00 | +0.41 | +1.26 |
| VGG-11 middle | 0.30–0.45 | 0.35 | 0.50 | −0.21 | −0.14 | +0.23 | +0.72 |
| S1 replay | 0.00–0.15 | 0.00 | 0.40 | −0.79 | +0.12 | −0.23 | −0.79 |
| S2 replay | 0.00–0.35 | 0.20 | 0.36 | −0.13 | +0.14 | −0.06 | −0.05 |

- 최고 고정 w는 결과를 보고 고른 값이므로 배포할 수 있는 방법이 아닙니다.
- 초기 관측 후 고정 w는 device마다 처음 128개 offload 요청의 entropy로 정한 뒤 하루 끝까지 유지한 값입니다.
- **곡선의 모양:** 그림 1과 보조 그림 1b를 보면 near-optimal 범위의 폭은 0.10–0.30입니다. 곡선은 최고점 근처에서 평탄하지만, 모델과 학습 단계에 따라 평탄한 위치가 옮겨 갑니다.
- **β = 0.5:** 결과의 방향은 같고 차이는 더 작습니다. 예를 들어 S1에서 DriftGate − 초기 관측 후 고정 w는 +0.13 pp이고, K=500에서 DriftGate − 최고 고정 w는 −0.36 pp입니다(`tables/R19_state.json`).

### 2.2 학습 후반 (round > 30)

| 설정 | 최고 고정 w | DriftGate − corrected edge only | DriftGate − 초기 관측 후 고정 w | DriftGate − dev w 0.45 |
|---|---|---|---|---|
| S1 | 0.00 | −0.55 | +0.30 | +0.08 |
| S1-fast | 0.00 | −0.67 | +0.35 | +0.09 |
| stepwise change | 0.00 | −0.48 | +0.29 | +0.07 |
| random mobility | 0.00 | −0.65 | +0.28 | +0.03 |
| K=200 | 0.00 | −1.06 | +0.58 | +0.21 |
| K=500 | 0.00 | −1.31 | +0.69 | +0.25 |
| S2 | 0.45 | +0.53 | −0.08 | −0.02 |
| partial participation | 0.35 | +0.23 | +0.06 | −0.01 |

기본 CNN의 이동 시나리오 6개에서는 round 30 이후 corrected edge only(w = 0)가 가장 높고, DriftGate는 그보다 0.48–1.31 pp 낮습니다.

**전체 차이의 분해:** S1 첫날에서 DriftGate − corrected edge only는 +0.45 pp입니다. 평가 round가 31개 가운데 7개뿐인 round ≤ 30 구간이 +0.87 pp를 더하고, round > 30 구간이 −0.43 pp를 뺀 결과입니다(`tables/R19_decomposition.csv`).

### 2.3 시간대별 적절한 w (그림 2)

S1 첫날에서 시간대마다 near-optimal 범위는 다음과 같이 바뀝니다.

| 시간대 | near-optimal 범위 | DriftGate 평균 w |
|---|---|---|
| 출근 전 | 0.70–0.85 | 0.46 |
| 출근 | 0.15–0.45 | 0.43 |
| 낮 | 0.00–0.10 | 0.42 |
| 귀가 | 0.00–0.20 | 0.39 |
| 저녁 | 0.25–0.50 | 0.36 |

DriftGate와 그 시간대의 최고 고정 w(사후 선택)의 차이는 출근 전에 −1.25 pp, 낮에 −0.85 pp입니다. S1 재생에서는 출근과 낮의 near-optimal 범위가 0.00–0.05인데, DriftGate의 w는 0.40 근처에 머뭅니다(`tables/R19_time_bins.csv`).

### 2.4 oracle (사후 진단, 표 2)

| 진단 | 결과 |
|---|---|
| 공통 고정 oracle − DriftGate | S1 −0.03, S1-fast +0.02. K=500 +0.45, ResNet-20 shallow +0.94, S1 replay +0.81 |
| device별 고정 oracle − 공통 고정 oracle (device 사이의 차이로 얻을 수 있는 여유) | +0.12 ~ +0.67 |
| device와 시간대별 oracle − device별 고정 oracle (시간 적응으로 얻을 수 있는 여유) | 첫날 기본 CNN(CIFAR-10) 설정 +0.64 ~ +1.15, CIFAR-100과 다른 모델 +0.19 ~ +0.78, 재생 +0.13 ~ +0.17. device와 시간대마다 요청은 약 7,000–15,000개 |
| device 상수 − DriftGate | −0.14 ~ +0.09 |

마지막 줄의 device 상수는 device마다 DriftGate w의 평균을 상수로 쓴 값입니다. 이 값이 DriftGate와 거의 같으므로, DriftGate가 같은 device 안에서 w를 바꾸면서 얻는 정확도는 작습니다.

### 2.5 다른 seed로 고른 모델별 w (leave-one-seed-out, 진단)

같은 설정의 다른 seed에서 고른 w를 남은 seed에 쓰면, DriftGate보다 다음만큼 높습니다.
- K=500 +0.45, ResNet-20 shallow +0.67, CIFAR-100 +0.35, K=200 +0.23, ResNet-18 +0.17, S1 replay +0.79 pp
- 반대로 S1, stepwise change, random mobility에서는 DriftGate가 각각 0.07, 0.17, 0.12 pp 높습니다.

이 w는 독립 development 자료로 고른 값이 아니므로 배포할 수 있는 baseline으로 부르지 않습니다.

## 3. 상태 판정 (`tables/R19_state.json`)

| 상태 | β = 1 | β = 0.5 | 근거 |
|---|---|---|---|
| A (한 번 정한 w로 충분) | 아님 | 아님 | DriftGate − 초기 관측 후 고정 w가 여러 설정에서 0.1 pp를 넘습니다(K=500 +0.53, S1 replay −0.23 등) |
| B (모델 사이의 차이가 중요) | 아님 | 아님 | (i)은 성립합니다. 공통 near-optimal w가 없고, dev w 0.45가 10개 설정에서 최고 고정 w보다 0.1 pp 넘게 낮습니다. (ii)는 성립하지 않습니다. DriftGate와 초기 관측 후 고정 w 모두 K=200, K=500, CIFAR-100, ResNet-20, 두 replay에서 최고 고정 w보다 0.2–0.8 pp 낮습니다 |
| C (시간 적응이 도움) | 성립 | 성립 | S1, S1-fast, random mobility, K=200, K=500, VGG-11 두 분할에서 DriftGate − 초기 관측 후 고정 w > 0.1이 전체와 전환 device-round에서 함께 성립했고, H_time > 0.1입니다 |

C가 성립한 설정은 모두 학습 중인 첫날 기록이고, 초기 관측 후 고정 w가 약 0.50인 설정입니다. 반면 모델이 고정된 하루 재생 두 개에서는 DriftGate가 초기 관측 후 고정 w보다 낮았습니다. 그래서 C의 이득이 이동을 따라간 결과인지 학습 진행을 따라간 결과인지 구분하기 위해 5절의 실험을 했습니다.

## 4. 5절: 새 경로 세 날의 frozen replay

**설정:** S1은 env seed를 바꾼 새 경로로, S2는 같은 GeoLife 사용자의 다른 평일로 세 날을 만들었습니다. 다른 평일이 부족해 날을 다시 쓴 사용자는 날짜별로 1, 3, 6명입니다. 모델은 Round 15의 학습 종료 checkpoint로 고정했고, window, controller, 초기 관측 후 고정 w는 날짜 사이에 이어서 썼습니다. 재생은 run마다 GPU에서 약 0.6분이 걸렸습니다(24 run).

| 시나리오 | β | 행 | DriftGate | − dev w 0.45 | − 초기 관측 후 고정 w | − corrected edge only | 사후 최고 고정 w |
|---|---|---|---|---|---|---|---|
| S1 (5 seed) | 1 | 세 날 전체 | 71.93 | +0.12 (0.12) | **−0.23 (0.07), 5/5 seed 음수** | −0.75 (0.42) | 0.00 |
| S1 | 1 | 둘째·셋째 날 | 72.07 | +0.11 | −0.23 | −0.72 | 0.00 |
| S1 | 1 | 전환 device-round | 68.33 | +0.12 | −0.32 | −1.12 | 0.00 |
| S1 | 1 | away | 55.95 | +0.22 | −0.52 | −2.47 | 0.00 |
| S1 | 0.5 | 세 날 전체 | 71.21 | +0.11 | −0.09 (5/5 음수) | −0.31 | 0.05 |
| S2 (3 seed) | 1 | 세 날 전체 | 76.33 | +0.15 | **−0.05 (0.01), 3/3 음수** | −0.10 | 0.20 |
| S2 | 1 | 전환 device-round | 68.87 | +0.06 | −0.20 | −0.11 | 0.25 |
| S2 | 0.5 | 세 날 전체 | 75.73 | +0.13 | −0.03 (3/3 음수) | −0.01 | 0.20 |

**판정:** 사전 기준은 DriftGate − X ≥ +0.1 pp이고 모든 seed에서 양수인 것입니다. 이 기준을 초기 관측 후 고정 w와 corrected edge only에 대해서는 어느 행에서도 만족하지 못했습니다. development w 0.45보다 높은 이유는 그 값이 고정 모델의 최적값(0.0–0.25)에서 멀기 때문이며, 시간 적응의 이득이 아닙니다.

**w의 변화:** DriftGate의 평균 w는 날짜마다 0.396, 0.395, 0.397(S1)로 거의 같습니다. 초기 관측 후 고정 w는 0.365(S1)와 0.345(S2)였습니다.

## 5. 4절: 분할 위치 (기존 기록으로 한 탐색)

Round 13b의 ResNet-20과 VGG-11은 S1 시나리오에서 shallow와 middle의 두 분할 위치를 각각 정상적으로 학습한 기록입니다(seed 0–2, 독립 배포).

| 구성 | near-optimal 범위 | DriftGate 평균 w | DriftGate − 최고 고정 w |
|---|---|---|---|
| ResNet-20 shallow | 0.75–1.00 | 0.55 | −0.80 |
| ResNet-20 middle | 0.70–1.00 | 0.55 | −0.22 |
| VGG-11 shallow | 0.35–0.45 | 0.45 | −0.08 |
| VGG-11 middle | 0.30–0.45 | 0.50 | −0.21 |

- **backbone에 따른 차이:** 같은 backbone 안에서는 near-optimal 범위가 크게 겹칩니다. backbone이 다르면 범위가 크게 다릅니다(ResNet-20 0.70–1.00, VGG-11 0.30–0.45).
- **development w의 이전:** CNN의 development w 0.45를 그대로 쓰면 ResNet-20 shallow에서 DriftGate보다 0.50 pp 낮습니다. 그러나 DriftGate도 그 구성의 최고 고정 w보다 0.80 pp 낮습니다.

**분할 위치 하나를 더하는 데 필요한 run:** 뒤쪽 분할 위치를 더하려면 새 학습이 필요합니다. ResNet-20과 VGG-11에 분할 위치 하나씩, seed 3개로 6 run이 필요합니다. Round 13b의 같은 크기 run은 run당 100–165분이 걸렸습니다(GPU 한 장). 위의 결과로 보면 이 실험은 "DriftGate가 별도 tuning 없이 각 구성의 최고 고정 w에 가깝다"는 주장을 뒷받침하지 않을 가능성이 큽니다. 그래서 실행을 권하지 않습니다.

## 6. 네 가지 질문에 대한 답

**1. 고정 가중치가 실제로 취약해지는 조건**
- **학습 단계:** 첫날 초반에는 큰 w가, round 30 이후와 학습 종료 모델에서는 w ≈ 0이 유리합니다.
- **모델 구성:** near-optimal 범위가 0.00–0.25(K=500)에서 0.75–1.00(ResNet-20 shallow)까지 다릅니다.
- **크기:** 공통 development w 0.45는 설정에 따라 최고 고정 w보다 최대 1.30 pp(ResNet-20 shallow), 0.91 pp(S1 replay), 0.67 pp(K=500) 낮습니다.

**2. DriftGate가 유리한 이유**
- **보정:** 가장 큰 몫입니다(Round 18). 첫날의 S1, S2, S1-fast, partial participation, stepwise change, random mobility에서 DriftGate − corrected edge only는 +0.31 ~ +1.21 pp입니다. 이 차이는 큰 w가 유리한 학습 초반에서 주로 생깁니다(2.2절의 분해).
- **자동 설정:** 라벨 없이 정해지는 w가 기본 CNN의 이동 시나리오(S1, S1-fast, stepwise change, random mobility, partial participation, S2)에서는 사후 최고 고정 w와 −0.11 ~ +0.14 pp 이내입니다. 하지만 K=200, K=500, CIFAR-100, ResNet 계열, 재생에서는 0.2–0.8 pp 낮습니다.
- **시간 적응:** 이득을 확인하지 못했습니다. DriftGate의 w는 시간대에 따른 최적 w의 변화를 거의 따라가지 않습니다. 모델이 고정되면 한 번 정한 w보다 낮습니다. 첫날에 초기 관측 후 고정 w보다 높은 것은 그 고정값이 학습 전 모델로 정해진 탓입니다.

**3. 전체와 후반의 정확도를 질문받았을 때 답할 수 있는 수치 (β = 1)**
- **S1 첫날 전체:** DriftGate 68.31%입니다. 최고 고정 w보다 +0.06, development w 0.45보다 +0.11, corrected edge only보다 +0.45 pp입니다.
- **S1 round > 30:** DriftGate 67.10%이고, corrected edge only보다 −0.55 pp입니다.
- **S1 학습 종료 모델 재생:** 71.95%이고, corrected edge only보다 −0.79 pp, 초기 관측 후 고정 w보다 −0.23 pp입니다.
- **S1 새 경로 세 날 재생:** 71.93%이고, 초기 관측 후 고정 w보다 −0.23 pp, corrected edge only보다 −0.75 pp입니다.
- **S2:** 첫날 전체에서는 corrected edge only보다 +1.21 pp이고, 세 날 재생에서는 −0.10 pp입니다.
- 이 차이들은 같은 seed끼리 뺀 값의 평균입니다. 표준편차는 표에 있습니다.

**4. 추가 학습이 정당화되는가**
- 정당화되지 않습니다. 기존 방법과 원고를 정리할 단계입니다.
- 원고에서 DriftGate의 가중치는 별도 tuning 없이 정해지는 기본값으로 설명합니다. 기본 CNN의 첫날에서는 가장 좋은 공통 상수에 가깝지만, 학습 후반과 학습 종료 모델에서는 보정한 edge만 쓰는 쪽(w = 0)이 높다는 점을 함께 적습니다.
- 시간 적응이나 이동 추적을 기여로 쓰지 않습니다.

## 7. 원고 프레이밍에 반영할 점

"학습 상태, serving cell, 모델 구성이 달라지는 동안 device와 edge의 예측을 어떻게 결합해야 운영 전 구간의 정확도를 높일 수 있는가"는 검증하려는 질문으로 둘 수 있습니다. 이번 결과로 확인된 답은 다음과 같습니다.

- **보정한 결합의 효과:** 학습 초반을 포함한 첫날 전체에서 확인됩니다.
- **가중치의 역할:** 학습 단계와 모델 구성에 따라 적절한 w가 크게 달라지지만, 현재의 entropy 비율은 그 변화를 따라가지 않습니다.
- **수치를 쓰는 방식:** 운영 전체의 정확도는 첫날 전체(평가 round 가중)로 쓰되, 초반 이득과 후반 손실을 분해해 함께 제시합니다(2.2절). confidence baseline 대비 절감률과 strongest corrected baseline 대비 차이를 서로 바꿔 쓰지 않습니다.

## 8. 산출물

| 파일 | 내용 |
|---|---|
| `R19_plan.md`, `R19_multiday_protocol.md` | 결과 전에 저장한 정의, 판정 규칙, 프로토콜 |
| `tables/R19_weight_curves.csv`, `_per_seed.csv` | 고정 w 곡선 (설정 × β × 구간), near-optimal 범위, DriftGate와 초기 관측 후 고정 w |
| `tables/R19_table1_deployable.csv` | 표 1: 배포할 수 있는 방법과의 paired 차이, 고친 요청과 망가뜨린 요청, 요청 수 가중 정확도, 오류 수 |
| `tables/R19_table2_oracles.csv` | 표 2: 공통, device, device × 시간대 oracle과 device 상수 |
| `tables/R19_leave_one_seed_out.csv`, `R19_splits.csv`, `R19_decomposition.csv`, `R19_time_bins.csv`, `R19_summary.csv`, `R19_state.json` | 진단, home/away와 Main/OOP/OOR, 구간 기여, 시간대, 요약, 판정 |
| `tables/R19_multiday_table.csv`, `_curves.csv`, `_per_seed.csv`, `R19_multiday_verdict.json` | 세 날 frozen replay |
| `figures/R19_fig1_accuracy_vs_w_{main,models}.*`, `R19_fig1b_paired_difference.*`, `R19_fig2_time_bins.*` | 그림 1, 보조 그림 1b(같은 seed끼리 뺀 차이, 결과를 본 뒤 가독성을 위해 추가), 그림 2 |
| `scripts/` | `r19_analysis.py`, `r19_multiday_env.py`, `r19_multiday_replay.py`, `r19_multiday.py`, `r19_summary.py` |

재생 기록(`runs/phaseT19_multiday/*_evalprobs.npz`)은 서버에만 있습니다. 환경 파일과 S2의 날짜 선택표는 commit했습니다.
