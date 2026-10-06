# Round 19.1 통합 보고서: Round 17–19의 결과 정리

이 문서는 Round 17, 18, 19의 결과와 보고서를 하나로 묶은 것입니다. 새 실험은 하지 않았습니다. 수치는 세 Round에서 commit한 표와 판정 파일을 `scripts/r19_1_integrate.py`로 읽어 만든 통합 표(`tables/R19_1_*.csv`)에서 가져왔습니다. 원천 파일과 그 sha256, 마지막 commit은 `tables/R19_1_sources.csv`에 있습니다.

표의 값은 seed 평균(표준편차)이고, 차이는 같은 seed끼리 뺀 값의 평균(표준편차)입니다. 단위는 pp입니다. 정확도는 논문과 같이 평가 round마다 요청이 있는 device의 정확도를 평균한 뒤, 그 값을 round에 대해 평균해 구했습니다.

## 0. 결론

1. **이득의 근원:** DriftGate의 이득은 대부분 edge exit의 보정(prior correction)에서 나옵니다. 같은 offloading budget에서 보정하지 않은 결합 규칙보다 +0.60 ~ +1.88 pp 높습니다(첫날 CNN 설정 7개). 그러나 보정한 고정 가중치 규칙과의 차이는 −0.79 ~ +0.15 pp입니다.
2. **실패한 후보:** 세 Round에서 시험한 새 후보는 모두 기존 DriftGate를 넘지 못했습니다. 요청 종류 탐지기, 조건부 결합, BTFL 적응, 가중치의 시간 적응이 여기에 해당합니다.
3. **가중치 w의 역할:** DriftGate의 w는 offload된 요청에서 device exit에 주는 비중입니다. 이 w는 라벨 없이 정해지는 기본값이며, 조건 변화를 따라가는 적응 장치가 아닙니다.
   - 적절한 w는 모델 구성과 학습 단계에 따라 0에서 1 사이로 크게 달라집니다.
   - DriftGate의 w는 하루 내내 0.36–0.46에 머뭅니다.
   - 보정한 edge만 쓰는 쪽(corrected edge only, w = 0)이 학습 후반(round > 30)에는 DriftGate보다 0.48–1.31 pp 높고, 학습 종료 모델의 재생에서는 S1 0.79 pp, S2 0.05 pp 높습니다.
4. **원고에서 고칠 문장:** 원고의 offloading 문장("7개 설정에서 14–52% offloading으로 모든 full-offload 규칙을 넘는다")은 Round 15의 7개 규칙에 대해서만 성립합니다. 보정 변형을 비교 집합에 넣으면 5개 설정, 43–87%로 바뀝니다.
5. **다음 단계:** 추가 학습 실험은 정당화되지 않습니다. 남은 일은 원고를 정리하고 모바일에서 실측하는 것입니다.

## 1. 세 Round의 범위

| Round | 질문 | 방법 | 자료 | 결과 전 commit | 판정 |
|---|---|---|---|---|---|
| 17 | device의 자기 학습 데이터와의 feature 거리로 요청이 Main인지 가려 결합을 바꾸면 나아지는가 | 탐지기 D1–D4, gate 규칙 R-b/R-a/R-c/R-s | Round 15 checkpoint의 S1·S2 frozen replay(재생 기록을 feature와 함께 다시 만듦) | `5cce14e` | `NO_GO` |
| 18 | 비교 공백(보정 변형, 같은 budget, offloading 문장)을 메우고, 기존 출력으로 시험할 수 있는 결합 변형 하나가 기여를 강화하는가 | 같은 budget 비교, offloading 문장 재계산, 진단, conditional combination, BTFL 적응 | 첫날 10개 설정과 S1·S2 frozen replay의 기존 기록 | `45d250d` | 후보 실패 |
| 19 | 고정 가중치는 언제 취약하고, DriftGate의 w는 그 변화를 따라가는가 | 고정 w 격자, 라벨 없이 한 번 정한 w, oracle, 새 경로 세 날 frozen replay | 16개 설정(모델·분할 4개 포함)과 세 날 재생 24 run | `f56f501`, `2060276` | 시간 적응의 이득 없음 |

- **자료의 성격:** 세 Round 모두 새 학습은 하지 않았습니다. 평가에 쓴 seed와 요청은 이미 결과를 본 탐색 자료입니다.
- **세 날 replay의 성격:** Round 19의 세 날 replay는 기존 checkpoint에 새 경로를 적용한 검증이며, 새 학습 seed의 검증이 아닙니다.
- **원래 보고서:** `driftgate_tmc_r17_featgate/R17_stage0_report_ko.md`, `driftgate_tmc_r18_compare/R18_final_report_ko.md`, `driftgate_tmc_r19_weights/R19_report_ko.md`

## 2. 요청 종류를 가리는 탐지기는 결합을 개선하지 못했다 (Round 17, 18)

Round 17은 device가 가진 own class 학습 데이터를 기준으로 요청이 Main인지 판정하는 탐지기 네 개를 시험했습니다(`tables/R19_1_T1_detectors.csv`).

| 탐지기 | 정의 | 계산할 수 있는 곳 |
|---|---|---|
| D1 | device block feature f_d의 Mahalanobis 거리 | device (offload 전) |
| D2 | f_d의 k-NN cosine 거리(k = 10) | device (offload 전) |
| D3 | edge가 학습해 device로 보내는 로지스틱 head | device (offload 전) |
| D4 | edge block feature f_e의 class 원형 비교 | edge (offload 후에만) |

| 설정 | 점수 | Main 대 non-Main AUROC (전체 요청) | D4가 정의된 요청에서의 AUROC | 고친 요청 대 망가뜨린 요청의 AUROC | R-b(점수) − 17개 규칙 중 최고 |
|---|---|---|---|---|---|
| S1 재생 | D1 | 0.665 | 0.663 | 0.600 | −0.19 |
| S1 재생 | D2 | 0.716 | 0.721 | 0.616 | −0.30 |
| S1 재생 | D3 | 0.701 | 0.700 | 0.551 | −0.48 |
| S1 재생 | D4 | 0.838 (점수 정의 비율 0.90) | 0.840 | 0.622 | −0.06 |
| S1 재생 | device entropy | 0.701 | 0.707 | 0.407 | |
| S2 재생 | D1 | 0.657 | 0.657 | 0.618 | +0.10 |
| S2 재생 | D4 | 0.864 (점수 정의 비율 0.93) | 0.864 | 0.612 | +0.05 |
| S2 재생 | device entropy | 0.707 | 0.711 | 0.411 | |

AUROC는 수신자 조작 특성 곡선 아래 면적이고, R-b(D)는 모든 요청을 offload한 뒤 탐지기가 non-Main으로 판정한 요청만 보정한 edge로 답하는 규칙입니다.

- **판정:** Round 17의 사전 기준은 D1이나 D3의 AUROC ≥ 0.80이고, 1차 후보 R-b(D4)가 strongest reference보다 +0.5 pp 이상 높은 것이었습니다. 두 기준 모두 S1·S2에서 미달해 `NO_GO`입니다.
- **Main 판정과 답의 정답 여부는 다른 문제입니다(Round 18 진단).**
  - D4는 Main 요청을 가장 잘 가립니다(AUROC 0.84–0.86).
  - 그러나 DriftGate의 답을 corrected edge only로 바꿨을 때 고쳐지는 요청과 망가지는 요청은 거의 가리지 못합니다(AUROC 0.61–0.62).
  - S1 첫날에 답을 바꾸면 고쳐지는 요청이 2.11%, 망가지는 요청이 2.56%입니다. 두 답이 다른 요청은 6.6%입니다.
  - 두 답 가운데 정답을 고르는 oracle은 DriftGate보다 +0.7 ~ +3.3 pp 높습니다. 이 값은 실현 가능한 성능이 아닙니다.
- **learned weight의 학습 목표:** 기존 learned weight는 요청이 Main인지 맞히도록 학습했습니다. 어느 답이 맞는지를 학습한 것이 아닙니다.
- **D3의 한계:** 음성 표본이 다른 device의 block(h_j)으로 만든 feature이므로, 대상 device의 block(h_k)에서 생기는 분포와 다릅니다.

## 3. 이득은 보정에서 나오고, adaptive w의 몫은 작다 (Round 18)

### 3.1 같은 offloading budget

모든 규칙에 Round 15 C2 controller가 만든 같은 offload mask를 적용했습니다. β는 controller가 목표로 하는 offloading 비율입니다(`tables/R19_1_T2_same_budget.csv`).

| 설정 | β | DriftGate-P | − 보정하지 않은 규칙 중 최고 | − 보정한 고정 w 규칙 중 최고 | 그 규칙 |
|---|---|---|---|---|---|
| S1 | 0.5 | 67.73 | +0.66 (0.36) | +0.05 (0.09) | correction + w 0.2 |
| S1 | 1 | 68.31 | +0.95 (0.57) | +0.15 (0.15) | correction + w 0.2 |
| S2 | 1 | 71.70 | +1.63 (1.04) | −0.07 (0.13) | correction + w 0.5 |
| partial participation | 1 | 64.82 | +1.88 (0.59) | −0.03 (0.08) | correction + w 0.5 |
| stepwise change | 1 | 72.36 | +0.92 (0.26) | +0.14 (0.09) | correction + w 0.5 |
| random mobility | 1 | 71.09 | +1.52 (0.53) | +0.14 (0.06) | correction + w 0.5 |
| CIFAR-100 | 1 | 41.29 | +1.46 (0.37) | −0.45 (0.08) | correction + product |
| ResNet-18 | 1 | 66.64 | +0.18 (0.29) | −0.07 (0.09) | correction + w 0.5 |
| K=500 | 1 | 70.11 | −0.49 (0.22) | −0.42 (0.12) | corrected edge only |
| S1 재생 | 1 | 71.95 | −0.52 (0.47) | −0.79 (0.41) | corrected edge only |
| S2 재생 | 1 | 76.54 | +0.64 (0.81) | −0.13 (0.15) | correction + w 0.2 |

- **요청당 edge 호출 수:** 두 cell에 속한 요청은 두 번으로 셉니다. β = 0.5에서 0.53–0.98, β = 1에서 1.06–1.97입니다.
- **adaptive w의 몫:** DriftGate-P가 보정한 고정 w 규칙보다 0.05 pp 이상 높은 경우는 S1, stepwise change, random mobility뿐이며, 차이는 +0.05 ~ +0.15 pp입니다.

### 3.2 원고의 offloading 문장 (`tables/R19_1_T3_offloading_claims.csv`)

| full-offload 비교 집합 | 최고 정확도에 도달하는 첫날 설정 | 필요한 offloading | confidence-based offloading 대비 offload 요청 감소 |
|---|---|---|---|
| Round 15의 7개 규칙 | 7개 (S1, S2, S1-fast, partial participation, stepwise change, random mobility, ResNet-18) | 0.136–0.524 (그중 0.5 이하는 5개) | 43–86% |
| 보정 변형 6개를 더한 13개 규칙 | 5개 (S1, S1-fast, stepwise change, random mobility, ResNet-18) | 0.433–0.870 | 5–20% |

- 보정 변형을 넣으면 S2, partial participation, CIFAR-100, K=200, K=500에서는 어떤 threshold로도 도달하지 못합니다. 이 설정들에서는 DriftGate의 full-offload 정확도도 보정 변형보다 낮습니다.
- edge 호출 수로 센 감소율도 요청 수로 센 감소율과 거의 같습니다(5–20%).
- 43–86%는 기본 confidence threshold와 비교한 offload 요청 수의 감소입니다. 모든 방법의 최선 budget과 비교한 통신·지연·에너지 절감이 아닙니다.

## 4. 가중치 w는 조건 변화를 따라가지 않는다 (Round 19)

### 4.1 모델과 학습 단계에 따른 적절한 w

β = 1, 첫날 전체 구간입니다(`tables/R19_1_T4_weights.csv`). 최고 고정 w와 near-optimal 범위는 결과를 보고 고른 값입니다. near-optimal 범위는 최고 정확도에서 0.1 pp 이내인 w의 범위입니다. 초기 관측 후 고정 w는 device마다 처음 128개 offload 요청의 entropy로 정한 뒤 하루 끝까지 유지한 값입니다.

| 설정 | near-optimal 범위 | DriftGate 평균 w | DriftGate − 최고 고정 w | DriftGate − dev w 0.45 | DriftGate − 초기 관측 후 고정 w | DriftGate − corrected edge only |
|---|---|---|---|---|---|---|
| S1 | 0.20–0.45 | 0.42 | +0.06 | +0.11 | +0.19 | +0.45 |
| S2 | 0.50–0.60 | 0.38 | −0.11 | +0.02 | −0.12 | +1.21 |
| K=500 | 0.00–0.25 | 0.41 | −0.45 | +0.22 | +0.53 | −0.42 |
| CIFAR-100 | 0.60–0.70 | 0.38 | −0.36 | −0.08 | −0.16 | +0.59 |
| ResNet-18 | 0.55–0.70 | 0.59 | −0.20 | +0.15 | −0.14 | +4.92 |
| ResNet-20 shallow | 0.75–1.00 | 0.55 | −0.80 | +0.50 | +0.07 | +7.11 |
| VGG-11 shallow | 0.35–0.45 | 0.45 | −0.08 | 0.00 | +0.41 | +1.26 |
| S1 재생 | 0.00–0.15 | 0.40 | −0.79 | +0.12 | −0.23 | −0.79 |
| S2 재생 | 0.00–0.35 | 0.36 | −0.13 | +0.14 | −0.06 | −0.05 |

- **모든 설정에 맞는 w는 없습니다.** 공통으로 near-optimal인 w가 없습니다. S1 development seed로 고른 공통 w 0.45는 최고 고정 w보다 최대 1.30 pp(ResNet-20 shallow) 낮습니다.
- **학습 후반에는 w = 0이 가장 높습니다.** round > 30에서는 S1, S1-fast, stepwise change, random mobility, K=200, K=500에서 그렇고, DriftGate는 그보다 0.48–1.31 pp 낮습니다.
  - S1 첫날 전체에서 DriftGate − corrected edge only는 +0.45 pp입니다.
  - 이 값은 평가 round 7개뿐인 round ≤ 30 구간의 +0.87 pp와 round > 30 구간의 −0.43 pp를 합친 결과입니다.
- **시간대마다 적절한 w가 크게 바뀝니다(사후 진단, `tables/R19_1_T4b_time_bins.csv`).**
  - S1 첫날: 출근 전 0.70–0.85, 낮 0.00–0.10, 귀가 0.00–0.20
  - DriftGate의 평균 w: 0.46, 0.42, 0.39
  - S1 재생의 출근과 낮: near-optimal 범위 0.00–0.05, DriftGate w 0.40 근처
- **DriftGate가 w를 바꾸며 얻는 몫은 작습니다.** device마다 DriftGate w의 평균을 상수로 쓰면 DriftGate와 −0.14 ~ +0.09 pp만 다릅니다. 반면 device와 시간대마다 w를 고르는 oracle은 device마다 w를 하나 고르는 oracle보다 0.13–1.15 pp 높습니다.

### 4.2 시간 적응의 이득: 학습 단계를 따라간 결과였다

- **0단계 판정:** 사전 규칙에서는 상태 C(같은 device의 시간 적응이 도움)가 성립했습니다. 성립한 설정은 S1, S1-fast, random mobility, K=200, K=500, VGG-11이고, 모두 학습 중인 첫날 기록입니다.
- **판정의 원인:** 첫날의 초기 관측 후 고정 w는 거의 학습되지 않은 모델로 정해져 약 0.50이 됩니다. DriftGate는 이 값보다 나았을 뿐입니다.
- **세 날 frozen replay:** 학습이 끝난 모델로 새 경로 세 날을 재생하면 결과가 반대가 됩니다(`tables/R19_1_T5_three_day_replay.csv`).

| 시나리오 | β | 행 | DriftGate | − 초기 관측 후 고정 w | DriftGate가 높은 seed | − corrected edge only | − dev w 0.45 |
|---|---|---|---|---|---|---|---|
| S1 | 1 | 세 날 전체 | 71.93 | −0.23 (0.07) | 0/5 | −0.75 | +0.12 |
| S1 | 1 | 전환 device-round | 68.33 | −0.32 | 0/5 | −1.12 | +0.12 |
| S1 | 1 | away | 55.95 | −0.52 | 0/5 | −2.47 | +0.22 |
| S2 | 1 | 세 날 전체 | 76.33 | −0.05 (0.01) | 0/3 | −0.10 | +0.15 |
| S2 | 1 | 전환 device-round | 68.87 | −0.20 | 0/3 | −0.11 | +0.06 |

사전 기준(DriftGate − X ≥ +0.1 pp이고 모든 seed에서 양수)을 초기 관측 후 고정 w와 corrected edge only에 대해서는 어느 행에서도 만족하지 못했습니다. 그러므로 후반 이동에서 시간 적응의 이득은 주장하지 않습니다.

## 5. 그 밖의 후보와 선행연구 비교 (Round 18)

`tables/R19_1_T6_candidates.csv`에 네 후보의 판정을 모았습니다.

- **조건부 결합 (conditional combination)**
  - 정의: edge가 정한 own-class 확률 합을 유지하고, own class 안에서만 device와 edge를 섞습니다.
  - 판정: 지정한 56행 가운데 β = 1에서 3행, β = 0.5에서 9행만 통과해 엄격 판정에서 실패했습니다. 독립 검증 후보도 아닙니다.
  - 큰 손실: S2 첫날 −0.87 pp, ResNet-18 −4.20 pp, S1 출근 전 −2.35 pp
  - 오류 분해: 'non-Main을 Main으로 예측한 오류'가 줄고, 'Main을 non-Main으로 예측한 오류'가 늘었습니다.
- **BTFL inference adaptation**
  - BTFL은 모든 client가 공유하는 하나의 feature extractor를 가정합니다. 우리 split 모델은 device마다 block이 달라 이 가정이 깨지므로, 이 차이를 명시한 adaptation으로 계산했습니다.
  - 결과: 재생에서 DriftGate보다 S1 −1.77 pp, S2 −2.38 pp 낮습니다.
  - 원문과 코드의 차이 8가지와 faithful 비교에 필요한 재학습 조건은 `driftgate_tmc_r18_compare/R18_BTFL_compatibility.md`에 있습니다.

## 6. Main 비중에 따른 민감도 (Round 18)

β = 1의 예측을 고정한 채, device-round마다 Main 비중 s를 바꿔 다시 가중했습니다(`tables/R19_1_T7_main_share.csv`). S1에서 corrected edge only − DriftGate는 다음과 같습니다.

| Main 비중 s | 0.3 | 0.5 | 0.65 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| corrected edge only − DriftGate (pp) | +2.54 | +1.14 | +0.09 | −0.96 | −1.66 |

관측된 Main 비중은 0.65–0.74입니다. 이 결과는 고정된 예측을 다시 가중한 것이며, controller가 바뀐 요청 구성에 적응한 결과가 아닙니다.

## 7. 정정 사항

1. **strongest other의 비교 집합 (Round 17 → 18):** Round 17은 strongest other를 11개 규칙에서 골라, Round 15 B1의 보정 변형 5개가 빠졌습니다. 17개 규칙으로 다시 고르면 S2 재생의 기준은 correction + development w 0.2(76.66%)이고, R-b(D4)의 차이는 +0.12에서 +0.05 pp로 줄어듭니다. 판정(`NO_GO`)은 바뀌지 않습니다. 이후 표는 17개 규칙을 씁니다.
2. **seed 집합:** Round 17 표의 S1 재생 DriftGate 71.82%는 판정용 seed 1–4의 평균이고, Round 15의 71.95%는 seed 0–4의 평균입니다. seed별 값은 같습니다.
3. **D4의 위치:** D4는 edge의 f_e를 쓰므로 offload 전에 판정할 수 없습니다. Round 17에서도 모든 요청을 offload하는 규칙에만 썼습니다.
4. **AUROC 계산 방식:** Round 17의 AUROC는 run마다 요청을 모아 계산했으므로 device-round 가중치를 쓰지 않았습니다. D4의 값은 점수가 정의된 요청에서만 계산했습니다.
5. **offloading 문장:** 3.2절과 같이 비교 집합을 명시해야 합니다. 0.5 이하에서 성립하는 설정은 7개가 아니라 5개입니다.
6. **두 cell의 처리 경로와 44 B:** 시뮬레이터는 두 edge의 logit을 먼저 평균하고, 그 확률에 보정을 한 번 적용합니다. cell마다 보정한 확률을 평균하는 계산과는 다릅니다. 원고의 44 B는 결합을 맡은 한 endpoint가 돌려주는 payload입니다. edge 사이의 전송과 두 번째 호출은 따로 세야 합니다.

## 8. 원고에 쓸 주장과 쓰지 않을 주장

**쓸 수 있는 주장** (`tables/R19_1_claims.csv`의 C1–C8)

| ID | 주장 | 수치와 조건 |
|---|---|---|
| C1 | edge exit를 보정한 뒤 결합하면, 같은 offloading 비율에서 보정하지 않은 결합 규칙보다 정확하다 | 첫날 CNN 설정 7개, β = 0.5와 1에서 +0.60 ~ +1.88 pp |
| C2 | adaptive weight는 보정한 고정 weight와 비슷하다 (모델마다 따로 조정하지 않고 쓰는 기본값) | 차이 −0.79 ~ +0.15 pp |
| C3 | 보정하지 않은 full-offload 규칙들과 비교하면, 적은 offloading으로 그 최고 정확도에 도달한다 | 첫날 7개 설정, 0.136–0.524 (0.5 이하는 5개), confidence-based 대비 43–86% 감소 |
| C4 | 보정 변형까지 넣으면 도달하는 설정과 절감 폭이 줄어든다 | 5개 설정, 0.433–0.870, 5–20% 감소 |
| C5 | 학습 후반에는 보정한 edge만 쓰는 쪽이 더 정확하다 | round > 30, CNN 이동 설정 6개에서 DriftGate가 −0.48 ~ −1.31 pp |
| C6 | 첫날 전체의 이득은 학습 초반에서 주로 생긴다 | S1에서 corrected edge only 대비 +0.45 pp = 초반 +0.87 − 후반 0.43 |
| C7 | 학습 종료 모델로 이동이 계속되면 한 번 정한 w보다 낫지 않다 | 세 날 frozen replay에서 S1 −0.23, S2 −0.05 pp, 모든 seed 음수 |
| C8 | 학습 종료 모델에서는 corrected edge only가 DriftGate보다 높거나 같다 | 재생에서 S1 −0.79, S2 −0.05 pp |

**쓰지 않을 주장**
- DriftGate의 가중치가 시간, 이동, 학습 단계에 적응해 이득을 낸다.
- 학습된 모델에서 DriftGate가 일반적으로 가장 정확하다.
- 요청 종류 탐지나 출력 기반 신호로 결합을 더 개선할 수 있다(Round 16–18의 시도가 모두 실패했습니다).
- confidence baseline 대비 감소율을, strongest corrected baseline 대비 결과와 바꿔 쓰는 것.
- BTFL과 FedTHE·ZTW의 행을 원 방법 전체의 성능이라고 쓰는 것.

## 9. 남은 일

1. **원고 수정 (새 run 없음):** 7절의 정정 사항과 8절의 주장 범위를 반영합니다. 평가 결과는 첫날 전체를 주 지표로 두고, 초반 이득과 후반 손실을 분해해 함께 제시합니다.
2. **모바일 실측:** 장비가 준비되면 β = 0.25–1의 같은 요청 stream에서 측정합니다. 지연(평균과 p95), 요청당 에너지, uplink와 downlink byte, edge 호출 수를 재고, 한 cell과 두 cell의 경로를 나눕니다.
3. **선택 사항 (권하지 않음):** 결론을 바꾸기 위한 것이 아니라 비교를 보완하기 위한 run입니다.
   - 분할 위치 하나 추가: ResNet-20과 VGG-11에 각 한 위치, seed 3개로 6 run, run당 100–165분
   - faithful BTFL 비교: 전역으로 공유하는 device block으로 다시 학습
   - Main 비중을 약 0.5로 바꾼 요청 stream으로 S1 재생 5 run: 기존 checkpoint를 쓰며 run당 약 1분

## 10. 산출물

| 파일 | 내용 |
|---|---|
| `scripts/r19_1_integrate.py` | 세 Round의 표와 판정 파일을 읽어 통합 표를 만드는 스크립트 (재현: `python journal_expansion/artifacts/driftgate_tmc_r19_1_integrated/scripts/r19_1_integrate.py`) |
| `tables/R19_1_T1_detectors.csv` | 탐지기: AUROC, D4 정의 범위, 고친·망가뜨린 요청의 구별, R-b 차이 |
| `tables/R19_1_T2_same_budget.csv` | 같은 budget에서 보정하지 않은 규칙과 보정한 규칙 대비 차이 |
| `tables/R19_1_T3_offloading_claims.csv` | offloading 문장의 두 비교 집합 |
| `tables/R19_1_T4_weights.csv`, `R19_1_T4b_time_bins.csv` | 가중치: 구간별 차이와 시간대별 적절한 w |
| `tables/R19_1_T5_three_day_replay.csv` | 세 날 frozen replay |
| `tables/R19_1_T6_candidates.csv` | Round 17–19 후보의 판정 |
| `tables/R19_1_T7_main_share.csv` | Main 비중 민감도 |
| `tables/R19_1_claims.csv` | 원고에 쓸 주장 C1–C8의 수치 |
| `tables/R19_1_sources.csv` | 원천 파일, sha256, 마지막 commit |

세 Round의 원래 산출물과 보고서는 각 디렉터리에 그대로 두었습니다(`driftgate_tmc_r17_featgate`, `driftgate_tmc_r18_compare`, `driftgate_tmc_r19_weights`). 결과 commit은 Round 17 `ffb6c50`, Round 18 `b8e252b`, Round 19 `a463578`입니다.
