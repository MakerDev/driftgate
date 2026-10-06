# Round 18 A1: Round 17 비교와 평가 단위의 감사

이 문서는 Round 17 코드(`driftgate_tmc_r17_featgate/scripts/r17_stage0.py`, `r17_extract.py`)와 표를 다시 확인한 결과입니다. 수치는 `tables/R18_R17_corrected_margins.csv`와 `tables/R18_R17_seed_sets.csv`(`scripts/r18_audit.py`)에서 가져왔습니다. Round 15 재현 확인은 Round 17의 결과를 그대로 재사용했고, 재생 기록을 다시 만들지 않았습니다.

## 1. Round 17 탐지기와 규칙의 정의 (코드 기준)

| 이름 | 입력 | 통계와 문턱 | 점수를 계산할 수 있는 곳 | offload 전에 판정할 수 있는가 |
|---|---|---|---|---|
| D1 | f_d(x): device block 출력의 공간 평균 (128) | own class 평균, class 중심을 뺀 공유 공분산(÷n) + ε·I (ε = 10⁻³·trace/128). 점수는 own class까지의 최소 Mahalanobis 거리. 문턱은 학습 표본 점수의 (1 − α) 분위수, α = 0.10 | device | 가능 |
| D2 | f_d(x), L2 정규화 | 학습 표본 feature 가운데 10번째 이웃과의 cosine 거리. 학습 표본의 점수는 leave-one-out. 문턱은 D1과 같은 규칙, α = 0.10 | device (학습 표본 feature 약 1,000개를 보관) | 가능 |
| D3 | f_d(x) | edge가 학습한 로지스틱 회귀(가중치 129개). 양성은 k의 학습 표본(h_k), 음성은 home cell의 다른 거주 device의 학습 표본 중 label이 M_k 밖인 것(각자의 h_j). P(Main) < 0.5이면 non-Main | 학습은 edge, 점수 계산은 device | 가능 |
| D4 | f_e(x): server block 출력 relu(fc2) (128) | cell마다 거주 device의 학습 표본으로 만든 class 평균과 공유 공분산. s = min_{M_k} d − min_{그 밖} d. s > 0이면 non-Main. 점수를 정할 수 없으면 판정하지 않음 | edge (f_e는 edge에서만 생김) | 불가능 |

| 규칙 | offload 결정 | offload된 요청의 답 | 쓴 탐지기 |
|---|---|---|---|
| R-b | 모든 요청 | non-Main이면 보정한 edge만, 아니면 DriftGate | D1–D4 |
| R-a | non-Main이거나 H(p_d) > τ | DriftGate-P | D1–D3 |
| R-c | R-a와 같음 | non-Main이면 보정한 edge만, 아니면 DriftGate-P | D1–D3 |
| R-s | 모든 요청 | 가중치 w·exp(−(d − θ)/θ)로 줄인 DriftGate | D1 |

D4는 edge가 server block을 계산한 뒤에만 점수를 낼 수 있습니다. Round 17에서도 D4는 모든 요청을 offload하는 R-b에만 썼습니다. 따라서 D4로 offload 여부를 정하는 규칙은 구현하지 않았고, 그렇게 쓴 적도 없습니다.

## 2. 평가 단위와 seed

- 정확도는 논문과 같이 계산했습니다. 평가 round마다 요청이 있는 device의 정확도를 평균하고, 그 값을 round에 대해 평균합니다.
- Round 17의 판정용 seed는 S1 seed 1–4와 S2 seed 0–2입니다. S1 seed 0은 α와 1차 후보를 고르는 데 썼습니다.
- Round 17 표의 S1 재생 DriftGate 71.82%는 seed 1–4의 평균입니다. Round 15 표 A2의 71.95%는 seed 0–4의 평균입니다. seed별 값은 두 Round에서 같습니다(72.4387, 71.8785, 72.1511, 69.4814, 73.7840). 따라서 두 값의 차이는 seed 집합의 차이이며 성능 변화가 아닙니다. S2는 두 Round 모두 seed 0–2이고 76.5354%로 같습니다.

## 3. strongest other의 비교 집합

- Round 17은 지시문에 따라 strongest other를 다음 11개 규칙에서 골랐습니다.
  - Round 15의 표시 규칙 10개(DriftGate 제외): Confidence-based offloading, Device only, Edge only, Probability average, Logit sum, Lower-entropy exit, Logit-entropy weighting, Geometric ensemble with early exit, Label-shift EM, Learned weight
  - corrected edge only
- 그 결과 Round 15 B1의 변형 5개는 비교 집합에서 빠졌습니다: no correction + adaptive w, correction + w 0.5, correction + development w = 0.2, correction + learned weight, correction + product.
- **S2 재생의 원인:** Round 17 표에서 S2 재생의 strongest other가 corrected edge only(76.59%)가 된 것은 비교 집합에서 correction + development w = 0.2(76.66%)를 뺐기 때문입니다. 구현이나 평가 조건의 차이는 없습니다. 두 Round에서 corrected edge only의 값은 76.589%로 같습니다.
- **정정:** strongest reference를 위의 17개 규칙에서 다시 골랐습니다.
  - S1 재생: strongest reference는 그대로 corrected edge only(72.544%)입니다.
  - S2 재생: strongest reference는 correction + development w = 0.2(76.663%)로 바뀝니다.

| 설정 | Round 17 규칙 | 정확도 | − strongest other (11개 규칙) | − strongest reference (17개 규칙) |
|---|---|---|---|---|
| S2 재생 | R-b(D1) | 76.761 | +0.171 (0.179) | +0.097 (0.105) |
| S2 재생 | R-b(D2) | 76.715 | +0.126 (0.263) | +0.052 (0.152) |
| S2 재생 | R-b(D3) | 76.608 | +0.019 (0.243) | −0.056 (0.134) |
| S2 재생 | R-b(D4), 1차 후보 | 76.711 | +0.122 (0.102) | +0.048 (0.091) |
| S2 재생 | R-s(D1) | 76.657 | +0.068 (0.238) | −0.007 (0.134) |

Round 17의 판정은 두 집합 모두에서 `NO_GO`입니다. 1차 후보의 차이는 S1이 −0.057 pp, S2가 +0.048 pp로, 기준인 +0.5 pp에 미치지 못합니다. 앞으로의 표에서는 17개 규칙을 비교 집합으로 씁니다.

## 4. Round 17 AUROC의 계산 방식

- **표본의 단위:** run마다 모든 요청(모든 device와 평가 round)을 하나로 모았습니다. device-round 가중치는 쓰지 않았으므로, 요청이 많은 device-round의 비중이 정확도 계산보다 큽니다.
- **양성과 점수의 방향:** 양성은 non-Main 요청(OOP와 OOR)이고, 점수가 클수록 non-Main으로 봅니다. 동점은 평균 순위로 처리했습니다(`m14.auroc`).
- **seed 평균:** run마다 AUROC를 구한 뒤 판정용 seed에 대해 평균했습니다.
- **D4의 유효 표본:** D4의 0.838(S1)과 0.864(S2)는 점수가 정의된 요청만으로 계산한 값입니다. 그 비율은 S1 0.897, S2 0.926입니다. 같은 요청 집합에서 다른 점수의 AUROC를 다시 계산한 결과는 `tables/R18_detector_matched_support.csv`에 있습니다(B2).
