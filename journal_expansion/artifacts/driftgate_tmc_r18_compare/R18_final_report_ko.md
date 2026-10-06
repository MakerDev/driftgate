# Round 18 최종 보고서: 비교 공백의 정리와 결합 변형 한 개의 시험

이번 Round에서는 새 학습을 하지 않았습니다. 기존 기록, Round 15 checkpoint, Round 16 cache(Round 15 정의의 규칙 답), Round 17 feature 기록만 썼습니다.

- **A와 B:** 비교와 진단입니다.
- **C:** Round 17을 본 뒤 제안된 가설 하나를 기존 seed에서 시험한 탐색 결과이며, 독립 검증이 아닙니다.
- **D:** 선행연구 BTFL과의 비교를 준비한 작업입니다.

분석 계획과 판정 규칙은 결과를 보기 전에 commit했습니다(`45d250d`, `R18_plan.md`). 표의 값은 seed 평균(표준편차)이고, 차이는 같은 seed끼리 뺀 값입니다. request 수를 독립 표본으로 쓰는 신뢰구간은 만들지 않았습니다.

## 1. 기존 DriftGate 원고에 바로 쓸 수 있는 가장 강한 주장

1. **보정한 결합의 이득:** edge exit를 device의 class 비율로 보정한 뒤 두 exit를 결합하면, 같은 offloading 비율에서 보정하지 않은 결합 규칙보다 정확합니다.
   - 첫날 CNN 설정 7개(S1, S2, S1-fast, partial participation, stepwise change, random mobility, CIFAR-100)에서, DriftGate-P는 보정하지 않은 규칙 가운데 가장 강한 규칙보다 β = 0.5에서 +0.60 ~ +1.18 pp, β = 1에서 +0.84 ~ +1.88 pp 높습니다.
   - β는 controller가 목표로 하는 offloading 비율입니다. 보정하지 않은 규칙은 raw edge, Probability average, Logit sum, 보정 없이 adaptive w를 쓰는 규칙입니다(`R18_online_same_budget_summary.csv`).
   - ResNet-18에서는 +0.18 pp이고, K=200에서는 β = 1일 때 0.00 pp입니다. K=500에서는 β = 0.5일 때 −0.25 pp, β = 1일 때 −0.49 pp로 낮습니다.
2. **adaptive weight의 크기:** adaptive weight가 고정 weight보다 더하는 정확도는 작습니다.
   - 같은 budget에서 DriftGate-P와 보정한 고정 weight 규칙 가운데 가장 강한 규칙의 차이는 −0.79 ~ +0.15 pp입니다. 보정한 고정 weight 규칙은 correction + w 0.5, correction + development w 0.2, corrected edge only, correction + product, correction + learned weight입니다.
   - DriftGate-P가 이 규칙들보다 0.05 pp 이상 높은 경우는 S1, stepwise change, random mobility뿐이며, 그 차이는 +0.05 ~ +0.15 pp입니다. 그 밖에 양수인 칸(S1-fast β = 1, S2와 partial participation β = 0.5)은 +0.04 pp 미만입니다.
   - 따라서 adaptive weight는 모델마다 따로 조정하지 않고 쓰는 기본 선택으로 설명하는 것이 수치에 맞습니다. 독립된 큰 기여로 설명하지 않습니다.
3. **offloading 문장 (A3):** 원고의 "7개 설정에서 14–52%의 offloading으로 모든 full-offload 규칙을 넘는다"와 "43–86% 감소"는 Round 15의 7개 full-offload 규칙에 대해 정확히 재현됩니다. 보정 변형 6개를 비교 집합에 넣으면 다음과 같이 달라집니다(`R18_A3_offload_claims.csv`).

| 비교 집합 | 최고 정확도에 도달하는 설정 | 필요한 offloading | confidence-based offloading 대비 offload 요청 감소 | edge 호출 감소 |
|---|---|---|---|---|
| Round 15의 7개 규칙 | 7개 (CIFAR-100, K=200, K=500 제외) | 0.136–0.524 | 43–86% | 44–86% |
| 보정 변형 포함 13개 규칙 | 5개 (S1, S1-fast, stepwise, random mobility, ResNet-18) | 0.433–0.870 | 5–20% | 5–20% |

보정 변형을 넣으면 S2, partial participation, CIFAR-100, K=200, K=500, 그리고 두 재생에서는 어떤 threshold로도 최고 정확도에 도달하지 못합니다. 이 설정들에서는 DriftGate의 full-offload 정확도 자체가 보정 변형보다 낮습니다. 따라서 원고의 offloading 문장은 "보정하지 않은 full-offload 규칙들과 비교하면"이라는 비교 집합을 명시해야 합니다.

edge 호출은 두 cell에 속한 요청을 두 번으로 셉니다. 요청당 호출 수는 β = 0.5에서 0.53–0.98(random mobility가 가장 많음)이고, β = 1에서 1.06–1.97입니다.

## 2. 같은 budget에서 보정한 단순 대안과의 차이 (A2)

controller는 Round 15 C2를 그대로 쓰고, 목표 비율 β만 바꿨습니다(0.25, 0.5, 0.75, 1). β = 0.5의 offload mask는 Round 15와 같았고, β = 1에서 DriftGate-P의 답은 Round 15 DriftGate와 100% 같았습니다(가중치 차이 10⁻¹¹ 이하). 실제 offloading 비율은 0.498–0.500(β = 0.5)이었습니다. 비교한 규칙과 home/away, Main/OOP/OOR의 값은 `R18_online_same_budget.csv`에 있습니다.

| 설정 | β = 0.5: DriftGate-P − 가장 강한 reference | 그 reference | β = 1: 차이 | 그 reference |
|---|---|---|---|---|
| S1 | +0.050 (0.089) | correction + dev w 0.2 | +0.152 (0.152) | correction + dev w 0.2 |
| S2 | +0.006 (0.065) | correction + w 0.5 | −0.073 (0.131) | correction + w 0.5 |
| S1-fast | −0.028 (0.024) | correction + dev w 0.2 | +0.033 (0.072) | correction + dev w 0.2 |
| partial participation | +0.019 (0.034) | correction + w 0.5 | −0.026 (0.084) | correction + w 0.5 |
| stepwise change | +0.109 (0.062) | correction + w 0.5 | +0.142 (0.093) | correction + w 0.5 |
| random mobility | +0.119 (0.019) | correction + w 0.5 | +0.138 (0.056) | correction + w 0.5 |
| CIFAR-100 | −0.268 (0.028) | correction + product | −0.445 (0.076) | correction + product |
| ResNet-18 | −0.033 (0.043) | correction + w 0.5 | −0.067 (0.093) | correction + w 0.5 |
| K=200 | −0.205 (0.030) | corrected edge only | −0.231 (0.042) | correction + dev w 0.2 |
| K=500 | −0.362 (0.092) | corrected edge only | −0.487 (0.217) | no correction + adaptive w |
| S1 재생 | −0.347 (0.271) | corrected edge only | −0.791 (0.413) | corrected edge only |
| S2 재생 | −0.037 (0.115) | correction + dev w 0.2 | −0.128 (0.148) | correction + dev w 0.2 |

- **같은 정확도에 필요한 offloading:** 목표는 β = 1에서 가장 강한 reference의 정확도입니다. DriftGate-P가 이 목표에 도달하는 offloading은 다음과 같습니다. 이 값은 네 β 사이를 선형 보간한 것입니다(`R18_online_offload_needed.csv`).
  - 도달함: ResNet-18 0.188, S2 0.530, stepwise change 0.676, random mobility 0.704, S1 0.725, S1-fast 0.941
  - 도달하지 못함: partial participation, CIFAR-100, K=200, K=500, 두 재생
- **어느 쪽이 더 강한가:** 같은 budget의 정확도 이득과 정확도 목표에서의 offloading 절감은, 보정한 대안에 대해서는 둘 다 작습니다.
- **Main 비중에 따른 민감도 (E1):** 예측을 고정하고 Main 비중 s만 바꿔 다시 가중했습니다(`R18_main_share_sensitivity.csv`).
  - S1에서 DriftGate는 s = 0.8에서 corrected edge only보다 0.96 pp, s = 0.9에서 1.66 pp 높습니다.
  - s = 0.3에서는 2.54 pp, s = 0.5에서는 1.14 pp 낮습니다. 관측된 Main 비중은 0.65–0.74입니다.
  - 이 값은 고정된 예측을 다시 가중한 결과이며, controller와 window가 바뀐 요청 구성에 적응한 결과가 아닙니다.

## 3. 새 후보(conditional combination)의 판정

**후보의 정의:** 후보는 edge가 정한 own-class 확률 합 E를 유지하고, own class 안에서만 device와 edge를 결합합니다. 식은 p(c) = E·w·d_c/D + (1 − w)·e_c (c ∈ M_k), e_c (그 밖)입니다. D와 E가 0에 가까워 corrected edge로 돌아간 경우는 모든 run에서 0건이었습니다.

**판정:** 두 판정 모두 통과하지 못했습니다(`R18_conditional_decision.json`).
- **엄격 판정 (`FAIL`):** 지정한 56행에서 후보가 DriftGate와 strongest reference보다 모두 높아야 합니다. 실제로는 β = 1에서 53행, β = 0.5에서 47행이 이 조건을 만족하지 못했습니다.
- **독립 검증 후보 판정 (`FAIL`):** 핵심 8칸 가운데 S1·S2 첫날과 S2 재생에서 후보가 DriftGate보다 낮습니다. 재생 4칸 모두에서 strongest reference 대비 +0.5 pp에 미치지 못합니다.

| 핵심 행 | β = 1: 후보 − DriftGate | 후보 − strongest reference | β = 0.5: 후보 − DriftGate | 후보 − strongest reference |
|---|---|---|---|---|
| S1 첫날 | −0.083 (0.169) | +0.069 (0.085) | −0.034 (0.112) | +0.016 (0.030) |
| S2 첫날 | −0.866 (0.180) | −0.938 (0.305) | −0.476 (0.092) | −0.470 (0.149) |
| S1 재생 | +0.791 (0.317) | −0.001 (0.147) | +0.371 (0.205) | +0.024 (0.109) |
| S2 재생 | −0.062 (0.155) | −0.190 (0.140) | −0.058 (0.144) | −0.096 (0.108) |

**손실이 큰 행 (β = 1, 후보 − DriftGate):**
- ResNet-18: −4.20 (1.97)
- S1 출근 전: −2.35 (0.49)
- S2 출근 전·출근: −1.56
- S2 첫날: −0.87
- partial participation: −0.64
- CIFAR-100: −0.49

β = 1에서 후보가 DriftGate보다 높은 행은 다음과 같습니다.
- S1, S1-fast, stepwise change, random mobility의 round > 30과 round > 50: +0.42 ~ +1.45
- K=200과 K=500의 세 구간: +0.50 ~ +1.23
- S1의 낮과 귀가 시간대
- S1 재생 전체(+0.79)와 두 재생의 away 행

후보가 DriftGate와 strongest reference보다 모두 높은 행은 random mobility round > 30(+0.004), K=200 전체(+0.27), K=500 전체(+0.25)의 3행뿐입니다. 통제 후보(고정 w 0.5, 보정하지 않은 edge)의 값은 `R18_conditional_combination.csv`에 있습니다.

**오류 분해 (C3):** 같은 가중치로 오류를 네 종류로 나누었습니다(`R18_conditional_error_groups.csv`). 네 차이의 합과 전체 오류 차이의 오차는 10⁻¹⁴ 이하입니다. S1 첫날, β = 1에서 후보 − DriftGate는 다음과 같습니다.
- 정답이 Main인데 non-Main을 예측: +1.87 pp
- 정답이 non-Main인데 Main을 예측: −2.49 pp
- Main 안의 다른 class를 예측: −0.33 pp
- non-Main 안의 다른 class를 예측: +1.04 pp

후보와 DriftGate의 차이는 주로 own class 집합과 나머지 집합 사이에서 argmax가 바뀌는 데서 생깁니다.

**대수적 관계:** 후보는 own-class 안의 상대 확률에만 device 출력을 쓰는 재매개화이며, w = 0이거나 |M_k| = 1이면 corrected edge only와 같습니다. 이 점에서 기존의 weighted correction과 다른 새 원리라고 볼 근거는 없습니다.

**진단 (B):** base 답을 DriftGate, alternative 답을 corrected edge only로 두었습니다(`R18_action_utility.csv`).
- **답을 바꿀 때의 구조:** S1 첫날에 답을 바꾸면 고쳐지는 요청이 2.11%, 망가지는 요청이 2.56%입니다. 두 답이 다른 요청은 6.6%이고, 그 안에서 고쳐지는 비율이 32%, 망가지는 비율이 39%, 변화 없는 비율이 30%입니다.
- **headroom:** 두 답 가운데 정답을 고르는 oracle은 DriftGate보다 +0.7 ~ +3.3 pp 높습니다. 이것은 실현 가능한 정책의 성능이 아닙니다.
- **신호의 구별력:** 고쳐지는 요청과 망가지는 요청을 구별한 AUROC는 다음과 같습니다.
  - device entropy: 0.31–0.51 (대부분 0.5보다 작아 방향이 반대)
  - 보정한 edge의 Main 확률 합: 0.44–0.61
  - margin 차이: 0.52–0.60
  - Q_DG − E: 0.42–0.63
  - Round 17 탐지기 D1–D4(재생): 0.49–0.66
- **learned weight의 학습 목표:** Round 13b와 Round 15의 learned weight는 요청이 Main인지(kind = Main)를 맞히도록 학습했습니다. 어느 답이 맞는지를 학습한 것이 아닙니다.
- **D4의 유효 범위 (B2):** D4가 정의된 같은 요청에서도 D4의 AUROC(0.84, 0.86)가 entropy(0.71)와 D1–D3(0.66–0.74)보다 높습니다. device별 AUROC의 평균도 같은 순서입니다. D4가 정의되는 비율은 home 0.99–1.00, away 0.81–0.82이고, 한 종류의 요청만 있는 device는 없었습니다(`R18_detector_matched_support.csv`).

## 4. Round 17에서 정정한 비교 범위와 평가 단위

자세한 내용은 `R18_reference_audit.md`에 있습니다.

- **seed 집합:** Round 17 표의 S1 재생 DriftGate 71.82%는 판정용 seed 1–4의 평균입니다. Round 15의 71.95%는 seed 0–4의 평균이며, seed별 값은 같습니다.
- **비교 집합:** Round 17의 strongest other는 corrected edge only를 포함한 11개 규칙에서 골랐으므로, Round 15 B1의 보정 변형 5개가 빠졌습니다. 17개 규칙으로 다시 고르면 S2 재생의 strongest reference는 correction + development w 0.2(76.66%)가 됩니다.
  - 1차 후보 R-b(D4)의 차이는 S1 −0.057 pp, S2 +0.048 pp(이전 +0.122 pp)입니다.
  - `NO_GO`는 바뀌지 않습니다. 앞으로의 표는 17개 규칙을 비교 집합으로 씁니다.
- **D4의 위치:** D4는 edge의 f_e를 쓰므로 offload 전에 판정할 수 없고, Round 17에서도 full-offload 규칙(R-b)에만 썼습니다.
- **AUROC 계산 방식:** run마다 요청을 모아 계산했으므로 device-round 가중치는 쓰지 않았습니다. 양성은 non-Main이고, 판정용 seed의 평균입니다. D4의 값은 점수가 정의된 요청에서만 계산했습니다.

## 5. BTFL 비교

자세한 내용은 `R18_BTFL_compatibility.md`와 `tables/R18_BTFL_adaptation.csv`에 있습니다.

**BTFL의 전제:** BTFL은 공유된 전역 feature 위에 붙은 두 선형 head를 Beta 사후 평균 e로 섞습니다. e는 feature 이진 빈도표(DLE)의 우도비와 entropy 사건으로 정합니다.

**우리 split 모델과의 차이:** device block이 device마다 다르므로, 다른 device들의 빈도표를 평균한 global DLE는 공통 feature 가정을 지키지 못합니다. 이 차이를 명시한 'BTFL inference adaptation'을 S1·S2 재생에서 기본값으로 한 번 계산했습니다.
- 원문 수식을 따르고, λ = 16, 확률 혼합을 썼습니다.
- 원문이 정하지 않은 부분은 코드를 따랐습니다.
- H̄_l과 H̄_g는 checkpoint로 학습 표본에 forward pass를 해서 구했습니다.

| 설정 | BTFL adaptation | DriftGate | 차이 | strongest reference |
|---|---|---|---|---|
| S1 재생 | 70.18 (1.33) | 71.95 (1.56) | −1.77 (0.34) | corrected edge only 72.74 |
| S2 재생 | 74.16 (2.90) | 76.54 (2.52) | −2.38 (0.66) | correction + dev w 0.2 76.66 |

**가중치와 구별력:** edge head의 가중치 e는 평균 0.35–0.40이고, Main 요청에서 non-Main 요청보다 컸습니다. DLE 우도비로 non-Main을 가린 AUROC는 0.69–0.71입니다.

**원문과 코드의 차이:** 혼합 공간, u_g의 상수, τ̂의 압축, 사건 배정, 가지치기 규칙, 평활화, L1 학습 벌점이 다릅니다. 목록은 문서에 정리했습니다.

**faithful 비교에 필요한 것:** 전역으로 공유한 device block과 그 위의 개인 head로 다시 학습한 run이 필요합니다.

## 6. 모바일 실측 상태

실측 장비와 실행 환경이 준비되지 않아 측정하지 않았습니다. 가정값이나 선형 energy 모형으로 실측 표를 채우지 않았습니다.

**측정할 때 정해 둔 항목:**
- 지표: 평균과 p95 지연, 요청당 에너지, uplink와 downlink byte, edge 호출 수
- 공통 도달점: β = 0.25, 0.5, 0.75, 1에서 같은 요청 stream을 씁니다. 실제 offloading은 β와 거의 같습니다.
- 처리 경로: 한 cell과 두 cell의 처리 경로를 나눕니다.
- 측정 절차: 요청 간격, warm-up, idle 기준을 정하고 반복 측정합니다.
- 계산 시간: 서버 CPU에서 잰 3.2 µs와 device에서 잰 시간을 구분합니다.

**두 cell의 처리 경로 (코드에서 확인):** 시뮬레이터는 두 edge의 logit을 device 쪽 복사본에서 먼저 평균하고 softmax를 취합니다(`runner_r6.eval_client`). 보정은 분석 단계에서 그 평균 확률에 한 번 적용합니다. 따라서 cell마다 보정한 확률을 평균하는 계산과는 다릅니다.
- **배포에 필요한 전송:** 결합을 맡은 edge가 다른 edge의 logit(C개 값)을 받아야 합니다. device의 feature(CNN에서 32 KB)도 두 edge에 모두 전달되어야 합니다.
- **44 B의 범위:** 원고의 44 B는 결합을 맡은 한 endpoint가 돌려주는 application payload(보정한 확률 10개와 entropy 1개, float32)입니다. edge 사이의 전송과 두 번째 호출은 따로 세야 합니다.

## 7. 추가 작업이 필요한 경우의 최소 실행 목록

새 후보는 두 판정에서 실패했습니다. 지시문에 따라 새 fallback, warm-up 전환, 후보 조합을 만들지 않고, 기존 DriftGate를 유지합니다. 아래 작업은 필요할 때만 합니다.

1. **원고 수정 (새 run 없음):** offloading 문장에 비교 집합을 명시하고, adaptive weight의 기여를 1절의 크기에 맞춥니다.
2. **Main 비중 민감도의 후속 확인 (선택):** E1에서 결론이 Main 비중에 따라 바뀌었습니다(s ≤ 0.5에서 corrected edge only가 높음). 이를 실제 stream으로 확인하려면 다음을 실행합니다.
   - Round 15 checkpoint(S1 seed 0–4)를 고정한 채, 요청 생성의 OOP·OOR 비중을 바꿔 Main 비중 약 0.5인 요청 stream으로 재생합니다.
   - controller와 window를 다시 계산합니다.
   - 이것은 '새 경로 검증'(기존 모델, 새 요청 stream)이며, 새 학습 seed 검증이 아닙니다. run당 약 2분이 걸립니다.
3. **faithful BTFL 비교 (선택, 새 학습 필요):** S1·S2(seed 0–2)에서 device block을 전역으로 공유하고, 개인 head를 따로 학습한 run이 필요합니다. 같은 run에서 DriftGate도 다시 평가합니다.
4. **모바일 실측:** 6절의 항목입니다. 장비가 준비된 뒤에 합니다.

## 8. 산출물과 실행

**문서**
- `R18_plan.md`: 결과 전에 고정한 정의와 판정 규칙
- `R18_reference_audit.md`
- `R18_BTFL_compatibility.md`

**표 (`tables/`)**
- `R18_reference_manifest.csv`: run, Round 16 cache 파일의 크기와 수정 시각, 확인 항목
- `R18_R17_corrected_margins.csv`, `R18_R17_seed_sets.csv`
- `R18_online_same_budget.csv`: seed별 표와 요약표(`_summary`) 포함
- `R18_online_offload_needed.csv`, `R18_A3_offload_claims.csv`
- `R18_action_utility.csv`, `R18_detector_matched_support.csv`
- `R18_conditional_combination.csv`: seed별 표 포함
- `R18_conditional_error_groups.csv`, `R18_conditional_decision.json`
- `R18_main_share_sensitivity.csv`, `R18_BTFL_adaptation.csv`

**스크립트 (`scripts/`)**
- `r18_analysis.py`: `run`, `features`, `tables` 하위 명령
- `r18_audit.py`, `r18_btfl.py`, `r18_summary.py`
- 실행 명령은 `scripts/README.md`에 있습니다.

**계산 시간:** run마다 분석에 0.1–7분(K=500이 가장 김), BTFL adaptation에 약 25초가 걸렸습니다.
