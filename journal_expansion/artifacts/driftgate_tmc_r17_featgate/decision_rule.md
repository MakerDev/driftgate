# Round 17 0단계 판정 규칙과 분석 설정

이 문서는 Round 17 지시문 4절(0단계)을 실행하기 위해, 결과를 보기 전에 고정한 규칙입니다. 수치 설정은 `r17_config.json`에 같은 내용으로 적었고, 분석 스크립트(`scripts/r17_stage0.py`)는 그 파일을 읽습니다. 지시문 8절에 따라 α, k, ε, 분위수 규칙, D3의 음성 표본 정의, D4의 점수 정의는 결과를 본 뒤에 바꾸지 않습니다. 바꿔야 할 이유가 생기면 바꾸지 않고 보고서에 적습니다.

## 1. 기록과 seed

- **입력:** Round 15에서 저장한 학습 종료 checkpoint를 씁니다. S1은 seed 0–4, S2는 seed 0–2이며, 경로는 `runs/phaseT15_replay/r15_{s1,s2}_fixed040_s{seed}_ckpt.pt`입니다.
- **재생 방법:** Round 15의 재생 명령을 같은 flag로 다시 실행합니다. 출력 위치는 `runs/phaseT17_featgate/r17_{s1,s2}_replay_s{seed}`입니다(`scripts/r17_extract.py`). runner는 고치지 않습니다. 평가 함수를 바깥에서 감싸서 요청마다 다음 feature를 기록합니다.
  - f_d는 device block 출력(128×8×8)을 공간에 대해 평균한 128차원 벡터입니다.
  - f_e는 server block의 출력이며, 정확히는 server exit(fc3)에 들어가는 relu(fc2) 값입니다. 128차원입니다.
  - 두 cell에 속한 device는 cell마다 f_e를 기록합니다.
- **학습 표본의 feature:** 재생이 끝난 뒤(모델은 고정)에 각 device의 학습 표본 전체를 그 device의 block에 통과시킵니다. 각 cell에서는 거주 device(home cell이 그 cell인 device)의 학습 표본을 각자의 block과 그 cell의 server block에 통과시킵니다.
- **seed의 역할**
  - 설계용 seed는 S1 seed 0입니다. 이 seed에서 D1과 D2의 α, 그리고 1차 후보 규칙을 고릅니다.
  - 판정용 seed는 S1 seed 1–4와 S2 seed 0–2입니다. 표의 주된 값도 이 seed들로 계산합니다.
- **시작 확인:** 새 재생 기록의 DriftGate 정확도를 Round 15 표 A2의 재생 값과 비교합니다. 기준값은 S1 71.95%, S2 76.54%입니다. seed 평균의 차이가 0.05 pp 이내여야 하고, seed마다의 값은 Round 15 캐시와 비교해 적습니다. 새 재생의 출력 배열이 Round 15 재생 기록과 비트 단위로 같은지도 함께 적습니다. 0.05 pp를 넘으면 원인을 적고 멈춥니다.

## 2. 탐지기

모든 탐지기는 점수가 클수록 non-Main입니다. 통계는 0단계에서 학습 표본 전체로 한 번 만듭니다. 이 표본은 모델 학습에도 쓰였으므로, 문턱과 AUROC(수신자 조작 특성 곡선 아래 면적)가 낙관적일 수 있습니다.

### D1 (device, Mahalanobis)

- 공분산은 Σ = (1/n) Σ_i (f_i − μ_{y_i})(f_i − μ_{y_i})ᵀ입니다. 여기서 n은 device의 학습 표본 수이고, μ_c는 own class c의 평균입니다.
- Σ에 ε·I를 더합니다. ε = 10⁻³·trace(Σ)/128입니다.
- 점수는 d(x) = min_{c∈M_k} (f_d(x) − μ_c)ᵀ (Σ + εI)⁻¹ (f_d(x) − μ_c)입니다.
- 문턱 θ_k는 학습 표본의 d 값 가운데 (1 − α) 분위수입니다(numpy 선형 보간). d(x) > θ_k이면 non-Main으로 판정합니다.
- α ∈ {0.05, 0.10}은 설계용 seed에서 고릅니다.

### D2 (device, k-NN)

- f_d를 L2 정규화한 뒤, 학습 표본의 feature 가운데 10번째로 가까운 표본과의 cosine 거리(1 − cosine 유사도)를 점수로 씁니다.
- k의 후보값이 지시문에 10 하나뿐이므로 k = 10으로 둡니다. 지시문 4절의 "D2의 k를 고른다"는 문턱의 α를 D1과 같은 방식으로 고르는 것으로 처리합니다.
- 문턱을 정할 때 학습 표본 자신의 점수는 자기 자신을 뺀 나머지 표본 가운데 10번째 이웃으로 계산합니다(leave-one-out).

### D3 (edge가 학습하는 이진 head)

- device k의 양성은 k의 학습 표본이고, 각 표본은 h_k로 만든 f_d를 씁니다.
- 음성은 k의 home cell에 사는 다른 device j의 학습 표본 가운데 label이 M_k 밖인 것이고, 각 표본은 h_j로 만든 f_d를 씁니다. 음성 feature가 다른 device의 block에서 나왔다는 점은 보고서에 적습니다.
- 모델은 scikit-learn LogisticRegression입니다. L2 정규화 C = 1, lbfgs, class_weight = balanced, max_iter = 5000이고, f_d를 표준화하지 않고 그대로 씁니다. 가중치는 128개와 bias 1개로 모두 129개입니다.
- P(Main) < 0.5이면 non-Main으로 판정합니다. 점수는 1 − P(Main)입니다. 음성 표본이 없으면 head를 만들지 않고 non-Main 판정도 하지 않습니다.

### D4 (edge, class 원형 비교)

- cell z마다 거주 device의 학습 표본으로 f_e 공간의 class 평균(그 cell에 있는 class만)과 공유 공분산을 만듭니다. 공분산과 ε은 D1과 같은 식으로 구합니다.
- 요청이 cell z를 거치면 s_z = min_{c∈M_k∩C_z} d_c − min_{c∈C_z∖M_k} d_c를 계산합니다. C_z는 cell z에 원형이 있는 class의 집합이고, d_c는 Mahalanobis 거리입니다.
- 두 cell에 속한 device는 정의된 s_z를 평균합니다. 두 집합 가운데 하나가 비어 있어 점수를 정할 수 없으면 non-Main 판정을 하지 않습니다.
- AUROC는 점수가 정의된 요청에서 계산하고, 그 비율(coverage)을 함께 적습니다. D4는 모든 요청이 offload되는 R-b에만 씁니다.

AUROC는 run마다 모든 요청을 모아 계산합니다. 양성은 non-Main이고, 동점은 평균 순위로 처리합니다. 같은 계산을 home 요청과 away 요청에서 따로 하고, device exit entropy의 AUROC도 함께 적습니다.

## 3. 규칙

모든 규칙은 DriftGate의 보정(r = 0.5)을 씁니다. 가중치는 offload 방식에 따라 다릅니다. 모든 요청을 offload하는 규칙은 R13b F-auto 가중치(Round 15의 DriftGate)를 쓰고, 일부만 offload하는 규칙은 Round 15의 DriftGate-P 가중치를 씁니다. DriftGate-P 가중치는 같은 offload 집합의 window에서 두 entropy의 평균으로 만듭니다. 아래에서 p'_s는 보정한 server exit입니다.

| 이름 | offload 결정 | offload된 요청의 답 | offload되지 않은 요청의 답 |
|---|---|---|---|
| R-b(D) | 모든 요청 | non-Main이면 argmax p'_s, 아니면 DriftGate 평균 | 없음 |
| R-a(D, τ) | non-Main이거나 H(p_c) > τ | DriftGate-P 평균 | device exit |
| R-c(D, τ) | R-a와 같음 | non-Main이면 argmax p'_s, 아니면 DriftGate-P 평균 | device exit |
| R-s (D1) | 모든 요청 | w(x) = w (d ≤ θ_k), w·exp(−(d − θ_k)/θ_k) (d > θ_k)로 평균 | 없음 |

- τ는 Round 14의 격자 {0, 0.05, …, 2.30}에서 바꾸고, τ = ∞도 별도 점으로 둡니다. 곡선은 server 사용 비율 0.1, …, 1.0에서 Round 15와 같이 보간합니다. server 사용 비율은 offload된 요청의 비율입니다. 같은 표에 entropy 기반 DriftGate-P 곡선(H(p_c) > τ만으로 offload)을 둡니다.
- **거리 controller (β = 0.5)**
  - device마다 직전 128개 요청의 탐지기 점수에서 중앙값을 구해 문턱으로 씁니다. 요청의 순서는 round, 도착 순서이고, 현재 요청은 넣지 않습니다.
  - 직전 요청이 16개 미만이면 non-Main 판정만으로 offload합니다.
  - R-a와 R-c에 D1–D3을 써서 계산합니다.
  - 같은 표에 Round 15 C2의 entropy controller를 둡니다. entropy controller에서는 offload된 요청에 edge 답, Probability average, Logit sum, DriftGate-P를 씁니다.
- 종류 oracle은 Main 요청에 device exit를, 나머지 요청에 보정하지 않은 server exit를 씁니다.
- 기존 규칙 11개와 corrected edge only의 정의는 Round 15와 같습니다(`r16_stage1.reference_answers`).

## 4. 지표

정확도는 Round 15와 같이 계산합니다. 평가 round마다 요청이 있는 device의 정확도를 평균하고, 그 값을 재생의 모든 평가 round에 대해 평균합니다. home, away, Main, OOP, OOR도 Round 15의 정의를 따릅니다. 표에는 판정용 seed의 평균(표준편차)을 적고, 차이는 같은 seed끼리 뺀 값의 평균(표준편차)을 적습니다.

**strongest other**는 설정마다 다음 11개 규칙 가운데 seed 평균이 가장 높은 규칙입니다. DriftGate와 새 규칙은 이 집합에 넣지 않습니다.

- Confidence-based offloading, Device only, Edge only, Probability average, Logit sum, Lower-entropy exit
- Logit-entropy weighting, Geometric ensemble with early exit, Label-shift EM, Learned weight, corrected edge only

## 5. 설계용 seed에서 고르는 것

설계용 seed(S1 seed 0)에서 다음 세 가지를 고릅니다. 판정용 seed의 결과는 이 선택에 쓰지 않습니다.

1. D1의 α: R-b(D1)의 전체 정확도가 높은 쪽을 고르고, 같으면 0.05를 고릅니다.
2. D2의 α: R-b(D2)로 같은 방식으로 고릅니다.
3. 1차 후보: R-b(D1), R-b(D2), R-b(D3), R-b(D4), R-c(D1, ∞), R-c(D2, ∞), R-c(D3, ∞) 가운데 전체 정확도가 가장 높은 규칙을 고릅니다. 같으면 이 목록의 앞쪽을 고릅니다.

## 6. 0단계 판정 (`decision_stage0.json`)

판정용 seed에서 설정 S ∈ {S1, S2}마다 다음 두 값을 계산합니다.

- A(S)는 D1과 D3의 AUROC(seed 평균, 모든 요청) 가운데 큰 값입니다.
- B(S)는 1차 후보의 seed 평균 정확도에서 strongest other의 seed 평균 정확도를 뺀 값(pp)입니다.

| 판정 | 조건 |
|---|---|
| `GO` | S1과 S2 모두에서 A ≥ 0.80이고 B ≥ 0.5 |
| `HOLD` (보류) | S1과 S2 모두에서 B ≥ 0.5이고 A ≥ 0.75이지만, 한 설정 이상에서 A < 0.80 |
| `NO_GO` | 그 밖의 모든 경우 |

반올림한 값으로 판정하지 않습니다. 1차 후보가 아닌 R-b와 R-c(τ = ∞) 변형이 B ≥ 0.5를 만족하는지도 참고로 적지만, 판정에는 쓰지 않습니다. 0단계 보고서(중간 보고서 1)를 남긴 뒤 멈추고, 1단계는 사용자가 go를 확인한 뒤에 시작합니다.

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `tables/R17_T0_start_check.csv` | 재생의 재현과 DriftGate 값 |
| `tables/R17_S0_design.csv` | 설계용 seed에서 고른 α와 1차 후보 |
| `tables/R17_S0_auroc.csv`, `_per_seed.csv` | 탐지기와 entropy의 AUROC (전체, home, away) |
| `tables/R17_S0_accuracy.csv`, `_per_seed.csv` | 규칙의 정확도 (전체, home, away, Main, OOP, OOR)와 strongest other 및 DriftGate와의 차이 |
| `tables/R17_S0_curves.csv` | R-a, R-c, DriftGate-P의 offload 곡선과 τ = ∞인 점 |
| `tables/R17_S0_online.csv` | entropy controller와 거리 controller (β = 0.5) |
| `tables/R17_S0_cost.csv` | 탐지기의 요청당 계산량과 device 또는 edge가 보관하는 통계의 크기 |
| `tables/R17_S0_detector_checks.csv` | 문턱, non-Main 판정 비율, D3 head 수, D4 coverage |
| `decision_stage0.json`, `R17_stage0_report_ko.md` | 판정과 중간 보고서 1 |
