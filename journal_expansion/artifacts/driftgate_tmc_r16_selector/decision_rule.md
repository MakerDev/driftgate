# Round 16 판정 규칙과 1단계 분석 설정

이 문서는 Round 16 지시문 1–4절을 실행하기 위해 결과를 계산하기 전에 고정한 규칙입니다. 수치로 된 설정은 `r16_config.json`에 같은 내용으로 적었고, 분석 스크립트는 그 파일을 읽어서 계산합니다. 이 문서를 commit한 뒤에 계산한 결과를 보고 규칙을 바꾸지 않습니다. 계산 오류를 고쳐야 하면 고친 시점과 영향을 받은 결과를 따로 기록합니다.

## 1. 기록과 행

### 1.1 기록

| 설정 | 기간 | 기록 | seed | 역할 |
|---|---|---|---|---|
| S1 | 첫날 | `runs/phaseT15_replay/r15_s1_fixed040_s{}` (Round 15 재학습) | 0–4 | 판정 |
| S2 | 첫날 | `runs/phaseT15_replay/r15_s2_fixed040_s{}` (Round 15 재학습) | 0–2 | 판정 |
| S1-fast | 첫날 | `runs/phaseT12_fusion/r12_s1fast_fixed040_s{}` | 0–2 | 판정 |
| Partial participation | 첫날 | `runs/phaseT12_fusion/r12_s4part05_fixed040_s{}` | 0–2 | 판정 |
| Stepwise change | 첫날 | `runs/phaseT12_fusion/r12_t1_A_fx40_s{}` | 0–4 | 판정 |
| Random mobility | 첫날 | `runs/phaseT12_fusion/r12_t1_mob_fx40_s{}` (120 round) | 0–2 | 판정 |
| CIFAR-100 | 첫날 | `runs/phaseT13b_arch/r13b_c100_s{}` | 0–2 | 판정 |
| ResNet-18 | 첫날 | `runs/phaseT12_fusion/r12_e2_res_fx40_A_s{}` (device 16개) | 0–2 | 판정 |
| K=200 | 첫날 | `runs/phaseT12_fusion/r12_s3k200_fixed040_s{}` | 0–2 | 판정 |
| K=500 | 첫날 | `runs/phaseT12_fusion/r12_s3k500_fixed040_s{}` | 0–2 | 판정 |
| S1 replay | 재생 | `runs/phaseT15_replay/r15_s1_replay_s{}` | 0–4 | 판정 |
| S2 replay | 재생 | `runs/phaseT15_replay/r15_s2_replay_s{}` | 0–2 | 판정 |
| S1, S2 (Round 12 기록) | 첫날 | `runs/phaseT12_fusion/r12_s{1,2}_fixed040_s{}` | 0–4, 0–2 | 보조 표 |
| S1 개발 | 첫날 | `runs/phaseT10_prior/r10_T40_s{}` | 5–7 | S-A의 s 선택 |

지시문 3.1절에 따라 S1과 S2의 첫날 행에는 Round 15에서 재학습한 첫날 기록을 씁니다. 이 기록은 재생 기록과 같은 모델에서 나왔으므로, 시간대 표와 온라인 표에서 첫날과 재생을 같은 run끼리 비교할 수 있습니다. 논문 주표에 쓴 Round 12의 S1·S2 첫날 기록도 같은 방법으로 계산하지만, 보조 표로만 보고하고 판정에는 세지 않습니다. 설정 수와 seed 수는 늘리지 않습니다.

### 1.2 평가 행

- 기본 표(전체 offloading)는 판정용 첫날 10개 설정에 4개 구간을 곱한 40행과, S1·S2 재생 2행을 더해 42행입니다.
  - `day1_full`: 기록의 모든 평가 round
  - `day1_early`: round ≤ 30
  - `day1_gt30`: round > 30
  - `day1_gt50`: round > 50
  - `replay_full`: 재생 기록의 모든 평가 round
- 시간대 표는 S1과 S2 각각에서 첫날(재학습 기록)과 재생을 나누고, R15와 같은 5개 시간대로 나누어 20행입니다. 평가 round r의 시작 시각은 300 + 6(r − 1)분으로 계산합니다. 시간대 경계는 출근 전 [300, 450), 출근 [450, 570), 낮 [570, 960), 귀가 [960, 1140), 저녁 [1140, 1201)입니다.
- 온라인 표는 β = 0.5 controller를 쓰고, S1·S2의 첫날 전체와 재생 전체로 4행입니다.

각 행의 평가 round 목록은 `evaluation_manifest.csv`에 적습니다. 빈 구간이나 누락된 seed가 있으면 그 행은 통과로 세지 않고 `INCOMPLETE`로 둡니다.

정확도는 R15와 같이 계산합니다. 평가 round마다 요청이 있는 device의 정확도를 평균하고, 그 값을 구간 안의 round에 대해 다시 평균합니다. Geometric ensemble with early exit는 R15 정의를 따릅니다. 구간 값은 그 구간에서 Confidence-based offloading이 보이는 offloading 비율에 해당하는 곡선 값이고, 시간대 값은 round별 soft correctness(R15 시간대 표와 같은 정의)의 평균입니다.

## 2. 후보와 비교 규칙

### 2.1 후보

- 고정 후보는 w ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0}와 r ∈ {none, 0.1, …, 0.9}의 90개 쌍입니다. 답은 argmax_c [w p_d(c) + (1 − w) p'_e(c; r)]이고, r = none이면 보정하지 않은 p_e를 씁니다. prior correction, a_k, clipping은 R15와 같습니다.
- 기존 DriftGate도 별도 후보로 둡니다. 이 후보는 r = 0.5를 쓰고, 가중치로는 R15의 causal entropy weight를 그대로 씁니다. 전체 offloading에서는 R13b F-auto 가중치를, 온라인 controller에서는 R15의 DriftGate-P 가중치를 씁니다. calibration 점수도 현재 요청에 실제로 쓸 그 w로 계산합니다.
- 점수가 최댓값과 1e−12 이내로 같은 후보가 여럿이면 다음 순서로 고릅니다.
  1. 기존 DriftGate를 먼저 고릅니다.
  2. 고정 후보 중에서는 w가 작은 쪽을 고릅니다.
  3. w가 같으면 r을 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, none 순서로 고릅니다. 이 순서는 0.5에 가까운 쪽을 먼저 두고, 거리가 같으면 작은 r을 먼저 둔 것입니다. `none`은 같은 w 안에서 마지막입니다.

### 2.2 비교 규칙과 strongest reference

비교 규칙은 R15의 11개 규칙과 R15 B1의 6개 변형을 합친 17개입니다.

- R15의 11개 규칙: Confidence-based offloading, Device only, Edge only, Probability average, Logit sum, Lower-entropy exit, Logit-entropy weighting, Geometric ensemble with early exit, Label-shift EM, Learned weight, DriftGate
- R15 B1의 6개 변형: no correction + adaptive w, correction + w 0.5, correction + dev w 0.2, corrected edge only, correction + learned weight, correction + product

정의와 계수는 R15 스크립트와 같고, 설정마다 다시 맞추지 않습니다. strongest reference는 각 행에서 이 17개 가운데 seed 평균이 가장 높은 규칙 하나입니다. seed마다 최고 규칙을 따로 고르지 않습니다. 새 selector와 oracle는 이 집합에 넣지 않습니다.

온라인 표에서는 각 규칙이 offload된 요청에 답하는 연산을 같은 controller에 적용합니다. 연산이 같은 규칙은 하나로 합칩니다. Confidence-based offloading과 Edge only는 edge 답 하나로 합치고, Geometric ensemble은 Logit sum으로 합칩니다. Device only는 offloading 0인 기준점입니다. 이 기준점도 strongest reference 후보에 넣어서 판정을 보수적으로 둡니다. Label-shift EM과 Learned weight는 R15의 전체 요청 정의를 그대로 씁니다. 이 정의는 offload되지 않은 요청의 edge 출력도 사용하므로, 두 규칙이 온라인 표에서 실제보다 유리할 수 있습니다.

## 3. 판정

각 행의 차이는 새 방법의 seed 평균 정확도에서 strongest reference의 seed 평균 정확도를 뺀 값(pp)입니다. 같은 seed끼리 뺀 차이로 표준편차를 구합니다. 반올림한 값으로 판정하지 않고, 내부 부동소수점 비교 오차 1e−8 pp만 허용합니다.

| 판정 | 조건 |
|---|---|
| `PASS` | 모든 필수 행에서 차이 ≥ −1e−8 pp |
| `HOLD` | 차이 < −1e−8 pp인 행이 있지만, 모든 행에서 차이 ≥ −0.2 − 1e−8 pp |
| `NO_GO` | 차이 < −0.2 − 1e−8 pp인 행이 하나 이상 |
| `INCOMPLETE` | 필요한 행의 결과나 seed가 없음 |

기본 표, 시간대 표, 온라인 표를 각각 판정합니다. 세 표가 모두 `PASS`여야 전체 통과입니다. 모든 행에서 차이의 절댓값이 1e−8 pp 이하이면, 개선이 확인되지 않았다고 따로 표시합니다. 판정 대상은 주 후보 S-C입니다. 다른 selector의 판정도 같은 방식으로 계산해서 참고용으로 보고하지만, 결과를 본 뒤 주 후보의 이름이나 정의를 바꾸지 않습니다.

1단계에서 S-C의 결과는 다음과 같이 정리합니다.

- 세 표가 모두 `PASS`이고 실제 calibration 경로가 있으면 `GO_TO_IMPLEMENTATION_CHECK`입니다.
- 세 표가 모두 `PASS`이지만 경로가 없으면 `BLOCKED_CALIBRATION`입니다.
- 어느 표든 `NO_GO`이면 `SIMULATION_BELOW_CRITERION`입니다. 이는 모사 가설이 현재 채택 기준에 미달한다는 뜻입니다.
- 그 밖의 경우는 `HOLD` 또는 `INCOMPLETE`로 두며, `GO`로 바꾸지 않습니다.

이 판정은 모사 결과에 대한 판정이며, 논문 채택 판정이 아닙니다. 기존 DriftGate 논문의 성립 조건으로 소급해 적용하지도 않습니다.

## 4. 1단계 계산 규칙

### 4.1 시작 확인

R15 A1에 쓴 첫날 기록 34개에서 R15 A1의 11개 규칙과 B1의 6개 변형을 다시 계산합니다. 대상은 전체 구간입니다. 계산한 값은 R15 캐시의 같은 값과 정확도 비율 단위로 1e−12 이내에서 같아야 합니다. R15 캐시가 없는 run은 `R15_A1_rules_full_per_seed.csv`의 소수점 넷째 자리 반올림 단위에서 비교합니다. 재학습 변동에 쓰는 0.05 pp 허용치는 쓰지 않습니다. Round 15 재학습 기록과 재생 기록 16개도 같은 방법으로 R15 캐시와 대조합니다. 하나라도 불일치하면 selector 결과를 만들지 않고 원인부터 확인합니다.

### 4.2 모사 calibration 자료

device k의 평가 round e에서는, e보다 앞선 평가 round 가운데 device k가 받은 요청에 Main과 non-Main이 각각 8개 이상 있는 가장 최근 round e′를 고릅니다. 그 round의 요청 전체와 label을 calibration 표본으로 씁니다. 각 표본에는 round e′의 출력, prior, cell을 그대로 씁니다. 조건을 만족하는 round가 없으면 기존 DriftGate로 답합니다.

현재 요청과 현재 round의 정답은 선택에 쓰지 않습니다. 다음 항목을 기록합니다.

- 사용한 round와의 차이: 평가 round 수와 학습 round 수
- model이 바뀌었는지 여부: 첫날 기록에서는 평가 round 사이에 학습이 있으므로 항상 바뀝니다. 재생에서는 바뀌지 않습니다.
- cell 집합이 바뀌었는지 여부
- 반복 이미지 비율: 현재 요청 가운데 calibration 표본과 같은 이미지인 요청의 비율

요청 생성 규칙에 따라, 한 run 안에서 같은 label과 같은 class 내 순번을 가진 요청은 같은 이미지입니다. 반복 이미지 비율은 이 대응을 이용해 계산합니다.

### 4.3 Main 비율 추정 (전체 offloading)

T(x)는 보정 전 edge 예측이 M_k에 속하면 1입니다. calibration 표본에서 c11 = P(T = 1 | Main), c01 = P(T = 1 | non-Main)을 구합니다. ŝ는 clip_[0.1, 0.9]((q − c01)/(c11 − c01))입니다. q는 다음 window에 있는 앞선 요청들의 T 평균입니다.

window는 R15의 period window를 따르되, 현재 model과 cell 집합으로 얻은 관측만 남깁니다. R15의 period window는 현재 평가 round에서 앞선 요청이 8개 이상이면 그 요청들이고, 8개 미만이면 직전 평가 round의 요청 전체입니다.

- 첫날 기록에서는 평가 round 사이에 model이 바뀝니다. 따라서 직전 round의 관측은 버리고, 현재 round에서 앞선 요청만 씁니다. 앞선 요청이 1–7개뿐이어도 그 요청들로 q를 계산합니다.
- 재생에서는 device의 cell 집합이 직전 평가 round와 같을 때만 직전 round의 요청을 씁니다. cell 집합이 다르면 첫날 기록과 같이 현재 round에서 앞선 요청만 씁니다.
- 현재 요청은 window에 넣지 않습니다. 답을 정한 뒤에 관측에 추가합니다.

기존 DriftGate의 entropy window는 R15 정의를 그대로 유지합니다.

다음 경우에는 기존 DriftGate로 답합니다. 이때 0.5를 추정치처럼 쓰지 않습니다.

- window에 관측이 없는 경우
- calibration 표본이 없는 경우
- |c11 − c01| < 0.1인 경우
- 계산한 값이 유한하지 않은 경우

c11 − c01이 음수여도 절댓값이 0.1 이상이면 식에 그대로 넣습니다. clipping 비율과 fallback 비율은 원인별로 기록합니다.

### 4.4 Selector

J(a; s) = s·Acc_M(a) + (1 − s)·Acc_N(a)입니다. Acc_M과 Acc_N은 calibration 표본에서 후보 a가 Main 요청과 non-Main 요청에 낸 정확도입니다.

| 이름 | 규칙 | fallback |
|---|---|---|
| S-A | 고정 s로 90개 쌍과 DriftGate 중 J가 가장 큰 후보 | calibration 없음 |
| S-B0 | r = r(ŝ), w = 0 | 4.3절의 조건 |
| S-B5 | r = r(ŝ), w = 0.5 | 4.3절의 조건 |
| S-Bg | r = r(ŝ)인 9개 w와 DriftGate 중 J(·; ŝ)가 가장 큰 후보 | 4.3절의 조건 |
| S-C (주 후보) | 90개 쌍과 DriftGate 중 J(·; ŝ)가 가장 큰 후보 | 4.3절의 조건 |

- r(ŝ)는 1 − ŝ에 가장 가까운 숫자 r 격자값입니다. 거리가 같으면 0.5에 가까운 값을 고르고, 그래도 같으면 작은 값을 고릅니다. 선택 점수와 실제 예측에는 같은 r을 씁니다.
- S-A의 s는 개발 기록(S1 seed 5–7)에서 정합니다. s = 0.5와 s = 0.65로 S-A를 계산하고, `day1_full` 정확도의 seed 평균이 더 높은 s를 고릅니다. 두 평균의 차이가 1e−12 이하이면 0.5를 고릅니다. 고른 s는 모든 설정에 공통으로 적용합니다.

### 4.5 온라인 controller (β = 0.5)

모든 방법에 R15 C2 controller의 offloading 결정을 똑같이 적용합니다.

- τ는 device의 직전 128개 요청에서 계산한 device entropy의 중앙값입니다. 직전 요청이 16개 미만이면 τ = 0.8 nats로 둡니다. 현재 요청은 τ 계산에 넣지 않습니다.
- offload하지 않은 요청에는 device exit의 답을 씁니다.
- offload한 요청에서는 Z_j = O_j·T_j로 둡니다. O_j는 요청 j가 offload되었는지를 나타내고, T_j는 4.3절과 같이 보정 전 edge 예측이 M_k에 속하는지를 나타냅니다. window는 4.3절과 같은 규칙으로 앞선 모든 요청에서 만듭니다.
- calibration 표본에서 v_g(τ_j) = P(H_d > τ_j, T = 1 | g)와 u_g(τ) = P(H_d > τ | g)를 구합니다. 여기서 g는 Main 또는 non-Main입니다.
- ŝ_all = clip((q_Z − v̄_N)/(v̄_M − v̄_N))이고, ŝ_off = clip(ŝ_all·u_M/(ŝ_all·u_M + (1 − ŝ_all)·u_N))입니다. clip 범위는 [0.1, 0.9]입니다.
- Acc_M과 Acc_N은 현재 τ에서 H_d > τ인 calibration 표본으로 구합니다.
- 다음 경우에는 DriftGate-P로 답합니다.
  - calibration의 두 그룹 또는 H_d > τ로 고른 두 그룹 가운데 8개 미만인 그룹이 있는 경우
  - window 안에서 앞서 실제로 offload된 요청이 8개 미만인 경우
  - |v̄_M − v̄_N| < 0.1인 경우
  - 분모가 0이거나 값이 유한하지 않은 경우

추정 오차는 두 가지로 나누어 기록합니다. 하나는 ŝ_all과, 같은 window의 실제 Main 비율의 차이입니다. 다른 하나는 ŝ_off와, 현재 device-round에서 offload된 요청의 실제 Main 비율의 차이입니다.

### 4.6 Oracle (정답을 쓰는 진단)

- O1은 설정·구간마다 seed 평균이 가장 높은 격자 쌍 하나입니다.
- O2는 device-round마다, 그 device-round의 요청에서 정확도가 가장 높은 격자 쌍입니다. 이 값은 격자 안에서 device-round 동안 한 쌍을 유지하는 규칙들의 상한일 뿐입니다.
- O3는 device-round의 실제 Main 비율 s_true로 r = clip(1 − s_true, 0.1, 0.9)를 정하는 진단입니다. r은 연속값을 그대로 쓰고, w = 0, w = 0.5, 그리고 구간 전체에서 seed 평균이 가장 높은 격자 w를 비교합니다.

각 oracle은 기존 DriftGate, Probability average, corrected edge only, strongest reference와 seed끼리 비교합니다. oracle은 strongest reference 집합에 넣지 않습니다.

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `tables/R16_T0_start_check.csv` | 시작 확인 |
| `tables/R16_R_references.csv` | 행마다 17개 비교 규칙의 정확도와 strongest reference |
| `tables/R16_S_selectors.csv`, `tables/R16_S_selectors_per_seed.csv` | 기본 표 42행 (selector와 기존 DriftGate) |
| `tables/R16_S_time.csv`, `tables/R16_S_time_per_seed.csv` | 시간대 표 20행 |
| `tables/R16_S_online.csv`, `tables/R16_S_online_per_seed.csv` | 온라인 표 4행 |
| `tables/R16_S_supplementary_R12.csv` | Round 12 S1·S2 첫날 기록의 보조 표 (판정 제외) |
| `tables/R16_S_estimation.csv` | 추정 오차, clipping, fallback, 과거 자료의 오래된 정도, 반복 이미지 비율, 선택된 w와 r |
| `tables/R16_S_dev_sA.csv` | S-A의 s 선택 |
| `tables/R16_O_oracles.csv`, `tables/R16_O_oracles_per_seed.csv` | O1, O2, O3 |
| `tables/R16_S_judgement.csv` | 표마다의 판정 |
| `decision_stage1.json`, `R16_stage1_report_ko.md` | 1단계 판정과 보고서 |
