# DriftGate Round 13b 보고서: 관련 연구 기준선, 결합의 보완, 추가 구조와 조건

- 작성: 2026-10-04, 서버 `honeynaps`(RTX 4090 4장, GPU 0–3). 다른 프로젝트의 프로세스와 GPU, CPU, 메모리를 함께 썼다.
- A부는 새 run 없이 계산했다. 입력은 개발용 기록(Round 10, seed 5–7)과 Round 12의 run 50개다.
- B부의 새 run은 15개다(`runs/phaseT13b_arch/r13b_*`). 2026-10-04 16:29에 시작해 19:14에 모두 첫 시도에 끝났다. 조건 2는 돌리지 않았다(8.3절 1번).
- 코드 commit은 두 개다.
  - `6b505a7`: B부 실행 코드(ResNet-20, VGG-11, CIFAR-100). run을 띄우기 전에 commit했다.
  - `49a19f4`: 분석 스크립트 `scripts/r13b_baselines_analysis.py`. 결과를 보기 전에 commit하고 push했다.
- 결과를 본 뒤 분석 스크립트를 고치지 않았다. nan으로 나온 값 하나를 계산 오류로 의심해 고쳐 보았지만, 바뀐 값이 없어서 되돌렸다(8.3절 8번).
- 산출물은 `journal_expansion/artifacts/driftgate_tmc_r13b_baselines/`에 있다.

## 0. 이 보고서에서 쓰는 말

- **exit**: 모델 중간에 붙은 분류기다. client exit는 기기의 client block 끝에, server exit는 edge server의 server block 끝에 있다. p_c와 p_s는 두 exit의 class 확률이다.
- **요청 종류**: M_k는 클라이언트 k가 학습한 class의 목록이다.
  - Main은 정답이 M_k 안에 있는 요청이다.
  - OOP는 지금 cell의 다른 클라이언트만 학습한 class의 요청이다.
  - OOR은 그 밖의 class의 요청이다.
- **집과 밖**: 클라이언트가 자기 집 cell에 있으면 집, 다른 cell에 있으면 밖이다. S1, S2, B부처럼 Round 6 경로로 돈 run에만 있다.
- **보정한 server 확률 p'_s**: server exit가 학습할 때의 사전 확률(π_tr)에서 생긴 치우침을 지운 확률이다. a는 π_tr에서 M_k가 차지하는 몫이다. 보정 세기 r = 0.5를 쓴다.
- **기존 규칙**
  - B0은 entropy routing이다. client exit의 entropy가 0.8 nats 이하이면 client exit의 답을, 넘으면 server exit의 답을 쓴다. 지금 원고의 추론 방식이다.
  - B1은 client exit만, B2는 server exit만 쓴다.
  - B3은 두 exit 확률의 평균에서 가장 큰 class를 답으로 쓴다.
  - F(결합)는 0.2·p_c + 0.8·p'_s에서 가장 큰 class를 답으로 쓴다.
  - 종류 oracle은 정답 요청 종류를 아는 참고 규칙이다. Main 요청에는 client exit, 나머지 요청에는 server exit의 답을 쓴다.
- **관련 연구의 기준선**: 두 exit의 기록된 확률로 계산했다.
  - R-PoE는 FedRoD처럼 두 head의 logit을 더해 답한다. 여기서는 두 확률의 log를 더한다.
  - R-PoE-bal은 R-PoE에서 server 확률을 p'_s로 바꾼 것이다. FedRoD의 class 균형 손실을 사후 보정으로 근사했다.
  - R-THE는 FedTHE를 따른 규칙이다. 요청마다 entropy가 작은 exit의 답을 쓴다.
  - R-ZTW는 Zero Time Waste를 따른 곡선이다. client exit의 최대 확률이 q 이상이면 client exit의 답을 쓴다. 그렇지 않으면 server로 보내고 R-PoE의 답을 쓴다. 표 1에는 server 사용 비율이 B0과 같은 지점의 값을 적었다.
  - R-EM은 Round 10의 M1이다. 같은 평가 라운드에서 앞서 도착한 요청으로 EM을 돌려 Main 아닌 요청의 비율을 추정한다. 그 비율로 보정한 server 확률의 답을 쓴다.
  - R-LR은 Round 11의 R-결합이다. 로지스틱 회귀로 요청마다 P(Main)을 구하고, P·p_c + (1 − P)·p_s의 답을 쓴다. 회귀는 개발용 기록으로 한 번만 학습했다.
- **F의 절제와 보완**
  - F-nodebias는 보정 없는 F(w = 0.2)다.
  - F-eq는 w = 0.5인 F다.
  - F-auto는 w를 label 없이 정한 F다. 앞서 도착한 요청(창)에서 두 exit의 평균 entropy를 구하고, w = H̄_s / (H̄_c + H̄_s)로 둔다.
  - F-gate는 p_s(M_k) < t인 요청에 server exit의 답을 쓰고, 나머지 요청에 F의 답을 쓴다. t는 개발용 기록으로 0.05를 골랐다.
- **정확도**: 평가 라운드마다 클라이언트 정확도를 같은 비중으로 평균하고, 평가 라운드에서 다시 평균한다.
  - 후반 구간은 학습 라운드 30 이후의 평가 라운드다. 판정은 이 구간으로 한다.
  - 전체 구간은 모든 평가 라운드다.
  - pp는 정확도 백분율의 차이다. 값은 seed 평균이다.
- **server 사용 비율**: server exit의 출력이 필요한 요청의 비율이다. B1은 0, B0은 entropy가 0.8을 넘는 요청의 비율이다. 결합 규칙과 R-THE는 1이다.
- **곡선**: 모두 server 사용 비율에 따른 정확도다.
  - E는 B0의 문턱 τ를 바꾼 곡선이다.
  - SF는 entropy가 τ 이하이면 client exit, 넘으면 F의 답을 쓰는 곡선이다.
  - SB3은 SF에서 F 대신 B3을 쓴 곡선이다.

## 1. 판정 (시험용 기록, 후반 구간, seed 평균, 고정 λ 0.4)

| 판정 | 답 | 근거 |
|---|---|---|
| 1. 관련 연구의 기준선 가운데 F 이상인 규칙 | S1: 없음. S2: R-PoE-bal(+0.17 pp) | S1에서 기준선 − F는 R-PoE −0.88, R-PoE-bal −1.03, R-THE −2.53, R-EM −2.56, R-LR −1.27 pp다. S2에서는 R-PoE −0.89, R-PoE-bal +0.17, R-THE −3.13, R-EM −1.37, R-LR −2.48 pp다. |
| 1. 같은 확인, B부 | 다섯 묶음 모두 F 이상인 기준선이 있다 | ResNet-20 얕은 분할: R-PoE +0.90, R-PoE-bal +3.23, R-THE +1.61. ResNet-20 중간 분할: R-PoE +0.11, R-PoE-bal +0.36, R-THE +0.27. VGG-11 얕은 분할: R-PoE +0.74. VGG-11 중간 분할: R-PoE +1.39, R-THE +0.37, R-LR +0.39. CIFAR-100: R-PoE-bal +0.45 pp. |
| 2. F-auto 채택 | **아니오** | F < max(B1, B3)인 묶음 7개 가운데 F-auto > max(B1, B3)인 묶음은 ResNet-18 고정 0.4(64.02 대 64.00) 하나뿐이다. 나머지 묶음에서도 F-auto가 F − 0.3 pp보다 낮은 묶음이 다섯 개 있다: S1 고정 −0.42, S1 DriftGate −0.38, client mobility 고정 −0.35, K = 200 −0.73, S1-fast −0.50 pp. |
| 3. F-gate 채택 | **아니오** | t = 0.05. 밖 정확도의 F-gate − F는 S1 +0.03, S2 +0.02 pp다(기준 1.0 pp 이상). 전체 정확도의 차이는 S1 +0.02, S2 +0.01 pp다. |
| 4. 구조 (판정 없음) | | F − B0: ResNet-20 얕은 분할 +19.57, ResNet-20 중간 분할 +5.42, VGG-11 얕은 분할 +5.49, VGG-11 중간 분할 +1.30 pp. F − B3: 각각 −2.09, −0.30, +0.02, −1.08 pp. |
| 5. 조건 (판정 없음) | | F − B3: 조건 1(CIFAR-100) +2.82 pp, 기본 S1(Round 12, seed 0–4) +0.83 pp. 조건 2는 돌리지 않았다. |

- 판정 값과 근거는 `decision.json`에 있다.
- 판정 2에서 "F < max(B1, B3)인 묶음"은 정의대로 계산했다. 시험용 묶음 19개(Round 12의 14개와 B부의 5개) 가운데 7개가 해당한다.
  - Round 12: ResNet-18 두 묶음, client mobility의 DriftGate, K = 500
  - B부: ResNet-20 두 묶음, VGG-11 중간 분할
- 지시문의 괄호는 Round 12에서 ResNet-18 두 묶음만 해당한다고 적었다. 괄호대로 묶음을 나눠도 답은 "아니오"다(`decision.json`의 `for_information_parenthesis_reading`).

## 2. 표 1: 규칙별 정확도 (`tables/R13b_T1_rules.csv`, seed별 값은 `R13b_T1_rules_per_seed.csv`)

<<T1>>

규칙별 server 사용 비율은 `tables/R13b_T1_server_use.csv`에 있다.

## 3. 표 2: 집, 밖, 요청 종류별 정확도 (후반 구간, `tables/R13b_T2_splits.csv`)

<<T2>>

## 4. 표 3: server 사용 비율에 따른 정확도 (`tables/R13b_T3_curves.csv`, `R13b_T3_SF_ratios.csv`)

<<T3>>

- 두 비율의 seed별 값은 `R13b_T3_SF_ratios.csv`에 있다. B부 묶음의 곡선 값과 전체 구간의 곡선은 `R13b_T3_curves.csv`에 있다.
- nan은 곡선 SF가 곡선 E의 최고 정확도에 끝내 닿지 못한 seed가 있다는 뜻이다. 그런 seed는 ResNet-20 얕은 분할의 seed 0, 2와 중간 분할의 seed 2다.
  - 이 seed들에서 곡선 E의 최고점은 server 사용 0.0001–0.025인 지점에 있다.
  - 그 높이는 SF의 최고점보다 각각 0.094, 0.0001, 0.0007 pp 높다.
  - 나머지 seed에서는 SF가 server 사용 0에서 이미 닿는다.

## 5. 표 4: 구조와 조건 (`tables/R13b_T4_structures.csv`)

<<T4>>

- FLOPs는 이미지 하나의 값이다. 합성곱과 행렬곱만 세고, 곱셈과 덧셈을 각각 하나로 센다. batch normalization, 활성화, pooling, 덧셈은 세지 않았다.
- smashed data는 요청 하나를 server로 보낼 때 client block의 출력(float32)이다.
- 학습 시간과 평가 시간은 run 평균이다. GPU마다 run을 최대 5개 함께 돌렸고 다른 프로젝트도 같은 자원을 썼다. 그래서 같은 설정에서도 시간이 1.5배까지 차이가 난다.
- a는 π_tr에서 M_k가 차지하는 몫의 평균이다.
- 기본 CNN의 client exit에는 합성곱층이 하나 있다. 그래서 선형층 하나인 ResNet-20, VGG-11의 client exit보다 parameter와 FLOPs가 훨씬 크다.

## 6. 그림 (`figures/`, 영문 caption은 `figures/R13b_figure_captions.md`)

- `R13b_fig1_home_away.png`, `.pdf`: S1과 S2에서 규칙별 (집 정확도, 밖 정확도)의 산점도다.
- `R13b_fig2_curves.png`, `.pdf`: S1과 S2의 곡선 E, SF, SB3, R-ZTW다. 점선은 B0의 server 사용 비율이다.
- `R13b_fig3_structures.png`, `.pdf`: 구조와 분할 위치별 B0, B1, B2, B3, F의 막대다(후반 구간). Round 12의 기본 S1 CNN과 ResNet-18(단계적 구성 변화 시나리오)을 함께 넣었다.

## 7. 이득과 손실이 생긴 곳

값은 모두 후반 구간에서 (규칙 − F)이고, 단위는 pp다. 순서는 집, 밖, Main, OOP, OOR이다.

**관련 연구의 기준선 (S1, S2)**

- R-PoE-bal과 R-EM은 Main 요청에서 F보다 높고, OOP와 OOR 요청과 밖에서 더 크게 잃는다.
  - S1 R-PoE-bal: +0.23, −2.16, +1.23, −6.22, −4.12
  - S1 R-EM: −0.14, −4.84, +1.29, −16.30, −9.17
- S2에서 R-PoE-bal이 F보다 높은 이유는 집(+0.56)과 Main(+1.47)의 이득이 밖(−0.81), OOP(−5.19), OOR(−1.60)의 손실보다 컸기 때문이다.
- R-PoE, R-THE, R-LR은 반대 방향이다. OOP와 OOR에서 F보다 높고, 집과 Main에서 잃는다.
  - S1 R-PoE: −2.06, −0.01, −3.01, +2.74, +2.77
  - S1 R-THE: −6.52, +0.56, −9.81, +12.06, +9.08
  - S1 R-LR: −4.02, +1.08, −6.03, +8.45, +6.66
- R-ZTW는 B0과 같은 server 사용 비율(S1 0.91)에서 R-PoE와 거의 같다. 이 지점에서는 대부분의 요청이 server로 가기 때문이다.

**F의 보완 (S1, S2)**

- F-auto의 w는 S1에서 평균 0.42이고, 고정값 0.2보다 크다. 그래서 client exit의 비중이 커진다.
  - S1 F-auto: +0.31, −1.09, +0.83, −3.11, −2.18
  - Main과 집에서 얻은 것보다 OOP와 OOR에서 잃은 것이 커서 전체로는 −0.42 pp다.
- F-gate(t = 0.05)는 답을 바꾸는 요청이 적다. 다섯 칸 모두 차이가 0.06 pp 이하다.

**구조 (B부)**

- ResNet-20에서는 server exit가 약하다. 후반 구간의 B2는 얕은 분할 33.59%, 중간 분할 51.83%로 B1(58.07%, 57.84%)보다 낮다.
  - F는 server 쪽에 0.8의 비중을 두므로 B1과 B3보다 낮다. 얕은 분할에서 B3 − F는 +2.20, +1.90, +1.88, +1.75, +1.92로 다섯 칸 모두 양수다.
  - 두 분할 모두 OOP와 OOR 정확도는 어느 규칙에서나 17% 이하다.
- VGG-11 얕은 분할에서 F와 B3은 거의 같다(F − B3 = +0.02).
  - R-PoE는 F보다 0.74 pp 높다. 차이는 +0.87, +0.57, +0.76, −0.21, −0.51로 집, 밖, Main에서 생겼다.
- VGG-11 중간 분할에서 F는 B3보다 1.08 pp 낮다.
  - B3 − F는 +0.35, +1.76, +0.03, +3.27, +1.59다.
  - 손실은 밖과 OOP, OOR에서 생겼다. Main에서는 같다.
- server 사용 비율이 줄어들 때를 보면, 곡선 SF는 B0의 정확도에 다음 server 사용 비율에서 처음 닿는다.
  - ResNet-20: 0
  - VGG-11: 얕은 분할 0.20, 중간 분할 0.22
  - S1 기본 CNN: 0.23
  - CIFAR-100: 0.39

**조건 1 (CIFAR-100)**

- F − B3 = +2.82 pp로, 기본 S1의 +0.83 pp보다 크다.
  - F − B3은 +5.61, +0.74, +7.60, −7.11, −4.55다.
  - 이득은 집과 Main에서 생겼고, 손실은 OOP와 OOR에서 생겼다.
- 클라이언트당 Main class는 20개다(기본 S1은 2개).
- a의 평균은 0.246으로 기본 S1의 0.255와 거의 같다.
  - 지시문 4.2는 클라이언트의 class가 cell 전체에서 차지하는 몫이 작을수록 보정의 효과가 클 것이라고 예상했다.
  - 이 조건에서는 그 몫이 거의 줄지 않았다. 따라서 F − B3이 커진 원인을 그 몫의 차이로 볼 수는 없다.
- B0은 server 사용 비율이 99.5%다. class가 100개라 client exit의 entropy가 거의 언제나 0.8을 넘기 때문이다.

**곡선 (S1, S2)**

- S1에서 SF는 B0의 정확도(64.67%)에 server 사용 0.23에서 처음 닿는다. B0 자신은 server 사용 0.91에서 이 정확도를 낸다.
- S2에서는 SF가 server 사용 0.01에서 B0의 정확도에 닿는다.
- R-ZTW는 server 사용이 적을 때 SF보다 조금 높다. S1에서는 0.1–0.3에서 높고(0.3에서 65.25% 대 65.15%), S2에서는 0.1–0.2에서 높다(0.2에서 70.15% 대 70.10%). 그보다 server 사용이 크면 SF가 높다.

## 8. 실행, 확인, 구현, 불일치 기록

### 8.1 개발용 기록에서 정한 값

<<DEVT>>

- t = 0.05와 t = 0.02, 0.1의 차이는 0.01 pp 이하다.
- R-LR은 개발용 기록의 후반 구간 요청 9,700,604개로 한 번 학습했다. 계수는 `tables/R13b_dev_LR.csv`에 있다.
- 개발용 기록에서 다시 계산한 값은 이전 라운드의 값과 같다.
  - B0, B1, B2, B3(63.69, 61.52, 63.75, 65.86)는 Round 10과 같다.
  - R-EM(64.10)은 Round 10 M1의 seed별 값까지 같다.
- 개발용 기록에서 규칙별 값은 `tables/R13b_dev_rules.csv`에 있다. R-LR과 F-gate는 이 기록으로 정했으므로 이 표의 값은 학습에 쓴 자료에서 잰 값이다.

### 8.2 일치 확인

- **시작 전 확인 (지시문 1.3)**: Round 12 기록으로 다시 계산한 B0, B3, F는 Round 12 표 1의 후반 구간 값과 모든 묶음에서 0.05 pp 안에서 같다. 최대 차이는 0.0049 pp이고, Round 12 표가 소수 둘째 자리로 반올림된 탓이다(`tables/R13b_T0_r12_recompute_check.csv`).
- **기록 확인**
  - run 65개 모두에서 다시 계산한 B0(전체 구간)은 run JSON의 정확도와 같다.
  - M_k는 모든 평가 라운드에서 일관된다. Round 6 경로에서는 provenance 기록과도 같다.
  - Round 5 경로 run의 요청별 기록과 평가 함수의 불일치는 0건이다(`tables/R13b_T0_checks_manifest.csv`).
- **기본 S1 run의 bit 단위 동일성 (지시문 4.1)**
  - Round 13b 코드(`6b505a7`)의 기본 S1 run을 이전 코드(`23aab8f`)와 CPU에서 비교했다. 조건은 seed 0, 2 라운드, 평가는 매 라운드다.
  - 기록 flag를 끈 경우와 켠 경우 모두 run JSON, trace 배열, 요청별 기록이 같다(`precheck/bit_identity_check_default_S1.txt`).
  - 비교에서 뺀 항목은 run 이름, run_id, 실행 시간, probe 계산 시간(`probe_us`), 환경 파일 경로다. 두 환경 파일의 sha256은 같다.
- **테스트**: 기존 107개와 새 구조 테스트 4개를 합친 111개가 통과했다(`precheck/tests_before_launch.txt`).
- **실행 시점과 종료 뒤의 코드**: sha256이 같다(`precheck/code_identity_launch.txt`, `code_identity_after.txt`).
- **기록에서 B0과 F를 계산할 수 있는지 (지시문 4.3)**
  - run의 요청별 기록은 run이 끝날 때 한 번에 저장된다. 그래서 "첫 run이 평가 라운드 하나를 마친 뒤"에는 확인할 수 없다.
  - 대신 실행 전에 구조와 조건마다 2 라운드짜리 run(seed 0)을 돌리고, 그 기록에서 B0과 F를 계산했다. B0은 run JSON의 값과 같았다(`precheck/smoke_b0_f_from_records.txt`).
  - 이 짧은 run은 결과에 쓰지 않았다.
- **분석 스크립트의 smoke test (결과 전)**: Round 12 기록에서는 값을 보지 않았다. 유한성과 범위만 확인했고, 계산 일부는 직접 반복문과 대조했다(`precheck/analysis_smoke_blind.txt`).

### 8.3 구현에서 정한 것과 불일치 기록

1. **조건 2를 돌리지 않음 (지시문 4.2)**
   - 분할 코드(`nd1_partition`)는 클라이언트당 Main class 수를 `max(1, int(10 × 0.2)) = 2`로 정한다.
   - 이 값을 1 줄이면 1이 되는데, 지시문은 2보다 작게 하지 말라고 정한다. 그래서 값이 기본값 2로 남는다.
   - 분할 코드에는 Dirichlet 계수(class 치우침의 정도를 정하는 분포의 매개변수)도 없다.
   - 이 경우가 지시문의 "둘 다 해당하지 않으면 돌리지 않는다"에 가장 가깝다고 보고 돌리지 않았다. 그래서 B부의 run은 18개가 아니라 15개다.
2. **구조**
   - ResNet-20의 stage가 바뀌는 곳의 shortcut은 저장소의 BasicBlock을 따라 1×1 합성곱 사상으로 두었다. He 등(2016)의 CIFAR 판은 영 채움 shortcut을 쓴다. client exit를 뺀 전체 parameter는 272,474개로, 원 논문의 0.27M과 같은 규모다. VGG-11은 9,231,114개다.
   - VGG-11의 합성곱에는 bias를 두었다(torchvision의 vgg11_bn과 같음).
   - exit는 ResNet-18 중간 분할 구현을 따랐다. client exit는 global average pooling과 선형층, server exit는 남은 층, global average pooling, 선형층이다.
   - 최적화 설정은 기본 S1과 같다(SGD, 학습률 0.01, weight decay 1e-4, batch 32, local epoch 3).
   - 발산한 run이 없어서 학습률을 낮추지 않았다.
3. **CIFAR-100**: 분할 코드는 class 수를 인자로 받으므로 고치지 않았다. 데이터는 `src/datasets_ext.get_dataset("cifar100")`(CIFAR-100 정규화 값)으로 읽었다. 모델은 S1 기본 CNN이고 출력만 100 class다.
4. **창의 "직전 평가 라운드"**: 참여율 0.5, client mobility, K = 500처럼 어떤 평가 라운드에 요청이 없는 클라이언트가 있다. 이때 창의 "직전 평가 라운드"는 그 클라이언트가 마지막으로 요청을 받은 평가 라운드로 두었다. F-auto, R-LR의 x_SR, R-EM의 시작값에 모두 같은 규칙을 쓴다.
5. **R-ZTW**
   - q의 격자(0–1)만으로는 server 사용 비율이 1에 닿지 않는다. float16으로 저장한 최대 확률이 1.0인 요청이 있기 때문이다.
   - 그래서 모든 요청을 server로 보내는 끝점(= R-PoE)을 곡선에 더했다. 곡선 E의 τ = 0과 같은 역할이다.
   - 표 2의 R-ZTW 값은 B0과 같은 server 사용 비율에서 이웃한 두 q의 정답 여부를 같은 비중으로 보간해 구했다.
6. **entropy**
   - R-THE와 F-auto는 client exit의 entropy로 기록의 값(B0이 0.8과 비교하는 값)을 쓴다. server exit의 entropy는 기록된 확률로 계산했다.
   - R-THE에서 두 entropy가 같으면 client exit의 답을 쓴다.
7. **판정 2의 묶음**: 1절 아래에 적었다.
8. **결과를 본 뒤 분석 스크립트를 고친 곳**
   - "곡선 SF가 곡선 E의 최고 정확도에 처음 닿는 server 사용 비율"이 ResNet-20 두 묶음의 seed 평균에서 nan으로 나왔다.
   - 처음에는 이렇게 진단했다. 곡선 E의 최고점이 server 사용 0이고, SF와 E의 그 지점 값이 다른 순서의 합이라 1e-16만큼 달라 비교가 실패했다는 것이다. 그래서 비교에 1e-9의 허용 오차를 두는 수정을 넣고 run 65개를 모두 다시 계산했다.
   - 그러나 바뀐 값이 하나도 없었다. 표, `decision.json`, PNG 그림이 모두 같았다.
   - 다시 확인해 보니 nan은 실제 결과였다. 4절 표 아래에 적은 대로, 곡선 E의 최고점이 server 사용 0보다 조금 큰 지점에 있고 SF가 그 높이에 닿지 못한다.
   - 그래서 수정을 되돌렸다. 이 보고서의 산출물은 모두 결과 전에 commit한 스크립트(`49a19f4`)로 만든 것이다.
9. **메모리 부족**
   - 17:32와 17:34에 커널이 분석 작업자 두 개를 메모리 부족으로 종료했다. 직후에 재 보니 다른 프로젝트의 프로세스가 약 61 GB를, B부 run 15개가 약 36 GB를 쓰고 있었다.
   - 작업자 수를 줄여 다시 계산했고, 끝난 run은 캐시를 썼다. 값에는 영향이 없다.
   - 17:42에는 VS Code Insiders의 원격 서버 프로세스도 메모리 부족으로 종료되었다.
   - B부 run과 CPU 비교 run은 종료되지 않았다.

### 8.4 바뀐 파일과 줄

| 파일:줄 | 내용 |
|---|---|
| `journal_expansion/src/models_ext.py:197-291` | ResNet-20, VGG-11의 client와 server 모델, 분할 위치 |
| `journal_expansion/src/models_ext.py:309-325` | FlexModelFactory에 두 구조 추가 |
| `journal_expansion/src/models_ext.py:346` | `arch_stats`: parameter, FLOPs, smashed data. 난수 상태를 바꾸지 않는다. |
| `journal_expansion/src/runner_r6.py:257-262, 316, 323-327, 343-349, 652` | `dataset`, `model_family`, `split_point` 인자와 구조 통계 기록. 기본값에서는 이전과 같다. |
| `journal_expansion/scripts/run_r6.py:90-98, 125-126` | `--dataset`, `--model_family`, `--split_point`, `--learning_rate` |
| `journal_expansion/scripts/r13b_dispatch.py`, `r13b_supervisor.sh` | B부 run 15개의 실행 관리자와 cron supervisor |
| `journal_expansion/tests/test_r6.py:507-` | 새 구조의 모양, parameter 수, 난수 상태 테스트 |
| `artifacts/driftgate_tmc_r13b_baselines/scripts/r13b_baselines_analysis.py` | A부와 B부 분석 |
| `artifacts/driftgate_tmc_r13b_baselines/scripts/r13b_report_tables.py` | CSV를 이 보고서의 표로 옮기는 스크립트(계산 없음) |

## 부록 A: B부 run의 시간과 요청별 원본의 sha256 (원본은 서버에만 둠, `runs/phaseT13b_arch/<run>_evalprobs.npz`)

<<MANIFEST>>

## 부록 B: F-auto의 w

<<FAUTO>>
