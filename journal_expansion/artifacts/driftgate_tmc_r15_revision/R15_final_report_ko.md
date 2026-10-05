# DriftGate Round 15 보고서: 최종 검토, 보완, 원고 업데이트

- 작성: 2026-10-05, 서버 `honeynaps`(RTX 4090 4장).
- 분석 계획은 결과를 보기 전에 commit `58e839b`로 고정했다. 이후 commit은 다음과 같다.
  - A2 코드 `314aa2e`
  - bit 동일성 확인 `0f2550e`
  - 분석 스크립트 `67f35f9`(결과 전)
- 결과를 본 뒤 분석 스크립트(`scripts/r15_analysis.py`)는 고치지 않았다. 원고용 그림과 파생 표는 그 캐시를 읽는 별도 스크립트(`r15_paper_figures.py`, `r15_derived_tables.py`)로 만들었다.
- 산출물은 `journal_expansion/artifacts/driftgate_tmc_r15_revision/`에 있다.

## 0. 요약

**핵심 결과**

1. **첫날 전체 구간**: v27의 주표가 그대로 재현된다. Round 14와의 차이는 최대 0.00005 pp다. 새 규칙 Logit-entropy weighting을 더해 규칙이 11개가 되어도, DriftGate는 핵심 8개 설정 모두에서 평균이 가장 높다.
2. **학습 초반의 비중**: 첫날 우위의 상당 부분이 학습 초반에서 온다. S1에서 출근 전 6개 평가 시점(31개 중)이 차지하는 기여는 다음과 같다.
   - DriftGate − Confidence-based offloading 5.45 pp 중 3.39 pp
   - DriftGate − strongest baseline 0.95 pp 중 0.58 pp
3. **round > 30**: DriftGate가 1위인 설정은 8개 중 6개이고, strongest 대비 우위는 −0.21 ~ +1.14 pp다. round > 50에서는 5개이고, random mobility에서는 9위다.
4. **학습된 모델로 하루 재생(A2)**
   - S2: DriftGate가 1위다(Probability average보다 +0.64 pp).
   - S1: Probability average보다 0.33 pp 낮아 5위다. correction한 edge exit 단독(+0.79 pp)과 correction + w 0.2(+0.63 pp)도 DriftGate보다 높다.
5. **DriftGate의 구성 요소**
   - correction은 첫날 1.2–3.8 pp를 더한다(ResNet-18 0.2 pp). S1 재생에서는 오히려 0.5 pp를 깎는다.
   - adaptive w는 더 나은 고정 w와 비슷하다. run 안에서 w의 표준편차는 0.04이고, window 길이는 정확도에 거의 영향이 없다(0.02 pp 이하).
6. **offloading**: 첫날 β ≤ 0.5의 측정점에서 모든 full-offload 규칙을 넘는 설정은 5개다(v27은 7개). round > 30에서는 2개다.

**v27에서 바뀐 주장**

- "8개 모두에서 가장 정확하다"는 첫날 조건으로 한정했다.
- "7개 설정에서 절반 이하 offload"는 5개로 고쳤다.
- "Confidence-based offloading이 Device only보다 1.3 pp 낮다"는 삭제했다. 이 차이는 첫날에만 있다.
- oracle의 "visitor가 대가를 치른다"는 관찰로 바꿨다.
- FedTHE 설명은 원문의 식대로 고쳤다.
- correction과 product에 대한 단정을 조건부로 바꿨다.
- APFL 비교는 같은 seed로 맞췄다.
- 자세한 위치는 `R15_manuscript_changes_ko.md`에 있다.

## 1. 입력 확인

- 저장소에 `AGENTS.md`는 없다. `paper_writing_guidelines_v4(1).md`는 서버에서 찾지 못했다.
- v27 원고는 `journal_expansion/artifacts/DriftGate_TMC_v27.zip`에 있다(사용자가 넣은 미커밋 파일).
- 서버에 LaTeX가 없어서 tectonic 0.17.0을 `~/.local/opt/tectonic`에 내려받았다. v27도 이것으로 컴파일된다(12쪽).
- **시작 확인**: R15 스크립트로 다시 계산한 전체 구간 값은 Round 14 주표(R14_T1_main_long)의 100개 값과 최대 0.00005 pp 차이로 같다(`tables/R15_T0_start_check_vs_R14.csv`). run마다 다시 계산한 B0도 run JSON과 같다.
- **설정 대응**: v27의 "random mobility"는 Round 12의 "client mobility"(`--mobility`, Gauss–Markov, 120 라운드)와 같다. ResNet-18은 device 16개다. v27 표 2의 "50 devices"는 틀렸으므로 v28에서 고쳤다.
- **집계 방식**: 원고의 식 (4)(시점마다 device 평균, 다시 시점 평균)와 코드(`m13.weights`, 시점별 device 평균)와 표가 같은 집계를 쓴다. 요청 수로 가중한 pooled accuracy는 쓰지 않는다.

## 2. 작업 P: 비교 규칙

**확인된 사실**

- 비교 규칙의 실제 연산은 `R15_G_checks_ko.md`의 P1 표에 정리했다. 원고 표시 이름은 Confidence-based offloading, Probability average, Logit sum, Lower-entropy exit, Geometric ensemble with early exit, Label-shift EM(두 그룹 비율의 EM), Learned weight다.
- FedTHE 원문(§4.2)은 logit 가중합의 softmax에 entropy를 적용하고, feature alignment와 cosine 유사도로 정하는 비중을 함께 쓴다. v27의 "확률 혼합 entropy의 오목성" 논증은 FedTHE의 해가 아니다.
- **Logit-entropy weighting(LEW)**: FedTHE의 entropy 항만 두 exit의 logit에 적용한 규칙이다(w 격자 0.01, 동률이면 0.5에 가까운 값, 그다음 작은 값).
  - 정확도는 Lower-entropy exit와 거의 같다(S1 64.35 대 64.35).
  - CNN 설정에서 w*가 0 또는 1인 요청이 94.6–98.7%다. logit을 섞어도 entropy를 최소로 하는 해는 대부분 한쪽 exit다.
- **float16 복원의 정확도**: Round 12 기록은 float16 확률만 있어 log(max(p, 1e−8))로 복원했다. float16에서 0인 확률은 device 0.00–0.21%, edge 0.00–1.81%다(CIFAR-100). 재학습 run의 float32 log-softmax로 계산한 답과 float16 복원으로 계산한 답은 99.99% 같다.
- **LEW의 계산 시간**: CPU(NumPy)에서 요청당 19.5 µs(C = 10)다. DriftGate 결합은 2.1 µs다(`tables/R15_P2_cpu_time.csv`).
- **원 방법에서 생략한 요소**: FedRoD의 balanced 학습과 add-on head, ZTW의 cascade 연결·학습 weight·class bias, FedTHE의 feature alignment다. 이 생략을 원 방법 전체의 열세로 해석하지 않는다.

## 3. 작업 A1: 구간별 재계산 (`tables/R15_A1_rules_{full,gt30,gt50}.csv`)

| 설정 | strongest 대비, 첫날 | round > 30 | round > 50 | Confidence-based 대비, 첫날 / > 30 / > 50 |
|---|---|---|---|---|
| S1 | +0.95 (1위) | +0.36 (1위) | +0.22 (1위) | 5.45 / 2.31 / 1.74 |
| S2 | +1.63 (1위) | +1.14 (1위) | +1.09 (1위) | 6.52 / 3.92 / 3.28 |
| S1-fast | +0.84 (1위) | +0.18 (1위) | +0.03 (1위) | 5.63 / 2.26 / 1.64 |
| partial participation | +1.88 (1위) | +0.93 (1위) | +0.82 (1위) | 8.37 / 5.49 / 4.80 |
| stepwise change | +0.92 (1위) | −0.02 (2위) | −0.02 (2위) | 7.18 / 1.21 / 0.93 |
| random mobility | +1.52 (1위) | +0.11 (1위) | −1.82 (9위) | 12.53 / 3.70 / −1.07 |
| CIFAR-100 | +0.85 (1위) | +0.52 (1위) | +0.28 (1위) | 5.45 / 5.03 / 4.96 |
| ResNet-18 | +0.10 (1위) | −0.21 (6위) | −0.27 (6위) | 7.43 / 3.94 / 3.61 |
| K = 200 (규모) | +0.10 (1위) | −0.58 (5위) | −0.55 (9위) | 3.73 / 0.30 / −0.09 |
| K = 500 (규모) | −0.27 (4위) | −1.02 (9위) | −0.96 (9위) | 3.08 / −0.34 / −0.70 |

- 값은 pp이고 같은 seed끼리 뺀 평균이다. 표준편차는 CSV에 있다.
- 경계 round는 포함하지 않았다. 평가 round 목록은 CSV의 마지막 행에 있다.
- 시간대 기여(`R15_A1_time_of_day.csv`)는 (T_{s,j}/T_s)·Δ_{s,j}로 계산했다. 기여의 합은 seed별 하루 전체 차이와 1e−13 안에서 같다.
- Device only − Confidence-based offloading은 S1에서 첫날 +1.29 pp, round > 30에서 −3.03 pp다.

## 4. 작업 A2: 학습된 모델로 하루 재생 (`tables/R15_A2_replay.csv`, 그림 `figures/R15_fig_replay_time_of_day`)

**방법**

- Round 12의 S1(seed 0–4)과 S2(seed 0–2) 명령으로 첫날을 다시 학습했다. 하루 끝의 상태로 다음을 저장했다.
  - 모든 device의 client 모델
  - cell마다 server 모델과 client 평균
  - 마지막 학습 라운드의 label 개수
- 같은 하루를 학습과 aggregation 없이 재생했다. BatchNorm을 포함해 모든 파라미터를 고정했다. 이동과 요청은 첫날과 같고, device가 쓰는 cell 모델은 위치에 따라 바뀐다.
- 재생의 edge prior는 150 라운드의 label 개수로 만들었다.
- 확인한 것
  - 재학습한 첫날은 Round 12와 DriftGate 기준으로 0.05 pp 안에서 같다.
  - 5 라운드 smoke test에서 재생의 5 라운드째 기록이 첫날과 모든 배열에서 같다.

**결과**

| | S1 첫날 | S1 재생 | S2 첫날 | S2 재생 |
|---|---|---|---|---|
| Confidence-based offloading | 62.86 | 71.47 | 65.24 | 74.76 |
| Probability average | 67.32 | 72.27 | 69.93 | 75.90 |
| DriftGate | 68.32 | 71.95 | 71.74 | 76.54 |
| DriftGate − strongest | +0.95 | −0.33 (0.31) | +1.62 | +0.64 (0.81) |
| correction + w 0.2 | 68.16 | 72.58 | 71.07 | 76.66 |

**가능한 해석**

- 재생에서는 edge exit가 크게 좋아진다(S1 Edge only +8.8 pp). device exit가 더할 수 있는 몫이 줄어든다.
- S1 재생에서 DriftGate는 Probability average보다 집에 있는 시간대(출근 전, 저녁)에 약 1 pp 높고, 출근·낮 시간대에 0.65–1.10 pp 낮다. 밖에 있는 device의 OOP/OOR 요청에서 잃는 몫이 더 커진다는 해석과 맞는다.
- 다만 S2 재생은 같은 경로를 다시 쓴 것이므로 새 날짜의 독립 검증이 아니다.
- 학습 초반의 영향은 시간대 곡선(그림 4)에서 확인된다. 첫날의 출근 전 정확도가 재생보다 크게 낮다.

## 5. 작업 B: correction, 가중치, window (`tables/R15_B1_correction_weight.csv`, `R15_B2_sensitivity.csv`)

**확인된 사실 (첫날, variant − DriftGate, pp)**

| 변형 | S1 | S2 | random mobility | CIFAR-100 | ResNet-18 | S1 재생 | S2 재생 |
|---|---|---|---|---|---|---|---|
| correction 없음, adaptive w | −1.20 | −2.41 | −1.77 | −3.79 | −0.18 | +0.52 | −0.65 |
| correction + w 0.5 | −0.19 | +0.07 | −0.14 | +0.16 | +0.07 | −0.41 | −0.31 |
| correction + w 0.2(development) | −0.15 | −0.67 | −0.58 | −0.34 | −2.61 | +0.63 | +0.13 |
| correction한 edge만 | −0.45 | −1.21 | −1.21 | −0.59 | −4.91 | +0.79 | +0.05 |
| correction + learned weight(재학습) | −0.43 | −1.04 | −0.45 | −0.74 | −2.28 | −0.23 | −1.28 |
| correction + product | −0.41 | +0.06 | −0.31 | +0.44 | −0.05 | −0.83 | −0.54 |

- round > 30의 값은 CSV에 있다. 예를 들어 S1에서 correction한 edge만 쓰면 +0.55, w 0.2는 +0.45다.
- **민감도**
  - window 길이 N = 8, 32, 128(round 경계를 넘는 최근 offload 요청)은 전체 offloading과 온라인 β = 0.5 모두에서 정확도를 0.02 pp 이하로 바꾼다. cold start 비율은 0.01–0.05%다.
  - r = 0.3/0.7은 설정마다 다르다. S2는 r = 0.3이 +0.25, S1 재생은 r = 0.7이 +0.23이고, CIFAR-100은 r = 0.7이 −2.37이다.
  - run 안에서 w의 표준편차는 0.03–0.07이다.

**가능한 해석**

- correction의 이득은 edge exit가 약하거나(학습 초반) class 분포가 치우친(CIFAR-100) 조건에서 크다.
- adaptive w는 요청마다 바뀌기보다 모델마다 다른 값으로 수렴한다. 그래서 고정 w를 모델마다 잘 고르는 것과 비슷한 효과를 낸다.
- r 민감도는 실용적인 안정성을 보여 줄 뿐이고, Bayes 보정의 가정을 증명하지 않는다.

## 6. 작업 C: offloading (`tables/R15_C1_offload_evidence.csv`, `R15_C1_curves.csv`, `R15_C2_online_controller.csv`)

- **DriftGate의 부분 offloading 정의**: 원고 §5.3에 맞춰, 두 평균 entropy를 모두 window 안의 offload 요청에서 계산했다(DriftGate-P). Round 14의 정의(H̄_d는 모든 요청)도 비교용으로 계산했다.
- **첫날, 측정점 기준으로 A_full을 넘는 최소 offloading 비율**: S2 0.242, partial participation 0.136, stepwise change 0.474, random mobility 0.372, ResNet-18 0.225에서 β ≤ 0.5다. S1은 0.517(보간 0.484), S1-fast는 0.524, CIFAR-100은 0.953이다. 따라서 5개 설정이다. 처음 넘은 뒤 다시 내려가는 곡선은 없다(첫날).
- **round > 30**: S2 0.337, partial participation 0.443에서만 β ≤ 0.5다. stepwise change, ResNet-18, K = 200·500은 A_full에 닿지 않는다.
- **재생**: S1은 닿지 않고, S2는 0.509에서 닿는다.
- **geometric ensemble과의 차이**: 첫날 β ≥ 0.3에서 8개 설정 모두 양수다. round > 30에서는 S2, partial participation, random mobility만 그렇다.
- **온라인 controller(C2, 새 구현)**: device마다 직전 128개 요청의 device entropy 중앙값을 τ로 쓰고, 직전 요청이 16개 미만이면 0.8을 쓴다.
  - 실제 β는 0.498–0.500이다.
  - 결합 규칙 가운데 DriftGate는 S1·S2 첫날과 S2 재생에서 가장 높다. S1 재생에서는 Probability average보다 0.21 pp 낮다.
  - edge 호출은 두 cell 요청 때문에 offload 요청당 S1 1.13번, S2 1.06번이다. uplink byte는 호출 수 × 32 KB다.

## 7. 작업 E: 규모 (`tables/R15_E_scale.csv`, `R15_E_scale_reach.csv`)

- K = 50, 200, 500에서 거의 같은 값은 다음과 같다.
  - device-round 중 집에 있는 비율 56–58%
  - 요청 종류 Main/OOP/OOR 65/27/8%
  - a_k 0.25
  - cell당 device 11.5개, cell당 own class 7.2–7.3개
  - 두 cell에 속한 device 14–16%
  - B0 offloading 0.92–0.96
- 달라지는 값은 다음과 같다.
  - Edge only의 정확도(62.9 → 65.3 → 67.1%)
  - AUROC(0.645 → 0.663 → 0.671)
  - Device only의 정확도(64.2 → 62.6 → 63.1%)
- DriftGate의 home과 Main 정확도는 비슷하다(home 79.8/79.6/81.1).
- 원인은 이 표만으로 확정하지 않는다.

## 8. 작업 G: 구현과 문헌 (`R15_G_checks_ko.md`)

10개 항목의 코드 위치와 출처를 적었다. 주요 확인 결과는 다음과 같다.

- OOP와 OOR의 풀네임은 SplitOMC README에 있다. 우리 구현은 cell의 class 범위를 거주자의 class 합집합으로 정한다는 점에서 SplitOMC와 다르다.
- τ = 0.8은 SplitOMC 구현의 기본값이다.
- 구현의 window는 현재 요청을 넣지 않고 직전 round로 대체한다. v27 Algorithm 1과 달랐으므로 원고를 고쳤다.
- EM 0.23/0.51은 개발용 기록의 round ≥ 30 값이다.
- 3.2 µs의 측정 조건을 확인했다.
- 하위 10%는 규칙과 round마다 다시 뽑는다.
- Kim 등은 초록만 확인했다.

## 9. 작업 H: 모바일 비용

- 측정값이 없다. 장비를 사용하지 않았고, 원고의 모바일 절은 TODO 상태로 남겼다.
- 준비한 것은 다음과 같다.
  - `mobile/mobile_cost_schema.json`: 입력 schema. `measurement_type`은 measured/modelled/demo이고, 값이 없는 에너지는 null이다.
  - `scripts/r15_mobile_cost.py`: 입력 JSON만 바꾸면 정확도–지연, 정확도–에너지 두 패널을 그린다. 규칙마다 자기 offload 비용을 쓴다.
  - `mobile/demo/mobile_cost_demo.json`과 `figures/demo/`: v27의 가정값으로 만든 demo다. 그림에 DEMO를 표시했고 제출용 폴더와 분리했다.
- 필요한 측정은 다음과 같다.
  - 기기·runtime·split, 네트워크 RTT와 대역폭(정의 포함), 요청 간격, warm-up과 반복 수
  - 요청 스트림 재생의 평균·p95 지연
  - 에너지의 범위(기기 전체 또는 모델 연산)와 idle 차감 여부
  - 운영점: Device only, B0, 강한 결합 규칙, DriftGate 전체 offload, DriftGate 온라인 β = 0.5

## 10. 하지 않은 것

| 항목 | 이유 |
|---|---|
| D: S2·ResNet-18 seed 추가 | 지시문의 선택 사항. 예약하지 않았다. |
| A3: 둘째 날 계속 학습(151–300 라운드) | 선택 사항. 기존 결과가 없다. |
| F1: oracle 모델에 DriftGate 적용 | Round 7 run에 요청별 기록과 모델이 없다. 재학습을 자동으로 돌리지 않았다. |
| F2: 2×2 λ 통제와 focal visitor 실험 | 인과 주장을 관찰 범위로 낮췄으므로 필요하지 않다. |
| I: 실제 확률 출력 예시 그림 | 선택 사항. 하지 않았다. |
| 제출 규정(쪽수 상한) | 공식 TMC 안내를 확인하지 않았다. v28은 참고문헌 포함 14쪽이다. |

## 11. 원고 (`manuscript/v28/`, `R15_manuscript_changes_ko.md`)

- **원본 보존**: v27 원본은 `manuscript/v27_original/`에 그대로 있다.
- **v28 컴파일**: 14쪽이다. 정의되지 않은 참조, 인용, 넘친 상자는 없다.
- **표의 글자 크기**: 관련 연구 표(v27 표 3, v28 표 1)는 열을 다시 짜서 축소 없이 8pt로 넣었다. 설정, 구간, 규모 표도 축소 없이 넣었다.
- **새로 넣은 표와 그림**: 표 5(구간), 표 6(재생), 그림 4(재생 시간대), 표 8(correction을 맞춘 비교)이다.
- **그림 표기**: 그림의 clients 표기를 devices로 바꿨다.

## 12. 남은 항목

- 모바일 측정(H)과 원고 모바일 절의 TODO
- 주표의 평가 조건 결정: v28은 첫날을 주표로 두고 round > 30과 재생을 별도 표로 보고한다. 배포 조건(학습된 모델)을 주표로 바꾸려면 8개 설정 전부에서 재생을 해야 한다. 지금은 S1·S2만 있다.
- 사용자가 넣은 미커밋 파일은 건드리지 않았다. `DriftGate_TMC_v27.zip`, Round 14 폴더의 zip, 이전부터 있던 zip들이다.
