# Round 15 분석 계획 (결과를 보기 전에 고정)

작성: 2026-10-05. 이 문서는 Round 15의 새 계산과 새 run을 하기 전에 정한 설정과 정의를 적는다. 결과를 본 뒤에는 이 문서의 정의를 바꾸지 않는다. 바꿔야 하면 바꾼 이유와 영향을 보고서에 따로 적는다.

## 1. 입력과 확인한 사실

- 저장소: `/home/honeynaps/data/driftgate`. `AGENTS.md`는 없다. 작업 시작 시 HEAD는 `0a81a75`다.
- 원고: `journal_expansion/artifacts/DriftGate_TMC_v27.zip`(사용자가 넣은 미커밋 파일). 원본은 그대로 두고 `manuscript/v27_original/`에 풀었다.
- `paper_writing_guidelines_v4(1).md`는 서버에서 찾지 못했다. 원고 수정은 지시문 J절과 부록의 한국어 지침(보고서용)을 기준으로 한다.
- 서버에 LaTeX가 없다. 컴파일은 사용자 폴더에 내려받은 tectonic으로 한다.
- 체크포인트: Round 12의 고정 λ 0.4 run(S1, S2)은 모델을 저장하지 않았다. S1 DriftGate run은 클라이언트 0과 두 cell의 server 모델만 저장했다. 따라서 A2에는 재학습이 필요하다(4절).
- Round 7의 oracle run(`runs/phaseT7_gate/r7_O_s{5,6,7}`)에는 요청별 기록과 모델이 없다. F1은 재학습 없이는 할 수 없으므로 이번에는 하지 않는다.
- 설정 대응
  - random mobility는 Round 12의 client mobility(`r12_t1_mob_fx40_s*`)와 같다. `run_v2.py --mobility`(Gauss–Markov, `network/mobility.py`), `--schedule abrupt`, 120 라운드, 클라이언트 50개다.
  - ResNet-18은 `r12_e2_res_fx40_A_s*`이고 클라이언트가 16개다. 원고 Table 2의 "50 devices"는 이 설정과 맞지 않는다.

## 2. 설정과 seed

| 묶음 | 설정 | run | seed |
|---|---|---|---|
| 핵심 8개 | S1 | `r12_s1_fixed040_s*` | 0–4 |
| | S2 | `r12_s2_fixed040_s*` | 0–2 |
| | S1-fast | `r12_s1fast_fixed040_s*` | 0–2 |
| | partial participation | `r12_s4part05_fixed040_s*` | 0–2 |
| | stepwise change | `r12_t1_A_fx40_s*` | 0–4 |
| | random mobility | `r12_t1_mob_fx40_s*` | 0–2 |
| | CIFAR-100 | `r13b_c100_s*` | 0–2 |
| | ResNet-18 | `r12_e2_res_fx40_A_s*` | 0–2 |
| 규모 2개 | K = 200, K = 500 | `r12_s3k200_fixed040_s*`, `r12_s3k500_fixed040_s*` | 0–2 |

모두 고정 λ 0.4 run이다. 차이는 같은 seed끼리 빼고 평균과 표준편차(ddof = 1)를 구한다.

## 3. 평가 구간과 집계

- 구간 세 개: 전체, round > 30, round > 50. 경계 round는 포함하지 않는다. 실제 평가 round 목록을 표와 함께 저장한다.
  - 5 라운드마다 평가하는 설정: round > 30은 35–150(24개), round > 50은 55–150(20개)이다.
  - 10 라운드마다 평가하는 설정(stepwise change, random mobility, ResNet-18): round > 30은 40부터, round > 50은 60부터다.
  - 이전 라운드의 "후반 구간"(round ≥ 30)과 경계가 다르다.
- 정확도: 평가 시점마다 요청이 있는 device의 정확도를 같은 비중으로 평균하고, 다시 시점에 걸쳐 평균한다(원고 식 (4)). home/away, Main/OOP/OOR, 하위 10%는 Round 14의 집계를 그대로 쓴다.
- strongest baseline: 설정과 구간마다 DriftGate를 뺀 비교 규칙 가운데 평균이 가장 높은 하나를 고르고, 그 규칙과의 seed별 차이를 보고한다. seed마다 최강 규칙을 따로 고른 값은 "seed별 oracle 비교"로 따로 적는다.
- 시간대 기여(S1, S2): seed s에서 시간대 j의 평가 시점 수를 T_{s,j}, 전체를 T_s라 할 때 기여는 (T_{s,j}/T_s)·Δ_{s,j}다. 기여의 합이 seed의 전체 차이와 같은지 확인한다. 시간대는 Round 14와 같다(평가 round의 시작 시각 05:00 + 6(r − 1)분 기준).

## 4. A2: 학습된 모델로 하루 재생 (pretrained, frozen-model replay)

- 재학습 목록(실행 전에 고정): S1 seed 0–4, S2 seed 0–2. 명령줄은 Round 12의 `r12_s1_fixed040_s*`, `r12_s2_fixed040_s*` provenance를 그대로 쓰고, 이름(`r15_s1_fixed040_s*`, `r15_s2_fixed040_s*`), 출력 폴더(`runs/phaseT15_replay`), 두 flag만 더한다.
  - `--save_checkpoint`: 150 라운드 종료 시 device마다 설치된 client 모델, cell마다 server 모델과 client 평균, device의 소속 cell, 마지막 학습 라운드(150)의 cell별·전체 label 개수를 저장한다.
  - `--record_eval_logprobs`: 두 exit의 log-softmax를 float32로 기록한다(Logit-entropy weighting의 정확한 계산용).
  - GPU 학습은 bit 단위로 재현되지 않으므로 재학습한 첫날은 Round 12와 조금 다르다. 첫날과 재생을 비교할 때는 같은 재학습 run의 첫날을 쓴다.
- 재생(`--replay_from <checkpoint>`)
  - 모든 파라미터와 BatchNorm 통계를 고정한다(eval mode). 학습, optimizer update, 파라미터 aggregation을 하지 않는다.
  - 이동, 소속 cell, overlap, 요청 구성은 첫날 환경 파일을 그대로 재생한다. 요청 이미지와 도착 순서는 첫날과 같은 규칙으로 만들어지므로 같다.
  - device는 자기 client 모델을 계속 쓰고, 현재 cell의 server 모델(첫날 종료 상태)을 쓴다. 두 cell에 속하면 두 server의 logit을 평균한다(첫날과 같음).
  - edge prior: 체크포인트에 저장한 150 라운드의 label 개수로 π_tr을 만든다(첫날 마지막 학습 라운드의 정의와 같음). 두 cell이면 두 cell의 prior를 평균한다. 평가일의 요청 label은 쓰지 않는다.
  - 평가 시점은 첫날과 같다(1, 5, …, 150). window는 재생 기록 안에서만 계산하므로 평가 시작 시 비어 있다.

## 5. 규칙

내부 ID는 유지하고 원고 표시 이름을 바꾼다.

| ID | 원고 표시 | 연산 |
|---|---|---|
| B0 | Confidence-based offloading (SplitGP inference rule) | H(p_d) ≤ 0.8 nats이면 device exit, 아니면 edge exit |
| B1, B2 | Device only, Edge only | 한 exit |
| B3 | Probability average | argmax (p_d + p_e)/2 |
| R-PoE | Logit sum (FedRoD의 결합 연산) | argmax log p_d + log p_e (1e−8 floor) |
| R-THE | Lower-entropy exit | 요청마다 entropy가 작은 exit(같으면 device) |
| R-ZTW | Geometric ensemble with early exit (ZTW에서 가져온 부분: geometric 결합과 confidence threshold) | max p_d ≥ q이면 device, 아니면 logit sum. 주표는 B0과 같은 offloading 비율에서 보간 |
| R-EM | Label-shift EM (own/other 두 그룹의 비율 r̂를 EM으로 추정) | Round 10 M1, Round 13b 함수 |
| R-LR | Learned weight (9개 특징, correction 없음) | Round 13b 계수 그대로 |
| LEW (새 규칙) | Logit-entropy weighting | 아래 |
| DriftGate | DriftGate | Round 14의 DriftGate(= Round 13b F-auto) |

- **Logit-entropy weighting**: 요청마다 w ∈ {0, 0.01, …, 1}에서 H(softmax(w z_d + (1 − w) z_e))를 최소로 하는 w를 고르고, 그 혼합의 argmax를 답으로 쓴다. 동률이면 0.5에 가까운 w, 다시 동률이면 작은 w를 고른다. z는 log-softmax다. FedTHE의 entropy 항만 쓰는 추론 규칙이며 full FedTHE가 아니다(feature alignment 없음, 재학습 없음).
  - Round 12와 Round 13b 기록에는 float16 확률만 있다. 그래서 z = log(max(p, 1e−8))로 복원하고, float16에서 0이 된 확률의 비율을 함께 적는다. 이 값은 정확한 복원이 아니다.
  - 재학습한 S1, S2 run에서는 float32 log-softmax로 정확히 계산하고, float16 복원값과의 차이를 적는다.
  - 모든 요청을 offload하는 비교로 계산하고, 추가 계산 시간을 잰다.
  - 이 규칙을 주표에 넣으므로 strongest baseline, headline, offloading 교차점을 모두 11개 규칙 집합으로 다시 계산한다.

## 6. B: correction, 가중치, window

- **B1** 설정: S1, S2, random mobility, CIFAR-100, ResNet-18. 구간: 전체, round > 30. A2 재생 결과가 있으면 S1, S2 재생을 더한다.
  1. DriftGate
  2. correction 없음 + DriftGate의 adaptive w(같은 w)
  3. correction + w = 0.5
  4. correction + development에서 고른 고정 w: Round 11이 개발용 기록(Round 10, S1, seed 5–7)에서 고른 w = 0.2(r = 0.5). 모든 모델에 같은 w를 쓰는 공통 정책이다.
  5. correction한 edge만: argmax p'_e
  6. correction + learned weight: Round 13b의 9개 특징 가운데 edge 출력에 기대는 특징(edge 최대 확률, edge entropy, p_e(M_k), TV, 두 exit의 일치, x_SR)을 p'_e로 계산하고, 개발용 기록의 round ≥ 30 요청으로 로지스틱 회귀를 다시 학습한다(scikit-learn 기본 설정, 표준화). 답은 argmax P·p_d + (1 − P)·p'_e다.
  7. correction + product: argmax log p_d + log p'_e(지수 1, 1). Round 14의 "Calibration + product"와 같다.
- **B2** 민감도(B1과 같은 설정)
  - r ∈ {0.3, 0.5, 0.7}, 기존 window.
  - 고정 길이 window N ∈ {8, 32, 128}, r = 0.5. device별로 앞서 offload된 요청 가운데 최근 N개로 두 평균 entropy를 구한다. 평가 round의 경계를 넘어 이어지고, 현재 요청은 넣지 않는다. 앞서 offload된 요청이 8개 미만이면 w = 0.5다(cold start). 도착 순서는 평가 round, 같은 round 안의 도착 순서다.
  - 전체 offloading과 목표 β = 0.5(7절의 온라인 controller)에서 계산한다.
  - 정확도, Main/OOP/OOR, w의 평균과 표준편차, 요청 수, cold start 비율을 저장한다.

## 7. C: offloading

- **DriftGate의 부분 offloading 정의**: 원고 §5.3과 Algorithm 1에 맞춰, 두 평균 entropy를 모두 window 안에서 offload된 요청으로만 구한다. window는 기존 규칙(같은 round의 앞선 offload 요청이 8개 이상이면 그것, 아니면 직전 round의 offload 요청)이다. offload 요청이 없으면 그 device의 직전 w, 그것도 없으면 0.5다. 전체 offloading에서는 DriftGate와 같다. Round 14 곡선(H̄_d는 모든 요청)은 비교용으로 함께 계산한다.
- **C1** (핵심 8개, 규모 2개 따로; 구간 전체가 기본, round > 30은 CSV)
  - B0 기본 threshold의 정확도와 실제 offloading 비율
  - full-offload 집합: Edge only, Probability average, Logit sum(= geometric ensemble을 모두 offload), Lower-entropy exit, Label-shift EM, Learned weight, Logit-entropy weighting. 최고 평균 A_full
  - 기본 동작 집합: full-offload 집합 + B0(τ = 0.8) + Device only(β = 0) + Geometric ensemble with early exit(B0과 같은 β). 최고 평균 A_default
  - DriftGate 곡선(τ 격자는 SplitGP 곡선과 같음)이 A_full, A_default에 닿는(≥) 최소 β와 넘는(>) 최소 β. seed 평균 곡선 기준이 기본이고 seed별 값은 보조 CSV에 둔다. 격자의 측정점과 선형 보간값을 함께 적고, 교차 뒤 다시 내려가는지도 확인한다.
  - 같은 β에서 geometric ensemble 곡선과의 차이(β = 0.1–1.0)
  - "β ≤ 0.5에서 A_full을 넘는" 핵심 설정의 수(측정점 기준, 반올림 전 값)
- **C2** 온라인 budget controller(S1, S2, 재생 포함), 목표 β = 0.5
  - device마다 τ = 그 device의 직전 128개 요청의 device-exit entropy의 (1 − β) 분위수. 현재 요청은 넣지 않는다. 직전 요청이 16개 미만이면 τ = 0.8.
  - 같은 offload 결정에 Confidence-based(edge 답), Probability average, Logit sum, DriftGate(부분 offloading 정의)를 적용한다.
  - 정확도와 실제 β, edge 호출 수(두 cell 요청은 2번), uplink와 downlink byte를 따로 적는다.
  - 기존 구현에 없던 정책이다.

## 8. E, G, H, I

- **E**: K = 50, 200, 500에서 규칙별 정확도, home/away와 Main/OOP/OOR 정확도와 집계 비중, a_k 평균, cell당 device 수와 class 수, 두 cell에 속한 device 비율, B0 offloading 비율, device entropy의 AUROC를 기록에서 계산한다.
- **G**: 지시문 9절의 10개 항목을 코드 위치, 결과 출처, 논문 위치와 함께 확인한다. 공개 원문을 확보하지 못한 세부는 미확인으로 둔다.
- **H**: 모바일 측정값이 없으므로 입력 schema(`measurement_type`: measured/modelled/demo, 값이 없는 에너지는 null), 두 패널 그림 스크립트, 필요한 측정 목록을 만든다. 원고의 가정값은 demo 입력에만 두고 제출용 폴더와 분리한다.
- **I**: S1 seed 0의 round > 30에서 지시문 11절의 규칙으로 Main 사례 하나와 OOP 사례 하나를 고른다. 사례를 고친 뒤 다시 고르지 않는다.

## 9. 하지 않는 것

- D(seed 추가), A3(300 라운드 계속 학습), F2(외부효과 인과 실험), F1(oracle 재학습)은 하지 않는다.
- DriftGate의 정의(r = 0.5, 기존 window, τ = 0.8)를 바꾸지 않는다. 민감도 결과를 보고 기본값을 바꾸지 않는다.
- 이전 라운드의 산출물을 고치지 않는다.
