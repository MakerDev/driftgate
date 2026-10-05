# Round 15 원고 변경 기록 (v27 → v28)

- 원본: `manuscript/v27_original/`(사용자가 넣은 `DriftGate_TMC_v27.zip`을 그대로 푼 것)
- 수정본: `manuscript/v28/`, 컴파일 결과 `manuscript/v28/main.pdf`
- 컴파일 도구: tectonic 0.17.0. 서버에 TeX Live가 없어 사용자 폴더에 내려받았다.
- 컴파일 상태
  - 참고문헌을 포함해 14쪽이다(v27은 12쪽).
  - 정의되지 않은 참조와 인용은 없고, 넘친 상자도 없다.
  - Palatino 글꼴 대체 경고가 나오지만 v27에서도 같은 경고가 나온다.
  - 글자 크기는 아래첨자를 빼면 6.6pt 이상이다. 표 1(관련 연구, v27의 표 3), 표 3(설정), 표 5(구간), 표 9(규모)는 축소 없이 8pt로, 표 6(재생)은 7pt로 넣었다. 표 4와 표 8은 두 단 폭에 맞춰 조금 축소된다.
- 빨간 TODO는 모바일 측정과 관련된 곳(초록, 서론, §6.8, 표 11, 결론)에만 남아 있다.
- 제출 규정(쪽수 상한)은 확인하지 않았다. 투고 직전에 공식 TMC 안내에서 확인해야 한다.

## 1. 결과가 바뀌어 주장의 범위를 고친 곳

| 위치 | v27 | v28 | 근거 |
|---|---|---|---|
| 초록, 서론 7문단, 결론 | 8개 설정 모두에서 가장 정확하다 | 첫날(학습 초반 포함) 기준으로 그렇다고 한정했다. round > 30에서는 8개 중 6개에서 1위이고 우위는 −0.2 ~ +1.1 pp다. 학습된 모델로 재생하면 S2에서 +0.6 pp, S1에서 Probability average보다 −0.3 pp다. | `tables/R15_A1_rules_{full,gt30,gt50}.csv`, `R15_A2_replay.csv` |
| 초록, 서론, 결론 | 7개 설정에서 절반 이하만 offload하고도 모든 full-offload 규칙을 넘는다 | 평가한 threshold(측정점) 기준으로 5개 설정이다. S1과 S1-fast는 52%, CIFAR-100은 95%에서 처음 넘는다. | `R15_C1_offload_evidence.csv` |
| 서론 3문단, §2.3 | Confidence-based offloading이 Device only보다 1.3 pp 낮다 | 삭제했다. 이 차이는 첫날에만 있고, round > 30에서는 Device only가 3.0 pp 낮다. Edge only와 같다는 문장은 두 구간에서 모두 성립하므로 남겼다. | `R15_A1_rules_full.csv`, `R15_A1_rules_gt30.csv` |
| 서론 4문단, §2.4, §2.5, §3, §7 | oracle이 home 이득을 visitor 손실로 갚는다(인과 단정), "best version", "training-time adaptation does not help" | 시험한 location-aware 정책에서 home 이득과 away 손실이 함께 관찰되었다는 관찰로 바꿨다. 정책이 거주자와 방문자의 비율을 함께 바꾸므로 원인을 분리하지 못한다고 적었다. | 지시문 0절, 8절; `R15_G_checks_ko.md` F1 |
| §2.4 | APFL 62.9%가 고정 비율과 같다 | seed 0–2로 맞춰 고정 비율보다 0.3 pp 낮다고 고쳤다. v27은 seed 집합이 달랐다(APFL 0–2, 고정 0–4). | Round 6 `runs/phaseT6_S1/s1_apfl*`, `s1_fixed040_s*` |
| §6.2 | ResNet-18에서 0.10 pp 앞선다 | 평균 0.10 pp가 seed 간 표준편차 0.61 pp보다 작아 구별할 수 없다고 적었다. | `R15_A1_rules_full.csv` |
| §6.2(v27) | learned weight가 낮으므로 per-request weighting은 답이 아니다 | 삭제했다. correction을 맞춘 learned weight는 DriftGate보다 0.4–2.3 pp 낮다는 사실만 §6.6에 적었다. | `R15_B1_correction_weight.csv` |
| §5.3, §6.6 | 고정 weight 하나로는 두 모델에 맞출 수 없다 / adaptive w가 더 낫다 | adaptive w는 설정마다 더 나은 고정 w에 가깝다(첫날 w = 0.5와 ±0.2 pp 이내). w = 0.2는 ResNet-18에서 2.6 pp 낮지만 S1, S2 재생에서는 0.6, 0.1 pp 높다. run 안에서 w의 표준편차는 0.04다. | `R15_B1_*`, `R15_B2_sensitivity.csv` |
| §5.2, §6.6 | correction이 없으면 평균이 작동하지 않는다 | 첫날에는 correction을 빼면 1.2–3.8 pp 낮아지지만(ResNet-18은 0.2 pp), S1 재생에서는 0.5 pp 높아진다고 적었다. Bayes 해석의 두 가정(class-conditional 분포 불변, calibration)을 밝히고 heuristic으로 다룬다고 적었다. | `R15_B1_correction_weight.csv` |
| §1, §5.1, §6.6 | product는 정답을 항상 지운다 / 평균이 edge 정답을 지킨다 / 평균이 product보다 정확하다 | 조건(확률 크기)에 따라 서술했다. product는 S1, random mobility, ResNet-18, 두 재생에서 낮고, CIFAR-100(+0.4 pp)과 S2(+0.1 pp)에서는 높다. | `R15_B1_correction_weight.csv` |
| §6.7 | 규모 표 | strongest alternative(K = 500에서 Probability average가 DriftGate보다 0.27 pp 높음)를 넣었다. DriftGate가 Confidence-based offloading의 정확도에 닿는 비율을 원고 정의의 곡선으로 다시 계산했다(0.223, 0.311). 단일 edge 서버의 처리량을 잰 것이 아니라고 적었다. | `R15_E_scale.csv`, `R15_E_scale_reach.csv` |
| §7 Scope | 긴 배포에서는 아침이 저녁처럼 될 것으로 예상한다 | 예상을 지우고 재생 결과를 근거로 한 "Trained models" 문단을 새로 썼다. S2 재생은 같은 날을 다시 쓴 것이라 새 날짜의 검증이 아니라고 적었다. | `R15_A2_replay.csv` |

## 2. 비교 방법의 이름과 설명 (작업 P)

| 위치 | 변경 | 이유 |
|---|---|---|
| §6.1 규칙 목록, 표 4, 표 6, 표 7, 표 10, 그림 | SplitGP → Confidence-based offloading(SplitGP의 추론 규칙), Average → Probability average, FedRoD → Logit sum, FedTHE → Lower-entropy exit, ZTW → Geometric ensemble with early exit, Learned weight 설명 보완 | 내부 연산과 이름을 맞춘다. 원 방법 전체와 구분한다. |
| §6.1, §3, 표 3 | FedTHE를 "entropy가 낮은 exit 선택의 해"로 설명한 문장과 오목성 논증을 삭제했다. 원문(logit 가중합의 softmax entropy, feature alignment, cosine 유사도 비중)대로 설명했다. | 지시문 P2. 원 논문 §4.2 확인 |
| 표 4, §6.2 | 새 규칙 Logit-entropy weighting을 추가했다(규칙 11개). w*가 0 또는 1인 요청이 94.6–98.7%라서 Lower-entropy exit와 거의 같다고 적었다. | 지시문 P2 |
| §3, 표 1 | ZTW의 cascade 연결, 학습 weight, class별 bias를 적었다. 표 1의 class 보정 열에 "Trained class-specific bias"를 넣었다. FedRoD의 balanced 학습과 add-on head를 적었다. | 지시문 P1 |
| §3 | Kim 등(IEEE IoT J. 2025)과의 차이를 3문장 넣었다. 초록에서 확인한 범위(사전 학습, server 모델을 teacher로 한 online distillation, client 모델의 early exit)만 썼다. | 지시문 J17, G10 |

## 3. 정의, 구현, 용어 (작업 G, J)

| 위치 | 변경 |
|---|---|
| 서론 1문단, §4.1 | request를 "앱이 class label을 필요로 하는 입력 하나"로 정의했다. |
| 서론 2문단 | 인식 앱의 예(집과 상권에서 보는 물체가 다름)를 넣었다. 예시이며 측정한 분포가 아니다. |
| §2.1, §4.1 | own classes를 "device의 local training data에 있는 class"로 정의했다. OOP(out-of-preference), OOR(out-of-region)의 풀네임과 SplitOMC 출처를 넣었다. |
| §2.1, §6.1 | ρ = 0.1/0.8, 11.5%, 51%가 시뮬레이션 설정이라고 적었다. 반복은 §2.1과 §6.1의 정의 위치에만 남겼다. |
| §2.1 | λ를 "locally trained device block을 유지하는 비율"로 앞에서 정의하고 §4.2의 식으로 연결했다. |
| §2.3 | AUROC와 entropy의 단위(nats, 자연로그)를 첫 사용 위치에서 풀어 썼다. τ = 0.8이 SplitOMC 구현의 기본값이라고 적었다. |
| §2.4, 그림 2 caption | oracle의 0.70/0.15가 우리 controller가 쓰는 λ 범위의 상한과 하한이며, 첫 라운드부터 적용했다고 적었다. 추론 규칙은 Confidence-based offloading이고, seed는 개발용 3개라고 적었다. |
| §4.2 | π_k가 파라미터 평균의 정확한 prior가 아니라 근사라고 적었다. a_k를 [0.01, 0.99]로 자르고, 학습 표본이 없는 cell은 전체 분포를 쓴다고 적었다. |
| §5.3, §5.4, Algorithm 1 | window를 구현에 맞췄다. 현재 요청은 넣지 않는다. 같은 평가 기간의 앞선 offload 요청이 8개 이상이면 그 요청들을 쓰고, 아니면 직전 기간의 요청들을 쓰며, 둘 다 없으면 w = 1/2다. 두 평균 entropy를 같은 offload 요청에서 계산한다. Algorithm은 w를 먼저 계산하고 답을 낸 뒤 현재 요청을 window에 넣도록 순서를 바꿨다. |
| §5.5 | 3.2 µs의 측정 조건(NumPy, Threadripper PRO 7965WX 코어 하나, 요청 하나씩 20,000번, 보정·평균·argmax 포함)을 적었다. 44 bytes가 application payload이고, 두 cell 요청은 edge 호출이 2번(S1에서 offload 요청당 +0.13번)이라고 적었다. |
| 표 2 | ResNet-18은 device 16개, random mobility는 120 라운드라고 고쳤다. Devices 열을 넣었다. |
| §6.1 | 평가 조건 세 가지(첫날, round > 30/50, frozen replay)와 strongest alternative의 정의를 넣었다. |
| 그림 1, 2, 5, 6 | clients 표기를 devices로 바꿨다. 그림 글자를 1pt 키웠다. |
| 표 10 caption, §6.7 | 하위 10%가 규칙과 평가 round마다 다시 선정된다고 적었다. 개별 device의 무손실로 해석하지 않는다. |
| §6.8, 표 11 | 수치가 측정값이 아니라 비용 모형의 가정값이라고 적었다. 선형 모형이 radio tail과 idle 전력을 반영하지 못할 수 있다고 적었다. DriftGate β = 0.5의 정확도는 온라인 controller의 실측 β(0.50)에서 얻은 67.7%로 바꿨다. |

## 4. 새로 넣은 표와 그림

| 표·그림 | 내용 | 출처 |
|---|---|---|
| 표 5 (`tab:windows`) | 8개 설정의 첫날, round > 30, round > 50에서 strongest alternative 대비 우위와 순위, Confidence-based 대비 우위 | `R15_A1_rules_*.csv` |
| 표 6 (`tab:replay`) | S1, S2의 첫날(재학습)과 frozen replay의 규칙별 정확도 | `R15_A1_rules_full.csv`(재학습, 재생 열), `R15_A2_replay.csv` |
| 그림 4 (`fig:replay`) | S1, S2의 시간대 곡선: 첫날 대 재생 | `figures/paper/replay_time_of_day.pdf` |
| 표 8 (`tab:ablation`) | correction을 맞춘 변형 비교: 첫날 5개 설정과 재생 2개 | `R15_B1_correction_weight.csv` |
| §6.5 문단 | 온라인 budget controller(β = 0.5) 결과 | `R15_C2_online_controller.csv` |

## 5. 그대로 둔 것

- 문제 설정, 절 구성, DriftGate의 정의(r = 1/2, window 규칙, τ = 0.8)는 바꾸지 않았다.
- 모바일 측정 절은 TODO 상태로 두었다. 측정 입력 schema와 그림 스크립트는 `mobile/`과 `scripts/r15_mobile_cost.py`에 있다.
- 그림 7(offloading 곡선)은 원고 §5.3의 정의(두 평균 entropy를 모두 offload 요청에서 계산)로 다시 그렸다. v27 그림은 Round 14의 정의(H̄^d를 모든 요청에서 계산)였다.
