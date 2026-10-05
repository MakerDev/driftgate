# Round 16 1단계 중간 보고서: 기존 기록으로 계산한 oracle와 모사 selector

이 보고서는 Round 16 지시문 1–4절의 결과입니다. 새 학습 run은 시작하지 않았습니다. 지시문에 따라 여기서 멈추고 후속 지시를 기다립니다. 모든 수치는 `tables/`의 CSV에서 가져왔고, 차이는 같은 seed끼리 뺀 값의 평균(표준편차)이며 단위는 pp입니다.

## 1. 판정

주 후보 S-C는 세 표에서 모두 `NO_GO`입니다. 따라서 1단계 결과는 `SIMULATION_BELOW_CRITERION`입니다. 즉 모사 가설은 현재 채택 기준에 미달합니다(`decision_stage1.json`). 이 판정은 모사에 대한 판정이며, 논문 채택 판정이 아닙니다.

| 표 | 행 | strongest reference 이상인 행 | 가장 낮은 차이 | 판정 |
|---|---|---|---|---|
| 기본 표 (전체 offloading) | 42 | 26 | −5.73 pp (Partial participation, round ≤ 30) | `NO_GO` |
| 시간대 표 | 20 | 16 | −1.17 pp (S1 첫날, 출근) | `NO_GO` |
| 온라인 표 (β = 0.5) | 4 | 2 | −0.49 pp (S2 첫날) | `NO_GO` |

calibration 경로의 상태는 `A_AVAILABLE_PENDING_APPROVAL`입니다(`calibration_route.md`). probe pool은 평가 이미지와 겹치지 않고 모든 클래스를 포함하므로 경로 A를 구현할 수는 있습니다. 하지만 공용 labeled pool이라는 가정은 현재 원고에 없으므로, 이 가정을 쓰려면 사용자의 승인이 필요합니다.

다른 후보의 판정은 참고용입니다. S-A(고정 s = 0.65)는 온라인 표에서만 `PASS`이고, 기본 표와 시간대 표에서는 `NO_GO`입니다. S-B0, S-B5, S-Bg는 세 표 모두 `NO_GO`입니다. 같은 기준을 기존 DriftGate에 적용해도 기본 표 42행 중 6행만 strongest reference 이상이므로, 이 기준은 17개 규칙 가운데 행마다 가장 높은 규칙과 비교하는 엄격한 기준입니다.

## 2. 입력과 시작 확인

- 결과를 계산하기 전에 두 번 commit했습니다.
  - `107224c`: 판정 규칙, 분석 설정, 평가 manifest, calibration 경로 확인, 분석 스크립트
  - `f31f742`: 판정 스크립트
- 판정 기록은 첫날 10개 설정의 34개 seed-run과 S1·S2 재생 8개 run입니다. S1과 S2의 첫날 행에는 재생과 같은 모델에서 나온 Round 15 재학습 기록을 썼습니다. Round 12의 S1·S2 첫날 기록은 보조 표(`R16_S_supplementary_R12.csv`)에만 썼고, 결과의 방향은 같았습니다. 예를 들어 S1 첫날 전체에서 S-C는 +0.58 pp였고, round ≤ 30에서는 −1.30 pp였습니다.
- 시작 확인에서는 50개 run의 17개 규칙 값 850개를 기록에서 다시 계산했습니다. R15 캐시와 비교한 최대 차이는 0이었습니다. 즉 비트 단위로 같았습니다.
- S-A의 고정 비율 s는 개발 기록(S1 seed 5–7)으로 정했습니다. 첫날 전체 정확도가 s = 0.65에서 68.48%, s = 0.5에서 67.19%였으므로 0.65를 썼습니다.
- 계산 경로는 두 가지로 점검했습니다.
  - DriftGate 후보의 calibration 점수에는 w 구간을 이용한 계산을 씁니다. 이 계산은 실제 결합 답과 99.99995% 이상 일치했습니다.
  - Probability average, correction + w 0.5, corrected edge only, correction + dev w 0.2는 격자 쌍과도 같은 규칙입니다. 이 네 규칙의 격자 쌍 답은 R15 답과 100% 일치했습니다.

## 3. 결과

### 3.1 기본 표 (42행)

S-C가 strongest reference 이상인 행의 수를 구간별로 세면 다음과 같습니다.

| 구간 | 행 | 0 이상 | 차이의 범위 |
|---|---|---|---|
| `day1_full` | 10 | 8 | −3.86 ~ +1.59 |
| `day1_early` (round ≤ 30) | 10 | 0 | −5.73 ~ −0.38 |
| `day1_gt30` | 10 | 8 | −3.51 ~ +1.65 |
| `day1_gt50` | 10 | 8 | −3.18 ~ +1.31 |
| `replay_full` | 2 | 2 | +1.96 ~ +2.16 |

첫날 전체 구간과 재생 행은 다음과 같습니다.

| 행 | strongest reference (평균) | S-C | S-C − strongest | S-C − DriftGate |
|---|---|---|---|---|
| S1 첫날 | DriftGate (68.32) | 68.92 | +0.60 (0.29) | +0.60 (0.29) |
| S2 첫날 | correction + w 0.5 (71.81) | 72.30 | +0.49 (0.85) | +0.56 (0.97) |
| S1-fast | DriftGate (68.96) | 69.67 | +0.71 (0.36) | +0.71 (0.36) |
| Partial participation | correction + w 0.5 (64.85) | 60.98 | −3.86 (0.31) | −3.84 (0.27) |
| Stepwise change | DriftGate (72.36) | 73.50 | +1.14 (0.23) | +1.14 (0.23) |
| Random mobility | DriftGate (71.09) | 72.69 | +1.59 (0.35) | +1.59 (0.35) |
| CIFAR-100 | correction + product (41.73) | 41.76 | +0.02 (0.11) | +0.47 (0.04) |
| ResNet-18 | correction + w 0.5 (66.71) | 65.75 | −0.95 (0.96) | −0.89 (0.90) |
| K=200 | correction + dev w 0.2 (69.23) | 70.46 | +1.23 (0.26) | +1.46 (0.28) |
| K=500 | no correction + adaptive w (70.60) | 71.90 | +1.31 (0.09) | +1.79 (0.15) |
| S1 재생 | corrected edge only (72.74) | 74.90 | +2.16 (0.35) | +2.95 (0.38) |
| S2 재생 | correction + dev w 0.2 (76.66) | 78.62 | +1.96 (0.12) | +2.09 (0.05) |

- round ≤ 30 구간에서는 10개 설정 모두에서 S-C가 낮습니다. 이 구간의 strongest reference는 8개 설정에서 correction + product이고, 나머지 2개 설정(Stepwise change, ResNet-18)에서 Device only입니다.
- Stepwise change, random mobility, ResNet-18에서는 round ≤ 30의 요청이 모두 Main입니다. non-Main calibration 표본이 없으므로 규칙대로 모든 요청이 기존 DriftGate로 fallback합니다. 따라서 이 세 행의 S-C는 기존 DriftGate와 같습니다.
- round ≤ 30을 제외하면, 음수인 행은 Partial participation의 3행(−3.18 ~ −3.86)과 ResNet-18의 3행(−0.95 ~ −1.46)입니다.

### 3.2 시간대 표 (20행)

- 음수인 4행은 모두 첫날의 출근 전 또는 출근 시간대입니다.

  | 행 | S-C − strongest |
  |---|---|
  | S1 출근 전 | −0.95 (0.64) |
  | S1 출근 | −1.17 (0.58) |
  | S2 출근 전 | −1.05 (0.76) |
  | S2 출근 | −0.40 (0.30) |

- 첫날의 낮, 귀가, 저녁 6행은 +0.17 ~ +0.93입니다.
- 재생 10행은 모두 +1.03 ~ +2.52입니다.

### 3.3 온라인 표 (β = 0.5, 4행)

| 행 | strongest reference (평균) | S-C | S-C − strongest |
|---|---|---|---|
| S1 첫날 | DriftGate-P (67.74) | 67.25 | −0.49 (0.17) |
| S2 첫날 | DriftGate-P (71.81) | 71.31 | −0.49 (0.42) |
| S1 재생 | corrected edge only (71.55) | 72.05 | +0.50 (0.37) |
| S2 재생 | correction + dev w 0.2 (76.04) | 76.72 | +0.68 (0.04) |

실제 offloading 비율은 네 행 모두 0.499–0.500입니다. offload된 요청의 40–47%는 DriftGate-P로 fallback했습니다. 대부분(37–44%)은 |v̄_M − v̄_N| < 0.1 조건 때문입니다. v̄_g는 calibration 표본에서 device entropy가 τ를 넘고 동시에 edge 예측이 M_k에 속할 확률을 window의 τ들에 대해 평균한 값입니다.

### 3.4 Main 비율 추정

ŝ는 BBSE(black box shift estimation) 방식으로 구한 Main 비율입니다. 아래 표는 기본 표의 첫날 전체 구간과 재생에서 계산한 값입니다(`R16_S_estimation.csv`, seed 평균).

| 설정 | fallback | ŝ의 MAE | 0.9에서 잘림 | calibration과의 학습 round 차이 | 반복 이미지 |
|---|---|---|---|---|---|
| S1 첫날 | 6.5% | 0.10 | 22.5% | 5.0 | 98.2% |
| S2 첫날 | 7.3% | 0.09 | 30.4% | 5.0 | 97.8% |
| Partial participation | 13.1% | 0.22 | 30.6% | 5.0 | 98.2% |
| ResNet-18 | 33.3% | 0.20 | 28.5% | 11.8 | 96.9% |
| K=500 | 5.4% | 0.08 | 19.4% | 5.0 | 98.0% |
| S1 재생 | 2.9% | 0.03 | 10.0% | 5.0 (모델은 같음) | 98.2% |
| S2 재생 | 2.7% | 0.03 | 13.4% | 5.0 (모델은 같음) | 97.8% |

- MAE는 평균 절대 오차입니다. 여기서는 ŝ와, 현재 device-round 요청의 실제 Main 비율을 비교했습니다.
- 첫날의 ŝ는 random mobility(+0.01)를 제외한 설정에서 실제 비율보다 평균 0.01–0.05 낮습니다. 첫날의 fallback은 대부분 calibration 표본이 없는 경우와 |c11 − c01| < 0.1인 경우입니다.
- 온라인 controller에서 ŝ_off의 MAE는 첫날 0.22–0.24이고, 재생에서 0.06입니다.

### 3.5 Oracle

O1은 설정·구간마다 가장 좋은 고정 격자 쌍이고, O2는 device-round마다 가장 좋은 격자 쌍입니다. O3는 실제 Main 비율로 r을 정한 진단입니다.

- O2와 S-C의 차이는 42행의 중앙값으로 2.48 pp입니다.
  - 재생 두 행에서는 0.20과 0.21로 작습니다.
  - Partial participation 4행에서는 6.1–9.1로 큽니다.
- S-C는 42행 중 26행에서 O1보다 높습니다. S-C는 device와 요청마다 쌍을 바꿀 수 있으므로, 이 결과는 O1의 범위 밖에 있는 선택이 이득을 낸다는 뜻입니다.
- O3(실제 비율, w = 0)는 round > 50에서 ResNet-18(−0.03)을 제외한 9개 설정과 재생 2행에서 기존 DriftGate보다 높습니다(S1 round > 50: +1.06, S1 재생: +1.04). 반면 round ≤ 30에서는 10개 설정 모두에서 기존 DriftGate보다 낮습니다(−0.55 ~ −3.39). round ≤ 30에서 O3 가운데 가장 좋은 고정 w는 대부분 0.8이었습니다.

## 4. 지시문 4.7절에서 요구한 세 가지 설명

**S-C와 S-Bg의 차이.** S-Bg는 r을 1 − ŝ에 가장 가까운 격자값으로 고정하고 w만 고릅니다. S-C는 r과 w를 함께 고릅니다. S-C − S-Bg의 평균은 기본 표에서 +0.43 pp(42행 중 26행에서 S-C가 높음), 시간대 표에서 +0.60 pp(20행 중 15행), 온라인 표에서 +0.04 pp(4행 중 2행)입니다. S1 첫날에 S-C가 고른 숫자 r의 평균은 0.59입니다. 같은 구간에서 실제 Main 비율의 평균은 0.66이므로, r = 1 − s는 평균 0.34 근처입니다. 즉 calibration 점수는 추정 비율로 정한 보정 목표보다 큰 r을 고릅니다. 추정 비율을 그대로 보정 목표로 쓰는 S-B0도 세 표 모두 `NO_GO`입니다. 따라서 이번 모사에서는 "추정 비율을 보정 목표로 직접 쓰는 것이 유리하다"는 가정이 지지되지 않았습니다.

**Oracle 대비 남은 차이.** 기본 표 42행에서 S-C는 O2보다 중앙값으로 약 2.5 pp 낮습니다. 재생 두 행에서는 그 차이가 0.2 pp까지 줄어듭니다. 재생의 calibration은 같은 고정 모델로 거의 같은 이미지(98%)를 처리한 결과입니다. 그래서 실제 Main 비율을 주고 이전 round 자료로 고른 쌍은 O2와 0.05 pp밖에 차이 나지 않습니다(진단표 `R16_D_stale_calibration.csv`). 따라서 재생에서 S-C가 얻은 +2 pp는 이 모사가 가진 낙관성을 상당 부분 포함합니다. 첫날의 남은 차이는 다음 진단에서 보듯이, calibration 자료가 5 학습 round 전의 모델로 계산된 영향이 큰 것으로 보입니다. 실제 비율을 주고 고른 쌍과 O2의 차이는 S1에서 round ≤ 30에 1.26 pp, round > 30에 0.78 pp입니다. Partial participation에서는 그 차이가 3.0 pp이고, 기존 DriftGate보다 0.31 pp만 높습니다. 같은 비교에서 S1은 2.13 pp 높습니다.

**실제 구현에서 달라지는 입력.** 경로 A는 calibration 자료로 probe pool의 이미지를 쓰고, 현재 모델과 현재 cell로 다시 계산합니다. 그래서 1단계 모사와 두 방향에서 다릅니다.

- 모사에는 5–12 학습 round 전 모델의 출력이 들어갑니다. 경로 A에는 이 지연이 없습니다.
- 모사의 calibration 이미지는 현재 요청과 97–98% 같습니다. 경로 A에는 이 반복이 없습니다.
- non-Main 표본의 class 구성도 실제 요청과 달라집니다. Acc_N을 요청 구성에 맞추어 가중할지는 정하지 않았습니다.
- 경로 A에서는 calibration을 평가 round마다 다시 계산해야 합니다. device마다 최대 512장, 장당 32 KB의 feature를 edge로 보내야 합니다.

ŝ의 관측량 T와 window 규칙은 같습니다. c11과 c01은 pool에서 다시 구합니다. 두 차이의 효과가 서로 반대 방향일 수 있으므로, 실제 경로의 결과가 모사보다 좋을지 나쁠지는 이 결과만으로 예측할 수 없습니다.

## 5. 관측 사실, 가능한 설명, 확인하지 못한 사항

**관측 사실**

- S-C는 round > 30과 재생에서 대부분의 설정에서 strongest reference보다 높습니다. 예외는 Partial participation과 ResNet-18입니다.
- S-C는 round ≤ 30의 10개 행, 그리고 첫날 출근 전과 출근 시간대의 4개 행에서 모두 낮습니다.
- 온라인 controller에서 S-C는 첫날에 DriftGate-P보다 0.49 pp 낮고, 재생에서는 0.50–0.68 pp 높습니다.
- S-C가 기존 DriftGate를 고른 비율은 설정마다 0.15–1.1%입니다.

**가능한 설명 (확인한 범위)**

- 학습 초반에는 평가 round 사이의 5 학습 round 동안 모델이 크게 바뀝니다. 그래서 이전 round의 정확도표가 현재의 후보 순위를 덜 정확하게 반영합니다. 위의 S1 진단(round ≤ 30에서 1.26, round > 30에서 0.78)이 이 방향과 맞습니다.
- 학습 초반에는 Main 비율이 높아서 ŝ가 0.9에서 잘리는 비율도 큽니다. S1 round ≤ 30에서 이 비율은 48%입니다.
- Partial participation에서는 실제 비율을 주어도 이전 round 자료로 고른 쌍의 이득이 작습니다. 또 ŝ의 MAE가 0.22로 다른 설정의 두 배입니다. 두 요인이 함께 작용한 것으로 보입니다.
- 재생에서 S-C가 크게 이기는 것은 calibration이 사실상 현재 요청의 정답 정보를 담고 있기 때문인 것으로 보입니다.

**확인하지 못한 사항**

- Partial participation에서 이전 round의 calibration이 덜 맞는 원인은 확인하지 못했습니다. 평가 round에 학습에 참여했는지로 나누어도 차이가 거의 없었습니다.
- ResNet-18에서 S-C가 낮은 원인은 확인하지 못했습니다. ResNet-18은 평가 간격이 10 round라서 calibration과 약 12 학습 round 차이가 납니다. 또 round ≤ 30에서는 Main 요청만 있어서 fallback이 33%입니다.
- 경로 A에서 독립된 이미지를 쓰고 현재 모델로 계산했을 때의 성능은 측정하지 않았습니다.
- 온라인 추정에서 fallback이 40% 이상인 문제를 줄일 방법은 검토하지 않았습니다. 이 문제는 대부분 |v̄_M − v̄_N| < 0.1 조건에서 생깁니다.

## 6. 사용자가 정할 사항

지시문에 따라 2단계는 시작하지 않았습니다. 진행 방향은 다음 중에서 정해 주시면 됩니다.

1. **새 방법 탐색을 멈추고 v27 구성에 R15의 정정을 반영합니다** (지시문 7.2–7.3절). 1단계 판정과 같은 방향입니다.
2. **S-C로 2단계를 진행합니다.** 모사 판정은 미달이지만, 실제 경로에서는 calibration 지연이 사라집니다. 이 경우 경로 A(공용 labeled pool)를 쓴다는 가정을 승인하셔야 하고, Acc_N의 가중 방식을 결과 전에 정해야 합니다. 경로 A로 가면 R15 체크포인트로 S1·S2 재생부터 평가할 수 있습니다.
3. **다른 후보로 바꿉니다.** 세 표를 모두 통과한 후보는 없습니다. S-A는 온라인 표에서만 통과했습니다.

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `decision_rule.md`, `r16_config.json`, `evaluation_manifest.csv` | 결과 전에 고정한 규칙, 설정, 평가 행 |
| `calibration_route.md`, `tables/R16_C_pools.csv` | 경로 A와 B의 사전 확인 |
| `tables/R16_T0_start_check.csv` | 시작 확인 (850개 값) |
| `tables/R16_S_selectors.csv`, `_per_seed.csv` | 기본 표 42행 (6개 방법) |
| `tables/R16_S_time.csv`, `_per_seed.csv` | 시간대 표 20행 |
| `tables/R16_S_online.csv`, `_per_seed.csv`, `R16_S_online_offloading.csv` | 온라인 표 4행과 실제 offloading 비율 |
| `tables/R16_R_references.csv` | 행마다 17개(온라인은 15개) 비교 규칙과 strongest reference |
| `tables/R16_O_oracles.csv`, `_per_seed.csv` | O1, O2, O3 |
| `tables/R16_S_estimation.csv` | 추정 오차, clipping, fallback, calibration 지연, 반복 이미지, 선택된 w와 r |
| `tables/R16_S_supplementary_R12.csv` | Round 12 S1·S2 첫날 기록 (판정 제외) |
| `tables/R16_S_dev_sA.csv`, `R16_S_judgement.csv`, `R16_T1_run_checks.csv` | S-A의 s 선택, 표별 판정, run별 계산 확인 |
| `tables/R16_D_stale_calibration.csv` | 결과를 본 뒤 추가한 진단 (판정에 쓰지 않음) |
| `decision_stage1.json` | 1단계 판정 |
| `scripts/` | 분석, 판정, 진단, 단위 확인 스크립트와 실행 순서(`README.md`) |
