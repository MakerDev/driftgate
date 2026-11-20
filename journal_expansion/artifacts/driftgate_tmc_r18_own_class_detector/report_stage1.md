# Round 18 1단계 보고서: 준비와 점검

이 보고서는 지시문 6.1절의 1단계 결과를 적습니다. 중간 보고서이므로 표와 수치, 구현에서 정한 것, 걸린 시간만 적습니다. 판단 기준은 결과를 보기 전에 `decision_rule.md`로 commit했습니다(commit `fc1c157`). 저장소에는 이름이 같은 다른 작업(`driftgate_tmc_r18_compare`, 고정 가중치 비교)이 이미 있으며, 이 작업과는 폴더와 branch(`r18-own-class-detector`)가 다릅니다.

## 1. v32 표 4의 S1 값 재계산 (6.1의 1번)

| 규칙 | v32 표 4 | 기록에서 다시 계산한 값 | 차이 |
|---|---|---|---|
| Confidence-based offloading | 62.87 | 62.866 | 0.004 %p |
| v32의 방법 (DriftGate) | 68.31 | 68.315 | 0.005 %p |

- 기록은 Round 12의 S1 run(`runs/phaseT12_fusion/r12_s1_fixed040_s{0..4}`)이고, 계산은 Round 20 cache의 요청별 정답 개수에서 했습니다. 두 값 모두 0.05 %p 안에서 같습니다.
- seed별 값은 Confidence-based offloading이 63.29, 63.56, 62.84, 58.67, 65.96이고, v32의 방법이 69.36, 69.07, 68.16, 64.26, 70.73입니다.

## 2. Round 16과 Round 17의 요약 (6.1의 2번)

1. Round 16은 device의 요청 가운데 Main의 비율 s를 edge 예측("argmax가 M_k 안")의 BBSE로 추정하고, 그 값으로 요청마다 가중치 w와 보정 세기 r을 고르는 선택기(S-C)를 시험했습니다. 보정 자료는 직전 평가 라운드의 요청과 label이었습니다.
2. 판정은 세 표 모두 NO_GO(SIMULATION_BELOW_CRITERION)였습니다. 기본 표 42행 가운데 26행만 17개 규칙의 최강 값 이상이었고, 라운드 30 이하의 10행은 모두 음수였으며, partial participation에서 −3.9 %p, ResNet-18에서 −1.0 %p였습니다.
3. 재생에서 보인 +2 %p의 이득은 직전 라운드 요청의 98%가 같은 이미지라서 oracle에 가까워진 결과였습니다.
4. Round 17은 Round 15 checkpoint의 재생에서 feature 거리로 Main이 아닌 요청을 가려내는 네 탐지기를 시험했습니다.
5. AUROC(Main이 아닌 요청을 양성으로 둠)는 S1/S2에서 D1(device feature의 Mahalanobis) 0.665/0.657, D2(k-NN) 0.716/0.737, D3(edge가 학습한 logistic head, 음성은 같은 cell의 다른 device 표본) 0.701/0.675, D4(edge feature의 class 원형) 0.838/0.864, device entropy 0.70/0.71이었습니다.
6. 가장 나은 규칙 R-b(D4)는 보정한 edge 출력만 쓰는 규칙보다 S1에서 −0.06 %p, S2에서 +0.12 %p였고, 판정은 NO_GO였습니다. 거리 controller(β = 0.5)는 entropy 기반 DriftGate-P보다 0.45–1.45 %p 낮았습니다.
7. 이번 작업의 `cnn-scratch` + `block` 입력 + `logreg` + `cell` 음성 조합은 Round 17의 D3과 같은 구조입니다(차이: 이번에는 라운드마다 다시 학습하고, held-out 검증 통계가 있습니다). 따라서 `cnn-scratch`에서는 AUROC 0.7 안팎을 예상할 수 있습니다.
8. 이후 작업(`driftgate_tmc_r18_compare`, Round 19, Round 20)에서는 같은 offloading 예산에서 v32의 방법이 보정한 고정 가중치 규칙과 −0.79…+0.15 %p 안에서 같았습니다. 보정 없는 규칙보다 높은 이득은 보정에서 나왔습니다.

## 3. 구현 (6.1의 3번)

| 항목 | 구현 |
|---|---|
| held-out 10% | device마다 class별 10%(반올림, class마다 최소 1개)를 seed로 고정해 떼어 두고 학습에 쓰지 않습니다(`src/runner_r6.py`의 `holdout_frac`). |
| feature 기록 | device가 local 학습을 마친 직후(eval mode) 자기 학습 표본의 block 출력을 만들고, edge는 그 라운드에 설치한 device exit로 exit 벡터를 계산합니다(`src/r18oc_detector.py`의 `post_local_train`, `end_round`). |
| 판별기 학습 | 학습 라운드마다 device 18개 조합(입력 3 × 모델 2 × 음성 3)을 edge에서 다시 학습합니다. 학습에 참여하지 않은 device는 이전 판별기를 씁니다. |
| 검증 통계 | 모델을 설치한 뒤 held-out을 forward해 own/other의 64칸 histogram을 만듭니다. |
| 판별기 전달 | 학습 시점의 판별기와 histogram이 그 라운드의 평가부터 쓰입니다. 라운드 150의 상태는 checkpoint에 함께 저장되고, 재생은 이 상태를 고정해 씁니다. |
| 요청 점수 | 평가할 때 요청마다 판별기 18개(평가용 run은 주 조합의 일정 판 2개를 더해 20개), device energy, own-mahalanobis, own-knn을 기록합니다(`{run}_r18oc.npz`). |
| 3.3·3.4절의 규칙 | 곡선, 고정 문턱, 예산 controller, `edge-corrected-by-detector`는 분석 스크립트에 있습니다(`scripts/r18oc_analysis.py`). |
| 둘째 날 | 환경의 하루를 두 번 이어 붙이고(`--days 2`), 둘째 날의 평가 시점은 첫날과 같은 시각입니다. |

- MobileNetV2: torchvision의 IMAGENET1K_V1 가중치를 내려받았습니다. 64×64 입력에서 device block(`features[0]`–`features[6]`)의 출력은 32×8×8(stride 8), edge block의 입력도 같습니다. `features[0]`–`features[10]`까지 늘리면 출력은 64×4×4(stride 16)입니다. device block의 파라미터는 55,488개이고 edge block은 약 217만 개입니다.
- 옵션을 모두 끈 runner는 기존 main branch의 runner와 같은 결과를 냅니다. CPU에서 S1 seed 0을 1라운드 돌려 요청별 확률 14개 배열과 trace 배열이 모두 같았습니다(probe 시간 기록만 다름).
- 재생 run은 학습 run의 checkpoint 시점을 비트 단위로 재현합니다. 2라운드 시험에서 두 exit의 확률, 판별기 점수 20개, energy, own-mahalanobis, own-knn, histogram이 모두 같았습니다. 첫 시험에서는 재생 run의 device exit batch normalization 통계가 판별기 모듈의 forward에서 바뀌는 문제가 있었고, 모든 forward를 eval mode로 고정해 고쳤습니다.

## 4. sanity check (6.1의 4번)

| 점검 | 결과 |
|---|---|
| 판별기 자리에 실제 요청 종류(g = 1[Main])를 넣으면 `request-kind-oracle`과 같은가 | 같습니다. 문턱 0.5의 offload 집합이 Main이 아닌 요청과 정확히 같고, 정확도가 oracle의 직접 계산과 같습니다(시험 run 56.571%). 곡선 20점도 oracle 곡선과 같습니다. |
| 판별기 없이 `device-entropy` + `average-corrected-even`을 계산하면 기존 구현과 같은가 | 같습니다. Round 12 S1 seed 0에서 모든 요청을 offload한 경우와 Round 18 controller(β = 0.5)의 offload 집합 모두, (평가 라운드, device)별 정답 개수가 Round 20 cache와 정확히 같습니다(69.360%). 새 기록 형식의 시험 run에서도 스크립트의 값과 직접 계산이 같습니다. |
| 보정한 edge의 argmax를 own class 최댓값과 나머지 최댓값의 비교로 계산한 결과가 전체 계산과 같은가 | r = 0.5와 무작위 r에서 모두 요청의 100%가 같습니다. |

평가 요청의 label이 판별기 학습, 검증 통계, ŝ의 계산에 들어가지 않는 코드 경로는 다음과 같습니다.

- 판별기 학습(`DetectorBank.end_round`, `_train`)은 학습 표본의 label(`Y_train[train_idx]`)만 씁니다. 자기 표본은 양성이고, 다른 device의 표본 가운데 label이 M_k 밖인 것이 음성입니다. 가상 음성은 학습 표본의 class 통계에서 뽑습니다.
- 검증 통계(`_validate`)는 held-out 표본(학습 세트에서 떼어 둔 표본)의 label만 씁니다.
- 평가 중 hook(`begin_eval`, `batch`)은 device logit과 block 출력만 받습니다. 평가 label은 runner의 기존 정답 계산에만 쓰입니다.
- 분석의 r 계산(`detector_r`)은 offload 결정, θ, histogram만 씁니다. 요청 종류는 oracle 행, AUROC, 추정 오차의 정답값(s_true)에만 쓰입니다.
- M_k는 device가 학습한 class의 집합이며 평가 요청의 label이 아닙니다.

## 5. 시간과 계획 (6.1의 5번)

| run | 1라운드 시간(시험, GPU 한 대에 run 하나) | 개발용 run의 예상 시간(GPU마다 run 2–3개) |
|---|---|---|
| `cnn-scratch` S1 | 약 20초(판별기 8.6초 포함) | 약 2시간 |
| `pretrained-finetuned-block` S1 | 약 43초(판별기 8.5초 포함) | 약 4–7시간 |
| `pretrained-frozen-block` S1 | 시험은 CIFAR-100에서 약 45초 | 약 5–7.5시간 |

- 전체 계획은 `plan.md`에 있습니다. 개발용 run 9개와 재생 9개를 2026-11-20 16:54에 시작했습니다.

## 6. 아직 적지 않은 것

- held-out 10% 때문에 규칙의 수치가 v32와 달라진 크기는 `cnn-scratch`의 평가용 S1 run이 끝난 뒤 2단계 보고서에 적습니다. 이 비교는 설계 선택에 쓰지 않습니다.
