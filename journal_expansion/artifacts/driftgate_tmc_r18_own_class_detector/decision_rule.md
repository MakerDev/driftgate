# Round 18 판단 기준 (own class 판별기로 offloading을 정하는 방법)

이 문서는 지시문 7절을 옮기고, 그 기준을 적용하는 데 필요한 정의를 결과를 보기 전에 고정한 것입니다. 지시문 9.3에 따라 결과를 본 뒤에는 이 문서를 고치지 않습니다. 기준이 부적절하다고 판단되면 고치지 않고 최종 보고서에 이유를 적습니다. 작업 이름(Round 18)은 지시문의 이름을 따랐고, 저장소에 이미 있는 다른 작업 `driftgate_tmc_r18_compare`와는 폴더와 branch가 다릅니다.

## 0. seed, 조건, 기록

- **개발용 seed:** 지시문 2.4는 S1 seed 5–7을 개발용으로 정했습니다. 그런데 이 seed들은 이전 라운드(Round 7, 8, 10, 19, 20)에서 개발용으로 쓰였습니다. 그래서 쓰인 적이 없는 가장 작은 S1 seed인 8, 9, 10을 개발용 seed로 씁니다. provenance 기록에서 partition seed가 8 이상인 run은 없습니다. 환경 파일은 기존 생성기로 `runs/phaseT18oc_env/`에 만들었고, 생성기가 기존 S1 seed 5 파일을 그대로 다시 만드는 것을 먼저 확인했습니다(`scripts/r18oc_make_env.py`).
- **평가용 seed:** 2단계에서는 S1 0–2, S2 0–2, CIFAR-100(S1 배치) 0–2를 씁니다. 3단계에서는 S1과 S2를 0–4로 늘리고, 나머지 설정은 0–2를 씁니다.
- **조건:** 모든 학습 run은 같은 하루를 두 번 이어 학습합니다(300 라운드).
  - `first-day`: 라운드 1–150의 평가 시점 31개(1, 5, 10, …, 150)입니다.
  - `second-day`: 라운드 151–300의 평가 시점 31개(151, 155, 160, …, 300)입니다. 둘째 날의 평가 시점은 첫날과 같은 시각입니다.
  - `frozen-replay`: 라운드 150이 끝난 상태(모델, batch normalization 통계, 판별기, 검증 통계, edge의 class 개수)를 고정하고 첫날의 이동과 요청을 재생합니다. 평가 시점은 1, 5, …, 150입니다. 재생 중에는 학습, 판별기 갱신, 검증 통계 갱신이 없습니다.
- 요청 이미지, 요청 종류, 도착 순서는 Round 15와 같은 규칙으로 만듭니다. 둘째 날의 도착 순서는 라운드 번호가 다르므로 첫날과 다릅니다.

## 1. 곡선과 지표의 정의

### 1.1 offloading 곡선

- run 하나와 조건 하나에서 그 조건의 모든 평가 요청을 모읍니다. 점수 s가 클수록 먼저 offload합니다.
  - 판별기: s = −g입니다. 즉 g가 작을수록 먼저 offload합니다. 판별기가 없는 device의 요청은 g = −1이므로 항상 먼저 offload됩니다.
  - `device-entropy`: entropy, `device-max-prob`: −(최대 확률), `device-energy`: −logsumexp(logit), `own-mahalanobis`: 거리, `own-knn`: cosine 거리입니다.
- 비율 격자는 ρ ∈ {0.05, 0.10, …, 1.00}(20점)입니다. 비율 ρ에서는 점수가 큰 순서로 ⌈ρN⌉개 요청을 offload합니다. 동점은 요청마다 고정한 난수(numpy default_rng([partition seed, 18, 7]))로 정합니다. 이 방식은 θ를 전체 요청의 분위수로 훑는 것과 같습니다. 판별기의 θ는 offload된 요청 가운데 가장 큰 g와 offload되지 않은 요청 가운데 가장 작은 g의 중점입니다.
- 정확도는 v32 원고의 식(accuracy)입니다. 평가 라운드마다 요청이 있는 device의 정확도를 같은 비중으로 평균하고, 그 값을 조건의 평가 라운드에서 다시 평균합니다.
- 곡선은 seed마다 구하고, 판정에는 seed 평균 곡선을 씁니다. 격자 사이는 선형 보간합니다.

### 1.2 답 규칙

offload하지 않은 요청은 device exit로 답합니다. offload한 요청에는 다음 규칙 가운데 하나를 씁니다. a_k와 보정식은 v32 원고와 같습니다(π_k = Λ·cell 분포 + (1 − Λ)·전체 분포, Λ = 0.5, a_k는 [0.01, 0.99]로 자름).

| 이름 | 규칙 |
|---|---|
| `edge-raw` | 보정하지 않은 edge exit의 argmax |
| `edge-corrected-even` | r = 0.5로 보정한 edge exit의 argmax |
| `average-corrected-even` | v32의 방법입니다. r = 0.5로 보정한 edge 확률과 device 확률을 DriftGate 가중치로 평균합니다. 가중치는 Round 15의 DriftGate-P 정의(같은 offload 집합의 window에서 두 entropy의 평균, `r17_stage0.dgp_weight`)이고, 모든 요청을 offload하면 Round 13b F-auto 가중치(`r16_stage1.driftgate_weight`)와 같습니다. |
| `probability-average` | (p_d + p_e)/2의 argmax |
| `logit-sum` | log p_d + log p_e의 argmax (확률 하한 1e−8) |
| `edge-corrected-by-detector` | 지시문 3.4입니다(1.3절). |

### 1.3 `edge-corrected-by-detector`

- q는 그 device의 직전 128개 요청(현재 요청 제외, 평가 라운드와 도착 순서의 순서, 재생은 빈 상태에서 시작) 가운데 offload한 비율입니다. 직전 요청이 16개 미만이면 ŝ = 0.5로 둡니다.
- t(θ)는 other 분포에서 g < θ인 비율이고, u(θ)는 own 분포에서 g < θ인 비율입니다. 64칸 histogram에서 칸 안은 균등하다고 보고 선형 보간합니다. 사용하는 histogram은 그 평가 시점에 device가 쓰는 판별기의 것입니다.
- ŝ = (q − u)/(t − u)를 [0.02, 0.98]로 자릅니다. t − u < 0.1이면 ŝ = 0.5입니다. r = ŝ·t/(ŝ·t + (1 − ŝ)·u)를 [0.05, 0.95]로 자릅니다. 분모가 0이면 r = 0.5입니다. 판별기가 없는 device는 r = 0.5입니다.
- offload한 요청은 이 r로 보정한 edge exit의 argmax로 답합니다.

### 1.4 새 방법, 비교 대상, oracle

- **새 방법:** `detector-offload`(3.5절에서 고른 판별기, s = −g) + `edge-corrected-by-detector`입니다.
- **비교 대상:** 4.1절의 점수 5개(`device-entropy`, `device-max-prob`, `device-energy`, `own-mahalanobis`, `own-knn`)와 4.2절의 답 규칙 5개(`edge-raw`, `edge-corrected-even`, `average-corrected-even`, `probability-average`, `logit-sum`)를 조합한 25개 규칙입니다. v32의 주 방법(`device-entropy` + `average-corrected-even`)도 여기에 들어갑니다.
- B(ρ)는 비율 ρ에서 25개 규칙의 seed 평균 정확도 가운데 가장 높은 값입니다. 비율마다 따로 고르고, 고른 규칙의 이름을 적습니다. 비교 대상 곡선의 위쪽 경계는 각 ρ의 B(ρ)를 이은 곡선입니다.
- **참고 행:** Device only, Edge only(보정하지 않은 edge exit, 비율 1), Confidence-based offloading(0.8 nats)을 적습니다. 판정에는 쓰지 않습니다.
- **`request-kind-oracle`:** 점수는 1[요청이 Main이 아님]이고, 동점은 1.1절의 고정 난수로 정합니다. offload한 요청은 보정하지 않은 edge exit로 답합니다. 따라서 non-Main 비율에서 Main은 device exit, 나머지는 edge exit가 됩니다. 여지 판정(2.2절)에만 씁니다. 같은 점수에 `edge-corrected-even`을 쓴 판도 참고로 적습니다.

### 1.5 지표

- A* = B(1.0) − 0.5 %p입니다.
- `ratio-at-target`은 seed 평균 곡선이 A* 이상이 되는 첫 비율입니다. 첫 격자점 i에서 처음 A* 이상이 되면, i−1과 i 사이를 선형 보간한 비율입니다. 첫 점(0.05)에서 이미 A* 이상이면 0.05로 적습니다. 1.0까지 닿지 않으면 "없음"입니다.
- 절감률 S = 1 − (새 방법의 `ratio-at-target`) / (위쪽 경계의 `ratio-at-target`)입니다. 새 방법이 "없음"이면 S = 0입니다.
- G(ρ) = (새 방법의 seed 평균 정확도) − B(ρ)입니다. 표에는 같은 seed끼리 뺀 차이(새 방법 − B(ρ)로 고른 규칙)의 표준편차(ddof = 1)를 함께 적습니다.
- `accuracy-at-ratio`는 ρ = 0.25, 0.5, 0.75의 정확도입니다.
- AUROC: own class가 아닌 요청(Main이 아닌 요청)을 양성으로 두고, 점수가 클수록 양성으로 봅니다. 판별기는 1 − g를 씁니다. 동점은 평균 순위입니다. run마다 조건의 모든 요청에서 계산하고 전체, 집, 밖으로 나눕니다.
- 추정 오차: |ŝ − s_true|의 평균입니다. s_true는 같은 직전 128개 요청 가운데 Main이 아닌 요청의 비율입니다. 고정 문턱(θ = 0.5), 곡선의 ρ = 0.5, 예산 controller β = 0.5에서 계산하고 집과 밖으로 나눕니다.
- 예산 controller: device가 직전 128개 요청의 g에서 β 분위수(numpy 선형 보간)를 θ로 씁니다. 직전 요청이 16개 미만이면 θ = 0.5입니다.
- 모든 표에는 seed 평균과 같은 seed끼리 뺀 차이의 표준편차를 적습니다. 통계 검정은 하지 않습니다.

## 2. 지시문 7절

### 2.1 칸의 판정 (7.1)

칸 하나는 (설정, 모델 구성, 조건)입니다. 평가용 seed의 평균으로 판정합니다.

| 판정 | 기준 |
|---|---|
| 통과 | S ≥ 0.25 또는 G(0.5) ≥ 2.0 %p |
| 부분 통과 | 통과는 아니고, S ≥ 0.15 또는 G(0.5) ≥ 1.0 %p |
| 미달 | 그 밖 |

- 새 방법의 `ratio-at-target`이 "없음"이면 S는 0입니다. 위쪽 경계는 비율 1.0에서 B(1.0)이므로 A*에 반드시 닿습니다.

### 2.2 여지의 판정 (7.2)

- `request-kind-oracle`(1.4절)로 같은 S와 G(0.5)를 구합니다. oracle이 통과 기준에 못 미치는 칸은 "여지 없음"으로 적습니다. 그 칸에서는 어떤 판별기도 통과할 수 없다고 봅니다.

### 2.3 2단계 뒤의 진행 (7.3)

판정에 쓰는 칸은 사전학습 구성마다 S1과 S2의 `second-day`와 `frozen-replay`, 네 칸입니다.

| 상황 | 다음 작업 |
|---|---|
| 한 구성의 네 칸이 모두 통과 | 그 구성을 주 구성으로 3단계를 합니다. 두 구성이 모두 해당하면 `pretrained-finetuned-block`이 주 구성입니다. |
| 네 칸이 모두 부분 통과 이상인 구성이 있음 | 7.4절의 개선 시도 1번을 한 번 하고, 개발용 seed에서 더 나은 쪽으로 3단계를 합니다. |
| 미달인 칸이 있지만 여지는 있음 | 7.4절의 개선 시도를 차례로 합니다. 네 칸이 모두 부분 통과 이상이 되면 3단계를 합니다. |
| CIFAR-10의 네 칸이 "여지 없음"이지만 CIFAR-100의 `second-day`와 `frozen-replay`는 통과 | class가 많을 때만 문제가 남는다고 적고, CIFAR-100과 iNaturalist를 주 설정으로 3단계를 합니다. S1과 S2는 참고 결과로 둡니다. |
| 사전학습 구성에서 여지가 없거나, 개선 시도 3번 뒤에도 미달 | 3단계를 건너뛰고 4단계로 갑니다. 판정은 "중단 권고"입니다. `cnn-scratch`에서만 통과했다면 그 사실과 수치를 적습니다. |

- 네 칸 가운데 일부만 "여지 없음"이면, 그 칸을 뺀 나머지 칸으로 위 표를 적용합니다. 이 경우 최종 판정은 "조건부 권고" 이하로 둡니다.
- 위 표의 행은 위에서부터 차례로 검사하고, 처음 해당하는 행을 따릅니다. 두 구성이 서로 다른 행에 해당하면 더 앞의 행에 해당하는 구성을 따릅니다.
- 추가 조건: `first-day`에서 G(0.5)가 −0.5 %p보다 낮은 칸은 목록으로 적습니다. 진행을 막지는 않습니다.

### 2.4 개선 시도 (7.4, 최대 3번, 순서 고정)

각 시도는 S1의 개발용 seed(8–10)에서 먼저 돌립니다. `frozen-replay`의 offloading 비율 0.5에서 새 방법의 seed 평균 정확도가 기존 판보다 0.5 %p 이상 오르면 평가용 seed(S1, S2)로 넘어가고, 아니면 버리고 다음 시도로 갑니다.

1. `exit-with-other-class`: device exit에 "other" 출력을 하나 더하고, edge가 보낸 class별 평균과 대각 분산에서 뽑은 가상 음성 벡터로 local 학습 때 함께 학습합니다. 판별기는 이 출력의 확률을 입력에 더합니다.
2. `deeper-split`: device block을 `features[0]`–`features[10]`(출력 64채널, stride 16)으로 늘립니다. device 계산량과 feature 크기를 비용 표에 함께 적습니다.
3. 스스로 세운 가설 하나: 2단계 결과에서 실패 원인을 진단하고 그 원인 하나를 겨냥한 변경을 합니다. 돌리기 전에 가설, 변경, 기대하는 수치를 `decisions.md`에 적습니다.

### 2.5 최종 판정 (7.5)

| 판정 | 기준 |
|---|---|
| TMC 진행 권고 | 주 구성에서 3단계 설정의 `second-day`·`frozen-replay` 칸 가운데 3분의 2 이상이 통과이고 나머지가 부분 통과 이상입니다. iNaturalist 설정이 부분 통과 이상입니다. 200대에서 미달이 아닙니다. |
| 조건부 권고 | 위 기준에 못 미치지만 절반 이상의 칸이 부분 통과 이상입니다. 부족한 칸과, 그것을 메우는 데 필요한 작업을 적습니다. |
| 중단 권고 | 그 밖입니다. 지금까지의 결과로 쓸 수 있는 논문의 규모(letter, 학회)를 한 줄로 적습니다. |

- 기준의 경계에 걸리면 낮은 쪽 판정을 적고, 경계에 걸린 칸을 함께 적습니다. 경계에 걸린다는 것은 S 또는 G(0.5)가 기준값과 0.01(S) 또는 0.05 %p(G) 안에 있는 경우입니다.
- 3단계를 하지 않으면(2.3절의 마지막 행) 판정은 "중단 권고"입니다.

## 3. 설계 선택 (지시문 3.5)

- 고르는 값은 판별기 입력(`block`, `exit`, `concat`), 모델(`logreg`, `mlp64`), 음성 표본(`cell`, `global`, `both`)의 18개 조합입니다.
- 모델 구성마다 S1 개발용 seed 8–10의 `frozen-replay`에서 비율 0.5의 새 방법 정확도(seed 평균)를 비교하고 가장 높은 조합을 고릅니다.
- 가장 높은 조합과의 차이가 0.10 %p 이내인 조합이 있으면 그 가운데 가장 단순한 조합을 고릅니다. 단순한 순서는 모델(`logreg` < `mlp64`), 입력(`block` < `exit` < `concat`), 음성 표본(`cell` < `global` < `both`)의 순서로 비교합니다. 고른 조합과 고르지 않은 값의 결과를 모두 표로 남기고 `decisions.md`에 적습니다.
- 고른 조합은 그 모델 구성의 모든 평가용 run에 쓰고, 그 run에서는 이 조합에 `frozen-at-home`과 `every-10` 판도 함께 기록합니다.

## 4. 구현에서 정한 값 (결과를 보기 전에 고정)

| 항목 | 값 |
|---|---|
| 판별기 학습 | 전체 표본 Adam, 300 step, 학습률 0.01, 가중치 L2 1e−3, 라운드마다 새로 초기화 (logistic regression은 0, MLP는 PyTorch 기본 균등 분포, seed 고정) |
| 입력 표준화 | device마다 학습 행 전체(양성, cell 음성, 가상 음성)의 평균과 표준편차 |
| 양성 | device가 local 학습을 마친 모델(eval mode)로 만든 자기 학습 표본의 block 출력. exit 벡터는 그 라운드에 설치한 device exit로 계산 |
| cell 음성 | 같은 라운드에 같은 cell(두 cell이면 합집합)에서 학습한 다른 device의 학습 표본 가운데 label이 M_k 밖인 것. 그 device의 block 출력을 쓰고 exit 벡터는 device k의 exit로 계산. 8192개를 넘으면 seed를 고정한 무작위 부분집합 |
| cell 밖 음성 | device의 cell에 있는 학습 label에 없고 M_k에도 없는 class마다 64개. 모든 참여 device의 자기 벡터로 구한 class 평균과 대각 분산의 정규분포 |
| 가중치 | 양성 합 0.5, 음성 합 0.5 |
| 검증 통계 | 설치한 모델로 forward한 held-out. own은 자기 held-out, other는 다른 참여 device의 held-out 가운데 label이 M_k 밖인 것. 64칸 histogram |
| held-out | device마다 class별 10%(반올림, class마다 최소 1개), numpy default_rng([partition seed, 18, device]) |
| 학습 설정 | v32와 같음(SGD 0.01, momentum 0, weight decay 1e−4, batch 32, local epoch 3, λ = 0.4, Λ = 0.5, γ = 0.5) |
| MobileNetV2 | torchvision IMAGENET1K_V1 가중치. CIFAR 정규화를 ImageNet 정규화로 바꾸고 64×64로 bilinear 확대 |

이 값들은 지시문 9.2가 말하는 설정에 해당하므로 평가용 seed의 결과를 보고 바꾸지 않습니다.
