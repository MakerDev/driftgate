# Round 16 실제 calibration 경로의 사전 확인

이 문서는 지시문 4.6절에 따라 새 학습 없이 확인한 사실을 적습니다. 정확도 결과를 계산하기 전에 작성했고, 경로를 고를 때에는 자료의 가용성과 시스템 조건만 근거로 삼았습니다. 수치 근거는 `tables/R16_C_pools.csv`와 `scripts/r16_calibration_pool_check.py`에 있습니다.

## 1. 최종 평가 이미지와 겹치지 않는 labeled pool

시뮬레이터는 CIFAR 시험 세트(10,000장)를 run seed에 따라 클래스별로 나눕니다. 20%는 probe pool이 되고, 80%는 평가 pool이 됩니다(`src/disjoint_pools.make_pool_masks`). 평가 요청은 평가 pool에서만 만들어집니다(`runner_r6`: `eval_rb = RequestBuilder(eval_labels, ...)`).

| 데이터셋 | seed | probe pool | 평가 pool | 겹침 | probe pool의 클래스 | 클래스당 probe 이미지 |
|---|---|---|---|---|---|---|
| CIFAR-10 | 0–7 | 2,000 | 8,000 | 0 | 10개 전부 | 200 |
| CIFAR-100 | 0–2 | 2,000 | 8,000 | 0 | 100개 전부 | 20 |

- probe pool은 모든 seed에서 평가 pool과 겹치지 않습니다. 시험 세트의 이미지는 모두 두 pool 중 하나에만 속합니다.
- 모델 학습에는 CIFAR 학습 세트만 쓰였습니다. 따라서 probe pool은 학습 데이터와도 겹치지 않습니다.
- Round 12, 13b, 15의 고정 λ run(`mode = fixed`)에서도 probe pool은 device 신호를 기록하는 데 쓰였습니다(`record_device_signals = True`). 이 계산은 gradient 없이 별도 난수로 이루어지고, 고정 λ에서는 학습에 영향을 주지 않습니다(`runner_r6` docstring).
- 시뮬레이터 안에는 probe pool의 label이 있습니다. 그러나 v28 원고의 시스템 모형에는 공용 labeled pool이 없습니다. 현재 방법에서 probe의 역할은 label 없는 최근 요청을 흉내 내는 것입니다.

## 2. 실제 배포에서 자료를 가질 수 있는 장치

시뮬레이터가 모든 CIFAR 이미지를 읽을 수 있다고 해서, 서버나 device가 그 이미지에 접근할 수 있다고 가정하지 않습니다. 현재 시스템 모형에서 각 장치가 가진 자료는 다음과 같습니다.

- device k는 자기 학습 데이터(Main 클래스)와 device block h_k, device exit를 가집니다.
- edge는 cell의 edge block과 edge exit, M_k, 학습 label count를 가집니다.
- 다른 device의 raw 이미지와 label에 접근하는 장치는 없습니다.

공용 labeled pool을 쓰려면 이 목록에 자료가 하나 더 들어가야 합니다. CIFAR 크기에서 pool 전체(2,000장)는 이미지당 3,072 B이므로 약 6.1 MB입니다. 이 pool을 device에 미리 두거나 edge가 필요할 때 보내는 방식을 생각할 수 있지만, 지금 원고는 둘 다 가정하지 않습니다.

## 3. 대상 device의 h_k와 device exit로 처리하는 경로

지시문은 Main과 non-Main 자료를 모두 대상 device k의 h_k와 device exit로 처리하도록 요구합니다.

- 경로 A에서는 device k가 pool 이미지를 직접 h_k와 device exit에 통과시킵니다. 이어서 feature를 현재 cell의 edge로 보내 edge 출력을 얻습니다. 두 cell에 속한 device는 두 edge의 logit을 평균합니다. 다른 device의 h_j로 만든 feature는 쓰지 않습니다.
- 기본 CNN의 feature는 128×8×8개 값입니다. single precision으로 32 KB이므로, device마다 Main 256장과 non-Main 256장을 쓰면 calibration 한 번에 약 16 MB를 uplink로 보냅니다.
- 첫날 학습에서는 평가 round 사이에 모델이 바뀝니다. 따라서 평가 round마다 calibration을 다시 해야 합니다. 이 비용을 학습 uplink와 비교한 값은 2단계에서 측정합니다.
- 학습 종료 모델을 재생할 때에는 모델이 고정되어 있습니다. 그래서 같은 cell 조합이면 calibration 결과를 캐시로 재사용할 수 있습니다.

## 4. OOP와 OOR 클래스

probe pool에는 모든 클래스가 있습니다. 따라서 device k의 non-Main 표본에는 현재 cell의 OOP 클래스와 그 밖의 OOR 클래스가 모두 들어갈 수 있고, 가짜 표본을 만들 필요가 없습니다.

다만 실제 요청에서 non-Main의 구성은 calibration의 class 비율과 다릅니다. 실제 요청은 OOP가 ρ·|Main|이고, OOR이 0.3·ρ·|Main|이며, 각각 클래스에 고르게 나뉩니다. class-stratified로 뽑은 calibration 표본의 Acc_N을 이 구성에 맞추어 가중할지는 확인하지 못한 사항으로 남깁니다. 2단계에서 표본을 뽑기 전에 정해야 합니다.

## 5. 경로 B

경로 B에서는 각 device가 자기 학습 데이터의 10%를 학습에서 빼 둡니다.

- 기존 체크포인트는 학습 데이터 전체로 학습했습니다. 그러므로 경로 B는 재학습이 필요하고, 기존 체크포인트로는 독립 validation이 되지 않습니다.
- device k의 non-Main 표본은 다른 device가 가진 학습 데이터입니다. 따라서 device k의 h_k와 device exit를 소유 device로 보내서 그곳에서 계산해야 합니다. 기본 CNN에서 device block과 device exit는 0.28 M parameter이고, single precision으로 약 1.1 MB입니다.
- 이 전달은 device의 모델을 다른 device에 노출합니다. 현재 시스템 모형에는 이런 전달이 없습니다.

## 6. 결론 (결과를 보기 전)

| 경로 | 자료 | 시스템 조건 | 상태 |
|---|---|---|---|
| A: 공용 labeled pool (probe pool) | 있음. 평가 pool과 겹치지 않고, 모든 클래스가 있으며, 학습에 쓰지 않음 | 공용 labeled pool이라는 추가 자료 가정이 필요함. 현재 원고에는 없음 | 구현 가능. 이 가정을 쓰려면 사용자의 승인이 필요함 |
| B: device별 10% holdout | 재학습해야 생김 | 다른 device로 모델을 전달해야 함. 현재 시스템에는 없음 | 모델 전달이 허용될 때만 가능 |

지시문의 우선순위에 따르면 경로 A를 먼저 검토합니다. 경로 A로 진행하면 R15 체크포인트로 S1·S2 재생을 먼저 평가할 수 있습니다(지시문 5.4절). 하지만 "이미 허용된" 공용 labeled pool은 현재 시스템 정의에 없습니다. 그래서 1단계의 경로 상태를 `A_AVAILABLE_PENDING_APPROVAL`로 둡니다. 어느 경로도 사용자가 허용하지 않으면 2단계는 `BLOCKED_CALIBRATION`입니다.

1단계의 모사는 이전 평가 round의 요청과 label을 calibration 자료의 대용으로 씁니다. 요청 생성 규칙에 따라 device가 매 평가 round에 받는 Main 이미지는 같습니다. OOP와 OOR 이미지도 클래스별로 앞쪽 이미지를 쓰기 때문에 대부분 반복됩니다. 따라서 모사는 실제 경로 A보다 낙관적일 수 있습니다. 반복 이미지 비율은 `R16_S_estimation.csv`에 기록합니다.
