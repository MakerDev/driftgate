# DriftGate 최종 실험 보고서 (Round 5)

이 보고서는 IEEE TMC 원고에 들어갈 비모바일 실험을 모두 정리한다. 모든 수치는 원시 run JSON에서 스크립트로 계산한 CSV에서 가져왔다 (`scripts/r5_tables.py` → `tables/*.csv` → 이 보고서). 이 보고서에서 **DriftGate**는 relonly controller를 뜻한다. relonly는 TV 신호를 temporal 비교와 spatial 비교의 표준화 점수로만 λ와 Λ로 바꾸며, absolute branch가 없다.

용어와 표기는 다음과 같다.

- **pp**는 정확도 백분율의 차이(퍼센트포인트)다. **integrated accuracy**는 평가 라운드마다 잰 acc_total의 평균이다.
- 차이는 모두 같은 seed끼리 뺀 paired 차이이며, 괄호 안에 Student-t 95% 신뢰구간(CI)을, 그 뒤에 양수인 seed 수/전체 seed 수(n_pos/n)를 적는다.
- 설정 이름: stepwise composition change는 기존 Schedule A, client mobility는 기존 mobility-med, late abrupt change는 기존 post-convergence abrupt다.
- **absonly**는 TV 크기(평활한 d̂)로 λ = 0.70 − 0.55·d̂를 바로 정하는 controller(Round 4 B1)다. **APFL**은 기기의 라벨 손실로 λ를 학습하는 기준 방법(B4)이다.
- 모든 새 run은 controller용 표본과 평가용 표본을 나눈 disjoint pool로 실행했고, 64개 모두 두 집합이 겹치지 않았다(overlap = 0).

## 1. 주장별 판정표

| # | 주장 | 근거 표·그림 | 판정 | 핵심 수치 |
|---|---|---|---|---|
| 1 | traffic이 바뀌는 두 설정에서 DriftGate는 시험한 모든 고정 λ보다 높다 | 표 1, 그림 4 | 지지 | stepwise: 최고 고정값 λ=0.4 대비 +0.76 pp [+0.41, +1.11] (5/5), λ=0.2·0.3·0.5 대비 +0.83 pp에서 +1.23 pp. mobility: 최고 고정값 λ=0.4 대비 +0.65 pp [+0.47, +0.83] (3/3). 고정 λ와의 8개 비교 모두 CI 하한이 0보다 크다 |
| 2 | 가장 좋은 λ는 구간에 따라 뒤바뀌고, DriftGate는 어느 구간에서도 가장 나쁜 고정값 수준까지 떨어지지 않는다 | 표 2c, 그림 4 | 부분 지지 | 최고 고정값은 stepwise의 ρ=0 구간에서 λ=0.5, ρ>0 구간에서 λ=0.2이고, mobility에서는 ρ 변화 전 λ=0.6, 뒤 λ=0.2다. DriftGate의 평균은 7개 구간 모두에서 최저 고정값보다 높고, 6개 구간에서 CI 하한이 0보다 크다. stepwise ρ=0.4 상승 구간은 +0.38 pp [-0.20, +0.96] (4/5)로 CI가 0을 포함한다 |
| 3 | 같은 controller에서 TV는 entropy보다 높다 | 표 1, 표 3, 사전 확인 1 | 지지 | stepwise +1.56 pp [+1.18, +1.94] (5/5), mobility +2.02 pp [+1.26, +2.77] (3/3), SVHN +2.93 pp [+1.86, +4.01] (5/5) |
| 4 | 기기의 라벨로 λ를 학습하는 APFL식 방법은 traffic 변화를 따라가지 못한다 | 표 1, 표 4c, 표 4d, 그림 3 | 부분 지지 | stepwise의 ρ=0.8 구간(R61–90)에서 APFL의 평균 λ는 η=0.01에서 0.634, η=0.1에서 0.608이다. λ와 ρ의 Spearman 상관은 η=0.01에서 +0.681, η=0.1에서 +0.363이다. DriftGate의 같은 값은 0.229, -0.686이다. mobility에서는 APFL의 λ도 ρ 변화 뒤 내려가지만, 감소폭(η=0.01은 0.029, η=0.1은 0.046)이 DriftGate의 0.169보다 작다. ρ=0.8 구간 정확도는 DriftGate가 +6.95 pp, +6.89 pp(stepwise), +5.47 pp, +4.73 pp(mobility) 높다. 반대 결과: mobility의 integrated accuracy는 APFL η=0.1이 평균으로 더 높다(DriftGate − APFL -0.35 pp [-1.44, +0.74] (1/3)) |
| 5 | 비교 점수로 λ를 정하는 DriftGate는 TV 크기로 λ를 바로 정하는 controller보다 높다 | 표 4a | 지지 | DriftGate − absonly: stepwise +1.01 pp [+0.73, +1.29] (5/5), mobility +1.30 pp [+1.06, +1.53] (3/3), CIFAR-100 gradual +1.20 pp [+1.07, +1.33] (3/3) |
| 6 | 결과가 guard와 λ_max 값에 크게 좌우되지 않는다 | 표 4b | 부분 지지 | 기본값 대비 stepwise: guard 0.25 -0.00 pp [-0.16, +0.15], guard 1.0 -0.23 pp [-0.82, +0.35], λ_max 0.60 -0.51 pp [-1.15, +0.12], λ_max 0.80 -0.27 pp [-1.72, +1.18]. mobility: λ_max 0.60 +0.11 pp [-0.30, +0.52], λ_max 0.80 -1.15 pp [-1.51, -0.80](0/3). mobility의 λ_max 0.80은 같은 seed의 최고 고정값(λ=0.4)과의 차이도 -0.50 pp [-1.03, +0.03]이다 |
| 7 | 다른 데이터셋에서 DriftGate와 가장 좋은 고정값의 차이 | 표 3 | 판정 없음 | CIFAR-10 gradual: fixed 0.2 대비 -0.17 pp [-2.08, +1.73] (2/3) / CIFAR-100 gradual: fixed 0.15 대비 +0.17 pp [-0.20, +0.53] (3/3) / Tiny-ImageNet: fixed 0.15 대비 -0.57 pp [-0.95, -0.19] (0/3) / SVHN temporal: fixed 0.15 대비 -0.54 pp [-1.40, +0.31] (1/5). CI가 0을 포함하지 않는 설정: Tiny-ImageNet |
| 8 | ResNet-18에서도 DriftGate가 고정값보다 높다 | 표 5 | 지지 | DriftGate 62.39%, 고정 λ=0.4 대비 +3.19 pp [+0.37, +6.01] (3/3), 고정 λ=0.2 대비 +6.10 pp [+2.12, +10.07] (3/3). seed 3개, client 16명, 고정값은 두 개만 시험했다 |
| 9 | TV는 traffic 구성을 따라가고, entropy는 학습 진행을 따라간다 | 표 6a–6c, 그림 1 | 지지 | late abrupt change에서 TV는 shift에 +0.053 변하고 shift 전 35 라운드 동안 +0.008 변한다. entropy(H/ln C)는 shift에 +0.039 오르지만 shift 전 35 라운드 동안 -0.100 내려간다. ρ와의 Spearman은 TV +0.91/+0.85, entropy +0.44/-0.58(단계 변화/late abrupt) |

판정 기준은 다음과 같다. 해당 비교의 차이가 모두 주장 방향이고 95% CI가 0을 넘으면 **지지**, 방향은 맞지만 일부 비교의 CI가 0을 포함하거나 일부 조건에서 반대 결과가 나오면 **부분 지지**, 주요 비교가 반대 방향이면 **반대**로 판정했다.

## 2. 시작 전 확인

### 2.1 entropy 실행과 relonly 실행의 controller 경로

**결론: 같은 경로다. 따라서 entropy 재실행(13개)은 하지 않았다.**

- **실행 flag**: provenance에 기록된 명령을 보면 두 실행군의 flag는 `--signal`만 다르다(entropy는 `ent_client`, relonly는 `tv_dist`). 둘 다 `--mode selfcal --burn_in 10 --z_guard 0.5 --spatial_norm --disjoint_pools`를 쓰고, 둘 다 `--abs_cap`이 없다.
- **resolved config**: stepwise seed 0의 두 run JSON에서 `config`를 비교하면 `controller_signal`과 run_id만 다르고 나머지 항목은 모두 같다.
- **코드 경로**: `scripts/run_v2.py:177-178`은 `--abs_cap`이 있을 때만 `abs_cap=True`로 둔다. `src/runner.py:250-251`은 flag가 없으면 `abs_cap=False`, `abs_only=False`로 controller를 만든다. 이때 `src/controllers/self_calibrating.py:168-169`의 `use_abs`가 False이므로 absolute 후보를 계산하지 않고, `185-187`행에서 λ와 Λ가 relative 후보 그대로 정해진다. 상수는 `self_calibrating.py:31-33`(z0 = 1.5, τ = 0.75, EMA α = 0.3)과 `normalizers.py:21,26,49`(σ 하한, β = 0.05, 점수 clip [−2, 6])에서 두 실행이 같은 값을 쓴다.
- **동작 확인**: entropy 실행은 Round 4의 코드 수정(`--abs_only` flag, λ_rel/λ_abs 로깅)보다 먼저 돌았다. 그래서 기록된 controller 점수 z에 relative 매핑 λ = 0.70 − 0.55·sigmoid((z − 1.5)/0.75), Λ = 0.70 − 0.30·sigmoid((z − 1.5)/0.75)만 적용해 기록된 λ와 Λ를 재구성했다. 결과는 아래와 같다(`precheck/precheck_controller_path.csv`).

| 실행군 | seed 수 | warm-up 이후 cluster-라운드 | λ 최대 오차 | Λ 최대 오차 | warm-up 불일치 |
|---|---|---|---|---|---|
| entropy, stepwise | 5 | 625 | 1.1e-16 | 1.1e-16 | 0 |
| entropy, mobility | 3 | 475 | 1.1e-16 | 1.1e-16 | 0 |
| entropy, SVHN | 5 | 625 | 1.1e-16 | 1.1e-16 | 0 |
| relonly, stepwise | 5 | 625 | 1.1e-16 | 1.1e-16 | 0 |
| relonly, mobility | 3 | 475 | 1.1e-16 | 1.1e-16 | 0 |
| relonly, CIFAR-100 gradual | 3 | 625 | 1.1e-16 | 1.1e-16 | 0 |

모든 seed에서 오차가 부동소수점 반올림 수준(1e-16)이고, 처음 25 라운드는 모두 λ = 0.425, Λ = 0.55다. 따라서 기존 entropy 실행은 relonly controller에 신호로 entropy를 넣은 실행과 같다.

### 2.2 relonly 실행의 Λ 계산

B1 relonly 실행(stepwise 5개, mobility 3개, CIFAR-100 gradual 3개)에서 Λ는 λ와 같은 relative 점수 q로 정해졌다. 위 표의 relonly 행에서 Λ = 0.70 − 0.30·sigmoid((q − 1.5)/0.75)가 모든 cluster-라운드에서 성립하고, absolute 후보를 기록하는 `history["lam_abs"]`는 모든 라운드에서 비어 있다.

### 2.3 코드, GPU, 대기열

- **코드**: 저장소 HEAD는 `e82b96b`다. 하지만 `adaptive_splitomc_tmc/` 디렉터리 전체가 git에 추적되지 않아(`git status`에서 `??`) 이 commit만으로는 실행 코드를 특정할 수 없다. 그래서 run이 실행한 소스 파일 17개의 sha256을 `precheck/code_identity.txt`에 기록했고, 모든 run이 끝난 뒤 다시 계산해 바뀌지 않았음을 확인했다.
- **GPU**: 새 run 64개는 모두 `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 실행되어 물리 GPU 0만 썼다. GPU 1은 쓰지 않았다. 10분마다 워커를 다시 띄우던 cron supervisor가 GPU 1에도 워커를 띄우던 설정이었으므로, GPU 0 워커만 띄우도록 바꿨다(원본은 `scripts/supervisor.sh.bak_pre_r5_*`).
- **멈춘 run 정리**: 작업 시작 시점에 Round 4의 B3 λ = 0.15 run 12개가 Round 4 중단 요청 때 일시 정지(SIGSTOP)한 상태로 3일 넘게 GPU 메모리를 잡고 있었다. 이 중 6개는 GPU 1에 있어 GPU 0만 쓰는 조건에서는 이어서 돌릴 수 없었다. GPU 0의 6개도 GPU 1 워커와 함께 돌던 이전 대기열에 묶여 있어 함께 종료했다(목록: `runs/queue/killed_r5_20260928_0523.txt`). 보고에 쓰는 B3 run은 아직 시작하지 않았던 run을 포함해 10개를 GPU 0에서 처음부터 돌렸다. GPU 0에 있던 Jupyter 커널은 끄지 않았다(메모리가 충분했고, 이후 스스로 종료됐다).
- **대기열 정리**: 대기 중이던 B5 16개와 B6 18개를 취소했다(시작한 run은 없었다. 목록: `runs/queue/cancelled_r5_B5_B6_*.txt`). CIFAR-100 spatial의 relonly 실행과 server non-Main prediction rate 재실행은 대기열에 없었다. B3 λ = 0.15의 CIFAR-100 spatial run 3개는 정지 상태에서 종료했고, 보고에 넣지 않으므로 다시 돌리지 않았다. entnorm 결과는 추가로 분석하지 않았다.
- **B4 설정 확인**: 대기열의 B4 명령은 4절의 규칙과 같다. client마다 λ_k를 두고, 매 라운드 local 학습 뒤 θ(λ_k) = λ_k·θ_local + (1 − λ_k)·θ̄_k에서 local minibatch 1개로 ∂L/∂λ_k를 구해 λ_k ← clip(λ_k − η·∂L/∂λ_k, 0.15, 0.70)로 갱신한다(`src/apfl_baseline.py:7-9,23-28,68`). 초기값 0.425(`src/runner.py:258`), Λ = 0.5, η ∈ {0.01, 0.1}(모델 learning rate와 그 10배)이고 probe와 TV 신호는 쓰지 않는다.

## 3. 새 실험 결과

### 3.1 E1과 B3: 다른 데이터셋 (표 3, `tables/T3_transfer.csv`)

DriftGate는 E1의 relonly 실행(CIFAR-10 gradual, Tiny-ImageNet, SVHN)과 B1의 relonly 실행(CIFAR-100 gradual)을 쓴다. 고정 λ = 0.15는 B3 실행이고, 0.2와 0.4는 기존 disjoint 실행(Round 2–4)이다. 고정 λ 실행은 모두 Λ = 0.5다. 가장 좋은 고정값은 같은 seed 평균이 가장 높은 값이다.

| 설정 | 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |
|---|---|---|---|---|
| CIFAR-10 gradual | DriftGate | 3 | 62.68% (1.18) |  |
| CIFAR-10 gradual | fixed 0.15 | 3 | 62.78% (0.42) | -0.10 pp [-2.17, +1.98] (2/3) |
| CIFAR-10 gradual | fixed 0.2 (최고 고정값) | 3 | 62.86% (0.51) | -0.17 pp [-2.08, +1.73] (2/3) |
| CIFAR-10 gradual | fixed 0.4 | 3 | 62.14% (0.79) | +0.55 pp [-0.74, +1.83] (2/3) |
| CIFAR-100 gradual | DriftGate | 3 | 36.09% (1.36) |  |
| CIFAR-100 gradual | fixed 0.15 (최고 고정값) | 3 | 35.92% (1.44) | +0.17 pp [-0.20, +0.53] (3/3) |
| CIFAR-100 gradual | fixed 0.2 | 3 | 35.91% (1.42) | +0.18 pp [-0.13, +0.50] (3/3) |
| CIFAR-100 gradual | fixed 0.4 | 3 | 35.00% (1.33) | +1.09 pp [+1.02, +1.16] (3/3) |
| Tiny-ImageNet | DriftGate | 3 | 23.80% (0.21) |  |
| Tiny-ImageNet | fixed 0.15 (최고 고정값) | 3 | 24.37% (0.09) | -0.57 pp [-0.95, -0.19] (0/3) |
| Tiny-ImageNet | fixed 0.2 | 3 | 24.31% (0.09) | -0.52 pp [-0.89, -0.15] (0/3) |
| Tiny-ImageNet | fixed 0.4 | 3 | 23.82% (0.16) | -0.03 pp [-0.19, +0.14] (1/3) |
| SVHN temporal | DriftGate | 5 | 79.90% (1.68) |  |
| SVHN temporal | fixed 0.15 (최고 고정값) | 5 | 80.44% (1.71) | -0.54 pp [-1.40, +0.31] (1/5) |
| SVHN temporal | fixed 0.2 | 5 | 80.43% (1.69) | -0.53 pp [-1.34, +0.28] (1/5) |
| SVHN temporal | fixed 0.4 | 5 | 79.79% (1.69) | +0.11 pp [-0.67, +0.88] (3/5) |
| SVHN temporal | entropy | 5 | 76.97% (1.26) | +2.93 pp [+1.86, +4.01] (5/5) |

네 데이터셋에서 DriftGate와 가장 좋은 고정값의 차이는 -0.57 pp에서 +0.17 pp 사이다. Tiny-ImageNet에서는 DriftGate가 λ = 0.15보다 0.57 pp 낮고 CI가 0을 포함하지 않는다. 3개 데이터셋(CIFAR-100 gradual, Tiny-ImageNet, SVHN temporal)에서 가장 좋은 고정값은 시험한 범위의 끝인 λ = 0.15다. SVHN에서 DriftGate는 같은 controller에 entropy를 넣은 실행보다 +2.93 pp [+1.86, +4.01] (5/5) 높다.

### 3.2 E2: ResNet-18 middle split (표 5, `tables/T5_resnet18_middle.csv`)

기존 ResNet-18 분할 실험의 middle split 정의를 그대로 썼다(client는 stem과 stage 1–2, server는 stage 3–4와 fc, 분할 지점 feature는 128×16×16). 기존 ResNet-18 실험처럼 client는 16명이고, CIFAR-10 stepwise composition change를 150 라운드 돌렸다.

| 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |
|---|---|---|---|
| DriftGate | 3 | 62.39% (3.15) |  |
| fixed 0.2 | 3 | 56.29% (2.71) | +6.10 pp [+2.12, +10.07] (3/3) |
| fixed 0.4 (최고 고정값) | 3 | 59.20% (2.85) | +3.19 pp [+0.37, +6.01] (3/3) |

세 seed 모두에서 DriftGate가 두 고정값보다 높다. seed가 3개라 CI가 넓고, 고정값은 0.2와 0.4 두 개만 시험했으므로 이 표의 '최고 고정값'은 두 값 중 높은 쪽이다.

### 3.3 E3: 상수 민감도 (표 4b, `tables/T4b_constant_sensitivity.csv`)

한 번에 상수 하나만 바꿨다. 기준은 같은 seed(0–2)의 B1 relonly 실행이다. λ_max를 바꾸면 warm-up 동안 쓰는 중간값도 함께 바뀐다(λ_max 0.60이면 0.375, 0.80이면 0.475).

| 설정 | 변형 | 변형 정확도 | 기본값 정확도(같은 seed) | 변형 − 기본값 | 변형 − 최고 고정값(같은 seed) |
|---|---|---|---|---|---|
| stepwise composition change | guard 0.25 | 66.07% | 66.07% | -0.00 pp [-0.16, +0.15] (1/3) | +0.94 pp [+0.65, +1.24] (fixed 0.4) |
| stepwise composition change | guard 1.0 | 65.84% | 66.07% | -0.23 pp [-0.82, +0.35] (1/3) | +0.71 pp [+0.33, +1.09] (fixed 0.4) |
| stepwise composition change | λ_max 0.60 | 65.56% | 66.07% | -0.51 pp [-1.15, +0.12] (0/3) | +0.43 pp [-0.40, +1.26] (fixed 0.4) |
| stepwise composition change | λ_max 0.80 | 65.80% | 66.07% | -0.27 pp [-1.72, +1.18] (1/3) | +0.68 pp [-0.56, +1.91] (fixed 0.4) |
| client mobility | λ_max 0.60 | 59.34% | 59.23% | +0.11 pp [-0.30, +0.52] (2/3) | +0.76 pp [+0.50, +1.02] (fixed 0.4) |
| client mobility | λ_max 0.80 | 58.08% | 59.23% | -1.15 pp [-1.51, -0.80] (0/3) | -0.50 pp [-1.03, +0.03] (fixed 0.4) |

stepwise에서 네 변형과 기본값의 차이는 -0.51 pp에서 0.00 pp 사이이고, CI가 모두 0을 포함한다. 네 변형 모두 평균이 기본값보다 조금 낮다. mobility에서 λ_max 0.60은 기본값과 차이가 작다(+0.11 pp [-0.30, +0.52]). λ_max 0.80과 기본값의 차이는 -1.15 pp [-1.51, -0.80]이고, 기본값보다 높은 seed는 0/3개다. 이 변형과 같은 seed의 최고 고정값(λ = 0.4)의 차이는 -0.50 pp [-1.03, +0.03]이다.

### 3.4 B4: APFL식 λ 학습 (표 1, 표 4c, 표 4d)

| 설정 | 방법 | integrated accuracy (SD) | DriftGate − 방법 |
|---|---|---|---|
| stepwise composition change | APFL η=0.01 | 64.25% (2.00) | +1.68 pp [+1.37, +1.98] (5/5) |
| stepwise composition change | APFL η=0.1 | 64.97% (1.97) | +0.96 pp [+0.52, +1.39] (5/5) |
| client mobility | APFL η=0.01 | 58.00% (0.62) | +1.23 pp [+0.56, +1.89] (3/3) |
| client mobility | APFL η=0.1 | 59.58% (0.70) | -0.35 pp [-1.44, +0.74] (1/3) |

λ가 ρ를 따라 움직이는지 보려고 구간별 평균 λ와 ρ와의 Spearman 상관을 계산했다(`tables/T4c_apfl_lambda_by_segment.csv`). APFL의 λ는 client별 λ_k의 평균이고, DriftGate의 λ는 cluster 평균이다. 두 방법 모두 초기 25 라운드는 제외했다.

| 설정 | 방법 | 구간별 평균 λ | ρ와의 Spearman (SD) |
|---|---|---|---|
| stepwise composition change | DriftGate | 0.580 / 0.380 / 0.229 / 0.245 / 0.459 (R26–30 / R31–60 / R61–90 / R91–120 / R121–150) | -0.686 (0.176) |
| stepwise composition change | APFL η=0.01 | 0.614 / 0.640 / 0.634 / 0.613 / 0.589 (R26–30 / R31–60 / R61–90 / R91–120 / R121–150) | +0.681 (0.115) |
| stepwise composition change | APFL η=0.1 | 0.663 / 0.634 / 0.608 / 0.584 / 0.558 (R26–30 / R31–60 / R61–90 / R91–120 / R121–150) | +0.363 (0.088) |
| client mobility | DriftGate | 0.549 / 0.380 (R26–60 / R61–120) | -0.836 (0.000) |
| client mobility | APFL η=0.01 | 0.643 / 0.614 (R26–60 / R61–120) | -0.709 (0.111) |
| client mobility | APFL η=0.1 | 0.636 / 0.590 (R26–60 / R61–120) | -0.717 (0.019) |

stepwise에서 APFL의 λ는 ρ가 0.8일 때도 0.6 근처에 머물고, ρ와의 상관이 양수다. 즉 λ가 필요한 방향(ρ가 클 때 작은 λ)과 반대로 움직인다. mobility에서는 APFL의 λ도 ρ 변화 뒤 내려가고 ρ와의 상관도 음수다. 다만 감소폭(η=0.01은 0.029, η=0.1은 0.046)이 DriftGate의 0.169보다 작다.

구간별 정확도(`tables/T4d_apfl_segment_accuracy.csv`)를 보면 APFL은 ρ = 0 구간에서 DriftGate보다 높고, ρ가 큰 구간에서 크게 낮다.

| 설정 | 방법 | 구간 | APFL 정확도 | DriftGate − APFL |
|---|---|---|---|---|
| stepwise composition change | APFL η=0.01 | ρ=0 early | 58.79% | -0.56 pp [-1.04, -0.08] (1/5) |
| stepwise composition change | APFL η=0.01 | ρ=0.4 rising | 60.56% | +1.04 pp [-0.04, +2.12] (5/5) |
| stepwise composition change | APFL η=0.01 | ρ=0.8 | 52.81% | +6.95 pp [+4.92, +8.97] (5/5) |
| stepwise composition change | APFL η=0.01 | ρ=0.4 falling | 63.58% | +3.34 pp [+1.60, +5.08] (5/5) |
| stepwise composition change | APFL η=0.01 | ρ=0 late | 87.35% | -1.64 pp [-5.06, +1.77] (2/5) |
| stepwise composition change | APFL η=0.1 | ρ=0 early | 61.21% | -2.98 pp [-3.64, -2.31] (0/5) |
| stepwise composition change | APFL η=0.1 | ρ=0.4 rising | 60.42% | +1.18 pp [-0.07, +2.43] (4/5) |
| stepwise composition change | APFL η=0.1 | ρ=0.8 | 52.87% | +6.89 pp [+4.80, +8.98] (5/5) |
| stepwise composition change | APFL η=0.1 | ρ=0.4 falling | 64.17% | +2.74 pp [+1.08, +4.41] (5/5) |
| stepwise composition change | APFL η=0.1 | ρ=0 late | 87.45% | -1.75 pp [-5.10, +1.60] (2/5) |
| client mobility | APFL η=0.01 | ρ=0 (R1–60) | 63.29% | -2.41 pp [-3.20, -1.62] (0/3) |
| client mobility | APFL η=0.01 | ρ=0.8 (R70–120) | 51.84% | +5.47 pp [+3.66, +7.28] (3/3) |
| client mobility | APFL η=0.1 | ρ=0 (R1–60) | 65.58% | -4.70 pp [-6.37, -3.03] (0/3) |
| client mobility | APFL η=0.1 | ρ=0.8 (R70–120) | 52.58% | +4.73 pp [+3.34, +6.11] (3/3) |

mobility의 APFL η=0.1은 ρ = 0 구간에서 4.70 pp 앞서고 ρ = 0.8 구간에서 4.73 pp 뒤진다. 평가 라운드는 ρ = 0 구간이 7개, ρ = 0.8 구간이 6개라 integrated accuracy에서는 APFL이 평균 0.35 pp 높게 나온다(CI [-1.44, +0.74]).

## 4. 기존 로그 분석 결과

### 4.1 대표 비교: stepwise composition change와 client mobility (표 1, `tables/T1_main_results.csv`)

mobility의 고정 λ = 0.5는 Round 1 실행(`t1_mob_fx50`)을 썼다. Round 4의 같은 설정 실행(`b3_mob_fx50`)은 6절에 적은 이유로 쓰지 않았다.

| 설정 | 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |
|---|---|---|---|---|
| stepwise composition change | DriftGate | 5 | 65.93% (2.01) |  |
| stepwise composition change | fixed 0.2 | 5 | 64.70% (1.73) | +1.23 pp [+0.46, +2.00] (5/5) |
| stepwise composition change | fixed 0.3 | 5 | 65.10% (1.91) | +0.83 pp [+0.28, +1.37] (5/5) |
| stepwise composition change | fixed 0.4 (최고 고정값) | 5 | 65.17% (2.07) | +0.76 pp [+0.41, +1.11] (5/5) |
| stepwise composition change | fixed 0.5 | 5 | 64.86% (2.18) | +1.07 pp [+0.82, +1.32] (5/5) |
| stepwise composition change | entropy | 5 | 64.37% (1.75) | +1.56 pp [+1.18, +1.94] (5/5) |
| stepwise composition change | APFL η=0.01 | 5 | 64.25% (2.00) | +1.68 pp [+1.37, +1.98] (5/5) |
| stepwise composition change | APFL η=0.1 | 5 | 64.97% (1.97) | +0.96 pp [+0.52, +1.39] (5/5) |
| stepwise composition change | absonly | 5 | 64.92% (1.96) | +1.01 pp [+0.73, +1.29] (5/5) |
| client mobility | DriftGate | 3 | 59.23% (0.36) |  |
| client mobility | fixed 0.2 | 3 | 57.93% (0.45) | +1.30 pp [+0.94, +1.66] (3/3) |
| client mobility | fixed 0.4 (최고 고정값) | 3 | 58.58% (0.42) | +0.65 pp [+0.47, +0.83] (3/3) |
| client mobility | fixed 0.5 | 3 | 58.30% (0.39) | +0.93 pp [+0.84, +1.02] (3/3) |
| client mobility | fixed 0.6 | 3 | 57.95% (0.45) | +1.28 pp [+0.88, +1.69] (3/3) |
| client mobility | entropy | 3 | 57.21% (0.46) | +2.02 pp [+1.26, +2.77] (3/3) |
| client mobility | APFL η=0.01 | 3 | 58.00% (0.62) | +1.23 pp [+0.56, +1.89] (3/3) |
| client mobility | APFL η=0.1 | 3 | 59.58% (0.70) | -0.35 pp [-1.44, +0.74] (1/3) |
| client mobility | absonly | 3 | 57.93% (0.31) | +1.30 pp [+1.06, +1.53] (3/3) |

### 4.2 relonly와 absonly (표 4a, `tables/T4a_controller_relonly_vs_absonly.csv`)

| 설정 | DriftGate | absonly | DriftGate − absonly |
|---|---|---|---|
| stepwise composition change | 65.93% | 64.92% | +1.01 pp [+0.73, +1.29] (5/5) |
| client mobility | 59.23% | 57.93% | +1.30 pp [+1.06, +1.53] (3/3) |
| CIFAR-100 gradual | 36.09% | 34.89% | +1.20 pp [+1.07, +1.33] (3/3) |

### 4.3 구간별 정확도와 구간별 기여 (표 2a–2c, 그림 4)

구간은 평가 라운드로 나눴다. stepwise는 ρ = 0 초반 {1, 10, 20, 30}, ρ = 0.4 상승 {40, 50, 60}, ρ = 0.8 {70, 80, 90}, ρ = 0.4 하강 {100, 110, 120}, ρ = 0 후반 {130, 140, 150}이고, mobility는 ρ 변화 전 {1, …, 60}과 변화 후 {70, …, 120}이다. 표 2c는 구간마다 가장 좋은 고정값과 가장 나쁜 고정값, 그리고 DriftGate와의 paired 차이다.

| 설정 | 구간 | 최고 고정값 | 최저 고정값 | DriftGate | DriftGate − 최저 | DriftGate − 최고 |
|---|---|---|---|---|---|---|
| stepwise composition change | ρ=0 early | fixed 0.5 (57.93%) | fixed 0.2 (55.48%) | 58.23% | +2.74 pp [+1.82, +3.67] (5/5) | +0.30 pp [-0.09, +0.69] |
| stepwise composition change | ρ=0.4 rising | fixed 0.2 (62.58%) | fixed 0.5 (61.22%) | 61.60% | +0.38 pp [-0.20, +0.96] (4/5) | -0.98 pp [-1.57, -0.38] |
| stepwise composition change | ρ=0.8 | fixed 0.2 (61.40%) | fixed 0.5 (55.84%) | 59.76% | +3.92 pp [+2.76, +5.08] (5/5) | -1.64 pp [-2.88, -0.40] |
| stepwise composition change | ρ=0.4 falling | fixed 0.2 (68.02%) | fixed 0.5 (65.60%) | 66.92% | +1.32 pp [+0.28, +2.36] (5/5) | -1.11 pp [-1.93, -0.29] |
| stepwise composition change | ρ=0 late | fixed 0.5 (86.03%) | fixed 0.2 (79.08%) | 85.70% | +6.62 pp [+2.27, +10.97] (5/5) | -0.32 pp [-3.29, +2.64] |
| client mobility | ρ=0 (R1–60) | fixed 0.6 (62.92%) | fixed 0.2 (55.92%) | 60.88% | +4.96 pp [+2.97, +6.95] (3/3) | -2.05 pp [-3.01, -1.08] |
| client mobility | ρ=0.8 (R70–120) | fixed 0.2 (60.27%) | fixed 0.6 (52.14%) | 57.31% | +5.17 pp [+3.85, +6.49] (3/3) | -2.97 pp [-5.91, -0.02] |

DriftGate는 7개 구간 중 5개에서 그 구간의 최고 고정값보다 낮고, 이 차이의 CI는 0을 포함하지 않는다. 그러나 최고 고정값이 구간마다 다르므로, 한 고정값을 끝까지 쓰면 어떤 구간에서는 최저 수준이 된다. 표 2b는 integrated accuracy의 차이를 구간별 기여(구간 차이 × 구간의 평가 라운드 비율)로 나눈 것이며, 기여의 합은 전체 차이와 같다.

| 설정 | 비교 | 구간 1 | 구간 2 | 구간 3 | 구간 4 | 구간 5 | 합(= 전체 차이) |
|---|---|---|---|---|---|---|---|
| stepwise composition change | DriftGate − fixed 0.2 | +0.69 | -0.18 | -0.31 | -0.21 | +1.24 | +1.23 |
| stepwise composition change | DriftGate − fixed 0.3 | +0.48 | -0.18 | -0.11 | -0.16 | +0.79 | +0.83 |
| stepwise composition change | DriftGate − fixed 0.4 | +0.29 | -0.09 | +0.25 | -0.01 | +0.32 | +0.76 |
| stepwise composition change | DriftGate − fixed 0.5 | +0.08 | +0.07 | +0.73 | +0.25 | -0.06 | +1.07 |
| stepwise composition change | DriftGate − entropy | -0.10 | +0.18 | +1.25 | +0.58 | -0.35 | +1.56 |
| client mobility | DriftGate − fixed 0.2 | +2.67 | -1.37 |  |  |  | +1.30 |
| client mobility | DriftGate − fixed 0.4 | +0.94 | -0.29 |  |  |  | +0.65 |
| client mobility | DriftGate − fixed 0.5 | +0.03 | +0.90 |  |  |  | +0.93 |
| client mobility | DriftGate − fixed 0.6 | -1.10 | +2.39 |  |  |  | +1.28 |
| client mobility | DriftGate − entropy | -0.65 | +2.67 |  |  |  | +2.02 |

구간 1–5는 stepwise의 다섯 구간을 순서대로, mobility에서는 구간 1이 ρ 변화 전, 구간 2가 변화 후를 뜻한다. 단위는 pp다.

Main/OOP/OOR별 정확도(`tables/T2a_segment_accuracy.csv`)를 보면, ρ = 0.8 구간에서 작은 고정 λ는 OOP와 OOR 정확도가 높고 Main 정확도가 낮다. 아래는 그 구간의 값이다.

| 설정 | 방법 | 전체 | Main | OOP | OOR |
|---|---|---|---|---|---|
| stepwise composition change, ρ=0.8 | DriftGate | 59.76% | 74.26% | 52.68% | 22.79% |
| stepwise composition change, ρ=0.8 | entropy | 53.09% | 81.55% | 30.90% | 8.30% |
| stepwise composition change, ρ=0.8 | fixed 0.2 | 61.40% | 75.02% | 55.80% | 23.20% |
| stepwise composition change, ρ=0.8 | fixed 0.3 | 60.32% | 76.63% | 51.80% | 20.62% |
| stepwise composition change, ρ=0.8 | fixed 0.4 | 58.45% | 78.20% | 46.06% | 17.28% |
| stepwise composition change, ρ=0.8 | fixed 0.5 | 55.84% | 79.65% | 38.73% | 13.50% |
| client mobility, ρ=0.8 (R70–120) | DriftGate | 57.31% | 75.22% | 43.26% | 28.51% |
| client mobility, ρ=0.8 (R70–120) | entropy | 51.53% | 81.32% | 24.84% | 15.43% |
| client mobility, ρ=0.8 (R70–120) | fixed 0.2 | 60.27% | 70.88% | 53.49% | 37.69% |
| client mobility, ρ=0.8 (R70–120) | fixed 0.4 | 57.93% | 76.00% | 43.66% | 29.19% |
| client mobility, ρ=0.8 (R70–120) | fixed 0.5 | 55.35% | 78.62% | 35.71% | 22.91% |
| client mobility, ρ=0.8 (R70–120) | fixed 0.6 | 52.14% | 81.28% | 26.33% | 15.78% |

### 4.4 마지막 평가 라운드 정확도 (표 10, `tables/T10_last_round_accuracy.csv`)

절대 정확도를 확인하는 표다. integrated accuracy와 순위가 다를 수 있다.

| 설정 | 방법 | 마지막 평가 라운드 | seed | acc_total (SD) |
|---|---|---|---|---|
| stepwise composition change | DriftGate | 150 | 5 | 87.98% (2.77) |
| stepwise composition change | fixed 0.2 | 150 | 5 | 79.68% (1.57) |
| stepwise composition change | fixed 0.3 | 150 | 5 | 82.29% (1.97) |
| stepwise composition change | fixed 0.4 | 150 | 5 | 84.81% (2.35) |
| stepwise composition change | fixed 0.5 | 150 | 5 | 86.94% (2.79) |
| stepwise composition change | entropy | 150 | 5 | 88.15% (2.99) |
| stepwise composition change | APFL η=0.01 | 150 | 5 | 87.32% (1.99) |
| stepwise composition change | APFL η=0.1 | 150 | 5 | 87.32% (1.46) |
| stepwise composition change | absonly | 150 | 5 | 88.58% (2.43) |
| client mobility | DriftGate | 120 | 3 | 59.72% (1.28) |
| client mobility | fixed 0.2 | 120 | 3 | 62.23% (0.71) |
| client mobility | fixed 0.4 | 120 | 3 | 59.89% (0.95) |
| client mobility | fixed 0.5 | 120 | 3 | 56.92% (0.83) |
| client mobility | fixed 0.6 | 120 | 3 | 53.43% (0.95) |
| client mobility | entropy | 120 | 3 | 52.77% (0.95) |
| client mobility | APFL η=0.01 | 120 | 3 | 52.36% (1.52) |
| client mobility | APFL η=0.1 | 120 | 3 | 53.24% (2.30) |
| client mobility | absonly | 120 | 3 | 55.91% (1.08) |
| CIFAR-10 gradual | DriftGate | 150 | 3 | 63.17% (3.17) |
| CIFAR-10 gradual | fixed 0.15 | 150 | 3 | 65.70% (2.51) |
| CIFAR-10 gradual | fixed 0.2 | 150 | 3 | 65.42% (2.69) |
| CIFAR-10 gradual | fixed 0.4 | 150 | 3 | 62.03% (2.80) |
| CIFAR-100 gradual | DriftGate | 150 | 3 | 46.68% (1.10) |
| CIFAR-100 gradual | fixed 0.15 | 150 | 3 | 47.20% (1.37) |
| CIFAR-100 gradual | fixed 0.2 | 150 | 3 | 46.97% (1.26) |
| CIFAR-100 gradual | fixed 0.4 | 150 | 3 | 43.99% (0.93) |
| Tiny-ImageNet | DriftGate | 100 | 3 | 31.21% (0.70) |
| Tiny-ImageNet | fixed 0.15 | 100 | 3 | 32.29% (0.42) |
| Tiny-ImageNet | fixed 0.2 | 100 | 3 | 32.21% (0.47) |
| Tiny-ImageNet | fixed 0.4 | 100 | 3 | 31.13% (0.63) |
| SVHN temporal | DriftGate | 150 | 5 | 83.70% (2.72) |
| SVHN temporal | fixed 0.15 | 150 | 5 | 87.26% (1.04) |
| SVHN temporal | fixed 0.2 | 150 | 5 | 87.05% (1.04) |
| SVHN temporal | fixed 0.4 | 150 | 5 | 84.74% (1.13) |
| SVHN temporal | entropy | 150 | 5 | 76.96% (1.43) |
| ResNet-18 middle split (stepwise) | DriftGate | 150 | 3 | 89.13% (2.26) |
| ResNet-18 middle split (stepwise) | fixed 0.2 | 150 | 3 | 74.02% (2.42) |
| ResNet-18 middle split (stepwise) | fixed 0.4 | 150 | 3 | 81.92% (0.39) |

마지막 평가 라운드에서 DriftGate가 가장 좋은 고정값보다 높은 설정은 stepwise composition change(87.98% 대 fixed 0.5 86.94%), ResNet-18 middle split(89.13% 대 fixed 0.4 81.92%)이다. 가장 좋은 고정값보다 낮은 설정은 client mobility(59.72% 대 fixed 0.2 62.23%), CIFAR-10 gradual(63.17% 대 fixed 0.15 65.70%), CIFAR-100 gradual(46.68% 대 fixed 0.15 47.20%), Tiny-ImageNet(31.21% 대 fixed 0.15 32.29%), SVHN temporal(83.70% 대 fixed 0.15 87.26%)이다.
stepwise의 마지막 라운드에서는 entropy(88.15%), absonly(88.58%)가 DriftGate(87.98%)보다 높다.

### 4.5 λ 궤적 (그림 3, `tables/lambda_trajectories.csv`)

seed 안에서 cluster 평균을 먼저 내고 seed 평균을 냈다. stepwise에서 DriftGate의 λ는 ρ = 0.8 구간(R61–90)에 평균 0.229까지 내려갔다가 ρ가 0으로 돌아온 뒤 0.459로 올라간다. mobility에서는 ρ 변화 전 0.549, 뒤 0.380이다. entropy controller의 λ는 R26 이후 두 설정 모두 0.58–0.64 사이에 머문다.

### 4.6 신호 수준과 passive 신호 지표 (표 6a–6c, 그림 1)

평가용 stepwise(150 라운드)의 고정 λ = 0.4 실행(`t1_A_fx40`)은 신호를 기록하지 않았다. 그래서 신호를 매 라운드 기록한 passive 실행을 썼다. 이 실행은 고정 λ = 0.4, Λ = 0.5로 학습하며 controller가 신호에 반응하지 않는다. 단계 변화는 ρ를 20 라운드마다 바꾸는 100 라운드 변형이고(R21, R41, R61, R81에 변화), late abrupt change는 R51에 ρ가 0에서 0.8로 오른다. seed는 3개이고, controller용과 평가용 표본을 나누지 않은 실행이다.

| 설정 | 구간 | 라운드 | TV (SD) | H/ln C (SD) |
|---|---|---|---|---|
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | ρ=0 | R16–R20 | 0.2440 (0.0146) | 0.7855 (0.0359) |
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | ρ=0.4 | R21–R40 | 0.2746 (0.0205) | 0.7645 (0.0378) |
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | ρ=0.8 | R41–R60 | 0.3030 (0.0271) | 0.7305 (0.0343) |
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | ρ=0.4 | R61–R80 | 0.2912 (0.0268) | 0.6692 (0.0319) |
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | ρ=0 | R81–R100 | 0.2482 (0.0244) | 0.5897 (0.0350) |
| late abrupt change (ρ 0→0.8 at R51) | ρ=0 | R16–R50 | 0.2490 (0.0175) | 0.7311 (0.0374) |
| late abrupt change (ρ 0→0.8 at R51) | ρ=0.8 | R51–R100 | 0.3131 (0.0289) | 0.6834 (0.0300) |

| 신호 | R46–50 평균 | R51–55 평균 | shift 변화 (R51–55 − R46–50) | shift 전 35 라운드 변화 (R46–50 − R16–20) | 전체 변화 (R96–100 − R16–20) |
|---|---|---|---|---|---|
| TV | 0.2521 | 0.3053 | +0.0532 | +0.0078 | +0.0740 |
| H/lnC | 0.6848 | 0.7234 | +0.0386 | -0.0998 | -0.1355 |

| 설정 | 신호 | ρ와의 Spearman (SD) | raw-direction AUROC | direction-free AUROC |
|---|---|---|---|---|
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | TV | +0.907 (0.004) | 0.978 | 0.978 |
| 100-round stepwise variant (ρ changes at R21, R41, R61, R81) | entropy | +0.437 (0.008) | 0.643 | 0.643 |
| late abrupt change (ρ 0→0.8 at R51) | TV | +0.853 (0.000) | 1.000 | 1.000 |
| late abrupt change (ρ 0→0.8 at R51) | entropy | -0.578 (0.078) | 0.161 | 0.839 |

AUROC는 ρ ≥ 0.5인 라운드를 양성으로 두고 처음 15 라운드를 뺀 값이다. raw-direction AUROC는 신호가 클수록 ρ가 크다고 보는 방향으로 계산했다. entropy도 shift에서 조금 오르므로 traffic에 반응하지 않는 것은 아니다. 다만 그 폭이 학습 진행에 따른 감소보다 작아 drift 구간의 entropy가 drift 이전보다 낮다.

### 4.7 역할 실험 (표 7, 그림 5)

기존 3-seed 결과를 그대로 옮겼다. 이 실행들은 예전 controller(absolute branch 포함)와 controller·평가 공용 표본으로 돌았다. 표의 값은 정확도가 아니라 TV와 ρ의 Spearman 상관이다(처음 15 라운드 제외).

| 조건 | TV–ρ Spearman 평균 (SD) | seed별 값 |
|---|---|---|
| standard | +0.866 (0.097) | +0.754 +0.927 +0.916 |
| same-role exits | -0.213 (0.028) | -0.226 -0.233 -0.182 |
| same-role independent init | +0.151 (0.330) | +0.404 -0.222 +0.273 |
| weakened server | +0.019 (0.412) | -0.457 +0.258 +0.255 |

### 4.8 재현 설정표 (표 8, `tables/T8_reproduction_settings.csv`)

| 분류 | 항목 | 값 | 출처(파일:줄) |
|---|---|---|---|
| 학습 | optimizer | SGD (client block과 server block 각각) | configs/base_v3.yaml:19; train/trainer.py:86,90 |
| 학습 | learning rate와 schedule | 0.01, 전 라운드 고정 (scheduler 없음) | configs/base_v3.yaml:17; train/trainer.py:72-141 |
| 학습 | momentum, weight decay | 0.0, 1e-4 | configs/base_v3.yaml:18,20 |
| 학습 | local epoch | 3 epoch / 라운드 | configs/base_v3.yaml:15; train/trainer.py:94 |
| 학습 | batch 크기 | 32 | configs/base_v3.yaml:16 |
| 학습 | 참여율 | 1.0 (모든 client가 매 라운드 참여) | src/runner.py:284,290 |
| 학습 | 손실 가중치 γ | 0.5 (client exit와 server exit의 cross-entropy를 반씩) | configs/base_v3.yaml:23 |
| 학습 | client별 학습 표본 수 (seed 0) CIFAR-10 | min 620 / median 908 / max 2120 (Main classes per client: [2]) | data/partition.py:50-104 (nd1_partition 재구성) |
| 학습 | client별 학습 표본 수 (seed 0) CIFAR-100 | min 800 / median 945 / max 1419 (Main classes per client: [20]) | 같은 함수, 실제 학습 label 사용 |
| 학습 | client별 학습 표본 수 (seed 0) Tiny-ImageNet | min 1537 / median 1919 / max 2729 (Main classes per client: [40]) | 같은 함수, 실제 학습 label 사용 |
| 학습 | client별 학습 표본 수 (seed 0) SVHN | min 703 / median 1405 / max 2898 (Main classes per client: [2]) | 같은 함수, 실제 학습 label 사용 |
| 학습 | client별 Main class 비율 | 전체 class의 20% (CIFAR-10·SVHN 2개, CIFAR-100 20개, Tiny 40개). datasets_ext.py의 META 값(5, 10)은 config 값이 있어 쓰이지 않음 | configs/base_v3.yaml:31; src/runner.py:121 (setdefault) |
| 모델 | CNN client block | conv 3→32→64→64→128 (BN, conv2·conv4 뒤 max-pool) + 보조 exit(conv 128 + GAP + fc). 279,882 parameters | models/architectures.py:14-52 |
| 모델 | CNN split 위치 | conv4 + pool 뒤. 32×32 입력에서 feature (128, 8, 8) (Tiny 64×64 입력에서는 (128, 16, 16)) | models/architectures.py:37-53 |
| 모델 | CNN server block | conv 128→256 + BN, flatten, fc 256→128→C. 1,378,698 parameters | models/architectures.py:54-79 |
| 모델 | ResNet-18 middle split | client = stem(conv 3→64) + stage 1(64) + stage 2(128), feature (128, 16, 16), 676,682 parameters. server = stage 3(256) + stage 4(512) + fc, 10,498,570 parameters | src/models_ext.py:93-130 (RESNET_SPLITS middle=2) |
| 모델 | ResNet-18 실행의 client 수 | 16명 (기존 ResNet-18 분할 실험과 같음. static topology에서 4명이 두 cluster에 속함) | provenance: phaseF_arch res_middle_* (--num_clients 16) |
| 데이터 | 전처리 CIFAR-10 | 학습: RandomCrop(32, padding 4) + RandomHorizontalFlip + Normalize. 평가·probe: Normalize만 | data/partition.py:154-167 |
| 데이터 | 전처리 CIFAR-100, Tiny-ImageNet, SVHN | ToTensor + Normalize만 (augmentation 없음). Tiny는 64×64 그대로 | src/datasets_ext.py:101-141 |
| 데이터 | OOP와 OOR의 비율 | Main 표본 수를 기준으로 OOP 표본 = ρ배, OOR 표본 = 0.3ρ배 (class별로 나눠 채움). 즉 OOP:OOR = 1:0.3 | data/partition.py:107-143; configs/base_v3.yaml:37 |
| traffic | stepwise composition change의 ρ | 5등분 계단 0→0.4→0.8→0.4→0. 150 라운드에서 R31, R61, R91, R121에 바뀜 | src/schedules.py:31-32 |
| traffic | gradual의 ρ 식 | ρ(r) = 0.8 / (1 + exp(−(f − 0.5)/0.08)), f = (r − 1)/T | src/schedules.py:58-59 |
| traffic | client mobility의 ρ | abrupt 일정. 120 라운드에서 R61에 0→0.8 | src/schedules.py:42-43 |
| traffic | client mobility의 membership 변화 | 모든 client가 매 라운드 Gauss-Markov 모델로 이동(α=0.9, 속도 12, dt=10 s, 1000×1000 지도, seed 42). 5 라운드마다 각 client의 가장 가까운 edge 2개를 다시 구하고, 그 집합이 바뀐 client만 재배치. R5에 50명 전원이 static topology에서 이동 topology로 옮겨가고, 이후 5 라운드마다 1–10명이 바뀜(로그 기준) | src/runner.py:206-213, 343-355; network/mobility.py:17-63 |
| traffic | late abrupt change (passive 신호 실행) | abrupt 일정을 100 라운드로 실행. R51에 ρ 0→0.8 | src/schedules.py:42-43; rec_abrupt 실행 |
| cluster | cluster 구성 | edge server 5개, client 50명. cluster마다 10명 중 뒤쪽 5명이 다음 cluster에도 속해 20명이 두 cluster에 속함 | configs/base_v3.yaml:9-11; data/partition.py:17-47 |
| cluster | edge 사이의 scalar 교환 | 점수 평균: 이웃 edge(선형 0-1-2-3-4)와 1단계 평균. spatial 비교: 같은 라운드의 모든 cluster TV의 중앙값·MAD를 사용 | data/partition.py:189-199; src/controllers/self_calibrating.py:51-61, 157-164 |
| 초기 구간 | burn-in과 warm-up | burn-in 10 라운드 + warm-up 15 라운드 = 처음 25 라운드. 이 동안 λ=0.425, Λ=0.55 (범위의 중간값) | src/controllers/self_calibrating.py:91,132-136,181-182; 실행 flag --burn_in 10; src/runner.py:246 (warmup 15) |
| 평가 | 평가 라운드와 최종 예측 | R1과 10 라운드마다 (150 R: 16회, 120 R: 13회, 100 R: 11회). client exit의 entropy가 0.8을 넘으면 server exit 예측을 씀 | configs/base_v3.yaml:34-35; src/runner.py:564; eval/evaluator.py |

### 4.9 통신량 (표 9, `tables/T9_communication.csv`)

| 분류 | 항목 | bytes | 근거 | 출처 |
|---|---|---|---|---|
| 기본 split | CNN split activation (표본 1개, fp32) | 32,768 | feature (128, 8, 8) × 4 B | 계산 |
| 기본 split | ResNet-18 middle split activation (표본 1개, fp32) | 131,072 | feature (128, 16, 16) × 4 B | 계산 |
| 기본 split | client block / server block 크기 (CNN, fp32) | 1,119,528 / 5,514,792 | 279,882 / 1,378,698 parameters × 4 B | 계산 |
| 기본 split | 라운드당 모델 교환량 (두 cluster에 속한 client 1명, 한 방향) | 12,149,112 | client block 1개 + server block 2개. tables/communication_accounting.csv의 12,149,112 B와 정확히 같음 | journal_expansion/tables/communication_accounting.csv |
| 기본 split | server-block 사본을 맞추는 downlink (두 cluster client, 라운드당) | 11,029,584 | cluster 평균 server block 2개를 받음(한 cluster client는 5,514,792 B). 12,149,112 B는 한 방향의 크기이므로 이 downlink는 따로 더해야 함 | train/trainer.py:276 (set_server_state, cell-based 경로) |
| 기본 split | client block downlink (cluster 평균과 섞을 모델, 라운드당) | 1,119,528 | cluster 평균 client block 1개 | train/trainer.py:259-273 (mix_state_dicts, 271줄) |
| DriftGate 추가 | client → edge TV scalar | 4 | client가 속한 edge마다 라운드당 4 B (두 cluster client는 8 B) | src/signals/library.py:215-223 (aggregate_per_es) |
| DriftGate 추가 | edge ↔ 이웃 edge 점수 평균 | 4·deg(e) | 선형 graph라 deg(e)는 1 또는 2, 즉 edge당 4 B 또는 8 B | self_calibrating.py:63-71,164 |
| DriftGate 추가 | spatial 비교를 위한 cluster TV 공유 | 4·(Z−1) | spatial 점수는 같은 라운드의 모든 cluster TV를 쓰므로 edge마다 다른 Z−1개 cluster의 TV가 필요함. Z=5면 16 B | self_calibrating.py:51-61,157 |
| 추정치 | edge에서 server exit를 계산할 때 probe 업로드 (probe 64개) | 2,099,712 | 추정치. activation 64 × 32,768 B + client 확률 벡터 64 × C × 4 B (C=10). 확률 벡터 대신 edge가 server 확률 벡터 64 × 40 B를 내려보내도 같은 크기 | 계산 (현재 구현은 client가 가진 server-block 사본으로 계산하므로 이 전송이 없음) |

12,149,112 B는 client block 1개와 server block 2개의 크기를 더한 값과 정확히 같다. 이 숫자는 한 방향의 크기이므로, client가 가진 server-block 사본을 매 라운드 cluster 평균으로 맞추는 downlink(두 cluster에 속한 client는 11,029,584 B)는 포함하지 않는다. 마지막 행은 edge에서 server exit를 계산한다고 가정한 추정치이며, 현재 구현은 client가 가진 server-block 사본으로 TV를 계산하므로 이 전송이 없다.

## 5. 그림 목록과 영문 caption

모든 그림은 vector PDF와 300 dpi PNG로 `figures/`에 있다. 그림 안의 이름은 논문 용어(stepwise composition change, client mobility, late abrupt change, DriftGate)로 바꿨다.

| 파일 | 크기 | 내용 |
|---|---|---|
| `fig1_signal_dynamics.pdf` | double column | 신호 변화: 위 ρ, 아래 TV와 H/ln C. (a) 단계 변화, (b) late abrupt change |
| `fig2_system_overview.pdf` | double column | 시스템 개요. absolute branch 없음, controller는 cluster마다 계산 |
| `fig3_lambda_trajectories.pdf` | double column | λ 궤적: 위 ρ, 아래 DriftGate와 entropy의 λ, 고정 λ = 0.4와 0.2 점선 |
| `fig4_segment_accuracy.pdf` | double column | 구간별 정확도: stepwise 5구간, mobility 2구간 |
| `fig5_role_experiment.pdf` | single column | 역할 실험: TV와 ρ의 Spearman 상관 |

**Figure 1** (`fig1_signal_dynamics.pdf`, double column).
Signal values on one model trained with a fixed weight (λ = 0.4, Λ = 0.5, 3 seeds, bands show the seed SD).
The top panels show the share ρ of non-Main traffic. The bottom panels show the client-server TV (blue) and
the predictive entropy divided by ln C (orange). (a) A 100-round version of the stepwise composition change,
with ρ changing at rounds 21, 41, 61 and 81. TV rises from 0.244 at ρ = 0 to 0.303 at ρ = 0.8 and returns to
0.248 when ρ goes back to 0. Entropy falls in every segment, from 0.786 to 0.590, whatever ρ does.
(b) Late abrupt change, with ρ going from 0 to 0.8 at round 51. At the shift, TV rises by 0.053 and entropy
rises by 0.039 (mean of rounds 51 to 55 minus rounds 46 to 50). Over the 35 rounds before the shift, entropy
falls by 0.100 while TV changes by 0.008. Shaded rounds are the first 15 rounds, which the signal statistics skip.

**Figure 2** (`fig2_system_overview.pdf`, double column).
Overview of DriftGate. (a) Each client runs its client block and two exits on the same request. The client exit
gives p_c and the server block with its exit gives p_s. Their total variation distance, averaged over 64 recent
unlabeled requests, is the only signal. (b) Each serving cluster averages the TV values of its clients and runs its
own controller. The controller compares the cluster signal with its own recent values (temporal comparison) and
with the other clusters in the same round (spatial comparison), and sets the cluster weights λ_z and Λ_z. Edges
exchange only scalars. No labels and no ρ enter any controller. Two clusters are drawn as an example.

**Figure 3** (`fig3_lambda_trajectories.pdf`, double column).
Personalization weight λ chosen by DriftGate and by the same controller fed with entropy, averaged over clusters
and then over seeds (bands show the seed SD). (a) Stepwise composition change, 5 seeds. (b) Client mobility,
3 seeds, where ρ goes from 0 to 0.8 at round 61 while clients keep moving between clusters. Dotted lines mark
the fixed weights λ = 0.4 and λ = 0.2. The first 25 rounds (burn-in and warm-up, shaded) use λ = 0.425.
On the stepwise schedule DriftGate lowers λ to 0.23 on average while ρ = 0.8 and raises it again when ρ returns
to 0. Under client mobility it uses 0.55 before ρ changes (rounds 26 to 60) and 0.38 after (rounds 61 to 120). The entropy controller stays between
0.58 and 0.64 after warm-up in both settings.

**Figure 4** (`fig4_segment_accuracy.pdf`, double column).
Accuracy in each segment of the evaluation rounds (mean over the rounds of a segment, error bars show the seed
SD). (a) Stepwise composition change, 5 seeds. (b) Client mobility, 3 seeds. The best fixed λ changes with the
segment. On the stepwise schedule λ = 0.5 is best when ρ = 0 and λ = 0.2 is best when ρ > 0. Under client
mobility λ = 0.6 is best before ρ changes and λ = 0.2 is best after. DriftGate stays above the worst fixed λ in
every segment.

**Figure 5** (`fig5_role_experiment.pdf`, single column).
Spearman correlation between TV and ρ when the two exits lose their different roles (stepwise composition
change, 3 seeds, open circles show each seed and filled circles show the mean with the SD). With the standard
roles the correlation is 0.87. When both exits play the same role, start from independent weights, or the server
sees less non-Main data, the correlation drops to between -0.21 and 0.15. These runs used the earlier controller
and a shared probe and evaluation pool.

## 6. 불일치 기록

1. **Round 4 B3 run의 정지**: B3 λ = 0.15 run 12개가 Round 4 중단 때 일시 정지된 채 남아 있었다. GPU 1에 있던 6개를 포함해 모두 종료했고, 보고에 쓰는 run은 GPU 0에서 처음부터 다시 돌렸다. 종료한 run 중 CIFAR-100 spatial 3개는 보고에 넣지 않으므로 다시 돌리지 않았다.
2. **controller 설명과 코드의 순서**: 지시서는 temporal 점수와 spatial 점수 중 큰 값을 EMA로 평활한다고 적었다. 코드는 두 점수를 각각 EMA(α = 0.3)로 평활한 뒤 큰 값을 고르고, 그 값에 이웃 edge와의 1단계 평균을 적용한 다음 λ로 바꾼다(`self_calibrating.py:150-164`). 모든 DriftGate 수치는 이 코드로 얻었다.
3. **같은 설정의 중복 실행**: mobility 고정 λ = 0.5가 Round 1(`t1_mob_fx50`, GPU 1)과 Round 4(`b3_mob_fx50`, GPU 0)에서 같은 flag로 두 번 실행됐다. seed별 integrated accuracy 차이는 0.015–0.039 pp(Round 4 − Round 1: -0.015, -0.039, -0.024)이고, 첫 라운드 손실부터 소수 다섯째 자리에서 달라진다. GPU 연산의 비결정성 때문으로 보이며, 이 크기는 재현 오차의 기준으로 쓸 수 있다. 결과를 보기 전에 정한 규칙대로 Round 1 실행을 썼다.
4. **지시서의 메시지 목록**: 지시서는 DriftGate의 추가 메시지를 client당 4 B, edge당 4·deg(e) B로 적었다. 그러나 spatial 비교는 같은 라운드의 모든 cluster TV의 중앙값과 MAD를 쓰므로(`self_calibrating.py:51-61,157`), edge마다 다른 cluster의 TV 4·(Z − 1) B가 더 필요하다(Z = 5이면 16 B). 표 9에 따로 적었다.
5. **Round 4의 probe 업로드 추정**: Round 4의 통신 표(a6_comm.csv)는 edge에서 server exit를 계산할 때의 업로드를 32 KiB × n + 40 B로 적었다. TV를 계산하려면 표본마다 확률 벡터가 필요하므로 n × (32,768 + 40) B가 맞다. probe 64개이면 2,099,712 B다.
6. **commit hash**: 저장소 HEAD `e82b96b`에는 `adaptive_splitomc_tmc/` 디렉터리가 들어 있지 않다(추적되지 않음). 실행 코드는 sha256으로 특정했다(`precheck/code_identity.txt`).
7. **ResNet-18의 client 수**: E2는 기존 ResNet-18 분할 실험과 같이 client 16명으로 돌렸다. CNN 실험의 기본값(50명)과 다르다.
8. **신호 수준 실행**: 평가용 stepwise의 고정 λ = 0.4 실행에는 신호 기록이 없어, 100 라운드 변형(ρ 변화 R21, R41, R61, R81)의 passive 실행을 썼다. 이 실행은 controller용과 평가용 표본을 나누지 않았다.
9. **역할 실험의 실행 조건**: 역할 실험 run은 예전 controller(absolute branch 포함)와 공용 표본으로 돌았다. 지시서가 이 결과를 옮기도록 했으므로 TV–ρ 상관만 옮겼고, 정확도는 넣지 않았다.
10. **λ_max 변형의 warm-up 값**: controller는 warm-up 동안 λ 범위의 중간값을 쓰므로, λ_max를 바꾸면 warm-up λ도 0.425에서 0.375(λ_max 0.60) 또는 0.475(λ_max 0.80)로 바뀐다.
11. **Main class 수의 메타데이터**: `src/datasets_ext.py`의 META에는 CIFAR-100 5개, Tiny-ImageNet 10개가 적혀 있지만 쓰이지 않는다. runner가 config의 classes_per_client_frac = 0.2를 먼저 쓰므로(`src/runner.py:121`) 실제 Main class는 전체의 20%다.
12. **E3의 비교 기준**: 지시서대로 E3의 비교 기준은 같은 seed(0–2)의 B1 relonly 실행이다. 따라서 stepwise의 기준 정확도(66.07%)는 5-seed 평균(65.93%)과 다르다.

## 7. 새 run의 run_manifest (`tables/run_manifest.csv`)

Round 5에서 실행한 run은 64개이고, 64개가 모두 지정된 라운드까지 끝났으며 64개 모두 overlap이 0이다. 모든 run은 물리 GPU 0에서 실행했다.

| run_id | run 이름 | 실험 | arm | 설정 | seed | 완료 | 라운드 | 평가 횟수 | overlap |
|---|---|---|---|---|---|---|---|---|---|
| fixed_20260928_121705_9a2ca6 | b3_c10gsig_fx15_s0 | B3 | fixed 0.15 | CIFAR-10 gradual | 0 | yes | 150 | 16 | 0 |
| fixed_20260928_122550_2d16d7 | b3_c10gsig_fx15_s1 | B3 | fixed 0.15 | CIFAR-10 gradual | 1 | yes | 150 | 16 | 0 |
| fixed_20260930_050048_162cf3 | b3_c100gsig_fx15_s1 | B3 | fixed 0.15 | CIFAR-100 gradual | 1 | yes | 150 | 16 | 0 |
| fixed_20260930_050353_e7ccdf | b3_c100gsig_fx15_s2 | B3 | fixed 0.15 | CIFAR-100 gradual | 2 | yes | 150 | 16 | 0 |
| fixed_20260928_052530_8987ae | b3_svhn_fx15_s2 | B3 | fixed 0.15 | SVHN temporal | 2 | yes | 150 | 16 | 0 |
| fixed_20260928_052530_8b2904 | b3_svhn_fx15_s3 | B3 | fixed 0.15 | SVHN temporal | 3 | yes | 150 | 16 | 0 |
| fixed_20260928_052530_7b949b | b3_svhn_fx15_s4 | B3 | fixed 0.15 | SVHN temporal | 4 | yes | 150 | 16 | 0 |
| fixed_20260928_052530_ad173d | b3_tiny_fx15_s0 | B3 | fixed 0.15 | Tiny-ImageNet | 0 | yes | 100 | 11 | 0 |
| fixed_20260928_052530_bd27eb | b3_tiny_fx15_s1 | B3 | fixed 0.15 | Tiny-ImageNet | 1 | yes | 100 | 11 | 0 |
| fixed_20260928_162845_f631a0 | b3_tiny_fx15_s2 | B3 | fixed 0.15 | Tiny-ImageNet | 2 | yes | 100 | 11 | 0 |
| apfl_20260929_190723_666657 | b4_apfl_eta001_mob_s0 | B4 | APFL η=0.01 | client mobility | 0 | yes | 120 | 13 | 0 |
| apfl_20260929_201143_7fec73 | b4_apfl_eta001_mob_s1 | B4 | APFL η=0.01 | client mobility | 1 | yes | 120 | 13 | 0 |
| apfl_20260929_210443_e031d9 | b4_apfl_eta001_mob_s2 | B4 | APFL η=0.01 | client mobility | 2 | yes | 120 | 13 | 0 |
| apfl_20260929_215047_3a6601 | b4_apfl_eta010_mob_s0 | B4 | APFL η=0.1 | client mobility | 0 | yes | 120 | 13 | 0 |
| apfl_20260929_215047_79ad2d | b4_apfl_eta010_mob_s1 | B4 | APFL η=0.1 | client mobility | 1 | yes | 120 | 13 | 0 |
| apfl_20260929_215047_45f105 | b4_apfl_eta010_mob_s2 | B4 | APFL η=0.1 | client mobility | 2 | yes | 120 | 13 | 0 |
| apfl_20260928_184343_61ccec | b4_apfl_eta001_A_s0 | B4 | APFL η=0.01 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| apfl_20260928_185046_ba9139 | b4_apfl_eta001_A_s1 | B4 | APFL η=0.01 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| apfl_20260928_213439_bc3abd | b4_apfl_eta001_A_s2 | B4 | APFL η=0.01 | stepwise composition change | 2 | yes | 150 | 16 | 0 |
| apfl_20260928_214136_8c1b5d | b4_apfl_eta001_A_s3 | B4 | APFL η=0.01 | stepwise composition change | 3 | yes | 150 | 16 | 0 |
| apfl_20260928_222156_8646ee | b4_apfl_eta001_A_s4 | B4 | APFL η=0.01 | stepwise composition change | 4 | yes | 150 | 16 | 0 |
| apfl_20260928_232729_fe534f | b4_apfl_eta010_A_s0 | B4 | APFL η=0.1 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| apfl_20260928_233414_b4a77e | b4_apfl_eta010_A_s1 | B4 | APFL η=0.1 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| apfl_20260929_015853_8ec263 | b4_apfl_eta010_A_s2 | B4 | APFL η=0.1 | stepwise composition change | 2 | yes | 150 | 16 | 0 |
| apfl_20260929_023025_d4b43e | b4_apfl_eta010_A_s3 | B4 | APFL η=0.1 | stepwise composition change | 3 | yes | 150 | 16 | 0 |
| apfl_20260929_024012_330b9f | b4_apfl_eta010_A_s4 | B4 | APFL η=0.1 | stepwise composition change | 4 | yes | 150 | 16 | 0 |
| selfcal_20260928_164815_bb5ced | e1_relonly_c10gsig_s0 | E1 | DriftGate (relonly) | CIFAR-10 gradual | 0 | yes | 150 | 16 | 0 |
| selfcal_20260928_170108_048906 | e1_relonly_c10gsig_s1 | E1 | DriftGate (relonly) | CIFAR-10 gradual | 1 | yes | 150 | 16 | 0 |
| selfcal_20260928_180624_f11bc7 | e1_relonly_c10gsig_s2 | E1 | DriftGate (relonly) | CIFAR-10 gradual | 2 | yes | 150 | 16 | 0 |
| selfcal_20260928_052530_c3b535 | e1_relonly_svhn_s0 | E1 | DriftGate (relonly) | SVHN temporal | 0 | yes | 150 | 16 | 0 |
| selfcal_20260928_052530_8d44a7 | e1_relonly_svhn_s1 | E1 | DriftGate (relonly) | SVHN temporal | 1 | yes | 150 | 16 | 0 |
| selfcal_20260928_113813_75083f | e1_relonly_svhn_s2 | E1 | DriftGate (relonly) | SVHN temporal | 2 | yes | 150 | 16 | 0 |
| selfcal_20260928_115507_502a01 | e1_relonly_svhn_s3 | E1 | DriftGate (relonly) | SVHN temporal | 3 | yes | 150 | 16 | 0 |
| selfcal_20260928_121456_07c99d | e1_relonly_svhn_s4 | E1 | DriftGate (relonly) | SVHN temporal | 4 | yes | 150 | 16 | 0 |
| selfcal_20260928_052530_ef5ff9 | e1_relonly_tiny_s0 | E1 | DriftGate (relonly) | Tiny-ImageNet | 0 | yes | 100 | 11 | 0 |
| selfcal_20260928_162131_2ff7aa | e1_relonly_tiny_s1 | E1 | DriftGate (relonly) | Tiny-ImageNet | 1 | yes | 100 | 11 | 0 |
| selfcal_20260928_162907_1ec9e6 | e1_relonly_tiny_s2 | E1 | DriftGate (relonly) | Tiny-ImageNet | 2 | yes | 100 | 11 | 0 |
| selfcal_20260929_031630_9ad14c | e2_res_relonly_A_s0 | E2 | DriftGate (relonly) | ResNet-18 middle split, stepwise composition change | 0 | yes | 150 | 16 | 0 |
| selfcal_20260929_123101_8aef42 | e2_res_relonly_A_s1 | E2 | DriftGate (relonly) | ResNet-18 middle split, stepwise composition change | 1 | yes | 150 | 16 | 0 |
| selfcal_20260929_213738_d54fd0 | e2_res_relonly_A_s2 | E2 | DriftGate (relonly) | ResNet-18 middle split, stepwise composition change | 2 | yes | 150 | 16 | 0 |
| fixed_20260929_031932_59b16e | e2_res_fx20_A_s0 | E2 | fixed 0.2 | ResNet-18 middle split, stepwise composition change | 0 | yes | 150 | 16 | 0 |
| fixed_20260929_123302_9986da | e2_res_fx20_A_s1 | E2 | fixed 0.2 | ResNet-18 middle split, stepwise composition change | 1 | yes | 150 | 16 | 0 |
| fixed_20260929_213908_41a9ea | e2_res_fx20_A_s2 | E2 | fixed 0.2 | ResNet-18 middle split, stepwise composition change | 2 | yes | 150 | 16 | 0 |
| fixed_20260929_032312_dda380 | e2_res_fx40_A_s0 | E2 | fixed 0.4 | ResNet-18 middle split, stepwise composition change | 0 | yes | 150 | 16 | 0 |
| fixed_20260929_123309_dd1ed5 | e2_res_fx40_A_s1 | E2 | fixed 0.4 | ResNet-18 middle split, stepwise composition change | 1 | yes | 150 | 16 | 0 |
| fixed_20260929_214026_a5f010 | e2_res_fx40_A_s2 | E2 | fixed 0.4 | ResNet-18 middle split, stepwise composition change | 2 | yes | 150 | 16 | 0 |
| selfcal_20260929_223447_4a0ada | e3_lmax060_mob_s0 | E3 | DriftGate λ_max 0.60 | client mobility | 0 | yes | 120 | 13 | 0 |
| selfcal_20260929_223622_738de2 | e3_lmax060_mob_s1 | E3 | DriftGate λ_max 0.60 | client mobility | 1 | yes | 120 | 13 | 0 |
| selfcal_20260930_014436_3dadbe | e3_lmax060_mob_s2 | E3 | DriftGate λ_max 0.60 | client mobility | 2 | yes | 120 | 13 | 0 |
| selfcal_20260930_030959_d6f723 | e3_lmax080_mob_s0 | E3 | DriftGate λ_max 0.80 | client mobility | 0 | yes | 120 | 13 | 0 |
| selfcal_20260930_035351_7db315 | e3_lmax080_mob_s1 | E3 | DriftGate λ_max 0.80 | client mobility | 1 | yes | 120 | 13 | 0 |
| selfcal_20260930_044303_b8d303 | e3_lmax080_mob_s2 | E3 | DriftGate λ_max 0.80 | client mobility | 2 | yes | 120 | 13 | 0 |
| selfcal_20260929_043041_d40d6b | e3_guard025_A_s0 | E3 | DriftGate guard 0.25 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| selfcal_20260929_044453_aa7cb3 | e3_guard025_A_s1 | E3 | DriftGate guard 0.25 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| selfcal_20260929_072952_49cfef | e3_guard025_A_s2 | E3 | DriftGate guard 0.25 | stepwise composition change | 2 | yes | 150 | 16 | 0 |
| selfcal_20260929_084203_154f20 | e3_guard100_A_s0 | E3 | DriftGate guard 1.0 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| selfcal_20260929_084615_c77080 | e3_guard100_A_s1 | E3 | DriftGate guard 1.0 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| selfcal_20260929_104200_b3d23e | e3_guard100_A_s2 | E3 | DriftGate guard 1.0 | stepwise composition change | 2 | yes | 150 | 16 | 0 |
| selfcal_20260929_104352_d586f4 | e3_lmax060_A_s0 | E3 | DriftGate λ_max 0.60 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| selfcal_20260929_130424_e479b5 | e3_lmax060_A_s1 | E3 | DriftGate λ_max 0.60 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| selfcal_20260929_144456_f71c1f | e3_lmax060_A_s2 | E3 | DriftGate λ_max 0.60 | stepwise composition change | 2 | yes | 150 | 16 | 0 |
| selfcal_20260929_145729_29717b | e3_lmax080_A_s0 | E3 | DriftGate λ_max 0.80 | stepwise composition change | 0 | yes | 150 | 16 | 0 |
| selfcal_20260929_162300_ffefeb | e3_lmax080_A_s1 | E3 | DriftGate λ_max 0.80 | stepwise composition change | 1 | yes | 150 | 16 | 0 |
| selfcal_20260929_164735_f5d73b | e3_lmax080_A_s2 | E3 | DriftGate λ_max 0.80 | stepwise composition change | 2 | yes | 150 | 16 | 0 |

## 8. 파일 구성

- `DriftGate_final_report_ko.md`: 이 보고서
- `paper_numbers.csv`: 논문에 들어갈 숫자(항목, 값, 신뢰구간, seed 수, 출처 run)
- `tables/`: 표 1–10과 보조 표의 CSV, `run_manifest.csv`
- `figures/`: 그림 5개(PDF, PNG)와 `figure_captions.md`
- `precheck/`: 시작 전 확인 스크립트와 결과, 코드 sha256
- `scripts/`: 재생성 스크립트. 순서는 `r5_tables.py` → `r5_config_comm.py` → `r5_manifest.py` → `r5_figures.py` → `r5_report.py`이다. `scripts/launch/`에는 실행에 쓴 `enqueue_r5.py`, `r5_worker.py`, `r5_monitor.py`, `supervisor.sh`의 사본과 실제로 넣은 명령 목록(`enqueued_*_snapshot.txt`)이 있다.
- `README.md`: 파일 구성과 재생성 명령
