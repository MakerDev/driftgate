# DriftGate 추가 실험 보고서 (Round 6: 이동 시나리오, 규모, 견고성, 지연 시간)

이 보고서는 Round 6 지시문(이하 지시문)과 2026-10-01 사용자 결정에 따라 실행한 실험 218개를 정리한다. 모든 수치는 원시 run JSON과 trace 파일에서 스크립트로 계산한 CSV에서 가져왔다(`scripts/r6_tables.py`, `scripts/r6_r0_tables.py`, `scripts/latency_calc.py` → `tables/*.csv`, `r0/tables/*.csv` → 이 보고서). 불리한 결과도 그대로 적는다.

## 0. 실행 개요와 시간

- **실행 환경**: 서버 `honeynaps`, NVIDIA RTX 4090 24 GB 4장(사용자가 허락한 GPU 0–3), Python 3.12.7, PyTorch 2.7.1+cu126. 218개 run 모두 이 서버에서 돌았다.
- **실행한 run**: 지시문의 Round 6 run 151개와 R0 run 67개다. R0은 Round 4–5에서 controller를 쓴 run을, 이웃 점수 평균 없이 원래 명령줄 그대로 다시 돌린 실험이다(2절).
- **시작 전 측정과 예상 시간** (`precheck/round_time.csv`, `precheck/round_time_r0.csv`)
  - RTX 4090 한 장에서 run 하나만 돌릴 때 라운드당 시간은 K=50 9.7초, K=200 37.3초, K=500 92.1초였다. K는 클라이언트 수다.
  - R0의 설정별 라운드당 시간은 다음과 같다. 단계적 구성 변화 34.7초, client mobility 35.0초, CIFAR-10 gradual 35.8초, CIFAR-100 gradual 33.3초, SVHN 58.8초, Tiny-ImageNet 130.6초, ResNet-18 42.3초. R0은 Round 5와 같은 코드 경로를 쓰기 때문에 Round 6 run보다 느리다.
  - 한 GPU에 K=50 run 6개를 함께 올리면 run마다 4.05배 느려졌고, 처리량은 혼자 돌릴 때의 1.48배였다.
  - 사용자 결정 이후 남은 작업량은 R0 약 110 GPU-시간과 Round 6 약 100 GPU-시간이었다(혼자 돌릴 때 기준). GPU 4장과 처리량 1.4배를 가정해 약 37시간을 예상했다.
  - K=500 run은 GPU 메모리를 약 8.9 GB 쓴다. 한 GPU에 3개를 올리면 메모리가 부족해서, GPU당 2개로 제한했다.
- **실제 시간**: 큐 v3를 2026-10-01 18:58에 시작했고, 2026-10-02 19:47에 218개가 모두 끝났다(약 25시간). 그 전에 운영 실수로 GPU가 약 17시간 놀았다(8절).
- **결과 완료**: 218개 모두 150 라운드(R0은 원래 설정의 라운드 수)를 마쳤다. 모든 run에서 controller용 표본과 평가용 표본이 겹치지 않았다(overlap = 0, `tables/T10_run_manifest.csv`).

### 용어와 표기

- **DriftGate**는 relonly controller에서 이웃 점수 평균 단계를 뺀 최종 방법이다(2절). 실행 flag는 `--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --no_neighbor_avg --disjoint_pools`이다.
- **TV**(total variation distance)는 같은 요청에 대해 client exit와 server exit가 낸 확률 벡터의 거리다. d̄는 cluster에 속한 클라이언트들의 TV 평균이다.
- **pp**는 정확도 백분율의 차이(퍼센트포인트)다. **integrated accuracy**는 평가 라운드 {1, 5, 10, …, 150}(31개)에서 잰 acc_total의 평균이다.
- 차이는 모두 같은 seed끼리 뺀 paired 차이다. 괄호 안은 Student-t 95% 신뢰구간(CI)이고, 그 뒤는 DriftGate가 높았던 seed 수/전체 seed 수(n_pos/n)다.
- **가장 좋은 고정값**은 같은 시나리오에서 시험한 고정 λ 가운데 integrated accuracy의 seed 평균이 가장 높은 값이다. 비교에 쓰는 seed 집합에서 고른다.
- **entropy controller**는 DriftGate와 같은 controller에 TV 대신 client exit의 entropy를 넣은 방법이다.
- **APFL**은 클라이언트가 자기 라벨 손실로 λ_k를 학습하는 기존 방법이다(η = 0.01, 0.1).
- 시나리오 이름은 다음과 같다.
  - **S1**(commute mobility): 출퇴근 이동, K=50, cell 5개, 도보 속도
  - **S1-fast**: S1과 같고 이동 속도만 차량 속도
  - **S2**(GeoLife trace): GeoLife 실제 궤적
  - **S3**: 구역을 이어 붙인 규모 실험(K=200, 500)
  - **S4**: 견고성 실험(신호 손실, 신호 지연, 적은 요청량, 참여율)
- 시간대는 다음과 같다: 출근 전(05:00–07:30), 출근(07:30–09:30), 낮(09:30–16:00), 퇴근(16:00–19:00), 저녁(19:00–20:00).

## 1. 주장별 판정표

| # | 주장 | 근거 표·그림 | 판정 | 핵심 수치 |
|---|---|---|---|---|
| 1 | 출퇴근 이동(S1)에서 DriftGate는 시험한 모든 고정 λ(0.15–0.6)보다 높다 | T2, T4, 그림 3 | 반대 | DriftGate 62.26%. 가장 좋은 고정값 λ=0.4(62.86%) 대비 −0.59 pp [−0.93, −0.25] (0/5), λ=0.5 대비 −0.34 pp [−0.59, −0.09] (0/5). λ=0.15, 0.2, 0.3, 0.6과의 차이는 CI가 0을 포함한다(+0.22, −0.09, −0.50, +0.16 pp) |
| 2 | 실제 이동 궤적(S2)에서 DriftGate는 시험한 모든 고정 λ(0.15–0.6)보다 높다 | T2, 그림 3 | 반대 | DriftGate 64.95%. 가장 좋은 고정값 λ=0.6(65.33%) 대비 −0.38 pp [−3.76, +3.00] (2/3). DriftGate의 평균은 λ=0.3–0.6 네 값보다 낮고 λ=0.15, 0.2보다 높다. 여섯 비교 모두 CI가 0을 포함한다 |
| 3 | S1과 S2에서 DriftGate는 entropy controller와 APFL식 학습보다 높다 | T2 | 반대 | S1: entropy 대비 +0.54 pp [−0.04, +1.13] (4/5), APFL η=0.01 대비 +0.47 pp [+0.07, +0.88] (3/3), APFL η=0.1 대비 −0.09 pp [−0.34, +0.16] (1/3). S2: entropy 대비 −0.21 pp [−2.32, +1.89] (2/3), APFL η=0.01 대비 −0.94 pp [−3.31, +1.44] (1/3), APFL η=0.1 대비 −1.33 pp [−3.88, +1.21] (0/3). 여섯 비교 가운데 CI 하한이 0보다 큰 것은 S1의 APFL η=0.01 하나다 |
| 4 | DriftGate는 출근 시간대에 hub cell의 λ를 낮추고, 퇴근 뒤 다시 올린다 | T5, 그림 2 | 부분 지지 | S1 hub cell의 시간대별 평균 λ: 출근 전 0.425(warm-up), 출근 0.435, 낮 0.342, 퇴근 0.363, 저녁 0.450. 반응 시간 평균 84분(hub, 5개 cell 모두 도달), 회복 시간 0분(hub, 회복 시점이 정의된 3개 cell). 주거 cell은 반응 83분(20개 가운데 19개 도달), 회복 28분(15개). λ는 낮에 내려가고 저녁에 올라가지만, 출근 시간대의 hub λ는 warm-up 값과 거의 같고 가장 낮은 λ는 오후에 나타난다 |
| 5 | 차량 속도(S1-fast)에서도 DriftGate는 고정 λ 0.2, 0.4, 0.6보다 높다 | T6, 그림 4 | 반대 | DriftGate 62.93%, 가장 좋은 고정값 λ=0.4(63.35%) 대비 −0.42 pp [−0.87, +0.02] (0/3). 같은 seed의 S1(도보)에서는 −0.44 pp [−1.04, +0.15] (0/3)였다 |
| 6 | K=200과 K=500에서도 DriftGate는 고정 λ 0.2, 0.4, 0.6보다 높고, edge 하나의 추가 비용은 K에 따라 늘지 않는다 | T7, 그림 4 | 반대 | 정확도: K=200은 λ=0.2 대비 −0.57 pp [−1.73, +0.59] (0/3), K=500은 λ=0.2 대비 −0.24 pp [−0.57, +0.10] (0/3). 비용: edge 하나가 라운드마다 점수와 λ를 계산하는 시간의 중앙값은 K=50, 200, 500에서 35.5, 62.0, 125.9 µs이고, edge 하나의 신호 byte는 61.5, 122.0, 242.9 byte다. 둘 다 cell 수 L이 커지면 함께 늘어난다 |
| 7 | 신호 손실 0.3, 신호 지연 3 라운드, 적은 요청량에서도 DriftGate는 S1의 가장 좋은 고정값보다 높다 | T8, 그림 4 | 반대 | S1 고정 λ=0.4(seeds 0–2) 대비 손실 0.3 −0.38 pp [−0.82, +0.06] (0/3), 지연 3 −0.43 pp [−0.96, +0.09] (0/3), 적은 요청량 −0.49 pp [−1.15, +0.16] (0/3). 같은 seed의 S1 DriftGate와 비교하면 세 조건 모두 변화가 0.07 pp 이내다(+0.06, +0.01, −0.05 pp) |
| 8 | 참여율 0.7과 0.5에서도 DriftGate는 같은 참여율의 고정 λ 0.2, 0.4, 0.6보다 높다 | T8, 그림 4 | 반대 | 참여율 0.7: λ=0.4 대비 −0.33 pp [−0.86, +0.19] (0/3). 참여율 0.5: λ=0.6 대비 −0.40 pp [−1.59, +0.80] (1/3) |
| 9 | DriftGate와 가장 좋은 고정값의 offload 비율과 E2E 지연 시간은 얼마나 다른가 | T9, 그림 5 | 판정 없음 | S1: offload 비율 DriftGate 90.5%, λ=0.4 92.0%. 평균 E2E(uplink 50 Mbps, RTT 20 ms) 23.8 ms 대 24.2 ms, p95 둘 다 26.2 ms. S2: offload 75.6% 대 69.6%(λ=0.6), 평균 E2E 20.0 ms 대 18.5 ms, p95 둘 다 26.2 ms. 클라이언트 쪽 계산 시간은 서버 CPU에서 잰 임시값이다 |
| 10 | DriftGate의 이득이 밖에 있는 클라이언트에서 나오는가, 집에 있는 클라이언트의 정확도는 가장 좋은 고정값과 얼마나 다른가 | T3 | 판정 없음 | S1(λ=0.4 대비): 밖 −0.76 pp [−1.82, +0.29] (1/5), 집 −0.75 pp [−2.36, +0.85] (2/5). S2(λ=0.6 대비): 밖 +1.49 pp [−0.50, +3.48] (3/3), 집 −1.14 pp [−6.59, +4.31] (1/3). 하위 10% 클라이언트 정확도는 S1 −0.95 pp [−1.72, −0.18] (0/5), S2 +0.26 pp [−0.56, +1.09] (2/3) |

판정 기준은 Round 5 보고서와 같다. 주장 방향의 차이가 모두 나오고 95% CI가 0을 넘으면 **지지**, 방향은 맞지만 일부 비교의 CI가 0을 포함하거나 일부 조건에서 반대 결과가 나오면 **부분 지지**, 주요 비교에서 DriftGate가 높지 않으면 **반대**로 판정했다. 주장 2, 5–8은 DriftGate의 평균이 가장 좋은 고정값보다 낮으므로, CI가 0을 포함하더라도 반대로 판정했다.

## 2. 2026-10-01 결정: 이웃 점수 평균 제거와 R0 재실행

### 2.1 결정 내용과 코드 변경

이전 controller는 cluster마다 temporal 점수와 spatial 점수를 EMA로 평활해 큰 값을 고른 뒤, 그 값을 이웃 edge의 값과 한 번 평균해서 q로 썼다. 사용자는 이 평균 단계를 빼기로 결정했다. 이웃 평균이 있으면 S2처럼 모든 edge가 이웃일 때 모든 cluster가 같은 q를 받고, S1에서도 hub의 높은 점수와 주거 cell의 낮은 점수가 서로 섞이기 때문이다. 이제 q_z는 cluster z 자신의 두 점수를 EMA(새 값의 비중 0.3)로 평활한 뒤 고른 큰 값이다. spatial 점수를 위해 edge끼리 d̄를 공유하는 단계와 나머지 상수는 그대로 두었다. entropy controller도 같은 controller를 쓴다.

기본 동작은 바꾸지 않고 flag `--no_neighbor_avg`를 추가했다. 바뀐 곳은 다음과 같다.

| 파일:줄 | 변경 |
|---|---|
| `journal_expansion/src/controllers/self_calibrating.py:82, 84–87` | `SelfCalController`에 `neighbor_avg=True` 인자 추가 |
| `journal_expansion/src/controllers/self_calibrating.py:168` | `neighbor_avg=False`이면 점수의 이웃 평균(`_consensus`)을 건너뛴다 |
| `journal_expansion/src/controllers/self_calibrating.py:174–175` | absolute 경로(absonly)의 raw 신호 이웃 평균도 건너뛴다 |
| `journal_expansion/src/runner.py:252, 612–613` | flag를 controller로 넘기고, 결과 JSON의 config에 기록한다 |
| `journal_expansion/scripts/run_v2.py:127, 163–164` | Round 5 경로의 flag |
| `journal_expansion/src/r6_controller.py:30, 32, 77` | Round 6 edge별 controller: q = 자기 점수 |
| `journal_expansion/src/runner_r6.py:264, 354` | flag 전달. 이웃 평균이 없으면 이웃 점수 메시지(손실 대상)도 없다 |
| `journal_expansion/scripts/run_r6.py:52, 85` | Round 6 경로의 flag |

flag 없이 실행하면 기존 동작과 bit 단위로 같다는 것을 테스트로 확인했다(`tests/test_r6.py`, 전체 98개 통과).

**absonly 경로 확인**: absonly는 TV 크기로 λ를 바로 정하는 비교 controller다. 원래 코드에서 absonly는 absolute 경로에서 raw 신호를 이웃과 한 번 평균했다(`self_calibrating.py:174–175`의 원래 코드 `sig_c = _consensus(signal_per_es, ...)`). 따라서 absonly도 이웃 평균을 거치므로 다시 돌렸다.

### 2.2 R0 재실행

R0은 다음 67개 run이다. 각 run의 명령줄은 원래 run의 provenance 명령줄(또는 Round 5 큐 snapshot)과 flag 단위로 같고, `--no_neighbor_avg`만 더했다(67개 모두 대조 완료). 결과는 `runs/phaseT6_R0/r0_<원래 이름>`에 있다.

- DriftGate 25개: 단계적 구성 변화(0–4), client mobility(0–2), CIFAR-10 gradual(0–2), CIFAR-100 gradual(0–2), Tiny-ImageNet(0–2), SVHN(0–4), ResNet-18 middle split(0–2)
- entropy controller 13개: 단계적 구성 변화(0–4), client mobility(0–2), SVHN(0–4)
- 상수 민감도 E3 18개: guard 0.25, 1.0(단계적 구성 변화 0–2), λ 상한 0.60, 0.80(단계적 구성 변화와 client mobility 0–2)
- absonly 11개: 단계적 구성 변화(0–4), client mobility(0–2), CIFAR-100 gradual(0–2)

고정 λ, APFL, passive 신호 실행, 역할 실험은 controller를 쓰지 않으므로 다시 돌리지 않았다.

### 2.3 이웃 평균이 있을 때와 없을 때의 차이

같은 seed끼리 비교했다(`r0/tables/R0_neighbor_avg_record.csv`). 값은 R0(이웃 평균 없음) − 기존 run(있음)이다. 사용자의 지시에 따라 기존 run(옛 서버)과 R0(새 서버)의 하드웨어 차이는 고려하지 않는다.

| 설정 | 방법 | 이웃 평균 있음 (%) | 없음 (%) | 차이 (pp) | 95% CI | 없음이 높은 seed |
|---|---|---|---|---|---|---|
| 단계적 구성 변화 | DriftGate | 65.93 | 65.81 | −0.11 | [−0.34, +0.11] | 2/5 |
| client mobility | DriftGate | 59.23 | 59.15 | −0.08 | [−0.15, −0.00] | 0/3 |
| CIFAR-10 gradual | DriftGate | 62.68 | 62.47 | −0.21 | [−0.89, +0.47] | 0/3 |
| CIFAR-100 gradual | DriftGate | 36.09 | 36.08 | −0.01 | [−0.08, +0.07] | 1/3 |
| Tiny-ImageNet | DriftGate | 23.80 | 23.76 | −0.04 | [−0.14, +0.06] | 1/3 |
| SVHN temporal | DriftGate | 79.90 | 79.48 | −0.42 | [−0.78, −0.05] | 0/5 |
| ResNet-18 middle split | DriftGate | 62.39 | 62.16 | −0.22 | [−2.48, +2.04] | 2/3 |
| 단계적 구성 변화 | guard 0.25 | 66.07 | 65.83 | −0.24 | [−0.36, −0.13] | 0/3 |
| 단계적 구성 변화 | guard 1.0 | 65.84 | 65.61 | −0.23 | [−0.37, −0.08] | 0/3 |
| 단계적 구성 변화 | λ 상한 0.60 | 65.56 | 65.48 | −0.08 | [−0.73, +0.58] | 2/3 |
| 단계적 구성 변화 | λ 상한 0.80 | 65.80 | 65.58 | −0.22 | [−0.85, +0.42] | 1/3 |
| client mobility | λ 상한 0.60 | 59.34 | 59.28 | −0.06 | [−0.10, −0.01] | 0/3 |
| client mobility | λ 상한 0.80 | 58.08 | 58.10 | +0.02 | [−0.22, +0.26] | 2/3 |
| 단계적 구성 변화 | entropy controller | 64.37 | 64.35 | −0.01 | [−0.24, +0.22] | 3/5 |
| client mobility | entropy controller | 57.21 | 57.37 | +0.16 | [−0.04, +0.36] | 3/3 |
| SVHN temporal | entropy controller | 76.97 | 77.15 | +0.18 | [−0.33, +0.68] | 3/5 |
| 단계적 구성 변화 | absonly | 64.92 | 64.93 | +0.01 | [−0.06, +0.07] | 3/5 |
| client mobility | absonly | 57.93 | 57.93 | +0.00 | [−0.04, +0.04] | 2/3 |
| CIFAR-100 gradual | absonly | 34.89 | 34.89 | +0.01 | [−0.03, +0.04] | 2/3 |
| S1 (Round 6, 결정 전에 돈 run) | DriftGate | 61.81 | 61.52 | −0.29 | [−0.72, +0.14] | 1/4 |
| S1 (Round 6, 결정 전에 돈 run) | entropy controller | 61.38 | 61.00 | −0.38 | [−1.02, +0.25] | 1/4 |

- 이웃 평균을 빼자 DriftGate는 일곱 설정 모두에서 평균이 0.01–0.42 pp 낮아졌다. SVHN(−0.42 pp)과 client mobility(−0.08 pp)는 CI가 0 아래에 있다.
- guard 변형 두 개도 0.23–0.24 pp 낮아졌고 CI가 0 아래에 있다.
- entropy controller와 absonly는 거의 변하지 않았다.
- 마지막 두 행은 Round 6 S1에서 결정 전에 이웃 평균을 켠 채로 돈 seed 0–3의 run과 같은 seed의 최종 run을 비교한 것이다. 두 run 모두 이 서버에서 돌았다.

### 2.4 Round 5 표를 R0으로 다시 계산한 결과

Round 5 표 코드(`artifacts/driftgate_tmc_final/scripts/r5_tables.py`, 수정하지 않음)를 controller run의 경로만 R0으로 바꿔 실행했다. 결과는 `r0/tables/`에 있다. 표 1, 2a–2c, 3, 4a–4d, 5, 6a–6c, 7, 10, `lambda_trajectories.csv`, `paper_numbers.csv`가 모두 새로 계산되었다. Round 5 산출물은 수정하지 않았다.

| 비교 | Round 5 (이웃 평균 있음) | R0 기준 (없음) |
|---|---|---|
| 단계적 구성 변화: DriftGate − 가장 좋은 고정값(λ=0.4) | +0.76 pp [+0.41, +1.11] (5/5) | +0.64 pp [+0.47, +0.81] (5/5) |
| client mobility: DriftGate − 가장 좋은 고정값(λ=0.4) | +0.65 pp [+0.47, +0.83] (3/3) | +0.57 pp [+0.42, +0.73] (3/3) |
| 단계적 구성 변화: DriftGate − entropy | +1.56 pp [+1.18, +1.94] (5/5) | +1.46 pp [+1.13, +1.79] (5/5) |
| client mobility: DriftGate − entropy | +2.02 pp [+1.26, +2.77] (3/3) | +1.78 pp [+0.89, +2.67] (3/3) |
| SVHN: DriftGate − entropy | +2.93 pp [+1.86, +4.01] (5/5) | +2.33 pp [+1.38, +3.29] (5/5) |
| client mobility: DriftGate − APFL η=0.1 | −0.35 pp [−1.44, +0.74] (1/3) | −0.43 pp [−1.45, +0.60] (1/3) |
| DriftGate − absonly (단계적 / mobility / CIFAR-100) | +1.01 / +1.30 / +1.20 pp | +0.89 / +1.22 / +1.19 pp (모두 CI > 0) |
| Tiny-ImageNet: DriftGate − 고정 λ=0.15 | −0.57 pp [−0.95, −0.19] (0/3) | −0.61 pp [−1.08, −0.14] (0/3) |
| SVHN: DriftGate − 고정 λ=0.15 | −0.54 pp [−1.40, +0.31] (1/5) | −0.96 pp [−1.80, −0.12] (0/5) |
| CIFAR-10 gradual: DriftGate − 고정 λ=0.2 | −0.17 pp [−2.08, +1.73] (2/3) | −0.38 pp [−2.96, +2.20] (2/3) |
| CIFAR-100 gradual: DriftGate − 고정 λ=0.15 | +0.17 pp [−0.20, +0.53] (3/3) | +0.16 pp [−0.22, +0.54] (3/3) |
| ResNet-18: DriftGate − 고정 λ=0.4 | +3.19 pp [+0.37, +6.01] (3/3) | +2.96 pp [+0.38, +5.54] (3/3) |
| E3 λ 상한 0.60 − 기본값 (단계적) | −0.51 pp [−1.15, +0.12] | −0.35 pp [−0.68, −0.02] |
| E3 λ 상한 0.80 − 기본값 (mobility) | −1.15 pp [−1.51, −0.80] (0/3) | −1.05 pp [−1.20, −0.91] (0/3) |

- Round 5의 대표 결과(단계적 구성 변화와 client mobility에서 모든 고정 λ, entropy, absonly보다 높다)는 R0에서도 유지된다. 다만 차이는 비교에 따라 0.01–0.60 pp 줄었다.
- SVHN에서는 가장 좋은 고정값과의 차이가 −0.54 pp에서 −0.96 pp로 커졌고, CI가 0 아래로 내려갔다.

## 3. 환경 확인 (지시문 3.3, 3.9)

### 3.1 class group (3.3)

- static topology에서 cluster g의 기본 구성원은 client 10g–10g+9다. 각 cluster의 기본 구성원 10명이 받은 Main class의 합집합이 class group g다.
- seeds 0–4에서 class group은 4–7개 class로 이루어지고, 두 group이 공유하는 class는 0–5개(seed별 평균 2.4–4.2개)다. 8개 이상의 class를 가진 group은 어느 seed에도 없다.
- 결과를 보기 전에 한 번 판단한 결론은, cell마다 class 범위가 구분되므로 기존 nd1 분할을 그대로 쓴다는 것이다. {2g, 2g+1} 분할로는 바꾸지 않았다.
- 목록: `env_check/class_groups.csv`(cell별 class group), `env_check/s1_partition_clients.csv`(클라이언트별 Main class와 표본 수, 620–2,857개)

### 3.2 시나리오별 환경 (3.9)

| 시나리오 | 집에 있는 비율 | 두 cell에 속한 비율 | 라운드당 소속이 바뀐 클라이언트 | probe 요청 수 평균 | n < 64 비율 | n = 0 비율 | OOR 집합이 빈 비율(집) |
|---|---|---|---|---|---|---|---|
| S1 | 0.571 | 0.139 | 1.19 | 58.3 | 0.343 | 0 | 0.025 |
| S1-fast | 0.562 | 0.108 | 0.73 | 58.5 | 0.327 | 0 | 0.027 |
| S2 | 0.746 | 0.062 | 1.28 | 58.9 | 0.290 | 0 | 0.004 |
| S3 K=200 | 0.543 | 0.165 | 5.68 | 58.1 | 0.342 | 0 | 0.020 |
| S3 K=500 | 0.546 | 0.156 | 14.39 | 58.4 | 0.338 | 0 | 0.002 |

- S1에서 집에 있는 클라이언트의 비율은 05:00의 1.0에서 낮에 약 0.35로 내려갔다가 20:00에 1.0으로 돌아온다. hub cell의 인원은 낮에 최대 약 32명이다.
- S1의 cell 평균 ρ는 0.1에서 낮에 0.45–0.6으로 오른 뒤 0.1로 돌아온다. ρ는 요청 구성을 정하는 매개변수다.
- 실제 요청에서 Main이 아닌 요청의 비율은, 집에 있는 클라이언트에서 약 11.5%로 정의(ρ=0.1)와 같다.
- OOR(현재 cell의 어느 클라이언트도 학습하지 않은 class) 집합이 비어 OOR 몫을 OOP에서 뽑은 경우는 전체 요청 pool의 0.9–3.8%였다. OOP 집합이 빈 경우는 S2에서만 0.8% 있었다.
- 그림: `env_check/env_check_<시나리오>.png`, 시간별 값: `env_check/env_check_<시나리오>.csv`

## 4. S2 데이터와 전처리

- **데이터**: GeoLife Trajectories 1.3(Microsoft Research). 배포 주소에서 받았다(313,164,406 byte, sha256 `1107c5ac…86bdb6`). 전처리는 지시문 S2의 1–10번을 그대로 따랐다(`scripts/r6_trace_env.py`).
- **점 수**: 전체 24,876,978개 가운데 영역 필터와 속도 필터를 통과한 점은 18,487,422개다.
- **후보와 선택**: 후보 사용자는 135명으로 50명 이상이어서 T-Drive는 쓰지 않았다. 사용자별로 가장 좋은 하루를 골라 점수 순으로 50개를 선택했다. 점수는 150–77이고, 선택 목록은 `runs/phaseT6_env/S2_selection.csv`에 있다.
- **edge 위치**: k-means(L=5, n_init=10, random_state=0)로 정했고, 평균 위치 기준 km 좌표는 (−2.51, 3.12), (−7.36, −16.33), (8.58, −7.54), (19.31, 8.54), (−21.05, 10.87)이다.
- **cell별 주민 수**: 38, 4, 5, 2, 1명으로 치우쳐 있다. 대부분의 사용자가 베이징 하이뎬 지역에 산다. 빈 cell은 없다.
- **중복 궤적**: 사용자 112와 163의 2008-06-20 기록은 기록 수(6,399개)가 같은 중복 궤적으로 보이며, 둘 다 선택되었다. 지시문에 중복 처리 규칙이 없어서 그대로 두었다.
- **이동 패턴**: S2는 S1보다 출퇴근 패턴이 약하다. 집에 있는 비율은 낮에도 약 0.6이고, 주민이 가장 많은 cell 0의 평균 ρ는 하루 내내 0.1–0.22다.

## 5. 새 실험 결과

### 5.1 S1과 S2의 대표 결과 (표 T2, 그림 3)

| 방법 | S1 정확도 (%) | S1: DriftGate − 방법 | S2 정확도 (%) | S2: DriftGate − 방법 |
|---|---|---|---|---|
| DriftGate | 62.26 | | 64.95 | |
| 고정 λ 0.15 | 62.04 | +0.22 [−1.09, +1.53] (4/5) | 64.32 | +0.63 [−0.95, +2.21] (3/3) |
| 고정 λ 0.2 | 62.35 | −0.09 [−1.16, +0.98] (4/5) | 64.58 | +0.37 [−1.19, +1.93] (2/3) |
| 고정 λ 0.3 | 62.76 | −0.50 [−1.14, +0.14] (0/5) | 64.99 | −0.04 [−1.84, +1.77] (2/3) |
| 고정 λ 0.4 | 62.86 (가장 좋은 고정값) | −0.59 [−0.93, −0.25] (0/5) | 65.21 | −0.26 [−2.57, +2.05] (2/3) |
| 고정 λ 0.5 | 62.60 | −0.34 [−0.59, −0.09] (0/5) | 65.28 | −0.32 [−3.35, +2.70] (2/3) |
| 고정 λ 0.6 | 62.11 | +0.16 [−0.11, +0.42] (3/5) | 65.33 (가장 좋은 고정값) | −0.38 [−3.76, +3.00] (2/3) |
| entropy controller | 61.72 | +0.54 [−0.04, +1.13] (4/5) | 65.17 | −0.21 [−2.32, +1.89] (2/3) |
| APFL η=0.01 (seeds 0–2) | 62.30 | +0.47 [+0.07, +0.88] (3/3) | 65.89 | −0.94 [−3.31, +1.44] (1/3) |
| APFL η=0.1 (seeds 0–2) | 62.86 | −0.09 [−0.34, +0.16] (1/3) | 66.29 | −1.33 [−3.88, +1.21] (0/3) |

Schedule A(단계적 구성 변화) 열은 2.4절의 R0 기준 값이다. S2의 seed 사이 표준편차는 3.4–4.7 pp로 커서, S2의 CI가 넓다.

### 5.2 시간대별 정확도와 기여 (표 T4)

S1에서 DriftGate − 고정 λ 0.4의 차이 −0.59 pp는 시간대별 기여로 다음과 같이 나뉜다. 기여의 합은 전체 차이와 같다.

| 시간대 | 평가 라운드 수 | 기여 (pp) | 95% CI |
|---|---|---|---|
| 출근 전 05:00–07:30 | 6 | +0.00 | [−0.05, +0.05] |
| 출근 07:30–09:30 | 4 | −0.02 | [−0.09, +0.05] |
| 낮 09:30–16:00 | 13 | −0.35 | [−0.49, −0.21] |
| 퇴근 16:00–19:00 | 6 | −0.17 | [−0.27, −0.07] |
| 저녁 19:00–20:00 | 2 | −0.06 | [−0.12, +0.01] |

- 차이는 대부분 낮과 퇴근 시간대에서 생긴다.
- 같은 낮 시간대 정확도는 DriftGate 60.68%, 고정 λ 0.4 61.52%, 0.3 61.82%다.
- S2에서는 모든 시간대의 기여가 CI에 0을 포함한다(낮 −0.14 pp, 퇴근 −0.29 pp).

### 5.3 λ의 반응 (표 T5, 그림 2)

| 시나리오 | 방법 | cell | 반응 시간 평균 | 도달하지 못한 cell | 회복 시간 평균 |
|---|---|---|---|---|---|
| S1 | DriftGate | hub | 84.0분 (14.0 라운드) | 0/5 | 0.0분 (3개 cell) |
| S1 | DriftGate | 주거 | 83.4분 (13.9 라운드) | 1/20 | 28.0분 (15개 cell) |
| S1 | entropy controller | hub | 110.4분 (18.4 라운드) | 0/5 | 0.0분 (3개 cell) |
| S1 | entropy controller | 주거 | 220.5분 (36.8 라운드) | 12/20 | 3.3분 (18개 cell) |

- **정의(5.2절)**: 반응 시간은 cluster 평균 ρ가 라운드 26 이후 처음으로 0.3을 넘은 라운드부터, λ가 처음으로 0.425 이하가 될 때까지의 시간이다. 회복 시간은 라운드 111 이후 ρ가 처음으로 0.3 아래로 내려간 라운드부터, λ가 처음으로 0.425를 넘을 때까지의 시간이다. 회복 시작 시점에 λ가 이미 0.425를 넘어 있으면 회복 시간은 0이다. cell별 값과 두 시점의 λ는 `tables/T5a_lambda_response_per_cell.csv`에 있다.
- **S1의 시간대별 평균 λ(DriftGate)**
  - hub: 출근 전 0.425(warm-up), 출근 0.435, 낮 0.342, 퇴근 0.363, 저녁 0.450
  - 주거: 0.425, 0.480, 0.359, 0.385, 0.450
  - 같은 시간대의 cell 평균 ρ는 hub에서 0.12, 0.48, 0.55, 0.44, 0.12다.
- **warm-up 직후의 λ**: λ는 warm-up(처음 25 라운드, 07:30까지)이 끝난 직후 약 0.6으로 올랐다가 낮 동안 천천히 내려간다. 그래서 출근 시간대에는 hub λ가 warm-up 값보다 낮지 않다.
- **S2**: 이웃 평균이 없으므로 cell마다 다른 λ를 받는다(그림 2 오른쪽). 반응 시간은 cell에 따라 0–158분이다.

### 5.4 cell과 시간대별 가장 좋은 고정 λ (표 T11, 6번 분석)

S1과 S2의 고정 λ run에서, cell 종류(S1은 hub와 주거 cell, S2는 cell별)와 시간대마다 정확도를 계산했다. 평가 라운드에 클라이언트가 속한 cell을 기준으로 삼았고, 두 cell에 속한 클라이언트는 두 cell 모두에 넣었다.

| 시나리오 | cell | 출근 전 | 출근 | 낮 | 퇴근 | 저녁 |
|---|---|---|---|---|---|---|
| S1 | hub | 0.6 | 0.4 | 0.3 | 0.4 | 0.6 |
| S1 | 주거 | 0.6 | 0.6 | 0.3 | 0.5 | 0.6 |
| S2 | cell 0 (주민 38) | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 |
| S2 | cell 1 (4) | 0.6 | 0.6 | 0.15 | 0.2 | 0.3 |
| S2 | cell 2 (5) | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| S2 | cell 3 (2) | 0.6 | 0.6 | 0.6 | 0.6 | 0.15 |
| S2 | cell 4 (1) | 0.6 | 0.15 | 0.15 | 0.15 | 0.15 |

- **같은 시각, 다른 cell**: S1에서는 출근과 퇴근 시간대에 cell 종류에 따라 가장 좋은 값이 다르고, 나머지 세 시간대에는 같다. S2에서는 다섯 시간대 모두 cell마다 다르다.
- **같은 cell, 다른 시간대**: S1의 hub와 주거 cell, S2의 cell 0, 1, 3, 4에서 시간대마다 가장 좋은 값이 바뀐다. S2의 cell 2만 하루 내내 0.6이다.
- **차이의 크기**: 가장 좋은 값과 두 번째 값의 차이는 대부분 0.5 pp보다 작다. S1 낮 시간대의 hub에서는 0.22 pp, 주거 cell에서는 0.17 pp다(`tables/T11b_best_fixed_by_cell_and_time.csv`).

### 5.5 추가 지표 (표 T3)

| 시나리오 | 방법 | 하위 10% | 집 | 밖 | Main | Main 아님 | offload 비율 |
|---|---|---|---|---|---|---|---|
| S1 | DriftGate | 46.27 | 69.13 | 53.85 | 71.16 | 36.10 | 90.56 |
| S1 | 고정 λ 0.4 | 47.22 | 69.88 | 54.61 | 72.31 | 36.28 | 92.05 |
| S1 | entropy controller | 44.48 | 69.87 | 52.09 | 73.20 | 30.19 | 81.89 |
| S2 | DriftGate | 45.83 | 70.30 | 50.72 | 74.25 | 32.54 | 75.54 |
| S2 | 고정 λ 0.6 | 45.57 | 71.44 | 49.23 | 76.30 | 28.00 | 69.62 |
| S2 | entropy controller | 46.03 | 70.85 | 50.07 | 74.95 | 31.55 | 73.71 |

단위는 %다. 모든 방법의 값과 DriftGate − 각 방법의 paired 차이는 `tables/T3a_extra_metrics.csv`, `tables/T3b_extra_metrics_paired.csv`에 있다. 고정 λ가 커질수록 Main 요청의 정확도는 오르고, Main이 아닌 요청의 정확도와 offload 비율은 내려간다.

### 5.6 이동 속도, 규모, 견고성 (표 T6–T8, 그림 4)

| 조건 | DriftGate (%) | 가장 좋은 고정값 | DriftGate − 가장 좋은 고정값 | 같은 seed의 S1 DriftGate 대비 |
|---|---|---|---|---|
| S1 (seeds 0–2) | 62.77 | λ=0.4 63.22% | −0.44 [−1.04, +0.15] (0/3) | |
| S1-fast (차량 속도) | 62.93 | λ=0.4 63.35% | −0.42 [−0.87, +0.02] (0/3) | |
| K=200 (L=20) | 64.84 | λ=0.2 65.40% | −0.57 [−1.73, +0.59] (0/3) | |
| K=500 (L=50) | 66.83 | λ=0.2 67.06% | −0.24 [−0.57, +0.10] (0/3) | |
| 신호 손실 p=0.1 | 62.79 | S1 λ=0.4 63.22% | −0.43 [−0.98, +0.12] (0/3) | +0.01 [−0.04, +0.07] |
| 신호 손실 p=0.3 | 62.84 | S1 λ=0.4 63.22% | −0.38 [−0.82, +0.06] (0/3) | +0.06 [−0.09, +0.22] |
| 신호 지연 1 라운드 | 62.80 | S1 λ=0.4 63.22% | −0.42 [−0.96, +0.12] (0/3) | +0.02 [−0.03, +0.08] |
| 신호 지연 3 라운드 | 62.78 | S1 λ=0.4 63.22% | −0.43 [−0.96, +0.09] (0/3) | +0.01 [−0.06, +0.08] |
| 적은 요청량 (μ ~ U[8, 24]) | 62.72 | S1 λ=0.4 63.22% | −0.49 [−1.15, +0.16] (0/3) | −0.05 [−0.17, +0.07] |
| 참여율 0.7 | 59.43 | 같은 참여율 λ=0.4 59.76% | −0.33 [−0.86, +0.19] (0/3) | −3.34 [−5.13, −1.56] |
| 참여율 0.5 | 56.24 | 같은 참여율 λ=0.6 56.63% | −0.40 [−1.59, +0.80] (1/3) | −6.54 [−7.29, −5.78] |

- 열한 조건 모두 DriftGate의 평균이 가장 좋은 고정값보다 0.24–0.57 pp 낮다. CI는 모두 0을 포함한다.
- 신호 손실, 신호 지연, 적은 요청량에서는 DriftGate 자체의 정확도가 S1과 거의 같다(차이 0.07 pp 이내).
- 참여율을 낮추면 DriftGate와 고정 λ가 함께 낮아진다.
- 규모(T7)에 따른 신호 byte와 controller 계산 시간은 다음과 같다.

| K | L | 라운드당 신호 byte (클라이언트 TV / d̄ 공유 / 합계) | edge 하나의 신호 byte | edge 하나의 controller 계산 시간 (중앙값 / p95) |
|---|---|---|---|---|
| 50 | 5 | 228 / 80 / 308 | 61.5 | 35.5 / 108.8 µs |
| 200 | 20 | 921 / 1,520 / 2,441 | 122.0 | 62.0 / 138.5 µs |
| 500 | 50 | 2,344 / 9,800 / 12,144 | 242.9 | 125.9 / 254.2 µs |

- d̄ 공유는 edge마다 4(L−1) byte이므로 cell 수 L에 비례해 늘어난다. 계산 시간도 spatial 점수의 중앙값을 L개 값으로 계산하기 때문에 L과 함께 늘어난다.
- 계산 시간은 run 16개가 서버를 함께 쓰는 동안 Python에서 잰 값이다.

## 6. 지연 시간, 비용, 기기 측정 준비물

### 6.1 서버에서 잰 값 (`tables/T9a_server_timing.csv`)

모든 run이 끝난 뒤 GPU가 비어 있을 때 재었다. 20회 예열한 뒤 200회 실행했다.

| 항목 | 중앙값 | p95 | 하드웨어 |
|---|---|---|---|
| A(요청 하나의 중간 특징 크기) | 32,768 byte | | [1, 128, 8, 8] fp32 |
| t_s(edge GPU), 요청 1개 | 0.109 ms | 0.114 ms | RTX 4090 |
| t_s(edge GPU), 요청 64개 | 0.112 ms | 0.117 ms | RTX 4090 |
| t_c(host CPU, 임시값), 요청 1개 | 0.877 ms | 0.938 ms | AMD Ryzen Threadripper PRO 7965WX, 24 threads |
| t_c(host CPU, 임시값), 요청 64개 | 8.40 ms | 10.85 ms | 같음 |
| t_s(host CPU, 기기 사본의 임시값), 요청 64개 | 1.42 ms | 1.51 ms | 같음 |
| 로컬 학습 한 라운드(host CPU, 임시값) | 1.345 s | | 표본 908개, epoch 3, batch 32 |

t_c는 client block과 client exit를 실행하는 시간이고, t_s는 server block과 server exit를 실행하는 시간이다.

### 6.2 E2E 지연 시간 (`tables/T9_latency.csv`, `T9_latency_by_slot.csv`)

요청 하나의 E2E(요청이 도착한 뒤 예측이 나올 때까지) 지연 시간은 지시문 6.2의 식으로 계산했다. client exit로 끝나는 요청은 t_c이고, offload하는 요청은 t_c + 8A/B + RTT + t_s(edge)다. B는 uplink 대역폭, RTT는 왕복 지연 시간이다.

| 시나리오 | 방법 | offload 비율 | 평균 E2E: B=10 / 50 / 100 Mbps (RTT 20 ms) | 평균 E2E: B=50 (RTT 50 ms) | p95 E2E (RTT 20 ms) |
|---|---|---|---|---|---|
| S1 | DriftGate | 90.5% | 42.8 / 23.8 / 21.5 ms | 51.0 ms | 47.2 / 26.2 / 23.6 ms |
| S1 | 고정 λ 0.4 | 92.0% | 43.5 / 24.2 / 21.8 ms | 51.8 ms | 47.2 / 26.2 / 23.6 ms |
| S1 | entropy controller | 81.9% | 38.8 / 21.6 / 19.5 ms | 46.2 ms | 47.2 / 26.2 / 23.6 ms |
| S2 | DriftGate | 75.6% | 35.9 / 20.0 / 18.0 ms | 42.7 ms | 47.2 / 26.2 / 23.6 ms |
| S2 | 고정 λ 0.6 | 69.6% | 33.1 / 18.5 / 16.7 ms | 39.4 ms | 47.2 / 26.2 / 23.6 ms |
| S2 | entropy controller | 73.7% | 35.0 / 19.6 / 17.6 ms | 41.7 ms | 47.2 / 26.2 / 23.6 ms |

- 이 모델에서는 요청 하나의 지연 시간이 두 값(client exit, offload) 가운데 하나다. 모든 방법이 요청의 5% 넘게 offload하므로, p95는 모든 방법에서 offload 지연 시간과 같다.
- 평균 E2E는 offload 비율에 비례해 달라진다.
- 클라이언트 쪽 시간 t_c에는 host CPU 임시값을 썼다. 기기에서 잰 값과 실측 네트워크 지연이 들어오면 `latency_calc.py`로 다시 계산한다(6.4절).

### 6.3 DriftGate의 라운드당 추가 비용 (`tables/T9_overhead.csv`)

| 배치 | 추가 비용 | 로컬 학습 대비 | 6분 라운드 대비 |
|---|---|---|---|
| (a) 기기에 server block 사본: t_c(64) + t_s(64, 기기) | 9.8 ms | 0.73% | 0.003% |
| (b) edge에서 실행, B=10 Mbps: 64(A + 4C) = 2,099,712 byte 업로드 + t_s(64, edge) | 1,680 ms | 124.9% | 0.47% |
| (b) B=50 Mbps | 336 ms | 25.0% | 0.09% |
| (b) B=100 Mbps | 168 ms | 12.5% | 0.05% |

C는 class 수(10)다. 신호 메시지는 클라이언트당 4 byte다. 기기 쪽 시간과 로컬 학습 시간은 host CPU 임시값이다.

### 6.4 기기 측정 준비물 (`device/`)

| 파일 | 내용 |
|---|---|
| `models/driftgate_s1_s0_weights.pth` | S1 DriftGate seed 0의 마지막 모델. client 0의 client block + client exit, hub cell의 server block + server exit |
| `models/client_block_exit.{pt,ptl,onnx}`, `models/server_block_exit.{pt,ptl,onnx}` | TorchScript, PyTorch Mobile lite interpreter, ONNX(opset 17, batch 크기 가변) |
| `models/MANIFEST.csv` | sha256과 원래 모델과의 최대 출력 차이. TorchScript 0, lite 7.6e-6 이하, ONNX 6.4e-6 이하(onnxruntime 1.30.0, batch 1과 64) |
| `bench_device.py` | Jetson에서 CPU와 CUDA로 t_c, t_s(사본), 로컬 학습 한 라운드 시간을 잰다. 출력은 6.2의 기기 측정 CSV 형식이다 |
| `offload_server.py` | edge 서버에서 중간 특징을 받아 server block을 실행하고 logit과 서버 계산 시간을 돌려준다 |
| `offload_client.py` | Python 표준 라이브러리만 쓴다. 32,768 byte를 500회 보내고 왕복 시간, 서버 계산 시간, 네트워크 시간을 기록한다(Jetson, Android Termux) |
| `r6_device_models.py` | 모델 구조 사본 |
| `export_models.py` | 모델 파일을 다시 만드는 스크립트 |
| `README_device_ko.md` | 설치 방법, 실행 순서, 명령, 결과를 보고서에 반영하는 방법 |

## 7. 그림 목록과 영문 caption

그림은 모두 vector PDF와 300 dpi PNG로 `figures/`에 있다. caption은 `figures/figure_captions.md`에 있으며, 스크립트가 표의 값으로 만든다.

1. **fig1_scenario_commute**: Commute mobility scenario with five cells (four residential cells and one hub). (a) Cells with a 1 km edge range and the daily paths of five clients of seed 0 (numbers mark the start at 05:00). (b) Number of clients in each cell by time of day. The hub holds up to 32 clients during the day. (c) Cell mean rho, the share parameter of requests outside the client's own classes (0.1 at home, 0.8 away). Values are means over five seeds.
2. **fig1_scenario_geolife**: GeoLife trace scenario built from 50 weekday user days of GeoLife Trajectories 1.3 (135 candidate users). (a) Five edges placed by k-means and the paths of the five clients that move the most. (b) Number of clients in each cell by time of day. (c) Cell mean rho. Residents per cell are 38, 4, 5, 2, 1.
3. **fig2_lambda_trajectories**: Cell mean rho (top) and lambda chosen by DriftGate and the entropy controller (middle and bottom). Left: commute mobility, hub cell and the mean of the residential cells, lambda of both methods. Right: GeoLife trace, one line per cell (DriftGate in the middle, entropy controller at the bottom). Each edge sets lambda from its own cell score only. Dashed lines mark fixed lambda 0.4 and 0.2. Lines are means over seeds.
4. **fig3_accuracy_by_time**: Accuracy by time of day for DriftGate, the best fixed lambda of each scenario and the entropy controller. Vertical lines separate the pre-commute, commute, daytime, return and evening periods. Lines are means over seeds.
5. **fig4_scale_robustness**: Accuracy difference between DriftGate and the best fixed lambda with 95% confidence intervals over matched seeds, for larger systems (K clients, L cells), vehicle speed, signal loss, signal delay, a low request rate and partial participation. Numbers give the mean difference and the count of seeds where DriftGate is higher.
6. **fig5_latency**: (a) Share of evaluation requests sent to the edge for each method. (b, c) Mean and 95th percentile end-to-end latency in commute mobility for uplink bandwidths of 10, 50 and 100 Mbps with a 20 ms round-trip time. Every method sends more than 5% of its requests to the edge, so with this two-value latency model the 95th percentile equals the latency of an offloaded request for all three methods. Client-side timings are host placeholders until device measurements are added.

환경 확인 그림 `env_check/env_check_<시나리오>.png`는 보고서용 확인 자료이며 논문 그림 목록에는 넣지 않았다.

## 8. 불일치 기록

지시문의 정의와 다르게 구현했거나, 지시문에 없는 결정을 내린 곳을 적는다. 자세한 경과는 `WORKLOG.md`에 있다.

### 8.1 실행 위치와 방법 정의

1. **코드 위치**: 지시문 2절은 옛 서버의 `/disk2/Yujin/adaptive_splitomc_tmc/`를 적었다. 실행은 새 서버의 저장소(`/home/honeynaps/data/driftgate`, GitHub `MakerDev/driftgate`)에서 했다. 두 코드는 경로 줄만 다르다.
2. **이웃 점수 평균 제거(2026-10-01 사용자 결정)**: 지시문 1.3의 4단계를 바꿨다(2절). 결정 전에 이웃 평균을 켠 채로 돈 Round 6 run 8개(S1 DriftGate와 entropy controller의 seeds 0–3)는 기록으로만 남기고 표에는 쓰지 않았다. 이 8개는 2.3절 비교표의 마지막 두 행에만 쓴다.
3. **이웃 규칙 제거에 따른 변경**: 3.2와 S2 8번, S3의 이웃 규칙은 controller가 쓰지 않는다. S4 신호 손실은 (1) 클라이언트 TV와 (2) d̄ 공유 두 종류만 잃는다. 환경 파일에 이미 들어 있는 이웃 점수 손실 배열(`lost_nbr`)은 쓰지 않는다. 5.2의 통신량에서 이웃 교환 항목을 뺐다.

### 8.2 환경과 요청 정의

4. **cell의 class 범위**: 1.2의 "현재 cluster의 다른 클라이언트가 학습한 class"를 cell 주민의 Main class 합집합(class group)으로 해석했다. OOP = (현재 cell들의 class group 합집합) − Main, OOR = (전체 클라이언트의 Main 합집합) − Main − (현재 cell들의 class group 합집합)이다. 기존 코드의 ES 범위(ES마다 무작위로 뽑은 40–70% class)는 쓰지 않았다.
5. **요청 표본**: 기존 규칙으로 요청 pool을 만들고, 그 가운데 n = min(N, 64)개를 비복원으로 뽑았다. 이 규칙은 Main 표본 전부, OOP ρ배, OOR 0.3ρ배이고, class마다 pool 앞쪽 표본을 쓴다.
6. **S3 분할 seed**: "구역마다 seed·1000 + d"와 "구역 0의 분할은 S1과 같다"가 seeds 1, 2에서 충돌한다. 구역 0은 S1과 같은 seed를, 구역 d ≥ 1은 seed·1000 + d를 썼다.
7. **일정**: 점심과 볼일의 귀가 이동은 도착 시각 + 머무는 시간에 출발한다. 그 시각이 다음 이동(퇴근) 이후이면 귀가 이동을 뺐다. 절단 정규분포는 기각 표본 추출로 뽑았다.
8. **난수 흐름**: 네 흐름을 `SeedSequence([env_seed, 흐름 번호, layout])`으로 나눴다. 속도, 요청률, 참여 여부, 손실은 균등 난수(분위수)를 먼저 뽑고 조건별 범위로 바꿨다. 그래서 S1과 S1-fast는 같은 일정과 같은 속도 분위수를 쓰고, 참여율 0.5의 참여자는 0.7의 참여자의 부분집합이다.
9. **빈 cell**: 참여 멤버가 없는 cell은 평균을 내지 않고 이전 상태를 유지한다. Λ 혼합의 전체 평균은 그 라운드에 평균을 낸 cell들로만 계산했다. 아직 아무도 들어오지 않은 cell은 공통 초기 가중치를 갖도록 두었다.
10. **참여하지 않은 클라이언트의 이동**: 모델을 받지 않으므로 이전 cell의 server block을 유지하고, 다시 참여하는 라운드에 현재 위치로 다시 연결한다. 평가와 요청 구성은 항상 현재 위치를 따른다.
11. **신호 지연**: edge의 d̄를 d 라운드 늦게 controller에 넣었다. 처음 d 라운드에는 신호가 없어서 λ=0.425를 유지한다.
12. **spatial 점수의 값이 3개 미만일 때**: 원래 controller처럼 temporal 점수만 쓴다.
13. **고정 λ와 APFL arm**: 기존 관례대로 probe 신호를 계산하지 않았다. 따라서 5.1절의 d_k, q 같은 controller 기록은 DriftGate와 entropy controller run에만 있다.
14. **가장 좋은 고정값을 고르는 seed 집합**: 비교에 쓰는 seed 집합에서 골랐다. S4의 신호 손실, 지연, 적은 요청량은 S1 고정 λ run의 seeds 0–2에서 골랐고, 그 결과는 seeds 0–4에서 고른 값(λ=0.4)과 같다.
15. **S2 중복 궤적**: GeoLife의 사용자 112와 163의 같은 날 기록이 둘 다 선택되었다(4절).

### 8.3 구현과 실행

16. **학습 데이터 캐시**: 학습 속도를 높이려고, 미리 변환한 학습 tensor에서 batch를 꺼내는 `CachedLoader`를 Round 6 runner에 썼다. 같은 DataLoader/RandomSampler를 index에 돌리므로, CPU에서 2 라운드 뒤 weight와 난수 상태가 원래 경로와 bit 단위로 같다(`tests/test_r6.py`). R0은 Round 5 코드 경로를 그대로 썼다.
17. **추가 패키지**: S2 k-means에 scikit-learn 1.9.1을, ONNX 출력 확인에 onnxruntime 1.30.0을 가상 환경에 추가했다.
18. **운영 실수로 생긴 공백(약 17시간)**: 2026-10-01 02:10–15:10에 GPU가 비기를 기다리던 명령이 자기 자신을 실행 중인 job으로 세어서 끝나지 않았다. 15:30–18:57에는 시작 전 측정을 완료 알림 없이 띄워서 다음 단계로 넘어가지 못했다. 결과에는 영향이 없다.
19. **메모리 부족으로 다시 돌린 run**:
    - 큐 v3를 시작한 직후 GPU 1–3에 K=500 run이 3개씩 올라가서, K=500 run 3개(`s3k500_fixed060_s0/s1/s2`)를 라운드 1–2에서 멈추고 처음부터 다시 돌렸다.
    - 2026-10-02 11:53에는 K=200 run 5개(`s3k200_fixed040_s1`, `fixed060_s1`, `fixed020_s2`, `fixed040_s2`, `fixed060_s2`)가 시작 직후 GPU 메모리 부족으로 실패해서 처음부터 다시 돌렸다.
    - 이후 워커가 GPU별 메모리 합계를 23 GB 아래로 유지하게 했다.
20. **워커 로그의 빈칸**: 워커를 세 번 교체하는 동안 이미 돌고 있던 job은 끝까지 돌았다. 다만 그 job들의 종료 기록(END)이 워커 로그에 없어서, run manifest의 종료 코드와 종료 시각이 일부 비어 있다. 이 run들의 결과 JSON과 provenance는 모두 있다.
21. **R5 표 재계산 방식**: Round 5 표 코드를 수정하지 않고, controller run의 경로만 R0으로 바꿔 실행했다(`scripts/r6_r0_tables.py`).
22. **지연 시간의 기기 값**: 기기 측정값이 아직 없어서 t_c, t_s(기기), 로컬 학습 시간에 host CPU 임시값을 썼다.

## 9. run manifest와 코드 식별

- **run manifest**: `tables/T10_run_manifest.csv`에 218개 run의 run_id, 시나리오, arm, seed, model_seed, 환경 파일, 완료 여부, 라운드 수, 평가 라운드 수, overlap, 장치, 시작·종료 시각, 실행 시간이 있다.
  - 218개 모두 complete = yes, overlap_ok = yes다.
  - 장치는 모두 `honeynaps, RTX 4090`이다.
  - run 하나의 실행 시간은 33–718분이다.
- **코드 식별(sha256 목록)**
  - 시작 전: `precheck/code_identity_before.txt`(git `ca49036`)
  - 실행 시작 시: `precheck/code_identity_launch.txt`(git `b8cf03d`)
  - 모든 run이 끝난 뒤: `precheck/code_identity_after.txt`(git `c99f0fb`)
  - 시작 시와 끝난 뒤 사이에 바뀐 실행 코드는 `--no_neighbor_avg` flag(2절)와 큐·워커 스크립트다.
  - 각 run의 provenance에는 그 run이 시작한 시점의 git commit이 기록되어 있다.
- **재생성 스크립트**
  - 환경: `journal_expansion/scripts/r6_make_env.py`, `r6_trace_env.py`
  - 환경 확인: `scripts/r6_env_check.py`
  - 표: `scripts/r6_tables.py`, `scripts/r6_r0_tables.py`
  - 지연 시간: `scripts/r6_server_timing.py`, `scripts/latency_calc.py`
  - 그림: `scripts/r6_figures.py`
  - 논문 숫자: `paper_numbers_r6.csv`(Round 6), `r0/tables/paper_numbers.csv`(Round 5 표의 R0 기준 값)
