# Round 6 작업 기록

보고서 `DriftGate_R6_report_ko.md`의 불일치 기록과 재현 정보의 근거가 되는 결정을 시간 순서로 적는다.

## 2026-09-30 시작 전

### 실행 위치와 코드
- 지시문 2절은 `/disk2/Yujin/adaptive_splitomc_tmc/`(옛 서버)의 코드를 쓰라고 적었다. 이 작업은 새 서버 `honeynaps`의 저장소 `/home/honeynaps/data/driftgate`(GitHub `MakerDev/driftgate`)에서 했다. 두 코드는 경로 줄만 다르다(`MIGRATION/MIGRATION_LOG.md`, `verify_code_identity.py` OK).
- 이 저장소는 git으로 추적되므로 sha256 목록과 함께 commit hash도 기록한다.
  - 시작 전: `precheck/code_identity_before.txt` (git HEAD `ca49036`, 기존 소스 20개)
  - 실행 시작 시: `precheck/code_identity_launch.txt` (기존 20개는 시작 전과 같고, 새 파일 12개 추가)
  - 모든 run이 끝난 뒤: `precheck/code_identity_after.txt` (예정)
- GPU: 사용자가 허락한 GPU 0–3(RTX 4090 24 GB × 4). 지시문의 "Round 5와 같은 설정"은 파일 큐, 워커, cron supervisor를 쓰는 방식으로 해석했다. 옛 서버의 "GPU 0만" 제약은 옛 서버의 사정이었다.

### 3.3 class group 확인 (결과를 보기 전에 한 번 판단)
- seeds 0–4에서 static topology의 cluster g 기본 구성원은 client 10g–10g+9이다. group 크기는 4–7개 class이고, 두 group이 공유하는 class는 0–5개(평균 2.4–4.2개)이다. 8개 이상의 class를 가진 group은 어느 seed에도 없다.
- 판단: group이 cell마다 구분되므로 기존 nd1 분할을 그대로 쓴다. {2g, 2g+1} 분할로 바꾸지 않는다.
- 전체 목록: `env_check/class_groups.csv`, `env_check/s1_partition_clients.csv`

### 정의를 구현하면서 정한 것
- **cell의 class 범위(scope)**: class group(그 cell 주민의 Main class 합집합)을 cell의 범위로 쓴다. OOP = scope(Z_k^t) − Main, OOR = all_used − Main − scope(Z_k^t). Z_k^t가 cell 두 개이면 두 group의 합집합이다. 기존 코드의 `es_scope`(ES마다 뽑은 40–70% class)는 쓰지 않는다. 지시문 1.2의 "현재 cluster의 다른 클라이언트가 학습한 class"를 cell 주민의 class로 해석했다(3.3의 "cell마다 class 범위" 표현과 맞춘 것).
- **요청 표본**: 기존 규칙(Main 표본 전부, OOP ρ배, OOR 0.3ρ배, class마다 pool의 앞쪽 표본)으로 요청 pool을 만들고, 그 가운데 n = min(N, 64)개를 비복원으로 뽑는다. 평가 요청 수와 규칙은 기존과 같다.
- **3.6의 빈 집합 규칙**: OOR 집합이 비면 OOR 몫을 OOP에서, OOP 집합이 비면 OOP 몫을 OOR에서 뽑는다. 빈도는 run마다 `probe_fallback`, `eval[].fallback`으로 기록한다.
- **일정**: 점심과 볼일의 귀가 이동은 도착 시각 + 머무는 시간에 출발한다. 그 시각이 다음 이동(퇴근) 이후이면 귀가 이동을 뺀다. 절단 정규분포는 기각 표본 추출로 뽑는다.
- **난수 흐름**: 네 흐름을 `SeedSequence([env_seed, 흐름 번호, layout])`로 나눴다. 속도, 요청률, 참여, 손실은 균등 난수(분위수)를 먼저 뽑고 조건별 범위로 바꾼다. 그래서 S1과 S1-fast는 같은 일정과 같은 속도 분위수를 쓰고, a=0.5 참여자는 a=0.7 참여자의 부분집합이며, p=0.1 손실은 p=0.3 손실의 부분집합이다.
- **S3 분할 seed**: 지시문은 "구역마다 seed·1000 + d"와 "구역 0의 분할은 S1과 같다"를 함께 적었다. seed 1, 2에서는 두 규칙이 충돌하므로, 구역 0은 S1과 같은 seed를, 구역 d ≥ 1은 seed·1000 + d를 쓴다.
- **S3 "가장 가까운 다른 구역"**: 구역 중심(hub) 사이 거리가 가장 짧은 구역이며, 같으면 균등하게 고른다.
- **빈 cell**: 참여 멤버가 없는 cell은 평균을 내지 않고 이전 평균을 유지한다. Λ 혼합의 전체 평균 θ̄_global은 그 라운드에 평균을 낸 cell들로만 계산한다.
- **아직 아무도 들어오지 않은 cell**: 처음 들어오는 클라이언트가 공통 초기 블록을 받도록, 모든 cell의 평균을 공통 초기 가중치로 둔다(기존 mobility 코드에서는 모든 cell이 첫 라운드에 평균을 내서 이 경우가 없었다).
- **참여하지 않은 클라이언트의 이동**: 모델을 받지 않으므로 server block을 바꾸지 않는다(이전 cell의 블록 유지). 다시 참여하는 라운드에 현재 위치로 rewire한다. 요청 구성(ρ, OOP/OOR)은 항상 현재 위치를 따른다.
- **신호 지연**: edge의 d̄을 d 라운드 늦게 controller에 넣는다(라운드 t−d의 소속으로 평균한 값). 처음 d 라운드에는 신호가 없어서 λ=0.425를 유지한다.
- **spatial 점수에 받은 값이 3개 미만**: 원래 controller와 같이 temporal 점수만 쓴다.
- **고정 λ, APFL arm**: 기존 관례대로 probe 신호를 계산하지 않는다(5.1의 d_k, q 등은 DriftGate와 entropy arm에만 있다).

### 구현
- 새 파일: `src/r6_env.py`(환경), `src/r6_requests.py`(분할, class group, 요청), `src/r6_controller.py`(edge별 DriftGate), `src/runner_r6.py`, `scripts/run_r6.py`, `scripts/r6_make_env.py`, `scripts/r6_trace_env.py`, 큐(`enqueue_r6.py`, `r6_worker.py`, `supervisor_r6.sh`, `r6_monitor.py`), `tests/test_r6.py`. 기존 파일은 고치지 않았다.
- 확인(`tests/test_r6.py`, 18개 통과):
  - edge별 controller는 메시지를 모두 받으면 원래 `SelfCalController`와 λ, Λ, q가 bit 단위로 같다(선형, star, 완전 그래프, seed 3개).
  - 모두 참여하고 빈 cell이 없으면 학습 라운드가 `run_one_global_round`와 bit 단위로 같다(CPU).
  - 평가 함수가 `evaluate_one_client`와 같은 정확도를 낸다.
  - 학습 데이터 캐시(`CachedLoader`): 같은 DataLoader/RandomSampler를 index 목록에 돌리고, 미리 변환한 tensor에서 batch를 꺼낸다. 2 라운드 뒤 weight와 torch 난수 상태가 원래 경로와 bit 단위로 같다. 속도는 약 2.2배 빨라졌다.
- 추가 패키지: scikit-learn 1.9.1(S2 k-means). numpy 2.5.1은 바뀌지 않았다.

### S2 데이터
- GeoLife Trajectories 1.3을 Microsoft 배포 주소에서 받았다(313,164,406 bytes, sha256 `1107c5ac…86bdb6`). `data/traces` → `/home/honeynaps/data/driftgate_datasets/traces` (git 제외).
- 점 24,876,978개 → 필터 뒤 18,487,422개, 후보 사용자 135명 ≥ 50 → GeoLife를 쓴다(T-Drive 쓰지 않음).
- 선택 50명의 점수 150–77. k-means edge의 cell별 주민 수 [38, 4, 5, 2, 1]로 치우쳐 있다(대부분 하이뎬 지역). 빈 cell은 없다.
- GeoLife 안의 중복: 사용자 112와 163의 2008-06-20 기록이 같은 기록 수(6,399개)로 둘 다 선택되었다. 지시문에 중복 처리 규칙이 없어서 그대로 둔다.
- **S2에서 모든 edge 쌍이 이웃이면**, 1.3의 4단계(이웃과 한 번 평균)가 모든 cell의 점수를 같은 값(전체 평균)으로 만든다. 따라서 S2에서 DriftGate는 모든 cell에 같은 λ를 준다. 지시문대로 실행하고 보고서에 적는다.

### 3.9 환경 확인 요약 (`env_check/`)
- S1: 집에 있는 비율 1.0 → 낮 0.35 → 1.0, hub 인원 최대 약 31명, cell 평균 ρ 0.1 → 0.45–0.6 → 0.1. Main이 아닌 요청 비율은 집에서 약 11.5%. n < 64 비율 34%, n = 0 없음. OOR 대체 3.1%, OOP 대체 0%.
- S1-fast: 같은 일정, 소속 변화가 짧게 끝난다.
- S2: 집에 있는 비율은 낮에도 약 0.6, 중심 cell의 평균 ρ는 0.1–0.22.
- S3: K=200 소속 변화 라운드당 5.7명, K=500 14.4명.

### 라운드당 시간과 예상 시간 (`precheck/round_time.csv`)
- RTX 4090 한 장에서 혼자 돌 때 라운드당 K=50 9.7초, K=200 37.3초, K=500 92.1초(평가 포함).
- 전체 151 run의 합: 약 116 GPU-시간(혼자 기준). 한 GPU에 K=50 run 6개를 함께 올리면 run마다 4.05배 느려지고 처리량은 1.48배이다. GPU 4장, GPU당 워커 4개로 약 20시간(완전히 채웠을 때), K=500의 꼬리를 넣으면 24–30시간으로 예상한다.

## 2026-09-30 실행 중

- 22:18 151개 run 투입. GPU 0–3, GPU당 워커 4개, K=500 동시 실행은 GPU당 2개, K=200은 3개로 제한. cron supervisor(10분)로 워커를 되살린다. 실행 코드는 commit `b8cf03d`.
- 23:28–23:32 첫 16개(S1 seed 0, 1의 8개 arm) 완료, 모두 exit 0, overlap 0, run당 약 68–70분.
- 표 스크립트를 부분 결과로 점검했다. 5.2의 반응·회복 시간 정의에서는, 변화 시작 시점에 λ가 이미 0.425 이하이면 반응 시간이 0이 되고 회복 시작 시점에 이미 0.425를 넘으면 회복 시간이 0이 된다. 해석을 돕도록 T5a에 두 시점의 λ와 라운드 25 이후 최소 λ 열을 추가했다(정의는 바꾸지 않음).
- 6.3 모델 파일: S1 DriftGate seed 0의 마지막 모델에서 client 0의 client block + exit와 hub cell의 server block + exit를 내보냈다(`device/models/`). TorchScript는 원래 모델과 출력이 같고(차이 0), lite interpreter는 최대 7.6e-6, ONNX(onnxruntime 1.30.0, batch 1과 64)는 최대 6.4e-6 차이다. 확인을 위해 onnxruntime 1.30.0을 가상 환경에 추가했다.

## 2026-10-01 00:40 사용자 결정: 이웃 점수 평균을 뺀다

사용자가 DriftGate에서 이웃 점수 평균을 빼기로 결정했다(결과를 보기 전의 결정, 새 정의를 논문의 방법으로 쓴다).

### 확인한 코드 (보고서에 적을 것)
- 이웃 평균의 식: `journal_expansion/src/controllers/self_calibrating.py:63–71` `_consensus(vals, neighbors, steps)`:
  new[z] = (v_z + Σ_{w ∈ N(z), w가 이번 라운드에 값이 있음} v_w) / (1 + |{그런 w}|), steps번 반복.
  점수에는 `:164` `zc = _consensus(z, self.neighbors, self.consensus_steps)`로 적용되고, 그 결과가 q(`last_z`)이다.
  steps = `configs/base_v3.yaml:48` `consensus_steps: 1` → `runner.py:247`.
- 이웃 구조: `journal_expansion/src/runner.py:217–223`. `--topology`를 주지 않으면 `data/partition.py:189–199` `es_neighbors_sequential`(선형 0–1–2–3–4)이다. Round 4·5의 모든 run은 `--topology`를 쓰지 않았다.
- Round 6 구현: `journal_expansion/src/r6_controller.py:75–77`(같은 식), 이웃 구조는 `r6_env.py:93–97` `neighbor_matrix`(중심 거리 ≤ 2 km, S2는 모든 쌍) → `runner_r6.py:257–258`.
- entropy arm: 같은 `SelfCalController`에 `--signal ent_client`만 다르므로 `:164`를 거친다.
- absonly: `--abs_only`는 `abs_cap`도 켠다(`run_v2.py:156–158`). λ를 정하는 absolute 경로가 `:170–171`에서 raw 신호 d̄에 같은 이웃 평균을 적용한 뒤 EMA로 평활한다. 따라서 absonly도 이웃 평균을 거친다 → 지시대로 absonly 11 runs도 다시 돌린다.

### 구현
- `SelfCalController(neighbor_avg=True)`: False이면 `zc = dict(z)`, absolute 경로는 `sig_c = dict(signal_per_es)`. 기본값의 동작은 그대로다.
- `run_v2.py --no_neighbor_avg` → `runner.py`가 `neighbor_avg=False`로 넘기고 결과 JSON `config.no_neighbor_avg = True`에 기록한다.
- `EdgeDriftGate(neighbor_avg=False)`: q = 자기 점수. 이웃 점수 메시지가 없으므로 S4 손실 (3)은 쓰지 않는다. `run_r6.py --no_neighbor_avg`.
- 테스트(`tests/test_r6.py`) 3개 추가: 두 controller가 새 정의에서 bit 단위로 같고 이웃 구조와 무관함, 기본값은 여전히 평균함, absonly가 자기 신호의 EMA를 씀. 전체 98개 통과.
- 이 변경은 실행 중인 프로세스에 영향을 주지 않는다. 이후 시작하는 고정 λ/APFL run은 selfcal 경로를 쓰지 않으므로 동작이 같다.

### 실행 계획 변경 (큐 v2, `scripts/enqueue_r6_v2.py`, `runs/queue_r6/enqueued_snapshot_v2.txt`, 218 runs)
- 00:40 구 정의 DriftGate/entropy 36줄을 큐에서 빼서 `runs/queue_r6/held_old_definition.txt`에 보관했다.
- 이미 끝났거나 거의 끝난 구 정의 run(S1 seed 0–3의 DriftGate, entropy)은 기록으로 남기고 표에는 넣지 않는다. 막 시작한 `s1_driftgate_s4`, `s1_entropy_s4`는 PID로 멈췄다(provenance status가 running으로 남음).
- 순서: P0 67 → R6 P1 고정 λ/APFL → R6 P1 DriftGate/entropy(새 정의, arm 이름 `driftgate_own`, `entropy_own`) → R6 P2(표 순서). P0 안에서는 오래 걸리는 run을 앞에 둔다(Tiny-ImageNet, ResNet-18, SVHN, ...).
- P0: Round 4·5 원래 명령줄(provenance 기록과 R5 snapshot)에 `--no_neighbor_avg`만 더했다. `run_v2.py` 경로, seed, rounds, disjoint pool이 같다. 저장 위치 `runs/phaseT6_p0/`, 이름 `p0_<원래 이름>`. absonly의 원래 명령에는 `--spatial_norm`이 없고, 그대로 따른다.
- 주의: P0는 RTX 4090(honeynaps)에서, 비교 대상인 원래 run은 RTX 3090 Ti(ubuntu20)에서 돌았다. 이웃 평균이 있을 때와 없을 때의 paired 차이에는 하드웨어 차이(부동소수점 연산 순서)가 함께 들어간다.
- R6: 3.2의 이웃 규칙과 S2 전처리 8번의 이웃 규칙은 controller에 쓰지 않는다(env 파일의 `neighbors` 배열은 남아 있지만 읽지 않는다). S4 신호 손실은 (1) 클라이언트 TV와 (2) d̄ 공유만 대상이다. 추가 통신량에서 이웃 교환 항목을 뺀다.
