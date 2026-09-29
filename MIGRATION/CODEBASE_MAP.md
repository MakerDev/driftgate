# 코드베이스 지도

저장소는 두 층으로 되어 있다.

- **루트 (v3/v4 학습 스택)**: ICTC 2026 논문(Adaptive-SplitOMC) 때 만든 SplitOMC 재현 코드다. 데이터 분할, 모델, 학습, 평가를 담당한다. journal 확장 작업에서도 이 코드를 그대로 import해서 쓰며, 내용은 고치지 않았다.
- **`journal_expansion/` (TMC 확장, DriftGate)**: 2026-07-11 이후의 모든 작업이 여기에 있다. 신호 라이브러리, controller, 확장된 실험 루프(runner), 큐 시스템, 분석 스크립트, run 기록, 보고서가 들어 있다.

## 1. 실행 흐름

```
journal_expansion/scripts/run_v2.py          CLI. 인자를 해석해서 run_experiment_v2() 호출
  └─ journal_expansion/src/runner.py          run_experiment_v2(): 한 run 전체
       ├─ data/partition.py                   ND1 분할, edge server 토폴로지(50 clients, 5 ES, overlap 50%)
       ├─ journal_expansion/src/datasets_ext.py  CIFAR-100/SVHN/Tiny-ImageNet 등 (--dataset 줄 때)
       ├─ models/architectures.py             기본 CNN 분할 모델 (Client 280K / Server 1.38M params)
       ├─ journal_expansion/src/models_ext.py    ResNet-18 분할(--model_family resnet18 --split_point ...)
       ├─ journal_expansion/src/schedules.py     ρ(t) 스케줄: A, abrupt, gradual_sigmoid, recurring, ...
       ├─ journal_expansion/src/disjoint_pools.py  test 표본을 class별로 controller 20% / 평가 80%로 분리
       ├─ journal_expansion/src/signals/library.py  probe 64개로 client/server exit 신호 계산 (tv_dist 등 16종)
       ├─ journal_expansion/src/controllers/self_calibrating.py  SelfCalController (DriftGate controller)
       │    └─ controllers/normalizers.py     GuardedRobustNormalizer (temporal z, β=0.05, guard, MAD σ)
       ├─ journal_expansion/src/apfl_baseline.py  --mode apfl (B4 기준 방법)
       ├─ train/trainer.py                    SplitOMCClient, EdgeServer, 라운드 학습/집계, λ·Λ 혼합
       ├─ network/mobility.py                 Gauss-Markov 이동, client 재배치(rewire)
       ├─ eval/evaluator.py                   entropy routing(eth=0.8) 평가, Main/OOP/OOR 정확도
       └─ journal_expansion/src/provenance.py   run_id, git hash, config SHA, GPU 이름 등 기록
```

한 라운드의 순서는 traffic(ρ) 결정 → 신호 계산(직전 라운드 끝 모델, causal) → controller가 λ, Λ 결정 → 학습과 집계 → 평가(10 라운드마다, 1 라운드 포함)이다.

## 2. run_v2.py의 주요 모드와 인자

| 인자 | 의미 |
|---|---|
| `--mode fixed --lambda_val L --big_lambda_val 0.5` | 고정 λ, Λ |
| `--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm` | **DriftGate(relonly)**. `--abs_cap`를 주지 않으면 absolute branch가 꺼진다 |
| `--mode selfcal ... --abs_cap` | full controller(absolute branch 포함). 최종 보고서에서 제외 |
| `--mode selfcal ... --abs_only` | absonly(TV 크기로 λ를 바로 정함). B1 비교용 |
| `--mode selfcal --signal ent_client ...` | entropy를 신호로 넣은 같은 controller |
| `--mode apfl --apfl_eta E --big_lambda_val 0.5` | APFL식 λ_k 학습 (B4) |
| `--mode passive` | 고정 λ=0.4로 학습하면서 모든 신호만 기록(Fig. 1의 `rec_*` run) |
| `--lam_min`, `--lam_max` | λ 범위 (기본 0.15, 0.70). E3의 λ_max 민감도 |
| `--schedule A` | stepwise composition change (150 rounds, ρ 0→0.4→0.8→0.4→0) |
| `--mobility --mobility_speed_level med --schedule abrupt --rounds 120` | client mobility (ρ가 라운드 61에 0→0.8) |
| `--dataset cifar10\|cifar100\|svhn\|tinyimagenet --schedule gradual_sigmoid` | transfer 설정 |
| `--model_family resnet18 --split_point middle --num_clients 16` | ResNet-18 middle split (E2) |
| `--disjoint_pools` | probe/평가 pool 분리. **모든 새 run에 필수** (overlap=0 assert) |
| `--probe_n 64 --seed S --model_seed 100+S` | R5 공통 설정 |

전체 인자는 `python journal_expansion/scripts/run_v2.py -h`로 본다. R5의 정확한 명령줄은 `journal_expansion/scripts/enqueue_r5.py`가 만든다.

## 3. run 결과 JSON (`<output_dir>/<run_name>.json`)

run이 **끝날 때 한 번만** 저장한다. 중간 checkpoint가 없으므로, 도중에 죽은 run은 JSON이 없고 처음부터 다시 돌려야 한다.

| 키 | 내용 |
|---|---|
| `round`, `mean_loss`, `rho_trace` | 라운드별 번호, 학습 손실, ρ |
| `lamdas`, `big_lamdas` | 라운드별 cluster(ES)별 λ, Λ (`{"0": 0.425, ...}`) |
| `controller_z` | 라운드별 cluster별 평활 점수 q |
| `lam_rel`, `lam_abs` | R4부터 기록. relonly에서는 `lam_abs`가 비어 있다 |
| `signals_per_es` | 신호 이름별, 라운드별 cluster 평균 신호 |
| `eval` | 평가 라운드 목록. 각 항목에 `acc_total`, `acc_main/oop/oor`, `worst_cell_acc`, `per_client_acc`, `routing`, `rho` 등 |
| `pool_manifest`, `probe_eval_overlap`, `probe_used_count`, `eval_used_count` | disjoint pool 확인값. overlap은 0이어야 한다 |
| `config` | method, λ, schedule, controller 모드/신호, seed, `run_id` |
| `total_time_sec` | 실행 시간 |

`integrated accuracy` = `eval`의 `acc_total` 평균이다. 모든 표가 이 정의를 쓴다.

run마다 `journal_expansion/provenance/<run_id>.json`과 `provenance/all_runs.jsonl` 한 줄이 추가된다. 같은 설정을 다시 돌리면 새 run_id가 생기고, 결과 JSON은 절대 덮어쓰지 않는다.

## 4. 큐 시스템 (Round 5)

| 파일 | 역할 |
|---|---|
| `journal_expansion/scripts/enqueue_r5.py` | R5의 64개 job 명령줄을 만들어 `runs/queue_r5/{heavy,light}.txt`에 추가한다. JSON이 이미 있거나 큐에 같은 줄이 있으면 건너뛴다. `--snapshot-only`는 64줄 전체를 `enqueued_*_snapshot.txt`에 쓰고 큐는 건드리지 않는다 |
| `journal_expansion/scripts/r5_worker.py <heavy\|light> <id>` | flock으로 큐에서 한 줄을 꺼내 실행한다. heavy 동시 실행 수를 `R5_HEAVY_MAX`(기본 5)로 제한한다. `runs/queue_r5/STOP`이 있으면 멈춘다. 로그는 `runs/queue_r5/logs/{worker<id>,<run_name>}.log` |
| `journal_expansion/scripts/supervisor.sh` | cron이 10분마다 실행한다. 큐에 job이 있는데 워커가 없으면 워커 묶음을 다시 띄운다 |
| `journal_expansion/scripts/r5_monitor.py` | 진행 상황 요약 |

heavy는 Tiny-ImageNet(약 3.8 GB, run당 약 20시간)과 ResNet-18(run당 9–19시간) job이고, light는 CIFAR/SVHN CNN job(약 1.5 GB, run당 3.5–7.5시간)이다. 시간은 옛 서버에서 GPU 하나에 8개 워커가 함께 돌 때의 값이다.

Round 4 이전의 큐(`runs/queue/`, `queue_worker.sh`, `resnet_worker.sh`, `timed_worker.sh`, `enqueue_r4.py` 등)는 기록으로만 남아 있다. 다시 쓰지 않는다.

## 5. 분석 스크립트와 산출물 (Round 5)

모두 `journal_expansion/artifacts/driftgate_tmc_final/`에 있다.

| 순서 | 스크립트 | 만드는 것 |
|---|---|---|
| 0 | `precheck/precheck.py` | 시작 전 확인 1·2: entropy와 relonly run의 λ, Λ를 controller 코드로 다시 계산해서 기록값과 비교 → `precheck_controller_path.csv`. `code_identity.txt`는 R5 코드의 sha256 |
| 1 | `scripts/r5_tables.py` | raw JSON → `tables/T1`–`T7`, `T10`, `lambda_trajectories.csv`, `paper_numbers.csv`. 빠진 run이 있는 arm은 빈칸으로 둔다 |
| 2 | `scripts/r5_config_comm.py` | `tables/T8_reproduction_settings.csv`, `T9_communication.csv` (config/코드의 파일:줄 출처 포함) |
| 3 | `scripts/r5_manifest.py` | `tables/run_manifest.csv` (R5 run 64개의 완료 여부, overlap, 시작/종료 시각) |
| 4 | `scripts/r5_figures.py` | `figures/fig1`–`fig5` (PDF + 300 dpi PNG). 캡션은 `figures/figure_captions.md` |
| 5 | (미작성) | `DriftGate_final_report_ko.md` — R5 지시문 8절 |

`r5_tables.py`는 R5의 새 run뿐 아니라 이전 라운드의 run도 읽는다. 예를 들면 `runs/phaseT1_disjoint`(고정 λ, entropy), `runs/phaseT3_fixedref`(transfer 고정 λ), `runs/phaseT4_B1_views`(relonly, absonly), `runs/signal_benchmark/rec_*`(passive), `runs/phaseR_role`, `runs/phaseC_signals`가 있다. 그래서 `journal_expansion/runs/` 전체를 저장소에 포함했다.

## 6. 폴더 요약

```
.
├── CLAUDE.md, README.md          새 서버용 안내 (옛 v3 문서는 docs/legacy_v3_ictc/)
├── MIGRATION/                    서버 이전 문서와 도구 (이 파일)
├── configs/base_v3.yaml          기본 하이퍼파라미터 (R5 run이 그대로 씀)
├── data/ models/ train/ eval/ network/   v4 학습 스택 (journal 작업이 import)
├── controller/ scripts/ run_all.sh tests/test_all.py   ICTC 시절 코드 (R5와 무관, 테스트는 유지)
├── results/ docs/ figure/ logs/  ICTC 시절 결과와 기록 (고정)
└── journal_expansion/
    ├── README.md                 journal 확장 초기 안내 (Gate A/B 시절)
    ├── src/                      runner, signals, controllers, baselines, evaluation, network, ...
    ├── scripts/                  run_v2.py, enqueue_*.py, 워커, 분석 스크립트 (라운드별)
    ├── tests/                    test_journal.py, test_r4.py 등 (루트 tests/test_all.py 포함 75개 통과)
    ├── configs/                  phase별 설정 기록
    ├── runs/                     모든 run 결과 JSON, 신호 npz, 큐, 로그 (기록, 수정 금지)
    ├── provenance/               run별 provenance (기록, 수정 금지)
    ├── reports/                  phase/라운드별 보고서 (기록)
    ├── tables/ figures/          초기 phase의 표와 그림
    ├── exported/                 on-device 벤치마크용 모델 export
    └── artifacts/
        ├── driftgate_tmc_export/        2026-09-14 export (R3)
        ├── driftgate_tmc_export_r4/     Round 4 (RESUME_NOTE.md 포함)
        ├── DriftGate_TMC_최종패키지/    패키지본 (+ .zip)
        └── driftgate_tmc_final/         **Round 5 최종 산출물 위치**
```
