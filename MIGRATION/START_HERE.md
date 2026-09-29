# 새 서버에서 시작하기

이 저장소는 2026-09-29에 옛 서버(`ubuntu20`, RTX 3090 Ti 2장)에서 GitHub로 옮겼다. 옛 서버에서는 GPU 1을 쓸 수 없어서 GPU 0 하나로 Round 5를 돌렸다. 사용자는 GPU가 더 많은 서버에서 작업을 이어가려 한다.

새 서버의 에이전트가 할 일은 두 가지다.

1. **이전 후처리**: 환경, 데이터, 경로, GPU 설정을 새 서버에 맞추고 동작을 확인한다 (1–7단계).
2. **Round 5 완수**: 남은 run을 마치고, 표와 그림을 다시 만들고, 한국어 최종 보고서를 쓴다 (8–10단계).

진행하면서 한 일과 결정을 `MIGRATION/MIGRATION_LOG.md`에 날짜와 함께 적는다(파일이 없으면 만든다).

## 0단계. 읽을 순서

1. 이 문서
2. `MIGRATION/PROJECT_CONTEXT.md`: 연구 내용, 경과, 반드시 지킬 규칙
3. `MIGRATION/ROUND5_DIRECTIVE.md`: 사용자의 Round 5 지시문 원문과 한국어 작성 지침
4. `MIGRATION/ROUND5_STATUS.md`: Round 5 진행 상황과 남은 run
5. `MIGRATION/CODEBASE_MAP.md`: 코드 구조, run JSON 형식, 큐 시스템, 분석 스크립트
6. 필요할 때: `SERVER_PATHS.md`(경로 표지와 치환 도구), `DATASETS.md`, `ENVIRONMENT.md`, `OLD_SERVER_STATE.md`, `claude_memory/`(옛 서버 에이전트의 메모리)

## 1단계. 사용자에게 먼저 확인할 것

아래 세 가지는 에이전트가 정할 수 없다. 한 번에 묻는다.

1. **GPU**: 새 서버에서 이 작업에 써도 되는 GPU 번호(nvidia-smi index)와, 다른 사용자와 나눠 쓰는지 여부. 옛 서버에서는 "GPU 1은 절대 쓰지 않는다"는 제약이 있었다.
2. **남은 run을 어느 서버에서 돌릴지**: 옛 서버는 저장소를 올린 뒤에도 R5를 계속 실행한다(`OLD_SERVER_STATE.md`). 선택지는 8단계에 있다.
3. **데이터 위치**: 데이터셋을 둘 경로(`DATA_ROOT`). 옛 서버에서 직접 복사할 수 있는지도 묻는다.

## 2단계. 위치 정하기와 메모리 가져오기

```bash
export REPO_ROOT=/path/to/driftgate        # git clone한 폴더 (이 문서가 있는 저장소의 루트)
export DATA_ROOT=/path/to/datasets
cd "$REPO_ROOT"
git log --oneline | head -3                # 옛 서버 이후 추가 commit(결과 동기화 등)이 있는지 확인
```

옛 서버 에이전트의 메모리를 새 서버의 Claude Code 메모리로 복사한다. 다음 세션부터 자동으로 불러온다.

```bash
SLUG=$(python -c "import re,sys; print(re.sub(r'[^A-Za-z0-9]', '-', sys.argv[1]))" "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)")
mkdir -p ~/.claude/projects/$SLUG/memory
find MIGRATION/claude_memory -name '*.md' ! -name README.md -exec cp -n {} ~/.claude/projects/$SLUG/memory/ \;
```

메모리 안의 `/disk2/Yujin/...` 경로는 옛 서버 기준이다. `/disk2/Yujin/adaptive_splitomc_tmc`를 `$REPO_ROOT`로 바꿔 읽는다. 자세한 내용은 `MIGRATION/claude_memory/README.md`에 있다.

## 3단계. 환경

`MIGRATION/ENVIRONMENT.md`대로 Python 3.12와 고정 버전 패키지를 설치한다.

```bash
pip install -r MIGRATION/requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4단계. 데이터셋

`MIGRATION/DATASETS.md`대로 CIFAR-10, CIFAR-100, SVHN, Tiny-ImageNet-200을 `$DATA_ROOT`에 받는다. md5와 표본 수를 확인하고, 저장소 루트에 `data_cache` 심볼릭 링크를 만든다.

## 5단계. 경로 치환과 코드 동일성 확인

```bash
python MIGRATION/tools/rewrite_server_paths.py --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"          # dry run
python MIGRATION/tools/rewrite_server_paths.py --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" --apply
python MIGRATION/tools/verify_code_identity.py  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"         # OK가 나와야 한다
grep -rn "/disk2/Yujin" --include=*.py --include=*.sh --include=*.yaml . | grep -v -e "^./journal_expansion/runs/" -e "^./MIGRATION/" -e "SERVER-PATH:EXTERNAL" | grep -v "^./logs/"
```

- 치환 도구는 "Old paths left in executable code: 0"을 출력해야 한다.
- 마지막 grep에서 나오는 줄은 docstring이나 주석이어야 한다. 실행되는 줄이 나오면 직접 고치고 `MIGRATION_LOG.md`에 적는다.
- `verify_code_identity.py`가 OK를 내면, 새 서버의 학습 코드는 R5 run이 실행한 코드와 같다(경로 줄만 다름).
- 치환 결과를 commit한다. commit 메시지에 새 서버 이름을 적는다.

`SERVER_PATHS.md`의 "수동으로 결정할 14줄"은 7단계에서 처리한다.

## 6단계. 테스트와 짧은 실행 확인

```bash
# 단위 테스트 75개 (CPU, 수 초). 옛 서버의 복사본에서 치환 후 75개 모두 통과했다
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py journal_expansion/tests -q -p no:cacheprovider

# GPU에서 relonly 2 라운드 실행 (허락받은 GPU 번호 사용)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<GPU> python -u journal_expansion/scripts/run_v2.py \
  --mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm --disjoint_pools \
  --schedule A --rounds 2 --probe_n 64 --seed 0 --model_seed 100 \
  --run_name smoke_relonly --output_dir journal_expansion/runs/_migration_check --device cuda:0
python -c "import json; h=json.load(open('journal_expansion/runs/_migration_check/smoke_relonly.json')); print(len(h['round']), h['probe_eval_overlap'], [round(e['acc_total'],4) for e in h['eval']])"
# 기대: 2 0 [...] (라운드 2개, overlap 0)
```

Tiny-ImageNet과 ResNet-18도 한 번씩 2 라운드로 돌려서 데이터 경로와 메모리를 확인한다(`--dataset tinyimagenet --schedule gradual_sigmoid`, `--model_family resnet18 --split_point middle --num_clients 16 --schedule A`). 확인용 run은 모두 `runs/_migration_check/`에 두고 표에는 넣지 않는다.

선택 사항: 새 GPU에서 결과가 얼마나 달라지는지 보려면, 옛 서버에서 끝난 run 하나(예: `b4_apfl_eta001_A_s0`, 약 4.5시간)를 같은 명령줄로 `runs/_migration_check/`에 다시 돌려 integrated accuracy를 비교한다. 이 차이는 보고서의 불일치 기록에 적을 수 있다. 사용자에게 시간을 쓸지 먼저 묻는다.

## 7단계. GPU와 워커 설정

1. `journal_expansion/scripts/supervisor.sh` 9–11줄의 기본값을 1단계에서 허락받은 값으로 고친다.
   - `GPUS`: 쓸 GPU 번호 목록(예: `"0 1 2 3"`)
   - `NHEAVY`, `NLIGHT`: GPU당 워커 수. 옛 서버(24 GB)에서는 heavy 3 + light 5를 썼고, GPU당 heavy 동시 실행은 `R5_HEAVY_MAX`=5로 제한했다. Tiny-ImageNet job은 약 3.8 GB, ResNet-18 약 2.5 GB, CIFAR/SVHN CNN 약 1.5 GB를 쓴다. 메모리가 다르면 조정한다.
   - 남은 run이 25개 이하이므로 GPU당 워커를 많이 둘 필요는 없다. 한 GPU에 job을 많이 올리면 job마다 느려진다.
2. cron이 올바른 python을 찾도록 crontab에 PATH를 넣는다. 워커는 `python3`, job은 `python`을 부른다.
   ```
   */10 * * * * PATH=/path/to/env/bin:/usr/bin:/bin /bin/bash <REPO_ROOT>/journal_expansion/scripts/supervisor.sh >> <REPO_ROOT>/journal_expansion/runs/queue_r5/logs/supervisor_cron.log 2>&1
   ```
3. `journal_expansion/artifacts/driftgate_tmc_final/scripts/r5_manifest.py:41`의 device 열을 고친다. 지금은 모든 run에 "cuda:0 = physical GPU 0"을 쓴다. 옛 서버 run과 새 서버 run이 구분되도록, 워커 로그의 GPU 표시(`cuda:0/GPU<N>`)와 `journal_expansion/provenance/<run_id>.json`의 `env.gpu`, 그리고 서버 이름을 적게 한다. 옛 서버 run은 `ubuntu20, RTX 3090 Ti, GPU 0`이다.

## 8단계. Round 5 남은 run 마치기

### 8.1 현재 상태 확인

```bash
git pull                                   # 옛 서버가 결과를 더 보냈을 수 있다
python - <<'EOF'
import os, re
miss = []
for q in ("heavy", "light"):
    for l in open(f"journal_expansion/runs/queue_r5/enqueued_{q}_snapshot.txt"):
        rn = re.search(r"--run_name (\S+)", l).group(1)
        od = re.search(r"--output_dir \S*/(runs/\S+)", l).group(1)   # 옛 절대 경로에서 runs/... 부분만 사용
        if not os.path.exists(f"journal_expansion/{od}/{rn}.json"):
            miss.append(rn)
print(64 - len(miss), "done;", len(miss), "missing:", *miss)
EOF
```

8.3에서 snapshot 파일을 `archive_ubuntu20/`로 옮긴 뒤에는 경로를 `journal_expansion/runs/queue_r5/archive_ubuntu20/enqueued_{q}_snapshot.txt`로 바꾼다.

`journal_expansion/scripts/r5_monitor.py`도 진행 상황을 보여 주지만, 실패한 run을 큐에 다시 넣는 동작이 있다. 그래서 8.3에서 큐를 새로 만든 뒤에만 실행한다.

### 8.2 어느 서버가 남은 run을 맡을지 정한다

1단계에서 사용자가 답한 대로 한다.

- **A. 옛 서버가 모두 마친다 (권장)**: R5 run이 모두 같은 하드웨어에서 나온다. 사용자가 옛 서버에서 `MIGRATION/tools/sync_results_from_old_server.sh`를 실행해 결과를 push하면, 새 서버에서 `git pull`하고 9단계로 간다. 새 서버에서는 R5 run을 돌리지 않는다.
- **B. 새 서버가 남은 run을 모두 맡는다**: 사용자가 옛 서버의 R5를 멈춘 경우다(`OLD_SERVER_STATE.md` 4.2). 옛 서버에서 끝났지만 아직 push하지 않은 결과가 있으면 먼저 sync를 요청한다. 그다음 8.3을 따른다.
- **C. 나눠서 맡는다**: 옛 서버가 실행 중인 run은 옛 서버가 마치고, 아직 시작하지 않은 run은 새 서버가 맡는다. 이 경우 사용자가 옛 서버 큐(`runs/queue_r5/{heavy,light}.txt`)에서 새 서버가 맡을 job을 지워야 한다. 두 서버가 같은 run을 돌리지 않는지 확인한 뒤 8.3을 따르고, 새 서버 큐에서도 옛 서버가 맡은 run을 지운다.

B나 C에서는 한 arm의 seed가 두 하드웨어로 나뉠 수 있다. 이 사실을 `MIGRATION_LOG.md`와 보고서의 불일치 기록에 적는다.

### 8.3 큐를 새로 만들고 워커 시작 (B, C의 경우)

저장소에 들어 있는 큐 파일은 옛 서버의 절대 경로로 된 명령줄이라 실행할 수 없다.

```bash
Q=journal_expansion/runs/queue_r5
mkdir -p $Q/archive_ubuntu20
git mv $Q/heavy.txt $Q/light.txt $Q/enqueued_heavy_snapshot.txt $Q/enqueued_light_snapshot.txt $Q/archive_ubuntu20/
python journal_expansion/scripts/enqueue_r5.py --snapshot-only   # 64개 명령줄을 새 경로로 기록 (r5_manifest.py가 사용)
python journal_expansion/scripts/enqueue_r5.py                   # JSON이 없는 run만 heavy.txt / light.txt에 추가
wc -l $Q/heavy.txt $Q/light.txt                                   # 남은 run 수와 맞는지 확인 (C라면 옛 서버 몫을 지운다)
rm -f $Q/STOP
bash journal_expansion/scripts/supervisor.sh                      # 워커 시작
ps -eo pid,ppid,args | grep "[r]5_worker"                         # PPID가 1인지 확인
crontab -e                                                        # 7단계의 cron 줄 추가
```

run이 도는 동안 몇 시간마다 `python journal_expansion/scripts/r5_monitor.py`로 확인한다. 종료 코드는 0 = 64개 완료, 1 = 진행 중, 2 = 두 번 실패한 run이 있거나 워커 없이 run이 남음이다. 실패 원인은 `grep -l Traceback $Q/logs/*.log`로 찾는다.

오래 걸리는 run의 시간(옛 서버, GPU 하나에 워커 8개): ResNet-18 9–19시간, mobility 약 3.5시간, Schedule A 약 4.5–6시간, CIFAR-100 gradual 약 5시간.

## 9단계. 표, 그림, 보고서

64개 run이 모두 끝나면(`r5_monitor.py` 종료 코드 0) 저장소 루트에서 실행한다.

```bash
cd journal_expansion/artifacts/driftgate_tmc_final
python precheck/precheck.py            # 시작 전 확인 재실행 (결과가 바뀌지 않아야 한다)
python scripts/r5_tables.py            # tables/T1–T7, T10, lambda_trajectories.csv, paper_numbers.csv
python scripts/r5_config_comm.py       # tables/T8, T9
python scripts/r5_manifest.py          # tables/run_manifest.csv  (7단계 3번 수정 후)
python scripts/r5_figures.py           # figures/fig1–fig5 (.pdf, .png)
cd -
```

확인할 것:
- `run_manifest.csv`: 64개 모두 `complete = yes`, `overlap_ok = yes`
- 모든 표에 빈칸이 남지 않았는지. 빈칸이 있으면 어떤 run이 빠졌는지 찾는다.
- `figures/figure_captions.md`의 수치가 새 표 값과 같은지. 다르면 캡션을 고친다(규칙: 엠대시, 세미콜론, "anti-correlates" 금지).
- 그림을 직접 열어 라벨 겹침, 내부 이름 노출(relonly, Schedule A 등)이 없는지 본다.

그다음 `journal_expansion/artifacts/driftgate_tmc_final/DriftGate_final_report_ko.md`를 쓴다. 구성은 `ROUND5_DIRECTIVE.md` 8절을 따른다.
- 첫 쪽: 주장 9개의 판정표 (주장 / 근거 표·그림 / 판정 / 핵심 수치)
- 시작 전 확인 두 가지의 결과 (`ROUND5_STATUS.md` 3절, 파일과 줄 번호 포함)
- 새 실험 결과, 기존 로그 분석 결과
- 그림 목록과 영문 캡션
- 불일치 기록: 서버 이전, 하드웨어가 섞인 run, 7단계에서 고친 코드, 기타 지시문과 다르게 처리한 것
- run manifest
- 코드 식별: `code_identity.txt`의 sha256, 옛 서버 git HEAD `e82b96b`(프로젝트 폴더는 untracked), 새 GitHub 저장소의 commit hash
- 한국어는 `ROUND5_DIRECTIVE.md` 부록의 fluent-korean 지침을 따른다. 불리한 결과도 그대로 적는다.

## 10단계. 결과 올리기와 보고

- run 결과, 표, 그림, 보고서를 commit하고 push한다.
- 사용자에게 보고서 위치와 판정표 요약을 한국어로 알린다.
- 옛 서버 정리(cron 삭제, STOP 파일)는 사용자가 요청할 때만 안내한다 (`OLD_SERVER_STATE.md` 4.2).
