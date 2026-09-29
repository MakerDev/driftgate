# 서버 이전 기록

새 서버의 에이전트는 한 일과 결정을 날짜와 함께 아래에 이어서 적는다.

## 2026-09-29 — 옛 서버(ubuntu20)에서 한 일

- live checkout(`/disk2/Yujin/adaptive_splitomc_tmc`)은 R5가 돌고 있어서 고치지 않았다. 대신 `/disk2/Yujin/driftgate_github`로 복사하고, 이 복사본에서 작업하고 commit했다.
- 제외한 것: `data_cache/`, `journal_expansion/data_cache/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `.claude/`, 큐의 `queue.lock`, `running_heavy/`. 데이터셋은 모두 다시 받을 수 있어서 올리지 않았다 (`DATASETS.md`).
- 서버 의존 줄 97개를 찾았다. 그중 실행 코드 85줄에 `[SERVER-PATH:*]`, `[SERVER-GPU]` 표지를 붙였다. 표지를 붙인 뒤 `.py`는 AST가 같은지, `.sh`는 `bash -n`을 통과하는지 확인했다.
- 옛 문서(`CLAUDE.md`, `README.md`, `CLAUDE_CODE_PROMPT.md`, `EXECUTE_ALL.md`)는 `docs/legacy_v3_ictc/`로 옮겼다. 루트의 `CLAUDE.md`와 `README.md`는 새로 썼다.
- `r5_worker.py`와 `supervisor.sh`가 GPU 목록과 GPU당 워커 수를 변수로 받도록 고쳤다. 기본값은 옛 서버의 동작(GPU 0, heavy 3 + light 5)과 같다.
- 확인한 것. 모두 옛 서버에서 새로 clone한 복사본으로 확인했다.
  - 경로 치환: 71줄 바꿈, 남은 옛 경로 0
  - 코드 동일성: R5 소스 17개 모두 일치
  - 단위 테스트: 75개 모두 통과 (CPU)
  - GPU 0 확인 실행: relonly 2 라운드, overlap 0, acc 0.279 → 0.300
  - 큐 재생성: 남은 25개만 새 경로로 큐에 들어감, `r5_manifest.py` 39/64 완료 인식
- 저장소에 올린 시점의 R5 진행: 39/64 완료. 옛 서버에서 8개 실행 중, 17개 대기 (`ROUND5_STATUS.md`).

## 2026-09-29 — 새 서버(honeynaps)에서 한 일

### 서버 정보
- 호스트 `honeynaps`, Ubuntu 22.04.4, Linux 5.15.0-191, CPU 48코어, RAM 125 GB
- GPU: NVIDIA GeForce RTX 4090 24 GB × 4 (index 0–3), driver 580.178.04 (CUDA 13.0 지원)
- 저장소 루트 `REPO_ROOT=/home/honeynaps/data/driftgate`, 시작 commit `74a976d` (origin/main과 같음)
- 다른 프로젝트도 쓰는 공용 계정이다(홈 폴더에 다른 연구 데이터가 있음)

### 1단계. 사용자 답
- GPU: 처음 받은 답의 GPU 항목이 틀(`<번호들, 예: 0,1,2,3>`, `<예/아니오>`) 그대로여서 다시 물었다. 답: **GPU 0, 1, 2, 3 모두 사용 가능.** 다른 사람과 나눠 쓰는지는 답이 없었다. 물었을 때 4장 모두 비어 있었다.
- 6단계 선택 사항(옛 서버 run 하나를 새 GPU에서 다시 돌려 비교): 사용자 답 **하지 않음**.
- 남은 R5 run: **8.2의 A안**. 옛 서버가 모두 마치고 결과를 GitHub로 보낸다. 새 서버에서는 R5 run을 돌리지 않는다. 결과가 올라오면 9단계부터 한다.
- 데이터셋: 옛 서버에서 직접 복사할 수 없다. 공개 주소에서 다시 받는다.

### 2단계. 메모리
- `MIGRATION/claude_memory/*.md`(README 제외) 9개를 `~/.claude/projects/-home-honeynaps-data-driftgate/memory/`로 복사했다(`cp -n`, 기존 파일 없음).

### 3단계. 환경
- 서버에 conda가 없고 시스템 Python이 3.10.12여서, `pip3 install --user uv`(uv 0.12.20)로 CPython 3.12.7을 받고 가상 환경 `~/venvs/driftgate`를 만들었다.
- `uv pip install --index-strategy unsafe-best-match -r MIGRATION/requirements-lock.txt`로 설치했다. 버전이 옛 서버와 모두 같다: torch 2.7.1+cu126, torchvision 0.22.1+cu126, numpy 2.5.1, scipy 1.16.3, matplotlib 3.11.0, PyYAML 6.0.1, pytest 8.4.2, tqdm 4.66.5, pillow 10.4.0 (onnx 1.22.0도 설치).
- 드라이버 580.178.04가 cu126 wheel을 지원하므로 다른 CUDA wheel로 바꾸지 않았다. `torch.cuda.is_available()=True`, device 4개, cuDNN 90501.
- 가상 환경 안에 `python`과 `python3`가 모두 있다. 명령은 `source ~/venvs/driftgate/bin/activate` 뒤에 실행한다.

### 4단계. 데이터셋
- `DATA_ROOT=/home/honeynaps/data/driftgate_datasets`로 정했다. 사용자가 위치를 정해 주지 않아서, 저장소 옆(`/home/honeynaps/data`)에 다른 프로젝트 폴더와 겹치지 않는 이름으로 만들었다. 남은 공간 약 270 GB.
- Tiny-ImageNet-200: cs231n 주소에서 받았다. zip md5 `90528d7ca1a48142e341f4ef8d21d0de`, 248,100,043 bytes로 기록과 같다. 서버에 `unzip`이 없어서 Python `zipfile`로 풀었다. train JPEG 100,000개, val 이미지 10,000개.
- CIFAR-10/100과 SVHN은 torchvision으로 차례로 받으면 너무 느려서(toronto.edu에서 초당 약 90 KB), torchvision을 멈추고 wget으로 네 파일을 동시에 받았다. 받은 뒤 torchvision이 md5를 확인하고 압축을 푼다.
- md5는 모두 `DATASETS.md`와 같다: `cifar-10-python.tar.gz` c58f3010…, `cifar-100-python.tar.gz` eb9058c3…, `train_32x32.mat` e26dedcc…, `test_32x32.mat` eb5a983b…. CIFAR는 toronto.edu가 `cave.cs.toronto.edu`로 넘겨 주는 같은 파일이다(170,498,071 / 169,001,437 bytes).
- torchvision `download=True`를 다시 실행해서 md5를 확인하고 압축을 풀었다(다시 받지 않음).
- `ln -sfn $DATA_ROOT/cifar10 data_cache` (`.gitignore` 대상).
- `get_dataset()` 표본 수: cifar10 50,000/10,000, cifar100 50,000/10,000, svhn 73,257/26,032, tinyimagenet 100,000/10,000. 모두 기록과 같다.

### 5단계. 경로 치환과 코드 동일성
- `rewrite_server_paths.py --repo-root /home/honeynaps/data/driftgate --data-root /home/honeynaps/data/driftgate_datasets`: dry run 71줄, `--apply` 71줄 치환. "Old paths left in executable code: 0". GIT_ROOT는 기본값(저장소 루트)을 썼다.
- `verify_code_identity.py`: "OK: all files match the Round-5 code" (17개 파일, `datasets_ext.py`만 표지·경로를 되돌린 뒤 일치).
- 옛 경로 grep: `MIGRATION/` 밖의 실행 파일에서 0줄. `MIGRATION/tools/`에 남은 옛 경로는 도구의 기본값과 옛 서버에서 돌릴 sync 스크립트이므로 그대로 둔다. (이 서버의 `grep -r`은 경로 앞에 `./`를 붙이지 않아서 START_HERE의 `grep -v "^./..."` 필터가 걸러 내지 못한다. `sed 's|^\./||'`로 맞춘 뒤 확인했다.)
- 수동 결정 14줄은 7단계에서 다룬다.

### 6단계. 테스트와 짧은 실행 확인
- `CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py journal_expansion/tests -q -p no:cacheprovider`: 75 passed (2.74 s).
- GPU 확인 실행 3개. 모두 relonly 플래그, `--disjoint_pools --rounds 2 --probe_n 64 --seed 0 --model_seed 100`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 실행했고 결과는 `journal_expansion/runs/_migration_check/`에 있다(표에 넣지 않음).

  | run | GPU | 라운드 | overlap | acc_total (R1, R2) | 시간 | 같은 seed의 옛 서버 run (R1 acc / loss R1, R2) | 새 서버 loss R1, R2 |
  |---|---|---|---|---|---|---|---|
  | `smoke_relonly` (CIFAR-10 Schedule A) | 0 | 2 | 0 | 0.2798, 0.3002 | 84 s | `b1_relonly_A_s0`: 0.2795 / 0.8837, 0.5726 | 0.8838, 0.5726 |
  | `smoke_tiny_relonly` (Tiny-ImageNet gradual) | 1 | 2 | 0 | 0.0079, 0.0201 | 274 s | `e1_relonly_tiny_s0`: 0.0081 / 4.4805, 4.0875 | 4.4806, 4.0865 |
  | `smoke_resnet_relonly` (ResNet-18 middle, 16 clients, A) | 2 | 2 | 0 | 0.4739, 0.2680 | 101 s | `e2_res_relonly_A_s0`: 0.4730 / 0.4034, 0.2766 | 0.4029, 0.2772 |

  - 옛 서버의 확인 실행(0.279 → 0.300)과도 맞는다. 차이는 R1 acc 0.001 이하, loss 0.1% 안팎이다. GPU(RTX 4090 대 3090 Ti)가 달라서 bit 단위로 같지는 않다.
  - ResNet-18의 R2 acc 0.268은 R1(0.474)보다 낮다. 옛 run은 R2에서 평가하지 않아서(평가 라운드 1, 10, 20, …) 직접 비교할 수 없다. loss가 옛 run과 맞으므로 이전 문제로 보지 않는다. 옛 run은 R10에서 0.618이었다.
  - provenance: 세 run 모두 `env.gpu = NVIDIA GeForce RTX 4090`, `git_commit = fcd4f14…`(새 저장소 HEAD. GIT_ROOT 치환이 동작함), `status = completed`, `global_overlap = 0`. `provenance/all_runs.jsonl`에 세 줄이 추가되었다.
- GPU 공유: 확인 실행 중(21:44경)에 같은 계정의 다른 작업(`/home/honeynaps/data/shared/integrate_shared_ver4/somnum_release/device_verification_260929/scripts/run_pipeline_dump.py`)이 GPU 0–2에 프로세스당 약 2.2 GB로 올라왔다. 이 서버의 GPU는 다른 작업과 함께 쓰는 상황이다. 그 프로세스는 건드리지 않았다.

### 7단계. GPU와 워커 설정
- 7.1 `supervisor.sh` 기본값: GPU 답을 받은 뒤 `GPUS="0 1 2 3"`, GPU당 heavy 2 + light 4로 바꾸고 주석에 새 서버 정보를 적었다(`bash -n` 통과). GPU당 메모리는 최대 2×3.8 + 4×1.5 ≈ 13.6 GB로 24 GB 안이다. 전체 24 슬롯이면 남은 25개를 거의 동시에 돌릴 수 있다. A안에서는 이 서버에서 워커를 띄우지 않으므로, 계획이 B·C로 바뀔 때만 쓰는 값이다.
- 7.2 cron: **넣지 않았다.** A안이라 이 서버에서 되살릴 워커가 없다.
- 안전장치: `journal_expansion/runs/queue_r5/STOP`을 만들었다(`.gitignore` 대상이라 옛 서버로 전파되지 않음). 누가 `supervisor.sh`를 실행해도 워커가 뜨지 않는다.
- 7.3 `r5_manifest.py` device 열: 고정 문자열 "cuda:0 = physical GPU 0 (CUDA_VISIBLE_DEVICES=0)"을 `device()` 함수로 바꿨다. `provenance/<run_id>.json`의 `env.gpu`로 GPU 이름과 서버를 정하고(RTX 3090 Ti → ubuntu20, RTX 4090 → honeynaps, 표지 `[SERVER-GPU]`), 워커 로그 태그 `cuda:0/GPU<N>`에서 GPU 번호를 읽는다. 결과 예: `ubuntu20, RTX 3090 Ti, GPU 0`. 끝나지 않은 run 줄에는 워커 로그 태그만 적는다(시작 전이면 빈칸). provenance에는 hostname이 없어서 GPU 이름으로 서버를 가린다.
- 시험(출력은 scratchpad로 돌려서 기존 `tables/run_manifest.csv`는 건드리지 않음): 64개 중 39개 complete, overlap_ok 39, 39개 모두 `ubuntu20, RTX 3090 Ti, GPU 0`.

### 8단계. 남은 run (A안)
- 8.1 `git fetch`: 옛 서버에서 새로 올라온 commit 없음. 확인 스크립트 결과 39 done, 25 missing으로 `ROUND5_STATUS.md` 2절 목록과 같다. 워커 로그에 exit≠0이나 Traceback 없음.
- A안이라 run은 돌리지 않는다. 다만 **snapshot은 다시 만들었다.** 저장소의 `enqueued_{heavy,light}_snapshot.txt`에는 옛 절대 경로가 들어 있어서, 9단계의 `r5_manifest.py`가 `od.relative_to(JR)`에서 `ValueError`로 멈추기 때문이다(START_HERE는 B·C안에서만 snapshot을 다시 만들라고 적었지만 A안에도 필요하다).
  - `git mv`로 `heavy.txt`, `light.txt`, `enqueued_{heavy,light}_snapshot.txt`를 `journal_expansion/runs/queue_r5/archive_ubuntu20/`로 옮겼다.
  - `python journal_expansion/scripts/enqueue_r5.py --snapshot-only`로 snapshot 64줄(heavy 15, light 49)을 새 경로로 다시 썼다. 옛 snapshot의 경로만 바꾼 것과 `diff`가 없다.
  - `enqueue_r5.py`(플래그 없이)는 실행하지 않았다. 이 서버에는 `heavy.txt`, `light.txt`가 없다.
  - 옛 서버의 sync 스크립트는 큐 파일과 snapshot을 복사하지 않으므로(`--exclude`), 이 변경과 충돌하지 않는다.
- `r5_monitor.py`는 이 서버에서 실행하지 않는다. 옛 서버에서 올라온 워커 로그에 실패 기록이 있으면 이 서버의 `heavy/light.txt`에 run을 다시 넣기 때문이다. A안에서는 8.1의 확인 스크립트로 진행 상황을 본다.
- 9단계 스크립트 사전 점검(새 run 없음): `artifacts/driftgate_tmc_final/`을 scratchpad로 복사하고, 복사본의 `precheck.py`, `r5_tables.py`, `r5_config_comm.py`, `r5_manifest.py`, `r5_figures.py`를 저장소 루트에서 실행했다. 출력이 스크립트 위치 기준이라 저장소의 산출물은 바뀌지 않았다.
  - 다섯 개 모두 exit 0. precheck: "ALL MATCH relative-only mapping … True", `precheck_controller_path.csv`가 저장소 파일과 byte 단위로 같다.
  - R5 run에 의존하지 않는 표(T2a, T6a–c, T7, T8, T9)는 저장소의 첫 버전과 byte 단위로 같다. 차이는 첫 버전 이후 끝난 R5 run이 채우는 칸(T1의 APFL, T3, T4b, T5, T10)뿐이다. 최종 표는 64개가 모두 끝난 뒤 9단계에서 다시 만든다.

### 현재 상태와 다음 할 일 (2026-09-29 21:55경)
- 1–7단계 완료. 8단계는 A안(옛 서버가 남은 25개를 마침)이라, 옛 서버가 `sync_results_from_old_server.sh`로 결과를 push하기를 기다린다.
- 결과가 올라오면: `git pull` → 8.1 확인 스크립트로 64/64 확인(이 서버에서는 `r5_monitor.py`를 쓰지 않음) → 9단계(`precheck` → `r5_tables` → `r5_config_comm` → `r5_manifest` → `r5_figures`, 캡션 확인, 보고서) → 10단계.
