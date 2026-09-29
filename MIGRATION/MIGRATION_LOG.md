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
- GPU: 사용자가 붙여 준 답의 GPU 항목이 틀(`<번호들, 예: 0,1,2,3>`, `<예/아니오>`) 그대로여서 허락된 GPU 번호를 아직 모른다. 답을 받을 때까지 GPU를 쓰지 않는다.
- 남은 R5 run: **8.2의 A안**. 옛 서버가 모두 마치고 결과를 GitHub로 보낸다. 새 서버에서는 R5 run을 돌리지 않는다. 결과가 올라오면 9단계부터 한다.
- 데이터셋: 옛 서버에서 직접 복사할 수 없다. 공개 주소에서 다시 받는다.

### 2단계. 메모리
- `MIGRATION/claude_memory/*.md`(README 제외) 9개를 `~/.claude/projects/-home-honeynaps-data-driftgate/memory/`로 복사했다(`cp -n`, 기존 파일 없음).

### 3단계. 환경
- 서버에 conda가 없고 시스템 Python이 3.10.12여서, `pip3 install --user uv`(uv 0.12.20)로 CPython 3.12.7을 받고 가상 환경 `~/venvs/driftgate`를 만들었다.
- `uv pip install --index-strategy unsafe-best-match -r MIGRATION/requirements-lock.txt`로 설치했다. 버전이 옛 서버와 모두 같다: torch 2.7.1+cu126, torchvision 0.22.1+cu126, numpy 2.5.1, scipy 1.16.3, matplotlib 3.11.0, PyYAML 6.0.1, pytest 8.4.2, tqdm 4.66.5, pillow 10.4.0 (onnx 1.22.0도 설치).
- 드라이버 580.178.04가 cu126 wheel을 지원하므로 다른 CUDA wheel로 바꾸지 않았다. `torch.cuda.is_available()=True`, device 4개, cuDNN 90501.
- 가상 환경 안에 `python`과 `python3`가 모두 있다. 명령은 `source ~/venvs/driftgate/bin/activate` 뒤에 실행한다.

### 4단계. 데이터셋 (진행 중, 아래에 결과 추가)
- `DATA_ROOT=/home/honeynaps/data/driftgate_datasets`로 정했다. 사용자가 위치를 정해 주지 않아서, 저장소 옆(`/home/honeynaps/data`)에 다른 프로젝트 폴더와 겹치지 않는 이름으로 만들었다. 남은 공간 약 270 GB.
- Tiny-ImageNet-200: cs231n 주소에서 받았다. zip md5 `90528d7ca1a48142e341f4ef8d21d0de`, 248,100,043 bytes로 기록과 같다. 서버에 `unzip`이 없어서 Python `zipfile`로 풀었다. train JPEG 100,000개, val 이미지 10,000개.
- CIFAR-10/100과 SVHN은 torchvision으로 차례로 받으면 너무 느려서(toronto.edu에서 초당 약 90 KB), torchvision을 멈추고 wget으로 네 파일을 동시에 받았다. 받은 뒤 torchvision이 md5를 확인하고 압축을 푼다.

### 5단계. 경로 치환과 코드 동일성
- `rewrite_server_paths.py --repo-root /home/honeynaps/data/driftgate --data-root /home/honeynaps/data/driftgate_datasets`: dry run 71줄, `--apply` 71줄 치환. "Old paths left in executable code: 0". GIT_ROOT는 기본값(저장소 루트)을 썼다.
- `verify_code_identity.py`: "OK: all files match the Round-5 code" (17개 파일, `datasets_ext.py`만 표지·경로를 되돌린 뒤 일치).
- 옛 경로 grep: `MIGRATION/` 밖의 실행 파일에서 0줄. `MIGRATION/tools/`에 남은 옛 경로는 도구의 기본값과 옛 서버에서 돌릴 sync 스크립트이므로 그대로 둔다. (이 서버의 `grep -r`은 경로 앞에 `./`를 붙이지 않아서 START_HERE의 `grep -v "^./..."` 필터가 걸러 내지 못한다. `sed 's|^\./||'`로 맞춘 뒤 확인했다.)
- 수동 결정 14줄은 7단계에서 다룬다.

### 6단계. 테스트 (CPU 부분)
- `CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py journal_expansion/tests -q -p no:cacheprovider`: 75 passed (2.74 s).
