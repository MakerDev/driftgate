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
