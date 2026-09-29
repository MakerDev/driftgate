# Claude Code 메모리 사본 (옛 서버)

옛 서버(`ubuntu20`)에서 이 프로젝트를 진행한 Claude Code 에이전트의 메모리 파일이다. 원래 위치는
`~/.claude/projects/-disk2-Yujin/memory/`였다. `project_server_migration.md`는 이전할 때 새로 추가했고,
`MEMORY.md`의 색인 두 줄(TMC expansion state, Server migration)을 이전 시점에 맞게 고쳤다. 나머지 파일은 원본 그대로다.

## 새 서버에서 가져오는 방법

Claude Code는 git 저장소 루트 경로를 프로젝트 이름으로 바꿔서 메모리 폴더를 찾는다.
경로의 영문자와 숫자가 아닌 문자를 모두 `-`로 바꾼 이름이다(예: `/disk2/Yujin` → `-disk2-Yujin`).

```bash
SLUG=$(python -c "import re,sys; print(re.sub(r'[^A-Za-z0-9]', '-', sys.argv[1]))" "$(git rev-parse --show-toplevel)")
mkdir -p ~/.claude/projects/$SLUG/memory
find MIGRATION/claude_memory -name '*.md' ! -name README.md -exec cp -n {} ~/.claude/projects/$SLUG/memory/ \;
```

## 읽을 때 주의할 점

- 메모리 속 `/disk2/Yujin/adaptive_splitomc_tmc/...`는 새 서버의 저장소 루트로 바꿔 읽는다.
- `adaptive_splitomc_v3`, `_v4`, `_v5` 폴더는 옮기지 않았다. 그 폴더를 가리키는 메모리(ICTC 시절 framing, v4 disagreement, v5 fairness)는 배경 정보로만 쓴다.
- 가장 최근 상태는 `project_tmc_expansion_state.md`의 마지막 항목(Round 4)과 `MIGRATION/ROUND5_STATUS.md`에 있다.
