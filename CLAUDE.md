# CLAUDE.md — DriftGate (IEEE TMC) 작업 저장소

> 연구자: 신유진 (Yujin Shin, 연세대 / UMich 방문). 사용자와는 한국어로 소통한다.
> 옛 v3 ICTC 파이프라인용 CLAUDE.md는 `docs/legacy_v3_ictc/CLAUDE.md`로 옮겼다. 그 문서의 `run_all.sh` 절차는 더 이상 쓰지 않는다.

## 이 저장소를 처음 여는 에이전트라면

이 저장소는 2026-09-29에 옛 서버(`ubuntu20`)에서 옮겨 왔다. **먼저 `MIGRATION/START_HERE.md`를 읽고 순서대로 진행한다.** 서버 이전의 후처리(환경, 데이터셋, 경로 치환, GPU 설정, 확인 실행)를 마친 뒤 Round 5 실험을 끝내고 최종 보고서를 쓰는 것이 현재 목표다.

`MIGRATION/MIGRATION_LOG.md`가 있으면 이전 세션이 어디까지 했는지 먼저 확인한다.

## 핵심 문서

| 문서 | 내용 |
|---|---|
| `MIGRATION/START_HERE.md` | 새 서버에서 할 일의 단계별 목록 |
| `MIGRATION/PROJECT_CONTEXT.md` | 연구 내용, 경과, 반드시 지킬 규칙 |
| `MIGRATION/ROUND5_DIRECTIVE.md` | 사용자의 Round 5 지시문 원문, 한국어 작성 지침 |
| `MIGRATION/ROUND5_STATUS.md` | Round 5 진행 상황 (64 run 중 완료/남은 목록) |
| `MIGRATION/CODEBASE_MAP.md` | 코드 구조, run JSON 형식, 큐, 분석 스크립트 |
| `MIGRATION/SERVER_PATHS.md` | 서버 의존 경로 표지와 치환 도구 |

## 항상 지킬 규칙 (요약, 자세한 내용은 PROJECT_CONTEXT.md 4절)

- DriftGate = relonly controller: `--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm` (`--abs_cap` 없음). 상수는 동결되어 있다.
- 모든 새 run은 `--disjoint_pools`를 쓰고 overlap=0을 확인한다.
- 기존 run, 보고서, export는 수정하거나 덮어쓰지 않는다. `journal_expansion/runs/`와 `journal_expansion/provenance/`는 기록이다.
- 조건은 결과를 보기 전에 고정한다. 중간 결과를 보고 arm을 바꾸지 않는다.
- paired 차이는 같은 seed끼리만 계산한다(Student-t 95% CI, n_pos, SD ddof=1). seed 수나 protocol이 다른 값끼리는 빼지 않는다.
- 모든 수치는 raw JSON → script → CSV → 표·그림으로 만든다. 불리한 결과도 그대로 보고한다.
- 쓸 GPU는 사용자가 허락한 번호만 쓴다. 옛 서버에서는 GPU 1 사용 금지였다.
- 오래 도는 작업은 `setsid nohup ... < /dev/null &`로 띄우고 PPID=1을 확인한다. cron supervisor로 워커를 되살린다. `pkill -f`는 쓰지 않고 PID로 종료한다.
- 명령은 저장소 루트에서 실행한다(`./data_cache`, `configs/base_v3.yaml` 상대 경로).
