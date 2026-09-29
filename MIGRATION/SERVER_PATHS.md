# 서버 의존 경로와 GPU 설정

옛 서버(`/disk2/Yujin/...`, RTX 3090 Ti 2장)에 의존하는 코드 줄에는 줄 끝에 표지 주석을 붙였다. 새 서버의 에이전트는 이 표지를 기준으로 경로를 바꾼다.

## 표지 종류

| 표지 | 옛 값 | 새 서버에서 할 일 | 자동 치환 |
|---|---|---|---|
| `# [SERVER-PATH:REPO_ROOT]` | `/disk2/Yujin/adaptive_splitomc_tmc` | 새 저장소 루트(이 README가 있는 폴더)로 바꾼다 | 예 |
| `# [SERVER-PATH:DATA_ROOT]` | `/disk2/Yujin/datasets` | 새 데이터 위치로 바꾼다 (`DATASETS.md`) | 예 |
| `# [SERVER-PATH:GIT_ROOT]` | `"/disk2/Yujin"` | provenance가 `git rev-parse HEAD`를 실행할 git 폴더. 새 서버에서는 저장소 루트가 곧 git 루트이다 | 예 |
| `# [SERVER-PATH:EXTERNAL]` | `/disk2/Yujin/adaptive_splitomc_v3`, `_v4` | 옮기지 않은 옛 형제 프로젝트다. ICTC 시절 스크립트(`scripts/gen_integrated.py`)에서만 쓰므로 그대로 둔다 | 아니요 |
| `# [SERVER-GPU]` | GPU 2장, GPU 0만 사용 등의 가정 | 새 서버의 GPU 수와 사용 허가 범위에 맞게 직접 고친다 | 아니요 |

표지는 주석이라 코드 동작에 영향이 없다. 표지를 붙인 뒤 모든 `.py` 파일의 AST가 원본과 같은지, 모든 `.sh` 파일이 `bash -n`을 통과하는지 확인했다.

전체 목록은 `MIGRATION/server_paths.tsv`에 있다(97행: 파일, 줄, 종류, 표지 여부, 원문). `marked_inline = no`인 12행은 표지를 붙이지 않은 곳이다. 모두 문서, 로그, 백업 파일이거나 docstring 안의 줄이라 실행에 영향이 없다.

## 기록 파일은 바꾸지 않는다

다음 파일에도 옛 경로가 들어 있지만 이력 기록이므로 **절대 고치지 않는다**. 치환 도구도 이 폴더들을 건너뛴다.

- `journal_expansion/runs/**` : run JSON, run 로그, 큐 파일(`queue*/…`), `enqueued_*_snapshot.txt`
- `journal_expansion/provenance/**` : run별 provenance 기록
- `logs/**`, `journal_expansion/reports/**`, 각 `artifacts/**/*.md`

단, 큐 파일 두 가지는 새 서버에서 다시 만들어야 한다. 내용이 옛 절대 경로로 된 명령줄이라서 그대로 실행할 수 없기 때문이다.

- `journal_expansion/runs/queue_r5/heavy.txt`, `light.txt`: 보관 후 비우고 `enqueue_r5.py`로 다시 만든다.
- `journal_expansion/runs/queue_r5/enqueued_{heavy,light}_snapshot.txt`: `r5_manifest.py`가 이 파일의 `--output_dir` 절대 경로로 run JSON을 찾는다. 보관 후 `enqueue_r5.py --snapshot-only`로 다시 만든다.

절차는 `START_HERE.md` 8단계에 있다.

## 자동 치환 도구

저장소 루트에서 실행한다. `--apply`가 없으면 바꿀 줄만 보여 주는 dry run이다.

```bash
python MIGRATION/tools/rewrite_server_paths.py --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"          # dry run
python MIGRATION/tools/rewrite_server_paths.py --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT" --apply  # 실제 치환
python MIGRATION/tools/verify_code_identity.py  --repo-root "$REPO_ROOT" --data-root "$DATA_ROOT"         # R5 코드와 같은지 확인
```

- 표지가 있는 줄에서, 표지 앞의 코드 부분만 바꾼다. 표지는 남기므로 나중에 다시 옮길 때 `--old-repo-root` 등으로 한 번 더 실행할 수 있다.
- 치환 뒤 `.py`는 `ast.parse`, `.sh`는 `bash -n`으로 확인한다. 실패하면 그 파일을 원래대로 되돌리고 멈춘다.
- 마지막에 실행 코드에 남은 옛 경로 수를 출력한다. 0이어야 한다(`EXTERNAL`은 제외).
- 옛 서버에서 복사본으로 시험한 결과: 71줄 치환, 남은 옛 경로 0, 수동 결정 14줄, 치환 후 `pytest tests/test_all.py journal_expansion/tests` 75개 통과, 코드 동일성 확인 통과.

`verify_code_identity.py`는 `journal_expansion/artifacts/driftgate_tmc_final/precheck/code_identity.txt`에 기록된 sha256(R5 run이 실행한 17개 소스 파일)과 현재 파일을 비교한다. 표지 주석과 경로 치환만 되돌려서 비교하므로, 이 도구가 OK를 내면 새 서버의 run은 R5와 같은 코드로 실행된다. 현재 17개 중 표지가 붙은 파일은 `journal_expansion/src/datasets_ext.py` 하나뿐이다.

## 수동으로 결정할 14줄

치환 도구가 마지막에 목록으로 보여 준다. Round 5를 마치는 데 필요한 것은 앞의 세 항목뿐이다.

옛 서버의 `r5_worker.py`와 `supervisor.sh`는 GPU 0을 코드에 고정했다. 저장소에 올린 판에서는 이 두 파일만 GPU를 변수로 받도록 고쳤다. 기본값으로 실행하면 옛 서버와 같게 동작한다. 달라진 점은 워커 번호 형식(`g0h1`, `g0l1` 등)과 heavy 계수 폴더 이름(`running_heavy_gpu<N>`)뿐이다. 두 파일은 학습 코드가 아니므로 `code_identity.txt`의 17개 파일과 무관하다.

| 파일:줄 | 내용 | 할 일 |
|---|---|---|
| `journal_expansion/scripts/supervisor.sh:9–11` | R5에 쓸 GPU 목록(`GPUS`, 기본 `0`)과 GPU당 heavy/light 워커 수(기본 3/5) | 새 서버에서 허락받은 GPU와 메모리에 맞게 기본값을 고친다 (`START_HERE.md` 7단계) |
| `journal_expansion/scripts/r5_worker.py:25` | 워커가 쓸 GPU를 `R5_GPU` 환경 변수로 받는다(기본 `0`). supervisor가 GPU마다 값을 넣어 준다 | 보통은 그대로 둔다. `R5_HEAVY_MAX`(기본 5)는 GPU당 heavy job 동시 실행 한도다 |
| `journal_expansion/artifacts/driftgate_tmc_final/scripts/r5_manifest.py:41` | manifest의 device 열에 "cuda:0 = physical GPU 0"을 고정 문자열로 쓴다 | 서버 이름과 GPU 이름을 적도록 고친다. 옛 서버 run과 새 서버 run을 구분할 수 있어야 한다 |
| `journal_expansion/scripts/launch_{downstream,gatec,pilots,v3b}.py` | `SLOTS`에 `cuda:0`, `cuda:1`을 적어 둔 옛 pilot 실행기 | Round 5와 무관하다. 다시 쓸 때만 고친다 |
| `journal_expansion/scripts/measure_overhead.py:44, 98` | 기본 device가 `cuda:1` | 다시 쓸 때만 고친다 |
| `run_all.sh:19` | v3 ICTC 파이프라인의 `DEVICE1=cuda:1` | Round 5와 무관하다 |
| `scripts/gen_integrated.py:4–5` | 옮기지 않은 `adaptive_splitomc_v3/v4` 결과를 읽는다 | 그대로 둔다. 이 스크립트는 새 서버에서 실행할 수 없다 |

## 상대 경로 가정

- 여러 스크립트가 저장소 루트를 작업 디렉터리로 가정한다(`./data_cache`, `configs/base_v3.yaml`, `journal_expansion/...`). 명령은 항상 저장소 루트에서 실행한다.
- `configs/base_v3.yaml`의 `data_root: ./data_cache`는 상대 경로라 표지가 없다. `DATASETS.md`대로 `data_cache` 심볼릭 링크를 만든다.
