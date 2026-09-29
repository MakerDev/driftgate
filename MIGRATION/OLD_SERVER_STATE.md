# 옛 서버 상태 (저장소를 올린 시점)

새 서버의 에이전트는 옛 서버에 직접 접속할 수 없다고 가정한다. 옛 서버에서 해야 하는 일은 사용자에게 요청한다.

## 1. 기본 정보

| 항목 | 값 |
|---|---|
| 호스트 이름 | `ubuntu20` |
| 작업 폴더(live checkout) | `/disk2/Yujin/adaptive_splitomc_tmc` |
| 데이터 | `/disk2/Yujin/datasets` |
| git | `/disk2/Yujin`의 git 저장소(HEAD `e82b96b`). 이 프로젝트 폴더는 그 저장소에서 추적되지 않았다 |
| GitHub로 올린 복사본 | `/disk2/Yujin/driftgate_github` (live checkout을 복사하고 경로 표지를 붙인 폴더. 이 폴더에서 처음 push했다) |
| GPU | RTX 3090 Ti 24 GB × 2. **GPU 1은 사용자가 따로 쓰므로 R5에서 쓰지 않았다** |

## 2. 실행 중인 것

- **R5 워커**: live checkout의 `journal_expansion/scripts/r5_worker.py`(GPU 0 고정판). heavy 3개, light 5개가 GPU 0에서 돈다.
- **cron**: 사용자 crontab에 다음 한 줄이 있다. 10분마다 워커가 살아 있는지 보고, 큐에 job이 남았는데 워커가 없으면 다시 띄운다.
  ```
  */10 * * * * /bin/bash /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/scripts/supervisor.sh >> /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/queue/logs/supervisor_cron.log 2>&1
  ```
- 따라서 사용자가 멈추지 않는 한 옛 서버는 R5 큐의 남은 job(`ROUND5_STATUS.md` 2절)을 끝까지 실행한다. 저장소를 올린 시점 기준으로, 남은 job을 모두 마치는 데 하루 정도 걸릴 것으로 보였다.

## 3. 이전 라운드의 큐 상태 (기록)

- `runs/queue/STOP`이 있다. R4 큐 워커는 새 job을 집지 않는다. `runs/queue/queue.txt`는 비어 있다.
- `runs/queue/queue_r4_backup_20260928_0525.txt`: R5 시작 때 비운 R4 큐의 백업
- `runs/queue/cancelled_r5_B5_B6_20260928_0525.txt`: R5 지시로 취소한 B5, B6 job
- `runs/queue/killed_r5_20260928_0523.txt`: R5 시작 때 종료한 run 목록. 3일 동안 SIGSTOP 상태로 멈춰 있던 B3 run들이고, 그중 5개는 GPU 1에 올라가 있었다. 이 가운데 CIFAR-100 spatial을 뺀 B3 run은 R5 큐에서 다시 돌렸다.
- `journal_expansion/scripts/supervisor.sh.bak_pre_r5_20260928_0524`: R5 이전 supervisor(GPU 0, 1 모두 사용)의 백업

## 4. 옛 서버에서 나중에 할 수 있는 일 (사용자에게 요청)

### 4.1 옛 서버에서 끝난 run 결과를 GitHub로 보내기

옛 서버에서 R5 run이 더 끝나면, 사용자가 옛 서버에서 다음을 실행한다. live checkout의 run 기록(JSON, npz, 로그, provenance)을 GitHub 복사본으로 옮기고 commit, push한다. 코드는 옮기지 않는다.

```bash
cd /disk2/Yujin/driftgate_github
bash MIGRATION/tools/sync_results_from_old_server.sh
```

새 서버에서는 `git pull`로 받는다. `all_runs.jsonl`과 R5 워커 로그는 `.gitattributes`에서 union merge로 지정했다. 그래서 두 서버가 같은 파일에 줄을 추가해도 충돌 없이 합쳐진다. 새 서버의 워커 로그 이름(`workerg0h1.log` 등)은 옛 서버의 이름(`worker0.log`, `worker10.log` 등)과 겹치지 않는다.

같은 run을 두 서버가 모두 돌렸다면, 나중에 끝난 쪽의 JSON이 git에서 충돌한다. 이런 경우가 생기지 않도록 `START_HERE.md` 8단계에서 어느 서버가 어떤 run을 맡을지 먼저 정한다.

### 4.2 옛 서버의 R5 실행 멈추기

사용자가 원할 때만 한다. 실행 중인 run은 checkpoint가 없으므로, 멈추면 그 run의 진행분은 사라진다.

```bash
touch /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/queue_r5/STOP   # 워커가 현재 job을 마치면 종료
# 즉시 멈추려면 run_v2.py 프로세스를 PID로 확인한 뒤 종료한다 (pkill -f 패턴은 셸 자신을 죽일 수 있다)
crontab -e   # supervisor 줄 삭제
```

### 4.3 새 서버로 데이터셋 직접 복사하기 (선택)

두 서버 사이에 ssh가 되면 다시 받지 않고 복사할 수 있다.

```bash
rsync -a /disk2/Yujin/datasets/{cifar10,cifar100,svhn,tiny-imagenet-200} <new-host>:<DATA_ROOT>/
```
