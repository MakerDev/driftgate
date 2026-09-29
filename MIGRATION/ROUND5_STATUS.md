# Round 5 진행 상황 (저장소를 올린 시점: 2026-09-29 18시경 KST)

Round 5 지시문 원문은 `ROUND5_DIRECTIVE.md`에 있다. 이 문서는 지시문의 각 항목을 어디까지 마쳤는지 정리한다.

## 1. 요약

| 항목 | 상태 |
|---|---|
| 3절 대기열 정리 (B5, B6, CIFAR-100 spatial relonly, non-Main rate 재실행, entnorm 분석 취소) | 완료. 옛 R4 큐(`runs/queue/`)는 STOP 상태이고 취소 대상은 R5 큐에 넣지 않았다 |
| 2절 시작 전 확인 1 (entropy run과 relonly run의 controller 경로) | 완료. 같은 경로임을 확인했다 (아래 3절) |
| 2절 시작 전 확인 2 (B1 relonly의 Λ) | 완료. Λ는 relative 점수로 정해졌다 (아래 3절) |
| 새 run 64개 (B3 재실행 10, B4 16, E1 11, E2 9, E3 18) | **39개 완료, 25개 남음** (아래 2절) |
| 5절 기존 로그 분석, 6절 표, 7절 그림 | 스크립트 완성. R5 run이 0개일 때(09-28 05:41) 한 번 생성했다. **모든 run이 끝난 뒤 다시 생성해야 한다** |
| 8절 보고서 `DriftGate_final_report_ko.md` | **아직 쓰지 않았다** |

## 2. 새 run 64개

run 정의(명령줄)는 `journal_expansion/scripts/enqueue_r5.py`에 있다. 모두 `--disjoint_pools --probe_n 64 --model_seed 100+seed`를 쓴다.

| 실험 | 저장 폴더 (`journal_expansion/runs/`) | run 수 | 완료 |
|---|---|---|---|
| B3 재실행: 고정 λ=0.15, Λ=0.5 (SVHN s2–4, CIFAR-10 gradual s0–1, CIFAR-100 gradual s1–2, Tiny s0–2) | `phaseT4_B3_fixedgrid/` | 10 | 8 |
| B4: APFL η=0.01, 0.1 (Schedule A s0–4, mobility s0–2) | `phaseT4_B4_apfl/` | 16 | 10 |
| E1: relonly transfer (CIFAR-10 gradual s0–2, Tiny s0–2, SVHN s0–4) | `phaseT5_E1_relonly_transfer/` | 11 | 11 |
| E2: ResNet-18 middle split, Schedule A (relonly, 고정 0.2, 고정 0.4 × s0–2) | `phaseT5_E2_resnet/` | 9 | 3 |
| E3: guard 0.25/1.0 (A s0–2), λ_max 0.60/0.80 (A s0–2, mobility s0–2) | `phaseT5_E3_const/` | 18 | 7 |
| 합계 | | 64 | 39 |

B3 폴더에는 Round 4의 B3 run도 함께 있다. R5에 속한 run은 위 10개뿐이다.

### 남은 25개

옛 서버(GPU 0 하나)에서 저장소를 올릴 때의 상태다.

- **옛 서버에서 실행 중 (8개)**
  - `e2_res_relonly_A_s1`, `e2_res_fx20_A_s1`, `e2_res_fx40_A_s1`: 09-29 12:31 시작. 라운드 80/150 근처였고, 남은 시간은 약 4–5시간이었다.
  - `e3_lmax060_A_s1`, `e3_lmax060_A_s2`, `e3_lmax080_A_s0`, `e3_lmax080_A_s1`, `e3_lmax080_A_s2`: 09-29 13:04–16:47 시작. 각 run은 6시간 안팎 걸린다.
- **옛 서버 큐에서 대기 중 (17개)**
  - heavy: `e2_res_{relonly,fx20,fx40}_A_s2`
  - light: `b4_apfl_eta001_mob_s0–2`, `b4_apfl_eta010_mob_s0–2`, `e3_lmax060_mob_s0–2`, `e3_lmax080_mob_s0–2`, `b3_c100gsig_fx15_s1–2`

옛 서버는 저장소를 올린 뒤에도 R5 큐를 계속 실행한다. 옛 서버의 cron supervisor가 살아 있고, 워커가 죽어도 10분 안에 다시 띄운다. 그래서 **새 서버에서 남은 run을 돌리기 전에, 옛 서버에서 그 run이 이미 끝났거나 실행 중인지 사용자에게 확인해야 한다.** 판단 방법은 `START_HERE.md` 8단계에 있다.

run이 끝나면 JSON이 생긴다. 중간 checkpoint는 없다. 따라서 저장소에 JSON이 없는 run은 새 서버에서 처음부터 다시 돌려야 한다.

## 3. 시작 전 확인 결과 (이미 완료)

`journal_expansion/artifacts/driftgate_tmc_final/precheck/`에 결과가 있다.

- **확인 1**: entropy run(Schedule A s0–4, mobility s0–2, SVHN s0–4)과 B1 relonly run(Schedule A s0–4, mobility s0–2, CIFAR-100 gradual s0–2)의 λ, Λ를 relonly 규칙으로 다시 계산했다. 계산에는 기록된 `controller_z`를 썼다. warm-up 이후 모든 cluster-round에서 기록값과의 최대 오차가 1.1e-16이었고, warm-up 불일치는 0건이었다. 따라서 두 run 묶음은 absolute cap 없이 같은 상수로 같은 코드 경로를 거쳤다. entropy를 relonly 경로로 다시 돌릴 필요가 없다.
  - 코드 근거: entropy arm의 명령줄(`journal_expansion/scripts/enqueue_task1.py:13`)에는 `--abs_cap`이 없다. `run_v2.py:177–178`은 `--abs_cap`이 있을 때만 `ckw["abs_cap"] = True`로 둔다. controller는 `self_calibrating.py:168`의 `use_abs = self.abs_cap or self.abs_only`가 거짓이면 absolute branch를 쓰지 않는다. 보고서에 적기 전에 줄 번호를 한 번 더 확인한다.
- **확인 2**: B1 relonly run에서 Λ도 같은 z로 정해졌다(오차 1.1e-16). absolute 후보 기록 `lam_abs`는 모든 라운드에서 비어 있었다.
- **코드 동일성**: `code_identity.txt`에 R5 run이 실행하는 소스 17개의 sha256을 기록했다. 옛 서버의 git HEAD는 `e82b96b`였지만, 이 프로젝트 폴더는 그 git 저장소에서 추적되지 않았다(untracked). 그래서 commit hash만으로는 코드를 특정할 수 없고, sha256 목록으로 특정한다. 새 GitHub 저장소의 첫 commit hash도 보고서에 함께 적는다.

## 4. 이미 생성된 표와 그림 (첫 버전)

`journal_expansion/artifacts/driftgate_tmc_final/` 아래에 있다. R5 run이 하나도 끝나지 않은 09-28 05:41–05:49에 만들었으므로, R5 run에 의존하는 칸(APFL, E1–E3, B3 fx15 일부)은 비어 있다.

- `tables/T1`–`T10`, `lambda_trajectories.csv`, `paper_numbers.csv`, `run_manifest.csv`
- `figures/fig1`–`fig5` (PDF, PNG), `figures/figure_captions.md` (영문 캡션, 수치 포함)

첫 버전에서 확인된 대표값은 참고용이다(최종값이 아니다).

- Schedule A: DriftGate 65.93% (5 seeds). 고정 0.2 대비 +1.23 pp, CI [+0.46, +2.00], 5/5.

모든 run이 끝나면 `r5_tables.py` → `r5_config_comm.py` → `r5_manifest.py` → `r5_figures.py` 순서로 다시 생성한다. 캡션의 수치도 새 표 값으로 다시 확인한다.

## 5. 남은 작업

1. 남은 25개 run 완료 (옛 서버 또는 새 서버. `START_HERE.md` 8단계)
2. 실패 run 점검: `grep -l Traceback journal_expansion/runs/queue_r5/logs/*.log`, `python journal_expansion/scripts/r5_monitor.py`
3. 표, 그림, manifest 재생성 (`START_HERE.md` 9단계)
4. 보고서 `DriftGate_final_report_ko.md` 작성 (지시문 8절, 부록의 한국어 지침 준수)
5. `paper_numbers.csv`와 캡션 수치가 새 표와 일치하는지 확인
