# Round 4 — 일시 중단 상태 및 재개 절차 (2026-09-24)

## 중단 시점 상태
- 큐 워커 정지: `runs/queue/STOP` 생성 → 워커는 현재 job이 끝나면 종료하고 새 job을 집지 않음.
  `runs/queue/queue.txt`는 **그대로 보존**(51 job 대기: B3 나머지 + B4 16 + B5 16 + B6 18).
- 중단 시 실행 중이던 14개(B3 fx15 SVHN/transfer)는 **자연 종료**되도록 두었음(checkpoint 없음 → kill 시 손실).
  종료 후 JSON은 `runs/phaseT4_B3_fixedgrid/`에 쌓임.
- 자동 재분석 모니터(background)는 정지시킴.
- 완료 현황(중단 시): B1 25/25 ✅, B2 8/8 ✅, B3 15/28(+14 실행 중), B4 0/16, B5 0/16, B6 0/18.

## 재개 절차 (순서대로)
1. `rm /disk2/Yujin/adaptive_splitomc_tmc/journal_expansion/runs/queue/STOP`
   → cron supervisor(10분 주기)가 워커 14개를 자동 재기동하고 queue.txt의 51 job을 이어서 실행.
   (즉시 재기동하려면 `bash journal_expansion/scripts/supervisor.sh` 실행.)
2. 완료 모니터 재가동(백그라운드):
   `ls runs/phaseT4_*/*.json | wc -l` 이 111이 될 때까지 대기 후
   `cd artifacts/driftgate_tmc_export_r4/scripts && python3 r4_partA.py && python3 r4_partB.py && python3 make_figures_r4.py`
3. 111 완료 후 남은 산출물 작성: `DriftGate_R4_report_ko.md`(8개 주장 판정표 + Part A/B 결과 + 그림 목록 + 불일치),
   `figures/figure_captions_r4.tex`의 F4(b) caption 확정, `data/run_manifest.csv` 최종화(스크립트가 생성).
4. 실패 run 점검: `grep -l Traceback runs/queue/logs/b[1-6]_*.log`; 재실행은 `python scripts/enqueue_r4.py`
   (이미 JSON이 있는 job은 건너뜀).

## 코드 변경 (재개 시 그대로 사용; 테스트 `pytest tests/test_r4.py tests/test_journal.py` 33 pass)
- `src/controllers/self_calibrating.py`: `abs_only`, `last_lam_rel/last_lam_abs`, SIGNAL_RANGE["ent_client_norm"]=1.0
- `src/runner.py`: `abs_only` 전달, `ent_client_norm` 계산·기록, `history["lam_rel"/"lam_abs"]`, `apfl` 분기
- `src/apfl_baseline.py`(신규), `scripts/run_v2.py`: `--abs_only`, `--apfl_eta`, mode `apfl`(record=False)
- `scripts/enqueue_r4.py`(111 job 정의; 재실행 안전)

## 중간 결과 (B1·B2 완료분, disjoint, paired)
- **B1 view removal(같은 TV 신호):** full − absonly = +0.92(A, 5/5) / +1.27(mob, 3/3) / +1.22(C100g, 3/3) → absolute만으로는 부족.
  full − relonly = −0.09(A, CI [−0.18,−0.01]) / −0.02(mob) / +0.01(C100g) → **absolute view를 빼도 차이 없음(A는 오히려 relonly가 근소 우위).**
  full − nospatial(C100sp) = −0.01, worst-cluster +0.003 → **spatial view를 빼도 차이 없음.**
  ⇒ 기존 "+4.02/+3.40/+1.85 pp absolute 기여"는 신호 혼동 비교였고, 동일 신호에서는 재현되지 않음.
- **B2 entnorm(H/lnC, full controller):** TV − entnorm = +0.02(A, CI 포함 0, 3/5) / **+0.46(mob, 3/3, CI [+0.19,+0.74])**.
  기존 entropy arm(relative-only)과의 +1.47/+2.00 격차는 대부분 controller 불공정에서 온 것. entnorm은 absolute 후보가
  항상 결정(활성 1.0; H/lnC≈0.6–0.8 → λ_abs≈0.26–0.37).
- **B3 부분:** mobility fixed 0.5/0.6 = 58.30/57.95 < fixed 0.4 = 58.58 → 이동 전 이득이 높은 fixed λ로 설명되지 않음.
  SVHN fixed 0.4 = 79.79 < 0.2 = 80.43; 0.15는 2/5 완료.

## 산출물 위치
`artifacts/driftgate_tmc_export_r4/{data,figures,scripts}`, 그림 F1–F7(+captions tex) 생성됨(F4(b)는 B1 반영됨).
