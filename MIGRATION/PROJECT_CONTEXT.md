# 프로젝트 맥락: DriftGate (IEEE TMC 투고 예정)

연구자는 신유진(Yujin Shin, 연세대, UMich 방문 중)이다. 이 저장소는 ICTC 2026 논문(Adaptive-SplitOMC)을 IEEE TMC 저널 논문(DriftGate)으로 확장한 작업 전체를 담고 있다. 이 문서는 새 에이전트가 앞선 작업을 다시 조사하지 않고 Round 5를 이어갈 수 있도록, 지금까지의 결정과 규칙을 정리한다.

## 1. 문제와 방법

**설정.** SplitOMC(Rizwan et al., IEEE/ACM ToN 2026)는 여러 cell(edge server, ES)이 겹치는 영역에서 분할학습(split learning)을 한다. 각 client는 client block과 client exit를 갖고, 자신이 속한 ES마다 server block 사본과 server exit를 갖는다. 평가 요청은 세 종류로 나뉜다. Main은 client의 주 class이다. OOP는 client가 속한 ES의 범위 안에 있지만 Main이 아닌 class이다. OOR는 ES 범위 밖의 class이다(`data/partition.py:114–115`). ρ는 요청 중 Main이 아닌 요청(OOP+OOR)의 비율이다.

**개인화 가중치.** 매 라운드 client block을 λ·(자기 모델) + (1−λ)·(cluster 평균)으로 섞는다. Λ는 cluster 평균과 전체 cluster 평균을 섞는 비중이다. ρ가 크면 공유 모델이 필요하므로 λ를 낮추는 것이 유리하고, ρ가 작으면 개인화가 유리하므로 λ를 높이는 것이 유리하다. 가장 좋은 고정 λ는 traffic 구성에 따라 달라진다.

**DriftGate.** 라벨과 ρ를 보지 않고 cluster마다 λ를 정하는 controller다.
- 신호: 각 client가 최근 unlabeled 요청 64개(probe)에 대해 client exit와 server exit 확률 벡터의 TV(total variation distance)를 계산한다. cluster 평균이 d̄이다. 두 exit는 역할이 다르다(client는 Main 전문, server는 일반화). 그래서 Main이 아닌 요청이 늘면 두 exit의 예측이 더 많이 갈라진다.
- 최종 controller = **relonly**. temporal 점수와 spatial 점수 중 큰 값을 평활해서 λ를 정한다. 식과 상수는 `ROUND5_DIRECTIVE.md` 1절에 있다. absolute branch는 Round 4 B1에서 효과가 없다고 확인되어 논문에서 뺐다.
- 비교 대상: 고정 λ grid, 같은 controller에 entropy(H 또는 H/ln C)를 넣은 것, absonly(TV 크기로 λ를 바로 정함), APFL식 λ 학습.

## 2. 연구 경과

| 시기 | 단계 | 핵심 결과 | 기록 |
|---|---|---|---|
| ~2026-06 | ICTC v3/v4 | entropy 기반 adaptive λ. Schedule A에서 고정 λ보다 +1.33 pp | `results/`, `docs/`, `docs/legacy_v3_ictc/` |
| 07-11~07-13 | TMC Gate A/B, pilot | 재현 확인(0.6669 vs 0.6671). 신호 비교에서 entropy는 drift와 반대로 움직이고(AUROC 0.161), client–server 불일치 신호는 ρ를 따라감. 옛 μ 보정이 test 관측에 맞춰진 누수를 발견 → 자기 보정(self-calibrating) controller로 전환 | `reports/phase0_*`, `phase1_calibration.md`, `phase2_signal_benchmark.md` |
| 07-17~07-20 | Gate C/D, 동결 | temporal 점수 + spatial 점수 + absolute branch(DV-2)로 동결. 신호는 TV로 확정 | `reports/phaseB_*`, `phaseC_*`, `dv_frozen_specification.md` |
| 07-20~08-02 | SVHN frozen holdout, Phase F–I | 신호는 SVHN에서도 entropy보다 우수(+2.40 pp). 그러나 고정 λ=0.2가 DriftGate보다 약 1 pp 높음 → "모든 고정 λ보다 높다"는 주장을 전 설정으로 일반화하지 않기로 함 | `reports/phaseE_frozen_holdout.md`, `FINAL_REPORT_2026-08-02.md` |
| 09-06 | 최종 평가 Round 1 | probe/평가 pool 분리(disjoint)에서도 핵심 결과 유지. adaptive Λ의 기여는 약 0 | `reports/DriftGate_TMC_final_evaluation_closure_report.md` |
| 09-07 | Round 2 | server non-Main prediction rate(hard)와 TV 비교 → TV를 주 신호로 유지 | `reports/final_closure_round2.md` |
| 09-14 | Round 3 (export) | 논문용 수치 문서와 그림 6개 | `artifacts/driftgate_tmc_export/` |
| 09-24 | Round 4 | B1: absolute branch를 빼도 차이 없음(full−relonly −0.09/−0.02/+0.01 pp) → **relonly로 방법 확정**. absonly는 부족(+0.9~1.3 pp 낮음). 사용자가 중간에 일시 중단 | `artifacts/driftgate_tmc_export_r4/RESUME_NOTE.md` |
| 09-28~ | **Round 5 (진행 중)** | 비모바일 실험 마무리. 최종 표, 그림, 한국어 보고서 | `ROUND5_DIRECTIVE.md`, `ROUND5_STATUS.md` |

모든 경로는 `journal_expansion/` 기준이다. 라운드별 상세 수치는 해당 보고서와 `MIGRATION/claude_memory/project_tmc_expansion_state.md`에 있다.

## 3. Round 5 첫 버전 표에서 본 현재 수치 (최종값 아님)

R5 run이 끝나기 전에 기존 run만으로 만든 표(`artifacts/driftgate_tmc_final/tables/`)의 값이다. 최종 보고서에서는 모든 run이 끝난 뒤 다시 계산한 값을 쓴다.

| 설정 | DriftGate | 가장 좋은 고정 λ | 차이 (95% CI, n_pos) |
|---|---|---|---|
| stepwise composition change (Schedule A, 5 seeds) | 65.93% | 0.4: 65.17% | +0.76 pp [+0.41, +1.11], 5/5 |
| client mobility (3 seeds) | 59.23% | 0.4: 58.58% | +0.65 pp [+0.47, +0.83], 3/3 |
| Schedule A, entropy 대비 | | | +1.56 pp, 5/5 |
| Schedule A / mobility / CIFAR-100 gradual, absonly 대비 | | | +1.01 / +1.30 / +1.20 pp, 모든 seed 양수 |
| CIFAR-100 gradual (3 seeds) | 36.09% | 0.2: 35.91% | +0.18 pp [−0.13, +0.50] |

transfer 설정에서는 가장 좋은 고정값과 거의 같거나 조금 낮을 수 있다. SVHN에서는 옛 결과상 고정 0.2가 약 1 pp 높았다. 지시문 7번 주장은 이 차이를 판정 없이 수치로만 적는다.

## 4. 반드시 지킬 규칙

지시문과 이전 라운드에서 사용자가 정한 규칙이다. 어기면 결과를 다시 만들어야 한다.

**방법과 상수**
- DriftGate = relonly 실행이다. 플래그는 `--mode selfcal --signal tv_dist --burn_in 10 --z_guard 0.5 --spatial_norm`이고 `--abs_cap`는 쓰지 않는다.
- controller 상수는 동결되어 있다. warm-up 15, burn-in 10, β=0.05, guard 0.5, z clip [−2, 6], EMA α=0.3, z0=1.5, τ=0.75, λ∈[0.15, 0.70], Λ∈[0.40, 0.70], warm-up 값 λ=0.425, Λ=0.55. E3에서 지정한 arm만 한 상수씩 바꾼다.
- controller는 라벨과 ρ를 보지 않는다. 코드 API와 테스트가 이를 강제한다.

**실험 절차**
- 모든 새 run은 `--disjoint_pools`를 쓰고, overlap=0 assert를 통과해야 한다.
- 조건은 결과를 보기 전에 고정한다. gate run을 두지 않는다. 중간 결과를 보고 arm을 추가, 삭제, 변경하지 않는다.
- 기존 run과 기존 export는 수정하거나 덮어쓰지 않는다. 같은 설정을 다시 돌리면 새 run_id가 생기고, 결과 JSON은 건너뛰기만 한다.
- 비교 기준이 되는 기존 run은 다시 돌리지 않는다.

**통계**
- integrated accuracy = 평가 라운드별 `acc_total`의 평균.
- paired 차이는 같은 seed끼리 계산한다. Student-t 95% CI와 n_pos(양수 seed 수)를 함께 적는다. SD는 ddof=1.
- seed 수가 다른 값끼리는 빼지 않는다. protocol(disjoint와 same-pool)이 다른 값끼리도 빼지 않는다.
- 모든 수치는 raw JSON → script → CSV → 표·그림 순서로 만든다. 손으로 옮겨 적은 수치를 쓰지 않는다.

**보고**
- 불리한 결과도 그대로 적는다. 주장 판정은 지지, 부분 지지, 반대 중 하나로 한다.
- 최종 보고서와 표에서 제외할 항목은 `ROUND5_DIRECTIVE.md` 9절에 있다(full controller, entnorm, non-Main rate, CIFAR-100 spatial, weight timing, Λ 고정 변형, 네트워크 지연/손실/잡음, 개발 단계 비교).
- 그림에서는 내부 이름을 논문 용어로 바꾼다. Schedule A → stepwise composition change, mobility-med → client mobility, post-convergence abrupt → late abrupt change, relonly → DriftGate.
- 영문 캡션에는 엠대시와 세미콜론을 쓰지 않는다. "anti-correlates"라는 표현도 쓰지 않는다.
- 한국어 보고서는 `ROUND5_DIRECTIVE.md` 부록의 fluent-korean 지침을 따른다.
- 원고(tex) 수정, 원고 문장이나 framing 제안, 모바일 기기 측정은 Round 5 범위가 아니다.

**GPU와 장기 실행**
- 옛 서버에서는 사용자가 GPU 1을 쓰지 말라고 했다(GPU 0만 사용). 새 서버에서 쓸 GPU 범위는 사용자에게 확인한다.
- 오래 도는 작업은 `setsid nohup ... < /dev/null &`로 띄우고 PPID=1인지 확인한다. tmux나 에이전트 세션에 묶인 프로세스는 세션이 끝나면 죽을 수 있다. 워커가 죽어도 이어지도록 cron supervisor를 둔다.
- `pkill -f <패턴>`은 명령을 실행하는 셸 자신과도 일치해서 셸을 죽일 수 있다. 프로세스는 PID로 확인해서 종료한다.

## 5. 사용자와의 작업 방식

- 사용자는 한국어로 지시한다. 보고와 설명도 한국어로 한다.
- 지시문은 번호가 매겨진 절 단위로 온다. 각 절을 빠짐없이 처리하고, 처리하지 못한 항목은 이유와 함께 보고서의 불일치 기록에 적는다.
- 사용자가 예상 수치를 제시하면, raw 로그에서 다시 계산해서 일치 여부를 확인하고 다르면 그대로 보고한다. 이전 라운드에서 여러 번 이런 불일치(seed 집합 차이, protocol 차이, 끝점 읽기 오류)를 찾았다.
