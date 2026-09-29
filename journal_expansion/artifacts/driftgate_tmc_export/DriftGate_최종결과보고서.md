# DriftGate — 최종 결과 보고서 (추가실험 종합)

> TMC 원고를 위한 최종 결과물. **모든 수치는 저장소의 원시 run JSON에서 코드로 재생성**했다(`scripts/build_data.py`
> 외). Canonical `integrated = mean(acc_total over eval rounds)`. Commit `e82b96b`, config `configs/base_v3.yaml`,
> method·parameter 고정. **Disjoint = controller/evaluation sample 분리(primary), Same-pool = 과거 공통 pool(보조)**;
> 두 protocol을 하나의 paired difference에 섞지 않는다. 3-seed와 5-seed 평균을 서로 빼지 않는다. 그림은 `figures/`에
> vector PDF + 300 dpi PNG로 첨부(그림 목록은 §7). 근거·불일치·남은 작업은 각각 `reproducibility/`, §6.

---

## 1. 한 페이지 결론

- **주 신호 = client–server TV (dual-exit divergence).** runtime label을 쓰지 않는 label-free 신호로 cluster별 λ를
  조절한다. Disjoint protocol에서 CIFAR-10 Schedule A **+0.66 pp**(5/5 seed), medium mobility **+0.63 pp**(3/3)로 최고
  fixed λ를 이기고, entropy 대비 각각 **+1.47 / +2.00 pp**다(§3.1).
- **이득은 adaptive λ에서 온다.** adaptive λ를 유지한 채 Λ=0.5로 고정해도 결과가 거의 같다(Schedule A −0.071 pp,
  mobility +0.070 pp, 두 CI 모두 0 포함). full 구현은 λ·Λ를 모두 조절하나, **adaptive Λ의 독립적 정확도 이득은 확인되지
  않는다**(§3.7, 그림 F5b).
- **더 단순한 server-only 신호(server non-Main prediction rate)는 대체가 아니라 조건부 대안이다.** class 수가 작고 실제
  drift가 있는 SVHN(+1.19), Schedule A(+0.61), CIFAR-10 gradual(+0.23), mobility(+0.19)에서는 TV와 같거나 높지만,
  class가 많은 CIFAR-100 spatial(−1.44)·gradual(−0.69)에서는 낮다. **CIFAR-100/Tiny에서는 fixed 대비로도** rate가
  낮다(§3.8, 그림 F6).
- **적용 조건:** 온라인 분포 변화가 실재하고 label이 없는 personalized split learning. 지속적으로 높은 drift가 이어지는
  peak 구간에서는 사전 고정 robust λ=0.2가 더 높다(§3.2). "TV가 항상 최고", "모든 shift 해결", "adaptive Λ의 독립
  기여"는 주장하지 않는다.

---

## 2. 실제 평가 정의 (코드 확인; 근거는 `reproducibility/evidence_index.md`)

- **최종 예측 = entropy-routed dual-exit.** 각 sample에서 client exit의 예측 entropy가 threshold `eth=0.8`을 넘으면
  server exit(연결된 server 복제들의 **logit 평균**의 argmax), 아니면 client exit을 최종 예측으로 쓴다. routing 결정은
  **label을 쓰지 않는다**(entropy만). `eth=0.8`은 config 기본값으로 모든 방법에 동일 → `acc_total`은 실제 unlabeled
  routing의 end-to-end 정확도다.
- **Main/OOP/OOR는 예측 경로 선택에 쓰이지 않는다.** 실제 label은 `acc_main/oop/oor`와 exit별 정확도의 **분해 보고**에만
  쓴다.
- **집계:** client 내부는 sample 가중, cluster/전체는 **client 평균**; 중첩 client는 per-cell 집계에서 primary edge
  server에만 1회.
- **라운드 순서:** ① traffic(환경, controller 비가시) → ② **직전(r−1) 모델**로 probe에서 signal 계산(causal) → ③
  controller가 λ,Λ 산출 → ④ 학습·집계 → ⑤ **갱신된 모델로 평가**. probe·평가 입력은 eval 모드+no-grad로 BN/state 불변.
- **신호:** TV `½Σ|p_c−p_s|`∈[0,1], entropy = client exit의 `mean H(p)`(nats, [0, ln C]), server non-Main rate
  `mean 1[argmax p_s ∉ C_Main]`. Main 목록은 **training partition에서만** 생성(evaluation/probe label·ρ 미사용, mobility
  중 고정; 코드 확인).
- **Disjoint pool:** class별 20% controller / 80% evaluation, seed 고정, 매 run overlap=0 assert(신규 89 run 전부 0).
- **Controller 상수(resolved config와 일치, 불일치 없음):** warm-up 15, burn-in 10, α=0.3, β=0.05, guard z=0.5, z0=1.5,
  τ=0.75, λ∈[0.15,0.70], Λ∈[0.40,0.70], scale `max(1e-3,0.10·max(|m|,0.05),1.4826·MAD)`, score clip [−2,6], γ=0.5.

---

## 3. 추가실험 결과

### 3.1 Disjoint 핵심 성능 (표 3-1, 3-2)

**표 3-1. Primary 정확도 (mean±SD %, integrated).** 근거 `data/primary_performance.csv`.

| 설정 | seeds | DriftGate TV | fixed λ0.4,Λ0.5 | entropy | adaptive λ·Λ=0.5 | fixed λ0.3 | fixed λ0.5 | 사전 고정 λ0.2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CIFAR-10 Schedule A | 5 | 65.84±2.03 | 65.17±2.07 | 64.37±1.75 | 65.91±2.01 | 65.10±1.91 | 64.86±2.18 | 64.70±1.73 |
| CIFAR-10 medium mobility | 3 | 59.21±0.37 | 58.58±0.42 | 57.21±0.46 | 59.14±0.40 | — | — | 57.93±0.45 |

Schedule A: 150 rounds, ρ 0→0.4(R31)→0.8(R61)→0.4(R91)→0(R121). Mobility: 120 rounds.

**표 3-2. TV − reference (동일 seed, disjoint, Student-t 95% CI).** 근거 `data/primary_paired_diff.csv`.

| 비교 | 평균 | 95% CI | 양의 seed |
|---|--:|:--:|--:|
| Schedule A: TV − fixed λ0.4 | **+0.664 pp** | [+0.31, +1.01] | 5/5 |
| Schedule A: TV − entropy | **+1.470 pp** | [+1.05, +1.89] | 5/5 |
| mobility: TV − fixed λ0.4 | **+0.629 pp** | [+0.44, +0.82] | 3/3 |
| mobility: TV − entropy | **+1.995 pp** | [+1.18, +2.81] | 3/3 |

### 3.2 지속 고drift(ρ=0.8) 구간 (표 3-3)

**표 3-3. Schedule A ρ=0.8 구간 accuracy(%), disjoint 5-seed.** 구간 = ρ=0.8인 eval round(R70/80/90) 평균.
근거 `tables/final_closure_round2/high_nonmain_segment.csv`.

| 방법 | ρ=0.8 구간 |
|---|--:|
| 사전 고정 λ0.2 | **61.40** |
| server non-Main rate | 60.60 |
| adaptive λ, Λ=0.5 | 59.84 |
| Full DriftGate TV | 59.80 |
| fixed λ0.4 | 58.45 |
| entropy | 53.09 |

Peak paired(5/5): TV−entropy **+6.71 pp**, TV−fixed λ0.4 **+1.35 pp**, TV−fixed λ0.2 **−1.60 pp**. 지속 고drift 구간에서는
최대 generalization을 하는 robust 저-λ 고정이 더 높다(설계 목적대로). adaptive의 이점은 entropy 대비 및 전체 trajectory
평균에서 나타난다. 이 구간 값을 headline 평균과 합치지 않는다.

### 3.3 Passive signal quality — 그림 F1 (표 3-4)

`figures/fig01_signal_dynamics`: TV는 ρ를 추적(상승·복귀), entropy는 ρ와 무관하게 수렴에 따라 단조 감소하고 abrupt
후에는 역방향. canonical `evaluate_signal`(warm-up 15, drift-active ρ≥0.5, raw-direction AUROC).

**표 3-4. 신호–ρ 관계.** 근거 `tables/final_closure_round2/passive_signal_quality.csv`.

| 신호·trajectory | Spearman | Raw AUROC | Direction-free AUROC |
|---|--:|--:|--:|
| TV, Schedule A (passive fixed-weight rec_A, 3 seeds) | +0.907 | 0.978 | 0.978 |
| TV, post-convergence abrupt (rec_abrupt) | +0.852 | 1.000 | 1.000 |
| entropy, Schedule A | +0.437 | 0.643 | 0.643 |
| entropy, post-convergence abrupt | −0.578 | 0.161 | 0.839 |

entropy raw AUROC 0.16은 raw level이 abrupt 수렴 뒤 **역방향**이라는 뜻이며 정보 없음이 아니다(direction-free 0.84).
공통 entropy-controller trajectory에서의 4-신호 비교(다른 trajectory, 중복 측정 아님): TV Schedule A +0.783 / SVHN
+0.433, server rate SVHN +0.866 등 — setting 의존적(표 5-B in `passive_signal_quality.csv`).

### 3.4 λ 적응 메커니즘 — 그림 F3

`figures/fig03_lambda_adaptation`(ρ + λ; **accuracy-vs-round panel은 요청에 따라 제거**): TV는 ρ 상승 시 λ를 낮춰
generalize하고 복귀 시 올린다. entropy는 λ를 ~0.63으로 올려 고정(역방향). mobility에서는 membership 변화(~R60) 후 TV가
λ를 0.4 아래로 내린다. 정확도 결과는 표 3-1/3-2.

### 3.5 역할 ablation — 그림 F4a (표 3-5)

**표 3-5. server exit 역할 조작에 따른 TV–ρ Spearman (same-pool, 3 seeds, mean±SD).** 근거 `data/f4_role.csv`.

| 조건 | TV–ρ Spearman |
|---|--:|
| standard (separated) | +0.866 ± 0.097 |
| same-role exits | −0.213 ± 0.028 |
| same-role independent init | +0.151 ± 0.330 |
| weakened server | +0.019 ± 0.412 |

역할을 분리해야 TV–ρ 상관이 유지된다(standard ≫ 교란 role). 정확도·capacity도 함께 바뀌는 조작이므로 신호 품질 단독
결론으로 확대하지 않는다. server-only 신호의 별도 1-seed 역할 진단(role 무관하게 ρ 추적, 0.92/0.93/0.89)은 TV의
역할-분리 메커니즘과 동일하게 설명하지 않는다.

### 3.6 Absolute-view 기여 — 그림 F4b (표 3-6)

**표 3-6. absolute view 추가(개발 단계: dual=TV+absolute vs main=relative-only ST·delta_hard; same-pool 3 seeds).**
근거 `data/f4_absolute_contribution.csv`. 개발 단계 controller 간 비교이며 신호도 다르다(단일 component 제거 아님).

| 설정 | full(dual) | relative-only(main) | full − relative | 95% CI |
|---|--:|--:|--:|:--:|
| CIFAR-100 gradual | 35.40 | 31.38 | **+4.02 pp** | [+3.45, +4.59] |
| CIFAR-100 spatial | 36.06 | 32.66 | **+3.40 pp** | [+2.99, +3.80] |
| Tiny-ImageNet | 24.09 | 22.24 | **+1.85 pp** | [+1.24, +2.47] |

### 3.7 Weight timing과 Λ 기여 — 그림 F5 (표 3-7, 3-8)

**표 3-7. adaptive − mean/shuffle control (same-pool replay: weight 순서를 바꿔 실제 재학습, 3 seeds).**
근거 `data/f5_timing.csv`.

| 설정 | − global mean λ | − per-cluster mean λ | − shuffled λ |
|---|--:|--:|--:|
| Schedule A | +0.41 [−0.10,+0.92] | +0.44 [+0.06,+0.81] | +0.42 [−0.06,+0.90] |
| gradual | +0.46 [−0.10,+1.02] | +0.49 [−0.16,+1.14] | +0.81 [−0.39,+2.00] |
| static spatial | +0.33 [+0.25,+0.41] | +0.01 [−0.13,+0.16] | +0.05 [−0.35,+0.45] |

static spatial의 이득은 좋은 per-cluster 평균에서 오며(per-cluster/shuffled ≈ 0) 시간 조절이 아니다.

**표 3-8. Full controller − (Λ=0.5) variant (방향 = full minus fixed-Λ, disjoint).** 근거 `data/f5_lambda.csv`.

| 설정 | full − (Λ=0.5) | 95% CI | full 높은 seed |
|---|--:|:--:|--:|
| Schedule A | **−0.071 pp** | [−0.23, +0.09] | 2/5 |
| medium mobility | **+0.070 pp** | [−0.05, +0.19] | 3/3 |

두 CI 모두 0 포함 → adaptive Λ의 독립적 정확도 이득 미확인. 별도 fixed(λ,Λ) local grid 3-seed(A +0.784, mobility
+0.56, same-pool)는 primary 5-seed와 분리 보관(Schedule A CI는 0 포함).

### 3.8 Server-only 신호 vs TV: 7환경 + matched fixed — 그림 F6 (표 3-9, 3-10)

**표 3-9. server non-Main rate − TV (disjoint, paired, 사전 정한 순서).** 근거 `data/f6_server_signal.csv`.

| 설정 | rounds | seeds | TV (%) | rate (%) | rate − TV (pp) | 95% CI |
|---|--:|--:|--:|--:|--:|:--:|
| SVHN temporal | 150 | 5 | 79.39 | 80.58 | **+1.187** | [+0.26, +2.11] |
| CIFAR-10 Schedule A | 150 | 5 | 65.84 | 66.45 | **+0.610** | [−0.14, +1.36] |
| CIFAR-10 gradual | 150 | 3 | 62.72 | 62.96 | +0.234 | [−0.74, +1.21] |
| medium mobility | 120 | 3 | 59.21 | 59.40 | +0.195 | [−1.07, +1.46] |
| Tiny-ImageNet | 100 | 3 | 23.93 | 23.86 | −0.074 | [−0.14, −0.01] |
| CIFAR-100 gradual | 150 | 3 | 36.10 | 35.41 | **−0.693** | [−1.02, −0.37] |
| CIFAR-100 spatial | 150 | 3 | 36.44 | 35.00 | **−1.440** | [−1.78, −1.10] |

Family 평균(mobility는 family로 묶어 1회): CIFAR-10 temporal +0.42, SVHN +1.19, mobility +0.19, Tiny −0.07,
CIFAR-100 −1.07 → rate가 TV보다 높은 family 3/5. CIFAR-100 spatial에서 rate의 absolute candidate가 최종 λ를 결정한
cluster-round 비율 0.998(포화)이 한 원인.

**표 3-10. Matched fixed reference (disjoint, 3 seeds, best = λ0.2).** 근거 `data/fixed_refs_transfer.csv` (신규 24 run).

| 설정 | best fixed(λ0.2) | TV − fixed | rate − fixed |
|---|--:|--:|--:|
| CIFAR-10 gradual | 62.86 | −0.13 | +0.10 |
| CIFAR-100 gradual | 35.91 | **+0.19** | **−0.50** |
| CIFAR-100 spatial | 36.44 | −0.00 | **−1.44** |
| Tiny-ImageNet | 24.31 | −0.38 | −0.46 |

class가 많은 CIFAR-100/Tiny에서 **direct rate는 best fixed보다도 낮고**(−0.50/−1.44/−0.46), adaptive TV는 fixed와 대등
이상(spatial ≈0, gradual +0.19). rate의 약점이 TV뿐 아니라 fixed 대비로도 확인된다.

---

## 4. Communication (tensor 기반, `tables/communication_accounting.csv`·`overhead.csv`)

- underlying SplitOMC: smashed activation **32,768 B/sample(32 KiB)**, model exchange **12,149,112 B/round**(= 12.15 MB
  decimal = 11.59 MiB, overlap-2 config 라운드 전체).
- **DriftGate 추가 통신 = client당 4 B(32-bit scalar)/round + edge consensus 4·degree B/ES/round**, activation 추가
  upload 0(server 복제가 로컬). Probe forward(probe_n=64): client fwd 0.83 ms, probe fwd 1.53 ms, 16-신호 6.9 ms.
- 실제 모바일 latency/energy는 이번 범위에 없다(payload estimate만; 실측값 미기입).

---

## 5. 신규 실행 요약

- **Round 1(disjoint 핵심):** Schedule A/mobility × {TV, entropy, a1(Λ=0.5), fixed grid} 재실행(overlap=0).
- **Round 2(신호 확장):** server non-Main soft/hard × {Schedule A, mobility, SVHN}, C100/Tiny/CIFAR-10 gradual TV·rate,
  role 진단, passive signal quality, high-drift segment. 35 run.
- **Round 3(이번):** code-verified 평가 정의, 검증된 수치 export, 그림 F1–F6, **matched fixed reference 24 run**
  (fixed λ0.2·0.4, Λ0.5, disjoint, seeds 0–2, C10 gradual·C100 gradual/spatial·Tiny; 24/24 완료).
- 전 신규 run overlap=0, λ/Λ bounds 내, 누락 라운드 0. Provenance는 각 JSON `config.run_id`,
  `tables/final_closure*/run_manifest.csv`.

---

## 6. 불일치·남은 작업

**주요 불일치(`reproducibility/discrepancies.md`):** ① 역할 TV–ρ same-role −0.21(과거 기록 −0.44 등, 3-seed 재계산,
정성적 std≫교란 role 유지) ② Schedule A timing +0.41(과거 +0.90; gradual/spatial은 정확 일치 → matched-source 페어링
정합) ③ entropy Schedule A Spearman "0.50"→REPLACED(A +0.44/abrupt −0.58, 부호 반전). **headline·Λ 방향·entropy
정의·평가 방식 오류는 없음.** 과거 same-pool "TV−entropy @ρ0.8 +3.64 pp"는 disjoint 5-seed에서 +6.71 pp(protocol
변경이지 오류 아님).

**남은 작업(범위상 미실행, 정직 표기):** ① absolute-only(`λ=λ_abs`) controller의 disjoint 정면 비교 — 전용 arm 미배선.
② 충분 학습 후 closed-loop 사전학습 비교(§6.3) — checkpoint 미저장. ③ 실제 모바일 latency/energy 측정 — 장비 필요.
모든 검토 항목 완료가 게재 가능성을 보장하지 않는다.

---

## 7. 첨부 그림 (`figures/`, vector PDF + 300 dpi PNG)

| 파일 | 내용 | 폭 |
|---|---|---|
| `fig01_signal_dynamics` | TV vs entropy 신호 dynamics (Schedule A / abrupt) | 181 mm |
| `fig02_system_overview` | 시스템 도식(dual-exit·cluster·controller·λ/Λ) | 145 mm |
| `fig03_lambda_adaptation` | ρ → λ 적응 (accuracy-vs-round panel 제거) | 181 mm |
| `fig04_mechanism_and_views` | (a) role ablation, (b) absolute-view 기여 | 180 mm |
| `fig05_timing_and_lambda` | (a) weight-timing controls, (b) adaptive-Λ 기여 | 180 mm |
| `fig06_server_signal_comparison` | 7환경 server rate − TV forest | 86 mm |

그림별 영문 caption·배치·조건은 `DriftGate_TMC_figure_guide_ko.md`. 표별 근거·코드 줄번호는
`reproducibility/evidence_index.md`.
