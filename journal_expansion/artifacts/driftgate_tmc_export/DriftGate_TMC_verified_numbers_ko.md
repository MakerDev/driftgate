# DriftGate TMC — 검증된 주요 수치 (한국어)

> 이 문서 하나로 TMC 원고에 필요한 수치와 설정을 사용할 수 있도록 정리했다. 모든 수치는
> 저장소의 **원시 run JSON에서 코드로 재생성**했으며(`scripts/build_data.py`), 그림과 동일한 tidy
> 데이터(`data/*.csv`)를 근거로 한다. Canonical `integrated = mean(acc_total over eval rounds)`.
> Commit `e82b96b` · config `configs/base_v3.yaml`. 근거 ID는 `reproducibility/evidence_index.md` 참조.
> **Disjoint = controller/evaluation sample 분리, Same-pool = 과거 공통 pool(보조).** 두 protocol을
> 하나의 paired difference에 섞지 않는다. 3-seed와 5-seed 평균을 서로 빼지 않는다.

---

## 1. 현재 확정된 결론

- **주 신호 = client–server TV(dual-exit divergence).** Runtime label을 쓰지 않는 label-free 신호로,
  drift가 없을 때는 personalization(큰 λ), OOP/OOR가 늘면 generalization(작은 λ)로 cluster별 λ를 조절한다.
  Disjoint protocol에서 CIFAR-10 Schedule A **+0.66 pp**(5/5 seed), medium mobility **+0.63 pp**(3/3)로
  최고 fixed λ를 이기고, entropy 대비 각각 **+1.47 / +2.00 pp**다.
- **더 단순한 server-only 신호(server non-Main prediction rate)는 대체가 아니라 조건부 대안이다.** class 수가
  작고 실제 drift가 있는 SVHN(+1.19 pp), Schedule A(+0.61), CIFAR-10 gradual(+0.23), mobility(+0.19)에서는
  TV와 같거나 높지만, class가 많은 CIFAR-100 spatial(−1.44)·gradual(−0.69)에서는 낮다(그림 F6).
- **Adaptive Λ의 독립적 정확도 이득은 확인되지 않았다.** Adaptive λ를 유지한 채 Λ=0.5로 고정해도 결과가
  거의 같다(Schedule A −0.071 pp, mobility +0.070 pp, 두 CI 모두 0을 포함). 이득은 **adaptive λ**에서 온다.
  단, 실제 full 구현은 λ와 Λ를 모두 조절한다(정확히 기록).
- **적용 조건:** 온라인 분포 변화가 실재하고 label이 없는 personalized split learning. 지속적으로 높은 drift가
  이어지는 peak 구간에서는 사전 고정 robust λ=0.2가 더 높다(§4). 광범위한 shift를 모두 해결한다고 주장하지 않는다.
- **아직 준비되지 않은 핵심 항목:** (i) CIFAR-10 gradual·CIFAR-100·Tiny-ImageNet의 disjoint matched fixed
  reference(재학습 실행 중, §3.3·§5.2 표에 표시), (ii) absolute-only controller의 disjoint 정면 비교(§6.2,
  전용 arm 미배선 — 남은 작업), (iii) 실제 모바일 latency/energy 측정(이번 범위 밖).

---

## 2. 실제 평가 정의와 설정 (코드 확인)

함수 경로·줄번호는 `reproducibility/evidence_index.md`. 아래는 논문에 넣을 수 있는 사실 서술이다.

### 2.1 최종 예측과 accuracy

- **최종 예측은 entropy-routed dual-exit다.** 각 sample에서 client exit의 예측 entropy가 threshold `eth=0.8`을
  넘으면 server exit(연결된 server 복제들의 **logit 평균**의 argmax)을, 넘지 않으면 client exit(argmax)을 최종
  예측으로 쓴다. Routing 결정은 **label을 쓰지 않는다**(entropy만 사용). `eth=0.8`은 config 기본값으로 모든
  방법에 동일 적용된다. → `acc_total`은 실제 unlabeled routing의 end-to-end 정확도다.
- **Main/OOP/OOR 구분은 예측 경로 선택에 쓰이지 않는다.** 실제 label은 `acc_main/acc_oop/acc_oor`와
  exit별 정확도를 **분해 보고**하는 데에만 쓴다. 따라서 total accuracy는 서비스 성능을 임의로 바꾼 값이 아니다.
- **집계 단위:** client 내부는 sample 가중 정확도, cluster/전체는 **client 평균**(client마다 동일 가중). 중첩
  client는 per-cell 집계에서 자신의 primary edge server에만 1회 들어가 중복되지 않는다.
- **평가 시점:** 각 round의 순서는 ① 현재 traffic 구성(환경, controller 비가시) → ② **직전(r−1) 모델**로 round-r
  probe에서 signal 계산(causal) → ③ controller가 λ,Λ 산출 → ④ 학습·집계 → ⑤ **갱신된 모델로 평가**. ρ·membership·
  probe·λ·모델·평가의 시간 인덱스가 일치한다.
- 로그에서 전체 평균, Main/OOP/OOR별, client/server exit별 정확도, routing 변형을 모두 확보할 수 있다.

### 2.2 학습·배포 데이터 관계

- 매 round의 labeled Main data는 **미리 나눈 client별 train partition의 재사용**이다(`nd1_partition`이 반환한
  고정 train index). Traffic·mobility 변화는 평가쪽 OOP/OOR 구성과 membership을 바꾸며, 학습 partition 자체는 고정.
- Probe와 evaluation 입력은 **eval 모드 + no-grad**로 처리되어 gradient·BN running statistics·기타 state를 바꾸지
  않는다(signal은 순전파만).
- **Primary protocol은 disjoint pool이다.** class별 test pool을 seed 기준 고정 20% controller / 80% evaluation으로
  나누고, 같은 sample ID가 양쪽에 들어가지 않음을 매 run에서 assert(overlap=0). 사용한 89개 신규 run 전부 overlap=0.
- Main class 목록은 **local training partition에서만** 생성(`nd1_partition(train_labels)`); evaluation label,
  probe label, 현재 traffic label, ρ를 쓰지 않고 mobility 중에도 고정. Dataset별 Main 비율 0.20(코드 확인,
  `data/main_class_metadata_audit.csv`).

### 2.3 신호와 controller (scale 공정성)

- **TV** `D_TV = ½Σ_j|p_c(j)−p_s(j)|`, 범위 [0,1]. **Entropy**는 client exit의 per-sample Shannon entropy를 batch
  평균한 `mean H(p)`(nats, 범위 [0, ln C]); routing에도 동일 per-sample entropy 사용. 그림 F1은 scale 공정성을 위해
  entropy를 정의된 `H/ln C`로 정규화해 [0,1]로 표시한다(임의 min–max 정규화 아님).
- **Controller 상수(resolved config와 일치):** warm-up 15, burn-in 10(초기 제외; warm-up과 구분), score/absolute
  EMA α=0.3, baseline update β=0.05, guard z=0.5(v3c 비대칭; 기본 1.5의 CLI override), mapping center z0=1.5,
  temperature τ=0.75, λ∈[0.15,0.70], Λ∈[0.40,0.70], scale rule `max(1e-3, 0.10·max(|m|,0.05), 1.4826·MAD)`,
  standardized score clip [−2,6], edge scalar 1-step, γ=0.5. **문서값과 resolved config 불일치 없음.**
- **매핑:** `λ_rel = 0.70 − 0.55·sigmoid((q−1.5)/0.75)`, `λ_abs = 0.70 − 0.55·d̂`, `λ = min(λ_rel, λ_abs)`;
  Λ은 같은 형태로 [0.40,0.70]. q = temporal/spatial standardized score의 max를 EMA, d̂ = TV의 EMA. server non-Main
  rate `mean 1[argmax p_s ∉ C_Main]`, soft mass `mean Σ_{j∉Main} p_s(j)`.

---

## 3. 주요 성능표 (disjoint)

**표 3-A. Primary 정확도 (mean±SD, integrated total accuracy %).** 근거: `data/primary_performance.csv`.

| 설정 | seeds | DriftGate TV | fixed λ=0.4, Λ=0.5 | entropy | adaptive λ·Λ=0.5 | fixed λ=0.3 | fixed λ=0.5 | 사전 고정 λ=0.2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CIFAR-10 Schedule A | 5 | 65.84±2.03 | 65.17±2.07 | 64.37±1.75 | 65.91±2.01 | 65.10±1.91 | 64.86±2.18 | 64.70±1.73 |
| CIFAR-10 medium mobility | 3 | 59.21±0.37 | 58.58±0.42 | 57.21±0.46 | 59.14±0.40 | — | — | 57.93±0.45 |

Schedule A는 150 rounds, ρ: 0→0.4(R31)→0.8(R61)→0.4(R91)→0(R121)으로 구현되어 있다(round는 config 확인값).
Medium mobility는 120 rounds. SD는 seed 간 sample SD(ddof=1). λ=0.2는 결과를 보기 전에 정한 사전 고정값.

**표 3-B. Paired difference: TV − reference (동일 seed, disjoint).** 근거: `data/primary_paired_diff.csv`.
CI는 seed-level paired difference의 Student-t 95% (n=3은 CI가 넓다는 한계를 감안).

| 비교 | 평균 차이 | 95% CI | 양의 seed |
|---|--:|:--:|--:|
| Schedule A: TV − fixed λ=0.4 | **+0.664 pp** | [+0.31, +1.01] | 5/5 |
| Schedule A: TV − entropy | **+1.470 pp** | [+1.05, +1.89] | 5/5 |
| Medium mobility: TV − fixed λ=0.4 | **+0.629 pp** | [+0.44, +0.82] | 3/3 |
| Medium mobility: TV − entropy | **+1.995 pp** | [+1.18, +2.81] | 3/3 |

---

## 4. Peak 구간과 class-group 결과

**표 4-A. Schedule A ρ=0.8 지속 구간 정확도(%).** 구간 = ρ=0.8인 eval round(R70/80/90) `acc_total` 평균, disjoint 5-seed.
근거: `tables/final_closure_round2/high_nonmain_segment.csv`.

| 방법 | ρ=0.8 구간 accuracy |
|---|--:|
| 사전 고정 λ=0.2 | **61.40** |
| server non-Main rate | 60.60 |
| adaptive λ, Λ=0.5 | 59.84 |
| Full DriftGate TV | 59.80 |
| fixed λ=0.4 | 58.45 |
| entropy | 53.09 |

Peak paired 차이(5/5): TV−entropy **+6.71 pp**, TV−fixed λ0.4 **+1.35 pp**, TV−fixed λ0.2 **−1.60 pp**. 즉 지속 고drift
구간에서는 최대 generalization을 하는 robust 저-λ 고정이 더 높다(설계 목적대로). Adaptive의 이점은 entropy 대비
(수렴사 회피) 및 전체 trajectory 평균에서 나타난다(그림 F3). 이 구간 수치를 headline 평균과 합치지 않는다.

---

## 5. Signal·role·three-comparison·timing·Λ 기여

### 5.1 Passive signal quality (그림 F1, 표 5-A)

**표 5-A. 신호–ρ 관계(canonical `evaluate_signal`, warm-up 15, drift-active ρ≥0.5, raw-direction AUROC).**
근거: `data/passive_signal_quality`(→ `tables/final_closure_round2/passive_signal_quality.csv`).

| 신호·model trajectory | Spearman | Raw AUROC | Direction-free AUROC |
|---|--:|--:|--:|
| TV, Schedule A (passive fixed-weight rec_A, 3 seeds) | +0.907 | 0.978 | 0.978 |
| TV, post-convergence abrupt (rec_abrupt, 3 seeds) | +0.852 | 1.000 | 1.000 |
| entropy, Schedule A (rec_A) | +0.437 | 0.643 | 0.643 |
| entropy, post-convergence abrupt (rec_abrupt) | −0.578 | 0.161 | 0.839 |

해석: entropy의 raw AUROC 0.16은 abrupt 수렴 뒤 raw level이 **역방향**(drift 중 entropy가 초기보다 낮음)이라는
뜻이며 정보가 없다는 뜻이 아니다(direction-free 0.84). entropy–ρ 방향은 schedule에 따라 뒤집힌다(A +0.44, abrupt −0.58).

**표 5-B. 공통 entropy-controller trajectory(동일 model trajectory, 단 이 trajectory는 entropy controller의 영향을
받음 — "neutral backbone"이라 부르지 않음).** 서로 다른 trajectory의 값(0.907 vs 0.783)을 같은 조건의 중복 측정으로
취급하지 않는다.

| 신호 | Schedule A Spearman | SVHN Spearman | SVHN raw AUROC |
|---|--:|--:|--:|
| TV | +0.783 | +0.433 | 0.876 |
| server non-Main rate | +0.700 | +0.866 | 1.000 |
| server non-Main probability mass | +0.373 | +0.709 | 0.974 |
| entropy | +0.327 | −0.764 | 0.190 |

### 5.2 Role 분석 (그림 F4a, 표 5-C)

**표 5-C. server exit 역할 조작에 따른 TV–ρ Spearman (same-pool, 3 seeds, mean±SD).** 근거: `data/f4_role.csv`.

| 조건 | TV–ρ Spearman | 바뀐 것 |
|---|--:|---|
| standard (separated) | +0.866 ± 0.097 | 없음(정상 dual-exit) |
| same-role exits | −0.213 ± 0.028 | server exit이 client와 같은 역할 |
| same-role independent init | +0.151 ± 0.330 | 위 + 독립 초기화 |
| weakened server | +0.019 ± 0.412 | server non-Main coverage 약화 |

역할을 분리하지 않으면 TV–ρ 상관이 무너진다. 이 4개는 정확도·capacity도 함께 달라지는 조작이므로 신호 품질 단독
결론으로 확대하지 않는다. server-only 신호(hard rate)의 별도 1-seed 역할 진단은 role과 무관하게 ρ를 추적했으므로
(Spearman 0.92/0.93/0.89) TV의 역할-분리 메커니즘과 **동일하게 설명하지 않는다**(`data`→round2 `server_role_nonmain_signal.csv`).

### 5.3 Three-comparison(absolute view) 기여 (그림 F4b, 표 5-D)

**표 5-D. absolute view 추가(개발 단계 비교: dual=TV+absolute vs main=relative-only ST·delta_hard; same-pool 3 seeds).**
근거: `data/f4_absolute_contribution.csv`. **개발 단계 controller 간 비교이며 신호도 다르다(단일 component 제거가 아님).**

| 설정 | full(dual) | relative-only(main) | full − relative | 95% CI |
|---|--:|--:|--:|:--:|
| CIFAR-100 gradual | 35.40 | 31.38 | **+4.02 pp** | [+3.45, +4.59] |
| CIFAR-100 spatial | 36.06 | 32.66 | **+3.40 pp** | [+2.99, +3.80] |
| Tiny-ImageNet | 24.09 | 22.24 | **+1.85 pp** | [+1.24, +2.47] |

### 5.4 Weight timing (그림 F5a, 표 5-E)

**표 5-E. adaptive − mean/shuffle control (same-pool replay, weight 순서를 바꿔 실제 재학습·재집계, 3 seeds).**
근거: `data/f5_timing.csv`. mean control은 전체 trajectory를 보고 구성한 분석용 비교임을 명시한다.

| 설정 | − global mean λ | − per-cluster mean λ | − shuffled λ |
|---|--:|--:|--:|
| Schedule A | +0.41 [−0.10,+0.92] | +0.44 [+0.06,+0.81] | +0.42 [−0.06,+0.90] |
| gradual | +0.46 [−0.10,+1.02] | +0.49 [−0.16,+1.14] | +0.81 [−0.39,+2.00] |
| static spatial | +0.33 [+0.25,+0.41] | +0.01 [−0.13,+0.16] | +0.05 [−0.35,+0.45] |

해석: static spatial의 이득은 **좋은 per-cluster 평균을 고른 것**에서 오며(per-cluster/shuffled ≈ 0), 시간적 조절 때문이
아니다. 시간 변화 환경(A/gradual)에서는 mean/shuffle 대비 양의 방향이지만 3-seed CI가 0을 포함하는 경우가 있다.

### 5.5 Adaptive Λ 기여 (그림 F5b, 표 5-F)

**표 5-F. Full controller − (Λ=0.5) variant, 방향 고정 = full minus fixed-Λ (disjoint).** 근거: `data/f5_lambda.csv`.

| 설정 | full − (Λ=0.5) | 95% CI | full 높은 seed |
|---|--:|:--:|--:|
| Schedule A | **−0.071 pp** | [−0.23, +0.09] | 2/5 |
| medium mobility | **+0.070 pp** | [−0.05, +0.19] | 3/3 |

두 CI 모두 0을 포함 → adaptive Λ의 독립적 정확도 이득은 확인되지 않는다. 별도의 fixed (λ,Λ) local grid 3-seed
결과(Schedule A +0.784 pp, mobility +0.56 pp)는 primary 5-seed와 분리해 보관하며(same-pool·다른 grid), Schedule A CI는
0을 포함한다.

---

## 6. Server non-Main rate와 TV: 7개 환경 (그림 F6)

**표 6-A. server non-Main rate − TV (disjoint, paired, 사전 정한 dataset 순서).** 근거: `data/f6_server_signal.csv`.

| 설정 | rounds | seeds | TV (%) | rate (%) | rate − TV (pp) | 95% CI |
|---|--:|--:|--:|--:|--:|:--:|
| SVHN temporal | 150 | 5 | 79.39 | 80.58 | **+1.187** | [+0.26, +2.11] |
| CIFAR-10 Schedule A | 150 | 5 | 65.84 | 66.45 | **+0.610** | [−0.14, +1.36] |
| CIFAR-10 gradual | 150 | 3 | 62.72 | 62.96 | +0.234 | [−0.74, +1.21] |
| medium mobility | 120 | 3 | 59.21 | 59.40 | +0.195 | [−1.07, +1.46] |
| Tiny-ImageNet | 100 | 3 | 23.93 | 23.86 | −0.074 | [−0.14, −0.01] |
| CIFAR-100 gradual | 150 | 3 | 36.10 | 35.41 | **−0.693** | [−1.02, −0.37] |
| CIFAR-100 spatial | 150 | 3 | 36.44 | 35.00 | **−1.440** | [−1.78, −1.10] |

Family 평균(mobility는 slow/med/fast를 먼저 family로 묶고 medium만 보유): CIFAR-10 temporal +0.42, SVHN +1.19,
mobility +0.19, Tiny −0.07, CIFAR-100 −1.07 → rate가 TV보다 높은 family 3/5. class 수 하나로 단정하지 않되, CIFAR-100
spatial에서 rate의 **absolute candidate가 최종 λ를 결정한 cluster-round 비율이 0.998**로 포화된 것이 한 원인이다
(warm-up+burn-in 이후 25 round부터, 분모=전체 cluster-round; `data/f6`와 round2 `hard_vs_tv_all_settings.csv`).

**표 6-B. Matched fixed reference (disjoint, 3 seeds).** 근거: `data/fixed_refs_transfer.csv`(재학습, fixed λ0.2·0.4,
Λ0.5). best = 두 λ 중 평균이 높은 쪽(모두 λ0.2).

| 설정 | best fixed(λ0.2) | TV − fixed | rate − fixed |
|---|--:|--:|--:|
| CIFAR-10 gradual | 62.86 | −0.13 | +0.10 |
| CIFAR-100 gradual | 35.91 | **+0.19** | **−0.50** |
| CIFAR-100 spatial | 36.44 | −0.00 | **−1.44** |
| Tiny-ImageNet | 24.31 | −0.38 | −0.46 |

Schedule A/mobility/SVHN의 disjoint fixed도 보유(A fixed λ0.4=65.17, mobility=58.58, SVHN 사전 고정 λ0.2=80.43).
**주목:** class가 많은 CIFAR-100에서는 adaptive TV가 best fixed와 대등하거나 근소 우위(spatial ≈0, gradual +0.19)인
반면, direct server rate는 fixed보다 낮다(−0.50, −1.44) — 즉 CIFAR-100에서 rate의 약점은 TV뿐 아니라 fixed 대비로도
확인된다. 다른 protocol의 same-pool 값으로 빈칸을 채우지 않는다.

---

## 7. Architecture·mobility·network·corruption·communication

- **Communication(§5.6, tensor 확인, `tables/communication_accounting.csv`·`overhead.csv`):** underlying SplitOMC의
  smashed activation 32,768 B/sample(=32 KiB), model exchange **12,149,112 B/round**(= 12.15 MB decimal = 11.59 MiB,
  overlap-2 config의 라운드 전체 교환량). **DriftGate 추가 통신 = client당 4 B(32-bit scalar)/round + edge consensus
  4·degree B/ES/round**, activation 추가 upload 0(server 복제가 client 로컬에 있어 p_s 계산에 추가 전송 없음).
  Probe forward 시간(probe_n=64): client fwd 0.83 ms, probe fwd 1.53 ms, 전체 16-신호 6.9 ms(overhead.csv).
- **Architecture/mobility/network/corruption:** 과거 same-pool 결과가 있으며(ResNet split, slow/fast mobility,
  delay/loss/topology, Gaussian corruption), 각 조건의 absolute accuracy·matched 차이·seed는 supporting 자료로
  `reproducibility/evidence_index.md`에 연결한다. 이들은 이번 검증의 disjoint 핵심 비교와 분리해 보조로 보고한다.
- **모바일 측정:** 실측 latency/energy는 이번 범위에 없다. payload는 위 tensor 기반 estimate로만 제시하고 실측값을
  채우지 않는다.

---

## 8. 본문에 사용할 대표 문장 (영문, 확인된 수치 범위 내)

- "Under a disjoint controller/evaluation split, DriftGate's TV-driven weighting improves integrated accuracy over the
  best fixed λ by +0.66 pp (5/5 seeds) on CIFAR-10 Schedule A and +0.63 pp (3/3) under medium mobility, and over an
  entropy controller by +1.47 and +2.00 pp."
- "The gain comes from adapting λ: freezing Λ at 0.5 leaves accuracy unchanged (−0.07 pp on Schedule A, +0.07 pp under
  mobility; both 95% CIs include zero), so we describe DriftGate as adaptive-λ-driven while noting the full controller
  also adapts Λ."
- "TV tracks the traffic shift ρ (Spearman 0.91 on Schedule A) whereas predictive entropy keeps falling with
  convergence and even anti-correlates after an abrupt shift (raw AUROC 0.16)."
- "A simpler server-only signal (the server's non-Main prediction rate) matches or beats TV where the class count is
  small and drift is real (SVHN +1.19 pp, Schedule A +0.61 pp) but is weaker on many-class CIFAR-100 (−1.44 pp
  spatial, −0.69 pp gradual), where its absolute view saturates."
- 반대 방향 문장(예: "TV always best", "adaptive Λ independently improves accuracy")은 근거가 없으므로 쓰지 않는다.

---

## 9. 남은 작업 (미확인 항목만)

1. **CIFAR-10 gradual·CIFAR-100 gradual/spatial·Tiny-ImageNet의 disjoint matched fixed reference** — fixed λ0.2·0.4
   재학습 24 run 실행 중(GPU idle 활용, 기존 작업 미방해). 완료 시 §6 표와 `table_settings.csv` 자동 갱신.
2. **Absolute-only controller의 disjoint 정면 비교(§6.2)** — `λ=λ_abs` 전용 arm이 코드에 미배선. 기존 same-pool의
   dual vs relative-only(개발 단계) 비교만 보유(표 5-D). 최종 component-removal 주장을 위해서는 Λ=0.5 고정하에 full
   three-comparison vs absolute-only의 disjoint 3-seed 비교가 필요.
3. **충분히 학습한 뒤의 closed-loop 적응(§6.3)** — 사전학습 checkpoint 재사용 조건의 downstream 정확도 비교가 별도
   필요(현재 post-convergence는 신호 품질 위주로 보유).
4. **실제 모바일 latency/energy 측정** — 장비 실험 필요(이번 범위 밖).

이상은 미확인 항목에 한한다. §2 평가 정의, §3~6 disjoint 수치, §5 signal/role/timing/Λ, §7 communication은 확인 완료.
