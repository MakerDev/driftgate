# DriftGate TMC — 그림 가이드 (한국어)

> 모든 그림은 `scripts/make_figures.py`(F1,F3–F6)와 `scripts/make_fig02_schematic.py`(F2)가 `data/*.csv`에서
> 재생성한다. PDF는 vector(embedded font, `pdf.fonttype=42`), PNG는 300 dpi preview. 색: TV `#0072B2`,
> entropy `#D55E00`, fixed λ0.4 `#4D4D4D`, fixed λ0.2 `#999999`, server non-Main `#009E73`. 실제 기록된 round에만
> marker를 찍고 그 사이 선은 시각적 연결이다(smoothing·spline 없음). 실제 삽입 크기에서 육안 검증 완료.

전체 재생성:
```
cd artifacts/driftgate_tmc_export/scripts
python build_data.py && python make_figures.py && python make_fig02_schematic.py
```

---

## F1 — 신호가 traffic 변화와 연결되는 모습
- **파일:** `figures/fig01_signal_dynamics.pdf` / `.png`  · 실제 크기 **181.6 × 90.1 mm (double-column)**.
- **독자의 질문:** 지속 학습 중 TV와 entropy의 방향이 왜 다르게 나타나는가?
- **panel:** (a) Schedule A, (b) post-convergence abrupt. 각 열 위=실제 ρ, 아래=같은 model trajectory·같은 요청에서
  계산한 TV와 `H/ln C`(entropy 정규화). warm-up 15 round는 옅은 음영.
- **가장 중요한 확인:** TV는 ρ 상승 시 오르고 복귀 시 내려가 drift를 추적한다. entropy는 ρ와 무관하게 수렴에 따라
  단조 감소하며, abrupt에서는 ρ가 0.8로 뛰어도 계속 감소(수렴사).
- **source data:** `data/f1_signal_dynamics.csv` (rec_A / rec_abrupt passive fixed-weight, 3 seeds).
- **조건·n·오차:** passive fixed-weight backbone(fixed λ, 동일 model trajectory), 3 seeds; band = seed 간 SD(ddof=1).
  correlation/AUROC 수치는 본문 표 5-A(warm-up 이후, drift-active ρ≥0.5)에서 제공한다.
- **영문 caption:** *Signal dynamics under continued training. Top: traffic-shift level ρ. Bottom: client–server TV
  (blue) and normalized predictive entropy H/ln C (orange), computed on the same fixed-weight model trajectory and the
  same requests (3 seeds; bands are seed SD). TV rises and falls with ρ, whereas entropy declines monotonically with
  convergence and even anti-correlates after the abrupt shift in (b). Warm-up rounds are shaded.*
- **배치:** double-column. Introduction 또는 Motivation에서 "entropy가 왜 부족한가"를 설명하는 위치. **Main 추천.**

## F2 — 시스템 도식
- **파일:** `figures/fig02_system_overview.pdf` / `.png`  · **145.4 × 79.0 mm (double-column 폭에 여유 있게 배치)**.
- **독자의 질문:** 어떤 정보가 어디에서 계산되어 어느 model mixing을 바꾸는가?
- **panel:** (a) 온-디바이스 dual-exit: 입력→client block→{client exit p_c} 및 {feature→server block+exit p_s}→TV.
  (b) 2개 serving cluster + 중첩 client(c3), client별 TV→cluster signal→DriftGate controller(temporal·spatial·absolute,
  scalar 교환)→λ/Λ mixing→weight sharing. 실선=data/prediction·signal 흐름, 파선=parameter sharing. controller로
  label·ρ가 들어가지 않음을 범례에 명시.
- **가장 중요한 확인:** client probability는 server block을 통과하지 않고 병렬 계산된다(코드 일치). λ=local↔cluster,
  Λ=cluster↔global. cluster 수(2)는 예시.
- **source:** 코드 구조(`eval/evaluator.py`, `src/runner.py`, `src/controllers/self_calibrating.py`).
- **영문 caption:** *DriftGate overview. (a) On-device dual exits: the client block feeds a client exit p_c and,
  through the server block, a server exit p_s; their total variation D_TV is the unlabeled drift signal. (b) Per-cluster
  control: client TV values are averaged into a cluster signal that the controller turns, via temporal/spatial/absolute
  views, into λ (local↔cluster) and Λ (cluster↔global) mixing weights. Solid = data/prediction and signal flow, dashed
  = parameter sharing; no labels or ρ enter the controller. Two clusters shown as an example.*
- **배치:** double-column, Method 첫머리. **Main 추천.**

## F3 — traffic 변화 → λ 적응 (accuracy-vs-round panel 제거됨)
- **파일:** `figures/fig03_lambda_adaptation.pdf` / `.png`  · **181.2 × 88 mm (double-column)**.
- **독자의 질문:** traffic 변화가 cluster별 λ 조절로 어떻게 이어지는가?
- **panel:** 왼=Schedule A(5 seeds), 오=medium mobility(3 seeds). (a) ρ, (b) TV·entropy의 실제 λ trajectory와
  fixed λ=0.4 기준선(seed 내 cluster 평균 후 seed 평균). **accuracy-vs-round panel은 제거**했다(사용자 요청).
  정확도 결과는 numbers doc 표 3-A/3-B에서 읽는다.
- **가장 중요한 확인:** TV는 ρ가 오르면 λ를 낮춰(generalize) drift에 대응하고 복귀 시 올린다. entropy는 λ를 ~0.63으로
  올려 고정(역방향). mobility에서는 membership 변화(~R60) 후 TV가 λ를 0.4 아래로 내린다.
- **source:** `data/f3_rho.csv`, `data/f3_traj_lambda.csv` (disjoint t1_* runs). `f3_traj_acc.csv`는 표 근거로 보존.
- **조건·n·오차:** disjoint protocol; A 5 seeds / mobility 3 seeds; λ는 매 round(seed 내 cluster 평균 후 seed 평균).
- **영문 caption:** *Traffic-shift to λ adaptation under the disjoint protocol. Left: CIFAR-10 Schedule A (5 seeds);
  right: medium mobility (3 seeds). (a) ρ. (b) per-cluster-averaged λ for TV and entropy with the fixed λ=0.4
  reference. TV lowers λ as ρ rises and restores it afterwards, while entropy raises λ regardless. (Accuracy is
  reported in the results tables.)*
- **배치:** double-column, Method/Evaluation의 메커니즘 설명. **Main 추천.**

## F4 — 역할 차이와 absolute view 기여
- **파일:** `figures/fig04_mechanism_and_views.pdf` / `.png`  · **179.9 × 70.9 mm (double-column)**.
- **독자의 질문:** TV가 왜 의미가 있고, relative view만으로 충분한가?
- **panel:** (a) server-role ablation의 TV–ρ Spearman(horizontal dot; standard는 강조색, 나머지 회색; same-role exits는
  SD가 작아 오차막대 짧음), (b) absolute view 추가의 accuracy 기여(full−relative-only, 95% CI).
- **가장 중요한 확인:** 역할을 분리해야 TV–ρ 상관이 유지된다(standard +0.87 vs 교란 role ≤ +0.15). absolute view는
  many-class 전이 설정에서 +1.85~+4.02 pp 기여.
- **source:** `data/f4_role.csv`(same-pool 3 seeds), `data/f4_absolute_contribution.csv`(개발 단계 dual vs main, same-pool 3 seeds).
- **조건·n·오차:** (a) 3-seed mean±SD(same-pool). (b) 개발 단계 controller 비교(dual=TV+absolute, main=relative-only
  ST·delta_hard — 신호도 다름); 95% CI. 두 panel의 두 방법 absolute accuracy는 표 5-C/5-D.
- **영문 caption:** *(a) Server-role ablation: TV–ρ Spearman correlation across role perturbations (same-pool, 3 seeds;
  mean±SD). Separating the exits' roles is necessary for TV to track ρ. (b) Contribution of the absolute view: full
  controller minus a relative-only variant on transfer settings (development-version comparison, same-pool, 3 seeds;
  95% CI).*
- **배치:** double-column, Method/Analysis. **Main 추천**(단 (a),(b)의 protocol 차이를 caption에 유지).

## F5 — weight timing과 Λ 기여
- **파일:** `figures/fig05_timing_and_lambda.pdf` / `.png`  · **179.6 × 69.0 mm (double-column)**.
- **독자의 질문:** 좋은 평균 weight를 고른 효과와 시간에 맞춰 조절한 효과를 구분할 수 있는가?
- **panel:** (a) adaptive − {global mean λ, per-cluster mean λ, shuffled λ}(Schedule A/gradual/static spatial, dot+CI,
  0 기준선), (b) full − (Λ=0.5) paired difference(Schedule A/mobility, 방향 = full minus fixed-Λ).
- **가장 중요한 확인:** static spatial의 이득은 좋은 per-cluster 평균에서 오고(per-cluster/shuffled≈0) 시간 조절이
  아니다. adaptive Λ의 독립 기여는 확인되지 않는다(두 CI 0 포함).
- **source:** `data/f5_timing.csv`(same-pool replay, 3 seeds), `data/f5_lambda.csv`(disjoint).
- **조건·n·오차:** (a) same-pool replay(weight 순서를 바꿔 실제 재학습), 3 seeds, 95% CI. (b) disjoint, A 5 / mob 3
  seeds, 95% CI. 서로 다른 protocol을 하나의 pooled effect로 합치지 않는다. CI가 0을 포함하면 독립 기여의 확정으로
  해석하지 않는다.
- **영문 caption:** *(a) Weight-timing controls: adaptive minus mean-matched and shuffled λ policies (same-pool replay
  that retrains with the altered weight order; 3 seeds, 95% CI). The static-spatial gain comes from good per-cluster
  means, not timing. (b) Adaptive-Λ contribution: full controller minus a Λ=0.5 variant (disjoint; 5/3 seeds). Both
  CIs include zero.*
- **배치:** double-column. 공간이 부족하면 (b)는 표 5-F로 옮기고 (a)만 single-column으로 export 가능. **Main 추천**(또는 (a)만).

## F6 — 더 단순한 server-only 신호와의 비교
- **파일:** `figures/fig06_server_signal_comparison.pdf` / `.png`  · **85.8 × 77.0 mm (single-column, 89 mm 이내)**.
- **독자의 질문:** server non-Main rate가 TV를 대체할 수 있는가?
- **panel:** 7개 환경의 `rate − TV` paired difference와 95% CI(forest). 0 기준선; rate 우위=green, TV 우위=blue.
  정렬은 사전 정한 dataset/scenario 순서(SVHN→…→CIFAR-100 spatial).
- **가장 중요한 확인:** class 수가 작고 drift가 실재하는 설정(SVHN/Schedule A/CIFAR-10 gradual/mobility)에서는 rate가
  TV와 같거나 높고, class가 많은 CIFAR-100에서는 낮다. 대체가 아니라 조건부 대안.
- **source:** `data/f6_server_signal.csv` (disjoint; A·SVHN 5 seeds, 나머지 3 seeds).
- **조건·n·오차:** disjoint, per-seed paired, 95% CI(n=3 CI 넓음). absolute accuracy·matched fixed·entropy는 표 6-A.
- **영문 caption:** *Direct server-only signal vs dual-exit TV across seven settings (disjoint, paired; 95% CI). The
  server non-Main prediction rate matches or beats TV where the class count is small and drift is real (SVHN, CIFAR-10)
  but is weaker on many-class CIFAR-100. Green = rate higher, blue = TV higher; n per row.*
- **배치:** single-column, Discussion(신호 단순화 논의). **Main 추천.**

---

## Supporting 후보 (필요 시)
- `supp05_fixed_grid`: 실제 fixed λ별 integrated accuracy curve + adaptive 기준선(`tables/canonical_scheduleA_fixed_grid.csv`).
- corruption/network/split-point: 숫자가 적으면 그림 대신 compact table(표 7절 근거)로 제공. 모바일 실측이 없으므로
  mobile overhead 그림은 만들지 않는다.

## 검증 메모
- 모든 PDF를 실제 삽입 크기에서 육안 확인: 잘린 label·겹친 legend 없음(F3 panel label·F6 legend 위치 조정 완료).
- 삽입 후 일반 text ≥ 7 pt(tick/legend 7.5–8, axis label 9, panel label 10). 89/181 mm 기준 충족.
- plot 값과 Markdown 값은 동일 `data/*.csv`에서 나오므로 일치(별도 반올림/subset/smoothing 없음).
