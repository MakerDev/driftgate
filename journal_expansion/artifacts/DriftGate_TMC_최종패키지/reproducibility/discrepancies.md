# Discrepancies — 프롬프트 대조값 vs 원시 재계산

프롬프트의 수치는 과거 원고의 **대조용 기록**이다. 아래는 원시 run에서 재계산해 달라진 항목과 원인·올바른 값·영향이다.
차이가 없는 항목은 "재현"으로 짧게 적는다. 재계산은 모두 `scripts/build_data.py` + round-1/2 스크립트.

## 재현(값 일치, 그대로 사용 가능)

- **Primary paired (표 3-B):** TV−fixedλ0.4 A **+0.664**, mob **+0.629**; TV−entropy A **+1.470**, mob **+1.995**. 프롬프트와 일치.
- **Absolute view 기여 (표 5-D):** C100 gradual **+4.02**(프롬프트 ≈+4.0), C100 spatial **+3.40**(≈+3.4), Tiny **+1.85**(≈+1.9). 일치.
- **Λ 기여 (표 5-F):** full−(Λ=0.5) A **−0.071**, mob **+0.070**. 부호·크기 일치(2회 검증). CI 모두 0 포함.
- **Weight timing gradual/spatial (표 5-E):** gradual +0.46/+0.49/+0.81, static spatial +0.33/+0.01/+0.05. 프롬프트와 정확 일치.
- **Server rate − TV, 7환경 (표 6-A):** SVHN +1.187, A +0.610, C10gsig +0.234, mob +0.195, Tiny −0.074, C100gsig −0.693,
  C100sp −1.440. 프롬프트와 일치. C100sp abs-active fraction 0.998도 재확인.
- **TV passive Spearman/AUROC (표 5-A):** A Spearman +0.907, AUROC 0.978; abrupt AUROC 1.000; entropy abrupt AUROC 0.161.
  프롬프트와 일치. entropy direction-free 0.839.
- **역할 standard:** TV−ρ +0.866 (프롬프트 +0.84). 일치(3-seed).

## 차이(원인·올바른 값·영향)

### D1. 역할 ablation TV−ρ correlation (표 5-C, 그림 F4a) — 일부 다름, 결론 유지
- 프롬프트: same-role −0.44, indep +0.08, weak +0.21.
- 재계산(same-pool 3 seeds, `phaseR_role`): same-role exits **−0.213±0.028**, indep **+0.151±0.330**, weak **+0.019±0.412**.
- 원인: 프롬프트 값은 다른 seed 수/run 집합(1-seed 진단 가능성)에서 나온 것으로 보임. 재계산은 3-seed same-pool.
- 영향: **정성적 결론 동일**(standard +0.87 ≫ 교란 role 모두 ≤ +0.15). indep/weak은 seed 간 분산이 커(±0.33/±0.41)
  개별 값의 신뢰구간이 넓다. 표·그림에 3-seed mean±SD를 그대로 싣고 프롬프트 숫자는 쓰지 않는다.

### D2. Weight timing — Schedule A만 다름 (표 5-E, 그림 F5a)
- 프롬프트: Schedule A global mean +0.90, shuffled +1.24.
- 재계산(matched source = replay가 파생된 `d2_dual_A`, 3 seeds): global +0.41, per-cluster +0.44, shuffled +0.42.
- 원인: gradual/spatial은 정확히 일치 → 내 pairing(control을 그 파생 원본 trajectory와 짝지음)이 정합적. Schedule A의
  프롬프트 값은 다른 adaptive source(예: dvsig_tv_A)나 다른 metric window에서 나온 것으로 추정.
- 영향: **matched-source 3-seed 값(+0.41~+0.44)을 보고**한다. 정성적 결론(시간 환경에서 adaptive가 mean/shuffle보다
  양의 방향, static spatial은 timing 무관) 유지. 프롬프트 A 값은 사용하지 않는다.

### D3. Entropy Schedule A Spearman (표 5-A) — REPLACED
- 프롬프트: entropy Spearman "≈0.50".
- 재계산: Schedule A **+0.437**, abrupt **−0.578**(schedule에 따라 부호가 뒤집힘).
- 원인: 단일 대표값으로 요약 불가(방향이 schedule 의존). 영향: 본문에서 A/abrupt를 구분해 보고(REPLACED).

## Protocol 전환 (차이가 아니라 조건 변경, 혼동 주의)

- **High-drift 신호 격차:** 과거 same-pool "TV−entropy @ρ=0.8 = +3.64 pp"는 disjoint 5-seed에서 **+6.71 pp**로 재계산.
  이는 오류 정정이 아니라 protocol(같은 pool→disjoint)·seed(→5) 변경이다. 두 값을 하나의 비교로 섞지 않는다(표 4-A는 disjoint).
- **F6 절대 정확도의 protocol:** 표 6-A의 TV/rate는 disjoint. matched fixed reference 중 A/mob/SVHN만 disjoint 보유,
  나머지 4개는 재학습 중(§additional_runs). same-pool 값으로 채우지 않는다.

## 결론
헤드라인(§3), Λ 비교 방향(§5.5), entropy 정의(§2.3·§5.1), 평가 방식(§2)에 영향을 주는 **오류는 없다**. D1·D2는
보조 분석의 대조값 차이로, 원시 재계산 값으로 대체해 보고하며 정성적 결론은 유지된다. D3은 단일 요약 불가로 REPLACED.
