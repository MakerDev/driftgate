# Round 15 주장과 근거의 대응

각 주장에 근거 표와 그림, 성립하는 조건, v28에서 고친 표현을 적는다. 값은 seed 평균이고, 차이는 같은 seed끼리 뺀 값의 평균(표준편차)이다.

| # | 주장 (v28 표현) | 근거 | 성립 조건 | v27에서 바뀐 점 |
|---|---|---|---|---|
| 1 | 두 exit는 다른 요청에서 정확하다(Device only: Main 86.3, OOP 7.5; Edge only: Main 72.3, OOP 40.8) | 그림 1, `R15_A1_splits.csv` | 첫날과 round > 30 모두(round > 30: Main 87.1 대 75.8, OOP 8.6 대 42.3) | 없음 |
| 2 | device entropy는 Main 아닌 요청을 약하게 구분한다(AUROC 0.64) | `R15_E_scale.csv`(S1 0.645) | 첫날, 요청 전체 | "구분하지 못한다"를 "약하게 구분한다"로 |
| 3 | Confidence-based offloading은 Edge only와 비슷하다 | 표 4, `R15_A1_rules_gt30.csv` | 첫날 62.87 대 62.91, round > 30 64.80 대 64.85 | "Device only보다 1.3 pp 낮다"는 첫날에만 성립해서 삭제 |
| 4 | location-aware 학습 정책에서 home +3.7, away −1.7 pp가 함께 관찰되었다 | 그림 2(Round 7 T2), `R15_G_checks_ko.md` F1 | S1, seed 5–7, B0 추론 | 인과("visitors pay") 삭제. 2×2 통제나 focal visitor 실험은 하지 않음(F2 미실행) |
| 5 | 첫날 DriftGate는 8개 설정 모두에서 평균이 가장 높다(strongest 대비 0.1–1.9 pp) | 표 4, `R15_A1_rules_full.csv` | 첫날(학습 초반 포함), 11개 규칙 | 규칙 10 → 11(Logit-entropy weighting 추가). ResNet-18의 0.10(0.61)은 구별 불가로 명시 |
| 6 | 첫날 우위의 상당 부분은 학습 초반에서 온다 | `R15_A1_time_of_day.csv`, 표 5 | S1: 출근 전 6/31 시점이 DriftGate − B0 5.45 중 3.39, strongest 대비 0.95 중 0.58을 차지 | 새 주장 |
| 7 | round > 30에서 8개 중 6개 설정에서 1위, 우위 −0.21 ~ +1.14 pp | 표 5, `R15_A1_rules_gt30.csv` | round > 30(평가 round 35 또는 40부터) | 새 주장 |
| 8 | 학습된 모델 재생에서 S2는 1위(+0.64), S1은 Probability average보다 −0.33 pp(5위) | 표 6, 그림 4, `R15_A2_replay.csv` | 재학습한 S1(5 seed), S2(3 seed), 150 라운드 종료 모델, 같은 하루 재생 | 새 주장. v27의 "아침이 저녁처럼 될 것"이라는 예상을 대체 |
| 9 | DriftGate는 집에 있는 device와 Main 요청에서 얻고 밖과 OOP/OOR에서 잃는다 | 그림 5, 표 7, `R15_A1_splits.csv` | 첫날, S1·S2. S1 재생에서도 같은 방향(home +1.0, away −2.0 대 Probability average) | away 정확도와 OOP/OOR 정확도를 구분해서 서술 |
| 10 | 첫날 측정한 threshold 가운데 β ≤ 0.5에서 모든 full-offload 규칙을 넘는 설정은 5개다 | `R15_C1_offload_evidence.csv` | 첫날, 원고 §5.3의 정의(DriftGate-P), 측정점 | v27의 7개 → 5개. round > 30에서는 2개, S1 재생은 닿지 않음, S2 재생은 0.51 |
| 11 | 첫날 offloading 0.3 이상에서 geometric ensemble보다 높다(8개 설정) | `R15_C1_offload_evidence.csv` P−geo 열 | 첫날만. round > 30에서는 S2, partial participation, random mobility만 | 조건 명시 |
| 12 | 온라인 controller(β = 0.5)로도 DriftGate가 결합 규칙 가운데 가장 높다 | `R15_C2_online_controller.csv` | S1·S2 첫날, S2 재생. S1 재생에서는 Probability average보다 0.2 pp 낮음 | 새 주장. sweep과 controller를 구분 |
| 13 | prior correction은 첫날 1.2–3.8 pp를 더한다(ResNet-18 0.2 pp) | 표 8, `R15_B1_correction_weight.csv` | 첫날. S1 재생에서는 correction을 빼면 +0.5 pp | "없으면 작동하지 않는다" 삭제, Bayes 가정 명시 |
| 14 | adaptive w는 더 나은 고정 w에 가깝다 | 표 8, `R15_B2_sensitivity.csv` | 첫날 w = 0.5와 ±0.2 pp. w = 0.2는 ResNet-18에서 −2.6, S1·S2 재생에서 +0.6·+0.1 | "고정 w로는 안 된다"를 크기에 맞춰 수정 |
| 15 | window 길이(8/32/128)는 정확도를 0.02 pp 이하로 바꾼다 | `R15_B2_sensitivity.csv` | 전체 offloading과 온라인 β = 0.5 | 새 주장. r 민감도는 실용 안정성으로만 서술 |
| 16 | 규모가 커져도 Confidence-based 대비 +3.1 ~ +5.4 pp. strongest 대비 우위는 K = 500에서 −0.27 pp | 표 9, `R15_E_scale.csv`, `R15_E_scale_reach.csv` | 첫날, cell 수를 device 수에 비례해 늘린 배치 | strongest 열 추가. 서버 처리량 측정이 아님을 명시 |
| 17 | 하위 10%의 평균도 오른다 | 표 10, `R15_A1_bottom10.csv` | 첫날, 규칙과 round마다 다시 선정 | 개별 device의 무손실이 아니라고 명시 |
| 18 | DriftGate의 추가 계산은 요청당 3.2 µs다 | §5.5, Round 12 `R12_T5_fusion_cpu_time.csv` | NumPy, Threadripper PRO 7965WX 코어 하나, 요청 하나씩 | 측정 조건 명시 |
| 19 | 모바일 지연과 에너지 | 없음 | 측정하지 않음 | TODO 유지. 비용 모형의 가정값임을 명시 |
