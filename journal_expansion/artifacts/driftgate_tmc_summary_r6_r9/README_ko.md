# DriftGate Round 6–9 결과 묶음

논문 방향을 논의하는 데 필요한 Round 6–9 결과를 모은 묶음이다. run 원본, 학습 코드, 요청별 원본 기록은 넣지 않았다. 원본과 스크립트는 저장소 `MakerDev/driftgate`의 `journal_expansion/artifacts/`에 있다.

## 읽는 순서

1. `DriftGate_R6_R9_summary_ko.md`: 다섯 단계(Round 6, R0, Round 7, 8, 9)의 결과와 판정을 한 문서로 정리했다. 0절(결론 요약)과 3절(라운드를 가로질러 보이는 사실)을 먼저 읽으면 된다.
2. 수치를 확인하려면 `reports/`의 라운드별 보고서를 본다. 정의, 판정 규칙, 불일치 기록, run manifest가 들어 있다.
3. 판정 값은 `decisions/`, 그림은 `figures/`, 표는 `tables/`에 있다.

## 파일

| 경로 | 내용 |
|---|---|
| `reports/DriftGate_R6_report_ko.md` | Round 6(이동 시나리오, 규모, 견고성, 지연 시간)과 R0 |
| `reports/DriftGate_R7_gate_report_ko.md` | Round 7(기기 단위 학습 λ) |
| `reports/DriftGate_R8_check_report_ko.md` | Round 8(추론 비율과 τ, server 사용 비율을 맞춘 비교) |
| `reports/DriftGate_R9_routing_report_ko.md` | Round 9(요청마다 exit를 고르는 규칙) |
| `decisions/R7_decision.json`, `R8_decision_v2.json`, `R9_decision.json` | 미리 정한 규칙의 판정 값 |
| `figures/R6_*` | λ 궤적, 시간대별 정확도, 규모와 견고성, 지연 시간, 시나리오 |
| `figures/R7_*` | 집과 밖의 λ_k, 시간대별 정확도 |
| `figures/R8_*` | 정확도와 server 사용 비율 곡선, probe 수에 따른 AUROC와 G |
| `figures/R9_*` | 배치 A 규칙별 정확도, 배치 B 곡선 |
| `figures/*captions*.md` | 그림의 영문 caption |
| `tables/R6_*`, `tables/R0_*` | Round 6 대표 결과, 시간대 기여, cell·시간대별 가장 좋은 고정 λ, 규모, 견고성, 지연 시간, R0 비교, 논문용 수치 |
| `tables/R7_*` | run별 값, 같은 seed 차이, λ, AUROC, 시간대 분해 |
| `tables/R8_*` | G, 비교 지점의 값, τ = 0.8의 값, exit별 정확도, 같은 정확도의 server 사용, 통제, 시간대 분해, probe |
| `tables/R9_*` | 배치 A와 B의 결과, 나눈 정확도, oracle 점, 가려내기 표 |
