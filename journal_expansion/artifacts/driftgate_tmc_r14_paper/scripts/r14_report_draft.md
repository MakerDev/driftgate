# DriftGate Round 14 보고서: 원고 결과 절의 표와 그림

- 작성: 2026-10-04. 새 run은 돌리지 않았다.
- 입력은 세 가지다.
  - Round 12의 기록(run 47개: 고정 λ 0.4 run 31개, DriftGate의 λ 조절 run 16개)
  - Round 13b B부의 CIFAR-100 run 3개
  - Round 13b의 로지스틱 회귀 계수, Round 7과 Round 11의 기존 표
- 분석 스크립트 `scripts/r14_paper_analysis.py`는 결과를 보기 전에 commit `a0c1802`로 고정하고 push했다. 결과를 본 뒤에는 그림 세 개의 배치만 고쳤다(6절 12번).
- 산출물은 `journal_expansion/artifacts/driftgate_tmc_r14_paper/`에 있다.
  - 표 8개는 `tables/R14_T1_*.csv`부터 `R14_T8_*.csv`까지다.
  - 그림 6개는 `figures/`에 PDF와 PNG로 있다. 영문 caption은 `figures/R14_figure_captions.md`에 있다.
- 지시문대로 이 보고서에는 표 1, 4, 5만 옮겼다. 해석은 쓰지 않았다.

## 0. 이 보고서에서 쓰는 말

- 모든 값은 하루 전체 구간(모든 평가 라운드)의 정확도다.
  - 정확도는 평가 라운드마다 요청이 있는 클라이언트의 정확도를 같은 비중으로 평균하고, 평가 라운드에서 다시 평균한 값이다.
  - 표의 값은 seed 평균이고, 괄호 안은 seed 사이의 표준편차(ddof = 1)다.
  - 차이(예: DriftGate − SplitGP)는 같은 seed끼리 먼저 빼고 평균과 표준편차를 구했다.
- 규칙 이름은 원고의 이름을 쓰고, 괄호에 이전 라운드의 이름을 적었다.
  - SplitGP(B0)는 client exit의 entropy가 0.8 nats 이하이면 client exit의 답을, 넘으면 server exit의 답을 쓴다.
  - Device only(B1)는 client exit만 쓰고, Edge only(B2)는 server exit만 쓴다. Average(B3)는 두 exit 확률의 평균에서 답을 고른다.
  - FedRoD(R-PoE), FedTHE(R-THE), ZTW(R-ZTW), Label-shift EM(R-EM), Learned per-request weight(R-LR)는 지시문 3.1절의 규칙이다.
  - DriftGate는 지시문 3.2절의 규칙이고, Round 13b의 F-auto와 같다.
- server 사용 비율은 server exit의 출력이 필요한 요청의 비율이다.

## 1. 시작 전 확인 (지시문 1.1)

- 다시 계산한 B0, B3, DriftGate의 전체 구간 값을 Round 13b 표 1(B0, B3, F-auto)과 비교했다. 설정 10개 모두 0.05 pp 안에서 같았다. 최대 차이는 0.0047 pp다(`tables/R14_T0_start_check.csv`). pp는 정확도 백분율의 차이다.
- 같은 표에 나머지 규칙(FedRoD, FedTHE, ZTW, Label-shift EM, Learned per-request weight와 절제 규칙)의 차이도 참고로 적었다. 최대 차이는 0.0051 pp다.
- run마다 다시 계산한 B0는 run JSON의 정확도와 같다(`tables/R14_T0_runs.csv`).
  - M_k는 모든 평가 라운드에서 일관된다.
  - K = 500의 클라이언트 번호는 Round 12의 방법으로 되살렸다.

## 2. 표 1: 주 결과 (`tables/R14_T1_main.csv`, 긴 형식은 `R14_T1_main_long.csv`)

<<T1>>

- 마지막 행의 "best other baseline"은 설정마다 SplitGP를 뺀 기준선 여덟 개 가운데 평균이 가장 높은 규칙이다. 그 규칙의 이름을 칸 끝에 적었다.

## 3. 표 4: 선택적 offload (`tables/R14_T4_curves.csv`, `R14_T4_reach.csv`)

선택적 offload는 client exit의 entropy가 문턱을 넘는 요청만 edge로 보내는 방식이다. 아래 표는 server 사용 비율 0.1–1.0에서의 정확도다. 나머지 설정의 값은 CSV에 있다.

<<T4>>

## 4. 표 5: 하위 10% 클라이언트 (`tables/R14_T5_bottom10.csv`)

평가 라운드마다 클라이언트 정확도를 정렬해 아래쪽 ceil(0.1 × 클라이언트 수)명의 평균을 구하고, 평가 라운드에서 다시 평균했다.

<<T5>>

## 5. 나머지 표와 그림

| 산출물 | 내용 |
|---|---|
| `tables/R14_T2_ablation.csv` | 표 2: S1, client mobility, ResNet-18에서 절제 규칙 여섯 개 |
| `tables/R14_T3_home_away_kinds.csv` | 표 3: S1, S2에서 규칙 여섯 개의 집, 밖, Main, OOP, OOR 정확도 |
| `tables/R14_T6_time_of_day.csv` | 표 6: S1, S2의 시간대별 정확도와 집에 있는 클라이언트의 비율 |
| `tables/R14_T7_training_adjustment.csv` | 표 7: DriftGate의 답을 쓸 때 (λ 조절 run) − (고정 λ 0.4 run) |
| `tables/R14_T8_motivation.csv` | 표 8: S1에서 Device only와 Edge only의 집, 밖, 요청 종류별 정확도, client exit entropy의 AUROC, SplitGP의 server 사용 비율 |
| `figures/R14_fig1_motivation` | 그림 1: motivation (한 단 폭) |
| `figures/R14_fig2_device_oracle` | 그림 2: Round 7 기기 단위 oracle의 이득 (한 단 폭의 0.6) |
| `figures/R14_fig3_main` | 그림 3: 설정 10개의 DriftGate − SplitGP (두 단 폭) |
| `figures/R14_fig4_home_away` | 그림 4: S1, S2의 (집 정확도, 밖 정확도) 산점도 (두 단 폭) |
| `figures/R14_fig5_offload` | 그림 5: S1, S2의 선택적 offload 곡선 (두 단 폭) |
| `figures/R14_fig6_time_of_day` | 그림 6: S1의 하루 동안의 정확도와 집에 있는 클라이언트의 비율 (두 단 폭) |

## 6. 구현에서 정한 것

1. **Learned per-request weight의 계수**: `R13b_dev_LR.csv`에 유효숫자 6자리로 적힌 평균, 표준편차, 계수, 절편을 그대로 썼다. P(Main) = sigmoid(Σ 계수 × (특징 − 평균) / 표준편차 + 절편)이다. 개발용 기록에서 Round 13b의 값과 소수 둘째 자리까지 같았다(`precheck/smoke_development_records.txt`).
2. **DriftGate 곡선의 w**: 지시문 3.3절을 다음과 같이 구현했다.
   - 창은 3.2절과 같다. H̄_c는 창의 모든 요청으로, H̄_s는 창 안에서 edge로 보낸 요청(entropy > τ)으로만 구한다.
   - 창은 있지만 보낸 요청이 없으면, 같은 클라이언트의 직전 요청의 w를 쓴다. 직전 요청은 (평가 라운드, 도착 순서)로 본 바로 앞 요청이고, 평가 라운드를 넘어서도 이어진다. 직전 요청이 없으면 0.5를 쓴다.
   - 창이 없으면(첫 평가 라운드에서 앞선 요청이 8개 미만) 3.2절과 같이 0.5를 쓴다.
   - τ = 0인 끝점은 DriftGate와, τ = ∞인 끝점은 Device only와 같은 값이다(개발용 기록에서 확인).
3. **곡선이 SplitGP의 정확도에 처음 닿는 비율**: Round 13b의 `first_reach`를 썼다. 곡선의 점을 server 사용 비율 순으로 놓고, 이웃한 두 점 사이를 선형으로 보간했다. Device only의 정확도가 SplitGP 이상인 seed에서는 server 사용 0인 점이 이미 닿으므로 값이 0이다.
4. **ZTW**: 표 1의 값은 ZTW 곡선을 SplitGP와 같은 server 사용 비율(전체 구간)에서 보간한 값이다.
5. **시간대**: 평가 라운드 r의 시작 시각(05:00 + 6(r − 1)분, `src/r6_env.py`)으로 나눴다. 평가 라운드는 1, 5, 10, …, 150이다.
   - 시간대별 평가 라운드 수: 출근 전 6, 출근 4, 낮 13, 귀가 6, 저녁 2
   - 집에 있는 클라이언트의 비율: 그 평가 라운드에 요청이 있는 클라이언트 가운데 자기 집 cell에 있는 클라이언트의 비율
6. **하위 10%**: 그 평가 라운드에 요청이 있는 클라이언트만 정렬했다.
7. **AUROC (표 8)**: run마다 모든 요청에서 client exit의 entropy로 Main 아닌 요청을 가려내는 AUROC를 구했다. 같은 값은 평균 순위로 처리했다. Round 11의 값(개발용 기록, seed 5–7)은 참고로 같은 표에 적었다.
8. **표 7**: 같은 seed의 두 run끼리 뺐다. S1과 단계적 구성 변화는 seed 0–4, client mobility와 ResNet-18은 seed 0–2다.
9. **그림 2**: Round 7 표 `T2_paired_differences.csv`의 "O − F" 행(seed 5–7의 값)을 다시 그렸다. 막대는 평균, 오차 막대는 seed 사이의 표준편차, 점은 seed별 값이다. Round 7 표의 신뢰구간은 쓰지 않았다.
10. **그림 6**: 집에 있는 클라이언트의 비율은 오른쪽 축의 옅은 배경으로 그렸다. 시간대의 경계는 점선으로 표시했다.
11. **색**: 그림 전체에서 규칙마다 같은 색과 표지를 썼다. DriftGate는 붉은색(#d7263d)과 굵은 선이다.
12. **결과를 본 뒤 고친 그림 배치**
    - 그림 3: 설정 이름이 겹쳐서 긴 이름을 두 줄로 나눴다.
    - 그림 6: SplitGP와 Edge only의 선이 거의 겹쳐서 Edge only를 점선으로 바꾸고 규칙마다 표지를 달리했다.
    - 그림 2: seed별 점이 오차 막대와 겹쳐서 점의 간격을 넓혔다.
    - 고친 스크립트로 전체를 다시 계산했다. 표는 고치기 전과 모두 같았다.
