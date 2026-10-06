# BTFL과의 비교 준비 (Round 18 D)

## 1. 확인한 1차 자료

- 논문: Zhou & Liu, "BTFL", KDD 2025. arXiv 2503.06633(HTML v1 전문과 부록 포함)을 읽었습니다.
- 공식 구현: github.com/ZhouYuCS/BTFL, commit `ada7b55`. 주된 파일은 `pcode/local_training/BTFL_worker.py`(아래에서 W:줄번호)와 `pcode/master.py`입니다.
- 원문과 코드를 읽고 정리하는 작업은 별도 에이전트가 했고, 그 내용을 이 문서에 옮겼습니다. 판정에 쓴 값은 모두 원문 수식 번호와 코드 줄번호로 확인할 수 있습니다.

## 2. BTFL의 추론 방법 (원문과 코드)

| 항목 | 내용 |
|---|---|
| 전제 구조 | FedAvg로 학습한 전역 feature extractor V와 전역 head w_g가 있습니다. 개인 head w_p는 고정된 V 위에서 cross-entropy로 따로 학습한 선형 head입니다(Sec. 4.1, W:150-219). 두 head는 **같은 feature z = f(x; V)**를 입력으로 받습니다. 원문은 head 앞에 ReLU가 있다고 가정합니다(App. B). |
| 학습 단계에서 준비하는 것 | H̄_l과 H̄_g는 두 head의 학습 데이터 평균 entropy입니다. local DLE는 z의 각 차원을 이진화(round(tanh z))한 값의 빈도표이며, 서버는 이를 평균해 global DLE를 만듭니다. |
| 요청마다 하는 계산 | (a) Y_g와 Y_l, 그리고 두 entropy를 구합니다. (b) local DLE와 global DLE로 우도 Q_l, Q_g를 구합니다(Eq. 6). (c) HBU: τ = Q_l/Q_g와 entropy 조건으로 사건을 세어 Beta(α, β) 사전 분포를 갱신합니다(Eq. 9–11). (d) CBU: τ̂ = q̂_l^{exp((H_l − H̄_l)/H̄_l)} / q̂_g^{exp((H_g − H̄_g)/H̄_g)}입니다(Eq. 14–15). (e) DPI: e = ∫ m̂·pdf(m̂ \| z) dm̂를 구하고, Y = e·Y_g + (1 − e)·Y_l로 답합니다(Eq. 19–21). |
| 요청 사이에 유지하는 상태 | (α, β)만 유지합니다. H̄는 학습 단계의 값으로 고정됩니다. |
| 기울기 갱신 | 없습니다(optimization-free). |
| 기본값 | 원문은 λ = 16을 유일한 하이퍼파라미터로 둡니다. 코드의 기본값은 U = 5, q_level = 16, kb = 1이고, 실행 스크립트는 q_level = 2, kb = 4(CNN), U는 5–3000을 씁니다. |

원문과 코드는 다음 점에서 다릅니다.

1. 코드는 확률이 아니라 log 확률을 섞습니다(W:422-426).
2. 코드의 u_g는 지수에 H̄_g 대신 ln 100을 씁니다(W:292).
3. 코드에는 τ̂를 kb·tanh로 압축하는 단계가 있지만, 원문에는 없습니다(W:302).
4. EXD와 IND 사건을 α와 β에 배정하는 방식이 원문 Eq. 11과 코드에서 반대입니다. 코드의 배정이 m = P(EXD)라는 정의와 맞습니다.
5. 가지치기 규칙이 다릅니다. 원문은 합이 3이 되게 하고, 코드는 U/10 + 2가 되게 합니다.
6. 코드는 DLE에 평활화(0.6·P + 0.2)를 쓰고, global DLE를 자기 자신을 뺀 평균으로 만듭니다.
7. 코드는 HBU를 batch 단위로 갱신하고, 시험 세트가 바뀔 때마다 초기화합니다.
8. 코드는 학습할 때 feature에 L1 벌점을 더합니다. 원문에는 이 벌점이 없습니다.

원문은 현재 표본의 사건을 자기 예측 전에 반영하는지 밝히지 않습니다. 코드는 예측한 뒤에 반영합니다.

## 3. DriftGate의 split 모델에 적용할 때의 차이

| 항목 | BTFL | DriftGate의 split 모델 |
|---|---|---|
| feature | 모든 client가 공유하는 전역 V의 출력 z | device마다 다른 block h_k의 출력. h_k는 λ = 0.4로 자기 모델과 cell 평균을 섞은 개인 모델입니다 |
| 두 head의 입력 | 같은 z에 붙은 선형 head 두 개 | 같은 h_k(x)를 받지만, device exit와 edge block + edge exit는 서로 다른 비선형 경로입니다 |
| global DLE | 다른 client들의 z 빈도표의 평균 | 다른 device의 빈도표는 h_j의 좌표에서 만든 것이라 h_k의 좌표와 대응하지 않습니다 |
| class prior 보정 | 없음 | edge 출력을 device의 own class 비율로 보정합니다 |
| 경로와 offloading | 두 head가 모두 client에 있음 | 두 exit가 device와 edge에 나뉘어 있고, 일부 요청만 offload합니다 |
| 결합 가중치의 근거 | feature 우도와 entropy 사건의 Beta 사후 분포 | 최근 offload 요청에서 두 exit의 평균 entropy 비율 |

이 차이 때문에 이번 비교는 'BTFL inference adaptation'입니다. 원 방법의 성능이라고 부르지 않습니다. 특히 다른 device들의 h_j에서 만든 빈도표를 global DLE로 쓰므로, BTFL의 공통 feature 가정은 지켜지지 않습니다.

## 4. 적용한 adaptation (S1·S2 재생, 기본값 한 번)

구현은 `scripts/r18_btfl.py`이고, 결과는 `tables/R18_BTFL_adaptation.csv`에 있습니다.

- **head:** 개인 head는 device exit p_d이고, 전역 head는 보정하지 않은 edge exit p_e입니다(두 cell이면 기록된 평균 logit).
- **feature:** z는 f_d(device block 출력의 공간 평균, 128차원)입니다. ReLU 뒤의 max pool 출력이므로 음수가 없습니다.
- **DLE:** local DLE는 device k의 학습 표본(h_k)으로 만들고, global DLE는 다른 49개 device의 local DLE를 평균합니다(코드와 같음).
- **학습 데이터의 평균 entropy:** H̄_l과 H̄_g는 Round 15 학습 종료 checkpoint로 device k의 학습 표본에 forward pass를 해서 구했습니다. 이때 edge는 home cell의 server 모델을 썼습니다. 새 학습은 하지 않았습니다.
- **계산 순서:** 원문의 수식을 따르되, 원문이 정하지 않았거나 수치적으로 계산할 수 없는 부분은 코드를 따랐습니다(평활화, 적분 구간 [0.01, 0.99], α에 EXD를 세는 방식). 상태는 device마다 하루의 시작에서 α = β = 1로 두고, 요청마다 예측한 뒤에 갱신합니다. 가지치기는 λ = 16(원문)이고, 답은 확률을 섞어서 냅니다(원문 Eq. 21).
- **확인:** 학습 데이터의 분할과 학습 표본 feature가 Round 17 기록과 정확히 같았습니다(최대 차이 0).

| 설정 | BTFL adaptation | DriftGate | 차이 | Probability average | strongest reference (β = 1) |
|---|---|---|---|---|---|
| S1 재생 | 70.18 (1.33) | 71.95 (1.56) | −1.77 (0.34) | 72.27 | corrected edge only 72.74 |
| S2 재생 | 74.16 (2.90) | 76.54 (2.52) | −2.38 (0.66) | 75.90 | correction + dev w 0.2 76.66 |

- **가중치 e의 분포:** edge head의 가중치 e는 평균 0.35(S1)와 0.40(S2)이었습니다. Main 요청(0.38, 0.43)에서 non-Main 요청(OOP 0.30–0.34, OOR 0.26–0.29)보다 컸습니다.
- **우도비의 구별력:** −log(Q_l/Q_g)로 non-Main을 가린 AUROC는 0.69(S1)와 0.71(S2)입니다.
- **feature 이진화:** 요청 feature에서 값이 1인 차원의 비율은 0.70–0.78이었습니다.

## 5. 비용

결합을 device에서 하면 edge는 C개의 logit을 보냅니다. edge에서 하면 device는 p_d(C개 값)와 이진화한 feature(128 bit, 16 B)를 보내고, edge는 DLE 표 두 개(각 128개 값)를 보관합니다. 공통으로 필요한 것은 H̄_l, H̄_g, (α, β)입니다. global DLE를 만들려면 각 device의 빈도표(128개 값)를 서버로 모아야 하고, 학습 데이터에서 H̄_l과 H̄_g를 구하는 forward pass가 한 번 필요합니다.

## 6. faithful 비교에 필요한 것 (이번에는 실행하지 않음)

BTFL을 그대로 비교하려면 모든 device가 같은 좌표의 feature를 갖도록 학습해야 합니다. 이를 위한 최소 설정은 다음과 같습니다.

1. S1과 S2(각 seed 0–2)에서 device block을 모든 device가 공유하는 전역 모델로 학습하고(개인화 없음), device exit만 고정된 block 위에서 개인 head로 따로 학습한 run. cell 평균만 공유하는 λ = 0은 cell마다 좌표가 달라지므로 이 조건을 만족하지 않습니다.
2. BTFL 코드의 L1 feature 벌점과 기본 하이퍼파라미터(U, q_level = 2, kb)를 그대로 쓴 추론

이 경우 DriftGate의 개인화 정도(λ = 0.4)와 학습 조건이 달라지므로, 같은 run 안에서 DriftGate도 다시 평가해야 합니다. FedTHE와 ZTW의 행도 원 방법 전체가 아니라 각 방법의 추론 규칙을 가져온 비교라는 점을 원고에 유지합니다.
