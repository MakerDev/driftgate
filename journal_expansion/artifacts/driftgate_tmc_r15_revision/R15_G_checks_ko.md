# Round 15 작업 P, G, F1: 구현과 문헌 확인

확인한 코드 위치, 결과 출처, 논문 위치를 적는다. 원문을 확보하지 못한 세부는 미확인으로 둔다.

## P1. 비교 규칙의 실제 연산과 원 논문의 관계

| 내부 ID | 실제 연산(코드) | 원 논문에서 가져온 부분 | 생략한 부분 | 원고 표시 |
|---|---|---|---|---|
| B0 | client exit entropy ≤ 0.8 nats이면 client exit, 아니면 edge exit(`src/runner_r6.py:eval_client`, `eval/evaluator.py:compute_entropy`) | SplitGP·SplitOMC의 entropy 기반 조기 종료 추론 규칙 | SplitGP 학습 알고리즘 전체(학습은 multi-cell personalized split learning) | Confidence-based offloading (SplitGP inference rule) |
| B3 | argmax (p_d + p_e)/2 | 없음 | | Probability average |
| R-PoE | argmax log p_d + log p_e(1e−8 floor) = 두 logit의 합의 argmax | FedRoD의 예측 결합(일반 head logit + 개인 head logit, 원문 §4.2 식 (9) 아래) | FedRoD의 balanced softmax 학습(식 (6)), 일반 head 위에 더해지도록 개인 head만 학습하는 방식, hypernetwork 변형(§4.3) | Logit sum |
| R-THE | 요청마다 entropy가 작은 exit(같으면 device) | 없음. FedTHE의 해가 아니다(P2) | | Lower-entropy exit |
| LEW (새 규칙) | 요청마다 w ∈ {0, 0.01, …, 1}에서 H(softmax(w z_d + (1 − w) z_e)) 최소 | FedTHE의 entropy 항(원문 §4.2: logit 가중합의 softmax에서 entropy) | feature alignment 항, cosine 유사도로 정하는 손실 비중, test history descriptor, e의 최적화 절차 | Logit-entropy weighting |
| R-ZTW | device 최대 확률 ≥ q이면 device, 아니면 logit sum. 주표는 B0과 같은 offloading 비율 | ZTW의 기하 평균 결합(식 (2))과 confidence 문턱 조기 종료(식 (3)) | cascade 연결(식 (1)), 학습한 exit 가중치 w_j, class별 bias b_i(class balancing), 결합 예측에 대한 문턱(우리는 device exit 단독의 최대 확률에 문턱) | Geometric ensemble with early exit |
| R-EM | own/other 두 그룹의 비율 r(Main 아닌 비율)을 EM으로 추정(Saerens 등의 EM을 두 그룹의 비율에 적용; 개별 class prior가 아님). 앞서 도착한 요청 8개 이상이면 그 요청들, 아니면 직전 round의 batch 추정. 업데이트 q_i = (r/b)p_e(O_k) / [((1−r)/a)p_e(M_k) + (r/b)p_e(O_k)], r ← mean q_i(`artifacts/driftgate_tmc_r13b_baselines/scripts/r13b_baselines_analysis.py: em_prefix, r_causal`) | Saerens 등(2002)의 EM prior 추정 | 개별 class prior 추정 | Label-shift EM (two-group) |
| R-LR | 9개 특징의 로지스틱 회귀로 P(Main), 답 argmax P p_d + (1 − P) p_e. correction 없음. target = Main 여부, loss = log loss(scikit-learn 기본, L2), 학습 자료 = 개발용 기록(Round 10, seed 5–7)의 round ≥ 30 요청 9,700,604개(`R13b_dev_LR.csv`) | 없음 | | Learned weight |

## P2. FedTHE 설명

- FedTHE 원문 §4.2(https://arxiv.org/html/2205.10920v4): 예측은 ŷ = e·ŷ^g + (1 − e)·ŷ^l이고 ŷ^g, ŷ^l은 두 head의 logit이다. entropy 손실은 이 logit 가중합의 softmax에서 계산한다. feature alignment 손실 e‖h'_n − h^g‖² + (1 − e)‖h'_n − h^l‖²와 cos(p(ŷ^g), p(ŷ^l))로 정하는 손실 비중이 함께 쓰인다. h'_n은 test feature와 history descriptor의 이동 평균이다.
- v27 §6.1의 "확률 혼합의 entropy는 w에 대해 오목하므로 한쪽 끝에서 최소"라는 논증은 확률의 산술 혼합에 대한 것이다. FedTHE는 logit을 섞으므로 이 논증으로 FedTHE의 해를 대신할 수 없다. 원고에서 고친다.
- feature alignment를 그대로 적용하지 못하는 이유: 두 exit는 같은 feature를 쓰지 않는다(device exit는 device block 출력, edge exit는 edge block 출력). FedTHE의 descriptor는 두 head가 공유하는 feature extractor의 출력 공간에서 정의되므로, 두 head의 descriptor를 같은 공간에서 비교할 수 없다. 공통 공간을 따로 정의하는 적응은 가능할 수 있으나 이번에는 하지 않았다.

## G. 원고에 필요한 구현·문헌 정보

1. **OOP, OOR**: SplitOMC 저장소 README(https://github.com/atifrizwan91/SplitOMC, commit f8a1ff4)는 "out-of-preference (OOP), out-of-region (OOR)"로 적는다. 저장소의 `Evaluator.py:get_ratio_based_data`는 OOP를 edge server의 class 범위(scope) 안에 있으면서 client의 학습 class가 아닌 class로, OOR을 그 밖의 class로 정의한다. 범위는 서버마다 무작위로 정한다. 우리 구현(`src/r6_requests.py` 1–16행)은 cell의 class 범위를 그 cell에 사는 device들의 Main class 합집합으로 정한다. 요청 수는 Main 요청 전부, OOP ρ|Main|, OOR 0.3ρ|Main|로, SplitOMC의 비율 구조(OOR = 0.3·OOP 비율)를 따른다.
2. **τ = 0.8 nats**: SplitOMC 저장소의 기본값(`main_SplitOMC.py:65`, `ethrange = [0.8]`, 탐색 범위 0.05–2.3)과 같고, 우리 설정 `configs/base_v3.yaml`의 `eth_default: 0.8`("SplitOMC reference-aligned config")이다. development에서 정한 값이 아니다. entropy는 두 구현 모두 자연로그다(nats).
3. **window**(`r13b_baselines_analysis.py: window_means`): 같은 device, 같은 평가 round에서 현재 요청보다 앞서 도착한 요청이 8개 이상이면 그 요청 전부(최대 길이 제한 없음, round 단위), 아니면 그 device가 마지막으로 요청을 받은 평가 round의 요청 전부를 쓴다. 둘 다 없으면 w = 0.5다. 현재 요청은 넣지 않는다. cell이 바뀌어도(handover) window를 비우지 않는다. 첫날 run에서는 평가 round 사이에 모델이 바뀌지만 window를 비우지 않는다. v27 Algorithm 1은 현재 요청의 entropy를 window에 넣은 뒤 w를 계산하고, 8개가 모일 때까지 w = 1/2라고 적는다. 구현과 다르므로 원고를 구현에 맞춘다.
4. **§5.3과 §5.4**: §5.3은 두 평균 entropy를 최근 offload된 요청으로 정의한다. §5.4는 H̄^e만 offload된 요청으로 계산한다고 다시 적는다. 부분 offloading에서 두 평균을 모두 offload된 요청으로 계산하도록(Round 15의 DriftGate-P) 문장을 맞춘다. Round 14의 곡선은 H̄^d를 window의 모든 요청으로 계산했으므로, 원고 곡선은 Round 15의 정의로 다시 만든다.
5. **w 평균 0.42**: S1, seed 0–4, 전체 구간, 모든 요청의 평균(Round 13b `R13b_Fauto_w.csv`, seed별 0.416–0.425). **EM 0.23과 실제 0.51**: Round 10 `R10_T3_r_estimation.csv`, 개발용 기록(seed 5–7), 밖에 있는 device, round ≥ 30 통합 0.230 대 0.510(전체 구간 0.233).
6. **3.2 µs**: Round 12 `R12_T5_fusion_cpu_time.csv`(C = 10에서 3.22 µs). 측정 조건은 다음과 같다.
   - CPU는 AMD Ryzen Threadripper PRO 7965WX(이 서버)이고, numpy float32로 계산했다.
   - Dirichlet로 만든 가짜 확률 20,000개를 요청 하나씩 처리했다(batch 1). 별도의 warm-up은 없고, 반복 1회의 평균이다.
   - 시간에는 prior correction(곱하고 정규화), 혼합, argmax가 들어간다. entropy 계산과 window 갱신은 들어가지 않는다.
   - 묶음으로 계산하면 요청당 39 ns다.
7. **π_k**: 평가 round와 같은 학습 round의 label 개수(`runner_r6.py` HK: local epoch 수 × device 데이터의 class 개수, 그 round에 학습한 device만)로 만든다. cell 분포와 전체 분포를 Λ = 0.5로 섞고, 두 cell이면 두 cell의 π를 평균한다. label 개수가 0인 cell은 전체 분포를 쓴다. a = π(M_k)를 [0.01, 0.99]로 자른다. 파라미터 평균은 확률 prior의 같은 선형 평균을 보장하지 않으므로, 원고는 "edge exit가 학습한 class 분포의 근사"로 적는다.
8. **요청과 seed**: 평가 요청은 시험 이미지 가운데 probe와 겹치지 않는 80%(`src/disjoint_pools.py`)에서 class마다 앞쪽 이미지를 고른다(결정적). 도착 순서는 `default_rng([seed, 10, round, device])`다. seed가 정하는 것은 Main class 분할, 모델 초기화(model_seed), S1 계열의 합성 이동 경로다. S2의 GeoLife 경로는 seed와 관계없이 같고(집에 있는 비율의 seed 간 표준편차 0), 인식 요청의 class 구성은 합성이다.
9. **하위 10%**: 규칙마다, 평가 round마다 다시 정렬해 아래쪽 ceil(0.1 n)명을 고른다. 같은 device 집합이 아니다. 하위 분위의 평균이 오른 것을 모든 device의 무손실로 해석하지 않는다.
10. **Kim 등(IEEE IoT J. 2025, DOI 10.1109/JIOT.2025.3601814)**: IEEE 원문은 열람하지 못했고 Semantic Scholar의 초록만 확인했다. 초록에 적힌 내용은 세 가지다.
    - pretraining 단계에서 개인화 SFL 학습과, 서버의 큰 모델을 위한 surrogate target을 만든다.
    - online learning 단계에서 서버의 큰 모델을 teacher로 하는 knowledge distillation으로 client 모델을 갱신해 label 분포 변화에 대응한다.
    - 실시간 추론에서는 client 모델만 거치는 early exit를 쓸 수 있다.
    - 문턱, 손실식, 비교 방법의 세부는 미확인이다. 이 방법은 학습을 바꾸는 방법이고, Round 15의 비교는 같은 학습된 모델에서 추론 규칙만 바꾸는 비교라 질문이 다르다.

## F1. Round 7 oracle과 APFL

- **oracle**: `runs/phaseT7_gate/r7_O_s{5,6,7}`, `--oracle_home_away`, 집 λ = 0.70, 밖 λ = 0.15(DriftGate controller의 λ 상한과 하한, `configs/base_v3.yaml`의 `lam_max`, `lam_min`), Λ = 0.5, 첫 라운드부터 적용. seed 5–7(개발용 seed)이다.
- **비교 대상**: 같은 seed의 고정 λ 0.4(`r7_F_s*`)이다. 추론 규칙은 runner의 기본 평가(B0, entropy 0.8)다.
- **기록**: 요청별 기록과 모델이 없으므로 oracle 모델에 DriftGate를 적용하려면 다시 학습해야 한다. 이번에는 하지 않았다.
- **APFL**: Round 6 `runs/phaseT6_S1/s1_apfl001_s{0,1,2}`(학습률 0.01) 62.30%, `s1_apfl010_s{0,1,2}`(학습률 0.1) 62.86%이다(B0 추론, 전체 구간). v27은 이 값을 고정 λ 0.4의 seed 0–4 평균 62.86%와 같다고 적었는데 seed 집합이 다르다. 같은 seed 0–2에서 고정 λ 0.4는 63.21%이고, APFL(0.1)은 0.35 pp 낮다.
