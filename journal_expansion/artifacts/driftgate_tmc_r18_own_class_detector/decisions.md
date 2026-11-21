# Round 18 결정 기록

지시문 0.2의 4번에 따라 스스로 내린 결정을 적습니다. 결정마다 날짜, 선택지, 고른 것, 근거를 적습니다.

## D1. 작업 이름과 위치 (2026-11-20)
- 선택지: 지시문의 이름(Round 18)을 그대로 쓰기, 또는 번호를 바꾸기.
- 고른 것: 지시문의 이름과 위치(`artifacts/driftgate_tmc_r18_own_class_detector/`, branch `r18-own-class-detector`)를 그대로 씁니다.
- 근거: 저장소에는 이미 다른 Round 18(`driftgate_tmc_r18_compare`, 고정 가중치 비교)이 있습니다. 폴더와 branch가 달라서 기존 산출물을 건드리지 않습니다. 보고서에는 두 작업을 구분해 적습니다.

## D2. 개발용 seed (2026-11-20)
- 선택지: 지시문의 S1 seed 5–7, 또는 쓰이지 않은 seed.
- 고른 것: S1 seed 8, 9, 10입니다.
- 근거: seed 5–7은 Round 7, 8, 10, 19, 20에서 개발용으로 쓰였습니다(지시문 2.4는 겹치면 바꾸라고 합니다). partition seed가 기록된 provenance 1,956개 가운데 seed가 8 이상인 run은 없습니다. 환경 파일은 기존 생성기로 만들었고, 생성기가 기존 seed 5 파일을 같은 배열로 다시 만드는 것을 확인했습니다.

## D3. cnn-scratch의 기존 run 재사용 (2026-11-20)
- 선택지: Round 15·17의 S1/S2 run을 재사용하기, 또는 새로 돌리기.
- 고른 것: 새로 돌립니다.
- 근거: 기존 run에는 held-out 분리, 판별기 학습, 둘째 날(300 라운드), 라운드 150의 판별기 상태가 없습니다. 이 기록 없이는 `edge-corrected-by-detector`와 `second-day` 칸을 계산할 수 없습니다.

## D4. 구현에서 정한 값 (2026-11-20)
- 판별기 학습(전체 표본 Adam 300 step, 학습률 0.01, L2 1e−3), 입력 표준화, cell 음성의 상한 8192개, 직전 요청이 16개 미만일 때 ŝ = 0.5, histogram 안의 선형 보간, MobileNetV2 입력 처리(ImageNet 정규화, 64×64 bilinear)를 결과를 보기 전에 정했습니다. 값은 `decision_rule.md` 4절에 있습니다.
- 양성 벡터: 지시문은 "그 라운드에 device k가 보낸 학습 표본의 벡터"라고 합니다. 구현은 device가 local 학습을 마친 모델(eval mode)로 자기 학습 표본을 forward한 block 출력을 씁니다. local 학습의 마지막 pass에서 보낸 smashed data와는 batch normalization의 mode와 epoch 안의 파라미터 변화만큼 다릅니다. exit 벡터는 그 라운드에 설치한 device exit(device가 edge에 올려 보내는 파라미터)로 계산합니다.
- cell 음성의 상한: hub cell에서는 한 device의 cell 음성이 2만 개를 넘습니다. 학습 시간과 GPU 메모리를 묶기 위해 8192개를 넘으면 seed를 고정한 무작위 부분집합을 씁니다. 양성과 음성의 비중이 같으므로 음성 분포의 기댓값은 바뀌지 않습니다.
- 검증 통계의 other: 다른 device의 held-out은 cell을 가리지 않고 모든 참여 device에서 모읍니다. edge 사이의 교환 byte를 비용에 넣습니다.
- 재생에서 batch normalization 통계가 바뀌지 않도록 판별기 모듈의 모든 forward를 eval mode로 고정했습니다(첫 시험에서 재생 run의 device exit 통계가 바뀌는 문제를 찾아 고쳤습니다).

## D5. 개발용 run을 먼저 띄우는 순서 (2026-11-20)
- 선택지: 1단계의 분석 수준 점검(oracle 대입, 기존 구현과의 일치)을 모두 끝낸 뒤 run을 띄우기, 또는 판단 기준과 학습 코드를 commit한 직후 개발용 run을 띄우고 그동안 분석 스크립트를 쓰기.
- 고른 것: 후자입니다. 개발용 run의 결과는 분석 스크립트와 점검 결과를 commit하기 전에는 읽지 않습니다.
- 근거: 300라운드 run 하나가 수 시간 걸리므로 벽시계 예산을 아낄 수 있습니다. 학습 코드의 점검(옵션을 끈 runner의 비트 동일성, 재생의 비트 단위 재현)은 run을 띄우기 전에 끝냈습니다.

## D6. 개선 시도 1·2번의 구현을 미리 준비 (2026-11-20)
- 선택지: 2단계 판정이 나온 뒤 구현하기, 또는 개발용 run이 도는 동안 미리 구현하고 시험해 두기.
- 고른 것: 미리 구현했습니다(`--other_class`, `--device_split stride16`, 기본값은 꺼짐). 실제로 쓸지는 2단계 판정(7.3절)에 따릅니다.
- 시도 1번의 구현에서 정한 값: device exit의 마지막 linear에 "other" 출력 하나를 더합니다. local 학습의 device exit 항은 (C+1)개 출력의 cross entropy이고, batch마다 실제 표본 수만큼 가상 exit 벡터를 M_k 밖의 class에서 균등하게 뽑아 "other"를 정답으로 하는 cross entropy를 γ 가중으로 더합니다. 가상 벡터는 직전 라운드에 edge가 모든 참여 device의 exit 벡터로 구한 class 평균과 대각 표준편차에서 뽑습니다(첫 라운드에는 없음). 판별기의 세 입력에는 모두 P(other)를 한 열로 더합니다. device 예측(확률, entropy)은 실제 class의 logit만 씁니다.
- 시도 2번: device block을 `features[0]`–`features[10]`(64×4×4, stride 16)으로 늘립니다. 다른 설정은 같습니다.
- 확인: 두 옵션 모두 2라운드 시험이 끝났고, 시도 1번은 재생이 학습 run을 비트 단위로 재현했습니다.

## D7. iNaturalist 자료 (2026-11-20, 결과를 보기 전)
- 선택지: (a) Hsu 등(ECCV 2020)의 지역별 분할, (b) 2021 `train_mini`와 그 위도·경도, (c) 2021 전체 `train`을 흘려 받으며 고른 사진만 저장, (d) CIFAR-100 superclass 대체 경로.
- 고른 것: (c)입니다. 자료 준비는 3단계를 할 경우에 대비해 2단계 run이 도는 동안 미리 합니다.
- 근거: (a)는 iNaturalist 2017 사진(약 186 GB)이 필요하고, 남은 디스크는 약 150 GB입니다. (b)는 종마다 사진이 50장뿐이어서, 북미 상위 100종을 모아도 5,000장이고 (지역, 종)별 사진 수의 중앙값이 0장입니다. 50대 device의 학습과 요청을 만들 수 없습니다. 2021 `train`은 종마다 152–300장이고 annotation에 위도·경도가 있습니다. 240 GB tar를 디스크에 저장하지 않고 읽으면서 고른 30,000장만 96×96으로 저장하면 디스크 문제가 없습니다. 이 선택은 지시문의 대체 경로("2021과 그 위도·경도")를 따르되 `train_mini` 대신 `train`을 쓰는 변경입니다.
- 구성에서 정한 값: 대륙은 관측이 가장 많은 북미(경도 −130…−50, 위도 7…84, 하와이 제외), 지역은 (위도, 경도 × cos 평균 위도)의 k-means 5개(n_init 10, random_state 0), 종은 다섯 지역의 사진 수 순서로 100종입니다. 사진 수가 상한 300에서 같은 종이 1,095개라서 동률은 numpy default_rng(18)의 순열로 정했습니다. 결과는 `tables/R18oc_inat_regions.csv`, `tables/R18oc_inat_species.csv`에 있습니다.
- 확인한 수치: 다섯 지역의 사진 수는 9,338, 12,278, 6,377, 1,056, 951장입니다. 지역 3과 4는 사진이 적습니다. 두 지역의 class 분포가 겹치는 정도(최솟값의 합)는 0.03–0.27입니다.

## D8. iNaturalist 설정의 구성 (2026-11-20, 결과를 보기 전)
- 지시문 6.4가 정하지 않은 값을 다음과 같이 정했습니다(`src/r18oc_inat.py`).
- 사진의 분할: (종, 지역)마다 30%를 요청용, 나머지를 학습용으로 나눕니다(사진이 2장 이상이면 요청용 최소 1장).
- cell과 지역의 대응: 하루 동안 device가 머문 횟수가 많은 cell부터, 100종의 사진이 많은 지역부터 짝을 짓습니다. S1에서는 hub cell이 가장 많이 방문되므로 사진이 가장 많은 지역이 hub에 대응합니다.
- 집: 지시문은 50대를 다섯 cell에 고르게 배정하라고 합니다. S1 환경은 cell마다 10대가 살므로 그대로 맞습니다. S2(GeoLife)는 경로를 그대로 쓰라는 조건과 함께 집 cell이 38/4/5/2/1대로 고르지 않습니다. 경로를 그대로 쓰는 조건을 우선하여 S2의 집 배정은 바꾸지 않습니다.
- 요청: 평가 라운드마다 device당 400개를 복원 추출합니다. 집에서는 90%가 own class(집 지역의 요청용 사진, 없으면 그 종의 모든 요청용 사진), 10%가 집 지역의 분포입니다. 밖에서는 현재 cell 지역의 분포이고, 두 cell에 속하면 요청마다 두 cell 가운데 하나를 같은 확률로 고릅니다.
- held-out: class별 10%(최소 1개)를 떼되, class마다 학습 표본을 1개 이상 남깁니다. CIFAR에서는 class마다 표본이 수백 개라 결과가 바뀌지 않습니다.
- 입력: 96×96 uint8을 GPU에 두고, device 모델이 255로 나눈 뒤 ImageNet 정규화를 합니다(확대 없음).

## D9. 중간 산출물과 데이터셋의 저장 위치 (2026-11-20, 사용자 지시)
- 사용자 지시: 중간 결과물과 데이터셋은 저장소가 아니라 `/home/honeynaps/8T/2_Data/3_DataPreprocessed` 아래에 폴더를 만들어 저장합니다.
- 선택지: (a) 돌고 있는 run을 멈추고 새 경로로 다시 띄우기, (b) 스크립트의 경로를 모두 새 폴더로 바꾸기, (c) 파일을 옮기고 저장소의 원래 경로에 링크를 두기.
- 고른 것: (c)이고, 폴더 이름은 `driftgate_r18_own_class_261120`입니다. run 출력(3.6 GB)과 분석 cache(84 MB)를 복사해 같음을 확인한 뒤 원래 경로를 링크로 바꿨습니다. 돌고 있던 run 8개의 로그는 열린 파일이라 run이 끝날 때까지 `runs/phaseT18oc.moving/logs`에 쓰이고, 새 폴더에서는 링크로 보입니다. run이 끝나면 로그를 새 폴더로 옮기고 `phaseT18oc.moving`을 지웁니다. iNaturalist 자료(1.6 GB, 추출 중)도 같은 방법으로 옮겼고, 추출 결과는 링크를 따라 새 폴더에 저장됩니다. 저장소 밖에 있는 원래 폴더는 지시문 9절에 따라 지우지 않고 `inat2021_r18.moved_to_8T`로 남깁니다.
- 근거: (a)는 3시간 넘게 돈 run 8개를 버립니다. (b)는 이미 `jobs.json`에 들어간 명령의 경로와 어긋납니다. run 출력은 저장소에 commit하지 않고, 마지막에 파일마다 sha256을 표로 남깁니다(Round 13b와 같은 방식).

## D10. 시험용 `decision_stage2.json`의 삭제 (2026-11-20)
- 상황: commit fd31a79에 가짜 cache로 표 스크립트를 시험할 때 나온 `decision_stage2.json`이 실수로 들어갔습니다. 이 파일의 판정은 실제 run의 결과가 아닙니다.
- 고른 것: 저장소에서 지우고(`git rm`, 기록은 history에 남음), 실제 판정 파일은 평가용 run이 끝난 뒤 같은 스크립트로 만듭니다.

## D11. `cnn-scratch`의 설계 선택 (2026-11-20 21:05, 개발용 seed 8–10, 3.5절)
- 선택지: 판별기 입력 3개 × 모델 2개 × 음성 표본 3개의 18개 조합입니다.
- 고른 것: `exit|mlp64|both`입니다.
- 근거: `frozen-replay`의 비율 0.5에서 새 방법의 seed 평균 정확도가 70.85%(SD 2.70)로 가장 높습니다. 2등인 `exit|mlp64|cell`은 70.66%로 0.19 %p 낮아 0.10 %p 동률 범위 밖이므로, 단순성 규칙을 적용할 후보가 없습니다. 가장 낮은 조합은 `exit|mlp64|global`의 67.82%입니다. 전체 표는 `tables/R18oc_design_selection_cnn.csv`에 있습니다.

## D12. 개선 시도(7.4)의 비교 방식 (2026-11-20, 평가용 seed의 결과를 보기 전)
- 선택지: 개선 시도의 run에서 18개 조합을 다시 골라 비교하기, 또는 그 모델 구성에서 3.5절로 고른 조합을 그대로 쓰기.
- 고른 것: 고른 조합을 그대로 씁니다. "기존 판"은 같은 모델 구성의 개발용 run(S1 seed 8–10, 변경 없음)이고, 두 판 모두 그 조합의 `frozen-replay` 비율 0.5 정확도를 같은 seed끼리 비교합니다. seed 평균 차이가 0.5 %p 이상이면 평가용 seed(S1·S2 seed 0–2)로 넘어가고, 그 run들로 2단계의 판정 칸 네 개를 다시 계산합니다(`r18oc_tables.py attempt_dev`, `attempt_cells`).
- 대상: 2단계 판정에서 행 2 또는 3에 해당하는 사전학습 구성입니다. 두 구성이 모두 해당하면 두 구성 모두에 시도합니다.
- 근거: 조합을 다시 고르면 비교에 선택 효과가 섞이고, 지시문 9.2의 설정 고정과도 어긋날 수 있습니다. 두 선택지가 비슷하면 단순한 쪽을 고르라는 원칙에도 맞습니다.

## D13. `pretrained-finetuned-block`의 설계 선택 (2026-11-20 23:55, 개발용 seed 8–10, 3.5절)
- 선택지: 판별기 입력 3개 × 모델 2개 × 음성 표본 3개의 18개 조합입니다.
- 고른 것: `exit|mlp64|both`입니다.
- 근거: `frozen-replay`의 비율 0.5에서 가장 높은 조합은 `concat|mlp64|both`의 72.09%(SD 1.96)이고, `exit|mlp64|both`는 72.07%(SD 1.95)로 0.02 %p 낮습니다. 0.10 %p 동률 범위 안의 후보가 이 둘이므로 단순성 순서(block < exit < concat)에 따라 `exit`를 고릅니다. 가장 낮은 조합은 `block|logreg|global`의 70.10%입니다. 전체 표는 `tables/R18oc_design_selection_ft.csv`에 있습니다.

## D14. stepwise change와 random mobility의 구현 (2026-11-21 00:10, 2단계 평가용 결과를 보기 전)
- 상황: v32는 두 설정을 옛 실행기(`src/runner.py`, `--schedule A`와 `--mobility --schedule abrupt`)로 돌렸습니다. 이 실행기에는 Round 18의 모델 구성, held-out 표본, 판별기, 요청 기록이 없습니다.
- 선택지: (a) 옛 실행기에 Round 18의 기능을 모두 옮기기, (b) 두 설정을 Round 6 실행기가 읽는 환경 파일로 만들기.
- 고른 것: (b)입니다(`r18oc_make_env.py stepwise`, `randmob`, 파일은 `runs/phaseT18oc_env/`). stepwise change는 옛 실행기의 고정 소속(`make_es_topology(50, 5, 50)`: cell 0–3의 device 5대씩이 다음 cell에도 속함)과 모든 device의 ρ = 0, 0.4, 0.8, 0.4, 0(30라운드씩)입니다. random mobility는 옛 실행기의 Gauss–Markov 모델(1 km 지도, edge 5개, α 0.9, 12 m/s, 라운드당 10초, v32와 같은 seed 42)과 5라운드마다 가까운 edge 2개로 바꾸는 규칙, ρ = 0(라운드 1–75)과 0.8(라운드 76–150)입니다. 하루는 150라운드이고 run은 이를 두 번 되풀이합니다(300라운드).
- 옛 실행기와 다른 점: 요청 수는 S1과 같은 Poisson 횟수이고, OOP는 v32 2절의 정의(현재 cell의 다른 device가 학습한 class)를 따릅니다. random mobility의 v32 run은 120라운드였지만 지시문 6.3에 따라 300라운드로 돌립니다.
- 근거: (a)는 학습, 기록, 분석 경로를 두 벌 두게 되어 다른 설정과 같은 코드로 비교한다는 조건이 깨집니다. 5라운드 시험 run이 두 환경에서 모두 끝났습니다. 환경 파일은 크기가 작은 입력이라 저장소에 둡니다(D9의 run 출력과 다름).

## D15. `pretrained-frozen-block`의 설계 선택 (2026-11-21 00:40, 개발용 seed 8–10, 3.5절)
- 선택지: 판별기 입력 3개 × 모델 2개 × 음성 표본 3개의 18개 조합입니다.
- 고른 것: `exit|mlp64|both`입니다.
- 근거: `frozen-replay`의 비율 0.5에서 가장 높은 조합은 `concat|mlp64|both`의 71.80%(SD 1.78)이고, `exit|mlp64|both`는 71.78%(SD 1.82)로 0.02 %p 낮습니다. 0.10 %p 동률 범위 안의 후보가 이 둘이므로 단순성 순서(block < exit < concat)에 따라 `exit`를 고릅니다. 세 번째인 `concat|mlp64|cell`은 71.64%로 범위 밖입니다. 가장 낮은 조합은 `block|mlp64|global`의 69.47%입니다. 전체 표는 `tables/R18oc_design_selection_fz.csv`에 있습니다.

## D16. 개선 시도 1·2번의 개발용 run을 미리 넣기 (2026-11-21 00:45, 2단계 평가용 결과를 보기 전)
- 상황: 7.3절에서 행 2나 3이 나오면 개선 시도를 해야 하고, 개발용 run 하나에 약 7시간이 걸립니다. 평가용 run이 끝난 뒤 시도를 시작하면 시도마다 반나절이 더 듭니다.
- 선택지: (a) 2단계 판정을 본 뒤에 시도 run을 넣기, (b) 지금 시도 1·2번의 개발용 run(S1 seed 8–10, 두 사전학습 구성, 고른 조합 `exit|mlp64|both`)을 job 목록의 뒤에 넣기.
- 고른 것: (b)입니다. dispatcher는 목록 순서대로 띄우므로 평가용 run이 먼저 돌고, 시도 run은 남는 GPU에서 돕니다.
- 지키는 것: 시도의 결과는 7.3절의 판정이 행 2 또는 3일 때에만 쓰고, 순서(1 → 2 → 3)와 판정 방식(D12)은 그대로입니다. 행 1, 4, 5가 나오면 이 run들은 보고서에 "쓰지 않음"으로 적습니다. 시도 run은 개발용 seed만 쓰므로 평가용 seed의 결과를 보고 설정을 고르는 일과 관계가 없습니다.

## D17. 2단계 판정 (7.3, 2026-11-21 16:15, `decision_stage2.json`, commit b4884cc)
- 결과: 두 사전학습 구성 모두 판정 칸 네 개(S1·S2 × `second-day`·`frozen-replay`)가 미달입니다. 새 방법은 네 칸 모두에서 A*에 닿지 못해 S = 0이고, G(0.5)는 `pretrained-finetuned-block`이 −0.02, −1.73, −0.81, −1.38 %p, `pretrained-frozen-block`이 −1.87, −1.96, −1.75, −1.77 %p입니다(S1 second-day, S1 frozen-replay, S2 second-day, S2 frozen-replay 순서).
- 여지(oracle이 통과): `pretrained-finetuned-block`은 S1·S2의 `frozen-replay`(oracle S 0.508, 0.414)에만 있고, `second-day` 두 칸(oracle S 0.000, −0.016)에는 없습니다. `pretrained-frozen-block`은 S1의 두 칸(oracle S 0.411, 0.485)에만 있고, S2의 두 칸(oracle S 0.000)에는 없습니다.
- 적용: 2.3절 표를 위에서부터 검사하면 두 구성 모두 행 3("미달인 칸이 있지만 여지는 있음")입니다. 두 구성이 같은 행이므로 D12에 따라 두 구성 모두에 시도 1 → 2 → 3을 합니다. 여지 없는 칸을 빼고 판정하므로, 판정 대상은 `pretrained-finetuned-block`의 S1·S2 `frozen-replay`와 `pretrained-frozen-block`의 S1 `second-day`·`frozen-replay`입니다. 일부 칸이 여지 없음이므로 최종 판정은 "조건부 권고" 이하입니다.
- 시도끼리는 쌓지 않습니다: 각 시도는 변경 없는 판(D12의 "기존 판")에 변경 하나만 더해 비교하고, 앞선 시도가 관문(+0.5 %p)을 넘었더라도 다음 시도에 포함하지 않습니다. 한 시도에서 한 구성의 판정 대상 칸이 모두 부분 통과 이상이 되면 그 시도와 그 구성으로 3단계를 합니다.
- 참고: `cnn-scratch`도 여섯 칸 모두 미달입니다(G(0.5) −0.20…−0.91 %p). `first-day`에서 G(0.5)가 −0.5 %p보다 낮은 칸은 6개입니다(S1 `cnn-scratch`, S1·S2·CIFAR-100 `pretrained-finetuned-block`, S2·CIFAR-100 `pretrained-frozen-block`).

## D18. 개선 시도 3번의 가설 (7.4의 3번, 2026-11-21 16:18, 결과를 보기 전에 적음)
- 진단(`scripts/r18oc_diagnose.py`, `tables/R18oc_stage2_diagnosis_*.csv`, commit b4884cc): 비율 0.5에서 envelope와의 차이는 순위보다 답 규칙에서 크게 납니다. 같은 답 규칙끼리 비교하면 판별기 순위는 `device-max-prob` 순위보다 `edge-raw`에서 −0.21…+1.33 %p, `edge-corrected-even`에서 −0.70…+0.07 %p입니다. 같은 `device-max-prob` 순위에서 `edge-corrected-even`을 device 출력과 섞는 `average-corrected-even`으로 바꾸면 `frozen-replay`에서 +1.06…+1.92 %p입니다. 새 방법의 답 규칙(`edge-corrected-by-detector`)은 device 출력을 쓰지 않으며, `edge-corrected-even`보다 0.15…0.58 %p만 높습니다.
- 요청 종류로 나누면: `frozen-replay`의 비율 0.25에서 새 방법이 Main 요청에서 얻는 정확도는 `pretrained-frozen-block` S1에서 85.7%이고, Main 요청을 하나도 보내지 않는 oracle(device 답과 같음)의 87.0%보다 낮습니다. 판별기가 보낸 Main 요청은 r(0.59 안팎) 때문에 다른 class 쪽으로 기운 edge 답을 받고, device가 맞혔던 답을 잃습니다. 섞는 규칙(`device-max-prob` + `average-corrected-even`)은 같은 칸에서 Main 89.8%입니다.
- 가설: 실패의 주된 원인은 보낸 요청의 답에서 device 출력을 버리는 것입니다. 판별기의 r로 보정한 edge 출력에 device 출력을 섞으면, 판별기 순위의 이점은 유지하면서 비교 방법들이 섞기에서 얻는 이득을 되찾습니다.
- 변경(답 규칙 하나, 이름 `average-corrected-by-detector`): 보낸 요청의 답을 argmax(w·p_device + (1 − w)·p_edge(r))로 바꿉니다. p_edge(r)는 새 방법과 같은 r(같은 θ, 같은 held-out 히스토그램, 같은 창과 clip)로 보정한 edge 출력이고, w는 v32의 방법(`average-corrected-even`)이 같은 비율에서 쓰는 가중치(`r17.dgp_weight`, 비율 1.0에서는 `r16.driftgate_weight`)입니다. 순위(g가 작은 요청부터), 판별기의 입력·모델·음성 표본·학습, 보내지 않은 요청의 답은 바꾸지 않습니다. 분석만 바뀌므로 새 학습 run은 없습니다.
- 관문과 평가: D12와 같습니다. S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 고른 조합(`exit|mlp64|both`)의 새 규칙과 `edge-corrected-by-detector`를 같은 seed끼리 비교해 평균이 0.5 %p 이상 오르면, 평가용 run(S1·S2 seed 0–2)의 기존 기록에서 판정 칸을 다시 계산합니다. 칸의 계산(envelope, A*, S, G, 여지)은 `r18oc_tables.cell`을 그대로 쓰고 새 방법의 곡선만 바꿉니다.
- 기대하는 수치: 개발용 seed에서 +0.8…+1.5 %p(두 구성 모두 관문 통과)를 기대합니다. 평가용 seed의 판정 대상 칸에서는 G(0.5)가 지금의 −1.38…−1.96 %p에서 −0.8…+0.3 %p로 오르고, S는 0에 머물 것으로 봅니다. 따라서 판정 대상 칸이 모두 부분 통과 이상이 될 가능성은 낮다고 봅니다.
- 순서: 시도 1·2의 관문을 먼저 계산하고, 그 뒤에 이 시도의 개발용 계산을 합니다. 이 가설과 기대 수치는 계산 결과를 본 뒤에 고치지 않습니다.

## D19. 개선 시도에서 여지를 판정하는 run (2026-11-21 16:25, 시도 1·2번의 결과를 보기 전)
- 상황: D17은 2단계 run에서 여지 있는 칸을 판정 대상으로 적었습니다. 시도 1·2번은 학습을 새로 하므로, 같은 칸의 oracle 곡선이 2단계와 다를 수 있습니다.
- 선택지: (a) 2단계 run에서 정한 여지를 그대로 쓰기, (b) 시도의 평가용 run에서 여지를 다시 계산하기.
- 고른 것: (b)입니다. `r18oc_tables.attempt_cells`(결과를 보기 전에 commit한 코드)가 이렇게 계산합니다. 시도 3번은 같은 run에서 답 규칙만 바꾸므로 oracle 곡선이 2단계와 같고, 여지도 2단계와 같습니다.
- 근거: 여지는 run의 device와 edge 출력에서 나오는 성질이므로, 판정하는 run과 같은 run에서 계산해야 앞뒤가 맞습니다. 이미 commit된 코드를 바꾸지 않는 쪽이기도 합니다.

## D20. 개선 시도 1번, `pretrained-finetuned-block`의 관문 (2026-11-21 18:17, `tables/R18oc_attempt1_ft_dev.{csv,json}`, `R18oc_attempt1_ft_dev_context.csv`)
- 결과: S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 새 방법의 정확도는 72.07%에서 66.64%로, 같은 seed끼리 −5.43 %p(SD 2.06; seed별 −7.81, −4.13, −4.36)입니다. 관문(+0.5 %p)을 넘지 못하므로 이 구성의 시도 1번은 평가용 run을 돌리지 않습니다(D12).
- 같은 run의 다른 수치: Device only는 70.24%에서 64.09%로, 그 run의 envelope B(0.5)는 74.54%에서 67.86%로 내려갔습니다. 판별기의 AUROC도 0.761에서 0.683으로 내려갔습니다.
- 해석: "other" 출력의 손실(γ = 0.5, 실제 표본과 같은 수의 가상 exit 벡터)이 finetuned 구성에서는 exit와 block을 함께 바꾸므로, 판별기뿐 아니라 device 모델 자체의 정확도를 낮춘 것으로 봅니다. 시도의 설정(γ, 가상 벡터의 수)은 결과를 본 뒤에 바꾸지 않습니다.
- 남은 일: `pretrained-frozen-block`의 시도 1번은 seed 9, 10의 cache가 생기면 같은 방식으로 판정합니다.

## D21. 개선 시도 2번, `pretrained-frozen-block`의 관문 (2026-11-21 18:36, `tables/R18oc_attempt2_fz_dev.{csv,json}`, `R18oc_attempt2_fz_dev_context.csv`)
- 결과: S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 새 방법의 정확도는 71.78%에서 74.28%로, 같은 seed끼리 +2.50 %p(SD 0.06; seed별 +2.46, +2.57, +2.48)입니다. 관문(+0.5 %p)을 넘으므로 D12에 따라 평가용 run(S1·S2 seed 0–2와 각 replay, 12개, `r18oc_jobs.py attempt_eval 2`)을 job 목록에 넣었습니다.
- 같은 run의 다른 수치: Device only는 66.45%에서 73.06%로, 그 run의 envelope B(0.5)는 74.01%에서 77.31%로 올랐습니다(B(0.5) 규칙은 두 run 모두 `device-max-prob`/`logit-sum`). 판별기의 AUROC는 0.865에서 0.894로 올랐습니다.
- 해석: device 쪽 층이 늘어 device 모델이 강해졌고, 이득은 새 방법과 비교 방법에 함께 돌아갑니다. 개발용 seed에서 새 방법과 그 run의 B(0.5)의 차이는 −2.23 %p에서 −3.03 %p로 오히려 벌어졌으므로, 평가용 seed의 판정 칸이 부분 통과 이상이 될 가능성은 낮다고 봅니다. 그래도 관문은 D12의 규칙대로 적용합니다.
- 순서: 시도 2번의 평가용 run이 도는 동안 `pretrained-frozen-block`의 시도 1번과 `pretrained-finetuned-block`의 시도 2번 관문을 계산합니다. 3단계에 쓰는 시도는 D17대로 1 → 2 → 3의 순서에서 판정 대상 칸이 모두 부분 통과 이상인 첫 시도입니다.

## D22. 개선 시도 1번, `pretrained-frozen-block`의 관문 (2026-11-21 19:49, `tables/R18oc_attempt1_fz_dev.{csv,json}`, `R18oc_attempt1_fz_dev_context.csv`)
- 결과: S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 새 방법의 정확도는 71.78%에서 70.94%로, 같은 seed끼리 −0.85 %p(SD 0.29; seed별 −0.55, −0.85, −1.13)입니다. 관문(+0.5 %p)을 넘지 못하므로 평가용 run을 돌리지 않습니다(D12). 이로써 시도 1번은 두 구성 모두에서 관문을 넘지 못했습니다(D20).
- 같은 run의 다른 수치: Device only는 66.45%에서 64.98%로, 그 run의 B(0.5)는 74.01%에서 72.97%로, 판별기의 AUROC는 0.865에서 0.851로 내려갔습니다.
- 해석: block을 고정한 구성에서도 "other" 출력의 손실은 exit의 정확도를 조금 낮추고, 판별기의 구별 능력은 높이지 못했습니다. finetuned 구성(D20)보다 떨어진 폭이 작은 것은 손실이 exit에만 닿기 때문으로 봅니다.
- 다음: 시도 3번은 D18의 순서대로 `pretrained-finetuned-block`의 시도 2번 관문을 계산한 뒤에 관문을 계산합니다. 그 전에 시도 3번의 cache(`r18oc_attempt3.py run`)를 미리 만들어 둡니다. `run`은 새 방법의 곡선이 2단계와 같은지만 출력하고, 시도 3번의 정확도는 출력하지 않습니다.

## D23. 개선 시도 2번, `pretrained-finetuned-block`의 관문 (2026-11-21 20:37, `tables/R18oc_attempt2_ft_dev.{csv,json}`, `R18oc_attempt2_ft_dev_context.csv`)
- 결과: S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 새 방법의 정확도는 72.07%에서 72.74%로, 같은 seed끼리 +0.67 %p(SD 0.56; seed별 +1.20, +0.70, +0.09)입니다. 관문(+0.5 %p)을 넘으므로 D12에 따라 평가용 run 12개(`r18oc_jobs.py attempt_eval 2`)를 job 목록에 넣었습니다. seed 하나(+0.09)는 관문 아래이고 평균도 관문에 가깝지만, 관문은 seed 평균으로 정했으므로 그대로 적용합니다.
- 같은 run의 다른 수치: Device only는 70.24%에서 72.47%로 올랐고, 그 run의 B(0.5)는 74.54%에서 74.57%로 거의 같습니다. 판별기의 AUROC는 0.761에서 0.804로 올랐습니다. 개발용 seed에서 새 방법과 그 run의 B(0.5)의 차이는 −2.47 %p에서 −1.83 %p로 줄었습니다.
- 해석: frozen 구성(D21)과 달리 envelope는 오르지 않고 판별기의 구별 능력과 새 방법만 올랐습니다. 그래도 차이가 −1.83 %p로 남아 있어, 평가용 seed의 판정 칸이 부분 통과(G(0.5) ≥ +1.0 %p 또는 S ≥ 0.15) 이상이 될 가능성은 낮다고 봅니다.
- 다음: 시도 1·2번의 관문이 모두 계산되었으므로, D18의 순서대로 시도 3번의 관문을 계산합니다.

## D24. 개선 시도 3번의 관문과 판정 칸 (2026-11-21 20:38, `tables/R18oc_attempt3_{ft,fz}_{dev,cells}.{csv,json}`)
- 관문: S1 개발용 seed 8–10의 `frozen-replay`, 비율 0.5에서 `pretrained-finetuned-block`은 +1.09 %p(SD 0.45; seed별 +1.19, +0.59, +1.48), `pretrained-frozen-block`은 +0.99 %p(SD 0.78; +0.41, +0.67, +1.87)입니다. 두 구성 모두 관문을 넘었고, D18에서 기대한 +0.8…+1.5 %p 안에 있습니다.
- 판정 칸(평가용 seed, 2단계 run의 기록, envelope와 여지는 2단계와 같음): 판정 대상 칸의 G(0.5)는 `pretrained-finetuned-block`의 S1·S2 `frozen-replay`가 −0.96, −0.36 %p(2단계 −1.73, −1.38), `pretrained-frozen-block`의 S1 `second-day`·`frozen-replay`가 −0.97, −1.20 %p(2단계 −1.87, −1.96)입니다. S는 `pretrained-frozen-block`에서 0이고, `pretrained-finetuned-block`에서는 새 방법이 A*에 닿지만 envelope보다 늦게 닿아 음수(−1.98, −0.72)입니다. 네 칸 모두 미달이므로 시도 3번으로는 3단계에 가지 않습니다.
- D18의 기대와 비교: G(0.5)는 기대 범위(−0.8…+0.3 %p) 안에 한 칸(−0.36), 아래에 세 칸(−0.96, −0.97, −1.20)이 있어 기대보다 조금 낮습니다. "S는 0에 머물고 판정 대상 칸이 모두 부분 통과 이상이 될 가능성은 낮다"는 예상은 맞았습니다.
- 해석: 보낸 요청의 답에 device 출력을 섞으면 2단계 격차의 절반 정도(0.8–1.0 %p)를 되찾지만, envelope에는 이미 같은 섞기(`average-corrected-even`)와 `logit-sum`이 들어 있으므로, 판별기 순위의 이점만으로는 envelope를 넘지 못합니다.
- 다음: 7.3절의 표에서 남은 것은 시도 2번의 평가용 칸입니다. 시도 2번의 판정 대상 칸이 한 구성에서라도 모두 부분 통과 이상이면 그 구성과 시도 2번으로 3단계를 합니다. 그렇지 않으면 시도 3번까지 미달이므로 행 5에 따라 3단계를 건너뛰고 4단계로 가며, 판정은 "중단 권고"입니다. 시도끼리는 쌓지 않습니다(D17).

## D25. 개선 시도 2번, `pretrained-frozen-block`의 판정 칸 (2026-11-22 02:00, `tables/R18oc_attempt2_fz_cells.{csv,json}`)
- 판정 대상 칸: D19에 따라 여지는 시도 2번의 평가용 run에서 다시 계산했습니다. 여지가 있는 칸은 S1 `frozen-replay`와 S2 `frozen-replay`이고, 2단계에서 여지가 있던 S1 `second-day`는 이 run에서 여지가 없습니다(oracle S −0.364).
- 결과: G(0.5)는 S1 `frozen-replay`가 −2.77 %p(SD 0.88; 2단계 −1.96), S2 `frozen-replay`가 −1.82 %p(SD 0.93; 2단계 −1.77)이고, S는 두 칸 모두 0입니다. 두 칸 모두 미달이므로 이 구성의 시도 2번으로는 3단계에 가지 않습니다. 여지가 없는 S1 `second-day`의 G(0.5)는 −3.10 %p(2단계 −1.87)입니다.
- 같은 칸의 다른 수치: 비율 0.5에서 새 방법의 정확도는 S1 `frozen-replay`에서 72.17%에서 74.23%로, S2 `frozen-replay`에서 75.08%에서 77.71%로 올랐습니다. 그러나 envelope B(0.5)가 74.14%에서 77.00%로, 76.84%에서 79.53%로 더 많이 올랐습니다(규칙은 2단계와 같은 `device-max-prob` 계열).
- D21의 예상과 비교: 개발용 seed에서 보인 "이득이 비교 방법에도 함께 돌아가 격차가 벌어진다"는 양상이 평가용 seed에서도 그대로 나타났습니다.
- 다음: `pretrained-finetuned-block`의 시도 2번 평가용 run(12개)이 끝나면 같은 방식으로 판정합니다. 그 구성의 판정 대상 칸이 모두 부분 통과 이상이 아니면 D24대로 4단계로 가며, 판정은 "중단 권고"입니다.
