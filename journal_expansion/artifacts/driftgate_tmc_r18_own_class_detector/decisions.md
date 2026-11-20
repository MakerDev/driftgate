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
