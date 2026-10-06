# Round 19 5절: 세 날의 frozen replay 프로토콜 (결과 전에 저장)

0단계의 판정에서 상태 C가 성립했습니다(`tables/R19_state.json`). C가 성립한 설정은 모두 학습이 진행되는 첫날 기록이었고, 모델이 고정된 재생에서는 DriftGate가 초기 관측 후 고정한 w보다 낮았습니다. 그래서 학습이 끝난 모델로 서비스하는 동안 이동이 계속될 때, DriftGate의 가중치가 고정 w보다 나은지 확인합니다. 이 실험은 **모델 파라미터를 고정한 여러 날의 replay**입니다. 계속 학습하는 운영이나 새 학습 seed의 검증으로 해석하지 않습니다. 새 학습은 하지 않습니다.

## 1. 모델, 날짜, seed

- **모델:** Round 15의 학습 종료 checkpoint입니다(S1 seed 0–4, S2 seed 0–2). device와 edge 모델은 정상적으로 함께 학습된 쌍입니다.
- **날짜:** seed마다 세 날을 연속으로 재생합니다. 날짜마다 새 경로와 요청 stream을 쓰고, 날마다 결과를 보고 checkpoint나 파라미터를 바꾸지 않습니다.
- **S1의 새 날짜:** 기존 이동 모델(`r6_env.build_synthetic("S1", env_seed)`)을 쓰고, env_seed = 9000 + 10·seed + day로 둡니다. 이 seed들은 이전에 쓰지 않았습니다.
  - home cell과 device 배치는 위상으로 고정되어 있어 checkpoint와 같습니다(스크립트에서 확인).
  - Main class도 같습니다. partition seed를 run seed로 두기 때문입니다.
  - 이동 계획, 속도, 요청 수, 참여는 새로 뽑습니다.
- **S2의 새 날짜:** 같은 GeoLife 사용자 50명의 다른 평일을 씁니다.
  - day d에는 원래 날을 뺀 후보 평일 가운데 d번째로 점수가 높은 날을 씁니다. 후보 규칙과 점수는 `scripts/r6_trace_env.py`와 같습니다.
  - 다른 평일이 부족한 사용자는 있는 날을 돌아가며 쓰고, 하나도 없으면 원래 날을 반복합니다. 그런 사용자는 day 1·2·3에 각각 1, 3, 6명입니다(`runs/phaseT19_multiday/env/S2_day_selection.csv`).
  - 위치 변환의 원점, edge 위치, membership 규칙, home cell, class group은 원래 파일과 같습니다. 요청 수는 env_seed = 9000 + 10·seed + day로 뽑습니다.
- **요청 생성:** Main/OOP/OOR 생성 규칙은 기존과 같습니다. Main 비중을 바꾸지 않습니다. 한 device가 같은 cell에서 받는 요청 이미지는 날짜가 달라도 같습니다. 이는 기존 요청 생성 규칙 때문입니다.
- **재생 명령:** 날짜마다 Round 15 재생과 같은 flag로 runner를 실행합니다(`scripts/r19_multiday_replay.py`).

## 2. 날짜 사이의 상태

분석(`scripts/r19_multiday.py`)에서 세 날을 이어 하나의 stream으로 만듭니다. 순서는 (날짜, round, 도착 순서)입니다.

- DriftGate의 window, β = 0.5 controller의 history, DriftGate-P 가중치는 날짜 사이에 초기화하지 않습니다.
- 초기 관측 후 고정 w는 첫날의 처음 128개 offload 요청으로 정하고, 셋째 날 끝까지 유지합니다.

## 3. 비교와 행

- **비교:** 같은 checkpoint, 요청 stream, offload mask에서 seed끼리 paired로 비교합니다.
  - DriftGate
  - development w(0단계에서 S1 development seed로 고른 값)
  - 초기 관측 후 고정 w
  - corrected edge only(w = 0)
  - Round 11의 w = 0.2
  - 고정 w 전체 곡선(0–1, 간격 0.05)
  - 사후 진단: device 상수, 사후 최고 고정 w
- **β:** 1과 0.5입니다.
- **행:** 세 날 전체, 날짜별(1, 2, 3), 둘째·셋째 날, 전환 device-round와 나머지, home/away입니다.
  - 전환 device-round는 device의 cell 집합이 직전 평가 round(평가 간격 한 개, 날짜 경계 포함)와 다른 device-round입니다.
  - 한 device-round는 한 번만 셉니다.

## 4. 판정 (기술적 기준)

비교 대상 X(development w, 초기 관측 후 고정 w, corrected edge only)마다 다음 조건을 봅니다.

- DriftGate − X의 seed 평균이 +0.1 pp 이상이다.
- 모든 seed에서 DriftGate − X가 양수이다.

이 조건이 세 날 전체, 둘째·셋째 날, 전환 device-round에서 함께 성립하면 "개선"으로 적습니다. 그렇지 않으면 이 프로토콜에서 후반 이동의 이득을 주장하지 않습니다. 0.1 pp는 통계적 기준이 아닙니다. 통계적으로 차이가 확인되지 않았다는 사실만으로 동등하다고 선언하지 않습니다.
