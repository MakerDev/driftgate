# Round 18 상태

- **현재 단계:** 2단계(예비 실험)의 개발용 run 진행 중입니다(2026-11-20 16:54 시작).
- **끝난 일:** 1단계의 학습 코드(MobileNetV2 분할, held-out, 이틀 일정, 판별기, 라운드 150 checkpoint, 판별기를 포함한 재생), 옵션을 끈 runner의 비트 동일성 확인(CPU 1라운드), 재생의 비트 단위 재현 확인, 개발용 seed 8–10의 환경 파일, `decision_rule.md` commit입니다.
- **다음 일:** 개발용 run이 끝나면 분석(`scripts/r18oc_analysis.py run`)과 3.5절의 선택, commit, 평가용 run 추가(`scripts/r18oc_jobs.py eval`)입니다. 1단계 점검과 분석 스크립트는 commit했습니다.
