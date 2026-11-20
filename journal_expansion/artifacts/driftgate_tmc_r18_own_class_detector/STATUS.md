# Round 18 상태

- **현재 단계:** 2단계(예비 실험)의 개발용 run 진행 중입니다(2026-11-20 16:54 시작). CNN은 약 21:00, MobileNetV2는 약 23:00–24:00에 끝날 것으로 봅니다.
- **끝난 일:** 1단계 전체(학습 코드, 점검, `decision_rule.md`, `report_stage1.md`, 분석·표 스크립트 commit), 개선 시도 1·2번 코드, iNaturalist 선택 규칙과 설정 코드, 4단계 모바일 도구(내보내기, TFLite 확인, 측정 스크립트, 안내서), 중간 산출물과 데이터셋을 8T 디스크로 옮기기(D9).
- **저장 위치:** run 출력, 분석 cache, iNaturalist 자료, 내보낸 모델은 `/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120/`에 있습니다. 저장소의 `runs/phaseT18oc`와 `cache`는 그 폴더를 가리키는 링크입니다.
- **진행 중인 배경 작업:** dispatcher(`runs/phaseT18oc/dispatch.log`), 자동 분석(`cache/autoanalyse.log`), iNaturalist 사진 추출(`datasets/inat2021_r18/extract.log`, 21:00–22:00 끝 예상).
- **다음 일:** 구성마다 개발용 seed 3개의 분석이 끝나면 `r18oc_tables.py select <구성>`으로 3.5절의 선택을 하고 commit한 뒤 `r18oc_jobs.py eval <구성>=<조합>`으로 평가용 run을 넣고 dispatcher를 다시 띄웁니다. 개발용 run이 모두 끝나면 `runs/phaseT18oc.moving`의 로그를 8T 폴더로 옮깁니다.
