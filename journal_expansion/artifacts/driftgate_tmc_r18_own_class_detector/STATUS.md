# Round 18 상태

- **현재 단계:** 2단계(예비 실험)의 평가용 run 진행 중입니다. 세 구성의 설계 선택이 끝났습니다(D11, D13, D15: 세 구성 모두 `exit|mlp64|both`).
- **끝난 일:** 1단계 전체, 2단계 개발용 run과 설계 선택, 중간 산출물과 데이터셋을 8T 디스크로 옮기기(D9, 옛 폴더 `runs/phaseT18oc.moving`은 확인 뒤 삭제), iNaturalist 사진 30,000장 추출, 3단계 준비(S2 seed 3–4 환경, stepwise change와 random mobility 환경 D14, 3단계 job 목록 `r18oc_jobs.py stage3`, 3단계 표 `r18oc_stage3.py`, 곡선 그림 `r18oc_figures.py`, 보고서 표 `r18oc_report.py`).
- **저장 위치:** run 출력, 분석 cache, iNaturalist 자료, 내보낸 모델은 `/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120/`에 있습니다. 저장소의 `runs/phaseT18oc`와 `cache`는 그 폴더를 가리키는 링크입니다.
- **진행 중인 배경 작업:** dispatcher(`runs/phaseT18oc/dispatch.log`, job 90개), 자동 분석(`cache/autoanalyse.log`). job 목록의 끝에는 개선 시도 1·2번의 개발용 run이 있습니다(D16, 2단계 판정이 행 2나 3일 때만 씀).
- **다음 일:** 평가용 run의 분석이 모두 끝나면 `r18oc_tables.py stage2`와 `holdout`을 돌리고 `report_stage2.md`를 쓴 뒤, 7.3절의 판정을 `decisions.md`에 적고 그에 따라 진행합니다.
