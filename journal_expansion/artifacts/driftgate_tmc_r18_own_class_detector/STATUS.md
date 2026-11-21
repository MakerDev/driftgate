# Round 18 상태

- **현재 단계:** 2단계가 끝났고, 7.3절의 판정은 두 사전학습 구성 모두 행 3입니다(D17, `report_stage2.md`). 판정 칸이 모두 미달이지만 일부 칸에 여지가 있으므로, 개선 시도를 1번부터 차례로 두 구성 모두에 합니다.
- **끝난 일:** 1단계 전체, 2단계의 개발용 run, 설계 선택(D11, D13, D15: 세 구성 모두 `exit|mlp64|both`), 평가용 run 24개와 분석, 2단계 표와 그림, 실패 원인의 진단 표, 개선 시도 3번의 가설(D18)과 코드(`scripts/r18oc_attempt3.py`), 3단계 준비(환경, job 목록, 표 스크립트).
- **저장 위치:** run 출력, 분석 cache, iNaturalist 자료, 내보낸 모델은 `/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120/`에 있습니다. 저장소의 `runs/phaseT18oc`와 `cache`는 그 폴더를 가리키는 링크입니다.
- **진행 중인 배경 작업:** dispatcher(`runs/phaseT18oc/dispatch.log`)가 개선 시도 1·2번의 개발용 run 12개(두 구성 × S1 seed 8–10)를 돌리고 있고, 자동 분석(`cache/autoanalyse.log`)이 끝난 run을 차례로 분석합니다.
- **다음 일:** 시도 1·2번의 cache가 모두 생기면 `r18oc_tables.py attempt_dev 1|2 <구성>`으로 관문(+0.5 %p)을 계산합니다. 관문을 넘은 시도는 평가용 run(S1·S2 seed 0–2)을 돌려 `attempt_cells`로 판정 칸을 다시 봅니다. 그 뒤 시도 3번(`r18oc_attempt3.py run`, `dev`, `cells`)을 같은 방식으로 판정합니다. 여지 있는 칸이 모두 부분 통과 이상이 되는 시도가 없으면 3단계를 건너뛰고 4단계로 가며, 판정은 "중단 권고"입니다.
- **개선 시도의 결과 (2026-11-21 18:17 기준):** 시도 1번은 `pretrained-finetuned-block`에서 관문을 넘지 못했습니다(−5.43 %p, D20). `pretrained-frozen-block`의 시도 1번과 두 구성의 시도 2번은 개발용 run을 기다리고 있습니다.
- **개선 시도의 결과 (2026-11-21 18:40 기준):** 시도 2번은 `pretrained-frozen-block`에서 관문을 넘었습니다(+2.50 %p, D21). 평가용 run 12개가 job 목록에 들어갔습니다.
