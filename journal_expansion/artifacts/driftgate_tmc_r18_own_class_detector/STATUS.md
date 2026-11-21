# Round 18 상태

- **현재 단계:** 끝났습니다(2026-11-22). 최종 판정은 "중단 권고"이고(D28), 결과와 해석은 `report_final.md`에 있습니다. 아래 줄들은 진행하는 동안 적은 기록입니다.
- **끝난 일:** 1단계 전체, 2단계의 개발용 run, 설계 선택(D11, D13, D15: 세 구성 모두 `exit|mlp64|both`), 평가용 run 24개와 분석, 2단계 표와 그림, 실패 원인의 진단 표, 개선 시도 3번의 가설(D18)과 코드(`scripts/r18oc_attempt3.py`), 3단계 준비(환경, job 목록, 표 스크립트).
- **저장 위치:** run 출력, 분석 cache, iNaturalist 자료, 내보낸 모델은 `/home/honeynaps/8T/2_Data/3_DataPreprocessed/driftgate_r18_own_class_261120/`에 있습니다. 저장소의 `runs/phaseT18oc`와 `cache`는 그 폴더를 가리키는 링크입니다.
- **진행 중인 배경 작업:** dispatcher(`runs/phaseT18oc/dispatch.log`)가 개선 시도 1·2번의 개발용 run 12개(두 구성 × S1 seed 8–10)를 돌리고 있고, 자동 분석(`cache/autoanalyse.log`)이 끝난 run을 차례로 분석합니다.
- **다음 일:** 시도 1·2번의 cache가 모두 생기면 `r18oc_tables.py attempt_dev 1|2 <구성>`으로 관문(+0.5 %p)을 계산합니다. 관문을 넘은 시도는 평가용 run(S1·S2 seed 0–2)을 돌려 `attempt_cells`로 판정 칸을 다시 봅니다. 그 뒤 시도 3번(`r18oc_attempt3.py run`, `dev`, `cells`)을 같은 방식으로 판정합니다. 여지 있는 칸이 모두 부분 통과 이상이 되는 시도가 없으면 3단계를 건너뛰고 4단계로 가며, 판정은 "중단 권고"입니다.
- **개선 시도의 결과 (2026-11-21 18:17 기준):** 시도 1번은 `pretrained-finetuned-block`에서 관문을 넘지 못했습니다(−5.43 %p, D20). `pretrained-frozen-block`의 시도 1번과 두 구성의 시도 2번은 개발용 run을 기다리고 있습니다.
- **개선 시도의 결과 (2026-11-21 18:40 기준):** 시도 2번은 `pretrained-frozen-block`에서 관문을 넘었습니다(+2.50 %p, D21). 평가용 run 12개가 job 목록에 들어갔습니다.
- **개선 시도의 결과 (2026-11-21 19:55 기준):** 시도 1번은 `pretrained-frozen-block`에서도 관문을 넘지 못했습니다(−0.85 %p, D22). 시도 3번의 cache를 `scripts/r18oc_attempt3_batch.sh`로 만들고 있습니다(`cache/attempt3_batch.log`). 시도 3번의 관문은 `pretrained-finetuned-block`의 시도 2번 관문 뒤에 계산합니다.
- **개선 시도의 결과 (2026-11-21 20:50 기준):** 시도 2번은 `pretrained-finetuned-block`에서도 관문을 넘었습니다(+0.67 %p, D23). 시도 3번은 두 구성 모두 관문을 넘었지만(+1.09, +0.99 %p), 평가용 seed의 판정 칸은 모두 미달입니다(D24). 남은 것은 시도 2번의 평가용 run 24개(두 구성)이고, 끝나면 `r18oc_tables.py attempt_cells 2 <구성>`으로 판정합니다.
- **개선 시도의 결과 (2026-11-22 02:00 기준):** 시도 2번은 `pretrained-frozen-block`의 평가용 seed에서 판정 대상 칸(S1·S2 `frozen-replay`) 두 칸 모두 미달입니다(G(0.5) −2.77, −1.82 %p, S 0; D25). 남은 것은 `pretrained-finetuned-block`의 시도 2번 평가용 run 12개입니다.
- **4단계 준비 (2026-11-22 02:30 기준):** `pretrained-finetuned-block`을 내보냈고(D26), ONNX와 TFLite float32 출력이 PyTorch와 2.15e−5 안에서 같습니다. 가정값으로 그린 지연 그림, `figures/README_ko.md`, 최종 보고서의 표와 요약 그림을 만드는 `scripts/r18oc_final.py`를 commit했습니다. `pretrained-finetuned-block`의 시도 2번 평가용 run(오전 10–11시 예상)이 끝나면 판정 칸을 계산하고 `report_final.md`와 `r18_review_bundle.zip`을 만듭니다.
- **마무리 (2026-11-22 06:10 기준):** `pretrained-finetuned-block`의 시도 2번도 판정 대상 칸 두 개(S1 `second-day`·`frozen-replay`)가 모두 미달입니다(G(0.5) −1.41, −1.85 %p; D27). 7.3절의 마지막 행에 따라 3단계를 건너뛰었고, 판정은 "중단 권고"입니다(D28). `report_final.md`, `r18_review_bundle.zip`, `tables/R18oc_files_sha256.csv`를 만들었습니다. 배경 작업은 남아 있지 않습니다.
