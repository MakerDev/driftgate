# Round 18 그림 목록

그림마다 무엇을 보여 주는지 한 문장씩 적었습니다. 모든 그림은 `tables/`의 CSV만 읽어 그렸습니다. 그림 안의 기호와 이름은 `report_final.md`의 머리말에서 설명합니다. 줄인 이름 ft는 `pretrained-finetuned-block`, fz는 `pretrained-frozen-block`입니다.

| 그림 | 보여 주는 것 |
|---|---|
| `R18oc_stage2_S1_cnn-scratch.png` | S1의 `cnn-scratch`에서 세 조건별로 offloading 비율에 따른 정확도를 새 방법, envelope, v32 방법, request-kind oracle의 seed 평균 곡선으로 비교합니다. |
| `R18oc_stage2_S2_cnn-scratch.png` | S2의 `cnn-scratch`에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_S1_pretrained-finetuned-block.png` | S1의 ft에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_S2_pretrained-finetuned-block.png` | S2의 ft에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_CIFAR-100_pretrained-finetuned-block.png` | CIFAR-100의 ft에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_S1_pretrained-frozen-block.png` | S1의 fz에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_S2_pretrained-frozen-block.png` | S2의 fz에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_stage2_CIFAR-100_pretrained-frozen-block.png` | CIFAR-100의 fz에서 같은 네 곡선을 세 조건별로 비교합니다. |
| `R18oc_final_judged_cells.png` (`.pdf`) | 사전학습 두 구성의 S1·S2 `second-day`·`frozen-replay` 칸에서 2단계와 개선 시도 2·3번의 G(0.5)를 부분 통과(+1.0 %p)와 통과(+2.0 %p) 기준과 함께 보여 줍니다. |
| `R18oc_mobile_assumed_S1_pretrained-finetuned-block_frozen-replay.png` (`.pdf`, `.json`) | S1의 ft `frozen-replay`에서 요청당 평균 지연과 정확도의 관계를 v32 원고의 가정값으로 계산해 보여 주며, 측정값이 아니므로 "가정값"이라고 표시했습니다. |
| `R18oc_mobile_assumed_S2_pretrained-finetuned-block_frozen-replay.png` (`.pdf`, `.json`) | S2의 ft `frozen-replay`에서 같은 가정값으로 요청당 평균 지연과 정확도의 관계를 보여 주며, 역시 "가정값"이라고 표시했습니다. |
