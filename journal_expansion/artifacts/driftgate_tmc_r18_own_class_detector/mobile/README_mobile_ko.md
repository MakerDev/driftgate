# Round 18 모바일 측정 안내

이 폴더는 지시문 6.5절의 1번에 해당하는 측정 도구입니다. 측정은 사용자가 갤럭시 스마트폰과 Jetson에서 직접 합니다. 여기의 시간과 에너지 값은 아직 하나도 측정되지 않았습니다. `measurement_assumed.json`에 있는 값은 v32 원고의 가정값이며, 이 값으로 만든 그림에는 "가정값"이라는 표시가 찍힙니다.

## 1. 재는 값

| 항목 | 뜻 | 얻는 방법 |
|---|---|---|
| device block 시간 | 이미지 한 장을 device block에 통과시키는 시간(ms) | `device_block` 모델 |
| device exit 시간 | block 출력에서 device exit의 logit을 만드는 시간(ms) | `device_exit` 모델 |
| 판별기 시간 | block 벡터와 exit 벡터에서 g를 만드는 시간(ms) | `detector` 모델 |
| device 전체 시간 | 위 세 단계를 한 번에 실행하는 시간(ms) | `device_side` 모델 |
| edge 시간 | edge block과 분류기를 실행하는 시간(ms) | `edge_block` 모델(edge 장비에서) |
| uplink, downlink byte | offload한 요청 하나가 보내는 smashed data와 받는 edge logit의 크기 | `export/export_manifest.json`의 `bytes_per_offloaded_request` |
| 요청당 에너지 | device 계산의 에너지와 offload 한 번에 더 드는 에너지(mJ) | 아래 4절 |
| 네트워크 | uplink·downlink 속도(Mbps)와 왕복 지연(ms) | iperf3, ping |

결과는 `measurement_template.json` 하나에 모읍니다. 측정하지 않은 항목은 null로 둡니다.

## 2. 모델 내보내기 (서버에서)

```bash
cd /home/honeynaps/data/driftgate && source ~/venvs/driftgate/bin/activate
M=journal_expansion/artifacts/driftgate_tmc_r18_own_class_detector/mobile
python $M/r18oc_export_mobile.py <runs/phaseT18oc/주 구성 run의 _ckpt_r150.pt> <모델 구성> '<입력|모델|음성 표본>' --out $M/export
~/venvs/r18_tflite/bin/python $M/r18oc_tflite_check.py $M/export
```

- 첫 명령은 ONNX 모델 5개(`device_block`, `device_exit`, `detector`, `device_side`, `edge_block`)를 만들고, test 이미지 64장에서 PyTorch 출력과의 최대 차이가 10⁻⁴ 이하인지 확인합니다.
- 둘째 명령은 같은 모델을 TFLite(float32)로 바꾸고 같은 기준으로 확인합니다. TFLite 모델의 4차원 입력과 출력은 NHWC 순서입니다(예: 이미지 [1, 32, 32, 3]).
- 확인 결과는 `export/export_manifest.json`의 `onnx_ok`, `tflite_ok`에 적힙니다.
- MobileNetV2 모델은 입력을 64×64로 확대합니다. 내보낸 모델은 이 확대를 scale factor 2로 적었으며, 학습에 쓴 크기 지정 방식과 같은 값을 냅니다(내보내기 스크립트가 확인합니다).

## 3. 갤럭시 스마트폰 (Android, TFLite)

1. TensorFlow가 배포하는 Android용 `benchmark_model`(android_aarch64)을 받아 휴대폰에 올립니다.
   ```bash
   adb push android_aarch64_benchmark_model /data/local/tmp/benchmark_model
   adb shell chmod +x /data/local/tmp/benchmark_model
   adb push export/*_float32.tflite /data/local/tmp/
   ```
2. 모델마다 시간을 잽니다. CPU 4 thread를 기본으로 하고, GPU delegate를 쓰면 `--use_gpu=true`를 더해 따로 기록합니다.
   ```bash
   for m in device_block device_exit detector device_side; do
     adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/${m}_float32.tflite \
       --num_threads=4 --warmup_runs=50 --num_runs=500 > logs/${m}.log
   done
   python parse_benchmark_model.py logs --out phone_latency.json
   ```
3. 화면 밝기, 비행기 모드 여부, 충전 상태, 기기 온도를 측정 전에 고정하고 JSON의 `device`와 `energy`에 적습니다.

## 4. 요청당 에너지

- 가장 정확한 방법은 외부 전원 측정기(예: Monsoon)로 휴대폰에 전원을 공급하고 전력을 기록하는 것입니다.
- 측정기가 없으면 배터리의 전압과 전류를 0.1초마다 읽는 `android_power_sample.sh`를 씁니다. 충전 중에는 전류 값이 맞지 않으므로 충전을 끄고 잽니다.
  ```bash
  bash android_power_sample.sh export/device_side_float32.tflite 60 power/device_side 4
  ```
- 요청당 에너지 = (부하 중 평균 V·I − 대기 중 평균 V·I) × 측정 시간 ÷ 실행 횟수입니다. 실행 횟수는 benchmark 로그의 count입니다. 단위(μV, μA, mA)는 기기마다 다르므로 확인해서 적습니다.
- offload 한 번에 더 드는 에너지(`energy_offload_extra_mJ`)는 smashed data를 실제로 edge에 보내는 실험으로 잽니다. 같은 시간 동안 요청을 보내는 경우와 보내지 않는 경우의 전력 차이를 요청 수로 나눕니다. 무선 상태(대기, 연결 유지)에 따라 값이 크게 달라지므로 요청 간격을 함께 적습니다.

## 5. Jetson (ONNX Runtime)

```bash
pip install onnxruntime-gpu   # JetPack에 맞는 wheel
python measure_onnx.py export --out jetson_latency.json \
    --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider --tegrastats --power_seconds 30
```

- 모델마다 평균, 95번째, 99번째 백분위 시간을 적습니다. `--tegrastats`를 주면 대기 전력을 먼저 재고, 모델을 반복 실행하는 동안의 평균 전력에서 대기 전력을 빼서 요청당 에너지를 계산합니다.
- edge 시간은 edge로 쓸 장비에서 `edge_block` 모델로 같은 명령을 실행해 얻습니다.

## 6. 네트워크

```bash
iperf3 -c <edge 주소> -t 30        # uplink
iperf3 -c <edge 주소> -t 30 -R     # downlink
ping -c 100 <edge 주소>            # 왕복 지연
```

## 7. 그림

```bash
python r18oc_mobile_plot.py measurement_template.json --curves ../tables/<곡선 CSV> --reference ../tables/<참고 행 CSV> \
    --setting S1 --config <모델 구성> --condition frozen-replay --variant '<입력|모델|음성 표본>' --out ../figures/mobile_S1
```

- 정확도 대 평균 지연과 정확도 대 평균 에너지 그림을 만듭니다. 지연 모델은 v32 원고와 같은 단일 edge 순차 경로입니다. 모든 요청은 device 쪽을 실행하고, offload한 요청에는 uplink 전송, 왕복 지연, edge 실행, downlink 전송이 더해집니다. 두 cell에 속한 요청의 두 번째 edge 호출과 대기열은 넣지 않습니다.
- `measurement_type`이 `measured`가 아니면 그림에 "가정값"이 찍힙니다. 에너지 값이 없으면 에너지 그림에 "에너지 미측정"이 찍힙니다.
- `measurement_assumed.json`의 device 시간 11.8 ms는 v32의 CNN 가정값이며 MobileNetV2의 측정값이 아닙니다.
