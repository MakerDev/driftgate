# 기기 측정 안내 (Round 6, 지시문 6.3)

이 폴더의 파일로 Galaxy 스마트폰 2대, Jetson 보드, edge 서버에서 지연 시간을 잰다. 측정 결과 CSV를 `latency_calc.py`에 넣으면 보고서의 지연 시간 표(T9)가 host 임시값 대신 실측값으로 다시 계산된다.

## 1. 파일

| 파일 | 내용 |
|---|---|
| `models/driftgate_s1_s0_weights.pth` | S1(출퇴근 이동) DriftGate seed 0의 마지막 모델. client 0의 client block + client exit, hub cell의 server block + server exit |
| `models/client_block_exit.{pt,ptl,onnx}` | 입력 이미지 [B,3,32,32] → (client logit [B,10], 중간 특징 [B,128,8,8]) |
| `models/server_block_exit.{pt,ptl,onnx}` | 입력 중간 특징 [B,128,8,8] → server logit [B,10] |
| `models/MANIFEST.csv` | 파일 크기, sha256, 원래 모델과의 출력 차이 |
| `r6_device_models.py` | 모델 구조 사본. `bench_device.py`와 `offload_server.py`가 쓴다 |
| `bench_device.py` | Jetson에서 t_c, t_s(사본), 로컬 학습 한 라운드 시간을 잰다 |
| `offload_server.py` | edge 서버에서 중간 특징을 받아 server block을 실행하고 logit을 돌려준다 |
| `offload_client.py` | 중간 특징 32,768 byte를 보내고 응답까지의 시간을 500회 기록한다. Python 표준 라이브러리만 쓴다 |
| `export_models.py` | 위 모델 파일을 다시 만드는 스크립트(서버에서 실행) |

- `.pt`는 TorchScript, `.ptl`은 PyTorch Mobile lite interpreter용, `.onnx`는 opset 17이며 batch 크기가 가변이다.
- 요청 하나를 offload하면 중간 특징 A = 128×8×8×4 = 32,768 byte를 보낸다.

## 2. 측정 순서

### 2.1 edge 서버에서 offload 서버 실행

GPU가 있는 edge 서버(또는 실험용 PC)에 이 폴더를 복사하고 PyTorch를 설치한 뒤 실행한다.

```bash
pip install torch numpy                 # 이미 있으면 생략
python3 offload_server.py --weights models/driftgate_s1_s0_weights.pth --port 5555 --device cuda
```

- 서버는 요청마다 server block 계산 시간을 재서 응답에 함께 보낸다. 그래서 클라이언트는 전체 왕복 시간에서 서버 계산 시간을 빼고 네트워크 시간만 따로 기록할 수 있다.
- 방화벽이 있으면 TCP 5555 포트를 연다. 기기와 서버는 측정하려는 무선망(예: Wi-Fi 5 GHz, LTE, 5G)으로 연결한다.

### 2.2 Jetson: 계산 시간 측정

JetPack에 맞는 PyTorch를 설치한 뒤(NVIDIA의 Jetson용 wheel) 실행한다.

```bash
python3 bench_device.py --weights models/driftgate_s1_s0_weights.pth \
    --device-name "Jetson Orin Nano 8GB" --backends cpu cuda --out device_timing.csv
```

- 요청 1개와 64개에 대해 client block(t_c)과 server block 사본(t_s)을 각각 20회 예열한 뒤 200회 실행하고, 중앙값과 p95를 적는다.
- 로컬 학습 한 라운드(t_train_round)는 표본 908개(Round 6 클라이언트의 중앙값), batch 32, local epoch 3으로 잰다.
- 전원 모드에 따라 결과가 달라지므로 `sudo nvpmodel -q`로 모드를 확인하고, CSV의 device 이름에 함께 적는다(예: "Jetson Orin Nano 8GB, 15W").

### 2.3 Galaxy 스마트폰: 계산 시간 측정

스마트폰에서는 PyTorch를 Python으로 실행하기 어렵다. 다음 가운데 하나로 재고, 결과를 `device_timing.csv`에 한 줄로 직접 적는다.

1. **PyTorch Mobile 벤치마크 앱이나 `speed_benchmark_torch`**: `models/client_block_exit.ptl`, `models/server_block_exit.ptl`을 입력 크기 [1,3,32,32], [64,3,32,32], [1,128,8,8], [64,128,8,8]로 실행한다.
2. **ONNX Runtime**(Android 앱 또는 `onnxruntime_perf_test`): `models/*.onnx`를 같은 입력 크기로 실행한다. 이 경우 backend 열에 "onnxruntime-cpu" 또는 "onnxruntime-nnapi"처럼 적는다.

CSV 형식은 다음과 같다. 앞의 7개 열은 필수이고, 없는 값은 비워 둔다. 스마트폰에서 학습 시간을 재지 않았으면 `t_train_round_s`를 비워 둔다. 그러면 `latency_calc.py`는 그 기기의 학습 대비 비율을 계산하지 않는다.

```
device,backend,t_c_b1_ms,t_c_b64_ms,t_s_b1_ms,t_s_b64_ms,t_train_round_s,source
Galaxy S24,pytorch-mobile-cpu,1.9,41.0,0.8,12.5,,measured
```

### 2.4 네트워크 지연 측정 (Jetson과 스마트폰)

offload 서버가 켜져 있는 상태에서 각 기기에서 실행한다. 스마트폰에서는 Termux 앱을 설치하고 `pkg install python`으로 Python을 설치한 뒤 실행한다.

```bash
python3 offload_client.py --host <edge 서버 IP> --port 5555 \
    --device-name "Galaxy S24" --network-name "WiFi 5GHz" --out net_latency.csv
```

- 요청 10개로 예열한 뒤 500회를 기록한다. 요청 사이에는 100 ms를 쉰다(`--interval-ms`).
- 출력 열: device, network, i, bytes_sent, bytes_recv, rtt_total_ms, server_compute_ms, network_ms, unix_time
- `network_ms` = 전체 왕복 시간 − 서버 계산 시간. `latency_calc.py`는 이 값을 8A/B + RTT 대신 쓴다.
- 기기와 망마다 파일을 따로 두는 것이 편하다(예: `net_latency_s24_wifi.csv`, `net_latency_s24_lte.csv`).

## 3. 결과를 보고서에 반영

측정 파일을 이 폴더에 두고, 저장소 루트에서 실행한다.

```bash
# 기기 계산 시간과 모델식 네트워크(B ∈ {10, 50, 100} Mbps, RTT ∈ {20, 50} ms)
python journal_expansion/artifacts/driftgate_tmc_r6/scripts/latency_calc.py \
    --device-csv journal_expansion/artifacts/driftgate_tmc_r6/device/device_timing.csv --device "Galaxy S24"

# 실측 네트워크 표본을 쓸 때
python journal_expansion/artifacts/driftgate_tmc_r6/scripts/latency_calc.py \
    --device-csv .../device_timing.csv --device "Galaxy S24" \
    --net-csv .../net_latency_s24_wifi.csv
```

- 출력: `tables/T9_latency.csv`(방법별 offload 비율, 평균·p95 E2E 지연 시간), `tables/T9_latency_by_slot.csv`(시간대별 평균), `tables/T9_overhead.csv`(DriftGate의 라운드당 추가 비용, 두 배치).
- `--device-csv`를 주지 않으면 서버 CPU에서 잰 host 임시값(`tables/T9a_host_placeholder_device.csv`)을 쓰고, 표의 `device_timing` 열에 "host placeholder"라고 적는다.
- edge GPU의 server block 시간(t_s edge)은 서버에서 잰 `tables/T9a_server_timing.csv`를 쓴다. 다른 edge 장비를 쓰면 그 장비에서 `scripts/r6_server_timing.py`를 다시 실행한다.
