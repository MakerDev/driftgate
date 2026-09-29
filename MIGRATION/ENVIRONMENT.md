# 실행 환경

옛 서버에서 Round 1–5의 모든 run을 실행한 환경은 다음과 같다. 새 서버에서도 가능하면 같은 버전을 설치한다.

| 항목 | 옛 서버 값 |
|---|---|
| OS | Linux 5.15.0-139-generic (Ubuntu) |
| Python | 3.12.7 (Anaconda, `/home/ubuntu/anaconda3/bin/python`) |
| PyTorch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| numpy / scipy | 2.5.1 / 1.16.3 |
| matplotlib | 3.11.0 (그림 스크립트가 이 버전에서 확인됨) |
| PyYAML / pytest / tqdm / pillow | 6.0.1 / 8.4.2 / 4.66.5 / 10.4.0 |
| GPU | NVIDIA GeForce RTX 3090 Ti 24 GB × 2 (index 0, 1) |
| NVIDIA driver / CUDA | 580.82.09 / 12.6 wheel |

정확한 버전 목록은 `MIGRATION/requirements-lock.txt`에 있다. 루트의 `requirements.txt`는 v3 시절의 느슨한 하한 목록이므로 새 환경 설치에는 쓰지 않는다.

## 설치 순서

```bash
# 1) 가상 환경 (conda 예시; venv도 무방)
conda create -n driftgate python=3.12.7 -y
conda activate driftgate

# 2) 드라이버 확인: CUDA 12.6 wheel은 driver >= 560 계열이면 동작한다
nvidia-smi

# 3) 고정 버전 설치
pip install -r MIGRATION/requirements-lock.txt
```

드라이버가 CUDA 12.6 wheel을 지원하지 않으면 torch 2.7.1의 다른 CUDA wheel(cu118, cu128 등)을 설치한다. 이 경우 torch 버전은 2.7.1로 유지하고, 바꾼 wheel을 `MIGRATION/MIGRATION_LOG.md`에 적는다.

## 결과 비교에 대한 주의

GPU 종류, 드라이버, cuDNN 버전이 바뀌면 같은 seed라도 부동소수점 연산 순서가 달라져서 run 결과가 bit 단위로 같지 않다. 정확도 차이는 보통 seed 사이 편차보다 훨씬 작지만, 다음 규칙을 지킨다.

- 옛 서버에서 끝난 run은 다시 돌리지 않는다. Round 5 지시문도 "비교 기준이 되는 기존 run은 다시 돌리지 않고 기존 JSON을 쓴다"고 정했다.
- 새 서버에서 돌린 run은 run manifest에 서버 이름과 GPU 이름을 적는다. run JSON과 짝을 이루는 `journal_expansion/provenance/<run_id>.json`에는 `gpu` 필드(`torch.cuda.get_device_name(0)`)가 자동으로 기록된다.
- 한 arm의 seed들이 두 서버에 나뉘어 실행되었다면 보고서의 불일치 기록에 그 사실을 적는다.
- 확인용으로 옛 서버에서 끝난 run 하나를 새 서버에서 다시 돌려 볼 수 있다. 이 run은 별도 폴더(예: `runs/_migration_check/`)에 두고 표에는 넣지 않는다. 자세한 방법은 `START_HERE.md` 7단계에 있다.
