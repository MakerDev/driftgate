# 데이터셋

## 결론: 업로드한 데이터셋은 없다

이 프로젝트가 쓰는 데이터셋은 모두 공개 주소에서 다시 받을 수 있다. 그래서 GitHub 저장소에는 데이터셋을 올리지 않았고, 새 서버에서 아래 방법으로 다시 받는다. 직접 만든 데이터 파일(파생 데이터)도 없다. 데이터 분할(client별 표본, probe/eval pool)은 매 run마다 코드가 seed로부터 다시 만든다.

| 데이터셋 | 사용하는 실험 | 받는 방법 | 크기 |
|---|---|---|---|
| CIFAR-10 | Schedule A, mobility, CIFAR-10 gradual, ResNet-18 등 대부분 | torchvision 자동 다운로드 | 163 MiB (tar) |
| CIFAR-100 | CIFAR-100 gradual / spatial | torchvision 자동 다운로드 | 161 MiB (tar) |
| SVHN | SVHN temporal / spatial (frozen holdout) | torchvision 자동 다운로드 | 약 245 MiB (.mat 2개) |
| Tiny-ImageNet-200 | Tiny-ImageNet gradual | 수동 다운로드 후 압축 해제 | 237 MiB (zip), 해제 후 약 480 MB |
| GTSRB, STL-10 | 초기 pilot에서만 사용, Round 5와 무관 | torchvision 자동 다운로드 | 필요할 때만 |

## 코드가 데이터를 찾는 위치

데이터 위치는 두 군데다.

1. **`DATA_ROOT`**: `journal_expansion/src/datasets_ext.py:27`에 적혀 있다. 옛 서버 값은 `/disk2/Yujin/datasets`였다. `--dataset` 옵션을 준 run(CIFAR-10 gradual, CIFAR-100, SVHN, Tiny-ImageNet)이 이 위치를 쓴다. `rewrite_server_paths.py --data-root`로 새 값으로 바꾼다.
2. **`./data_cache`**: `configs/base_v3.yaml`의 `data_root: ./data_cache`이다. 저장소 루트 기준 상대 경로이며, `--dataset` 옵션이 없는 run(Schedule A, mobility, ResNet-18 등 CIFAR-10 기본 설정)이 이 위치의 CIFAR-10을 쓴다. 작업 디렉터리가 저장소 루트여야 한다(`r5_worker.py`가 `os.chdir`로 맞춘다).

두 위치의 CIFAR-10은 같은 파일이다(md5 동일). 한 번만 받고 `data_cache`를 심볼릭 링크로 만들면 된다.

`DATA_ROOT` 아래의 기대 구조는 다음과 같다.

```
$DATA_ROOT/
├── cifar10/cifar-10-python.tar.gz, cifar-10-batches-py/
├── cifar100/cifar-100-python.tar.gz, cifar-100-python/
├── svhn/train_32x32.mat, test_32x32.mat
└── tiny-imagenet-200/          # zip을 풀어서 생긴 폴더 그대로 (train/, val/, wnids.txt, words.txt)
    ├── train/<wnid>/images/*.JPEG
    └── val/images/*.JPEG, val/val_annotations.txt
```

Tiny-ImageNet의 val 폴더는 재배치하지 않는다. 코드(`TinyImageNet` 클래스)가 `val/val_annotations.txt`를 읽어서 원래 구조 그대로 쓴다.

## 다운로드 명령

저장소 루트에서 실행한다. `DATA_ROOT`는 새 서버에서 정한 경로로 바꾼다.

```bash
export DATA_ROOT=/path/to/datasets          # 새 서버의 데이터 위치
mkdir -p "$DATA_ROOT"

# CIFAR-10, CIFAR-100, SVHN: torchvision이 받고 압축을 푼다
python - <<EOF
from torchvision import datasets
R = "$DATA_ROOT"
for tr in (True, False):
    datasets.CIFAR10(f"{R}/cifar10", train=tr, download=True)
    datasets.CIFAR100(f"{R}/cifar100", train=tr, download=True)
for sp in ("train", "test"):
    datasets.SVHN(f"{R}/svhn", split=sp, download=True)
print("torchvision datasets ready")
EOF

# Tiny-ImageNet-200 (Stanford CS231n 배포본)
wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip -O "$DATA_ROOT/tiny-imagenet-200.zip"
unzip -q "$DATA_ROOT/tiny-imagenet-200.zip" -d "$DATA_ROOT"

# 기본 CIFAR-10 위치(./data_cache)를 같은 파일로 연결
ln -sfn "$DATA_ROOT/cifar10" data_cache
```

cs231n 주소가 막혀 있으면 같은 파일을 다른 미러에서 받는다(예: Hugging Face의 `zh-plus/tiny-imagenet`은 parquet 형식이라 구조가 다르므로 쓰지 않는다). 받은 zip이 아래 md5와 같은지 반드시 확인한다.

옛 서버에서 새 서버로 직접 복사할 수 있다면(scp, rsync) 그렇게 해도 된다. 옛 서버 경로는 `/disk2/Yujin/datasets/{cifar10,cifar100,svhn,tiny-imagenet-200,tiny-imagenet-200.zip}`이다.

## 검증 (md5)

옛 서버에 있던 파일의 md5이다. 새로 받은 파일이 같아야 한다.

| 파일 | md5 |
|---|---|
| `cifar10/cifar-10-python.tar.gz` | `c58f30108f718f92721af3b95e74349a` |
| `cifar100/cifar-100-python.tar.gz` | `eb9058c3a382ffc7106e4002c42a8d85` |
| `svhn/train_32x32.mat` | `e26dedcc434d2e4c54c9b2d4a06d8373` |
| `svhn/test_32x32.mat` | `eb5a983be6a315427106f1b164d9cef3` |
| `tiny-imagenet-200.zip` | `90528d7ca1a48142e341f4ef8d21d0de` (248,100,043 bytes) |

```bash
md5sum "$DATA_ROOT"/cifar10/*.tar.gz "$DATA_ROOT"/cifar100/*.tar.gz "$DATA_ROOT"/svhn/*.mat "$DATA_ROOT"/tiny-imagenet-200.zip
```

표본 수도 확인한다. 옛 서버 로그(`journal_expansion/runs/svhn_download.log`)에는 SVHN이 train 73,257 / test 26,032로 기록되어 있다. CIFAR-10/100은 50,000 / 10,000, Tiny-ImageNet은 train 100,000 / val 10,000이다.

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "journal_expansion")
from src.datasets_ext import get_dataset
for n in ("cifar10", "cifar100", "svhn", "tinyimagenet"):
    tr, te, meta = get_dataset(n)
    print(n, len(tr), len(te), meta)
EOF
```
