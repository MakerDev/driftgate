"""Round 5 — T8 reproduction-settings table and T9 communication table.
Every value is read from config/code (source file:line given) or computed from the code
(parameter counts, tensor shapes, partition sizes). Items that do not exist are marked "없음".
"""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path("/disk2/Yujin/adaptive_splitomc_tmc"); JR = ROOT / "journal_expansion"  # [SERVER-PATH:REPO_ROOT]
HERE = Path(__file__).resolve().parent.parent; TAB = HERE / "tables"; TAB.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(JR)); sys.path.insert(0, str(ROOT))
import torch, yaml
from models.architectures import ModelFactory
from src.models_ext import ResNetClient, ResNetServer
from data.partition import make_es_topology, nd1_partition
cfg = yaml.safe_load(open(ROOT / "configs/base_v3.yaml"))

# ---------------- computed facts
cnn_c, cnn_s = ModelFactory("cifar10", 10, 32).make_client(), ModelFactory("cifar10", 10, 32).make_server()
PC = sum(p.numel() for p in cnn_c.parameters()); PS = sum(p.numel() for p in cnn_s.parameters())
_, rep = cnn_c(torch.zeros(1, 3, 32, 32)); CNN_REP = tuple(rep.shape[1:])
rc, rs = ResNetClient(10, "middle"), ResNetServer(10, "middle")
_, rrep = rc(torch.zeros(1, 3, 32, 32)); RES_REP = tuple(rrep.shape[1:])
RPC = sum(p.numel() for p in rc.parameters()); RPS = sum(p.numel() for p in rs.parameters())

def partition_sizes(ds):
    """per-client train sample counts (seed 0) with the real training labels."""
    if ds == "cifar10":
        from data.partition import get_cifar10
        tr, _ = get_cifar10(data_root=str(ROOT / "data_cache")); ncls = 10
    else:
        from src.datasets_ext import get_dataset
        tr, _, meta = get_dataset(ds); ncls = meta["num_classes"]
    y = np.array(tr.targets)
    c2es, es2c = make_es_topology(cfg["num_clients"], cfg["num_edge_servers"], cfg["overlap_percentage"], seed=0)
    idx, mains, _ = nd1_partition(y, ncls, cfg["num_clients"], c2es, es2c,
                                  classes_per_es_frac=(cfg["classes_per_es_min_frac"], cfg["classes_per_es_max_frac"]),
                                  classes_per_client_frac=cfg["classes_per_client_frac"], seed=0)
    n = np.array([len(v) for v in idx.values()]); m = sorted({len(v) for v in mains.values()})
    return f"min {n.min()} / median {int(np.median(n))} / max {n.max()} (Main classes per client: {m})"
sizes = {}
for ds in ("cifar10", "cifar100", "tinyimagenet", "svhn"):
    try: sizes[ds] = partition_sizes(ds)
    except Exception as e: sizes[ds] = f"계산 실패: {type(e).__name__}"
c2es, _ = make_es_topology(50, 5, 50, seed=0); n_two = sum(1 for v in c2es.values() if len(v) == 2)
c2es16, _ = make_es_topology(16, 5, 50, seed=0); n_two16 = sum(1 for v in c2es16.values() if len(v) == 2)

T8 = [
 ("학습", "optimizer", "SGD (client block과 server block 각각)", "configs/base_v3.yaml:19; train/trainer.py:86,90"),
 ("학습", "learning rate와 schedule", "0.01, 전 라운드 고정 (scheduler 없음)", "configs/base_v3.yaml:17; train/trainer.py:72-141"),
 ("학습", "momentum, weight decay", "0.0, 1e-4", "configs/base_v3.yaml:18,20"),
 ("학습", "local epoch", "3 epoch / 라운드", "configs/base_v3.yaml:15; train/trainer.py:94"),
 ("학습", "batch 크기", "32", "configs/base_v3.yaml:16"),
 ("학습", "참여율", "1.0 (모든 client가 매 라운드 참여)", "src/runner.py:284,290"),
 ("학습", "손실 가중치 γ", "0.5 (client exit와 server exit의 cross-entropy를 반씩)", "configs/base_v3.yaml:23"),
 ("학습", "client별 학습 표본 수 (seed 0) CIFAR-10", sizes["cifar10"], "data/partition.py:50-104 (nd1_partition 재구성)"),
 ("학습", "client별 학습 표본 수 (seed 0) CIFAR-100", sizes["cifar100"], "같은 함수, 실제 학습 label 사용"),
 ("학습", "client별 학습 표본 수 (seed 0) Tiny-ImageNet", sizes["tinyimagenet"], "같은 함수, 실제 학습 label 사용"),
 ("학습", "client별 학습 표본 수 (seed 0) SVHN", sizes["svhn"], "같은 함수, 실제 학습 label 사용"),
 ("학습", "client별 Main class 비율", "전체 class의 20% (CIFAR-10·SVHN 2개, CIFAR-100 20개, Tiny 40개). datasets_ext.py의 META 값(5, 10)은 config 값이 있어 쓰이지 않음",
  "configs/base_v3.yaml:31; src/runner.py:121 (setdefault)"),
 ("모델", "CNN client block", f"conv 3→32→64→64→128 (BN, conv2·conv4 뒤 max-pool) + 보조 exit(conv 128 + GAP + fc). {PC:,} parameters", "models/architectures.py:14-52"),
 ("모델", "CNN split 위치", f"conv4 + pool 뒤. 32×32 입력에서 feature {CNN_REP} (Tiny 64×64 입력에서는 (128, 16, 16))", "models/architectures.py:37-53"),
 ("모델", "CNN server block", f"conv 128→256 + BN, flatten, fc 256→128→C. {PS:,} parameters", "models/architectures.py:54-79"),
 ("모델", "ResNet-18 middle split", f"client = stem(conv 3→64) + stage 1(64) + stage 2(128), feature {RES_REP}, {RPC:,} parameters. server = stage 3(256) + stage 4(512) + fc, {RPS:,} parameters",
  "src/models_ext.py:93-130 (RESNET_SPLITS middle=2)"),
 ("모델", "ResNet-18 실행의 client 수", f"16명 (기존 ResNet-18 분할 실험과 같음. static topology에서 {n_two16}명이 두 cluster에 속함)", "provenance: phaseF_arch res_middle_* (--num_clients 16)"),
 ("데이터", "전처리 CIFAR-10", "학습: RandomCrop(32, padding 4) + RandomHorizontalFlip + Normalize. 평가·probe: Normalize만", "data/partition.py:154-167"),
 ("데이터", "전처리 CIFAR-100, Tiny-ImageNet, SVHN", "ToTensor + Normalize만 (augmentation 없음). Tiny는 64×64 그대로", "src/datasets_ext.py:101-141"),
 ("데이터", "OOP와 OOR의 비율", "Main 표본 수를 기준으로 OOP 표본 = ρ배, OOR 표본 = 0.3ρ배 (class별로 나눠 채움). 즉 OOP:OOR = 1:0.3", "data/partition.py:107-143; configs/base_v3.yaml:37"),
 ("traffic", "stepwise composition change의 ρ", "5등분 계단 0→0.4→0.8→0.4→0. 150 라운드에서 R31, R61, R91, R121에 바뀜", "src/schedules.py:31-32"),
 ("traffic", "gradual의 ρ 식", "ρ(r) = 0.8 / (1 + exp(−(f − 0.5)/0.08)), f = (r − 1)/T", "src/schedules.py:58-59"),
 ("traffic", "client mobility의 ρ", "abrupt 일정. 120 라운드에서 R61에 0→0.8", "src/schedules.py:42-43"),
 ("traffic", "client mobility의 membership 변화", "모든 client가 매 라운드 Gauss-Markov 모델로 이동(α=0.9, 속도 12, dt=10 s, 1000×1000 지도, seed 42). 5 라운드마다 각 client의 가장 가까운 edge 2개를 다시 구하고, 그 집합이 바뀐 client만 재배치. R5에 50명 전원이 static topology에서 이동 topology로 옮겨가고, 이후 5 라운드마다 1–10명이 바뀜(로그 기준)",
  "src/runner.py:206-213, 343-355; network/mobility.py:17-63"),
 ("traffic", "late abrupt change (passive 신호 실행)", "abrupt 일정을 100 라운드로 실행. R51에 ρ 0→0.8", "src/schedules.py:42-43; rec_abrupt 실행"),
 ("cluster", "cluster 구성", f"edge server 5개, client 50명. cluster마다 10명 중 뒤쪽 5명이 다음 cluster에도 속해 {n_two}명이 두 cluster에 속함", "configs/base_v3.yaml:9-11; data/partition.py:17-47"),
 ("cluster", "edge 사이의 scalar 교환", "점수 평균: 이웃 edge(선형 0-1-2-3-4)와 1단계 평균. spatial 비교: 같은 라운드의 모든 cluster TV의 중앙값·MAD를 사용", "data/partition.py:189-199; src/controllers/self_calibrating.py:51-61, 157-164"),
 ("초기 구간", "burn-in과 warm-up", "burn-in 10 라운드 + warm-up 15 라운드 = 처음 25 라운드. 이 동안 λ=0.425, Λ=0.55 (범위의 중간값)", "src/controllers/self_calibrating.py:91,132-136,181-182; 실행 flag --burn_in 10; src/runner.py:246 (warmup 15)"),
 ("평가", "평가 라운드와 최종 예측", "R1과 10 라운드마다 (150 R: 16회, 120 R: 13회, 100 R: 11회). client exit의 entropy가 0.8을 넘으면 server exit 예측을 씀", "configs/base_v3.yaml:34-35; src/runner.py:564; eval/evaluator.py"),
]
with open(TAB / "T8_reproduction_settings.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["분류", "항목", "값", "출처(파일:줄)"]); w.writerows(T8)

CB, SB = PC * 4, PS * 4
T9 = [
 ("기본 split", "CNN split activation (표본 1개, fp32)", int(np.prod(CNN_REP)) * 4, f"feature {CNN_REP} × 4 B", "계산"),
 ("기본 split", "ResNet-18 middle split activation (표본 1개, fp32)", int(np.prod(RES_REP)) * 4, f"feature {RES_REP} × 4 B", "계산"),
 ("기본 split", "client block / server block 크기 (CNN, fp32)", f"{CB} / {SB}", f"{PC:,} / {PS:,} parameters × 4 B", "계산"),
 ("기본 split", "라운드당 모델 교환량 (두 cluster에 속한 client 1명, 한 방향)", CB + 2 * SB,
  "client block 1개 + server block 2개. tables/communication_accounting.csv의 12,149,112 B와 정확히 같음", "journal_expansion/tables/communication_accounting.csv"),
 ("기본 split", "server-block 사본을 맞추는 downlink (두 cluster client, 라운드당)", 2 * SB,
  "cluster 평균 server block 2개를 받음(한 cluster client는 5,514,792 B). 12,149,112 B는 한 방향의 크기이므로 이 downlink는 따로 더해야 함", "train/trainer.py:276 (set_server_state, cell-based 경로)"),
 ("기본 split", "client block downlink (cluster 평균과 섞을 모델, 라운드당)", CB, "cluster 평균 client block 1개", "train/trainer.py:259-273 (mix_state_dicts, 271줄)"),
 ("DriftGate 추가", "client → edge TV scalar", 4, "client가 속한 edge마다 라운드당 4 B (두 cluster client는 8 B)", "src/signals/library.py:215-223 (aggregate_per_es)"),
 ("DriftGate 추가", "edge ↔ 이웃 edge 점수 평균", "4·deg(e)", "선형 graph라 deg(e)는 1 또는 2, 즉 edge당 4 B 또는 8 B", "self_calibrating.py:63-71,164"),
 ("DriftGate 추가", "spatial 비교를 위한 cluster TV 공유", "4·(Z−1)", "spatial 점수는 같은 라운드의 모든 cluster TV를 쓰므로 edge마다 다른 Z−1개 cluster의 TV가 필요함. Z=5면 16 B", "self_calibrating.py:51-61,157"),
 ("추정치", "edge에서 server exit를 계산할 때 probe 업로드 (probe 64개)", 64 * (int(np.prod(CNN_REP)) * 4) + 64 * 10 * 4,
  "추정치. activation 64 × 32,768 B + client 확률 벡터 64 × C × 4 B (C=10). 확률 벡터 대신 edge가 server 확률 벡터 64 × 40 B를 내려보내도 같은 크기", "계산 (현재 구현은 client가 가진 server-block 사본으로 계산하므로 이 전송이 없음)"),
]
with open(TAB / "T9_communication.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["분류", "항목", "bytes", "근거", "출처"]); w.writerows(T9)
print("T8 rows", len(T8), "| T9 rows", len(T9)); [print("  ", k, v) for k, v in sizes.items()]
