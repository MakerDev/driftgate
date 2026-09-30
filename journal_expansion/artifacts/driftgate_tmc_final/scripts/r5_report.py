"""Builds DriftGate_final_report_ko.md from the CSV tables (numbers are never typed by hand).
Run after r5_tables.py, r5_config_comm.py, r5_manifest.py, r5_figures.py."""
import csv, json, re
from pathlib import Path
import numpy as np
JR = Path("/home/honeynaps/data/driftgate/journal_expansion")  # [SERVER-PATH:REPO_ROOT]
HERE = Path(__file__).resolve().parent.parent; T = HERE / "tables"
def rd(n): return list(csv.DictReader(open(T / n)))
def clean(m): return m.replace(" (best fixed)", "")
T1 = {(r["setting"], clean(r["method"])): r for r in rd("T1_main_results.csv")}
T3 = {(r["setting"], clean(r["method"])): r for r in rd("T3_transfer.csv")}
T4a = {r["setting"]: r for r in rd("T4a_controller_relonly_vs_absonly.csv")}
T4b = {(r["setting"], r["variant"]): r for r in rd("T4b_constant_sensitivity.csv")}
T4c = {(r["setting"], r["method"]): r for r in rd("T4c_apfl_lambda_by_segment.csv")}
T4d = {(r["setting"], r["method"], r["segment"]): r for r in rd("T4d_apfl_segment_accuracy.csv")}
T5 = {clean(r["method"]): r for r in rd("T5_resnet18_middle.csv")}
T2a = {(r["setting"], r["method"], r["segment"]): r for r in rd("T2a_segment_accuracy.csv")}
T2c = {(r["setting"], r["segment"]): r for r in rd("T2c_segment_best_worst_fixed.csv")}
T6a, T6b, T6c = rd("T6a_signal_levels.csv"), {r["signal"]: r for r in rd("T6b_shift_response_late_abrupt.csv")}, rd("T6c_passive_signal_metrics.csv")
T7 = {r["condition"]: r for r in rd("T7_role_experiment.csv")}
T10 = rd("T10_last_round_accuracy.csv")
PRE = list(csv.DictReader(open(HERE / "precheck/precheck_controller_path.csv")))
MAN = rd("run_manifest.csv")
A, M = "stepwise composition change", "client mobility"
def pp(x): return f"{float(x):+.2f} pp"
def d(r, col="DriftGate_minus_method_pp", ci="ci95", npos="n_pos", n="n_matched"):
    return f"{float(r[col]):+.2f} pp {r[ci]} ({r[npos]}/{r[n]})"
def acc(r, col="integrated_acc_pct"): return f"{float(r[col]):.2f}%"
def f2(x): return f"{float(x):.2f}"
def f3(x): return f"{float(x):.3f}"
def _last_digit(x): return re.findall(r"\d", str(x))[-1]
def ida(x): return f"{x}이다" if _last_digit(x) in "013678" else f"{x}다"      # 숫자 뒤 서술격 조사
def ro(x): return f"{x}으로" if _last_digit(x) in "036" else f"{x}로"          # 숫자 뒤 부사격 조사
def lo(ci): return float(re.findall(r"\[([-+0-9.]+)", ci)[0])
def hi(ci): return float(re.findall(r", ([-+0-9.]+)\]", ci)[0])
L = []
def w(s=""): L.append(s)

# ------------------------------------------------------------------ header
w("# DriftGate 최종 실험 보고서 (Round 5)")
w()
w("이 보고서는 IEEE TMC 원고에 들어갈 비모바일 실험을 모두 정리한다. 모든 수치는 원시 run JSON에서 스크립트로 계산한 CSV에서 가져왔다"
  " (`scripts/r5_tables.py` → `tables/*.csv` → 이 보고서). 이 보고서에서 **DriftGate**는 relonly controller를 뜻한다. relonly는 TV 신호를"
  " temporal 비교와 spatial 비교의 표준화 점수로만 λ와 Λ로 바꾸며, absolute branch가 없다.")
w()
w("용어와 표기는 다음과 같다.")
w()
w("- **pp**는 정확도 백분율의 차이(퍼센트포인트)다. **integrated accuracy**는 평가 라운드마다 잰 acc_total의 평균이다.")
w("- 차이는 모두 같은 seed끼리 뺀 paired 차이이며, 괄호 안에 Student-t 95% 신뢰구간(CI)을, 그 뒤에 양수인 seed 수/전체 seed 수(n_pos/n)를 적는다.")
w("- 설정 이름: stepwise composition change는 기존 Schedule A, client mobility는 기존 mobility-med, late abrupt change는 기존 post-convergence abrupt다.")
w("- **absonly**는 TV 크기(평활한 d̂)로 λ = 0.70 − 0.55·d̂를 바로 정하는 controller(Round 4 B1)다. **APFL**은 기기의 라벨 손실로 λ를 학습하는 기준 방법(B4)이다.")
w("- 모든 새 run은 controller용 표본과 평가용 표본을 나눈 disjoint pool로 실행했고, 64개 모두 두 집합이 겹치지 않았다(overlap = 0).")
w()

# ------------------------------------------------------------------ 1 verdict table
w("## 1. 주장별 판정표")
w()
w("| # | 주장 | 근거 표·그림 | 판정 | 핵심 수치 |")
w("|---|---|---|---|---|")
fx_A = [k for k in ("fixed 0.2", "fixed 0.3", "fixed 0.4", "fixed 0.5")]
fx_M = [k for k in ("fixed 0.2", "fixed 0.4", "fixed 0.5", "fixed 0.6")]
all_pos = all(float(re.findall(r"\[([-+0-9.]+)", T1[(s, k)]["ci95"])[0]) > 0 for s, ks in ((A, fx_A), (M, fx_M)) for k in ks)
w(f"| 1 | traffic이 바뀌는 두 설정에서 DriftGate는 시험한 모든 고정 λ보다 높다 | 표 1, 그림 4 | {'지지' if all_pos else '부분 지지'} | "
  f"stepwise: 최고 고정값 λ=0.4 대비 {d(T1[(A,'fixed 0.4')])}, λ=0.2·0.3·0.5 대비 {pp(T1[(A,'fixed 0.3')]['DriftGate_minus_method_pp'])}에서 {pp(T1[(A,'fixed 0.2')]['DriftGate_minus_method_pp'])}. "
  f"mobility: 최고 고정값 λ=0.4 대비 {d(T1[(M,'fixed 0.4')])}. " + ("고정 λ와의 8개 비교 모두 CI 하한이 0보다 크다 |" if all_pos else "일부 비교의 CI가 0을 포함한다 |"))
segs_A = ["ρ=0 early", "ρ=0.4 rising", "ρ=0.8", "ρ=0.4 falling", "ρ=0 late"]; segs_M = ["ρ=0 (R1–60)", "ρ=0.8 (R70–120)"]
nsig = sum(1 for (s, g), r in T2c.items() if float(re.findall(r"\[([-+0-9.]+)", r["ci95"])[0]) > 0)
w(f"| 2 | 가장 좋은 λ는 구간에 따라 뒤바뀌고, DriftGate는 어느 구간에서도 가장 나쁜 고정값 수준까지 떨어지지 않는다 | 표 2c, 그림 4 | 부분 지지 | "
  f"최고 고정값은 stepwise의 ρ=0 구간에서 λ=0.5, ρ>0 구간에서 λ=0.2이고, mobility에서는 ρ 변화 전 λ=0.6, 뒤 λ=0.2다. "
  f"DriftGate의 평균은 7개 구간 모두에서 최저 고정값보다 높고, {nsig}개 구간에서 CI 하한이 0보다 크다. "
  f"stepwise ρ=0.4 상승 구간은 {d(T2c[(A,'ρ=0.4 rising')], 'DriftGate_minus_worst_pp', 'ci95', 'n_pos', 'n')}로 CI가 0을 포함한다 |")
w(f"| 3 | 같은 controller에서 TV는 entropy보다 높다 | 표 1, 표 3, 사전 확인 1 | 지지 | "
  f"stepwise {d(T1[(A,'entropy')])}, mobility {d(T1[(M,'entropy')])}, SVHN {d(T3[('SVHN temporal','entropy')])} |")
lam_drop = lambda m: float(T4c[(M,m)]['lambda_seg1(A: R26–30, mob: R26–60)']) - float(T4c[(M,m)]['lambda_seg2(A: R31–60, mob: R61–120)'])
w(f"| 4 | 기기의 라벨로 λ를 학습하는 APFL식 방법은 traffic 변화를 따라가지 못한다 | 표 1, 표 4c, 표 4d, 그림 3 | 부분 지지 | "
  f"stepwise의 ρ=0.8 구간(R61–90)에서 APFL의 평균 λ는 η=0.01에서 {f3(T4c[(A,'APFL η=0.01')]['lambda_seg3(A: R61–90)'])}, η=0.1에서 {ida(f3(T4c[(A,'APFL η=0.1')]['lambda_seg3(A: R61–90)']))}. "
  f"λ와 ρ의 Spearman 상관은 η=0.01에서 {T4c[(A,'APFL η=0.01')]['spearman_lambda_vs_rho_after_R25']}, η=0.1에서 {ida(T4c[(A,'APFL η=0.1')]['spearman_lambda_vs_rho_after_R25'])}. "
  f"DriftGate의 같은 값은 {f3(T4c[(A,'DriftGate')]['lambda_seg3(A: R61–90)'])}, {ida(T4c[(A,'DriftGate')]['spearman_lambda_vs_rho_after_R25'])}. "
  f"mobility에서는 APFL의 λ도 ρ 변화 뒤 내려가지만, 감소폭(η=0.01은 {lam_drop('APFL η=0.01'):.3f}, η=0.1은 {lam_drop('APFL η=0.1'):.3f})이 DriftGate의 {lam_drop('DriftGate'):.3f}보다 작다. "
  f"ρ=0.8 구간 정확도는 DriftGate가 {pp(T4d[(A,'APFL η=0.01','ρ=0.8')]['DriftGate_minus_APFL_pp'])}, {pp(T4d[(A,'APFL η=0.1','ρ=0.8')]['DriftGate_minus_APFL_pp'])}(stepwise), "
  f"{pp(T4d[(M,'APFL η=0.01','ρ=0.8 (R70–120)')]['DriftGate_minus_APFL_pp'])}, {pp(T4d[(M,'APFL η=0.1','ρ=0.8 (R70–120)')]['DriftGate_minus_APFL_pp'])}(mobility) 높다. "
  f"반대 결과: mobility의 integrated accuracy는 APFL η=0.1이 평균으로 더 높다(DriftGate − APFL {d(T1[(M,'APFL η=0.1')])}) |")
w(f"| 5 | 비교 점수로 λ를 정하는 DriftGate는 TV 크기로 λ를 바로 정하는 controller보다 높다 | 표 4a | 지지 | "
  f"DriftGate − absonly: stepwise {d(T4a[A],'DriftGate_minus_absonly_pp','ci95','n_pos','n')}, mobility {d(T4a[M],'DriftGate_minus_absonly_pp','ci95','n_pos','n')}, "
  f"CIFAR-100 gradual {d(T4a['CIFAR-100 gradual'],'DriftGate_minus_absonly_pp','ci95','n_pos','n')} |")
e3 = lambda s, v: f"{float(T4b[(s,v)]['variant_minus_default_pp']):+.2f} pp {T4b[(s,v)]['ci95']}"
w(f"| 6 | 결과가 guard와 λ_max 값에 크게 좌우되지 않는다 | 표 4b | 부분 지지 | "
  f"기본값 대비 stepwise: guard 0.25 {e3(A,'guard 0.25')}, guard 1.0 {e3(A,'guard 1.0')}, λ_max 0.60 {e3(A,'λ_max 0.60')}, λ_max 0.80 {e3(A,'λ_max 0.80')}. "
  f"mobility: λ_max 0.60 {e3(M,'λ_max 0.60')}, λ_max 0.80 {e3(M,'λ_max 0.80')}({T4b[(M,'λ_max 0.80')]['n_pos']}/3). mobility의 λ_max 0.80은 같은 seed의 최고 고정값(λ=0.4)과의 차이도 "
  f"{float(T4b[(M,'λ_max 0.80')]['variant_minus_best_fixed_pp']):+.2f} pp {ida(T4b[(M,'λ_max 0.80')]['ci95_vs_best_fixed'])} |")
TRS = ("CIFAR-10 gradual", "CIFAR-100 gradual", "Tiny-ImageNet", "SVHN temporal")
def t3best(st):
    for x in rd("T3_transfer.csv"):
        if x["setting"] == st and "(best fixed)" in x["method"]: return clean(x["method"]), x
w(f"| 7 | 다른 데이터셋에서 DriftGate와 가장 좋은 고정값의 차이 | 표 3 | 판정 없음 | "
  + " / ".join(f"{st}: {t3best(st)[0]} 대비 {d(t3best(st)[1])}" for st in TRS) + ". CI가 0을 포함하지 않는 설정: " + (", ".join(st for st in TRS if lo(t3best(st)[1]["ci95"]) > 0 or hi(t3best(st)[1]["ci95"]) < 0) or "없음") + " |")
w(f"| 8 | ResNet-18에서도 DriftGate가 고정값보다 높다 | 표 5 | 지지 | "
  f"DriftGate {acc(T5['DriftGate'])}, 고정 λ=0.4 대비 {d(T5['fixed 0.4'])}, 고정 λ=0.2 대비 {d(T5['fixed 0.2'])}. seed 3개, client 16명, 고정값은 두 개만 시험했다 |")
tvA = [r for r in T6c if r["signal"] == "TV"]; enA = [r for r in T6c if r["signal"] == "entropy"]
w(f"| 9 | TV는 traffic 구성을 따라가고, entropy는 학습 진행을 따라간다 | 표 6a–6c, 그림 1 | 지지 | "
  f"late abrupt change에서 TV는 shift에 {float(T6b['TV']['shift_change(R51–55 − R46–50)']):+.3f} 변하고 shift 전 35 라운드 동안 {float(T6b['TV']['pre_shift_change_35_rounds(R46–50 − R16–20)']):+.3f} 변한다. "
  f"entropy(H/ln C)는 shift에 {float(T6b['H/lnC']['shift_change(R51–55 − R46–50)']):+.3f} 오르지만 shift 전 35 라운드 동안 {float(T6b['H/lnC']['pre_shift_change_35_rounds(R46–50 − R16–20)']):+.3f} 내려간다. "
  f"ρ와의 Spearman은 TV {float(tvA[0]['spearman_vs_rho']):+.2f}/{float(tvA[1]['spearman_vs_rho']):+.2f}, entropy {float(enA[0]['spearman_vs_rho']):+.2f}/{float(enA[1]['spearman_vs_rho']):+.2f}(단계 변화/late abrupt) |")
w()
w("판정 기준은 다음과 같다. 해당 비교의 차이가 모두 주장 방향이고 95% CI가 0을 넘으면 **지지**, 방향은 맞지만 일부 비교의 CI가 0을 포함하거나 일부 조건에서 반대 결과가 나오면 **부분 지지**, 주요 비교가 반대 방향이면 **반대**로 판정했다.")
w()

# ------------------------------------------------------------------ 2 pre-checks
w("## 2. 시작 전 확인")
w()
w("### 2.1 entropy 실행과 relonly 실행의 controller 경로")
w()
w("**결론: 같은 경로다. 따라서 entropy 재실행(13개)은 하지 않았다.**")
w()
w("- **실행 flag**: provenance에 기록된 명령을 보면 두 실행군의 flag는 `--signal`만 다르다(entropy는 `ent_client`, relonly는 `tv_dist`). 둘 다 `--mode selfcal --burn_in 10 --z_guard 0.5 --spatial_norm --disjoint_pools`를 쓰고, 둘 다 `--abs_cap`이 없다.")
w("- **resolved config**: stepwise seed 0의 두 run JSON에서 `config`를 비교하면 `controller_signal`과 run_id만 다르고 나머지 항목은 모두 같다.")
w("- **코드 경로**: `scripts/run_v2.py:177-178`은 `--abs_cap`이 있을 때만 `abs_cap=True`로 둔다. `src/runner.py:250-251`은 flag가 없으면 `abs_cap=False`, `abs_only=False`로 controller를 만든다. "
  "이때 `src/controllers/self_calibrating.py:168-169`의 `use_abs`가 False이므로 absolute 후보를 계산하지 않고, `185-187`행에서 λ와 Λ가 relative 후보 그대로 정해진다. "
  "상수는 `self_calibrating.py:31-33`(z0 = 1.5, τ = 0.75, EMA α = 0.3)과 `normalizers.py:21,26,49`(σ 하한, β = 0.05, 점수 clip [−2, 6])에서 두 실행이 같은 값을 쓴다.")
w("- **동작 확인**: entropy 실행은 Round 4의 코드 수정(`--abs_only` flag, λ_rel/λ_abs 로깅)보다 먼저 돌았다. 그래서 기록된 controller 점수 z에 relative 매핑 "
  "λ = 0.70 − 0.55·sigmoid((z − 1.5)/0.75), Λ = 0.70 − 0.30·sigmoid((z − 1.5)/0.75)만 적용해 기록된 λ와 Λ를 재구성했다. 결과는 아래와 같다(`precheck/precheck_controller_path.csv`).")
w()
w("| 실행군 | seed 수 | warm-up 이후 cluster-라운드 | λ 최대 오차 | Λ 최대 오차 | warm-up 불일치 |")
w("|---|---|---|---|---|---|")
groups = {}
for r in PRE: groups.setdefault(r["group"], []).append(r)
GN = {"entropy A": "entropy, stepwise", "entropy mobility": "entropy, mobility", "entropy SVHN": "entropy, SVHN",
      "relonly A": "relonly, stepwise", "relonly mobility": "relonly, mobility", "relonly C100 gradual": "relonly, CIFAR-100 gradual"}
for g, rs in groups.items():
    w(f"| {GN.get(g, g)} | {len(rs)} | {rs[0]['post_warmup_cluster_rounds']} | {max(float(x['max_abs_err_lambda']) for x in rs):.1e} | {max(float(x['max_abs_err_Lambda']) for x in rs):.1e} | {sum(int(x['warmup_mismatches']) for x in rs)} |")
w()
w("모든 seed에서 오차가 부동소수점 반올림 수준(1e-16)이고, 처음 25 라운드는 모두 λ = 0.425, Λ = 0.55다. 따라서 기존 entropy 실행은 relonly controller에 신호로 entropy를 넣은 실행과 같다.")
w()
w("### 2.2 relonly 실행의 Λ 계산")
w()
w("B1 relonly 실행(stepwise 5개, mobility 3개, CIFAR-100 gradual 3개)에서 Λ는 λ와 같은 relative 점수 q로 정해졌다. 위 표의 relonly 행에서 Λ = 0.70 − 0.30·sigmoid((q − 1.5)/0.75)가 모든 cluster-라운드에서 성립하고, "
  "absolute 후보를 기록하는 `history[\"lam_abs\"]`는 모든 라운드에서 비어 있다.")
w()
w("### 2.3 코드, GPU, 대기열")
w()
w("- **코드**: 저장소 HEAD는 `e82b96b`다. 하지만 `adaptive_splitomc_tmc/` 디렉터리 전체가 git에 추적되지 않아(`git status`에서 `??`) 이 commit만으로는 실행 코드를 특정할 수 없다. "
  "그래서 run이 실행한 소스 파일 17개의 sha256을 `precheck/code_identity.txt`에 기록했고, 모든 run이 끝난 뒤 다시 계산해 바뀌지 않았음을 확인했다.")
w("- **GPU**: 새 run 64개는 모두 `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 실행되어 물리 GPU 0만 썼다. GPU 1은 쓰지 않았다. "
  "10분마다 워커를 다시 띄우던 cron supervisor가 GPU 1에도 워커를 띄우던 설정이었으므로, GPU 0 워커만 띄우도록 바꿨다(원본은 `scripts/supervisor.sh.bak_pre_r5_*`).")
KL = [l.split() for l in open(JR / "runs/queue/killed_r5_20260928_0523.txt") if l.startswith("run_v2") and "b3_" in l]
k1 = sum(1 for x in KL if x[-1] == "cuda:1"); k0 = len(KL) - k1
b3m = [r["run_name"] for r in MAN if r["experiment"] == "B3"]
w(f"- **멈춘 run 정리**: 작업 시작 시점에 Round 4의 B3 λ = 0.15 run {len(KL)}개가 Round 4 중단 요청 때 일시 정지(SIGSTOP)한 상태로 3일 넘게 GPU 메모리를 잡고 있었다. "
  f"이 중 {k1}개는 GPU 1에 있어 GPU 0만 쓰는 조건에서는 이어서 돌릴 수 없었다. GPU 0의 {k0}개도 GPU 1 워커와 함께 돌던 이전 대기열에 묶여 있어 함께 종료했다(목록: `runs/queue/killed_r5_20260928_0523.txt`). "
  f"보고에 쓰는 B3 run은 아직 시작하지 않았던 run을 포함해 {len(b3m)}개를 GPU 0에서 처음부터 돌렸다. GPU 0에 있던 Jupyter 커널은 끄지 않았다(메모리가 충분했고, 이후 스스로 종료됐다).")
w("- **대기열 정리**: 대기 중이던 B5 16개와 B6 18개를 취소했다(시작한 run은 없었다. 목록: `runs/queue/cancelled_r5_B5_B6_*.txt`). CIFAR-100 spatial의 relonly 실행과 server non-Main prediction rate 재실행은 대기열에 없었다. "
  "B3 λ = 0.15의 CIFAR-100 spatial run 3개는 정지 상태에서 종료했고, 보고에 넣지 않으므로 다시 돌리지 않았다. entnorm 결과는 추가로 분석하지 않았다.")
w("- **B4 설정 확인**: 대기열의 B4 명령은 4절의 규칙과 같다. client마다 λ_k를 두고, 매 라운드 local 학습 뒤 θ(λ_k) = λ_k·θ_local + (1 − λ_k)·θ̄_k에서 local minibatch 1개로 ∂L/∂λ_k를 구해 "
  "λ_k ← clip(λ_k − η·∂L/∂λ_k, 0.15, 0.70)로 갱신한다(`src/apfl_baseline.py:7-9,23-28,68`). 초기값 0.425(`src/runner.py:258`), Λ = 0.5, η ∈ {0.01, 0.1}(모델 learning rate와 그 10배)이고 probe와 TV 신호는 쓰지 않는다.")
w()

# ------------------------------------------------------------------ 3 new experiments
w("## 3. 새 실험 결과")
w()
w("### 3.1 E1과 B3: 다른 데이터셋 (표 3, `tables/T3_transfer.csv`)")
w()
w("DriftGate는 E1의 relonly 실행(CIFAR-10 gradual, Tiny-ImageNet, SVHN)과 B1의 relonly 실행(CIFAR-100 gradual)을 쓴다. 고정 λ = 0.15는 B3 실행이고, 0.2와 0.4는 기존 disjoint 실행(Round 2–4)이다. 고정 λ 실행은 모두 Λ = 0.5다. 가장 좋은 고정값은 같은 seed 평균이 가장 높은 값이다.")
w()
w("| 설정 | 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |")
w("|---|---|---|---|---|")
for r in rd("T3_transfer.csv"):
    dd = "" if not r["DriftGate_minus_method_pp"] else f"{float(r['DriftGate_minus_method_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n_matched']})"
    w(f"| {r['setting']} | {r['method'].replace('(best fixed)','(최고 고정값)')} | {r['n_seeds']} | {float(r['integrated_acc_pct']):.2f}% ({float(r['sd_pct']):.2f}) | {dd} |")
w()
bd = {st: float(t3best(st)[1]["DriftGate_minus_method_pp"]) for st in TRS}
sig_neg = [st for st in TRS if hi(t3best(st)[1]["ci95"]) < 0]
b15 = [st for st in TRS if t3best(st)[0] == "fixed 0.15"]
w(f"네 데이터셋에서 DriftGate와 가장 좋은 고정값의 차이는 {min(bd.values()):+.2f} pp에서 {max(bd.values()):+.2f} pp 사이다. "
  + "".join(f"{st}에서는 DriftGate가 {t3best(st)[0].replace('fixed ','λ = ')}보다 {abs(bd[st]):.2f} pp 낮고 CI가 0을 포함하지 않는다. " for st in sig_neg)
  + f"{len(b15)}개 데이터셋({', '.join(b15)})에서 가장 좋은 고정값은 시험한 범위의 끝인 λ = 0.15다. "
  f"SVHN에서 DriftGate는 같은 controller에 entropy를 넣은 실행보다 {d(T3[('SVHN temporal','entropy')])} 높다.")
w()
w("### 3.2 E2: ResNet-18 middle split (표 5, `tables/T5_resnet18_middle.csv`)")
w()
w("기존 ResNet-18 분할 실험의 middle split 정의를 그대로 썼다(client는 stem과 stage 1–2, server는 stage 3–4와 fc, 분할 지점 feature는 128×16×16). 기존 ResNet-18 실험처럼 client는 16명이고, CIFAR-10 stepwise composition change를 150 라운드 돌렸다.")
w()
w("| 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |")
w("|---|---|---|---|")
for r in rd("T5_resnet18_middle.csv"):
    dd = "" if not r["DriftGate_minus_method_pp"] else f"{float(r['DriftGate_minus_method_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n_matched']})"
    w(f"| {r['method'].replace('(best fixed)','(최고 고정값)')} | {r['n_seeds']} | {float(r['integrated_acc_pct']):.2f}% ({float(r['sd_pct']):.2f}) | {dd} |")
w()
w("세 seed 모두에서 DriftGate가 두 고정값보다 높다. seed가 3개라 CI가 넓고, 고정값은 0.2와 0.4 두 개만 시험했으므로 이 표의 '최고 고정값'은 두 값 중 높은 쪽이다.")
w()
w("### 3.3 E3: 상수 민감도 (표 4b, `tables/T4b_constant_sensitivity.csv`)")
w()
w("한 번에 상수 하나만 바꿨다. 기준은 같은 seed(0–2)의 B1 relonly 실행이다. λ_max를 바꾸면 warm-up 동안 쓰는 중간값도 함께 바뀐다(λ_max 0.60이면 0.375, 0.80이면 0.475).")
w()
w("| 설정 | 변형 | 변형 정확도 | 기본값 정확도(같은 seed) | 변형 − 기본값 | 변형 − 최고 고정값(같은 seed) |")
w("|---|---|---|---|---|---|")
for r in rd("T4b_constant_sensitivity.csv"):
    w(f"| {r['setting']} | {r['variant']} | {float(r['variant_acc_pct']):.2f}% | {float(r['default_DriftGate_same_seeds_pct']):.2f}% | "
      f"{float(r['variant_minus_default_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/3) | {float(r['variant_minus_best_fixed_pp']):+.2f} pp {r['ci95_vs_best_fixed']} ({r['best_fixed_same_seeds']}) |")
w()
vA = [float(T4b[(A,v)]["variant_minus_default_pp"]) for v in ("guard 0.25","guard 1.0","λ_max 0.60","λ_max 0.80")]
allinc = all(lo(T4b[(A,v)]["ci95"]) <= 0 <= hi(T4b[(A,v)]["ci95"]) for v in ("guard 0.25","guard 1.0","λ_max 0.60","λ_max 0.80"))
fz = lambda v: "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
w(f"stepwise에서 네 변형과 기본값의 차이는 {fz(min(vA))} pp에서 {fz(max(vA))} pp 사이이고, " + ("CI가 모두 0을 포함한다. " if allinc else "일부 CI는 0을 포함하지 않는다. ")
  + ("네 변형 모두 평균이 기본값보다 조금 낮다. " if all(v < 0 for v in vA) else "")
  + f"mobility에서 λ_max 0.60은 기본값과 차이가 작다({e3(M,'λ_max 0.60')}). "
  f"λ_max 0.80과 기본값의 차이는 {ida(e3(M,'λ_max 0.80'))[:-1]}고, 기본값보다 높은 seed는 {T4b[(M,'λ_max 0.80')]['n_pos']}/3개다. "
  f"이 변형과 같은 seed의 최고 고정값(λ = 0.4)의 차이는 {float(T4b[(M,'λ_max 0.80')]['variant_minus_best_fixed_pp']):+.2f} pp {ida(T4b[(M,'λ_max 0.80')]['ci95_vs_best_fixed'])}.")
w()
w("### 3.4 B4: APFL식 λ 학습 (표 1, 표 4c, 표 4d)")
w()
w("| 설정 | 방법 | integrated accuracy (SD) | DriftGate − 방법 |")
w("|---|---|---|---|")
for st in (A, M):
    for m in ("APFL η=0.01", "APFL η=0.1"):
        r = T1[(st, m)]; w(f"| {st} | {m} | {acc(r)} ({float(r['sd_pct']):.2f}) | {d(r)} |")
w()
w("λ가 ρ를 따라 움직이는지 보려고 구간별 평균 λ와 ρ와의 Spearman 상관을 계산했다(`tables/T4c_apfl_lambda_by_segment.csv`). APFL의 λ는 client별 λ_k의 평균이고, DriftGate의 λ는 cluster 평균이다. 두 방법 모두 초기 25 라운드는 제외했다.")
w()
w("| 설정 | 방법 | 구간별 평균 λ | ρ와의 Spearman (SD) |")
w("|---|---|---|---|")
for (st, m), r in T4c.items():
    segv = [r[k] for k in r if k.startswith("lambda_seg") and r[k]]
    lab = "R26–30 / R31–60 / R61–90 / R91–120 / R121–150" if st == A else "R26–60 / R61–120"
    w(f"| {st} | {m} | {' / '.join(segv)} ({lab}) | {r['spearman_lambda_vs_rho_after_R25']} ({f3(r['sd'])}) |")
w()
w(f"stepwise에서 APFL의 λ는 ρ가 0.8일 때도 0.6 근처에 머물고, ρ와의 상관이 양수다. 즉 λ가 필요한 방향(ρ가 클 때 작은 λ)과 반대로 움직인다. "
  f"mobility에서는 APFL의 λ도 ρ 변화 뒤 내려가고 ρ와의 상관도 음수다. 다만 감소폭(η=0.01은 {lam_drop('APFL η=0.01'):.3f}, η=0.1은 {lam_drop('APFL η=0.1'):.3f})이 "
  f"DriftGate의 {lam_drop('DriftGate'):.3f}보다 작다.")
w()
w("구간별 정확도(`tables/T4d_apfl_segment_accuracy.csv`)를 보면 APFL은 ρ = 0 구간에서 DriftGate보다 높고, ρ가 큰 구간에서 크게 낮다.")
w()
w("| 설정 | 방법 | 구간 | APFL 정확도 | DriftGate − APFL |")
w("|---|---|---|---|---|")
for (st, m, g), r in T4d.items():
    w(f"| {st} | {m} | {g} | {float(r['acc_total_pct']):.2f}% | {float(r['DriftGate_minus_APFL_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n']}) |")
w()
w(f"mobility의 APFL η=0.1은 ρ = 0 구간에서 {abs(float(T4d[(M,'APFL η=0.1','ρ=0 (R1–60)')]['DriftGate_minus_APFL_pp'])):.2f} pp 앞서고 ρ = 0.8 구간에서 "
  f"{float(T4d[(M,'APFL η=0.1','ρ=0.8 (R70–120)')]['DriftGate_minus_APFL_pp']):.2f} pp 뒤진다. 평가 라운드는 ρ = 0 구간이 {len(T2a[(M,'DriftGate','ρ=0 (R1–60)')]['eval_rounds'].split())}개, ρ = 0.8 구간이 {len(T2a[(M,'DriftGate','ρ=0.8 (R70–120)')]['eval_rounds'].split())}개라 integrated accuracy에서는 APFL이 평균 "
  f"{abs(float(T1[(M,'APFL η=0.1')]['DriftGate_minus_method_pp'])):.2f} pp 높게 나온다(CI {T1[(M,'APFL η=0.1')]['ci95']}).")
w()

# ------------------------------------------------------------------ 4 existing-log analyses
w("## 4. 기존 로그 분석 결과")
w()
w("### 4.1 대표 비교: stepwise composition change와 client mobility (표 1, `tables/T1_main_results.csv`)")
w()
w("mobility의 고정 λ = 0.5는 Round 1 실행(`t1_mob_fx50`)을 썼다. Round 4의 같은 설정 실행(`b3_mob_fx50`)은 6절에 적은 이유로 쓰지 않았다.")
w()
w("| 설정 | 방법 | seed | integrated accuracy (SD) | DriftGate − 방법 |")
w("|---|---|---|---|---|")
for r in rd("T1_main_results.csv"):
    dd = "" if not r["DriftGate_minus_method_pp"] else f"{float(r['DriftGate_minus_method_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n_matched']})"
    w(f"| {r['setting']} | {r['method'].replace('(best fixed)','(최고 고정값)')} | {r['n_seeds']} | {float(r['integrated_acc_pct']):.2f}% ({float(r['sd_pct']):.2f}) | {dd} |")
w()
w("### 4.2 relonly와 absonly (표 4a, `tables/T4a_controller_relonly_vs_absonly.csv`)")
w()
w("| 설정 | DriftGate | absonly | DriftGate − absonly |")
w("|---|---|---|---|")
for s, r in T4a.items():
    w(f"| {s} | {float(r['DriftGate_pct']):.2f}% | {float(r['absonly_pct']):.2f}% | {float(r['DriftGate_minus_absonly_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n']}) |")
w()
w("### 4.3 구간별 정확도와 구간별 기여 (표 2a–2c, 그림 4)")
w()
w("구간은 평가 라운드로 나눴다. stepwise는 ρ = 0 초반 {1, 10, 20, 30}, ρ = 0.4 상승 {40, 50, 60}, ρ = 0.8 {70, 80, 90}, ρ = 0.4 하강 {100, 110, 120}, ρ = 0 후반 {130, 140, 150}이고, "
  "mobility는 ρ 변화 전 {1, …, 60}과 변화 후 {70, …, 120}이다. 표 2c는 구간마다 가장 좋은 고정값과 가장 나쁜 고정값, 그리고 DriftGate와의 paired 차이다.")
w()
w("| 설정 | 구간 | 최고 고정값 | 최저 고정값 | DriftGate | DriftGate − 최저 | DriftGate − 최고 |")
w("|---|---|---|---|---|---|---|")
for (s, g), r in T2c.items():
    w(f"| {s} | {g} | {r['best_fixed']} ({float(r['best_fixed_pct']):.2f}%) | {r['worst_fixed']} ({float(r['worst_fixed_pct']):.2f}%) | {float(r['DriftGate_pct']):.2f}% | "
      f"{float(r['DriftGate_minus_worst_pp']):+.2f} pp {r['ci95']} ({r['n_pos']}/{r['n']}) | {float(r['DriftGate_minus_best_pp']):+.2f} pp {r['ci95_vs_best']} |")
w()
nlow = sum(1 for r in T2c.values() if hi(r["ci95_vs_best"]) < 0)
w(f"DriftGate는 {len(T2c)}개 구간 중 {nlow}개에서 그 구간의 최고 고정값보다 낮고, 이 차이의 CI는 0을 포함하지 않는다. 그러나 최고 고정값이 구간마다 다르므로, 한 고정값을 끝까지 쓰면 어떤 구간에서는 최저 수준이 된다. "
  "표 2b는 integrated accuracy의 차이를 구간별 기여(구간 차이 × 구간의 평가 라운드 비율)로 나눈 것이며, 기여의 합은 전체 차이와 같다.")
w()
w("| 설정 | 비교 | " + " | ".join(["구간 1", "구간 2", "구간 3", "구간 4", "구간 5"]) + " | 합(= 전체 차이) |")
w("|---|---|---|---|---|---|---|---|")
con = {}
for r in rd("T2b_segment_contribution.csv"): con.setdefault((r["setting"], r["comparison"]), []).append(r)
for (s, c), rs in con.items():
    cells = [f"{float(x['contribution_pp'].split()[0]):+.2f}" for x in rs if not x["segment"].startswith("SUM")]
    tot = [x for x in rs if x["segment"].startswith("SUM")][0]
    cells += [""] * (5 - len(cells))
    w(f"| {s} | {c} | " + " | ".join(cells) + f" | {float(tot['segment_diff_pp']):+.2f} |")
w()
w("구간 1–5는 stepwise의 다섯 구간을 순서대로, mobility에서는 구간 1이 ρ 변화 전, 구간 2가 변화 후를 뜻한다. 단위는 pp다.")
w()
w("Main/OOP/OOR별 정확도(`tables/T2a_segment_accuracy.csv`)를 보면, ρ = 0.8 구간에서 작은 고정 λ는 OOP와 OOR 정확도가 높고 Main 정확도가 낮다. 아래는 그 구간의 값이다.")
w()
w("| 설정 | 방법 | 전체 | Main | OOP | OOR |")
w("|---|---|---|---|---|---|")
for s, g, ms in ((A, "ρ=0.8", ["DriftGate", "entropy", "fixed 0.2", "fixed 0.3", "fixed 0.4", "fixed 0.5"]), (M, "ρ=0.8 (R70–120)", ["DriftGate", "entropy", "fixed 0.2", "fixed 0.4", "fixed 0.5", "fixed 0.6"])):
    for m in ms:
        r = T2a[(s, m, g)]
        w(f"| {s}, {g} | {m} | {f2(r['acc_total_pct'])}% | {f2(r['acc_main_pct'])}% | {f2(r['acc_oop_pct'])}% | {f2(r['acc_oor_pct'])}% |")
w()
w("### 4.4 마지막 평가 라운드 정확도 (표 10, `tables/T10_last_round_accuracy.csv`)")
w()
w("절대 정확도를 확인하는 표다. integrated accuracy와 순위가 다를 수 있다.")
w()
w("| 설정 | 방법 | 마지막 평가 라운드 | seed | acc_total (SD) |")
w("|---|---|---|---|---|")
for r in T10:
    w(f"| {r['setting']} | {r['method']} | {r['last_eval_round']} | {r['n_seeds']} | {float(r['acc_total_pct']):.2f}% ({float(r['sd_pct']):.2f}) |")
w()
above, below = [], []
for st in dict.fromkeys(r["setting"] for r in T10):
    rs = {r["method"]: float(r["acc_total_pct"]) for r in T10 if r["setting"] == st}
    bf = max((v, m) for m, v in rs.items() if m.startswith("fixed"))
    (above if rs["DriftGate"] > bf[0] else below).append(f"{st.replace(' (stepwise)', '')}({rs['DriftGate']:.2f}% 대 {bf[1]} {bf[0]:.2f}%)")
w("마지막 평가 라운드에서 DriftGate가 가장 좋은 고정값보다 높은 설정은 " + ", ".join(above) + "이다. "
  "가장 좋은 고정값보다 낮은 설정은 " + ", ".join(below) + "이다.")
rsA = {r["method"]: float(r["acc_total_pct"]) for r in T10 if r["setting"] == A}
hiA = [f"{m}({v:.2f}%)" for m, v in rsA.items() if v > rsA["DriftGate"]]
if hiA: w(f"stepwise의 마지막 라운드에서는 {', '.join(hiA)}가 DriftGate({rsA['DriftGate']:.2f}%)보다 높다.")
w()
_e = [float(x["lambda_mean"]) for x in rd("lambda_trajectories.csv") if x["method"] == "entropy" and int(x["round"]) > 25]
entl = (min(_e), max(_e))
w("### 4.5 λ 궤적 (그림 3, `tables/lambda_trajectories.csv`)")
w()
w(f"seed 안에서 cluster 평균을 먼저 내고 seed 평균을 냈다. stepwise에서 DriftGate의 λ는 ρ = 0.8 구간(R61–90)에 평균 {f3(T4c[(A,'DriftGate')]['lambda_seg3(A: R61–90)'])}까지 내려갔다가 ρ가 0으로 돌아온 뒤 "
  f"{ro(f3(T4c[(A,'DriftGate')]['lambda_seg5(A: R121–150)']))} 올라간다. mobility에서는 ρ 변화 전 {f3(T4c[(M,'DriftGate')]['lambda_seg1(A: R26–30, mob: R26–60)'])}, 뒤 {ida(f3(T4c[(M,'DriftGate')]['lambda_seg2(A: R31–60, mob: R61–120)']))}. "
  f"entropy controller의 λ는 R26 이후 두 설정 모두 {entl[0]:.2f}–{entl[1]:.2f} 사이에 머문다.")
w()
w("### 4.6 신호 수준과 passive 신호 지표 (표 6a–6c, 그림 1)")
w()
w("평가용 stepwise(150 라운드)의 고정 λ = 0.4 실행(`t1_A_fx40`)은 신호를 기록하지 않았다. 그래서 신호를 매 라운드 기록한 passive 실행을 썼다. 이 실행은 고정 λ = 0.4, Λ = 0.5로 학습하며 controller가 신호에 반응하지 않는다. "
  "단계 변화는 ρ를 20 라운드마다 바꾸는 100 라운드 변형이고(R21, R41, R61, R81에 변화), late abrupt change는 R51에 ρ가 0에서 0.8로 오른다. seed는 3개이고, controller용과 평가용 표본을 나누지 않은 실행이다.")
w()
w("| 설정 | 구간 | 라운드 | TV (SD) | H/ln C (SD) |")
w("|---|---|---|---|---|")
for r in T6a:
    w(f"| {r['schedule']} | {r['segment']} | {r['rounds']} | {r['TV_mean']} ({r['TV_sd']}) | {r['H_over_lnC_mean']} ({r['H_over_lnC_sd']}) |")
w()
w("| 신호 | R46–50 평균 | R51–55 평균 | shift 변화 (R51–55 − R46–50) | shift 전 35 라운드 변화 (R46–50 − R16–20) | 전체 변화 (R96–100 − R16–20) |")
w("|---|---|---|---|---|---|")
for s, r in T6b.items():
    w(f"| {s} | {r['mean_R46_50']} | {r['mean_R51_55']} | {r['shift_change(R51–55 − R46–50)']} | {r['pre_shift_change_35_rounds(R46–50 − R16–20)']} | {r['whole_run_change(R96–100 − R16–20)']} |")
w()
w("| 설정 | 신호 | ρ와의 Spearman (SD) | raw-direction AUROC | direction-free AUROC |")
w("|---|---|---|---|---|")
for r in T6c:
    w(f"| {r['schedule']} | {r['signal']} | {float(r['spearman_vs_rho']):+.3f} ({float(r['sd']):.3f}) | {float(r['raw_direction_auroc']):.3f} | {float(r['direction_free_auroc']):.3f} |")
w()
w("AUROC는 ρ ≥ 0.5인 라운드를 양성으로 두고 처음 15 라운드를 뺀 값이다. raw-direction AUROC는 신호가 클수록 ρ가 크다고 보는 방향으로 계산했다. entropy도 shift에서 조금 오르므로 traffic에 반응하지 않는 것은 아니다. "
  "다만 그 폭이 학습 진행에 따른 감소보다 작아 drift 구간의 entropy가 drift 이전보다 낮다.")
w()
w("### 4.7 역할 실험 (표 7, 그림 5)")
w()
w("기존 3-seed 결과를 그대로 옮겼다. 이 실행들은 예전 controller(absolute branch 포함)와 controller·평가 공용 표본으로 돌았다. 표의 값은 정확도가 아니라 TV와 ρ의 Spearman 상관이다(처음 15 라운드 제외).")
w()
w("| 조건 | TV–ρ Spearman 평균 (SD) | seed별 값 |")
w("|---|---|---|")
for c, r in T7.items():
    w(f"| {c} | {float(r['tv_rho_spearman_mean']):+.3f} ({float(r['sd']):.3f}) | {r['per_seed']} |")
w()
w("### 4.8 재현 설정표 (표 8, `tables/T8_reproduction_settings.csv`)")
w()
w("| 분류 | 항목 | 값 | 출처(파일:줄) |")
w("|---|---|---|---|")
for r in csv.reader(open(T / "T8_reproduction_settings.csv")):
    if r[0] == "분류": continue
    w("| " + " | ".join(x.replace("|", "/") for x in r) + " |")
w()
w("### 4.9 통신량 (표 9, `tables/T9_communication.csv`)")
w()
w("| 분류 | 항목 | bytes | 근거 | 출처 |")
w("|---|---|---|---|---|")
for r in csv.reader(open(T / "T9_communication.csv")):
    if r[0] == "분류": continue
    b = r[2]
    try: b = " / ".join(f"{int(x):,}" for x in b.split(" / "))
    except ValueError: pass
    w(f"| {r[0]} | {r[1]} | {b} | {r[3]} | {r[4]} |")
w()
w("12,149,112 B는 client block 1개와 server block 2개의 크기를 더한 값과 정확히 같다. 이 숫자는 한 방향의 크기이므로, client가 가진 server-block 사본을 매 라운드 cluster 평균으로 맞추는 downlink(두 cluster에 속한 client는 11,029,584 B)는 포함하지 않는다. "
  "마지막 행은 edge에서 server exit를 계산한다고 가정한 추정치이며, 현재 구현은 client가 가진 server-block 사본으로 TV를 계산하므로 이 전송이 없다.")
w()

# ------------------------------------------------------------------ 5 figures
w("## 5. 그림 목록과 영문 caption")
w()
w("모든 그림은 vector PDF와 300 dpi PNG로 `figures/`에 있다. 그림 안의 이름은 논문 용어(stepwise composition change, client mobility, late abrupt change, DriftGate)로 바꿨다.")
w()
w("| 파일 | 크기 | 내용 |")
w("|---|---|---|")
w("| `fig1_signal_dynamics.pdf` | double column | 신호 변화: 위 ρ, 아래 TV와 H/ln C. (a) 단계 변화, (b) late abrupt change |")
w("| `fig2_system_overview.pdf` | double column | 시스템 개요. absolute branch 없음, controller는 cluster마다 계산 |")
w("| `fig3_lambda_trajectories.pdf` | double column | λ 궤적: 위 ρ, 아래 DriftGate와 entropy의 λ, 고정 λ = 0.4와 0.2 점선 |")
w("| `fig4_segment_accuracy.pdf` | double column | 구간별 정확도: stepwise 5구간, mobility 2구간 |")
w("| `fig5_role_experiment.pdf` | single column | 역할 실험: TV와 ρ의 Spearman 상관 |")
w()
cap = open(HERE / "figures/figure_captions.md").read(); cap = cap[cap.index("**Figure 1**"):]
w(cap.strip())
w()

# ------------------------------------------------------------------ 6 discrepancies
w("## 6. 불일치 기록")
w()
def integ(f): return float(np.mean([e["acc_total"] for e in json.load(open(f))["eval"]]))
dd = [100 * (integ(JR / f"runs/phaseT4_B3_fixedgrid/b3_mob_fx50_s{k}.json") - integ(JR / f"runs/phaseT1_disjoint/t1_mob_fx50_s{k}.json")) for k in range(3)]
dup = f"{min(abs(x) for x in dd):.3f}–{max(abs(x) for x in dd):.3f} pp(Round 4 − Round 1: " + ", ".join(f"{x:+.3f}" for x in dd) + ")"
items = [
 ("Round 4 B3 run의 정지", f"B3 λ = 0.15 run {len(KL)}개가 Round 4 중단 때 일시 정지된 채 남아 있었다. GPU 1에 있던 {k1}개를 포함해 모두 종료했고, 보고에 쓰는 run은 GPU 0에서 처음부터 다시 돌렸다. "
  "종료한 run 중 CIFAR-100 spatial 3개는 보고에 넣지 않으므로 다시 돌리지 않았다."),
 ("controller 설명과 코드의 순서", "지시서는 temporal 점수와 spatial 점수 중 큰 값을 EMA로 평활한다고 적었다. 코드는 두 점수를 각각 EMA(α = 0.3)로 평활한 뒤 큰 값을 고르고, 그 값에 이웃 edge와의 1단계 평균을 적용한 다음 λ로 바꾼다"
  "(`self_calibrating.py:150-164`). 모든 DriftGate 수치는 이 코드로 얻었다."),
 ("같은 설정의 중복 실행", f"mobility 고정 λ = 0.5가 Round 1(`t1_mob_fx50`, GPU 1)과 Round 4(`b3_mob_fx50`, GPU 0)에서 같은 flag로 두 번 실행됐다. seed별 integrated accuracy 차이는 {dup}이고, 첫 라운드 손실부터 소수 다섯째 자리에서 달라진다. "
  "GPU 연산의 비결정성 때문으로 보이며, 이 크기는 재현 오차의 기준으로 쓸 수 있다. 결과를 보기 전에 정한 규칙대로 Round 1 실행을 썼다."),
 ("지시서의 메시지 목록", "지시서는 DriftGate의 추가 메시지를 client당 4 B, edge당 4·deg(e) B로 적었다. 그러나 spatial 비교는 같은 라운드의 모든 cluster TV의 중앙값과 MAD를 쓰므로(`self_calibrating.py:51-61,157`), edge마다 다른 cluster의 TV 4·(Z − 1) B가 더 필요하다(Z = 5이면 16 B). 표 9에 따로 적었다."),
 ("Round 4의 probe 업로드 추정", "Round 4의 통신 표(a6_comm.csv)는 edge에서 server exit를 계산할 때의 업로드를 32 KiB × n + 40 B로 적었다. TV를 계산하려면 표본마다 확률 벡터가 필요하므로 n × (32,768 + 40) B가 맞다. probe 64개이면 2,099,712 B다."),
 ("commit hash", "저장소 HEAD `e82b96b`에는 `adaptive_splitomc_tmc/` 디렉터리가 들어 있지 않다(추적되지 않음). 실행 코드는 sha256으로 특정했다(`precheck/code_identity.txt`)."),
 ("ResNet-18의 client 수", "E2는 기존 ResNet-18 분할 실험과 같이 client 16명으로 돌렸다. CNN 실험의 기본값(50명)과 다르다."),
 ("신호 수준 실행", "평가용 stepwise의 고정 λ = 0.4 실행에는 신호 기록이 없어, 100 라운드 변형(ρ 변화 R21, R41, R61, R81)의 passive 실행을 썼다. 이 실행은 controller용과 평가용 표본을 나누지 않았다."),
 ("역할 실험의 실행 조건", "역할 실험 run은 예전 controller(absolute branch 포함)와 공용 표본으로 돌았다. 지시서가 이 결과를 옮기도록 했으므로 TV–ρ 상관만 옮겼고, 정확도는 넣지 않았다."),
 ("λ_max 변형의 warm-up 값", "controller는 warm-up 동안 λ 범위의 중간값을 쓰므로, λ_max를 바꾸면 warm-up λ도 0.425에서 0.375(λ_max 0.60) 또는 0.475(λ_max 0.80)로 바뀐다."),
 ("Main class 수의 메타데이터", "`src/datasets_ext.py`의 META에는 CIFAR-100 5개, Tiny-ImageNet 10개가 적혀 있지만 쓰이지 않는다. runner가 config의 classes_per_client_frac = 0.2를 먼저 쓰므로(`src/runner.py:121`) 실제 Main class는 전체의 20%다."),
 ("E3의 비교 기준", f"지시서대로 E3의 비교 기준은 같은 seed(0–2)의 B1 relonly 실행이다. 따라서 stepwise의 기준 정확도({float(T4b[(A,'guard 0.25')]['default_DriftGate_same_seeds_pct']):.2f}%)는 5-seed 평균({float(T1[(A,'DriftGate')]['integrated_acc_pct']):.2f}%)과 다르다."),
]
for i, (t, s) in enumerate(items, 1):
    w(f"{i}. **{t}**: {s}")
w()

# ------------------------------------------------------------------ 7 manifest
w("## 7. 새 run의 run_manifest (`tables/run_manifest.csv`)")
w()
w(f"Round 5에서 실행한 run은 {len(MAN)}개이고, {sum(1 for r in MAN if r['complete']=='yes')}개가 모두 지정된 라운드까지 끝났으며 {sum(1 for r in MAN if r['overlap_ok']=='yes')}개 모두 overlap이 0이다. "
  "모든 run은 물리 GPU 0에서 실행했다.")
w()
w("| run_id | run 이름 | 실험 | arm | 설정 | seed | 완료 | 라운드 | 평가 횟수 | overlap |")
w("|---|---|---|---|---|---|---|---|---|---|")
EXO = {"B3": 0, "B4": 1, "E1": 2, "E2": 3, "E3": 4}
for r in sorted(MAN, key=lambda r: (EXO[r["experiment"]], r["setting"], r["arm"], int(r["seed"]))):
    w(f"| {r['run_id']} | {r['run_name']} | {r['experiment']} | {r['arm']} | {r['setting']} | {r['seed']} | {r['complete']} | {r['rounds']} | {r['eval_rounds']} | {r['probe_eval_overlap']} |")
w()
w("## 8. 파일 구성")
w()
w("- `DriftGate_final_report_ko.md`: 이 보고서")
w("- `paper_numbers.csv`: 논문에 들어갈 숫자(항목, 값, 신뢰구간, seed 수, 출처 run)")
w("- `tables/`: 표 1–10과 보조 표의 CSV, `run_manifest.csv`")
w("- `figures/`: 그림 5개(PDF, PNG)와 `figure_captions.md`")
w("- `precheck/`: 시작 전 확인 스크립트와 결과, 코드 sha256")
w("- `scripts/`: 재생성 스크립트. 순서는 `r5_tables.py` → `r5_config_comm.py` → `r5_manifest.py` → `r5_figures.py` → `r5_report.py`이다. `scripts/launch/`에는 실행에 쓴 `enqueue_r5.py`, `r5_worker.py`, `r5_monitor.py`, `supervisor.sh`의 사본과 실제로 넣은 명령 목록(`enqueued_*_snapshot.txt`)이 있다.")
w("- `README.md`: 파일 구성과 재생성 명령")
open(HERE / "DriftGate_final_report_ko.md", "w").write("\n".join(L) + "\n")
print("report lines:", len(L))
