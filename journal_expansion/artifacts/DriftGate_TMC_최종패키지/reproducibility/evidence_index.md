# Evidence index — DriftGate TMC export

Commit `e82b96b` · config `configs/base_v3.yaml`. 경로는 저장소 루트(`/disk2/Yujin/adaptive_splitomc_tmc/`) 기준.
모든 export 수치는 `artifacts/driftgate_tmc_export/scripts/build_data.py`가 원시 run JSON에서 재생성한다.

## A. 코드 확인 (§2 평가 정의)

| 확인 항목 | 파일:함수/줄 | 결론 |
|---|---|---|
| 최종 예측 = entropy-routed dual-exit | `eval/evaluator.py::evaluate_one_client` (`route_to_server = entropy>eth`; `final = where(route, server_preds, client_preds)`) | label 미사용 routing |
| routing threshold eth=0.8 | `configs/base_v3.yaml: eth_default`; `src/runner.py:~552` eval 호출 | 전 방법 동일 |
| multi-server 결합 | `eval/evaluator.py::evaluate_one_client` (`server_logits = sum(list)/len`, then argmax) | logit 평균 |
| entropy 정의 | `eval/evaluator.py:18 compute_entropy` (`-(p*log_p).sum`, nats) | per-sample Shannon, [0,ln C] |
| acc_total 집계 | `eval/evaluator.py::evaluate_all_clients` (`mean over per-client`; per-cell by `primary_es`) | client 평균, 중첩 1회 |
| Main/OOP/OOR 사용처 | `evaluate_one_client` (true label로 counts[t] 분해만; routing엔 미사용) | 보고용 stratification |
| 라운드 순서·eval 시점 | `src/runner.py` 주석 `# -- 1..5` (329 traffic, 366 signal on end-of-(r-1), 435 controller, 458 train+agg, 537 eval) | causal, eval는 갱신 후 |
| main_classes 생성 | `src/runner.py:157 nd1_partition(...)`; `data/partition.py:50 nd1_partition`, `:82 rng.sample(scope)` (train scope) | training split만 |
| probe no-grad/eval | `src/signals/library.py:31 probe_forward` (`@torch.no_grad`, `.eval()`) | BN/state 불변 |
| TV/entropy 신호 | `src/signals/library.py:80 tv_dist`, `:60 ent_client` | TV[0,1], ent mean H(p) |
| server non-Main | `src/signals/library.py:174 server_nonmain_signals` | rate/soft mass, [0,1] |
| controller 매핑·상수 | `src/controllers/self_calibrating.py:12-14,32-33,45,174-181 (Z0=1.5,TAU_Z=0.75,α=0.3, λ=min(λ_rel,λ_abs))`; `src/controllers/normalizers.py:29-36,49-54 (β=0.05, guard, floor)` | 문서와 일치 |
| disjoint pool | `src/disjoint_pools.py::make_pool_masks` (class별 20/80, seed 고정); `src/runner.py` 끝 `assert overlap==0` | overlap=0 |
| communication | `tables/communication_accounting.csv`, `tables/overhead.csv` | activation 32 KiB, exchange 12,149,112 B/round, DriftGate +4 B scalar |

## B. 그림별 source run (원시 JSON)

| 그림 | data CSV | 원시 run (glob) | protocol · seeds |
|---|---|---|---|
| F1 | f1_signal_dynamics.csv | `runs/signal_benchmark/rec_A_s*`, `rec_abrupt_s*` | passive fixed-weight · 3 |
| F3 | f3_traj_{lambda,acc}.csv, f3_rho.csv | `runs/phaseT1_disjoint/t1_{A,mob}_{dg,ent,fx40}_s*` | disjoint · A 5 / mob 3 |
| F4a | f4_role.csv | `runs/phaseC_signals/dvsig_tv_A_s*`(standard), `runs/phaseR_role/role_{same_role,same_role_indep,weak_server}_s*` | same-pool · 3 |
| F4b | f4_absolute_contribution.csv | `runs/gated/d4_cifar100/d4_{dual,main}_{gsig,sp}_s*`, `runs/gated/d5_tinyimagenet/d5_{dual,main}_gsig_s*` | same-pool(dev) · 3 |
| F5a | f5_timing.csv | adaptive src `runs/gated/{d2_horizon/d2_dual_A, d1_unseen/d1_dual_gradual_sigmoid, d3_spatial/d3_dual_sp}_s*`; controls `runs/phaseB_decomp/{gmm,pcm,shuf}_{A,gsig,sp}_s*` | same-pool replay · 3 |
| F5b | f5_lambda.csv | `runs/phaseT1_disjoint/t1_{A,mob}_{dg,a1}_s*` | disjoint · A 5 / mob 3 |
| F6 | f6_server_signal.csv | `runs/phaseT1_disjoint/t1_*`, `runs/phaseT2_signal/t2_*`, `runs/phaseT2r2_signal/r2_*` (round-2 recompute) | disjoint · 3–5 |

## C. 표별 근거 (numbers doc)

| 표 | data/round2 CSV |
|---|---|
| 3-A primary | `data/primary_performance.csv` |
| 3-B paired | `data/primary_paired_diff.csv` |
| 4-A peak | `tables/final_closure_round2/high_nonmain_segment.csv` |
| 5-A/5-B passive | `tables/final_closure_round2/passive_signal_quality.csv` |
| 5-C role | `data/f4_role.csv` + `tables/final_closure_round2/server_role_nonmain_signal.csv` |
| 5-D absolute | `data/f4_absolute_contribution.csv` |
| 5-E timing | `data/f5_timing.csv` |
| 5-F Λ | `data/f5_lambda.csv` |
| 6-A server vs TV | `data/f6_server_signal.csv` (← `hard_vs_tv_all_settings.csv`) |
| metadata audit | `tables/final_closure_round2/main_class_metadata_audit.csv` |

## D. Metric 정의·집계

- integrated = mean(acc_total over eval rounds); eval rounds: A/SVHN/CIFAR-10·100 gsig/sp = {1,10,…,150}(16),
  mobility = {1,10,…,120}(13), Tiny = {1,10,…,100}(11) — 각 run config에서 읽음(hardcode 아님).
- paired difference: 동일 seed A−B → 차이들의 평균 + Student-t 95% CI; SD = seed 간 ddof=1.
- passive signal: `src/evaluation/signal_metrics.evaluate_signal`(warm-up 15, drift-active ρ≥0.5, raw-direction AUROC,
  direction-free = max(a,1−a)).
- abs-active fraction: warm-up+burn-in(25) 이후 cluster-round에서 `d̂ > sigmoid((z−z0)/τ)` 비율, SIGNAL_RANGE(tv/rate)=1.0.

## E. Provenance

각 run JSON의 `config.run_id`(`selfcal_YYYYMMDD_HHMMSS_hash` / `fixed_*`), seed = 파일명 `_s{n}`, model_seed = 100+seed.
신규 run manifest: `tables/final_closure_round2/run_manifest.csv`(round-2 35 run), `tables/final_closure/final_run_manifest.csv`(round-1).
