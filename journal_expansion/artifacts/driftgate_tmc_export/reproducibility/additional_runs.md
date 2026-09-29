# Additional runs — 이번 export에서 새로 실행한 것

원칙: 기존 결과로 답할 수 없는 근거 공백에만 새 run을 돌린다. 조건과 seed는 **결과 확인 전에 고정**했다. 기존 실행
작업을 종료하거나 GPU를 독점하지 않았다(파일-큐 워커에 추가 enqueue, GPU idle 활용).

## AR1. Matched fixed reference (disjoint) — 4개 전이 설정

- **질문:** F6/설정별 표에서 CIFAR-10 gradual·CIFAR-100 gradual/spatial·Tiny-ImageNet의 **disjoint** matched fixed
  reference가 비어 있었다. 기존에는 이들 설정의 fixed run이 same-pool만 존재하고, checkpoint가 저장돼 있지 않아
  (`find runs -name '*.pt' → 0`) evaluation-only 재평가가 불가능했다.
- **기존 자료로 해결되지 않은 이유:** disjoint fixed run 자체가 없음 + checkpoint 없음 → 재학습 필요.
- **결과 확인 전 고정한 조건:** fixed λ∈{0.2, 0.4}, Λ=0.5(사전 정의 후보), `--disjoint_pools`, seeds 0–2,
  각 설정의 기존 adaptive와 동일한 schedule/rounds/probe(C10gsig·C100gsig 150R, C100sp static+equal_spread 150R,
  Tiny 100R). 총 4×2×3 = **24 run**. 출력 `runs/phaseT3_fixedref/`.
- **상태:** **완료(24/24).** `scripts/fill_fixed_refs.py`가 `data/fixed_refs_transfer.csv`를 생성했고 numbers doc
  §6 표 6-B에 반영. 결과(best fixed = λ0.2, disjoint 3 seeds): C10 gradual 62.86, C100 gradual 35.91, C100 spatial
  36.44, Tiny 24.31. **핵심:** class가 많은 CIFAR-100/Tiny에서 direct rate는 best fixed보다 낮고(−0.50/−1.44/−0.46),
  adaptive TV는 fixed와 대등 이상(spatial ≈0, gradual +0.19). 다른 protocol 값으로 빈칸을 채우지 않았다.

## 새로 실행하지 않은 것(기존 자료로 해결 또는 범위 밖)

- **§6.2 absolute-only 정면 비교:** `λ=λ_abs` 전용 controller arm이 코드에 미배선. 새 mode를 만드는 것은 이 수치·그림
  작업의 범위를 넘어(method-인접 코드 변경) 실행하지 않았다. 기존 same-pool의 dual vs relative-only(개발 단계, 신호도
  다름) 비교만 표 5-D로 제공하고, disjoint 정면 비교는 남은 작업으로 명시(numbers doc §9-2).
- **§6.3 closed-loop 사전학습 비교:** 사전학습 checkpoint 미저장 + 전용 continual 설정 부재로, downstream 정확도
  비교를 새로 돌리지 않았다. 기존 post-convergence(신호 품질 위주, 표 5-A abrupt)만 제공. 남은 작업으로 명시(§9-3).
- **모바일 latency/energy:** 장비 실험 필요 → 범위 밖. payload는 tensor 기반 estimate(§7)로만 제시.
- **Role/absolute의 seed 확장:** role 진단은 server-only 신호가 기본 signal이 아니므로(TV_PRIMARY) 확장하지 않음.

## Provenance
새 run: commit `e82b96b`, config `configs/base_v3.yaml`, run_id `fixed_*`(각 JSON `config.run_id`), seed=파일명,
model_seed=100+seed. 완료 목록은 `runs/phaseT3_fixedref/`와 갱신될 `table_settings.csv`에 기록.
