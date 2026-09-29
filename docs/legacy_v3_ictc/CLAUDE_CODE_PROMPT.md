# Claude Code Starter Prompt

Copy and paste the entire block below into a fresh Claude Code session to kick off the autonomous execution.

---

## Prompt to send to Claude Code (Korean):

```
Adaptive-SplitOMC v3 자율 실행을 시작해.

지시사항:
1. `CLAUDE.md`와 `EXECUTE_ALL.md`를 읽어서 전체 파이프라인 구조를 이해해.
2. 그 다음 `bash run_all.sh`를 tmux 세션에서 실행해. 백그라운드로 돌아가면서 진행상황을 주기적으로 확인해줘.
3. 각 phase 끝날 때마다:
   - 단위 테스트 통과: 다음 phase로 진행
   - E0 게이트 통과: 다음 phase로 진행
   - E2 (CORE) 결과 검토: adaptive integrated이 best fixed보다 2pp 이상 이기는지 확인
   - 만약 어느 phase든 실패하면 STOP하고 나(사용자)에게 보고해.
4. 모든 실험 끝나면:
   - `docs/results_v3/paper_v3_filled.md`를 읽어서 결과 잘 정리됐는지 검증
   - `docs/results_v3/interpretation.md` 읽어서 narrative 적절한지 검증
   - 핵심 결과 3-5줄로 요약해서 나에게 알려줘
5. 만약 결과가 paper에서 원하는 방향과 다르면 (e.g., adaptive가 fixed보다 안 좋음), 가능한 원인을 분석하고 fix 시도해. fix 후 해당 phase 재실행. 최대 2번까지 시도.

하드웨어: 2× RTX 3090, cuda:0과 cuda:1 사용.
예상 wall-time: 30-40 시간.

진행 상황은 tmux 세션 `adaptive_v3`에서 확인. 30분마다 nvidia-smi와 logs/master.log 마지막 100줄을 확인해서 정상 진행 중인지 체크.

핵심 결과 파일:
- docs/results_v3/paper_v3_filled.md  (paper 초안, 실제 숫자 포함)
- docs/results_v3/interpretation.md   (auto narrative)
- docs/results_v3/tables/*.md         (표)
- docs/results_v3/figures/*.png       (그림)

문제 발생 시 즉시 멈추고 logs/master.log 마지막 부분과 함께 보고할 것.
```

## Prompt to send to Claude Code (English):

```
Start Adaptive-SplitOMC v3 autonomous execution.

Instructions:
1. There should be `adaptive_splitomc_v3.zip` in the current directory. Unzip and cd into it.
2. Read `CLAUDE.md` and `EXECUTE_ALL.md` to understand the full pipeline.
3. Then run `bash run_all.sh` inside a tmux session named `adaptive_v3`. Monitor progress.
4. At each phase boundary:
   - Unit tests pass: continue
   - E0 gate passes: continue
   - E2 CORE: verify adaptive integrated beats best fixed by ≥ 2 pp
   - If any phase fails: STOP and report to me
5. After all experiments complete:
   - Read docs/results_v3/paper_v3_filled.md to verify it's well-formed
   - Read docs/results_v3/interpretation.md to verify narrative
   - Summarize key results in 3-5 lines for me
6. If results contradict the paper hypothesis (e.g., adaptive < fixed), analyze possible causes and try to fix (up to 2 attempts), then re-run that phase.

Hardware: 2× RTX 3090, devices cuda:0 and cuda:1.
Estimated wall-time: 30-40 hours.

Check progress every 30 minutes via nvidia-smi and `tail -100 logs/master.log`.

Final artifacts:
- docs/results_v3/paper_v3_filled.md  (paper draft with real numbers)
- docs/results_v3/interpretation.md   (auto narrative)
- docs/results_v3/tables/*.md         (paper-ready tables)
- docs/results_v3/figures/*.png       (paper-ready figures)

On any problem, halt and report logs/master.log tail.
```

---

## Manual execution alternative

If user prefers to run manually instead of via Claude Code:

```bash
# Step 1: extract
unzip adaptive_splitomc_v3.zip
cd adaptive_splitomc_v3

# Step 2: verify environment
python -m pytest tests/test_all.py -v
# Should show: 12 passed

# Step 3: launch in tmux
tmux new-session -d -s adaptive_v3 'bash run_all.sh 2>&1 | tee logs/master.log'

# Step 4: monitor
tmux attach -t adaptive_v3
# Ctrl+B then D to detach

# Step 5: check progress later
tail -50 logs/master.log
ls -la results/*/
```

## Expected timeline

| Phase | Approx duration | What's happening |
|---|---|---|
| Pre-flight | 1 min | Unit tests |
| Phase 0 | 3 h | E0 smoke (4 configs, 50R each) |
| Phase 1 | 10-13 h | E1 (16 configs, 150R each, parallel across 2 GPUs) |
| Phase 2 | 7 h | E2 temporal Schedule A + B |
| Phase 3 | 10-12 h | E3 + E4 + E5 (chained) |
| Phase 4 | 5 min | Analysis + paper assembly |
| **Total** | **~32 h** | |

## Final deliverables location

After completion, deliverables live at:
- `docs/results_v3/paper_v3_filled.md` ← **start here**
- `docs/results_v3/interpretation.md`
- `docs/results_v3/tables/`
- `docs/results_v3/figures/`
- `results/` (raw JSON files, for re-analysis)
- `logs/` (run logs, for debugging)
