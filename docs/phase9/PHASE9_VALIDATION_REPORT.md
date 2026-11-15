# Phase 9 — Validation report

**Acceptance status: `ACCEPTED_WITH_LIMITATIONS`.** Engineering integration of Phases 4–8 on precomputed Phase 3 artifacts passes; raw-video execution and independent scientific evaluation are unavailable.
Execution mode of every recorded run: **`PRECOMPUTED_PHASE3_ARTIFACTS`** (Phase 3 outputs from the Colab run; no raw video, OpenCV, Ultralytics, `psutil` or GPU locally; none installed). This is not raw-video end-to-end
inference. Passing tests demonstrate software correctness under tested conditions; they do not establish scientific validity, generalisation, calibration, real-time operation or deployment readiness.
Branch `feature/phase9-end-to-end-validation`, base commit `cee4575` (Phase 8), `209b738` (Phase 7).

The results below are kept in separate classes and are never combined into one score.

## 1. Unit-test validation (synthetic fixtures)
`tests/phase9/test_contracts_and_gates.py`: Stage 3 contract (coordinate system, boxes, timestamps/variable frame rate, duplicates, ordering, truncated JSON), manifest hash/chain verification, history-only window and alignment gates, risk-record identity
and probability semantics, alert-output safety (no alert from invalid evidence), configuration validation, statistics / warm-up separation / stage clock / real-time rule, split-manifest validation (10 failure modes), independent-evaluation status levels,
partition isolation, metric-validity gates, and arithmetic of the trajectory and alert-level metrics on synthetic arrays (formula checks, not results).

## 2. Integration-test validation (actual PIE pilot artifacts)
`tests/phase9/test_pipeline_integration.py`, `tests/phase9/test_limitations_and_verification.py`: precomputed-artifact execution; reproduction of Phase 5 predictions (bitwise), Phase 7 records, Phase 8 events and alerts; window re-execution equal to the Phase 5 reference with
its provenance chain; deterministic repeat; source artifacts unchanged after repeated inference (30 input hashes before/after); end-to-end future-target scrambling changes no prediction, record or event; a sensitivity control shows predictions do change when history inputs
change; model/window provenance mismatch, truncated window artifact and corrupted checkpoint are rejected (`DATASET_PROVENANCE_MISMATCH`, `TRAIN_ARTIFACT_HASH`); missing Phase 3 input, corrupted Phase 3 rows, missing mid-pipeline reference and a stage without vehicle context fail clearly and stop dependants
(`NOT_RUN`) without producing alerts; missing frames, long gaps, all-insufficient evidence, track identity change, out-of-order timestamps at the alert boundary; CLI exit codes; execution-mode statement; unavailable timings are `null`, never 0; no physical-quantity keys; no generalisation claims.

## 3. Test commands and results (this machine, `pcg-baseline`, Python 3.9.25)
| Command | Exit | Result |
|---|---|---|
| `python -m pytest tests -q -p no:cacheprovider` (all suites: phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, task7) | 1 | **1381 passed, 1 failed, 8 skipped**, 421.5 s |
| `python -m pytest tests/phase9 -q -p no:cacheprovider` (before the last small edits; re-run after them, see §9) | 0 | 66 passed, 178 s |
| Full regression excluding Phase 9, first pass | 1 | 1313 passed, 3 failed, 8 skipped — two failures were NEW and caused by Phase 9 (direct `import torch` in `benchmark.py` / `orchestrator.py` violates the Stage 6.1 dependency guard); fixed by reaching torch through the guarded Phase 5 module, the guard tests were not touched |
**The single remaining failure** is `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (`UnboundLocalError: TTC_mask`, `utils/Interaction.py:173`). Verified pre-existing, not assumed: it fails identically on a detached checkout of Phase 5
commit `afdf822` (before Phases 6–9), and `utils/` and `tests/phase2/` have a zero-line diff between `afdf822` and the current HEAD. It is legacy baseline code outside Phases 3–9 and was not modified. The 8 skips are pre-existing (Phase 3/4/5).
Full console logs: `experiments/phase9_end_to_end_pie_video_0004/final_regression_test_log.txt`, `full_run_console.log`.

## 4. Full pipeline execution (engineering benchmark) — run `pie_video_0004_full`, exit code 0
Stages S3 (artifact validation), S4A, S4B, S5A, S5B, S6, S7, S8, S9 all `COMPLETED`; the re-executed Phase 4/5 data files are sha256-equal to the committed references (`causal_kinematics.jsonl`, 8 interaction NPZ files (7 edge relations + nodes), `windows.npz`, `graph_bank.npz`,
`rejected_candidates.npz`). Details and timings: `PHASE9_PERFORMANCE_REPORT.md`; manifest `experiments/phase9_end_to_end_pie_video_0004/run_manifest.json`; 34 output hashes and per-stage manifests in `verification_report.json`.

## 5. Comparison with the committed Phase 5 / 7 / 8 artifacts (exact equality)
| Artifact | Result |
|---|---|
| Phase 5 `predictions_neural.npz` | sha256 equal; all 11 arrays bitwise equal (single thread, CPU) |
| Phase 5 windows / graph bank / rejected candidates (re-executed) | all three data files sha256- and array-equal |
| Phase 7 records (8,233) | exactly equal as lists: pedestrian and vehicle ids, anchor frames, window ids, risk scores, categories, category status, masks, reasons, null probabilities |
| Phase 7 `risk_arrays.npz` | risk score, category, anchor, window id equal to the records; probabilities all null / `NOT_CALIBRATED` |
| Phase 8 events (1,200), alerts (28) | event ids, alert ids, (pedestrian, vehicle, frame), state transitions and ordering equal; every field except `provenance` equal; alert table exactly equal |
The counts 28 alerts / 1,200 events are read from the outputs and compared with the committed Phase 8 summary at test time; they are not hard-coded in the runner.

## 6. Every difference found
| # | Field / artifact | Difference | Explanation | Affects scientific validity? |
|---|---|---|---|---|
| D1 | Phase 8 events, `provenance.phase9_stage` | present in Phase 9 only (`"S8"`) | Phase 9 labels the producing stage | No, metadata |
| D2 | Phase 8 events, `provenance.phase9_execution_mode` | present in Phase 9 only (`PRECOMPUTED_PHASE3_ARTIFACTS`) | records the execution mode in each event | No, metadata |
| D3 | Phase 8 events, `provenance.phase5_predictions_sha256` | present in Phase 9 only | hash of the S5B predictions file that fed the run (equal to the reference file hash) | No, metadata |
| D4 | Recorded full run (`pie_video_0004_full`) events, 10 provenance keys (`coordinate_system`, `phase5_run_manifest_sha256`, `phase6_config_sha256`, `phase7_calibration_status`, `phase7_category_status`, `phase7_risk_report_sha256`, `phase7_schema`, `phase7_threshold_evidence`, `sample_status`, `phase9_stage`) | missing / different in that run | that run used the code revision before the provenance enrichment (this gap was found by the exact comparison and fixed); the inference replay run, produced after the fix, differs only in D1–D3 | No, metadata |
| D5 | Recorded full run manifest | lacks `execution_mode`, input-hash and CPU/RAM fields | added to the code afterwards; derived post hoc by `attribute` / `verify-run` without re-measurement (`performance_attribution.json`, `verification_report.json`) and demonstrated on the replay manifest | No, reporting only |
| D6 | Wall times | not reproducible across runs | normal timing variability | No |
No numerical array, identifier, ordering, risk value, mask, category or alert differs.

## 7. Determinism, isolation, safety checks (all passing)
Identical inputs and configuration give byte-identical predictions and identical event lists; inputs are hash-identical before and after repeated runs; future targets scrambled in the model's dataset and in the window dataset used by Phase 6 change nothing, while a history
perturbation does (so the isolation result is not vacuous); missing or invalid evidence stays masked and `INSUFFICIENT_EVIDENCE` (1,231 records, 473 transitions) never becomes a warning; track ids are never merged; coordinates stay `captured_1920x1080` pixels;
Phase 7 stays `NOT_CALIBRATED`; no TTC, PET, metric distance or severity value exists anywhere.

## 8. Scientific status (not validated)
In-sample pilot results: Phase 5 neural ADE 31.5 px vs B2 35.7, B0 54.3, B1 93.0 (in-sample), Phase 7 category counts, 28 Phase 8 alerts (25 closed by evidence loss, 3 by risk subsiding). Independent scientific evaluation: **`BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES`**
(one scene group, no annotations, no frozen split, no verified events). Blocked: independent trajectory metrics, conflict metrics, calibration (Brier/ECE/NLL), alert precision / recall / false alerts per hour / lead time, perception quality, raw-video execution and FPS.
A structural limitation remains: windows exist only where the target also has future samples, so the system is an offline replay.

## 9. Final re-run after the last edits
After the final edits (output inventory in `verify-run`, failure-register entries) `python -m pytest tests/phase9 tests/task7 -q -p no:cacheprovider` gave **652 passed, 0 failed, exit 0** (148.8 s): the 66 Phase 9 tests and the 586 `task7` tests
(including the Stage 6.1 dependency guards that the first Phase 9 version had broken). Log: `experiments/phase9_end_to_end_pie_video_0004/final_phase9_task7_rerun_log.txt`. The full-suite figure in §3 (1381 passed) was taken before these last edits, which touched no
other suite; the only failure then and now is the pre-existing Phase 2 test.
