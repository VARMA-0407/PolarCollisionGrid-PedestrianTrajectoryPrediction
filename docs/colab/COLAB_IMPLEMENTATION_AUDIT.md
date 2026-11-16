# Colab implementation audit — master PIE notebook

Date: 16 November 2026 · Branch: `feature/colab-full-pipeline-notebook` (cut from `feature/phase9-end-to-end-validation` @ `b351d30`) ·
Deliverables: [`notebooks/PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb`](../../notebooks/PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb),
[`GOOGLE_COLAB_NOTEBOOK_GUIDE.md`](GOOGLE_COLAB_NOTEBOOK_GUIDE.md), this audit.

**Execution status:** structurally validated; orchestration smoke-tested locally on synthetic data in precomputed-artifact mode (CPU, Windows,
Python 3.9). **Not executed on Google Colab, on a GPU, with YOLO/ByteTrack, or on real PIE videos.** No result in this document is a PIE result.

## 1. Repository files inspected

* Project records: `PROJECT_MASTER.md` (including the Phase 3–9 addenda), `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md`.
* Documentation: `docs/phase2/*` (schemas, split protocol), `docs/phase3/*` (Task 1 perception, Task 4 PIE evaluator, Colab execution guide),
  `docs/phase4/*` (coordinate contract, calibration audit incl. the PIE `utilities/pie_data.py` record, Task 2/3), `docs/phase5/*` (architecture, input contract,
  P5.3 windows, P5.7.1–P5.7.6, independent-evaluation status), `docs/phase6/*`, `docs/phase7/*`, `docs/phase8/*`, `docs/phase9/*` (architecture, integration guide, validation).
* Source: `pedsafety/perception/{runner,config,adapter,video_source,pie_perception_evaluator}.py`, `pedsafety/image_kinematics_run.py`,
  `pedsafety/interaction/{runner,input_adapter}.py`, `pedsafety/phase5/{runner,upstream,window_storage,training_data,training_scaler}.py`,
  `pedsafety/phase5/models/{training_harness,training_step,checkpoint,neural_evaluation}.py`, `pedsafety/phase5/evaluation/metrics.py`,
  `pedsafety/phase6/{runner,pairs}.py`, `pedsafety/phase7/{risk,runner}.py`, `pedsafety/phase8/{runner,schema}.py`,
  `pedsafety/phase9/{orchestrator,contracts,evaluation,benchmark,reporting}.py`, `pedsafety/schema.py`.
* Scripts: `scripts/run_perception.py`, `scripts/evaluate_pie_perception.py`, `scripts/build_colab_bundle.py`, the Phase 4/5 run scripts.
* Configuration: `configs/phase3/perception_default.json`, `configs/phase4/*`, `configs/phase5/*`, `configs/phase6–8/*`, `configs/phase9/pipeline_pie_video_0004.json`.
* Dependencies: `requirements*.txt`; dependency policy test `tests/task7/test_stage61_dependencies_version.py` (torch only under `pedsafety/phase5/models`).
* Tests and helpers: `tests/phase4`, `tests/phase5` (conftest, synthetic scene), `tests/phase9`.
* Existing Colab notebook `notebooks/phase3_task1_perception_colab.ipynb`, pilot Phase 3 artifacts (`PIE_pilot_inputs/`, git-ignored) and the committed pilot experiment records under `experiments/`.

## 2. Implementation inventory (before this work)

| Stage | Existing implementation / entry point | Inputs → outputs | GPU | Real PIE evidence | Limitation found | Colab-ready for all videos? |
|---|---|---|---|---|---|---|
| Video ingestion + YOLO + ByteTrack (Phase 3) | `scripts/run_perception.py` → `perception.runner.execute_run` | MP4 → `tracked_agents.jsonl`, `detections.jsonl`, `agent_states.jsonl` (with `--scene-id`), `run_manifest.json`, optional `annotated.mp4` | yes (Ultralytics) | one video (`set01/video_0004`, Colab T4) | one video per call; output folder name is timestamped; sequence id = file stem (collides across PIE sets); agent-state adapter keeps all frame results of a video in memory | per video only → wrapped |
| Perception vs PIE annotations | `scripts/evaluate_pie_perception.py` → `pie_perception_evaluator.run_evaluation` | annotation XML + detections/tracks → reports | no | one video | per video | yes, per video |
| Phase 4 Task 2 | `image_kinematics_run.run_image_kinematics` | agent states → `causal_kinematics.jsonl` + manifest | no | pilot video | per video; outputs must be absent/empty | yes, per video |
| Phase 4 Task 3 | `interaction.runner.run_interaction_representation` | Task 2 run + agent states → graph `.npz` + manifest | no | pilot video | re-verifies upstream files at their recorded paths | yes, per video (same scratch folder) |
| P5.3 windows | `phase5.runner.generate_windows` (`video_id`, `group_id` arguments) | Task 3 run → `windows.npz`, `graph_bank.npz` | no | pilot video | `PILOT_SMOKE` pseudo-targets only; `ANNOTATED_TARGET` refused; window needs 15 future samples | yes, per video |
| Phase 5 training | `phase5.models.training_harness.run_training` | ONE window run → checkpoints, scaler, log | CUDA path exists, never run | pilot, in-sample | **accepts one window run; `split_policy` admits only the pilot policy** | **no** → extension needed |
| Phase 5 evaluation | `neural_evaluation.run_neural_evaluation` / `evaluate_neural` | training run + window run → predictions, metrics | yes | pilot, in-sample | **`load_trained_model` requires the exact training window run (manifest hash)** → cannot score other videos | **no** (loader) → extension; `evaluate_neural` reused |
| Phase 6 | `phase6.pairs.build_conflict_pairs`; `phase6.runner` | predictions + windows → pair batch, report | no | pilot | runner hard-codes in-sample status; batch has no persistent format | API yes |
| Phase 7 | `phase7.risk.assess_risk`; `phase7.runner` | pair batch → risk output | no | pilot | runner hard-codes the pilot dataset string; calibration blocked | API yes |
| Phase 8 | `phase8.runner.run_alerts` | Phase 7 records → events, alerts | no | pilot | — | API yes |
| Phase 9 | `python -m pedsafety.phase9 run / verify-run / evaluation-status / attribute`; `phase9.orchestrator.Pipeline` | one Phase 3 folder + pilot reference runs | no | pilot, precomputed mode | **single video, bound to the pilot reference runs; S7/S8 provenance hard-codes `set01/video_0004` and precomputed mode; does not launch YOLO** | no for multi-video; gates and CLI reused |

## 3. What was added (no earlier-phase file was modified)

| File | Purpose |
|---|---|
| `pedsafety/multivideo/common.py` | SHA-256, atomic JSON, per-stage completion markers, verified publication local scratch → Drive, quarantine (never delete), git state |
| `pedsafety/multivideo/experiment.py` | experiment id, configuration lock (refuses changed scientific settings / code commit on resume), persistent stage tracker |
| `pedsafety/multivideo/inventory.py` | PIE inventory: six sets, MP4 counts/sizes/SHA-256 (resumable cache), duplicates, annotation archives and XML, safe extraction |
| `pedsafety/multivideo/split.py` | official split read from `pie_data.py` by AST (not executed); frozen recording-level split in the Phase 9 `phase9-group-split-1.0.0` layout; integrity / leak checks; tamper-evident content hash |
| `pedsafety/multivideo/perception_batch.py` | Phase 3 per video through `scripts/run_perception.py` (subprocess per video, sequence id `<set>_<video>`), Phase 9 Phase-3 contract, optional PIE evaluation, resume, verified reuse, precomputed-artifact registration |
| `pedsafety/multivideo/preprocess.py` | Task 2 → Task 3 → P5.3 per video with the Phase 9 S4A/S4B/S5A checks and per-sub-stage resume |
| `pedsafety/multivideo/training.py` | multi-video training on the train partition (reuses `TrainingConfig`, `ForecastingModel`, optimizer/scheduler builders, `train_step`/`eval_step`, `fit_feature_statistics`, checkpoint I/O); hash-verified model loader for held-out videos |
| `pedsafety/multivideo/evaluation.py` | per-video `evaluate_neural` (unchanged metrics: primary most-probable mode, separate oracle), per-partition aggregation, future-isolation check, test-access log |
| `pedsafety/multivideo/downstream.py` | Phase 6 / 7 / 8 per video with the Phase 9 S6–S8 contracts, partition-aware sample status, `calibrated_probability` guard |
| `pedsafety/multivideo/report.py` | experiment report JSON/Markdown, schema validation, marker table, real-time statement (`phase9.benchmark.realtime_assessment`), final status |
| `scripts/build_colab_full_pipeline_notebook.py` | generates the notebook deterministically |
| `scripts/validate_colab_notebook.py` | structural validator (parse, imports, signatures, paths, execution order, switches) |
| `tests/multivideo/*` | unit, integration (real Phase 4–8 code on synthetic videos) and notebook smoke tests |

### Stages integrated in the notebook
All: environment/Drive/GPU (1), inventory (2), provenance/experiment (3), official split + frozen recording-level split (4), Phase 3 raw-video or
precomputed (5), Phase 4 (6), Phase 5 windows / training / evaluation (7), Phase 6 (8), Phase 7 (9), Phase 8 (10), Phase 9 status, gate CLI and optional
single-video pilot regression (11), report (12), tests and final validation (13).

### Not integrated / blocked, and why
| Item | Status | Reason |
|---|---|---|
| Trajectory evaluation against PIE annotation ground truth | BLOCKED | `ANNOTATED_TARGET` is refused by the window generator; no verified PIE-id ↔ track-id mapping or frame alignment exists. Not implemented here (it would be a new scientific method, not orchestration). |
| Phase 9 independent evaluation (`evaluate_*_independent`) | BLOCKED (gate called unchanged) | split manifest annotation provenance / identity mapping / timestamp alignment / licence are honestly `verified: false` |
| Phase 6 supervised training, Phase 7 calibration, category validation | BLOCKED | no verified conflict labels; the calibration partition (if chosen) has no labels |
| Phase 8 lead time / false alarms | BLOCKED | no ground-truth event timestamps |
| Metric geometry, TTC, PET | BLOCKED | image space only; no validated calibration / ego-motion compensation |
| Real-time claim | NOT SUPPORTED | offline batch replay; window builder needs future frames; stages timed separately |
| Original Phase 9 runner on the full dataset | not used for multi-video | single-video, pilot-bound design; reused for gates (`evaluation-status`) and as an optional pilot regression |

## 4. Deviations from the original implementation (recorded in every experiment report)

1. Multi-video training wrapper: `split_policy = RECORDING_LEVEL_FROZEN_SPLIT_TRAIN_PARTITION_ONLY`; **video-homogeneous mini-batches** (a window's graph lives in its
   video's graph bank); batch order across videos shuffled per epoch with `default_rng([seed, epoch])`; per-epoch validation monitor logged but never used
   for checkpoint selection (`FINAL_EPOCH_ONLY` kept); GPU allowed (CUDA is not bitwise reproducible — recorded). Architecture, loss, optimizer, scheduler,
   hyperparameters and scaler procedure are unchanged.
2. Held-out evaluation loads the model through `load_multivideo_model` (verifies manifest, artifact hashes, configuration hashes, scaler, checkpoint) because
   the P5.7.5 loader is bound to the training window run. The metric computation is the unchanged `evaluate_neural` / P5.4 evaluator.
3. Phase 6–8 are called per video with the Phase 9 S6–S8 contracts; `sample_status` reflects the partition instead of the pilot's hard-coded in-sample string.
4. Phase 3 sequence ids are `<set>_<video>` (set through `video.sequence_id` of the per-video configuration); `detections.jsonl` is stored gzip-compressed
   (the PIE evaluation runs before compression; the uncompressed SHA-256 is recorded).
5. Per-video window runs keep the generator's built-in label `PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`; the experiment label
   `MULTI_VIDEO_RECORDING_SPLIT_PSEUDO_TARGET_DEVELOPMENT` and the partitions come from the frozen split.
6. Optional split scheme `PIE_OFFICIAL_CALIBRATION_FROM_VAL` (a whole official validation set becomes the calibration partition) — off by default.

## 5. Split methodology

* Unit = PIE recording set (`set01`–`set06`); all clips of a set go to one partition; nothing is split at frame, window or clip level.
* The official split is parsed from `pie_data.py` (`image_set_nums`: train `set01, set02, set04`; val `set05, set06`; test `set03` in the transcribed reference,
  confirmed against the source file at run time). If the file is absent the stage is BLOCKED unless the user explicitly accepts the unverified reference.
* The split is frozen (`data/splits/<split_id>.json`, id = content hash) before windows are generated; resuming verifies the same content hash.
* Limitation: the official split has no calibration partition; with the default scheme calibration stays empty and blocked. Only six recordings exist, so
  partition-level statistics rest on very few independent units (1 test recording in the official split).

## 6. Dependencies (Colab)

Kept from Colab: Python, torch (CUDA build), numpy, scipy, matplotlib, OpenCV, jsonschema, psutil. Installed only if missing / incompatible:
`ultralytics==8.4.212`, `lapx==0.13.1` (raw-video mode), `pytest>=8` (tests). torch and numpy are never reinstalled; incompatibility stops the run.
`requirements.txt` (pins numpy 1.26.4 for the original PCG code) is deliberately **not** installed on Colab. Every session's versions and `pip freeze` are
stored in `<experiment>/environment/`.

## 7. Tests executed (local, Windows, `pcg-baseline` environment: Python 3.9.25, torch 2.2.1+cpu, numpy 1.26.4)

| Command | Result |
|---|---|
| `python scripts/validate_colab_notebook.py` | PASS — 49 cells (34 code), 0 problems, 87 calls into `pedsafety` bound to their real signatures, all repository paths exist, no use-before-definition |
| `pytest tests/multivideo` | 22 passed: units (inventory, AST split parsing without execution, frozen split + tamper/leak detection, experiment lock, stage tracker, markers, Phase 3 wrapper with mocked CLI), integration (real Phase 4–8 code on 4 synthetic videos: resume, stale-config refusal, train-partition-only training, byte-identical resumed training on CPU, hash-verified loader, per-partition evaluation, Phase 6/7/8 per stage, schemas, report), notebook (regenerated-from-builder check, structural validation, **local execution of every notebook cell** in precomputed mode on 6 synthetic videos, then a resumed run that verifies and skips everything) |
| `pytest tests` (full suite) | **1403 passed, 8 skipped, 1 failed** in 12 min. The failure is the documented pre-existing defect `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (`utils/Interaction.py`), identical to the Phase 9 baseline (1381 passed + 22 new tests). No test was modified, skipped or weakened. |

Not runnable locally: raw-video Phase 3 (no OpenCV / Ultralytics / GPU in this environment — covered only by the mocked-CLI unit test), Google Drive
mounting, Colab package versions (numpy 2 / newer Python), real PIE data, GPU training, and the optional Phase 9 pilot regression (needs git-ignored
pilot reference artifacts).

## 8. Known compatibility issues / risks

* **Colab Python / numpy versions** differ from the tested ones (Python 3.9 / numpy 1.26 locally). No numpy-2-removed API was found by a source scan, but the
  suite has not run on numpy 2 / Python 3.11+; Section 13 runs it on Colab and marks the experiment FAILED on unexpected failures.
* **Ultralytics 8.4.212 / lapx 0.13.1** are the versions of `requirements-perception.txt`; availability on PyPI at run time is not verified here.
* **Phase 3 memory:** the existing agent-state adapter keeps all frame results of a video until it ends; each video runs in its own subprocess, but a very
  long clip still needs that memory once.
* **Phase 4 wall time:** CPU-bound; it dominated the pilot (89 % of the Phase 4–8 time on a laptop CPU). Expect hours for the full dataset.
* **Strict Phase 9 contract in Task 2** (`S4A_ROWS`: no quarantined row) may reject some real videos; kept strict on purpose and reported per video.
* **Branch not on GitHub:** the default `REPO_SOURCE="github"` needs the branch pushed by the user; otherwise use the git-bundle option.
* **Drive FUSE:** large reads are slower than local disk; videos are copied to local storage first and outputs are published with verification.

## 9. Remaining manual steps

1. Push `feature/colab-full-pipeline-notebook` (or create the git bundle) — the assistant did not push.
2. Upload the PIE clips, annotation archives and `pie_data.py` to `MyDrive/PedestrianSafety/data/raw/pie/`.
3. Open the notebook in Colab with a GPU runtime and Run all; note the `EXPERIMENT_ID`.
4. Recommended first: a small separate experiment (`VIDEO_SUBSET`, e.g. one clip per set) to check timings and Colab compatibility.
5. Review `EXPERIMENT_REPORT.md` / `final_validation.json`; address failed videos and re-run with the same `EXPERIMENT_ID`.

## 10. Scientific limitations (unchanged by this work)

Image-space pixels only (moving camera, no ego-motion compensation); perceived-track pseudo-targets; held-out recordings are not annotation-validated;
primary metrics use the most-probable mode and oracle metrics are separate; risk scores are uncalibrated (`calibrated_probability = null`); categories and
alert thresholds are provisional; no lead-time validation; no real-time claim; only six recordings (one official test recording).
