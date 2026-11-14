# Phase 3 Task 1: YOLO + ByteTrack Perception Frontend (local implementation complete; Colab GPU validation pending)

**Date:** 14 November 2026. **Status:** `LOCAL_COMPLETE_GPU_AND_REAL_VIDEO_VALIDATION_PENDING`.
**Scope:** video ingestion -> YOLO detection -> ByteTrack tracking -> structured image-space tracked-agent output, visualisation, configuration, tests and
performance instrumentation. No world-coordinate projection, forecasting, conflict prediction or alerting was implemented.

**What has and has not been shown.**
- Shown locally (CPU, Windows): the code runs; the real OpenCV source, the real ByteTrack and the real Ultralytics load/predict path work together; outputs
  satisfy the contracts; 112 Task 1 tests pass; regressions are unchanged.
- **Not shown:** anything on a GPU (the Colab notebook is prepared and has not been executed); detection quality (no real road video and no ground truth
  exist yet); any tracking-quality metric; any performance claim. The only YOLO runs used a randomly initialised model built locally from the model yaml,
  so their detections (none) mean nothing.
- Phase 2 is **not** formally closed; the exit decision still needs your explicit approval and its limitations are unchanged (below).

## 1. Files inspected

`PROJECT_MASTER.md` (supporting draft; its "No Phase 3 implementation has started" is stale); the canonical master
`Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md` (grepped, not read in full) and its `.pre_task10_backup.md`;
`docs/phase2/PHASE2_MASTER_CONSOLIDATION_CHECKLIST.md`, `docs/phase2/schemas/agent_state.schema.json`; `docs/phase3/PHASE3_STAGE0_READINESS_AUDIT.md`,
`PHASE3_STAGE1_EVALUATION_FOUNDATION.md`, `PHASE3_STAGE2A_PERCEPTION_FOUNDATION.md` (read in full); `pedsafety/perception/{types,sources,detector,tracker,pipeline,adapter,
performance,mocks,__init__}.py`; `pedsafety/errors.py`; the Stage 6.1 dependency test; the installed Ultralytics `byte_tracker.py`, `bytetrack.yaml` and `parse_bboxes`.

## 2. Files created

| File | Purpose |
|---|---|
| `pedsafety/perception/config.py`, `configs/phase3/perception_default.json` | validated JSON configuration (video, detector, tracker, pipeline); unknown keys rejected |
| `pedsafety/perception/video_source.py` | OpenCV `FrameSource`: streaming, `frame_index / fps` timestamps, explicit release |
| `pedsafety/perception/yolo_detector.py` | Ultralytics `Detector`: local weights only, device selection, original class id/name preserved |
| `pedsafety/perception/bytetrack_tracker.py` | `Tracker` around the real `BYTETracker` |
| `pedsafety/perception/visualize.py` | annotated frames and an incremental video writer |
| `pedsafety/perception/export.py` | tracked-agent and detection JSONL rows (image space only) |
| `pedsafety/perception/runner.py` | the single execution layer (`execute_run`), run directory and manifest |
| `pedsafety/perception/benchmark.py`, `runtime_info.py`, `runtime_errors.py` | performance passes, environment capture, error types |
| `pedsafety/perception/synthetic_video.py` | locally generated test video and a **mock** replay detector |
| `scripts/run_perception.py`, `benchmark_perception.py`, `make_synthetic_video.py`, `build_colab_bundle.py` | command-line entry points |
| `notebooks/phase3_task1_perception_colab.ipynb` | Colab execution interface (not executed) |
| `requirements-perception.txt` | perception dependencies, kept separate from `requirements.txt` and `requirements-pedsafety.txt` |
| `tests/phase3/test_task1_*.py` (6 files), `tests/phase3/task1_helpers.py` | tests |
| `docs/phase3/PHASE3_TASK1_PERCEPTION_FRONTEND.md`, `PHASE3_TASK1_EXECUTION_GUIDE.md` | this report and the local/Colab guide |

## 3. Files modified (all untracked Phase 2/3 files; no tracked file changed)

- `pedsafety/perception/types.py`: optional `class_id: Optional[int] = None` on `RawDetection` and `DetectionRecord` (last field, default None, so every existing call still
  works), plus type validation in `detection_problem`.
- `pedsafety/perception/pipeline.py`: one line passes `d.class_id` into `DetectionRecord`.
- `tests/task7/test_stage61_dependencies_version.py`: the "every third-party import is declared" test now (a) maps `cv2` to `opencv-python-headless`, and (b) lets
  `pedsafety/perception/` additionally use `requirements-perception.txt` while the core package must still use only `requirements-pedsafety.txt`. Without this the test
  failed in `pcg-baseline` (it flagged `torch`) once perception code existed. It is a change to an existing test; please confirm you accept it.

The approved agent-state schema is unchanged (SHA-256 `784c8ca9c059ccbf0ddbb8bfcaf42cdfbbd60b33acfc0136c78a00fde46c3c6c`, same as Stage 2A).

## 4. Architecture decisions

- **Stage 2A interfaces reused unchanged**: `PerceptionPipeline` owns validation; the real components plug in behind `FrameSource`, `Detector`, `Tracker`. The adapter and
  `PerformanceRecorder` are reused as is. `pedsafety.perception` still imports no CV/ML library (a test checks this); only the wrapper modules do, lazily.
- **Video source**: OpenCV, sequential, one frame in memory. `frame_index` counts delivered frames; `timestamp_s = frame_index / fps`, labelled `frame_index_over_fps`
  (exact for constant frame rate; an approximation for variable frame rate, since no container PTS is read). No FPS resampling. `fps_override` exists only for containers that
  report no FPS. Raster is named `captured_<W>x<H>`; frames are not resized by the source.
- **Detector**: Ultralytics YOLO. Weights are never downloaded (a bare name like `yolo11n.pt` is rejected if the file is absent, because Ultralytics would otherwise fetch it).
  `RawDetection.label` is the model's original class name and `class_id` its original index; **semantic mapping is only the configurable `LabelMap`**. Default map:
  person -> pedestrian, bicycle -> bike, car, bus, truck; **motorcycle is rejected unless you map it** (never silently treated as a bicycle). The default COCO class filter is
  `[0, 1, 2, 5, 7]`. `device='cuda'` fails loudly without CUDA; `'auto'` picks CUDA when present. An injected fake model marks the detector `is_mock=True`.
- **Tracker**: the real ByteTrack, `ultralytics.trackers.byte_tracker.BYTETracker` (Ultralytics' implementation of Zhang et al., ECCV 2022: Kalman filter, high-score then low-score
  association, lost-track buffer), driven with our validated detections, decoupled from YOLO inference. Version: Ultralytics 8.4.212. The wrapper only converts data. Output notes:
  ByteTrack emits only tracks matched in the current frame, so `is_predicted` is always false; `matched_detection_id` comes from ByteTrack's detection index; by default the reported box
  is the **matched detection's box** (`box_source='detection'`), not the Kalman estimate (`'kalman'` is available); `track_buffer` is in frames; `class_policy` is `'detection'`
  (a class change is then flagged and dropped by the pipeline) or `'first_seen'` (pins the class, counts flips). Detector `conf` (0.10) is deliberately at or below `track_low_thresh`
  (0.10) so ByteTrack's second association can receive low-score detections; a configuration warning fires otherwise.
- **Output contract** (`export.py`, one JSON line per frame and track): `video_id` (= sequence id), `frame_index`, `timestamp_s`, `timestamp_source`, `track_id`, `detector_class_id`,
  `detector_class_name`, `semantic_class`, `confidence`, `bbox_xyxy`, `bbox_format` (`xyxy_image_pixels_top_left_origin_y_down`), `image_width`, `image_height`, `coordinate_system`,
  `coordinate_space = image_pixels`, `matched_detection_id`, `is_predicted`, `detection_validity`, `tracking_validity`. No world coordinates, velocities or physical units exist in it;
  footpoints are for the later geometry stage to derive from the box. Schema-valid agent states come from the existing adapter when `--scene-id` is supplied (image-space only,
  `world_position_valid = false`, world and kinematic fields null; mock sources must carry "mock" in `source_dataset`).
- **Performance**: four separate passes (decode-only, detection-only with a no-op tracker, tracking-only replaying captured detections, combined including decode); mean, P50, P95, max,
  processing FPS (frames / elapsed), warm-up excluded, peak GPU memory via `torch.cuda` only when CUDA exists. A report is marked as supporting claims only if no component is a mock, the clock
  is real and hardware is declared; `--plumbing-only` forces it false. Scope is the perception frontend only, never the PCG predictor and never end-to-end system FPS.
- **One implementation for both environments**: the Colab notebook unpacks the verified ZIP and runs the same `scripts/run_perception.py` and `scripts/benchmark_perception.py`.

## 5. Dependency versions and environment

Environment `D:\envs\pedsafety-perception` (Python 3.11.18, kept as approved; `pcg-baseline` untouched).

| Package | Version |
|---|---|
| torch / torchvision | 2.15.0+cpu / 0.29.2+cpu |
| numpy / scipy / pillow | 2.4.6 / 1.17.1 / 13.0.0 |
| ultralytics | 8.4.212 (installed with `--no-deps` to avoid a second OpenCV wheel) |
| opencv-python-headless | 5.0.0.93 |
| lapx | 0.13.1 (`lap`-compatible, imported as `lap`) |
| jsonschema / pytest / matplotlib / PyYAML / psutil / polars | 4.26.0 / 9.2.0 / 3.11.2 / 6.0.3 / 7.2.2 / 2.1.0 |

`pip check` reports two expected lines (`opencv-python` is satisfied by the headless wheel under another name; `nvidia-ml-py` is an optional, unused helper). `ultralytics` created its settings
file under `%USERPROFILE%\AppData\Roaming\Ultralytics` on first import (outside the repository). Hardware: Intel i5-1235U, 12 logical CPUs, 15.7 GB RAM, no CUDA GPU, Windows 11.
`pcg-baseline`: Python 3.9.25, torch 2.2.1+cpu, numpy 1.26.4, pytest 8.4.2 (no OpenCV or Ultralytics, by design).

**Licensing (not legal advice, not independently verified here):** Ultralytics code and its pretrained weights are published under AGPL-3.0 (enterprise licence available): acceptable for coursework
and research use, but review before distributing or deploying. OpenCV is Apache-2.0. Record the weights' URL, date and SHA-256 in Colab (the notebook does).

## 6. Model and tracker configuration (`configs/phase3/perception_default.json`)

Detector: `weights/yolo11n.pt` (not present in the repository), device `auto`, conf 0.10, IoU 0.7, imgsz 640, max_det 300, class filter `[0, 1, 2, 5, 7]`.
Tracker: `track_high_thresh 0.25`, `track_low_thresh 0.10`, `new_track_thresh 0.25`, `track_buffer 30` frames, `match_thresh 0.8`, `fuse_score true`, `box_source detection`, `class_policy detection`.
Pipeline: non-strict, `out_of_frame reject` with 1 px tolerance. These are starting values from the Ultralytics defaults, not tuned.

**Model choice** (vendor-published COCO figures recalled from memory, not measured or re-verified here; treat as indicative): YOLO11n about 2.6 M parameters and 39.5 mAP50-95,
YOLO11s about 9.4 M and 47. Use `n` for CPU development and `s` on a T4 if the latency budget allows. GPU memory for either is small at 640 px; it is **not measured** until the Colab run.

## 7. Tests: commands and exact results

Local (all in `D:\BTech-Research\PCG-Baseline`):

```
D:\envs\pedsafety-perception\python.exe -m pytest tests/phase3 -k task1 -q -rs
   -> 112 passed, 1 skipped, 140 deselected in 26.04s
D:\envs\pedsafety-perception\python.exe -m pytest tests/phase3 tests/task7 --ignore=tests/phase3/test_measurement_helpers.py -q -rs
   -> 798 passed, 1 skipped in 157.06s
%USERPROFILE%\anaconda3\envs\pcg-baseline\python.exe -m pytest tests/phase3 tests/task7 tests/phase2 -q -rs
   -> 1 failed, 810 passed, 5 skipped in 172.01s   (the skips are Task 1 files needing OpenCV/Ultralytics)
%USERPROFILE%\anaconda3\envs\pcg-baseline\python.exe -m pytest tests/phase3 -k task1 -q -rs
   -> 56 passed, 5 skipped, 140 deselected in 0.96s
```

- The single skip in the perception environment is the optional real-weights/real-video test (needs `PEDSAFETY_WEIGHTS` and `PEDSAFETY_VIDEO`); it is **BLOCKED**, not passed.
- The one failure in `pcg-baseline` is the documented original defect `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (`UnboundLocalError: TTC_mask` in
  `utils/Interaction.py`), unchanged and unrelated.
- `tests/phase3/test_measurement_helpers.py` (12 tests) cannot run in the new environment: it needs `pandas`, which is deliberately not installed there. It runs and passes in `pcg-baseline`.
- A first `pcg-baseline` run also failed the Stage 6.1 dependency test (it flagged `torch`); that was fixed as described in section 3 and the test passes in both environments.
- **Mutation checks:** 7 safeguards were broken one at a time in a scratch copy (timestamp off by one, tracker skipped on empty frames, class filter not sent, detection validity forced false, manifest
  `is_mock` forced false, capture not released, motorcycle mapped to bike). All 7 were caught.
- Coverage: config validation (24 invalid cases); additive `class_id`; video ingestion (valid input, indices, timestamps, end of video, invalid path/file/arguments, release after exhaustion, early
  exit and context exit, several resolutions and frame rates, FPS override, truncated file); detector (structure, original labels, explicit mapping incl. motorcycle, thresholds, device selection, CUDA
  unavailable, bad payload, missing weights, unreadable weights, mock marking); real ByteTrack (creation, persistence, new agent, occlusion reappearance, loss beyond the buffer, empty frames, multiple
  agents and classes, no duplicate ids, low-score association, class policy, box source, reset, thresholds); export contract; visualisation; integration (video -> detector -> ByteTrack -> JSONL,
  schema-valid agent states, identity through a crossing and a detector dropout, annotated video frame count, CLI exit codes and cleanup); benchmark passes and claim refusal.

## 8. Video validation results (local, CPU)

Synthetic scene (`scripts/make_synthetic_video.py`): 640x360, 10 fps, 120 frames of moving rectangles (two pedestrians crossing, a 6-frame detector dropout, a low-confidence stretch, a car entering through
the frame edge, objects appearing and disappearing). **Plumbing only; rectangles are not road users.**

Run with the mock replay detector, real video source, real ByteTrack (`run_perception.py --backend replay --save-video --scene-id synthetic_scene`): 120 frames seen, 120 processed, 0 skipped, 612 detections,
608 track rows, 7 track ids, no issues, `agent_states.jsonl` written and schema-valid. Mapping tracks to the known boxes: the two crossing pedestrians kept one id each with no switch; the pedestrian
with a 6-frame dropout kept the same id afterwards; the car got a second id for one frame at the right frame edge, where its clipped box was only a few pixels wide (an edge artefact). The annotated video
has 120 frames at 10 fps and was inspected on a sample frame (boxes, `ID n class conf` labels, header with frame and time are drawn; the undetected pedestrian correctly has no box).

Run with the real YOLO wrapper on random-initialised weights (built locally with `YOLO("yolo11n.yaml")`, no download): the load, hash, CPU inference, ByteTrack, JSONL, manifest and annotated video all ran;
0 detections, as expected. Error paths: missing video and missing weights both exit with code 2 and a clear message and leave no empty run directory.

**Real road video: PENDING.** No detection quality, id-switch rate or miss rate is claimed anywhere.

## 9. Measured latency (local CPU; hardware: i5-1235U, 12 logical CPUs, 15.7 GB RAM, no GPU)

| Run | Frames | Combined mean / P50 / P95 (incl. decode) | Combined FPS | Note |
|---|---|---|---|---|
| mock replay detector, 640x360 | 120 | 2.36 / 2.32 / 3.09 ms | 423.6 | mock detector: **no claim**; shows decode about 0.9 ms and tracking about 1.3 ms per frame with 5 objects |
| random-init yolo11n, CPU, 640x360 | 60 | 55.48 / 54.62 / 60.77 ms | 18.0 | `--plumbing-only`: random weights, synthetic video, zero detections; **no claim** |

Stage split of the second row: decode 0.92 ms mean, detection 55.45 ms mean, tracking 0.09 ms mean (no detections to track). `supports_performance_claims` is false for both. GPU latency and peak GPU
memory are **not measured**. Results are in `outputs/perception_bench/` (untracked).

## 10. Colab deliverable (prepared, not executed; revision 2 after the pre-Colab audit)

`notebooks/phase3_task1_perception_colab.ipynb`: a bootstrap cell with status tracking and a failure-proof manifest writer, GPU check (stops without a GPU), Drive mount and a reserved unique
Drive experiment folder, ZIP SHA-256 verification (stops on any mismatch, unsafe path or extra/missing file; optional expected hash), pinned install, version capture, weights download
with a size and zip-header check and URL / date / size / SHA-256 / Ultralytics version / licence recorded, the Task 1 tests, input-video setup, `run_perception.py`,
`benchmark_perception.py`, and a final manifest. Revision 2 changes (notebook only): (1) a synthetic-video benchmark is run with `--plumbing-only` and a guard rejects a benchmark that
claims otherwise; (2) all outputs are produced under `/content/work` and copied to Drive at the end (or on failure) with read-back verification, and a real input video is copied off Drive
first; (3) every stage runs under a wrapper that records state, error and traceback and writes `colab_run_manifest.json` locally and to Drive even when ZIP verification, installation or the
weights download fails. The notebook JSON parses and all code cells compile; its cell logic was exercised locally with a stubbed Drive/GPU (a local simulation, **not** a Colab run). Execution steps:
`docs/phase3/PHASE3_TASK1_EXECUTION_GUIDE.md`. Unverified assumptions: the weights release URL; that the pinned `ultralytics==8.4.212` and `lapx==0.13.1` install on Colab's Python and NumPy.

## 11. Known limitations

- No GPU run, no real road video, no ground truth: no accuracy, tracking or performance claim is possible yet.
- `timestamp_s = frame_index / fps` is exact only for constant-frame-rate video; variable-frame-rate video would need container PTS (not implemented, PyAV was the alternative).
- A failed read cannot be told apart from end of file in OpenCV; a short video is reported via the advisory frame-count shortfall in `stats`.
- ByteTrack emits no coasting boxes, so a track that is not re-matched produces no rows; the schema still cannot carry predicted-only rows (Stage 2A rule).
- ByteTrack is class-agnostic: a class flip on one track is flagged and the row dropped (`class_policy='detection'`) or masked (`'first_seen'`); neither is tuned. A cyclist can appear as both `person`
  and `bicycle`, giving a pedestrian and a bike at one place; not de-duplicated.
- Boxes are image pixels only; footpoints are a ground-contact proxy; moving-camera video makes later metric work harder. Perception ids are per sequence only.
- COCO-pretrained YOLO is not adapted to the target scenes; thresholds are Ultralytics defaults, untuned.
- The Ultralytics `half` argument is only sent when requested on a GPU (recent versions deprecate it); behaviour of FP16 on the T4 is untested.
- `tests/phase3/test_measurement_helpers.py` does not run in the perception environment (needs `pandas`); the export tests and the other OpenCV tests are skipped in `pcg-baseline` by design.
- All Task 7, Stage 1, Stage 2A and Task 1 files are untracked and outside the 20-artifact manifest.

## 12. Unresolved issues and decisions for you

1. Accept the edit to the Stage 6.1 dependency test (section 3)?
2. Supply a road video (rights confirmed) and confirm the weights source URL for Colab.
3. Commit policy: nothing is committed; the untracked Phase 2/3 work still has no Git history.
4. Phase 2 exit decision remains unrecorded; HBS units, calibration, provenance and licence remain open.
5. Whether tracking-quality metrics (HOTA, IDF1) should come next, which needs a ground-truth-annotated clip and a fixed matching protocol.

## 13. Git status and preservation audit

- Branch `reproduction`, HEAD `7a6f56f`, unchanged; no commit, no push. `git diff` is empty (no tracked file modified). All new and modified files are untracked.
- Preservation: SHA-256 of all 53 git-tracked files plus every file under `experiments/`, `Data/`, `Store_Results/` and the schema (121 files) was taken before the first change and again after the last code
  change: **identical**. `utils/`, `train.py`, `test.py`, the HBS data and every checkpoint and result artifact are unchanged. `pcg-baseline` packages were not modified.
- New untracked directories: `notebooks/`, `outputs/` (synthetic video, run directories, benchmark JSON, Colab bundle; about 2 MB), `configs/phase3/`.

## 14. Recommended next task

Run the Colab notebook after review, with a road video you provide, then add detection and tracking metrics on a small annotated clip behind the Stage 2A evaluator interfaces
(Stage 2A plan, item 6), before any geometry or forecasting work.
