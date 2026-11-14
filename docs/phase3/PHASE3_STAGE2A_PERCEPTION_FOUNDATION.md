# Phase 3 Stage 2A: Perception Frontend Foundation

**Status:** COMPLETE_WITH_LIMITATIONS.
**Date:** 14 November 2026
**Scope:** architecture, interfaces, validation, mock-based tests and documentation. **No YOLO, ByteTrack, OpenCV or any model was integrated, no weights
or data were downloaded, no package was installed, and no detection, tracking or runtime performance was measured or claimed.**

## 1. Stage 1 verification results

| Check | Result |
|---|---|
| Files exist | `docs/phase3/PHASE3_STAGE0_READINESS_AUDIT.md`, `docs/phase3/PHASE3_STAGE1_EVALUATION_FOUNDATION.md`, `pedsafety/chronological.py` (187 lines), `pedsafety/evaluation.py` (103 lines), `tests/phase3/` (`conftest.py`, `test_stage1_evaluation_foundation.py`) |
| Stage 1 tests | `tests/phase3` ran 11 passed before Stage 2A work |
| Branch / HEAD | `reproduction` / `7a6f56ff513034924200423843d63896454e8cbc` (same as `origin/reproduction`) |
| `git log -5` | `7a6f56f`, `0735a0f`, `6da10f2`, `49e27e3`, `d1c9bd8` |
| Tracked vs untracked | `git ls-files` lists 53 files; **none** is under `pedsafety/`, `tests/`, `docs/`, `configs/` or `scripts/`. `git ls-files --others` lists the Stage 1 files, so **Stage 1 (and everything since Phase 2 Task 7) is untracked**. An empty `git diff` only shows that the 53 tracked files are unmodified; it says nothing about the new files. |
| Implementation read | `chronological.py` validates strict frame/time order, requires explicit increasing boundaries, never randomises, keeps a window only if its whole dependency interval lies in one period, and returns excluded windows. `evaluation.py` selects a candidate by `first` or a model score and raises on `best_of_k`, `oracle` or `ground_truth`. Both match what the Stage 1 document describes. |
| Safety | No Stage 1 file touches the schema, the PCG code or the data. Nothing unsafe found. |
| Minor inconsistency (not changed) | The `chronological.py` module docstring describes half-open intervals (`[train_end, validation_end)`), while the Stage 1 document and `SplitBoundaries.partition()` use an inclusive window-END check (`end <= boundary`). The code is self-consistent; only the prose differs. Recorded for Stage 1's owner; Stage 2A did not edit Stage 1 files. |

Stage 1's own status is COMPLETE_WITH_LIMITATIONS. This stage did not review or change it beyond the checks above.

## 2. Perception architecture

```
FrameSource ──► (FrameMeta, payload) ──► PerceptionPipeline ──► Detector ──► RawDetection
                                              │  validate frame      │
                                              │  validate detections ◄┘ → DetectionRecord (category, box policy, provenance)
                                              └► Tracker.update ──► TrackObservation → validate → TrackRecord
                                                                                             │
                                          tracks_to_agent_states (adapter) ◄─────────────────┘
                                              │  RawObservation → existing build_agent_states (Stage 2, approved schema unchanged)
                                              ▼
                                   agent-state records + Stage 2 sidecar + perception sidecar
                                              ▼
                              existing Stage 3 (projection) / Stage 4 (causal features) / Stage 5 (splits, windows)
```

Design rules:
- The pipeline **owns validation**; detectors and trackers are untrusted callables behind abstract interfaces.
- The forecasting side depends only on agent-state records. It never imports YOLO, ByteTrack, OpenCV, numpy or torch through this package
  (a test imports `pedsafety.perception` in a fresh interpreter and asserts none of `cv2`, `ultralytics`, `torch`, `torchvision`, `numpy`, `lap`,
  `filterpy`, `yolox` is loaded).
- The adapter reuses the **existing Stage 2 builder**, so identity namespacing, class mapping, box rules and schema validation are the already-tested ones.
- **No schema change was needed.** The approved schema file is byte-identical (SHA-256 `784c8ca9c059ccbf0ddbb8bfcaf42cdfbbd60b33acfc0136c78a00fde46c3c6c`),
  asserted by a test.

## 3. Module responsibilities (`pedsafety/perception/`)

| Module | Responsibility |
|---|---|
| `types.py` | Data contracts and validators: `FrameMeta`, `RawDetection`, `DetectionRecord`, `TrackObservation`, `TrackRecord`, `ComponentInfo`, `PerceptionClass`, `LabelMap`, `IssueCode`, `PerceptionIssue`, `PerceptionError`. Stdlib plus the shared bounding-box rule only. |
| `sources.py` | `FrameSource` interface (frames as metadata plus an opaque payload). No decoding. |
| `detector.py` | `Detector` interface. |
| `tracker.py` | `Tracker` interface (`update`, `reset`; causal). |
| `pipeline.py` | `PerceptionPipeline`, `PipelineConfig`, `FrameResult`, `PerceptionResult`. Frame ordering, detection and track validation, error policy, provenance binding, optional timing hooks. |
| `adapter.py` | `AdapterConfig`, `tracks_to_agent_states`, `PerceptionProvenance`. Track records to schema-valid agent states. |
| `metrics.py` | Detection and tracking EVALUATION INTERFACES, ground-truth contract, metric catalogue, `EvaluationReport` that refuses mocks. No metric implementation. |
| `performance.py` | `PerformanceRecorder` (injectable clock), `PerformanceSummary`, `percentile`. Measurement scaffolding only; the claim predicate is conservative. |
| `mocks.py` | Clearly labelled test doubles: `ScriptedFrameSource`, `synthetic_frames`, `MockDetector`, `MockTracker` (greedy IoU matcher; **not** ByteTrack). |

## 4. Data contracts

- **FrameMeta:** `source_id`, `sequence_id`, `frame_index`, `timestamp_s`, `width`, `height`, `coordinate_system` (names the pixel raster; never invented),
  `timestamp_source` (`pts`, `frame_index_over_fps`, `dataset_annotation`, `mock_clock`), optional `nominal_fps`.
- **RawDetection (detector output):** `detection_id` (unique in the frame), `label` (detector's string), `bbox_xyxy`, `confidence`.
- **DetectionRecord:** all frame fields plus `category` (one of pedestrian, car, truck, bus, bike), `label_raw`, `bbox_xyxy` (as used), `bbox_raw` and
  `clipped`, `confidence`, and the detector's `ComponentInfo`.
- **TrackObservation (tracker output):** `track_id` (positive int), `category`, `bbox_xyxy`, `matched_detection_id`, `is_predicted`, optional `tracking_confidence`.
- **TrackRecord:** frame fields, `track_id`, `category`, box, `detection_confidence` (of the matched detection), `matched_detection_id`, `is_predicted`,
  `tracking_confidence`, detector and tracker `ComponentInfo`.
- **ComponentInfo:** `kind`, `name`, `version`, **`is_mock` (explicit bool)**, free-form `parameters` (a real detector will record weights hash and thresholds).
- **Agent-state mapping:** `track_id` -> `<dataset>:<sequence>:<id>` (Stage 2 rule); `scene_id` from required `AdapterConfig`; `source_label` = category;
  schema class from the Stage 2 class map (car, truck, bus -> `vehicle`; bike -> `bicyclist`, an assumption recorded in Stage 2; pedestrian -> `pedestrian`);
  `bbox_xyxy`, `bbox_center`, `footpoint` from the box; `image_x`/`image_y` = footpoint; `image_coordinate_system` = the frame's raster name,
  unit `pixel`; `detection_confidence` from the matched detection; `tracking_confidence` only if supplied; `world_position_valid` always false, all world and
  kinematic fields null (no unit contract, so nothing is invented).
- **Sidecar (`PerceptionProvenance`):** input index, perception track id, matched detection id, detector, tracker and frame-source `ComponentInfo`, `is_mock`,
  pipeline and adapter versions. The schema cannot carry these.
- **Dropped rows:** predicted-only track rows (no matched detection) are dropped and counted, because they are estimates and the schema cannot mark a causal
  estimate (recorded Stage 4 safeguard #2).

## 5. Bounding-box and timestamp conventions

- **Box:** `(x_min, y_min, x_max, y_max)`, floats, image pixels of the raster named by `coordinate_system`, origin top-left, x right, **y down**. Never reordered;
  zero width or height is invalid; non-finite is invalid. Pixels are **not metres**; nothing here converts them.
- **Out-of-frame boxes:** default `reject` (a box leaving the frame by more than `bbox_tolerance_px`, default 0, is dropped with an issue). Optional `clip`
  keeps the clipped box, sets `clipped=True` and preserves the original in `bbox_raw`. A box with nothing left after clipping is rejected.
- **Footpoint:** bottom-centre `((x_min + x_max) / 2, y_max)`, a ground-contact proxy for upright agents; unreliable for cropped, occluded or non-upright boxes.
- **Pixel-centre convention:** not decided here. It belongs to the calibration raster (Stage 3 `RasterSpec`) and must be declared by a real frame source.
- **Timestamp:** seconds from the sequence origin, finite and >= 0, strictly increasing; `timestamp_source` records its origin. A frame rate is never guessed
  (`nominal_fps` is optional metadata). Frame indices are non-negative, strictly increasing, gaps allowed and preserved.
- **Confidence:** a detector score in [0, 1] (booleans and strings rejected); not a calibrated probability.
- **Track ids:** positive integers, unique within a sequence, never reused for another object; a track's class may not change.

## 6. Mock-based test results

`tests/phase3/test_stage2a_perception.py`: **94 passed**. Every component in it is a mock; no number is a detection, tracking or runtime result.
Coverage: frame order and timestamps (invalid, missing, non-increasing, sequence change, non-`FrameMeta` items, gaps, empty source, strict raise); empty frames
(tracker still updated); multiple detections per frame; box validity, out-of-frame reject/clip/tolerance; confidence bounds including endpoints; supported
labels, aliases and unsupported labels; duplicate detection ids; track persistence, disappearance and reappearance within and beyond `max_age`, class-change and
duplicate-id guards, invalid tracker output; detector and tracker exceptions in strict and non-strict modes; non-detection objects; component-kind checks;
adapter schema validity, class and coordinate mapping, sidecar provenance, required `scene_id`, mock-dataset rule, predicted-row dropping, tracking
confidence; **integration with the existing Stage 4 causal features** (4 px per 0.1 s gives 40 px/s, unit `pixel/s`, not physical); independence from CV/ML
imports; mock labelling; evaluator interfaces refusing to invent numbers; report refusal of mocks; performance arithmetic with a fake clock; claim predicate
(mock, fake clock, undeclared hardware and empty runs can never support a claim).

**Mutation checks:** 14 safeguards were broken one at a time in a scratch copy (frame-index order, timestamp order, mock-dataset rule, predicted rows written as
observations, mock refusal in reports, claims for mocks, claims with a fake clock, duplicate track ids, track class change, confidence bounds, unsupported class
defaulted, out-of-frame reject, duplicate detection ids, centre instead of footpoint). One survived the first run (fake clock masked by the mock check); a test was
added and it is now caught. All 14 are caught.

Regression (after the changes): `tests/task7` 581 passed; `tests/phase3` 105 passed (11 Stage 1 + 94 Stage 2A); schema 22 passed; full `tests/phase2` 33 passed and
1 failed (the original `Interaction.py` `TTC_mask` defect, unchanged).

## 7. Optional dependencies required for Stage 2B (NOT installed; versions to be chosen then)

| Need | Candidate | Note |
|---|---|---|
| Video decode and frame pixels | OpenCV (`opencv-python`) or PyAV/FFmpeg | A real `FrameSource` lives behind the interface; pick the one that gives presentation timestamps (`pts`) so `timestamp_source="pts"` is honest |
| Detector | Ultralytics YOLO | Licence terms of the library and of each weight file must be checked first (not verified in this stage) |
| Tracker | A ByteTrack implementation (the official code or a maintained port) | Needs a small wrapper that converts to `TrackObservation`; its licence must be checked (not verified) |
| Matching helper | `lap`/`scipy` (scipy 1.12 is already present) | Only if the chosen tracker needs it |
| Weights | A pretrained detector checkpoint | Download only with your approval; record URL, date, licence and checksum |
| Metrics libraries | e.g. a MOT-metrics package | Only after the matching protocol is fixed; implement behind `DetectionEvaluator`/`TrackingEvaluator` |
| Declared in requirements | none yet | Add a separate `requirements-perception.txt` in Stage 2B, not to `requirements.txt` |

## 8. Local versus Colab runtime considerations

| | Local (audited) | Colab (assumed, not verified) |
|---|---|---|
| Hardware | i5-1235U, 15.7 GB RAM, Intel Iris Xe, no CUDA | A GPU runtime, if the plan and quota allow |
| PyTorch | 2.2.1 CPU build | A CUDA build is available there, version to be checked |
| OpenCV / Ultralytics / tracker / FFmpeg | none installed | Installable per session |
| Suitable for | everything in Stage 2A; interface tests; adapter and downstream tests | detector inference, timing and training |
| Risks | CPU YOLO is slow; decode and inference would not represent a deployment target | Sessions are ephemeral; versions drift; storage is temporary unless Drive is used |

Rules for Stage 2B: declare the hardware and software versions with every timing; keep raw video and weights out of Git; record run configuration and hashes;
keep the perception environment separate from `pcg-baseline`. Colab availability and quota are **not assumed**; you must confirm them.

## 9. Dataset and licensing dependencies

- No real video is available locally; HBS has none. Stage 2B needs at least one small, legitimately obtained clip set with ground-truth boxes and track ids.
- PIE (public download, MIT-labelled repository; the video licence scope was not stated on the page) is the only candidate not gated by an access request.
  TITAN and inD need requests (non-commercial; inD forbids redistributing the data). Nothing was requested or downloaded.
- PIE and TITAN are moving-camera datasets. The calibration manifest assumes one static calibration, so metric velocity is not available from them without
  ego-motion handling. Perception output stays in image pixels; the metric policy is an open decision from Stage 0.
- Detector and tracker licences, and the licence of any weights, must be reviewed before use (not done).

## 10. Known limitations

- No real detector, tracker, frame source or decoder exists; the only runnable components are mocks. The mock tracker is **not** ByteTrack and says nothing
  about its behaviour.
- `ScriptedFrameSource` carries no pixels; the payload path to a real detector is untested.
- No metric is implemented (by design); the matching protocol (IoU rule, ignore regions, crowd handling) is undecided.
- Timing scaffolding is arithmetic-tested with a fake clock only. No FPS or latency has been measured and none may be claimed. Memory measurement covers the Python
  heap only; GPU memory needs a separate probe.
- Predicted-only tracks are dropped, so a real tracker's coasting boxes will not reach the forecasting pipeline until the schema/sidecar policy is revisited.
- Class coverage is the five requested categories; `motorcycle` is rejected by default. `bike` -> `bicyclist` remains an assumption.
- `scene_id` has to be supplied by the caller; a perception run alone cannot say which scene it saw, so Stage 5's scene grouping needs that metadata.
- Frame skipping on an invalid frame leaves a gap the tracker never sees; for a real tracker the effect on identity is untested.
- All Task 7, Stage 1 and Stage 2A files are untracked and outside the 20-artifact manifest.

## 11. Stage 2B implementation plan (needs your approval)

1. **Decisions:** compute plan (Colab or other), dataset for the first clip set, licence review of detector, tracker and weights, permission to create a separate
   environment on D: and to install packages.
2. **Environment:** new environment, `requirements-perception.txt`, a recorded environment manifest (versions, CUDA, hardware).
3. **Frame source:** `VideoFileFrameSource` using the chosen decoder, with `pts` timestamps, an explicit raster name, a decoding test on a tiny synthetic video
   generated locally (no download), and a check that frame count and timestamps match.
4. **Detector wrapper:** `UltralyticsDetector` behind `Detector`, weights path and SHA-256 in `ComponentInfo.parameters`, label map and thresholds declared,
   determinism check on fixed input; first run on a handful of frames from the approved clip, with outputs inspected and not reported as metrics.
5. **Tracker wrapper:** ByteTrack behind `Tracker`, reset semantics, id persistence checks on a short clip, and a decision on predicted-only rows.
6. **Evaluation:** fix the matching protocol, implement detection metrics (precision, recall, mAP) and tracking metrics (HOTA, IDF1, MOTA, identity switches) behind the
   interfaces, validate them against a small hand-computed example before real data.
7. **Performance:** timed runs with declared hardware, fixed resolution and batch size, warm-up excluded, processing FPS and latency distributions, Python-heap and
   GPU memory; no claim before the predicate `supports_performance_claims` is true and the setup is documented.
8. **Integration test:** real clip -> detections -> tracks -> agent states -> existing Stage 4 features, in image pixels only.
