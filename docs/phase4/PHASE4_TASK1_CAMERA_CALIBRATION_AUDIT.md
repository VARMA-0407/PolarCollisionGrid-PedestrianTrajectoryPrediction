# Phase 4 Task 1: Camera Calibration and Coordinate-System Audit (PIE `set01/video_0004`)

**Date:** 15 November 2026. **Branch:** `reproduction` (HEAD `7a6f56f`, unchanged). **Status:** `EVIDENCE_COLLECTION_PENDING`.
**Companion files:** `phase4_task1_calibration_audit.json` (machine-readable record), `PHASE4_COORDINATE_CONTRACT.md`, `PHASE4_CALIBRATION_EVIDENCE_CHECKLIST.md`.
**Scope:** audit only. No inference was rerun, no PIE data was downloaded, no homography was constructed, no code or Phase 3 output was changed.

## 1. Answer

| Representation | Supported for video_0004? | Basis |
|---|---|---|
| Image-space coordinates | **Yes. Authorized, and the safe operational representation.** | Boxes are in original 1920x1080 video pixels (verified from code); conventions defined in the contract. |
| Relative/projective road-plane coordinates | **Not yet.** | No ground-plane correspondences exist; distortion model and camera-motion policy are undecided. |
| Metric road-plane coordinates | **No.** | Only `K`, `D` values, a static height and pitch exist; no distortion model, no full pose, no scale evidence, no validation points, and the camera is vehicle-mounted and moving. |

The status is `EVIDENCE_COLLECTION_PENDING` rather than complete because (a) the actual video_0004 output records are in Google Drive and could not be inspected from this machine, so the Phase 3 coordinate findings rest on source code, and (b) the calibration evidence needed for any plane mapping does not exist yet.

## 2. Inspection performed

**Required documents:** all present: `PROJECT_MASTER.md`, `docs/phase2/task4_coordinate_systems_camera_geometry.md`, `docs/phase2/task3_agent_state_schema.md`, `docs/phase3/PHASE3_TASK1_PERCEPTION_FRONTEND.md`, `docs/phase3/PHASE3_TASK4_PIE_EVALUATION.md`, and the handoff file `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md` (Phase 3/4 plan sections). Discrepancies: `PROJECT_MASTER.md` still says no Phase 3 work has started (stale; not edited, as instructed); the handoff's Phase 4 item "calibrate camera and estimate road-plane homography" implicitly assumes a static camera, which PIE is not.

**Code:** `pedsafety/{calibration,projection,geometry,trajectory}.py`; `pedsafety/perception/{video_source,yolo_detector,pipeline,types,export,adapter}.py`; installed Ultralytics 8.4.212 `engine/predictor.py`, `models/yolo/detect/predict.py`, `utils/ops.py`. Local synthetic run records under `outputs/perception/20261114T122436Z_synthetic_replay/` (format check only).

**External (official PIE repository, master branch, fetched 15 Nov 2026 to the session scratchpad, not the repository):** `camera_params/calibration_data.json`, `camera_params/camera_params.png`, `README.md`, `utilities/pie_data.py`, and the `camera_params` directory page. The GitHub API was rate-limited, so the commit SHA was not resolved; file SHA-256 values are recorded in the JSON.

## 3. Task 1A: Repository audit

| Item searched | Finding |
|---|---|
| Camera calibration files | **None** for PIE in the repository. |
| Camera geometry / homography code | Stage 3 software exists: `geometry.py` (homography application with vanishing-line/denominator checks), `calibration.py` (`RasterSpec`, `RasterTransform`, `CalibrationManifest` restricted to `image_to_plane`, control points from independent sources only, `evaluate_metric_eligibility`), `projection.py` (sidecar projection; `world_*` filled only if `manifest_eligible` and promotion is enabled). Tested by `tests/task7/test_stage3_*.py`. It holds **no PIE values**. |
| Resize / crop / letterbox | Phase 3 code: none. Ultralytics internally letterboxes to `imgsz` (default 640) and maps boxes back to the original image with `ops.scale_boxes` (pad subtract, divide by gain, clip to `[0,W]x[0,H]`). |
| Footpoint | `((x_min+x_max)/2, y_max)` (`perception/types.py:184-186`, `:225-227`); used as `image_x/image_y` by the adapter. Not written to `tracked_agents.jsonl` / `detections.jsonl`. |
| Coordinate fields | Export: `bbox_xyxy`, `bbox_format = xyxy_image_pixels_top_left_origin_y_down`, `image_width/height`, `coordinate_system = captured_<W>x<H>`, `coordinate_space = image_pixels`. Agent states: `image_coordinate_system`, `image_coordinate_unit = pixel`, `world_* = null`, `world_position_valid = false`. Run manifest: `world_coordinates = "none: outputs are image-space pixels only"`. |
| Transformation tests | `tests/task7/test_stage3_{calibration,geometry,projection}.py`, Stage 2A/Task 1 export and adapter tests. |

**Conclusion (verified from code):** Phase 3 boxes and footpoints are in **original video pixel coordinates** of the captured raster, not model-input coordinates. Net transform from the captured frame to the exported numbers is the identity.

**Not yet verified on real data:** that the video_0004 records carry `captured_1920x1080`, 1920/1080 and in-range boxes, and which `imgsz` was used. Checklist section A gives a read-only Colab cell for this.

**Interpretation limitation (not a defect):** Ultralytics clips boxes to the image before Phase 3 sees them, so `clipped` in `detections.jsonl` is false for real YOLO output and does **not** mean "not truncated". Truncation must be inferred from border contact. Proposed isolated follow-up: compute a `touches_image_border` flag downstream in Phase 4 preprocessing; no exporter change.

## 4. Task 1B: PIE calibration audit

### 4.1 Verified facts
- `calibration_data.json` (288 bytes, SHA-256 `d6ee335b…2a92`) contains exactly `cam_height_mm = 1270`, `cam_pitch_deg = -10`, `dim = [1920, 1080]`, `K`, and `D` (4x1). **Every previously reported value matches exactly.**
- `camera_params.png` is a lens datasheet: Sony IMX178LQJ-C 1/1.8" sensor, 7.14 x 5.35 mm, EFL 3.88 mm +/-5%, F2.4, FOV 157 deg diagonal / 110 deg horizontal / 80.4 deg vertical, image circle 9.2 mm.
- The README only says `camera_params` "contains camera parameters". `utilities/pie_data.py` never reads the calibration file and hard-codes 1920x1080 for all videos. It reads ego-vehicle telemetry (OBD/GPS speed, heading, lat/long, yaw/roll/pitch, accelerometer, gyro) from `<video>_obd.xml`.
- The `camera_params` directory lists only those two files (same as Phase 2 Task 4's full tree listing on 11 Nov).
- PIE is recorded from a moving vehicle-mounted camera.

### 4.2 Interpretation of K and D
- `K`: standard pinhole matrix, `fx = 1004.84`, `fy = 1004.39`, `cx = 960.10`, `cy = 573.55`, zero skew, for a 1920x1080 raster (by `dim`). Pixel-centre convention and calibration residuals are undocumented.
- Consistency (inference): datasheet EFL / (7.14 mm / 1920 px) = 1043 px, with a +/-5% band of 991-1096 px that contains `fx`. This assumes the video spans the full sensor width. `cy` sits 33.6 px below the image centre (cause undocumented).
- **Distortion model: undocumented.** Two hypotheses:
  - **H-D1, OpenCV fisheye (Kannala-Brandt), `[k1..k4]`.** The 4x1 shape is the `cv2.fisheye.calibrate` output format. The FOV implied across 1920 px is about 116 deg (pure equidistant gives 109.5 deg), against 110 deg on the datasheet.
  - **H-D2, radial-tangential `[k1,k2,p1,p2]`.** This implies about 87-89 deg HFOV, inconsistent with the datasheet. It would also need unusually large tangential terms (0.040, 0.019).

  H-D1 is better supported, but it remains an inference. **Neither model is authorized** until the plumb-line test (checklist B) is done.

### 4.3 Extrinsics, plane, scale
| Item | Status |
|---|---|
| Height 1270 mm | value verified; reference point and surface undocumented; static |
| Pitch -10 deg | value verified; sign and reference axis undocumented; static, while real pitch varies with braking and grade |
| Roll, yaw, offsets, `R`, `t`, transform direction | unavailable |
| Road-plane equation, homography, ground correspondences | unavailable |
| Known distances, surveyed points, metric lane geometry | unavailable |
| Applicability to set01/video_0004 | undocumented (single global file) |

Diagnostic only, not authorized: assuming a pinhole camera, 10 deg downward tilt, zero roll and a flat road, the horizon would sit near image row **396.5 px**. Checking this against real frames (checklist C) tests the pitch sign and its variation.

### 4.4 Historical findings
Phase 2 Task 4 recorded the same file contents and the same absence of `R`/`t`/homography. This audit confirms it, and adds the datasheet transcription, the distortion-model analysis, the toolkit inspection, and the moving-camera implication.

## 5. Task 1C: Coordinate-system contract

Defined in `PHASE4_COORDINATE_CONTRACT.md`. Key points:
- **Image:** top-left origin, +u right, +v down, pixels, `[x_min,y_min,x_max,y_max]` continuous edge coordinates, footpoint bottom-centre, `pixel_center_half` (inferred; ambiguity <= 0.5 px, below the ~3 px detector quantum at `imgsz` 640), `as_captured` (distorted).
- **Camera:** not established. The OpenCV frame (+X right, +Y down, +Z forward, right-handed) is proposed as an assumption.
- **Road/world:** not authorized. Any future frame must declare `ego_ground_plane` vs `world_fixed`.
- **Projective:** not authorized. When it is, the unit is `projective_unit` (never metres) and the values go to the sidecar only.
- **Transforms and heading:** homographies are stored `image_to_plane`; `H' = H A^-1` and `K' = A K` for a verified raster transform `A` (here `A = I`). Image heading is `atan2(dv, du)`, positive clockwise on screen, range `[-pi, pi)`.

## 6. Task 1D: Ground-contact assessment

Based on code and the Phase 3 box semantics; the real video_0004 boxes are still to be checked (checklist A prints border-contact counts per class).

| Class | Recommendation |
|---|---|
| Pedestrian | Keep bottom-centre. Flag `y_max >= H-1` (feet cut off) and occlusion. Lower-body occlusion moves the point up the image, i.e. farther away on the ground. |
| Car | Relative proxy only. It marks the nearest visible bottom edge, not the vehicle centre or footprint. Not suitable for metric gap/TTC without a footprint model. |
| Bus / truck | As for car with larger bias; frequently truncated in ego view. No PIE ground truth for these classes. |
| Bicycle | Bottom-centre of the **bicycle** box (about midway between wheel contacts). When person and bicycle both fire for one cyclist, the person box bottom is elevated (pedal/shoe). |

Further failure modes: loose boxes (about 3 px quantisation, amplified near the horizon); partial visibility (no visibility score in the records); elevated agents (curbs, sidewalks, riders); road crown, grade and curbs that break the single-plane model. **No exporter defect was demonstrated**, so the exporter was not changed.

## 7. Task 1E: Feasibility matrix

| | Image space | Relative/projective | Metric road plane |
|---|---|---|---|
| Required | raster id/size, box convention, net transform, timestamps | >= 4 non-collinear ground correspondences + held-out, distortion domain, valid region, moving-camera policy | verified distortion model, full pose or verified conventions, independent scale, frame definition, held-out metric residuals, region, per-frame pose or proof static suffices |
| Available | all, from code | Stage 3 software only | `K`, `D` values; height and pitch values; datasheet |
| Missing | real-record confirmation | correspondences, distortion model, motion policy | distortion model, roll/yaw/`R`/`t`, conventions, per-video applicability, distances, validation, ego-motion |
| Risks | distortion left in; ego-motion mixed into image motion; pixel kinematics are not physical | arbitrary scale mistaken for metres; pitch changes between frames; horizon extrapolation | 1 deg pitch error gives metre-scale error far away; fisheye mis-modelling; unsafe TTC claims |
| Validation | checklist A | RANSAC inliers, fit/held-out px residuals, `cond(H)`, region | `evaluate_metric_eligibility` with independent points and a pre-declared threshold |
| **Status** | **AUTHORIZED** | **NOT AUTHORIZED (evidence pending)** | **NOT AUTHORIZED** |

Image-space coordinates remain the operational representation: `world_position_valid = false`, `kinematic_coordinate_space = "image"`.

## 8. Task 1F: Calibration acquisition plan (summary)

Full procedure is in the checklist. In order:
- **A.** Read-only record inspection in Colab.
- **B.** Plumb-line test of H-D1 vs H-D2 on frames read from the existing video.
- **C.** Horizon/vanishing-point check of pitch sign and dynamics.
- **D.** Independent ground correspondences. Preferred sources: the authors' rig calibration, a GPS-localised site survey (if feasible and safe), or documented marking standards (scale check only). Otherwise projective-only points from two independent annotators. Each needs >= 4 fit and >= 4 pre-selected held-out points.
- **E.** RANSAC estimation with residual, conditioning and region checks, then a `CalibrationManifest` run through `evaluate_metric_eligibility`.
- **F.** Obtain `video_0004_obd.xml`, the paper/supplementary, and the repository commit SHA.

No points or measurements were created.

## 9. Homography and image-processing requirements

Direction is `image_to_plane` (`x_plane ~ H x_image`), stored explicitly. The minimum is four non-collinear correspondences; 8+ is recommended, with held-out points disjoint from the fit set. Estimation uses normalised coordinates and RANSAC with a declared threshold and recorded inlier mask and count. Report forward residuals (fit and held-out) and backward pixel residuals, `cond(H)` and the denominator margin over the region. The valid region is an image polygon strictly below the horizon, and points outside are invalid. Use `H^-1` only when it is well-conditioned and named. For a verified raster transform `x' = A x`, `H' = H A^-1`. For Phase 3 output `A = I`: this was verified from code, and the pixel convention is inferred as `pixel_center_half`. Distortion must be handled by either (1) undistorting with a verified model and working in the `undistorted` domain, or (2) fitting in the `as_captured` domain and restricting to the fitted region. No distortion model is applied or assumed now.

## 10. Risks and limitations
1. Moving camera: no fixed world frame is possible from static calibration (high).
2. Wide-angle lens with an undocumented distortion model (high).
3. Projective or pixel values mislabelled as metres downstream (high).
4. `clipped=false` misread as not truncated (medium).
5. Bottom-centre is a poor contact proxy for vehicles and for occluded or elevated agents (medium).
6. Global calibration may not match this video (medium).
7. Phase 3 findings are from code; real records not yet inspected (medium).
8. PIE commit SHA not resolved (low; file hashes recorded).

## 11. Verification
See the final task response for the commands run. In summary: the JSON parses; every local path the JSON cites exists; protected files are unchanged (SHA-256 before/after over the git-tracked files, `experiments/`, `Data/`, `Store_Results/`, `outputs/`, `pedsafety/`, `configs/` and the Phase 2/3 docs); the relevant existing tests were run; the `git status` delta is only `docs/phase4/`.

## 12. Status
**`EVIDENCE_COLLECTION_PENDING`.** The audit questions are answered as far as the available evidence allows. Image space is the authorized representation. Closing the audit needs checklist A (real records), and any move beyond image space needs checklists B-E. Phase 4 Task 2 has not been started.
