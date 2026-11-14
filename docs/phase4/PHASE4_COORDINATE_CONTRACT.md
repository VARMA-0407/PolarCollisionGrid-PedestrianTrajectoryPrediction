# Phase 4 Coordinate Contract

**Version:** 1.0.0 (Phase 4 Task 1). **Date:** 15 November 2026.
**Applies to:** PIE `set01/video_0004` Phase 3 outputs and everything Phase 4 derives from them.
**Machine-readable record:** `docs/phase4/phase4_task1_calibration_audit.json` (`coordinate_conventions`, `image_transformations`).

This contract fixes what each coordinate frame means, how values move between frames, and which frames are currently **authorized**.
It extends Phase 2 Task 4 (`docs/phase2/task4_coordinate_systems_camera_geometry.md`) and the Task 3 schema; it does not change the schema.

## 0. Authorization summary

| Frame | Status for video_0004 | Allowed labels |
|---|---|---|
| A. Image (`captured_1920x1080`) | **AUTHORIZED** (operational default) | `pixel`; `kinematic_coordinate_space = "image"` |
| B. Camera (3-D) | **NOT ESTABLISHED** | none; do not emit |
| C. Road/world (metric) | **NOT AUTHORIZED** | none; `world_position_valid = false`, `world_*` null |
| D. Relative/projective plane | **NOT AUTHORIZED (evidence pending)** | sidecar only, unit `projective_unit`; never metres |

## A. Image coordinates

| Item | Definition | Evidence status |
|---|---|---|
| Frame id | `captured_<W>x<H>`, expected `captured_1920x1080` | from code (`pedsafety/perception/video_source.py:76`); real records not yet inspected |
| Raster | the decoded video frame, **no** resize, crop, pad or undistortion | from code |
| Origin | top-left corner of the top-left pixel | from code / project convention |
| +u | right | from code |
| +v | down | from code |
| Units | pixels | from code |
| Pixel-centre convention | `pixel_center_half`: pixel `(i, j)` covers `[i, i+1) x [j, j+1)`; its centre is `(i+0.5, j+0.5)`; valid domain `[0, W] x [0, H]` | **inference** (Ultralytics affine rescale + clip to `[0, W]`); ambiguity <= 0.5 px, below the ~3 px detector quantum |
| Distortion domain | `as_captured` (lens distortion present) | from code (nothing undistorts) |
| Bounding box | `bbox_xyxy = [x_min, y_min, x_max, y_max]`, continuous edge coordinates, `x_min < x_max`, `y_min < y_max`; format string `xyxy_image_pixels_top_left_origin_y_down` | from code |
| Centre | `((x_min + x_max)/2, (y_min + y_max)/2)` | from code |
| Footpoint | `((x_min + x_max)/2, y_max)`: bottom-centre ground-contact **proxy** | from code (`types.py:184-186`); proxy is a project assumption |
| `image_x`, `image_y` in agent states | the footpoint | from code (`adapter.py`, `track_to_observation`) |

**Model-input vs original pixels.** Boxes are in **original video pixels**. YOLO runs on a letterboxed image (for 1920x1080 at `imgsz=640`, expected scale 1/3 to 640x360, padded 12 px top/bottom to 640x384), and Ultralytics `ops.scale_boxes` removes the padding, divides by the gain and clips to the original image before the boxes reach Phase 3 (`ultralytics/models/yolo/detect/predict.py:130`, `utils/ops.py:220-264`). Net transform from captured raster to exported coordinates: **A = I**. The exact `imgsz` used must be confirmed from the Drive `run_manifest.json`.

**Truncation signal.** Because Ultralytics clips first, `detections.jsonl` field `clipped` is `false` for real YOLO output and does **not** mean "not truncated". Treat `x_min <= 1`, `x_max >= W-1` or `y_max >= H-1` as possible truncation (derived downstream; the exporter is not changed).

## B. Camera coordinates

**Not established for PIE.** PIE documents no camera-frame axes, handedness, or transform direction.

Proposed project convention, to be adopted only after extrinsics are verified (**assumption**):

- origin at the optical centre; +X right, +Y down, +Z forward along the optical axis; right-handed (OpenCV convention);
- length unit must be declared per use (`mm` or `m`); no default;
- extrinsic direction stored explicitly, preferably `X_c = R X_w + t` (world-to-camera), with the direction string recorded.

Ideal (undistorted, normalised) image point: `[x_n, y_n, 1]^T ~ K^-1 [u, v, 1]^T` applies **only after** distortion has been removed with the verified model. PIE's `K` refers (by its `dim` field) to a 1920x1080 raster; its pixel-centre convention is undocumented.

## C. Road / world coordinates

**Not authorized.** No origin, axes, handedness, unit or scale is evidenced for video_0004.

Rules:

1. `world_x`, `world_y` stay null and `world_position_valid` stays false.
2. A metric frame may be declared only through a `CalibrationManifest` (`pedsafety/calibration.py`) that passes `evaluate_metric_eligibility` with **independent** control points (survey, metric map, known distance), held-out points, a declared threshold in physical units, documented origin/axes/handedness/scale basis, a valid region and a known distortion domain. Even then the status is `manifest_eligible`, not "physically verified".
3. PIE uses a **moving, vehicle-mounted** camera. A plane derived from the mounting pose is **ego-vehicle-relative and time-varying**. Any future frame must say which it is:
   - `ego_ground_plane` (moves with the vehicle; requires per-frame or validated-static pose), or
   - `world_fixed` (requires ego-motion integration; out of current scope).
4. Ego-vehicle telemetry (OBD/GPS) describes the recording vehicle and must never be assigned to agents.

## D. Relative / projective coordinates

**Not authorized yet** (no ground correspondences exist).

When later authorized:

- produced only by an `image_to_plane` homography fitted to >= 4 non-collinear ground points (more for RANSAC and held-out checks);
- unit label `projective_unit` (or `relative`), **never** `m`; `world_position_valid = false`; stored in the Stage 3 projection **sidecar** only (`pedsafety/projection.py` already does this);
- arbitrary projective scale and, in general, arbitrary affine/projective distortion of distances and angles. Distances, speeds and TTC computed in this frame are **not physical**;
- valid only inside a declared image polygon strictly below the horizon line (the code rejects regions crossing the vanishing line).

## E. Transform direction and image transformations

| Rule | Statement |
|---|---|
| Homography direction | stored **image_to_plane**: `x_plane ~ H x_image` (enforced by `CalibrationManifest`). Inverse `x_image ~ H^-1 x_plane` only if `H` is well-conditioned and the inverse is named explicitly. |
| Raster change | if pixel coordinates are transformed `x' = A x` (resize, crop, pad), the homography becomes `H' = H A^-1`; intrinsics `K' = A K`. `A` must be expressed in the same pixel-centre convention (`RasterTransform.matrix` conjugates by a 0.5 px shift for `pixel_center_integer`). |
| Phase 3 -> Phase 4 | `A = I` (no image transformation between the captured raster and the exported boxes). No `H A^-1` update is needed **provided** the calibration raster is `captured_1920x1080`, `pixel_center_half`, `as_captured`. |
| Distortion | nonlinear; not representable by `A` or `H`. Choose one: (1) undistort points with the **verified** model, then fit/apply `H` in the `undistorted` domain; or (2) fit `H` directly in the `as_captured` domain and restrict it to its fitted region. Never mix domains (`resolve_effective_calibration` rejects a domain mismatch). No model is authorized today. |

## F. Heading and rotation

| Item | Definition | Status |
|---|---|---|
| Image heading | `atan2(dv, du)`: angle from +u toward +v | from code (`pedsafety/trajectory.py:195, 395`) |
| Positive rotation | toward +v, i.e. **clockwise on screen** (because +v points down) | from code |
| Range | `[-pi, pi)` (`wrap_angle`, `trajectory.py:66`) | from code |
| Image heading meaning | direction of image motion; on a moving camera it mixes ego-motion and agent motion; **not** a world heading | stated |
| World heading | not authorized; when a plane frame exists, `atan2(dy, dx)` from +x toward +y in that frame, with handedness recorded | pending |

## G. Required metadata on every Phase 4 record or sidecar

`image_coordinate_system` (raster id), `image_coordinate_unit = pixel`, `world_position_valid`, `kinematic_coordinate_space`, `kinematic_position_unit`, `kinematic_time_unit = s`; and in the projection sidecar: calibration id and SHA-256, direction, plane frame name, plane unit (or null = projective), distortion domain, pixel convention, valid-region id, residuals, reason code for every non-projection.
