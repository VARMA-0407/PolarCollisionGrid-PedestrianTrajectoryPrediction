# Phase 4 Calibration Evidence Checklist: PIE `set01/video_0004`

**Version:** 1.0.0 (Phase 4 Task 1). **Date:** 15 November 2026.
Each item is ticked only when the evidence exists **and** is recorded with its source. Nothing here may be filled from estimates, from agent tracks, or from the nominal values themselves.
Status legend: `[x]` verified in this audit · `[~]` partial (values present, semantics unverified) · `[ ]` missing.

## 1. Status at the end of Task 1

### Identity and raster
- [~] Video identity, 1920x1080, 30 fps, 9840 frames (user/Task 4 report; file not read locally)
- [x] PIE calibration `dim` = [1920, 1080]; PIE toolkit hard-codes 1920x1080
- [x] Phase 3 boxes are in original video pixels (code: OpenCV no-resize + Ultralytics `scale_boxes`)
- [ ] Actual video_0004 records show `coordinate_system = captured_1920x1080`, `image_width/height = 1920/1080`, boxes inside `[0,1920]x[0,1080]` (section A)
- [ ] `run_manifest.json` confirms `imgsz`, weights SHA-256, frames processed

### Intrinsics and distortion
- [x] `K` values re-verified byte-for-byte (`calibration_data.json`, SHA-256 `d6ee335b...2a92`)
- [x] `D` values re-verified (4x1)
- [ ] Distortion model documented (H-D1 fisheye vs H-D2 radial-tangential; section B)
- [ ] Calibration procedure / residuals for `K`, `D`
- [ ] Evidence that `K`, `D` apply to video_0004's readout (same camera unit, same crop)

### Extrinsics
- [~] Camera height 1270 mm (reference point/surface undocumented)
- [~] Camera pitch -10 deg (sign and reference axis undocumented)
- [ ] Roll, yaw, mounting offsets, `R`, `t`, transform direction
- [ ] Per-frame pitch/roll or proof that static pose suffices (moving vehicle)

### Ground plane and scale
- [ ] Road-plane equation or homography
- [ ] >= 4 non-collinear fit correspondences (target >= 8)
- [ ] >= 4 independent held-out correspondences
- [ ] Known physical distances / surveyed points / metric lane geometry
- [ ] Declared valid projection region (image polygon below horizon)
- [ ] Declared acceptance threshold (pixels for projective, metres for metric) **before** measuring

**Authorization now:** image space AUTHORIZED; relative/projective NOT AUTHORIZED (evidence pending); metric NOT AUTHORIZED.

## 2. Acquisition plan

### A. Read-only inspection of existing Phase 3 records (Colab, CPU, minutes)

Reads the Drive experiment folder; writes nothing there; runs no model. Expected checks: raster id, width/height, box ranges, number of boxes touching each border, footpoint distribution per class.

```python
import json, collections, pathlib
EXP = pathlib.Path("/content/drive/MyDrive/pedsafety_task1/PIE_Pilot/experiments/20261114T134036Z_pie_video_0004_yolo11n")
def rows(name):
    p = next(EXP.rglob(name))
    with open(p) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
man = json.loads(next(EXP.rglob("run_manifest.json")).read_text())
print("detector cfg:", man.get("config", {}).get("detector"))
print("video:", man.get("video"), "stats:", man.get("video_stats"))
stat = collections.Counter(); border = collections.Counter(); W = H = None
xmin = ymin = float("inf"); xmax = ymax = float("-inf")
for r in rows("tracked_agents.jsonl"):
    stat[(r["coordinate_system"], r["image_width"], r["image_height"], r["bbox_format"])] += 1
    W, H = r["image_width"], r["image_height"]
    x0, y0, x1, y1 = r["bbox_xyxy"]
    xmin, ymin, xmax, ymax = min(xmin, x0), min(ymin, y0), max(xmax, x1), max(ymax, y1)
    c = r["semantic_class"]; border[(c, "total")] += 1
    if y1 >= H - 1: border[(c, "bottom")] += 1
    if x0 <= 1: border[(c, "left")] += 1
    if x1 >= W - 1: border[(c, "right")] += 1
print("raster/format:", dict(stat))
print("box extent:", xmin, ymin, xmax, ymax)
print("border-touching:", dict(border))
a = next(rows("agent_states.jsonl"))
print({k: a[k] for k in ("image_coordinate_system", "image_coordinate_unit", "footpoint", "image_x", "image_y", "world_position_valid")})
```

Record the printed output verbatim in the Task 1 report addendum.

### B. Distortion-model test (Colab, reads frames from the existing `video_0004.mp4`; no inference)

1. Extract 5-10 frames spread over the video with straight scene structures near the image edges (building edges, poles, lane lines on straight road).
2. Undistort each frame twice: `cv2.fisheye.undistortImage(img, K, D, Knew=K)` (H-D1) and `cv2.undistort(img, K, D)` with `D` as `[k1,k2,p1,p2]` (H-D2).
3. On each result, mark >= 3 straight edges (manual polyline, >= 10 points each, at least one near an image corner) and compute the RMS perpendicular deviation from a fitted line.
4. Accept a model only if it is clearly straighter than the raw image and than the other model on every frame; record the residuals. If neither straightens lines, both are rejected and `D` is not used.

### C. Pitch and horizon check

1. On frames with a visible, level, straight road, locate the vanishing point of lane lines (after the accepted undistortion from B).
2. Compare its row with the nominal 396.5 px (pinhole, 10 deg down, zero roll). Report the per-frame deviation; convert to pitch via `atan((cy - v)/fy)`.
3. A consistent match supports the pitch sign; large frame-to-frame variation quantifies dynamic pitch and rules out a single static plane.

### D. Ground-plane correspondences for video_0004

Because the camera moves, survey-style control points cannot be placed in the scene after the fact. Acceptable independent sources, in order of preference:

| Source | How | Use |
|---|---|---|
| Authors' rig calibration (ground target / checkerboard with the same vehicle) | request from PIE authors | metric ego-plane (fit + held-out) |
| Location revisit survey | only if the GPS track (obd.xml) localises static features (crosswalk corners, stop lines, curb corners) and a site measurement is possible and safe | metric, frame-specific |
| Documented road-marking standard | only if the municipality's standard and the specific markings are confirmed (e.g. dashed-line length, crosswalk stripe width) | scale check, not sole calibration |
| Projective-only correspondences | corners of planar road markings marked by two annotators independently, plane coordinates arbitrary | projective frame (D) only |

For every correspondence record: `point_id`, frame index, image `(u, v)` in `captured_1920x1080` with pixel convention and distortion domain, plane `(X, Y)` with unit, source type, annotator, role (`fit`/`holdout`), and uncertainty. Agent tracks or detector boxes must never be control points (the `ControlPoint` class rejects such sources).

Design rules:
- at least 4 non-collinear fit points spread across the intended region (target 8+), no three nearly collinear, covering near and far range;
- at least 4 held-out points chosen **before** fitting, disjoint from fit points;
- all points on the same road surface (not curbs or sidewalks);
- for a moving camera, either all points from the same frame (per-frame H) or a demonstrated static pose for the frame range used.

### E. Estimation and validation protocol (when D exists)

1. Points in the chosen distortion domain; normalise coordinates (Hartley) or use OpenCV's internally normalised estimator.
2. `cv2.findHomography(img_pts, plane_pts, cv2.RANSAC, ransacReprojThreshold=<declared>)`, direction image_to_plane; record threshold, inlier mask, inlier count (require all fit points inliers or justify exclusions).
3. Residuals: forward plane-space residuals on fit and on held-out points; backward pixel residuals via `H^-1`; report RMS and max.
4. Conditioning: report `cond(H)` and the homogeneous denominator margin over the valid region; reject a region that approaches the vanishing line.
5. Valid region: image polygon containing the control points, strictly below the horizon with a margin; points outside are `projection_valid = false`.
6. Write a `CalibrationManifest` and run `evaluate_metric_eligibility`; only `manifest_eligible = true` plus an explicit config switch may promote values into `world_*`.

### F. Additional records to obtain (read-only)
- `video_0004_obd.xml` (if in the pilot): document fields, units, sampling rate, and whether pitch/roll exist and how they are referenced.
- PIE ICCV 2019 paper and supplementary: camera model and mounting description.
- PIE repository commit SHA for the calibration files (retry when the GitHub API is not rate-limited).

## 3. Exit gate for "metric road-plane authorized"

All of: section 1 boxes ticked; accepted distortion model (B); pose conventions confirmed (C, F); independent correspondences with held-out error below a threshold declared in advance (D, E); valid region; camera-motion policy stated; independent verification claim recorded. Until then Phase 4 runs in image space.
