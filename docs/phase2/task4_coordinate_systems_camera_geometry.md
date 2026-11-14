# Phase 2 Task 4 - Coordinate Systems and Camera Geometry Audit

**Status:** COMPLETED_WITH_LIMITATIONS  
**Date:** 11 November 2026  
**Scope:** Research, repository inspection, and technical design only. No
coordinate-transformation pipeline was implemented, no dataset was downloaded,
and no source, dataset, checkpoint, prediction artifact, manifest, or
canonical master file was modified.

## 1. Executive summary

The project must keep image, camera, and world/road-plane coordinates as
different reference systems. A pixel location is not a physical position, and
a homography is not automatically a metre-calibrated transform. Metric
coordinates require a verified coordinate frame, physical scale, calibration
evidence, and an explicitly validated projection region.

The official PIE GitHub repository was inspected directly. It contains:

```text
camera_params/calibration_data.json
camera_params/camera_params.png
```

The JSON contains image dimensions, an intrinsic matrix `K`, four distortion
values `D`, camera height in millimetres, and camera pitch in degrees. The
repository tree inspected here does not contain a road-plane homography,
rotation matrix, or translation vector. These files therefore establish
partial camera metadata, not a complete metric world-coordinate contract for
pedestrians or vehicles.

The provisional PIE recommendation from Tasks 1-3 remains provisional and has
not been approved for acquisition. inD remains a possible trajectory/map
supplement and TITAN remains a possible moving-camera alternative. Their
coordinate conventions must not be merged with PIE without dataset-specific
adapters and validation.

## 2. Coordinate-system definitions

| System | Meaning | Typical representation | Unit status |
|---|---|---|---|
| Image coordinates | Coordinates in a selected image raster, usually pixels with origin at the top-left. | `[u, v]`, bounding boxes, footpoints | Pixel units; affected by resize, crop, distortion, and frame convention. |
| Camera coordinates | A 3-D frame attached to the camera, commonly represented by optical-axis and sensor-axis conventions. | `[X_c, Y_c, Z_c]` | A physical length unit only after calibration and scale interpretation are established. |
| World/road-plane coordinates | A scene or road reference frame used for agent positions and motion. | `[X_w, Y_w]` on a plane, or `[X_w, Y_w, Z_w]` in 3-D | Metres or another declared physical unit only when independently established. |

These systems are related by a camera model, not by field names alone. The
Task 3 schema therefore keeps `image_coordinate_system`,
`image_coordinate_unit`, `world_coordinate_system`, and
`world_coordinate_unit` separate. `world_position_valid: true` must not be
used unless the world values and their frame/unit evidence are present.

### Pixel convention

The proposed default image convention is:

- origin `(0, 0)` at the top-left of the image raster;
- positive `u` to the right;
- positive `v` downward;
- boxes ordered `[x_min, y_min, x_max, y_max]`;
- centre `[ (x_min+x_max)/2, (y_min+y_max)/2 ]`;
- footpoint as a bottom-centre ground-contact proxy unless the source defines
  another point.

This is a project convention, not a claim that every dataset uses identical
box edge, pixel-centre, crop, or annotation conventions. Dataset adapters must
record the source convention and any one-half-pixel or resize transform.

The footpoint is generally more appropriate than the box centre for a
road-plane projection because it approximates the visible contact location.
It remains a noisy proxy: loose boxes, occlusion, truncation, slopes, and
annotation policy can move it away from the actual contact point.

### Heading convention

The Task 3 `heading` field is in radians, but its meaning is incomplete until
the dataset adapter defines the reference axis, positive rotation direction,
valid angle range, and coordinate space. For every adapter, these must be
recorded explicitly, for example: angle from positive world `X`, positive
counter-clockwise rotation, range `[-pi, pi)`, and `world` coordinate space.
An image-space heading derived from pixel motion is not interchangeable with
a world-space heading. The selected world-frame axes and their handedness
must therefore be part of the calibration/provenance record before heading
values are compared across datasets.

## 3. Image resizing, cropping, and lens distortion

For an original image point `[u, v]`, a resize with scale factors `s_x, s_y`
and crop origin `[c_x, c_y]` produces, under a simple pixel convention:

```text
u' = s_x (u - c_x)
v' = s_y (v - c_y)
```

More generally, represent the image-coordinate transformation in homogeneous
coordinates as:

```text
x' = A x
K' = A K
```

where `x = [u, v, 1]^T`, `x'` is the transformed point, and `A` is the
3x3 affine matrix for resize, crop, translation, and padding. For the simple
resize-plus-crop example:

```text
A = [[s_x, 0,   -s_x c_x],
     [0,   s_y, -s_y c_y],
     [0,   0,    1       ]]
```

Padding is represented by additional translation terms, such as
`[p_x, p_y]` after resizing. Thus, scaling focal lengths alone is sufficient
only for a pure resize with no crop, padding, aspect-ratio change, or other
pixel-coordinate translation. Principal-point offsets must also be
transformed, and non-uniform scaling changes `f_x` and `f_y` separately.

The relationship `K' = A K` assumes that `A` is a known 2-D affine
pixel-coordinate transform and that the same pixel-centre convention is used
before and after the operation. Implementations that define pixel centres at
half-integer coordinates, or that use library-specific resize alignment,
must preserve that convention in `A`; an undocumented half-pixel translation
can create systematic projection error. The same transform must be applied
to boxes, centres, footpoints, calibration intrinsics, and any homography
estimated in the original raster. If a network uses letterboxing, both
padding offsets and the resized dimensions must be retained.

If an existing homography maps original image coordinates to world-plane
coordinates,

```text
x_world ~ H x
```

then, after the image-coordinate transformation `x' = A x`, the equivalent
transformed image-to-world homography is:

```text
H' = H A^-1
```

For the reverse world-to-image mapping, if the original mapping is
`x ~ H x_world`, the transformed mapping is:

```text
H' = A H
```

The direction of every homography must be recorded explicitly. These updates
require `A` to be invertible; a lossy or rank-deficient coordinate operation
cannot be corrected by this formula without additional information.

Lens distortion means observed pixels do not follow the ideal pinhole model.
It is nonlinear, so a standard planar homography does not generally remove
distortion. The distortion coefficients must be interpreted using the
calibration library/model that produced them. The pipeline must choose one
consistent option:

1. undistort image points (and use the corresponding corrected intrinsics)
   before estimating or applying the homography; or
2. use a documented distortion-aware projection pipeline throughout.

A homography estimated directly from distorted pixels may approximate the
mapping within its fitted region, but it should not be treated as a
distortion-correct transform outside that region. Distortion effects become
more important away from the principal point and near image edges.

## 4. Camera calibration fundamentals

### 4.1 Intrinsics

The pinhole intrinsic matrix is commonly written:

```text
K = [[f_x, s, c_x],
     [  0, f_y, c_y],
     [  0, 0,   1]]
```

`f_x` and `f_y` are focal lengths in pixel units, `[c_x, c_y]` is the
principal point, and `s` is skew. Most modern cameras use negligible skew,
but it should not be silently discarded when the calibration source provides
it. Distortion parameters such as radial and tangential terms must retain
their ordering and model definition.

### 4.2 Extrinsics

Extrinsics relate camera and scene frames. A common convention is:

```text
X_c = R X_w + t
```

where `R` is a 3x3 rotation matrix and `t` is a 3x1 translation vector.
Other sources use camera-to-world transforms, a camera centre `C`, or
different axis directions. The report must record the direction of the
transform, handedness, units, and frame origin before use. `R` and `t` are
not interchangeable with a camera pitch angle alone.

### 4.3 Road-plane geometry

For a planar road surface, a plane can be expressed as:

```text
n^T X_c + d = 0
```

The plane assumption is useful for a locally flat road and a fixed camera.
It becomes unreliable on slopes, crowns, ramps, curbs, uneven terrain,
multi-level scenes, or when the camera moves. A single homography is valid
only for points on one plane under a fixed projective relationship. It does
not correctly map elevated body points or arbitrary 3-D points to a single
road plane.

Camera calibration identifies the camera model and pose parameters. A
road-plane homography identifies a 2-D projective mapping between one image
plane and one physical or arbitrary plane. A homography can be estimated
without a complete intrinsic/extrinsic calibration, but that does not make
the result a calibrated camera model or prove metric scale.

## 5. Homography mathematics

For homogeneous image point `x_image = [u, v, 1]^T` and homogeneous plane
point `x_world = [X, Y, 1]^T`:

```text
x_world ~ H x_image
```

`H` is a 3x3 matrix defined up to a non-zero scale factor. It therefore has
8 degrees of freedom. At least four non-collinear point correspondences are
needed for a basic estimate, and more correspondences are required for
residual checking and robustness.

OpenCV's `findHomography` is an appropriate implementation candidate when
its input convention, method, reprojection threshold, and output direction
are recorded. A robust method such as RANSAC can reject correspondence
outliers, but it does not repair bad labels, non-planarity, wrong point
ordering, or a wrong plane assumption.

Numerical requirements:

1. Normalize or use a numerically stable estimator for points with large
   coordinates.
2. Keep the source and destination point order fixed.
3. Record inlier masks, threshold, method, and calibration-region bounds.
4. Compute forward reprojection residuals on held-out correspondences.
5. Use `H^-1` only when it is well-conditioned and the inverse direction is
   explicitly documented.

The inverse mapping is:

```text
x_image ~ H^-1 x_world
```

when `H` is invertible. Points outside the calibration region may have large
errors even if the fitted inlier residual is small.

## 6. Metric-scale requirements

A homography alone determines only a projective plane up to its calibration
information and scale. The output may be an arbitrary plane coordinate
system, relative coordinates, or a normalized projective representation.
It must not be labelled metres merely because it has two numeric axes.

Metric output requires additional evidence, for example:

- surveyed road-plane points with known physical distances;
- a verified metric map or lane geometry;
- camera intrinsics and extrinsics with a defined world frame;
- a documented plane origin, axis direction, handedness, and unit;
- independent scale checks using known distances;
- held-out reprojection and ground-reference residuals below an approved
  threshold.

If these conditions are absent, the system should emit image-space
trajectories or explicitly named relative/projective coordinates with
`world_position_valid: false`. It must not fabricate metres from pixels,
camera height, GPS, or an arbitrary homography scale.

## 7. PIE calibration audit

### 7.1 Official repository evidence

The official repository tree was inspected at commit/ref `master` using the
GitHub repository API. It lists:

| Path | Format | Verified content |
|---|---|---|
| `camera_params/calibration_data.json` | JSON | `cam_height_mm`, `cam_pitch_deg`, `dim`, `K`, and `D`. |
| `camera_params/camera_params.png` | PNG | Camera/sensor specification image; not a machine-readable extrinsic or homography file. |

The verified JSON content is:

```json
{
  "cam_height_mm": 1270,
  "cam_pitch_deg": -10,
  "dim": [1920, 1080],
  "K": [
    [1004.8374471951423, 0.0, 960.1025514993675],
    [0.0, 1004.3912782107128, 573.5538287373604],
    [0.0, 0.0, 1.0]
  ],
  "D": [
    [-0.02748054291929438],
    [-0.007055051080370751],
    [-0.039625194298025156],
    [0.019310795479533783]
  ]
}
```

The repository README identifies `camera_params` as the location of camera
parameters. The inspected repository reference and tree do not list a homography JSON, a
full rotation matrix, or a translation vector. This is a finding about the
specific public repository reference inspected for this audit, not proof that
no supplementary calibration exists in another PIE release, archive, or
restricted distribution. `cam_height_mm` and
`cam_pitch_deg` are useful camera metadata, but their frame conventions and
relationship to a road plane must be verified before deriving a world
transform.

### 7.2 What PIE does and does not establish

| Question | Finding |
|---|---|
| Camera parameter files? | Yes: the two files above are present in the official repository. |
| Intrinsics? | Yes: `K` is supplied in the JSON. |
| Distortion? | Yes: `D` is supplied, but the exact distortion-model interpretation must be retained. |
| Image dimensions? | Yes: `[1920, 1080]` is supplied. |
| Camera height and pitch? | Yes, as named scalar fields; frame semantics require confirmation. |
| Full extrinsics? | Not found in the inspected repository tree. |
| Road-plane homography? | Not found in the inspected repository tree. |
| Metric agent world coordinates? | Not established by these files. |
| Ground-plane projection? | Potentially designable from additional scene correspondences and assumptions; not validated by this audit. |
| Additional missing information | World-frame origin/axes, complete pose convention, road-plane equation or surveyed correspondences, projection-region limits, and independent metric validation. |

The official repository is evidence that camera metadata exists; it is not
evidence that every annotated pedestrian or vehicle has a verified metric
road-plane position. PIE vehicle OBD/GPS or other vehicle telemetry, where
provided, describes the recording vehicle unless a separate agent-level
transform is proven. It must not be promoted to the world position of each
pedestrian or vehicle.

Official PIE sources:

- Repository: <https://github.com/aras62/PIE>
- README: <https://raw.githubusercontent.com/aras62/PIE/master/README.md>
- Calibration JSON:
  <https://github.com/aras62/PIE/blob/master/camera_params/calibration_data.json>
- Calibration image:
  <https://github.com/aras62/PIE/blob/master/camera_params/camera_params.png>
- Dataset page: <https://www.data.nvision2.eecs.yorku.ca/PIE_dataset/>

## 8. inD and TITAN comparison

These findings use the official source links recorded in the Task 1 and Task
2 research documents. The local repository contains no downloaded inD or
TITAN release, so their files were not independently inspected here.

| Dataset | Coordinate evidence | Camera relationship | Use in this project |
|---|---|---|---|
| inD | Official materials describe high-positioning-accuracy trajectories and Lanelet2/OpenDRIVE maps. | Drone-recorded trajectory/map data; this is not interchangeable with PIE onboard camera pixels. | Geometry and trajectory supplement after confirming units, frame origin, axes, and map alignment. |
| TITAN | Official release describes image annotations with `obj_track_id` and synchronized IMU/odometry. | Moving GoPro vehicle; IMU/odometry describes ego motion and is not automatically an agent world frame. | Heterogeneous moving-camera alternative after access, calibration, and coordinate validation. |

For inD, trajectory fields must be treated as dataset-defined coordinates
until the format documentation and release metadata establish unit and axis
semantics. For TITAN, synchronized accelerometer and angular-velocity data
must not be promoted to pedestrian/vehicle world positions without a
time-synchronized camera pose, calibration, and scene transform.

Official sources:

- inD: <https://levelxdata.com/ind-dataset/>
- inD paper: <https://ieeexplore.ieee.org/document/9304839>
- TITAN: <https://usa.honda-ri.com/titan>

## 9. Error sources and quality indicators

Major error sources include:

- wrong footpoint or contact-point estimate;
- bounding-box localization and truncation error;
- incorrect `K`, distortion model, or extrinsics;
- wrong road-plane assumption;
- road slope and non-planarity;
- camera motion or rolling-shutter effects;
- incorrect point correspondences;
- homography extrapolation outside the calibration region;
- timestamp mismatch between frames, annotations, and sensors;
- inconsistent resize, crop, padding, or coordinate origin.

The proposed audit metrics are:

| Indicator | What it measures | Required retention |
|---|---|---|
| Reprojection error | Pixel residual after forward projection. | Per-point residuals, aggregate statistic, inlier mask. |
| Ground-reference error | Error at surveyed or known road points. | Point identity, physical reference, units, residual. |
| Calibration residuals | Fit quality of intrinsics/distortion/extrinsics. | Calibration model, dataset split, residual distribution. |
| Valid projection region | Area where the transform was calibrated and tested. | Image polygon/road bounds and exclusion mask. |
| Invalid/unsupported mask | Points that cannot be justified for projection. | Per-agent/per-frame reason code. |
| Uncertainty | Sensitivity to point, calibration, and model errors. | Covariance, interval, or conservative quality score. |

Projected coordinates must carry a quality/confidence field or an equivalent
provenance record. A numeric output without uncertainty is not evidence that
the estimate is exact. Calibration-specific metadata should initially be
stored in a separate versioned calibration manifest/provenance sidecar,
rather than silently added to the Task 3 agent-state schema. The sidecar
must include at least:

- calibration ID and version;
- transform direction and coordinate-frame conventions;
- projection region and invalid mask;
- reprojection residuals and uncertainty;
- source image dimensions;
- resize, crop, padding, and pixel-centre transform.

Each projected observation must be traceable to the calibration manifest
version used to produce it.

## 10. Proposed project coordinate architecture

The architecture is compatible with the Task 3 schema without changing it:

1. Store raw image observations and source image dimensions.
2. Store the image coordinate-system identity and pixel unit.
3. Estimate footpoints with a method/version identifier and validity mask.
4. Apply an explicitly named dataset/camera transform only when calibration
   prerequisites are satisfied.
5. Store world frame identity, unit, transform version, projection region,
   residuals, and uncertainty.
6. Declare `world_position_valid: false` when calibration, correspondences,
   scale, or validity-region checks fail.
7. Declare kinematic coordinate space and unit separately from position.
8. Keep relative/projective trajectories distinct from metric trajectories.

Proposed calibration-manifest fields for future review, not Task 3 schema
changes:

```text
calibration_id
projection_method
projection_valid
projection_confidence
projection_reprojection_error
projection_region_id
source_image_width
source_image_height
resize_crop_transform
transform_direction
```

The manifest should also record the selected world-frame reference axis,
positive rotation direction, angle range, and whether each heading was
calculated in image or world space. No fields are being added to the Task 3
schema by this report.

When calibration is unavailable, the supported outputs are:

- image-space trajectory with pixel kinematics;
- explicitly named relative/projective trajectory;
- metric world-space trajectory only after verification.

## 11. Unresolved questions

1. What exact PIE release/version and camera sequence does the project intend
   to use?
2. Does PIE provide scene-specific camera pose, road-plane correspondences,
   or homographies outside the inspected repository tree?
3. What coordinate-frame convention and distortion model do `D`,
   `cam_height_mm`, and `cam_pitch_deg` use?
4. Are PIE annotations tied to one fixed camera configuration, and do all
   selected clips share the supplied calibration?
5. Which PIE points are valid ground-contact points under occlusion and
   truncation?
6. What physical reference establishes world-frame scale and axes?
7. Which inD release and map version will be used, and what are the exact
   trajectory units and axis directions?
8. Can TITAN camera pose be reconstructed from synchronized sensors with
   sufficient accuracy for the intended safety metrics?
9. What projection-region and residual thresholds will gate metric claims?
10. How will uncertainty propagate into TTC, PET, separation, and alert
    decisions?

## 12. Verified findings versus assumptions and recommendations

### Verified

- The repository is on branch `reproduction`.
- The project Python interpreter is Python 3.9.25 at
  `%USERPROFILE%\anaconda3\envs\pcg-baseline\python.exe`.
- The official PIE repository lists `camera_params/calibration_data.json`
  and `camera_params/camera_params.png`.
- The PIE JSON contains `K`, `D`, dimensions, camera height, and camera pitch.
- The inspected PIE tree does not list a homography or full `R`/`t` file.
- Task 3 separates image/world metadata and requires explicit world validity.
- No local inD or TITAN dataset files were found or downloaded.

### Assumptions

- Top-left image origin and downward-positive pixel `v` are the proposed
  project convention, not a universal dataset fact.
- A road-plane model may be useful for selected fixed-camera scenes, subject
  to scene-specific validation.
- `D` is a distortion parameter vector, but its exact model must be confirmed
  before use.

### Recommendations

- Keep PIE as a provisional candidate only until release terms and geometry
  semantics are approved.
- Build dataset-specific adapters and calibration manifests before mixing
  datasets.
- Gate metric safety metrics on calibration evidence, residual thresholds,
  scale checks, and valid-region masks.
- Preserve uncalibrated image/relative outputs instead of fabricating metres.

## 13. References and repository paths

Repository documents inspected:

- `docs/phase2/task1_dataset_research.md`
- `docs/phase2/task2_dataset_feasibility.md`
- `docs/phase2/task3_agent_state_schema.md`
- `docs/phase2/schemas/agent_state.schema.json`
- `docs/phase2/PHASE2_TASK1_TASK3_FINAL_REPORT.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/TASK7_HBS_COORDINATE_PROVENANCE_FINAL_AUDIT.md`
- `experiments/PCG_T4_5_Epoch_Run/reports/NEXT_IMPLEMENTATION_PHASE_ROADMAP.md`

External references:

- PIE repository and calibration files listed in Section 7.
- inD official page and paper listed in Section 8.
- TITAN official release listed in Section 8.
- OpenCV homography documentation:
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>

## Acceptance checklist

- [x] Image, camera, and world coordinates are distinguished.
- [x] Intrinsics and extrinsics are explained with conventions and units.
- [x] Homography assumptions, dimensions, degrees of freedom, estimation,
  inverse mapping, residuals, and RANSAC are explained.
- [x] Metric-scale requirements are explicit.
- [x] PIE calibration resources were inspected directly through the official
  repository API.
- [x] No claim states that PIE agent coordinates are metres.
- [x] Calibration uncertainty and invalid regions are addressed.
- [x] Proposed architecture is compatible with the Task 3 schema.
- [x] Unresolved questions are recorded.
- [x] No prohibited source or artifact was modified.
- [x] No dataset was downloaded or access application submitted.
- [x] Canonical master project file was not modified.

**Recommendation:** Task 4 is ready for review, not automatic approval. The
remaining approval gate is evidence for scene-specific pose, road-plane
geometry, metric scale, and uncertainty thresholds.
