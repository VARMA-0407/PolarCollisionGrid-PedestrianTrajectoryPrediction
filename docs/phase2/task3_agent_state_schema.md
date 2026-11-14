# Phase 2 Task 3 - Common Agent-State Representation

**Status:** COMPLETED_AND_VALIDATED  
**Date:** 11 November 2026  
**Scope:** Schema design only. No model or source implementation was changed.

## Recommendation

Use a **time-series observation table** as the canonical storage model:
one row per agent, sequence, and frame. Store dataset-specific source
identifiers and provenance alongside normalized fields. Build agent-level
track windows as a derived view for GATv2 and TCN. This avoids duplicating
values across nested track objects and makes missing observations, occlusion,
birth/death, and split boundaries explicit.

The schema is intentionally nullable for world-space and derived kinematic
fields. Missing values must remain missing; pixel coordinates must never be
labelled as metres. Image and world coordinate metadata are separate fields,
and world validity is conditional on non-null coordinates, a named coordinate
system, and an explicitly supported physical unit.

## Field contract

| Field | Type and unit | Required | Source | Online use | Leakage rule |
|---|---|---:|---|---|---|
| `track_id` | string | Yes | Tracker or annotation | Yes | Scope to sequence/scene. |
| `agent_id` | string or null | No | Dataset identity layer | Yes if stable | Never invent cross-scene identity. |
| `agent_class` | enum | Yes | Detector or annotation | Yes | Map labels explicitly. |
| `timestamp` | number, seconds | Yes | Video/metadata | Yes | Must be current or past time; seconds from the sequence time origin. |
| `frame_index` | integer | Yes | Video/annotation | Yes | Monotonic within sequence. |
| `sequence_id` | string | Yes | Dataset split builder | Yes | Used for grouping and purging. |
| `scene_id` | string | Yes | Dataset metadata | Yes | Never cross scene boundaries. |
| `bbox_xyxy` | exactly four floats, pixels, ordered `[x_min, y_min, x_max, y_max]` | Yes for image observations | Detector/annotation | Yes | Current-frame observation only. |
| `bbox_center` | exactly two floats, pixels, ordered `[x, y]` | Derived | Detector/annotation | Yes | Derived from current box. |
| `footpoint` | exactly two floats, pixels, ordered `[x, y]` | Optional | Derived from box/detector | Yes | Bottom-center ground-contact proxy unless the source defines another point. |
| `detection_confidence` | float [0,1] or null | Optional | Detector | Yes | Not a tracking score. |
| `image_x`, `image_y` | floats, pixels | Optional | Derived | Yes | Not metres. |
| `world_x`, `world_y` | floats when valid; null or absent when invalid | Optional | Calibration/annotation | Only when valid | Never fabricate from pixels; invalid world positions cannot retain numeric values. |
| `image_coordinate_system` | non-empty string or null | Optional | Camera/dataset metadata | Yes if image fields are used | Names the image reference frame. |
| `image_coordinate_unit` | `pixel` or null | Optional | Camera/dataset metadata | Yes if image fields are used | Image coordinates are never metres. |
| `world_coordinate_system` | non-empty string or null | Required when valid | Calibration/annotation metadata | Yes if valid | Must identify frame and transform. |
| `world_coordinate_unit` | `m`, `cm`, `mm`, or `ft`, or null | Required when valid | Calibration/annotation metadata | Yes if valid | Explicit physical unit; no `unknown` value is accepted when valid. |
| `world_position_valid` | boolean | Yes | Validation layer | Yes | False when calibration or value is unavailable. |
| `velocity_x`, `velocity_y` | floats or null | Optional | Annotation/derived | Yes if causal | Must declare kinematic space, position unit, and seconds as the time basis. |
| `speed` | float or null | Optional | Derived | Yes if causal | Same space/unit contract as velocity; never use future positions online. |
| `acceleration_x`, `acceleration_y` | floats or null | Optional | Derived | Yes if causal | Same space/unit contract; require declared smoothing/window. |
| `kinematic_coordinate_space` | `image`, `world`, or null | Required when kinematics are present | Derived/annotation metadata | Yes if causal | Prevents image and world measurements being treated as equivalent. |
| `kinematic_position_unit` | `pixel`, `m`, `cm`, `mm`, or `ft`, or null | Required when kinematics are present | Derived/annotation metadata | Yes if causal | Must match the declared kinematic space. |
| `kinematic_time_unit` | `s` or null | Required when kinematics are present | Dataset metadata | Yes if causal | All velocity/acceleration values use seconds. |
| `heading` | float or null, radians | Optional | Annotation/derived | Yes if valid | See heading semantics below. |
| `visibility` | number/string or null | Optional | Annotation/detector | Yes | Preserve source meaning; do not map unknown to visible. |
| `occlusion_status` | enum | Optional | Annotation/tracker | Yes | Do not convert unknown to visible. |
| `tracking_confidence` | float [0,1] or null | Optional | Tracker | Yes | Separate from detection confidence. |
| `observation_valid` | boolean | Yes | Validation layer | Yes | False for missing/invalid observations. |
| `interpolation_flag` | boolean | Yes | Preprocessing | Yes | Interpolated values are not raw observations. |
| `source_dataset` | enum/string | Yes | Ingestion | Metadata | Preserves provenance. |
| `source_label` | string/null | Optional | Ingestion | Metadata | Preserves original class. |

The companion JSON Schema is at
[`schemas/agent_state.schema.json`](schemas/agent_state.schema.json).

## Canonical record examples

### Observed pedestrian with valid image-space state

```json
{
  "track_id": "pie:seq_012:ped_0042",
  "agent_id": null,
  "agent_class": "pedestrian",
  "timestamp": 12.4,
  "frame_index": 372,
  "sequence_id": "pie:seq_012",
  "scene_id": "pie:route_03",
  "bbox_xyxy": [421.0, 188.0, 468.0, 326.0],
  "bbox_center": [444.5, 257.0],
  "footpoint": [444.5, 326.0],
  "detection_confidence": 0.94,
  "image_x": 444.5,
  "image_y": 326.0,
  "world_x": null,
  "world_y": null,
  "image_coordinate_system": "camera_image",
  "image_coordinate_unit": "pixel",
  "world_coordinate_system": null,
  "world_coordinate_unit": null,
  "world_position_valid": false,
  "velocity_x": null,
  "velocity_y": null,
  "speed": null,
  "acceleration_x": null,
  "acceleration_y": null,
  "heading": null,
  "visibility": "visible",
  "occlusion_status": "none",
  "tracking_confidence": 0.91,
  "observation_valid": true,
  "interpolation_flag": false,
  "source_dataset": "PIE",
  "source_label": "pedestrian"
}
```

### Vehicle with valid world-space trajectory

```json
{
  "track_id": "ind:recording_07:track_88",
  "agent_id": "88",
  "agent_class": "vehicle",
  "timestamp": 4.0,
  "frame_index": 80,
  "sequence_id": "ind:recording_07",
  "scene_id": "ind:location_01",
  "bbox_xyxy": null,
  "bbox_center": null,
  "footpoint": null,
  "detection_confidence": null,
  "image_x": null,
  "image_y": null,
  "world_x": 24.7,
  "world_y": 11.2,
  "image_coordinate_system": null,
  "image_coordinate_unit": null,
  "world_coordinate_system": "ind_local_map",
  "world_coordinate_unit": "m",
  "world_position_valid": true,
  "velocity_x": 3.1,
  "velocity_y": 0.2,
  "speed": 3.11,
  "acceleration_x": null,
  "acceleration_y": null,
  "kinematic_coordinate_space": "world",
  "kinematic_position_unit": "m",
  "kinematic_time_unit": "s",
  "heading": 0.06,
  "visibility": null,
  "occlusion_status": "unknown",
  "tracking_confidence": null,
  "observation_valid": true,
  "interpolation_flag": false,
  "source_dataset": "inD",
  "source_label": "car"
}
```

### Record without world coordinates

Set `world_position_valid` to `false`, use `null` for world values, and
retain the image-space values with `image_coordinate_system` and
`image_coordinate_unit: "pixel"`. The world metadata may remain null.

### Missing or occluded observation

Use the same identity and sequence metadata, set `observation_valid` to
`false` when no measurement is available, and do not silently copy the
previous position. An actual observation is a value directly supplied by the
detector or dataset annotation. An offline interpolated observation fills a
gap using a declared offline procedure and sets `interpolation_flag` to
`true`. A missing/invalid observation has no trustworthy measurement. A
causally estimated state is computed only from observations at or before the
current timestamp and must not be labelled as an actual observation.
Future-observation-dependent interpolation is prohibited for online
inference.

## Coordinate, heading, and kinematic semantics

- `timestamp` is measured in seconds from the sequence time origin; the
  source frame index and source sampling frequency must be retained in
  provenance.
- `track_id` is unique only within its declared `sequence_id` and `scene_id`
  unless the source explicitly proves a wider scope.
- Bounding boxes use `[x_min, y_min, x_max, y_max]` in image pixels.
- `visibility` preserves the source dataset's numeric or categorical
  definition. `occlusion_status: "unknown"` is not equivalent to visible.
- `source_label` preserves the original dataset label before normalization;
  it is provenance, not an extra model prediction.
- Heading is in radians. The reference axis, positive rotation direction,
  and angle range must be recorded per dataset adapter. The common schema
  does not assume that `0` points east, north, image right, or vehicle
  forward, nor that positive rotation is clockwise or counter-clockwise.
  An adapter should normalize to a documented convention, preferably
  `[-pi, pi)`, before cross-dataset modelling.
- Kinematics are either image-space (`pixel/s`, `pixel/s^2`) or world-space
  (`m/s`, `cm/s`, `mm/s`, or `ft/s`, and corresponding squared-time
  acceleration). The declared `kinematic_coordinate_space`,
  `kinematic_position_unit`, and `kinematic_time_unit` must agree. Image
  and world values must never be concatenated as if they shared units.
- Kinematic metadata is required only when at least one velocity, speed, or
  acceleration value is a non-null number. If all such measurements are
  null or absent, the metadata may also be null or absent.
- When image-space bounding geometry or image coordinates are supplied,
  `image_coordinate_system` must be a non-empty string and
  `image_coordinate_unit` must be `pixel`.
- Bounding-box preprocessing must reject records where `x_min > x_max` or
  `y_min > y_max`; array length validation is handled by the schema, while
  ordering is a preprocessing invariant.
- Online inference may use only current/past observations and causal state.
  Offline interpolation and future targets belong in separate views.
- `world_position_valid: true` requires numeric `world_x` and `world_y`,
  named world metadata, and a physical unit. `world_position_valid: false`
  requires `world_x` and `world_y` to be null or absent.

## Dataset mapping

| Common field group | PIE | inD | TITAN |
|---|---|---|---|
| Identity | Annotation/tracker identity; verify exact scope in release. | `trackId` and recording scope. | `obj_track_id` and clip scope. |
| Class | Pedestrian/vehicle labels and source labels. | Car, truck/bus, pedestrian, bicyclist classes. | Person, four-wheeled and two-wheeled vehicle labels. |
| Time | Video frame/timestamp. | Recording frame and metadata. | Image/frame timestamp and 10 Hz annotation. |
| Image geometry | Bounding boxes and derived footpoints; image metadata is adapter-supplied. | Nullable unless video is available. | Image dimensions and bounding-box fields in annotation files. |
| World geometry | Nullable until camera/calibration semantics are verified. | Local map/trajectory coordinates with provider-defined semantics. | Nullable unless a verified transform is supplied; IMU is ego-motion. |
| Kinematics | Derived causally from observed positions when allowed; space/unit must be declared. | Track velocities/headings are available in trajectory files; verify units from metadata. | Derived from tracks or source fields; confirm exact fields per release. |
| Quality | Visibility/occlusion/confidence where released or detector-derived. | Dataset metadata plus derived validity flags. | Action/state and annotation quality plus tracker-derived confidence. |

This table is a mapping plan, not a claim that every field is present in
every dataset.

## Online inference and leakage prevention

At time `t`, model inputs may use only observations with timestamps less
than or equal to `t`, plus causal tracker and filter state. Future ground
truth, future annotations, oracle best-of-K selection, and future conflict
labels are evaluation targets or offline diagnostics only. They must never
enter the online feature tensor.

Recommended derived views:

1. ingest raw dataset records;
2. normalize identities and class labels;
3. validate time and sequence boundaries;
4. create an observed-only causal feature view;
5. create a separate target view for future positions and conflict labels;
6. construct fixed-length agent windows for GATv2;
7. pass temporally ordered node features through the TCN.

GATv2 consumes the current interaction graph: node features can include
class, image/world position when valid, causal velocity, heading, confidence,
and observation masks. Edge features can include relative displacement,
relative velocity, distance, and validity masks. The TCN consumes each
agent's ordered history and masks; it must not receive target columns from
future frames.

## Design decisions

- IDs are scoped to a sequence or scene unless a dataset explicitly proves a
  global identity.
- Image and world coordinates are different fields with different units.
- `detection_confidence` and `tracking_confidence` are separate.
- Unknown, missing, invalid, and interpolated values are distinguishable.
- World values are nullable and require a validity flag.
- The schema stores provenance so dataset-specific assumptions remain visible.
- A time-series table is canonical; track windows are derived.

## Bounding-box preprocessing

The schema enforces the four-element shape of `bbox_xyxy`, but ordering is a
preprocessing invariant. The reusable
`tests/phase2/bounding_box_validation.py` helper accepts
`[x_min, y_min, x_max, y_max]` only when `x_min <= x_max` and
`y_min <= y_max`.

## Limitations

The schema cannot resolve missing calibration, licensing, or dataset
semantics. Those are acquisition and provenance decisions. It also does not
implement a detector, tracker, coordinate transform, GATv2 model, TCN, or
evaluation protocol.

## Validation evidence

The schema was validated with Python 3.9.25 in the `pcg-baseline` Conda
environment:

```text
python -c "import json; from jsonschema import Draft202012Validator; ..."
Draft2020-12 schema validation: PASS

python -m pytest tests\phase2\test_agent_state_schema.py -v
22 passed in 0.84s
```

The complete existing Phase 2 suite was also executed in that environment:

```text
python -m pytest tests\phase2 -v
collected 34 items
33 passed, 1 failed, 13 warnings in 2.82s
```

The one failure is the pre-existing
`test_ttc_occupancy_mask_collapses_agent_dimension` failure caused by the
uninitialized `TTC_mask` branch in `utils/Interaction.py`. That original PCG
source was not modified as part of this schema revision.
