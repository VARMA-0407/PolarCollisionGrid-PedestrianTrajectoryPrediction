# Task 7 — HBS Coordinate and Data Provenance Final Audit

## Status

**PARTIAL — DOCUMENTED LIMITATIONS**

The available repository evidence supports a structural CSV inventory and
loader-behaviour audit. It does not establish physical coordinate units,
camera calibration, original video FPS, complete provenance, or licensing.

## Dataset inventory

- Dataset file: `Data/HBS/hbs.csv`
- Size: 4,180,343 bytes
- Rows: 43,459
- Columns in the CSV: `frame_id`, `agent_id`, `pos_x`, `pos_y`, `label`,
  `scene_id`, `timestamp`, `vel_x`, `vel_y`
- Unique agents: 1,475
- Frames: 3,620
- Scene IDs observed: one scene (`scene_id` 0)
- Timestamp examples: 0.0, 0.5, 1.0, 1.5, 2.0 seconds
- Repository-contained HBS auxiliary files: none found beyond `hbs.csv`
- Original video files: not present in the repository

The existing CSV inspection reports the historical class counts as 28,123
pedestrian observations/1,115 agents, 14,414 car observations/331 agents,
and 922 bike observations/29 agents. Counts are retained as audit evidence;
they are not reinterpreted as a new dataset result.

## Provenance findings

`README.md` links to the HBS dataset landing page and states that HBS contains
pedestrian and vehicle trajectories. The repository does not contain the
downloaded dataset documentation, collection methodology, source video
metadata, or a local citation/license record. The original publisher,
dataset expansion, camera identity, and collection conditions therefore
remain unverified from repository evidence.

## Coordinate-system findings

The CSV exposes `pos_x`, `pos_y`, `vel_x`, and `vel_y`, but variable names and
numeric magnitudes do not prove whether values are pixels, normalized values,
camera-relative values, or world coordinates. Axis directions, origin, units,
and velocity semantics are unresolved. The data must not be labelled metres
or metres per second.

## Calibration and homography

No calibration file, homography matrix, camera intrinsic/extrinsic file, or
ground-plane metadata was found in the repository. Metric world-coordinate
reconstruction and physically interpretable TTC/PET values are therefore not
established.

## Temporal sampling

`utils/DataLoader.py` sets `downsample_step = 1`, assumes `fps = 2`, and uses
`timestamp = 1 / fps`, giving a 0.5-second implementation interval. The CSV
timestamps independently show 0.5-second spacing in the inspected rows. This
supports the CSV sampling interval, but does not prove the original recording
FPS or video frame rate.

## Entity coverage and loader limitations

The loader recognizes pedestrian labels `pedestrian` and `ped`, and vehicle
labels `car`, `cart`, and `veh`. It creates separate pedestrian and vehicle
arrays. Bikes are present in the CSV inventory but are not included by the
loader's recognized labels, so the baseline does not cover bikes as modeled
agents. The historical baseline is preserved unchanged.

## Licensing status

The dataset license or terms of use were not found in repository files.
Software-repository citation information is not equivalent to dataset
licensing. Dataset access and reuse conditions remain unresolved.

## Evidence references

- `Data/HBS/hbs.csv`
- `README.md`
- `utils/DataLoader.py`
- `experiments/PCG_T4_5_Epoch_Run/logs/task7_csv_inspection.json`
- `experiments/PCG_T4_5_Epoch_Run/logs/task7_data_file_inventory.txt`
- `experiments/PCG_T4_5_Epoch_Run/logs/task7_dataloader_source.txt`
- `experiments/PCG_T4_5_Epoch_Run/reports/task7_hbs_coordinate_provenance_audit.json`

## Unresolved questions

1. What are the authoritative HBS dataset citation and license?
2. Are `pos_x` and `pos_y` pixel, normalized, camera-relative, or metric?
3. What are the coordinate origin and axis directions?
4. Are velocity columns measured or derived, and in which units?
5. What was the original video FPS and camera configuration?
6. Is a calibration or homography available from the dataset publisher?
7. Is one CSV scene equivalent to one camera sequence?

## Recommended next actions

Obtain the official HBS documentation and terms of use, preserve it as
external provenance evidence, confirm coordinate and velocity semantics with
the dataset publisher or source paper, and obtain calibration/homography
metadata if metric safety indicators are required. Until then, report
coordinate-dependent indicators as uncalibrated surrogates.

