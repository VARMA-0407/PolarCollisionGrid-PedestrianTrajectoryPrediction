# Phase 2 Task 1 - Dataset Research and Selection

**Status:** COMPLETED_WITH_LIMITATIONS  
**Date:** 11 November 2026  
**Scope:** Research only. No dataset was downloaded, licensed, or accepted.

## Executive summary

PIE is the strongest provisional primary dataset for the intended
video-to-trajectory workflow because its official description confirms
continuous onboard video, pedestrian and vehicle annotations, and 30 FPS
recording. It also provides vehicle telemetry, but that telemetry describes
the recording vehicle; it must not be interpreted as world coordinates for
all agents.

TITAN is a strong alternative for heterogeneous interactions in moving-camera
scenes. Its official release describes 700 clips, 10 Hz annotation, object
track IDs, pedestrians, four-wheeled vehicles, two-wheeled vehicles, and
synchronized IMU data. Access is non-commercial and requires university
affiliation and a university email.

inD is valuable for metric-like trajectories and maps, but its official
description is a drone trajectory dataset, not a directly interchangeable
roadside video dataset. The public format emphasizes tracks, recording
metadata, background images, and maps. It should therefore be treated as a
supplementary geometry/evaluation source unless raw video access is confirmed
directly with the provider.

JAAD and BDD100K are useful perception and behavior resources, but are less
complete fits for the proposed interaction-aware forecasting study. rounD is
useful for trajectory and map validation at roundabouts, but has the same
drone-versus-roadside-video incompatibility as inD.

These are recommendations for approval, not approved dataset decisions.

## Comparison table

| Dataset | Raw video | Pedestrian/vehicle data | IDs and temporal data | Geometry | Access and fit |
|---|---|---|---|---|---|
| PIE | Yes; official site describes over 6 hours of onboard HD video at 30 FPS. Repository README reports approximately 74 GB for all clips. | Pedestrian and vehicle boxes, infrastructure/context annotations, and pedestrian behavior/intention labels. | Frame-level annotations and track-style identities are available through the annotation tooling; verify exact identity semantics before integration. | Official repository includes camera parameters. Vehicle GPS/OBD telemetry is ego-vehicle data, not universal agent world coordinates. | Best provisional video-primary option; MIT-labelled release information is reported by the official site/repository, but the exact scope of each asset should be checked before redistribution. |
| JAAD | Yes; official site describes 346 onboard clips at 30 FPS. | Pedestrian boxes with occlusion tags, behavior/action labels, attributes, and traffic-scene annotations. | Per-frame labels and pedestrian identities in annotation files; forecasting suitability requires additional sequence construction. | No verified public world-coordinate or homography contract found in the cited overview. | Good behavior/perception supplement; less suitable as the sole interaction-forecasting dataset. Video clips are described as CC BY 4.0 and annotation repository as MIT. |
| TITAN | Yes; official site describes 700 GoPro clips, 1920x1200 images, 10-20 seconds each. | 395,770 person, 146,840 four-wheeled vehicle, and 102,774 two-wheeled vehicle annotations; 50 action/state labels. | Official format includes `obj_track_id`; annotations are sampled at 10 Hz. | Synchronized GoPro IMU/odometry is available. This is ego-motion data, not automatically a calibrated world frame for every agent. | Strong heterogeneous alternative. Non-commercial university access and request approval are required. |
| inD | Public trajectory files, maps, and background images are confirmed. Public raw video availability is not confirmed by the official overview. | Approximately 8,200 vehicles and 5,300 vulnerable road users; classes include cars, trucks/buses, pedestrians, and bicyclists. | Track IDs, frame indices, positions, velocities, headings, and metadata are the core format. | Strong supplement: high-positioning-accuracy trajectories plus Lanelet2/OpenDRIVE maps. Do not equate this with video calibration. | Academic/research use is described as free; provider terms must be reviewed before redistribution. Excellent geometry supplement, not a direct PIE replacement. |
| rounD | Public overview confirms recordings and trajectory data; public raw video availability is not confirmed. | Approximately 13,740 road users across three roundabouts, including vehicles, pedestrians, bicycles, and motorcycles. | Trajectory tracks and metadata are provided. | Lanelet2/OpenDRIVE maps and infrastructure-relative trajectories are available. | Useful roundabout evaluation supplement. Free academic/non-commercial use is described; application and terms must be followed. |
| BDD100K | Yes; official download documentation reports 100,000 videos and approximately 1.8 TB for videos. | Detection, segmentation, and multi-object tracking annotations; broad road-scene coverage. | Video and tracking tasks exist, but the target labels and sequence protocol differ from PIE/TITAN. | GPS/IMU package is reported at approximately 3.9 GB and describes the recording vehicle. It is not a pedestrian/vehicle world-coordinate map. | Strong perception pretraining source, expensive to store, and not a clean replacement for interaction-focused trajectory data. |

## Individual assessments

### PIE

**Confirmed:** The official PIE page describes over six hours of continuous
onboard footage at 30 FPS, more than 300,000 labelled frames, pedestrian and
vehicle spatial annotations, infrastructure annotations, and intention labels.
The official annotation repository documents camera-parameter files and
annotation tooling.

**Useful for:** Video ingestion, pedestrian/vehicle detection and tracking
experiments, behavior/intention prediction, and trajectory forecasting from
image-space observations.

**Not established by the sources:** A universal metric world-coordinate
system, a complete scene homography contract, or the validity of using ego
vehicle GPS as the position of each pedestrian or vehicle.

### JAAD

JAAD is a behavior-centric onboard video dataset. Its per-frame boxes,
occlusion tags, actions, and traffic context are useful for perception and
crossing behavior. It is less directly suited to long-horizon
pedestrian-vehicle trajectory forecasting because the official overview is
organized around short clips and behavior labels rather than a common metric
trajectory protocol.

### TITAN

TITAN is the strongest alternative when heterogeneous agents, actions, and
moving-camera context are more important than a simple acquisition process.
Its object-track IDs and 10 Hz annotation are useful for temporal modelling.
The IMU provides synchronized ego-motion information, but a separate
calibration and coordinate interpretation step is still required before
metric safety claims.

### inD and rounD

Both datasets are valuable for trajectory, map, road-user-class, and
infrastructure-relative analysis. They were recorded from drones and should
not be described as equivalent to onboard roadside video. Their trajectories
and maps may support geometry-aware evaluation or simulation, but a
video-based detector/tracker trained on PIE cannot be assumed to transfer
without an explicit domain and coordinate protocol.

### BDD100K

BDD100K provides exceptional scale and environmental diversity for
perception. The storage requirement and broad task design make it unsuitable
as the first end-to-end forecasting dataset for this project. Its GPS/IMU
signals are associated with the ego vehicle and must not be promoted to
agent-level world coordinates.

## Compatibility analysis

Combining datasets will introduce differences in:

- camera viewpoint (onboard versus drone);
- frame rate and annotation frequency (for example 30 FPS, 10 Hz, or
  dataset-specific trajectory sampling);
- coordinate systems and units;
- label taxonomies and object-class definitions;
- identity scope and track termination rules;
- occlusion and visibility conventions;
- licensing and redistribution obligations.

The safest strategy is one primary video dataset plus one explicitly
separate supplementary dataset. A single shared schema may normalize field
names, but it cannot make the underlying measurements equivalent.

## Provisional recommendation

1. **Primary candidate for approval:** PIE, subject to confirming the current
   download terms, annotation identity semantics, and the usable calibration
   contract.
2. **Supplementary candidate:** inD for trajectory and map/geometry
   experiments, subject to confirming whether raw video is needed and what
   provider terms permit.
3. **Alternative primary candidate:** TITAN if university access and
   non-commercial restrictions are acceptable and the moving-camera setting is
   preferred.
4. **Behavior/perception supplement:** JAAD.
5. **Perception-scale supplement:** BDD100K.
6. **Roundabout trajectory supplement:** rounD.

No selection is approved until the acquisition and licensing audit in Task 2
is reviewed.

## Official sources

The following links are the traceable sources used for the claims above. The
repository records the URLs; no download was initiated.

- PIE dataset page: <https://www.data.nvision2.eecs.yorku.ca/PIE_dataset/>
- PIE annotation repository and README: <https://github.com/aras62/PIE>
- JAAD dataset page: <https://data.nvision2.eecs.yorku.ca/JAAD_dataset/>
- JAAD annotation repository and license notes: <https://github.com/ykotseruba/JAAD>
- TITAN official release: <https://usa.honda-ri.com/titan>
- inD official dataset page: <https://levelxdata.com/ind-dataset/>
- inD paper: <https://ieeexplore.ieee.org/document/9304839>
- rounD official dataset page: <https://levelxdata.com/round-dataset/>
- rounD paper: <https://ieeexplore.ieee.org/document/9294728>
- BDD100K toolkit: <https://github.com/bdd100k/bdd100k>
- BDD100K download documentation: <https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst>

## Remaining uncertainties and approval requirements

- Confirm the current license and permitted use for every asset actually
  downloaded, not only the code repository.
- Confirm PIE and TITAN calibration/coordinate semantics from the release
  documentation before metric TTC or PET claims.
- Confirm whether inD or rounD raw videos can be obtained under provider
  terms; do not infer availability from trajectory files.
- Define a cross-dataset label and sampling policy before any combined
  training.
- Approve one primary and one supplementary dataset before acquisition.

