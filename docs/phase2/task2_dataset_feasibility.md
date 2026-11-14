# Phase 2 Task 2 - Dataset Feasibility and Acquisition Audit

**Status:** COMPLETED_WITH_LIMITATIONS  
**Date:** 11 November 2026  
**Decision:** CONDITIONALLY_FEASIBLE  
**Scope:** Acquisition planning only. No dataset was downloaded.

## Candidate acquisition summary

| Candidate | Acquisition path | Access/licensing finding | Raw video and annotations | Planning conclusion |
|---|---|---|---|---|
| PIE | Use the official PIE site for videos and the official GitHub repository for annotations/tooling. | Official materials describe an MIT-labelled release; verify the exact terms attached to videos, annotations, and derived data before redistribution. | Official page confirms onboard video, pedestrian/vehicle annotations, and infrastructure/context labels. | Feasible as the primary candidate after terms and calibration review. |
| inD | Request/download through the official provider page and follow its academic/research terms. | Official overview describes free academic/research use; provider terms remain authoritative. | Trajectory CSVs, recording/track metadata, background images, and maps are confirmed. Public raw-video availability is not confirmed. | Feasible as a geometry/trajectory supplement; conditionally feasible for video work. |
| TITAN | Submit the official request using university affiliation and university email. | Non-commercial use; university affiliation and request approval are required. | Official release describes image clips, annotations with object track IDs, splits, and synchronized IMU data. | Conditionally feasible and potentially strong, but access is a gating dependency. |

## Acquisition requirements

### PIE

1. Review the current official dataset page and repository license files.
2. Confirm whether the intended video and annotation subsets can be used for
   academic research and publication.
3. Download a small approved subset first; do not mirror the approximately
   74 GB repository-reported video collection before storage is confirmed.
4. Preserve the release URL, access date, license text, archive checksums,
   and subset manifest.
5. Inspect the annotation format and camera-parameter files before defining
   world-coordinate fields.

### inD

1. Review the official provider terms and complete any required academic
   request.
2. Start with metadata, one recording, trajectory CSVs, background image, and
   map assets.
3. Confirm whether raw drone video is available to this project. The public
   overview should not be treated as proof that it is.
4. Preserve recording metadata and map versions with the subset manifest.
5. Keep inD in a separate split namespace from onboard-video datasets.

### TITAN

1. Use a university email and submit the official non-commercial request.
2. Wait for approval before downloading any archive.
3. Begin with one clip, its annotation CSV, split metadata, and synchronized
   sensor record.
4. Confirm the permitted publication and redistribution boundaries.
5. Record the release version, archive checksums, image dimensions, and
   sensor timestamp policy.

## Formats, frequencies, and geometry

| Dataset | Formats confirmed by official materials | Sampling | Geometry status |
|---|---|---|---|
| PIE | MP4 video and repository annotation data; camera-parameter files are present in the official repository. | Video described at 30 FPS. Exact annotation sampling and identity semantics must be checked in the selected release. | Camera parameters are available, but a universal metric world-coordinate contract is not assumed. |
| inD | Per-recording background image, recording metadata CSV, tracks metadata CSV, trajectories CSV, and Lanelet2/OpenDRIVE maps. | Frame-based trajectories; exact frequency must be read from each recording metadata file. | High-positioning-accuracy trajectories and maps are confirmed; this is not evidence of raw-video calibration. |
| TITAN | Image archives, annotation CSVs, split text files, and synchronized sensor CSVs. | Official release describes 10 Hz annotation and 1920x1200 images. | Ego IMU/odometry is available; agent world coordinates and camera calibration require explicit verification. |

## Storage planning

The following values distinguish published figures from planning estimates.

| Dataset | Published or provider-reported size | Local planning recommendation |
|---|---|---|
| PIE | Approximately 74 GB for all video clips is reported by the official repository README. | Keep the full archive on Google Drive or another backed-up volume. Start locally with a small subset and annotations only. Allow additional extracted-frame space if frames are materialized. |
| inD | No single archive size was used here because the provider page emphasizes trajectory/map assets and the exact release package was not downloaded. | Reserve a few GB for an initial trajectory/map subset; measure the approved archive before full acquisition. |
| TITAN | The official overview gives image dimensions and counts but no single total archive size in the reviewed material. | Treat total size as unknown until access is granted. Download one clip first and measure compressed and extracted sizes before scaling. |

These are not guarantees of final storage usage. Archive size, extraction
policy, duplicate frames, and generated features must be recorded in a
manifest. BDD100K is not part of the initial acquisition plan; its official
download documentation reports approximately 1.8 TB for videos, making it
unsuitable for the first local experiment on a 16 GB RAM laptop.

## Hardware feasibility

The known environment is Windows 11, Python 3.9.25, the `pcg-baseline`
Conda environment, an Intel i5-1235U, 16 GB RAM, and Intel Iris Xe graphics.
The practical plan is:

- use local storage for metadata, annotations, small video subsets, and
  deterministic preprocessing;
- use Google Drive and Colab Pro/Pro+ for larger archives or GPU-intensive
  extraction/model experiments;
- process videos in streaming chunks rather than loading complete clips into
  RAM;
- materialize only the frames required for a controlled experiment;
- keep raw archives outside Git and store only manifests, scripts, schemas,
  and small fixtures in the repository.

Expected bottlenecks are disk capacity, video decoding time, extracted-frame
duplication, and GPU memory during future model training. The current task
does not install a decoder, download data, or benchmark throughput.

## Proposed directory structure

Create these directories only when acquisition is approved:

```text
data/
  raw/
    pie/
    ind/
    titan/
  annotations/
    pie/
    ind/
    titan/
  calibration/
    pie/
    ind/
    titan/
  maps/
    ind/
  interim/
  processed/
  splits/
  manifests/
```

Raw video, downloaded archives, licensed annotations, and generated large
features should remain outside Git. The repository should contain a manifest
with source URL, release/version, acquisition date, license reference,
relative local path, byte size, checksum, and processing status.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| License terms differ between code, videos, and annotations. | Archive the exact terms and obtain approval before redistribution or publication. |
| Raw video is unavailable for inD/rounD. | Keep them as trajectory/map supplements and do not claim video parity. |
| Coordinate systems are incompatible. | Maintain dataset-specific coordinate adapters and never merge units without a documented transform. |
| Access request is denied or delayed. | Use PIE as the primary candidate and keep TITAN as an alternative, not a dependency for all work. |
| Storage exceeds laptop capacity. | Acquire one subset, stream processing, and use Google Drive/Colab for approved large archives. |
| Labels and identity semantics differ. | Define a mapping table and dataset-specific evaluation splits before training. |

## Exact next steps before downloading

1. Approve PIE as the provisional primary and choose whether inD or TITAN is
   the first supplement.
2. Review and archive the current official license/terms for the selected
   releases.
3. Confirm storage capacity and backup location.
4. Submit the TITAN request if TITAN remains in scope.
5. Confirm the required PIE video and annotation subset.
6. Create a manifest template and an acquisition log.
7. Download only a small, approved pilot subset.
8. Validate checksums, file formats, frame timing, and coordinate metadata.

## Final acquisition decision

**CONDITIONALLY_FEASIBLE.** PIE can reasonably support the proposed first
video experiment, but acquisition is conditional on license confirmation,
storage approval, and calibration/coordinate review. inD is feasible as a
trajectory/map supplement, not yet as a confirmed raw-video source. TITAN is
feasible only after university access approval and non-commercial terms are
accepted. No download should begin until the user approves the primary and
supplementary dataset choices.

## Official sources

- PIE: <https://www.data.nvision2.eecs.yorku.ca/PIE_dataset/>
- PIE repository: <https://github.com/aras62/PIE>
- JAAD repository: <https://github.com/ykotseruba/JAAD>
- TITAN: <https://usa.honda-ri.com/titan>
- inD: <https://levelxdata.com/ind-dataset/>
- rounD: <https://levelxdata.com/round-dataset/>
- BDD100K download documentation: <https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst>

