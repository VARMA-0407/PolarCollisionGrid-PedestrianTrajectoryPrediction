# Phase 3 Stage 0: Implementation Readiness Audit

**Date:** 14 November 2026
**Branch / HEAD at audit time:** `reproduction` / `7a6f56ff513034924200423843d63896454e8cbc`
**Nature:** inspection and documentation only. No dependency was installed, no dataset downloaded, no application submitted, no model trained,
and no existing file modified. This report is the only file created (plus the new `docs/phase3/` directory that holds it).
**Status of every statement:** "verified" means I checked it in this audit; "documented" means it comes from an existing project document that I did
not re-derive; "unverified" means I could not check it.

## 1. Executive summary

- **The repository is intact.** Tracked files match HEAD; the 20-artifact manifest verifies 20/20; schema and HBS hashes are unchanged; 581 Task 7 tests and
  22 schema tests pass; the full Phase 2 suite shows the one known original `Interaction.py` failure and nothing else.
- **The environment is NOT ready for perception or model work.** `pcg-baseline` has CPU-only PyTorch 2.2.1, no CUDA, no NVIDIA GPU (only Intel Iris Xe),
  no OpenCV, no Ultralytics, no ByteTrack implementation, no PyTorch Geometric and no ffmpeg. It is adequate for the data/evaluation code already written.
- **No video dataset is available locally.** The only local data is HBS (a CSV, no video, one scene, units and calibration unknown). PIE is publicly
  downloadable; TITAN and inD need manual-review access requests. Nothing has been requested or downloaded.
- **Dataset acquisition cannot start yet,** because the Task 2 preconditions (approved primary/supplement choice, archived terms, storage and backup
  location, subset definition) are not done. These are decisions for you, not engineering work.
- **Phase 2 limitations are unresolved, not closed:** no corrected evaluator, no hidden-future-input test, no chronological purge, no real-dataset
  integration, no real calibration, licence/provenance open, and the Phase 2 exit and Stage 6.1 have not been formally approved by you.
- **A decision with large consequences is not yet made:** PIE and TITAN are moving-camera datasets. The calibration manifest in `pedsafety` assumes one
  static calibration, so metric velocity and TTC are not obtainable from them without ego-motion handling. This decides whether the project makes
  metric conflict claims at all.
- **Recommended first Phase 3 implementation task:** the chronological purged split with real dependency-interval verification plus a hidden-future-input
  test harness (Section 9). It needs no GPU, no download and no new dependency.

## 2. Documents read, and referenced files that do not exist

| Requested file | Result |
|---|---|
| `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md` | Exists (82,116 bytes, SHA-256 `2bbd8377562aa7f0af0802c0d355dafc8e5c5ca82d1a952eb4fbf6815ff5b8d2`). Read the Phase 2 sections, tracker and closure addendum. |
| `PROJECT_MASTER.md` | Exists (supporting draft). Read in an earlier session; unchanged (SHA prefix `5c20b0fc`). |
| `docs/phase2/task7_implementation.md` | Exists. Covers Stages 1-6.1. |
| `docs/phase2/PHASE2_TASK1_TASK3_FINAL_REPORT.md` | Exists. Plus `task1_dataset_research.md`, `task2_dataset_feasibility.md`, `task4_coordinate_systems_camera_geometry.md` (header read). |
| `experiments/PCG_T4_5_Epoch_Run/reports/NEXT_IMPLEMENTATION_PHASE_ROADMAP.md` | Exists, read in full. |
| `PHASE2_MASTER_CONSOLIDATION_CHECKLIST.md` (repository root) | **Does not exist at that path.** The checklist exists at `docs/phase2/PHASE2_MASTER_CONSOLIDATION_CHECKLIST.md`; I used that. |
| `docs/phase3/` | Did not exist; created for this report. |

Also unresolved from Stage 6.1: the canonical master named in that request (`...Handoff(2)(2).md`) was never found, and you have not yet confirmed
that the `(2).md` file is the intended one. The canonical-master update has therefore not happened.

## 3. Repository status (verified)

- **Branch/HEAD:** `reproduction`, `7a6f56ff...`; `origin/reproduction` is the same commit.
- **Local branches:** `reproduction` (current), `audit-fixes` (same commit, 0 ahead, so the TTC fix is not committed anywhere), `baseline-original` and
  `master` (both `0735a0f`, tracking `upstream/master`).
- **Remotes:** `origin` = `VARMA-0407/PolarCollisionGrid-PedestrianTrajectoryPrediction` (your fork); `upstream` = `Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction`.
- **Latest 10 commits:** `7a6f56f` Add PCG T4 five-epoch reproduction artifacts; `0735a0f`, `6da10f2`, `49e27e3` README updates; `d1c9bd8` create the directory
  if it doesn't exist; `9081e47`, `f5cb452` updated requirement; `ca4f38a` address a future warning; `1ade5c9` change import file directory; `70a80da` create utils folder.
- **Tracked files:** 0 modified, 0 deleted (`git diff HEAD` empty).
- **Untracked (26 entries):** the two master files and the pre-Task-10 backup, `PROJECT_MASTER.md`, `configs/`, `docs/`, `scripts/`, `tests/`, `pedsafety/`,
  `requirements-pedsafety.txt`, `requirements-dev.txt`, the Phase 2 reports/logs, `manifest_v2.json`, `ade_fde_reconstruction.json`.
  **All Phase 2 and Task 7 deliverables are uncommitted and outside the 20-artifact manifest.** A disk failure or an accidental clean would lose them.
- **Historical artifacts (verified):** `verify_phase2_manifest.py --manifest experiments/PCG_T4_5_Epoch_Run/manifest_v2.json` -> 20 declared, 20 checked,
  PASS. `utils/Interaction.py` SHA-256 `127f9d47...` unchanged. Checkpoints 6.5 MB, results 33 MB. `Store_Results/` holds the original CollisionGrid,
  SocialLSTM and VanillaLSTM models.
- **`pedsafety/`:** 21 modules (about 4,900 lines) from Stages 1-6.1. 581 tests pass in `tests/task7`; 22 schema tests pass; `tests/phase2` shows 33 passed and
  1 failed (the original `TTC_mask` defect, unchanged).
- **Schema:** SHA-256 `784c8ca9...` unchanged. **HBS CSV:** `eccd7478...` unchanged.

## 4. Environment status (verified, read-only)

| Item | Finding |
|---|---|
| OS / hardware | Windows 11 (10.0.26300); Intel i5-1235U; 15.7 GB RAM; Intel Iris Xe graphics (no NVIDIA GPU) |
| Active project environment | `pcg-baseline`, Python 3.9.25 |
| Other conda environments | `base`, `genai` and `Reverse_engineering_projects` (both Python 3.12.13, numpy only; no torch/OpenCV/Ultralytics found by `pip list`) |
| PyTorch / torchvision | 2.2.1+cpu / 0.17.1+cpu (torchaudio 2.2.1+cpu) |
| CUDA | **Not available** (`torch.cuda.is_available()` is False; `nvidia-smi` not found) |
| OpenCV (`cv2`) | **Not installed** |
| Ultralytics (YOLO) | **Not installed** |
| ByteTrack implementation | **Not installed** (none of `lap`, `filterpy`, `yolox`, `boxmot`, `supervision` found) |
| PyTorch Geometric (GATv2) | **Not installed.** GATv2 can also be written in plain PyTorch, which is a design decision for you. |
| ONNX / TensorRT / video decoders (`av`, `decord`) / ffmpeg | None installed |
| Present and declared | numpy 1.26.4, scipy 1.12.0, pandas 2.2.1, scikit-learn 1.4.1, matplotlib 3.8.3, jsonschema 4.25.1, pytest 8.4.2, Pillow 11.3.0 |
| Requirements files | `requirements.txt` (original PCG pins, untouched), `requirements-pedsafety.txt`, `requirements-dev.txt` (Stage 6.1). There is no `pyproject.toml` or `setup.py`. |
| Disk | D: 293 GB total, **257 GB free** (project lives here). C: 182 GB total, **32 GB free (83% used)**; the conda environments live on C:. |
| Colab / Drive | The master says earlier training used a Colab T4. I cannot check Colab or Drive from here: **unverified**. |

**Readiness verdict:** ready for CPU data/evaluation work (parsing, splitting, evaluator logic, tests, inference with the 6.5 MB checkpoints). Not ready for
detection, tracking, video decoding or any GPU training. The local machine cannot realistically meet a real-time target for a YOLO detector; Phase G
numbers must be reported for the hardware actually used.

## 5. Dataset readiness

Public pages were fetched read-only on 14 November 2026 to check current access terms. Page text is treated as data. Where the existing project documents
and the pages agree I say so; where I could not check, I say so.

| | PIE | TITAN | inD | HBS |
|---|---|---|---|---|
| **Access** | Public download (YorkU server or Google Drive); annotations in the GitHub repo. No request found. | Request form; university affiliation and university email required; review process and time not stated. | Request form; manual review; org address and intended use required. | Local `hbs.csv` only. |
| **Licence / provenance** | Repo licence is MIT. The page does not say whether the *video* is covered by that licence. | "Non-commercial usage"; cite the CVPR 2020 paper; nothing more specified on the page. | Free for academic/research use; non-commercial; **no distribution of the data**; derivatives only if abstract (models, annotations without data); cite the paper. | **Unresolved.** README links a TU Braunschweig repository record. I could not read it: the page returned only a bot-verification screen, which I did not try to bypass. Provenance and licence are **unverified**. |
| **Video** | Yes; about 74 GB total, six route-based sets, 30 FPS (documented). | Image sequences (PNG, `images_anonymized.tar`), 1920x1200, 700 clips of 10-20 s. Total size not stated. | Raw video **not mentioned** on the page. Trajectories, maps and one background photo per location. | None in the repo. |
| **Annotations** | Repo `annotations` folder; format not described on the page. | Per-clip CSV with track IDs, boxes and attributes; 10 Hz; train/val/test splits. | Georeferenced trajectories with metadata; Lanelet2/OpenDRIVE maps. | `hbs.csv`: 43,459 rows, 1,475 agents, one scene, 0.5 s spacing (verified in Stage 6). |
| **Coordinates / calibration** | `camera_params/calibration_data.json` exists (documented in Task 4: intrinsics, distortion, camera height, pitch). The page describes no road-plane homography or world frame. **Onboard moving camera.** | Page describes no calibration or coordinate frame; bbox reference frame not specified. IMU in m/s^2 and rad/s is ego-motion. **Moving camera.** | Georeferenced, but units and reference system are not stated on the page (a linked format PDF likely does). Static drone view. | Units, origin, axes, calibration and original FPS **unknown**. |
| **Storage** | ~74 GB video; start with a subset; extracted frames add more | Unknown until a clip is measured | "A few GB" planning estimate (Task 2); unconfirmed | 4 MB |
| **Suitability** | Best provisional video-primary option (documented); image-space forecasting yes; metric claims not established | Strong heterogeneous alternative; gated by access; same metric limitation | Geometry/trajectory supplement; cannot support video perception unless video access is confirmed | Not suitable as a video dataset or for scene-level generalisation; usable only as a legacy trajectory benchmark with unverified units |

**Access status summary:** no request submitted, nothing downloaded. PIE is the only candidate whose acquisition does not wait on a third-party decision.
The Task 2 conclusion stands: **CONDITIONALLY_FEASIBLE**, subject to licence confirmation, storage approval and calibration review.
BDD100K and JAAD are not part of the initial plan.

## 6. Architecture dependency analysis

The proposed chain is: video ingestion -> YOLO -> ByteTrack -> geometric preprocessing -> interaction graph (GATv2) -> TCN -> multimodal prediction ->
conflict prediction -> calibration.

| # | Component | Needs real dataset evidence? | Needs GPU? | Can start now? | Depends on | Notes |
|---|---|---|---|---|---|---|
| 1 | Dataset acquisition | n/a | no | **Blocked on your decisions** | licence review, storage, primary choice | Governs everything with real video |
| 2 | Corrected non-oracle evaluation | HBS suffices for the *logic*; real metrics need a coordinate decision | no (CPU inference with existing checkpoints) | **Yes** | split/purge policy; HBS coordinate decision for metric metrics | Entirely independent of YOLO/tracking |
| 3 | Video ingestion | Needs sample video for timing/format validation; logic can use synthetic video | no | Partly (needs `cv2`/ffmpeg, i.e. new dependencies) | #1 for real clips | Frame timing policy matters for dt |
| 4 | YOLO detection | Yes (class mapping, evaluation) | **Yes** for training; CPU inference is slow | Not meaningfully | #1, #3; GPU access | Ultralytics licence terms to be checked (unverified here) |
| 5 | ByteTrack tracking | Yes (ID stability) | no for tracking itself | Interface and tests with synthetic detections: yes | #4 outputs, #1 | Tracking quality cannot be judged without annotated video |
| 6 | Geometric preprocessing | **Largely built** (`pedsafety` Stages 3-4) | no | Yes for static-camera data; **open design question for moving camera** | calibration availability | Current manifest assumes one static calibration; a moving camera yields no metric velocity |
| 7 | GATv2 interaction model | Needs state representation and splits; can be unit-tested on synthetic graphs | training: yes | Architecture tests only | #2, #6 | PyTorch Geometric not installed |
| 8 | TCN temporal model | Same as 7 | training: yes | Architecture tests only | #2, #6 | Plain PyTorch is enough |
| 9 | Multimodal trajectory prediction | Yes (metrics, mode selection) | yes | No | #2, #7, #8 | Must not use oracle best-of-N selection |
| 10 | Conflict prediction and calibration | **Yes**: conflict labels, coordinate semantics | yes | No | #2, #6, #9 | Metric TTC/PET claims forbidden without verified metric coordinates |

**Independent of real data (can be built and tested with synthetic fixtures):** the corrected evaluator and its hidden-future tests, the chronological purge,
tracker interface/association logic, and model-architecture shape/gradient tests.
**Requires real dataset evidence:** detector quality, tracker ID stability, calibration and moving-camera handling, conflict-label definition, probability
calibration, and every performance number.

## 7. Phase 2 unresolved limitations (not marked complete)

| Item | Status |
|---|---|
| Corrected non-oracle evaluator | **Not implemented.** Task 6 remains REVISION REQUIRED. |
| Hidden-future-input test | **Not implemented** (no test shows future annotations cannot reach rollout or mode selection). |
| Chronological purge | **Not implemented.** Stage 5 splits whole groups only; HBS has one group so it cannot be split. The recorded safeguard "verify actual dependency-interval overlap" is only moot for whole-group splits. |
| Real PIE/inD/TITAN integration | **Not done.** No real file of these datasets has been read. |
| Calibration | **None real.** Only invented fixtures. PIE's published metadata has no road-plane homography. |
| Dataset licensing | **Open:** HBS provenance/licence unresolved; PIE video scope and TITAN/inD terms not yet archived. |
| HBS coordinates | Units, origin, axes, original FPS unresolved; must not be called metres. Single scene; bikes excluded by the baseline loader. |
| Original PCG defect | `TTC_mask` in `utils/Interaction.py` unfixed; the `audit-fixes` branch carries no commit. |
| Phase 2 exit | The audit returned `PHASE2_EXIT_APPROVED_WITH_LIMITATIONS`. **You have not recorded approval,** Stage 6.1 is awaiting your review, and the canonical master has not been updated. |

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No GPU locally; model work impossible on this machine | Decide the compute plan (Colab Pro/other) before Stage 3; keep Stage 1-2 CPU-only; report hardware with every timing |
| Moving-camera datasets cannot give metric velocity/TTC | Decide early: image-space/uncalibrated conflict surrogates only, or add ego-motion/homography work. Never label such outputs metric |
| Licence terms differ per asset (PIE video vs code; TITAN/inD non-commercial, no redistribution) | Archive the exact terms with access date before any download; keep raw data outside Git |
| HBS provenance still unknown | You open the HBS landing page manually and archive the metadata (the tool could not read it) |
| Deliverables exist only in the working tree | Decide a commit/backup strategy (new branch, so `reproduction` stays clean) |
| Dependency sprawl (OpenCV, Ultralytics, tracker libraries) could disturb the baseline environment | Use a separate environment for Phase 3; leave `pcg-baseline` as the reproducible baseline |
| Over-claiming from HBS | Keep the HBS smoke-test restrictions; report only unit-labelled results |
| Evaluator written against oracle behaviour by habit | Hidden-future-input tests first, before any evaluator code |
| Disk: C: 83% full | Install new environments on D: or free space first |
| Third-party licences for YOLO/ByteTrack code (e.g. copyleft terms) | Check before adopting; not verified in this audit |

## 9. Recommended Phase 3 execution order

1. **Stage 0 decisions** (this report): close Phase 2 formally, fix the master-file question, choose commit strategy, choose primary dataset and the
   moving-camera/metric policy, choose compute plan.
2. **Stage 1: corrected evaluation foundation** (CPU, no downloads): chronological purged split with actual dependency-interval verification, plus a
   hidden-future-input test harness. No model changes.
3. **Stage 2: isolated non-oracle evaluator** on HBS with the existing checkpoints, deterministic selection, oracle numbers labelled separately, unit-labelled
   only (HBS units unverified).
4. **Dataset track in parallel (needs you):** HBS provenance retrieval, PIE terms archived, then one small PIE pilot subset with checksums and a manifest.
5. **Stage 3: perception frontend** in a new environment: video ingestion, then YOLO, then ByteTrack, with detector and tracker evaluated separately.
6. **Stage 4: geometry on real data:** camera/ego-motion policy, footpoint transform, extension of `pedsafety` for the chosen dataset adapter.
7. **Stage 5: interaction model** (GATv2, TCN, multimodal decoder) with ablations, only after the evaluator and splits exist.
8. **Stage 6: conflict labels and calibration;** Stage 7: end-to-end timing on declared hardware.

## 10. First implementation task recommendation

**Phase 3 Stage 1: "Chronological purged split and hidden-future-input test harness."**
- Add to `pedsafety` (new modules and tests only): a within-group chronological splitter with a purge derived from the actual dependency intervals of each
  window (history, target, feature look-back), a verifier that fails on any dependency overlap across a partition boundary, and a harness that proves
  future annotations cannot influence history features or rollout inputs.
- Acceptance: tests on synthetic data, HBS read-only smoke (single scene, now splittable chronologically but labelled as non-independent), mutation checks,
  protected-file audit unchanged. No GPU, no download, no new dependency, no change to Stages 1-6.1.
- Why first: it closes two recorded Phase 2 gaps, is a prerequisite for every later evaluation, and does not wait on access, licences or hardware.

## 11. Required user decisions

1. **Phase 2 exit:** approve or reject `PHASE2_EXIT_APPROVED_WITH_LIMITATIONS`, and accept or reject Stage 6.1.
2. **Canonical master:** confirm that `...Handoff(2).md` is canonical (the `(2)(2)` file was not found), then authorise the master update using the checklist.
3. **Commit strategy:** whether and where to commit the untracked deliverables (a new branch is suggested; no commit has been made).
4. **Primary dataset:** confirm PIE as primary, and which supplement (inD or TITAN) if any; decide whether to submit the TITAN/inD requests yourself.
5. **Metric policy:** accept that moving-camera data support image-space/uncalibrated surrogates only, or authorise ego-motion/calibration work.
6. **HBS provenance:** retrieve and archive the TU Braunschweig landing-page metadata and licence manually.
7. **Compute:** confirm the GPU plan (Colab or other) and whether a separate Phase 3 environment on D: may be created later (installs need your approval).
8. **GATv2 implementation choice:** plain PyTorch or PyTorch Geometric.
9. **Stage 1 approval:** approve the task in Section 10, or choose another first task.

## 12. Errors and anomalies found during this audit

- The checklist path given in the request does not exist (correct path is under `docs/phase2/`).
- The HBS landing page could not be read (bot-verification screen); I did not attempt to bypass it. HBS provenance remains unverified.
- No NVIDIA GPU, CUDA, OpenCV, Ultralytics or tracker library: expected given Phase 2 scope, but it means Stage 3 is blocked locally.
- Pre-existing and unchanged: the original `TTC_mask` defect; the uncommitted deliverables.
- Nothing in this audit changed any existing file.
