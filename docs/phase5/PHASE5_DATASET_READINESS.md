# Phase 5 Dataset Readiness

**Date:** 15 November 2026 (revised for P5.3: H = 6 feasibility rows, section 3). **Status:** `PILOT_ONLY — SCENE-DIVERSE DATA NOT YET AVAILABLE`.
Evidence: repository inspection at commit `5d58ab7`, local file search, Phase 2 Task 1/2 dataset documents, Phase 3/4 manifests. Nothing below claims an acquisition that has not happened.

## 1. Readiness table
| Dataset | Acquisition status | Video | Annotations | Camera / geometry | Scenes / groups | Videos | Usable agent categories | Processing status | Licence / access | Phase 5 suitability | Remaining work |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PIE pilot** `set01/video_0004` | Video and annotation XML in the user's Google Drive (not on this machine); Phase 3 outputs copied locally to `PIE_pilot_inputs/` (gitignored) | 1 video, 1920×1080, 30 fps, 9,840 frames (≈ 328 s) | `video_0004_annt.xml` (Drive); used only by the Phase 3 Task 4 evaluator, which has **not** been run on it | Global `calibration_data.json` (K, D, height, pitch) only; no validated extrinsics or road plane (Task 1) | 1 video → **no independent groups** | 1 | pedestrian 23,661 / vehicle 34,678 / bicyclist 7,049 nodes (Task 3) | Phase 3 → Task 2 → Task 3 complete (image space) | MIT-labelled release per official sources; exact terms for videos/annotations/derived data still to be archived (Phase 2 Task 2) | **Development and smoke testing only**; cannot support generalization claims | none for development; see full PIE |
| **PIE full release** | **Not acquired** (only `video_0004` processed) | Official sources: > 6 h at 30 fps, ≈ 74 GB of clips | Pedestrian/vehicle boxes, identities, context and intention labels (official sources) | Same global calibration file; per-video applicability undocumented | Organised in sets (`set01`, …); set semantics as recording sessions to be verified | not counted here (to be read from the release index) | same classes as pilot (to be confirmed per video) | Phase 3 inference per video needed (Colab T4: pilot took 609 s wall for 9,840 frames), then Task 2 and Task 3 | as above; storage on Drive | **Primary candidate** for scene-diverse Phase 5 data | licence archive; choose subset; run Phase 3/Task 2/Task 3 per video; Phase 3 Task 4 GT evaluation |
| **HBS** (`Data/HBS/hbs.csv`, PCG baseline) | In repository (tracked) | none (trajectory CSV only) | 43,459 rows, 1,475 agents, 3,620 frames; pedestrians, cars/carts, bikes | Position/velocity units and calibration **unresolved** (Phase 2 Task 7) | **1 scene** | – | pedestrian, vehicle (bike excluded by PCG loader) | Stage 6 HBS smoke pipeline exists; not in Task 3 format | provenance/licence unresolved | Historical PCG baseline only; not image space, not comparable to PIE pixels | resolve units/provenance before any reuse |
| **inD** | Not acquired | public raw video not confirmed | trajectories + Lanelet2/OpenDRIVE maps (official) | drone, metric map coordinates | multiple intersections | – | pedestrians, bicyclists, cars, trucks/buses | none | research use; terms to review | Possible metric-trajectory supplement; **different coordinate system** (not mixable with PIE pixels) | approval, acquisition, separate adapter |
| **TITAN** | Access **not requested** | image clips (official) | track ids, splits, IMU (official) | moving GoPro; ego IMU only | multiple | – | pedestrians, vehicles | none | non-commercial, request required | Alternative moving-camera source pending access | request access |

## 2. Scene diversity and independence
- The only processed data is **one video of one drive** (≈ 5.5 min). Windows from it share the scene, camera, lighting and ego trajectory; their tracks overlap in time. There are **no independent groups** for a train/validation/test split.
- Windows from the same track and from temporally adjacent tracks are strongly correlated; the effective sample size is closer to the number of distinct tracks than to the number of windows.
- Splitting this video into chronological blocks can exercise the split code, but such blocks are **not independent** (same scene and drive) and must be labelled as a smoke-test split, never as generalization evidence.

## 3. Window feasibility on the pilot (measured at kickoff)
Counted from the pilot Task 3 `nodes.npz` (local): a window requires a node of the track at every sampled frame (no interpolation); stride = step. Exploratory counts, not a dataset:

| Setting (step s, H, F) | Pedestrian windows (tracks) | Vehicle windows (tracks) | Bicyclist windows (tracks) |
|---|---|---|---|
| s=1 (30 Hz), H=15, F=45 (0.5 s / 1.5 s) | 6,573 (79) | 13,803 (119) | 1,824 (23) |
| s=1, H=15, F=30 (0.5 s / 1.0 s) | 8,245 (116) | 16,145 (155) | 2,281 (29) |
| ~~s=3 (10 Hz), H=5, F=15~~ — **superseded** (kickoff provisional H; its "0.5 s" label counted H × Δt, the first-to-last span is 0.4 s) | ~~2,482 (88)~~ | ~~5,034 (132)~~ | ~~684 (28)~~ |
| **s=3 (10 Hz), H=6, F=15 (0.5 s first-to-last / 1.5 s) — approved; per-track anchor phase (kickoff convention)** — recalculated estimate, P5.3 | 2,380 (83) | 4,893 (124) | 653 (28) |
| **same, global anchor grid `frame % 3 == 0` (adopted by the P5.3 implementation)** — recalculated estimate, P5.3 | 2,327 (84) | 4,807 (121) | 646 (25) |
| s=3, H=10, F=15 (1.0 s / 1.5 s) | 2,019 (70) | 4,404 (102) | 545 (22) |
| s=5 (6 Hz), H=8, F=12 (1.33 s / 2.0 s) | 999 (47) | 2,266 (68) | 261 (13) |
| s=15 (2 Hz), H=6, F=6 (PCG-like 3 s / 3 s) | 211 (27) | 590 (37) | 50 (6) |

**P5.3 reconciliation note.** The two bold H = 6 rows were recalculated on 15 November 2026 from the regenerated Task 3 `nodes.npz` (byte-identical to the kickoff input) with the same rule (node of the track at every sampled frame, no interpolation). They are **estimates**, not accepted-window counts; the actual implementation count, with rejection reasons, is in `PHASE5_TASK3_WINDOW_GENERATION.md`. Vehicle and bicyclist columns are what those types *would* yield as targets; P5.3 predicts pedestrians only. The other rows are exploratory alternatives, unchanged from the kickoff; their time labels use the kickoff's H × Δt convention (e.g. H=10 at 10 Hz spans 0.9 s first-to-last), and they are not affected by the approved configuration.

Contiguous-run lengths (frames without a gap): pedestrian median 7 (p90 49, max 450); vehicle median 5; bicyclist median 6. Longer horizons rapidly reduce the number of distinct pedestrian tracks.

## 4. Target source (open decision)
Two options, both legitimate if labelled:
1. **Perceived tracks** (Task 3 node positions at future frames): end-to-end consistent with the deployed pipeline; targets inherit detector/tracker errors (Phase 3 detection/tracking quality on PIE is **not yet measured**; the Task 4 evaluator has not been run on the real XML).
2. **PIE ground-truth annotations** (annotation XML, parsed by the Phase 3 Task 4 parser): cleaner targets, but a separate adapter and a different identity space; annotations cover pedestrians and some vehicle types only.
A decision is required before P5.3 (kickoff report, Q2).

## 5. Prerequisites before final training
1. Archive PIE licence terms for videos, annotations and derived data.
2. Select a PIE subset with enough independent groups (proposal: several sets, split by set, after verifying set semantics) and record it in a dataset manifest with archive hashes.
3. Run Phase 3 perception on each selected video (Colab T4), then Task 2 and Task 3 locally; one run directory per video, each with its manifest.
4. Run the Phase 3 Task 4 evaluator on at least the pilot XML to quantify detection/tracking quality (affects target reliability).
5. Decide the target source (section 4).
Until 1–3 are complete, Phase 5 may build and test code on the pilot but **must not run final training or report generalization results**.
