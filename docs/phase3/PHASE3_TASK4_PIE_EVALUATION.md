# Phase 3 Task 4: PIE Ground-Truth Detection and Tracking Evaluator

**Date:** 14 November 2026. **Status:** evaluator implemented and unit-tested locally; **not yet run on the real PIE `set01/video_0004` outputs** (they are in Google Drive / Colab, not on this machine).
No PIE metric exists yet, and none is claimed. Phase 2 remains not formally closed.

## 1. What was built

| File | Purpose |
|---|---|
| `pedsafety/perception/pie_perception_evaluator.py` | parsing, matching, detection and tracking metrics, report writers |
| `scripts/evaluate_pie_perception.py` | command-line interface |
| `tests/phase3/test_task4_pie_evaluator.py` | 42 focused tests on synthetic XML/JSONL (no PIE data, no model, no video) |
| `scripts/build_colab_bundle.py` (modified, additive) | adds the Task 4 files to the allow-list; new `--name` option so bundles of different tasks never share a file name |
| `docs/phase3/PHASE3_TASK4_PIE_EVALUATION.md` | this document |

**Deviation from the requested path.** You asked for `pedsafety/evaluation/pie_perception_evaluator.py`. `pedsafety/evaluation.py` already exists (Stage 1, imported by `tests/phase3/test_stage1_evaluation_foundation.py`),
and a package named `pedsafety/evaluation/` would shadow it and break those imports. The evaluator therefore lives in `pedsafety/perception/`, next to the other perception code. Say if you prefer converting
`evaluation.py` into a package instead; that moves a Stage 1 file.

The evaluator reads the existing JSONL files only. It never runs YOLO, ByteTrack or any model, reads no video, downloads nothing, hardcodes no Colab path, and stays in image pixels. It imports only the standard library plus
(lazily) `scipy`/`numpy`, both already declared in `requirements.txt`. It needs Python 3.9+ (checked on 3.9 and 3.11).

## 2. Evaluation policy

- **Ground truth:** CVAT-style XML; per-box `<attribute name="id">` is the identity (never the CVAT track id). `outside=1` ignored; malformed, non-finite, zero- or negative-area boxes discarded with counters; GT boxes clipped to the image
  (size taken from the predictions) with a counter.
- **Classes:** pedestrian -> pedestrian; vehicle `car` -> car; vehicle `bicycle` -> bike; vehicle `train` (or any other/missing type) -> unsupported, counted, never mapped. Traffic lights, signs, crosswalks: counted, ignored.
  Evaluated classes: pedestrian, car, bike. Truck and bus **predictions** have no PIE counterpart: they are out of scope (counted, shown in mismatch diagnostics) and are not false positives of the evaluated classes; a car labelled truck shows as a missed car plus a class-mismatch pair.
- **Occlusion:** partially occluded boxes retained (requirement). Fully occluded boxes are retained by default and `--exclude-fully-occluded` drops them; recall is always reported by occlusion level (pedestrian `occlusion` 0/1/2 -> none/partial/full; others use the CVAT `occluded` flag).
  Interpolated boxes (`keyframe=0`) are retained and counted.
- **Evaluated frames:** only frames where the annotation file has at least one valid non-outside box of any label (`annotated_any`, primary) **and** that lie inside the frame range the predictions cover (union of the two JSONL ranges and, if given, the run manifest's range).
  Frames beyond coverage are dropped and counted, never scored as misses; predictions in unannotated frames are never false positives. A stricter set (frames with at least one supported GT box, `supported_gt_only`) is reported as a sensitivity result. This is a limitation: the annotation file does not state that unannotated frames are empty.
- **Detection:** per frame, class-aware one-to-one assignment (most pairs first, then highest IoU; IoU >= threshold inclusive) using `scipy.optimize.linear_sum_assignment`, or an exact pure-Python Hungarian solver if SciPy is absent (a test checks both give identical reports). TP, FP, FN, precision, recall, F1, mean IoU of matches, overall and per class,
  at each threshold (default 0.50 and 0.75). Fixed operating point (every detection in the file, optional `--min-confidence`); **no AP or mAP** is computed.
- **Tracking** (same GT identities and frame set, per threshold): CLEAR-MOT with continuity (a valid previous correspondence is kept first), FP, FN, ID switches (change of predicted identity against each GT identity's last matched predicted identity; the memory persists across unmatched frames and annotation gaps),
  MOTA = 1 - (FN + FP + IDSW) / GT boxes, MOTP reported as the mean IoU of matches (higher is better). IDF1 uses **one global identity assignment** (per connected component of the overlap graph) maximising matched frames: IDTP, IDFP, IDFN, ID precision/recall/F1. GT boxes without an `id` or with a duplicate identity in one frame are excluded from tracking only (counted).
- **Tracked rows:** primary = `tracking_validity` true; diagnostics = detection-backed only (`is_predicted` false) and all rows. `is_predicted` rows are counted and reported. In the Task 1 exporter `is_predicted` implies `tracking_validity` false, so the primary variant excludes them and the all-rows diagnostic includes them.
- **Undefined values:** null with an explanation (no GT -> MOTA, ID recall and IDF1 are null; no prediction -> precision is null).
- **Extras:** a frame-alignment sensitivity check (predictions shifted by -2..+2 frames) is written to the detection report as a sanity check only; `--frame-offset` exists but defaults to 0.

## 3. Outputs

`pie_detection_metrics.json`, `pie_tracking_metrics.json`, `pie_evaluation_summary.md`, `pie_evaluation_manifest.json` in a new or empty output directory (an existing non-empty directory is refused, never overwritten).
The manifest records evaluator version, UTC time, input paths and SHA-256 (annotations, detections, tracked agents, optional run manifest), video id, annotation frame range, evaluated frame counts, IoU thresholds, class mapping, occlusion policy,
unsupported-class policy, matching method, software versions, the prediction run's info, and the limitations. Metric files and the Markdown contain no timestamps, so repeated runs are byte-identical (a test checks this).

## 4. Tests

```
D:\envs\pedsafety-perception\python.exe -m pytest tests/phase3/test_task4_pie_evaluator.py -q    -> 42 passed
%USERPROFILE%\anaconda3\envs\pcg-baseline\python.exe -m pytest tests/phase3/test_task4_pie_evaluator.py -q    -> 42 passed (Python 3.9)
D:\envs\pedsafety-perception\python.exe -m pytest tests/phase3 tests/task7 --ignore=tests/phase3/test_measurement_helpers.py -q -rs    -> 840 passed, 1 skipped in 148.84s
%USERPROFILE%\anaconda3\envs\pcg-baseline\python.exe -m pytest tests/phase3 tests/task7 tests/phase2 -q -rs    -> 1 failed, 852 passed, 5 skipped in 161.94s
```

The 840 and 852 are different runs in different environments and are not additive; the 42 new tests are inside both (the previous totals were 798 and 810). The one failure is the documented original `TTC_mask` defect in `utils/Interaction.py`; the 5 skips are Task 1 test files that need OpenCV or Ultralytics (absent from `pcg-baseline` by design); the 1 skip in the perception environment is the optional real-weights test.
Coverage: pedestrian and vehicle-subtype parsing, per-box identity, `outside=1`, partial and full occlusion, train exclusion and its diagnostics, ignored labels, malformed/zero-area/clipped boxes, IoU, one-to-one class-aware matching, solver equivalence, precision/recall/F1 arithmetic, IoU thresholds, class mismatch,
truck/bus scoping, occlusion stratification, empty input, no-GT frames, coverage limits, frame offset and alignment sensitivity, the prediction loader, global IDF1, ID switches (including across gaps and swaps), `is_predicted`/`tracking_validity` handling, missing/duplicate GT ids, determinism, end-to-end reports and manifest, CLI exit codes.
**Mutation checks:** 9 safeguards were broken one at a time in a scratch copy (framewise IDF1, switches forgetting gaps, `outside` not ignored, CVAT track id as identity, train mapped to car, no coverage restriction, truck counted as FP, partial occlusion dropped, wrong F1 formula); all 9 were caught.
Two defects were found and fixed while testing: F1 returned null instead of 0.0 when nothing matched, and IDF1 read 0.0 instead of null when there was no ground truth.

**Schema check on real pipeline output:** the CLI was also run on the actual `detections.jsonl` and `tracked_agents.jsonl` from the local synthetic Task 1 run against a PIE-style XML generated from the synthetic scene. It parsed them and produced all four files. That is a format check only (the predictions replay the ground truth), not a result.

## 5. Task 4 exit criteria

| Criterion | Status |
|---|---|
| Evaluator parses official PIE XML | implemented and tested on synthetic XML in the confirmed structure; **real `video_0004_annt.xml` not yet parsed** |
| Detection metrics from the actual `detections.jsonl` | **pending the Colab run** |
| Tracking metrics from the actual `tracked_agents.jsonl` | **pending the Colab run** |
| Reports and manifest saved | implemented and tested; pending the real run |
| Focused tests pass | yes (42) |
| Policy and limitations documented | yes |

Task 4 is **not** complete until the evaluator has been run on the real files and its four outputs inspected.

## 6. Limitations

- One video; nothing here supports a claim that the detector or tracker is accurate, robust or production-ready.
- The XML structure is the one you described; attribute layout (`<attribute name="id">` child elements, CVAT `outside`/`occluded`/`keyframe`) was tested on synthetic files only. If the real file differs, the parse counters and identity validation will show it (for example `boxes_missing_id`).
- Evaluation is restricted to annotated frames inside the predicted range; PIE does not say other frames are empty.
- Single operating point, no AP/mAP; no per-confidence analysis.
- Fully occluded and interpolated GT boxes are retained by default, which lowers recall; use the occlusion stratification or `--exclude-fully-occluded` to see the effect.
- PIE has no truck/bus class, and trains are unsupported: both are only diagnosed.
- Frame alignment (XML frame number equals predicted `frame_index`) is assumed; the sensitivity block is a sanity check, not proof.
- IoU-only matching; MOTP is mean IoU, not a distance. Image pixels only: no world coordinates, no metric accuracy.
- XML is read with the standard library parser, which is fine for the official file but is not hardened against hostile XML.

## 7. Exact Colab execution

The evaluator is new, so it is **not in the earlier Task 1 ZIP**. A new bundle `pedsafety_task4_bundle.zip` (+ `.manifest.json`) contains it; its SHA-256 is given in the handoff message.
Run these as notebook cells in the Colab session that has Drive mounted (a GPU runtime is not needed; the evaluator is CPU-only and takes seconds to minutes).

1. Upload `pedsafety_task4_bundle.zip` and `pedsafety_task4_bundle.manifest.json` to `MyDrive/pedsafety_task1/`.
2. Cell A: mount, verify, extract (stops on any mismatch):

```python
from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, os, pathlib, shutil, zipfile
DRIVE = pathlib.Path("/content/drive/MyDrive/pedsafety_task1")
EXPECTED_ZIP_SHA256 = "<paste the SHA-256 from the handoff message>"
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
zip_path, man_path = DRIVE / "pedsafety_task4_bundle.zip", DRIVE / "pedsafety_task4_bundle.manifest.json"
man = json.loads(man_path.read_text())
z = sha256(zip_path)
assert z == man["bundle_sha256"] == EXPECTED_ZIP_SHA256.strip().lower(), "bundle hash mismatch"
SRC = pathlib.Path("/content/pedsafety_src_task4")
shutil.rmtree(SRC, ignore_errors=True)
with zipfile.ZipFile(zip_path) as zf:
    assert set(zf.namelist()) == set(man["files"]), "ZIP contents differ from manifest"
    zf.extractall(SRC)
assert all(sha256(SRC / n) == e["sha256"] for n, e in man["files"].items()), "file hash mismatch"
os.chdir(SRC)
print("bundle verified:", z)
```

3. Cell B: run the evaluator tests (CPU, about a second; `scipy` and `pytest` are preinstalled on Colab, otherwise `!pip install -q scipy pytest`):

```python
!python -W ignore -m pytest tests/phase3/test_task4_pie_evaluator.py -q -p no:cacheprovider
```

4. Cell C: locate the prediction files and run the evaluation. Inputs are read from Drive; outputs are written locally and then copied to a new Drive folder. Nothing under the existing experiment is modified.

```python
import datetime, glob, subprocess, sys
EXP = pathlib.Path("/content/drive/MyDrive/pedsafety_task1/PIE_Pilot/experiments/20261114T134036Z_pie_video_0004_yolo11n")
def one(pattern):
    hits = sorted(glob.glob(str(EXP / "**" / pattern), recursive=True))
    assert len(hits) == 1, f"expected exactly one {pattern} under {EXP}, found {hits}"
    return hits[0]
DET, TRK, RUNM = one("detections.jsonl"), one("tracked_agents.jsonl"), one("run_manifest.json")
ANN = "/content/PIE_Pilot/annotations/annotations/set01/video_0004_annt.xml"      # re-upload/extract the PIE pilot first if the session was reset
assert os.path.isfile(ANN), "annotation XML missing in this session"
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOCAL_OUT = f"/content/work/pie_eval_{stamp}"
r = subprocess.run([sys.executable, "-W", "ignore", "scripts/evaluate_pie_perception.py", "--annotations", ANN, "--detections", DET, "--tracked-agents", TRK,
                    "--run-manifest", RUNM, "--output-dir", LOCAL_OUT, "--video-id", "video_0004", "--iou-thresholds", "0.50", "0.75"], capture_output=True, text=True)
print(r.stdout, r.stderr)
assert r.returncode == 0, "evaluation failed (see stderr above)"
DRIVE_OUT = EXP / "evaluation" / f"{stamp}_pie_eval"
shutil.copytree(LOCAL_OUT, DRIVE_OUT)                       # new directory; never overwrites
assert all(sha256(DRIVE_OUT / f) == sha256(pathlib.Path(LOCAL_OUT) / f) for f in os.listdir(LOCAL_OUT)), "Drive copy differs"
print("saved:", DRIVE_OUT)
```

5. Cell D: look at the results (read the Markdown first, then check `ground_truth.identity_validation`, `counters`, `frame_alignment_sensitivity_at_lowest_iou_threshold` and `frames.dropped_outside_prediction_coverage` in the JSON):

```python
print((DRIVE_OUT / "pie_evaluation_summary.md").read_text())
```

**What to check before trusting any number:** that `video_id` matched and the row counts are plausible; that the XML counters match your inspection (non-outside pedestrian, car, bicycle, train boxes; 2,161 annotated frames; range 0 to 7,979);
that the frame-alignment block shows offset `+0` as best; that GT identity validation lists no unexpected missing or inconsistent ids; and how many evaluated frames were dropped for lying outside the prediction coverage. If the earlier run processed fewer than 9,840 frames, that count will be large and the results apply only to the covered range.

**Expected output files** in `.../experiments/20261114T134036Z_pie_video_0004_yolo11n/evaluation/<UTC>_pie_eval/`: `pie_detection_metrics.json`, `pie_tracking_metrics.json`, `pie_evaluation_summary.md`, `pie_evaluation_manifest.json`.
