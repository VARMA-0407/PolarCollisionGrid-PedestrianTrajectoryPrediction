# Phase 3 Task 1: Execution Guide (Local Windows and Google Colab)

This guide separates what runs **locally** (CPU, Windows/VS Code) from what runs in **Colab** (T4 GPU). The Colab notebook runs the same
scripts as the local commands below; it contains no separate implementation. Nothing in this guide has been executed on a GPU.

All commands are run from the repository root `D:\BTech-Research\PCG-Baseline` in PowerShell unless marked otherwise.

## 0. Which command runs where

| Step | Where | Environment |
|---|---|---|
| Setup check, unit/integration tests of Task 1 | Local | `pedsafety-perception` |
| Synthetic video, CPU pipeline run, CPU benchmark | Local | `pedsafety-perception` |
| Regression suites (`tests/phase3`, `tests/task7`, `tests/phase2`) | Local | `pcg-baseline` (Python 3.9) |
| Build the Colab ZIP and hash manifest | Local | either (stdlib only) |
| Download YOLO weights | **Colab only** | notebook cell 7 |
| YOLO + ByteTrack on the GPU, GPU benchmark, GPU memory | **Colab only** | Colab T4 |
| Real-video detection quality | Blocked until you supply a road video | - |

## 1. Local setup and validation (Deliverable A)

### 1.1 Activate the environment

The environment is a path-based Conda environment on D:. Do not recreate it and do not touch `pcg-baseline`.

```powershell
conda activate D:\envs\pedsafety-perception
python --version        # Python 3.11.18
```

If `conda activate` is unavailable in the shell, call the interpreter directly instead: `D:\envs\pedsafety-perception\python.exe` in place of `python` below.

### 1.2 Dependency compatibility check

```powershell
python -m pip check
python -c "import torch, torchvision, numpy, cv2, ultralytics, lap; print(torch.__version__, torchvision.__version__, numpy.__version__, cv2.__version__, ultralytics.__version__, 'cuda', torch.cuda.is_available())"
python -c "from ultralytics.trackers.byte_tracker import BYTETracker; print('ByteTrack import ok')"
```

Expected: `pip check` prints two lines only, `ultralytics requires opencv-python, which is not installed` (the headless wheel provides `cv2` under
another distribution name) and `ultralytics requires nvidia-ml-py, which is not installed` (optional GPU-monitoring helper, not imported by this code).
CUDA prints `False` locally; do not expect a GPU here.

If the environment ever has to be rebuilt (only with your approval), the install order is:

```powershell
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps ultralytics==8.4.212
pip install opencv-python-headless==5.0.0.93 lapx==0.13.1 jsonschema pytest matplotlib pyyaml requests scipy polars psutil ultralytics-thop ultralytics-platform
```

`--no-deps` for Ultralytics prevents it from pulling a second OpenCV wheel (`opencv-python`) next to the headless one.

### 1.3 Unit and integration tests (Task 1)

```powershell
python -m pytest tests/phase3 -k task1 -q -rs
```

The one expected skip is the optional real-weights test (`test_optional_real_weights_real_video`); it needs `PEDSAFETY_WEIGHTS` and `PEDSAFETY_VIDEO`.
To run it later, with weights and a road video you obtained yourself:

```powershell
$env:PEDSAFETY_WEIGHTS = "D:\path\to\yolo11n.pt"
$env:PEDSAFETY_VIDEO   = "D:\path\to\road_clip.mp4"
python -m pytest tests/phase3/test_task1_integration.py -k optional_real -q -rs
```

### 1.4 Generate a synthetic video and run the pipeline on it (CPU, plumbing only)

```powershell
python scripts/make_synthetic_video.py --out outputs/synthetic/synthetic_scene.mp4
python scripts/run_perception.py --video outputs/synthetic/synthetic_scene.mp4 --backend replay --replay-gt outputs/synthetic/synthetic_scene.gt.json --save-video --scene-id synthetic_scene --run-name synthetic_replay
```

`--backend replay` uses a **mock** detector that replays the synthetic ground-truth boxes, so the real video source, real ByteTrack wrapper, pipeline,
exporter and visualiser are exercised together. It says nothing about YOLO or detection quality. Each run creates a new directory
`outputs\perception\<UTC timestamp>_<name>\`.

### 1.5 Run the real YOLO wrapper on CPU (needs a weights file you already have)

```powershell
python scripts/run_perception.py --video D:\path\to\road_clip.mp4 --model D:\path\to\yolo11n.pt --device cpu --max-frames 200 --save-video --run-name local_cpu_real
```

Nothing is downloaded: a missing weights path is an error (exit code 2). Local weights are not part of the Colab ZIP and must never be committed.

### 1.6 Inspect the outputs

```powershell
$run = Get-ChildItem outputs\perception -Directory | Sort-Object Name | Select-Object -Last 1
Get-ChildItem $run.FullName
Get-Content "$($run.FullName)\tracked_agents.jsonl" -TotalCount 3
Get-Content "$($run.FullName)\run_manifest.json" | Select-Object -First 60
python -c "import json,sys,collections; rows=[json.loads(l) for l in open(sys.argv[1])]; print(len(rows),'rows;', len({r['track_id'] for r in rows}),'track ids;', collections.Counter(r['semantic_class'] for r in rows))" "$($run.FullName)\tracked_agents.jsonl"
Invoke-Item "$($run.FullName)\annotated.mp4"
```

What to look for in the annotated video: a stable colour and `ID n` per object (a colour change is an ID switch), grey `det ...` boxes for detections that did
not become tracks (false or low-score detections), objects that appear without a box (missed detections), and the header with frame index and timestamp.

### 1.7 Local CPU performance report

```powershell
python scripts/benchmark_perception.py --video D:\path\to\road_clip.mp4 --model D:\path\to\yolo11n.pt --device cpu --max-frames 100 --warmup 3 --run-name local_cpu
```

It prints decode-only, detection-only, tracking-only and combined latency (mean, P50, P95, max), combined processing FPS, the declared hardware and software
versions, and writes `benchmark.json` (new timestamped directory under `outputs\perception_bench\`). Add `--plumbing-only` for any non-representative setup
(random weights, synthetic video); such a report can never be marked as supporting performance claims. These are CPU numbers; do not use them as GPU numbers.

### 1.8 Regression suites (separate environment)

```powershell
conda activate pcg-baseline
python -m pytest tests/phase3 tests/task7 tests/phase2 -q -rs
conda activate D:\envs\pedsafety-perception
```

Expected: only the documented baseline failure `tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension` (the original
`Interaction.py` defect). The Task 1 tests that need OpenCV or Ultralytics are skipped in `pcg-baseline`, by design.

### 1.9 Build the Colab source ZIP

```powershell
python scripts/build_colab_bundle.py --out-dir outputs/colab_bundle
```

It refuses to reuse an existing directory and prints the ZIP SHA-256. Rebuild into a new directory after any source change; the notebook rejects a ZIP whose
hash differs from its manifest.

## 2. Colab execution (Deliverable B) - run only after your review

Not executed yet. GPU validation does not exist until you run it.

1. In Google Drive create `MyDrive/pedsafety_task1/` and upload `pedsafety_task1_bundle.zip` and `pedsafety_task1_bundle.manifest.json` from
   `outputs\colab_bundle\`. Optionally upload a road video you have rights to (for example `MyDrive/pedsafety_task1/videos/road_clip.mp4`).
2. Upload `notebooks/phase3_task1_perception_colab.ipynb` to Colab (File -> Upload notebook), or open it from Drive.
3. Runtime -> Change runtime type -> **T4 GPU**.
4. Edit only the PARAMETERS cell: paste the ZIP SHA-256 from the handoff report into `EXPECTED_ZIP_SHA256` (it cannot be embedded in the notebook because the notebook is
   inside the ZIP), set `INPUT_VIDEO` if you uploaded a video, choose `WEIGHTS_FILE` (`yolo11n.pt` or `yolo11s.pt`).
5. Runtime -> Run all and approve the Drive prompt. The cells, in order: bootstrap (status tracking and the failure-proof manifest writer), GPU check, Drive mount
   (reserves a new `experiments/<UTC>_<name>/` folder), ZIP hash verification (stops on any mismatch), pinned installs, version capture, weights download with a size
   and header check and provenance, the Task 1 tests, input-video setup (a real video is copied from Drive to `/content/work/input` first), `run_perception.py`,
   `benchmark_perception.py`, and the final manifest and Drive copy.
6. **All work is produced on local Colab storage** under `/content/work/<UTC>_<name>/` (the annotated MP4 and JSONL are never written directly to Drive). At the end, or at
   the moment any stage fails, that folder is copied to `MyDrive/pedsafety_task1/experiments/<UTC>_<name>/`, read back and size/hash-checked; the result is recorded
   under `drive_copy` in the manifest. If the Colab VM dies before the copy, the local folder is lost.
7. Results in the Drive folder: `colab_run_manifest.json` (written on success **and** on failure: `status` COMPLETED / COMPLETED_WITH_FAILURES / ABORTED, `first_failed_stage`,
   `failed_stages`, `blocked_stages`, `errors` with a traceback tail, `blocked_validations`, basic and full environment, per-step state, `drive_copy`),
   `environment.json`, `weights_provenance.json`, `tests_summary.json`, a copy of the bundle manifest, `logs/` (`pip_install.log`, `pytest_task1.log`, `run_perception.log`,
   `benchmark.log`), `run/<UTC>_perception/` (`tracked_agents.jsonl`, `detections.jsonl`, `annotated.mp4`, `run_manifest.json`), `benchmark/<UTC>_perception/benchmark.json`,
   and `synthetic_scene.mp4` when no real video was supplied. No `agent_states.jsonl` is produced (the notebook passes no `--scene-id`).

Notes for the Colab run:
- The weights URL in the notebook points at the Ultralytics assets release and has **not been verified** from this machine. If it fails, the cell reports
  BLOCKED and stops; set `WEIGHTS_URL` to a valid source and re-run. The notebook records URL, UTC date, size, SHA-256, the Ultralytics version and the licence
  note; weights stay in `/content/weights` and are never written into the repository or the ZIP.
- Colab keeps its own CUDA PyTorch and OpenCV; the notebook does not reinstall them and records their versions. The pinned packages are
  `ultralytics==8.4.212` and `lapx==0.13.1`. Their compatibility with the Colab Python/NumPy of the day has not been tested.
- Without `INPUT_VIDEO` the YOLO run uses the synthetic rectangle video: it exercises the GPU path and gives timing only. The benchmark is run with `--plumbing-only`, so
  `benchmark.json` carries `supports_performance_claims: false` and a caveat, a guard stops the notebook if it ever says otherwise, and the manifest lists
  `real_video_validation` under `blocked_validations`. Detection quality is never validated by this notebook (no ground truth exists). Expect few or no detections on the synthetic video.
- A failed download is recorded as BLOCKED; a downloaded file under `MIN_WEIGHTS_BYTES` (1 MB) or without a zip/`PK` header is recorded as FAILED. Only the SHA-256 is recorded:
  there is no reference hash to verify the weights against.
