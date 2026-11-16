"""Build notebooks/PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb (the master Colab orchestration notebook).

  python scripts/build_colab_full_pipeline_notebook.py [--out notebooks/PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb]

The notebook contains NO pipeline implementation: every stage calls the repository code (``pedsafety.multivideo`` orchestration over the
existing Phase 3-9 public APIs, ``scripts/run_perception.py``, ``python -m pedsafety.phase9``). Re-running this builder regenerates the
notebook deterministically from the cell sources below.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "notebooks" / "PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb"

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# =====================================================================================================================================
md(r'''
# Real-Time AI-Based Pedestrian Safety Prediction — full PIE pipeline (Google Colab)

**Interaction-Aware Multimodal Trajectory and Conflict Forecasting from Video**

**Status of this notebook: structurally validated and smoke-tested locally on synthetic data in precomputed-artifact mode. It has NOT been executed
on Google Colab or on the real PIE videos until you run it.** Every number it produces is created by your run and stored in a new experiment folder.

This notebook is an **orchestration interface**. It contains no model, tracker or metric implementation: every stage calls the repository code
(`pedsafety.multivideo`, which wraps the existing Phase 3–9 public APIs; `scripts/run_perception.py`; `python -m pedsafety.phase9`).

## Objective
Process every downloaded PIE video (all six recording sets), freeze a leakage-free **recording-level** split, train the interaction-aware multimodal
forecaster on the training recordings only, and produce per-video conflict, risk and early-warning outputs plus a consolidated, machine-readable report —
with video-level and stage-level resume after a Colab disconnect.

## Architecture
```mermaid
flowchart LR
  A[PIE MP4 videos<br/>set01..set06] --> B[Video ingestion<br/>OpenCV decode]
  B --> C[YOLO11 detection]
  C --> D[ByteTrack tracking]
  D --> E[Agent states<br/>image pixels]
  E --> F[Phase 4 Task 2<br/>causal image kinematics]
  F --> G[Phase 4 Task 3<br/>interaction graphs]
  G --> H[P5.3 windows<br/>H=6, F=15 @10 Hz]
  H --> I[Heterogeneous GATv2]
  I --> J[Causal TCN]
  J --> K[Multimodal decoder<br/>K=5 modes + probabilities]
  K --> L[Phase 6 conflict<br/>pairs + indicators]
  L --> M[Phase 7 risk<br/>uncalibrated score]
  M --> N[Phase 8 early-warning<br/>state machine]
  N --> O[Phase 9 gates +<br/>experiment report]
```

| Section | Stage | Repository entry point |
|---|---|---|
| 1 | Colab environment, Drive, GPU, code, dependencies | `git`, `pip` (only missing packages) |
| 2 | Drive layout and PIE inventory (counts, sizes, SHA-256, annotations) | `pedsafety.multivideo.inventory.inventory_pie` |
| 3 | Repository provenance and experiment identity / lock | `pedsafety.multivideo.common.git_state`, `experiment.create_or_resume_experiment` |
| 4 | Official split discovery and frozen recording-level split | `pedsafety.multivideo.split` (Phase 9 `phase9-group-split-1.0.0` layout) |
| 5 | Phase 3 perception per video (YOLO + ByteTrack), optional PIE annotation evaluation | `scripts/run_perception.py` via `perception_batch`; `pie_perception_evaluator.run_evaluation` |
| 6 | Phase 4 Task 2 / Task 3 per video | `run_image_kinematics`, `run_interaction_representation` via `preprocess` |
| 7 | Phase 5 windows, multi-video training, evaluation | `generate_windows`; `training.train_multivideo`; `evaluation.evaluate_all` (`evaluate_neural`) |
| 8 | Phase 6 conflict representation | `build_conflict_pairs` via `downstream` |
| 9 | Phase 7 risk assessment (calibration gated) | `assess_risk` via `downstream` |
| 10 | Phase 8 early-warning events | `run_alerts` via `downstream` |
| 11 | Phase 9 orchestration status, gates, optional pilot regression | `python -m pedsafety.phase9 evaluation-status / run / verify-run` |
| 12 | Consolidated experiment report | `pedsafety.multivideo.report.build_report` |
| 13 | Tests and final validation | `pytest tests`, `report.validate_schemas`, `marker_table` |

## Dataset requirements
`<DRIVE_ROOT>/data/raw/pie/PIE_clips/set01 … set06/video_NNNN.mp4`, the PIE annotation XML (`annotations/…/setNN/video_NNNN_annt.xml`, or the
`annotations*.zip` archives — extracted to `data/interim/` if needed) and the PIE toolkit file `pie_data.py` (anywhere under `data/raw/pie/`, used only
to read the official split; it is parsed, never executed). Nothing is assumed: the number of clips per set is counted.

## Hardware
A Colab GPU runtime (T4 or better) for Phase 3 and training. Phase 4 runs on the CPU and dominates the wall time for long videos. Large intermediate
files are produced on local `/content` storage and published to Drive with SHA-256 verification; frames are never written to Drive.

## Outputs
Everything goes to `<DRIVE_ROOT>/experiments/<EXPERIMENT_ID>/` (never overwritten): inventory, frozen split, per-video Phase 3–8 artifacts with completion
markers, training checkpoints and logs, evaluation results, `experiment_report.json`, `EXPERIMENT_REPORT.md`, `final_validation.json`.

## Scientific limitations (preserved, not hidden)
* **Image-space pixels only** (`captured_1920x1080`, moving camera, no ego-motion compensation): no metres, no physical distance, TTC, PET or collision verification.
* **Forecasting targets are perceived-track pseudo-targets** (YOLO + ByteTrack), not PIE annotation ground truth: the `ANNOTATED_TARGET` adapter is not implemented.
  Held-out-recording metrics are therefore *pseudo-target diagnostics*; the Phase 9 independent-evaluation gate stays **BLOCKED**.
* **Primary metrics use the most-probable mode**; minADE@K / minFDE@K are oracle diagnostics, reported separately.
* **Risk scores are uncalibrated** (`calibrated_probability = null`, `NOT_CALIBRATED`); categories and alert thresholds are provisional engineering values.
* **No warning lead time** can be validated (no ground-truth event timestamps).
* **No real-time claim**: stages are timed offline and separately, and the window builder needs future frames (no causal online mode).
* The earlier Phase 3–9 work was a **single-video pilot** (`set01/video_0004`); none of its numbers is reused here as a full-dataset result.

## How to run / resume
1. Runtime → Change runtime type → **T4 GPU** (or better).
2. Edit **only** the central configuration cell below, then Runtime → Run all.
3. After a disconnect: set `EXPERIMENT_ID` to the id printed in Section 3 and Run all again. Completed videos and stages are verified and skipped;
   training continues from its newest checkpoint. Keep `WORK_ROOT` unchanged.
''')

md(r'''
## Central configuration — the only cell you need to edit
''')

code(r'''
# ===================================================== CENTRAL CONFIGURATION =========================================================
# ---- Google Drive layout ----------------------------------------------------------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/PedestrianSafety"          # project root on Drive (data/, experiments/, checkpoints/, logs/, outputs/)
PIE_RAW_DIR = DRIVE_ROOT + "/data/raw/pie"                        # holds PIE_clips/ and annotations/ (and pie_data.py somewhere below)
PIE_CLIPS_DIRNAME = "PIE_clips"
PIE_ANNOTATIONS_DIRNAME = "annotations"
REQUIRE_ANNOTATIONS = True            # the inventory fails when no *_annt.xml exists at all
ALLOW_INCOMPLETE_DATASET = False      # True = continue when sets/videos are missing (recorded as PARTIAL_DATASET)
HASH_VIDEOS = True                    # SHA-256 of every MP4 (cached on Drive; first run reads every video once)

# ---- split ------------------------------------------------------------------------------------------------------------------------
SPLIT_SCHEME = "PIE_OFFICIAL"         # "PIE_OFFICIAL" (train/val/test as published) | "PIE_OFFICIAL_CALIBRATION_FROM_VAL" (documented deviation)
CALIBRATION_SETS = None               # only for PIE_OFFICIAL_CALIBRATION_FROM_VAL; default = last official validation set
ALLOW_UNVERIFIED_SPLIT_REFERENCE = False   # True only if pie_data.py is unavailable (the split is then recorded as unverified)

# ---- repository code --------------------------------------------------------------------------------------------------------------
REPO_SOURCE = "github"                # "github" | "git_bundle" (a `git bundle` file on Drive) | "existing" (an existing checkout)
REPO_URL = "https://github.com/VARMA-0407/PolarCollisionGrid-PedestrianTrajectoryPrediction.git"
REPO_BRANCH = "feature/colab-full-pipeline-notebook"
EXPECTED_COMMIT = ""                  # optional commit SHA (prefix) that HEAD must equal; never checked out silently
GIT_BUNDLE_PATH = DRIVE_ROOT + "/code/PCG-Baseline.bundle"
EXISTING_REPO_DIR = ""
REPO_DIR = "/content/PCG-Baseline"
INSTALL_MISSING_DEPENDENCIES = True   # pip-installs only missing / incompatible packages (never torch or numpy)

# ---- experiment identity ----------------------------------------------------------------------------------------------------------
EXPERIMENT_ID = ""                    # "" = new experiment; set to the printed id to RESUME after a disconnect
EXPERIMENT_PREFIX = "pie_full"
ALLOW_RESUME_ACROSS_COMMITS = False
WORK_ROOT = "/content/work"           # local high-I/O scratch; keep unchanged when resuming
REQUIRE_GPU = True                    # stop early if no GPU runtime is attached
REQUIRE_ALL_VIDEOS = True             # a stage counts as complete only if every video succeeded
VIDEO_SUBSET = []                     # e.g. ["set01_video_0001", "set03_video_0001"] for a pilot run; [] = every inventoried video

# ---- stage switches (a disabled stage only VERIFIES existing outputs of this experiment; it never computes) --------------------------
RUN_DATASET_AUDIT = True
RUN_SPLIT_PREPARATION = True
RUN_PERCEPTION = True
RUN_PERCEPTION_PIE_EVALUATION = True
RUN_GEOMETRY = True
RUN_WINDOW_GENERATION = True
RUN_FORECASTING_TRAINING = True
RUN_FORECASTING_EVALUATION = True
EVALUATE_TEST_PARTITION = True
RUN_CONFLICT_PREDICTION = True
RUN_RISK_ASSESSMENT = True
RUN_EARLY_WARNING = True
RUN_END_TO_END_VALIDATION = True
RUN_PHASE9_PILOT_REGRESSION = False   # optional single-video regression of the original Phase 9 runner (needs the pilot Phase 3 folder)
PILOT_PHASE3_DIR = DRIVE_ROOT + "/data/interim/PIE_pilot_inputs"
RUN_TESTS = True
TEST_TIMEOUT_S = 3600
FULL_HASH_VERIFICATION = True         # final validation re-hashes every published artifact

# ---- Phase 3 perception -----------------------------------------------------------------------------------------------------------
PERCEPTION_MODE = "raw_video"         # "raw_video" (YOLO + ByteTrack on the MP4s) | "precomputed" (existing Phase 3 folders; NOT raw-video execution)
PRECOMPUTED_PHASE3_ROOT = ""          # for "precomputed": <root>/<set>/<set>_<video>/{tracked_agents,agent_states}.jsonl + run_manifest.json
PHASE3_REUSE_EXPERIMENT_IDS = []      # earlier experiments whose Phase 3 outputs (same configuration and weights) may be reused after SHA-256 verification
YOLO_WEIGHTS_FILE = "yolo11n.pt"      # the pilot used YOLO11n
WEIGHTS_DRIVE_DIR = DRIVE_ROOT + "/checkpoints/yolo"
DOWNLOAD_WEIGHTS_IF_MISSING = True
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/" + YOLO_WEIGHTS_FILE
EXPECTED_WEIGHTS_SHA256 = ""          # optional; when set, other weights are refused
PERCEPTION_DEVICE = "auto"            # "auto" | "cuda" | "cpu"
SAVE_ANNOTATED_VIDEO = False          # large files; enable for inspection of a few videos
WRITE_DETECTIONS_JSONL = True         # needed for the PIE annotation evaluation; stored gzip-compressed
COMPRESS_DETECTIONS = True
PERCEPTION_MAX_FRAMES = None          # smoke tests only: partial videos are labelled as such
PERCEPTION_TIMEOUT_S = None

# ---- Phase 5 training / evaluation (repository defaults; changing them changes the experiment) --------------------------------------
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 64
TRAIN_LEARNING_RATE = 0.003
TRAIN_DEVICE = "auto"
TRAIN_TORCH_THREADS = None            # None = repository default (1 thread: CPU-bitwise reproducible)
CHECKPOINT_EVERY_EPOCHS = 1           # frequent checkpoints = little lost work after a disconnect
MONITOR_MAX_WINDOWS = 4096            # validation-monitor subset per epoch (logged, never used for selection)
EVAL_DEVICE = "auto"
EVAL_BATCH_SIZE = 64
# Data-loader "workers": not applicable - the existing TrainingWindowDataset collates graphs in-process (no torch DataLoader is used).
''')

# ===================================================================================================================================== 1
md(r'''
# Section 1 — Google Colab environment setup
1.1 detects Colab and mounts Drive · 1.2 reports GPU / CUDA / RAM / disk / Python / PyTorch · 1.3 obtains the repository code (needed by every later
section) · 1.4 checks dependencies and installs **only** missing or incompatible ones · 1.5 verifies the imports.
''')

code(r'''
# ---- 1.1 Environment detection and Google Drive ----------------------------------------------------------------------------------------
import datetime, importlib, importlib.util, json, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:                        # the parent package "google" is absent outside Colab
    IN_COLAB = False
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
PRE_STAGES = {}                                    # sections that run before the experiment folder exists (copied into its status file)

def pre_stage(name, status, **detail):
    PRE_STAGES[name] = {"status": status, "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **detail}
    print("[%s] %s %s" % (status, name, json.dumps(detail, default=str)[:300] if detail else ""))

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("Not running inside Google Colab: Drive is not mounted; DRIVE_ROOT must be a local folder.")
if not Path(DRIVE_ROOT).is_dir():
    raise FileNotFoundError("DRIVE_ROOT %s does not exist. Create it on Drive and place the PIE data under %s" % (DRIVE_ROOT, PIE_RAW_DIR))
pre_stage("S1_ENVIRONMENT", "COMPLETED", in_colab=IN_COLAB, session=SESSION)
''')

code(r'''
# ---- 1.2 Hardware and software report ---------------------------------------------------------------------------------------------
def _cmd(c):
    try:
        r = subprocess.run(c, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

HW = {"python": sys.version.split()[0], "platform": platform.platform(), "in_colab": IN_COLAB, "cpu_count": os.cpu_count()}
try:
    import torch
    HW.update(torch=torch.__version__, torch_cuda_build=torch.version.cuda, cuda_available=bool(torch.cuda.is_available()))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        HW.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2), cuda_capability="%d.%d" % (p.major, p.minor))
except ImportError:
    HW.update(torch=None, cuda_available=False)
HW["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"])
try:
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
    HW.update(ram_total_gb=round(mem["MemTotal"] / 1e6, 2), ram_available_gb=round(mem["MemAvailable"] / 1e6, 2))
except OSError:
    HW.update(ram_total_gb=None)
for label, path in (("local_disk", "/content" if Path("/content").is_dir() else "."), ("drive", DRIVE_ROOT)):
    u = shutil.disk_usage(path)
    HW[label + "_free_gb"], HW[label + "_total_gb"] = round(u.free / 1e9, 1), round(u.total / 1e9, 1)
GPU_AVAILABLE = bool(HW.get("cuda_available"))
for k, v in HW.items():
    print("%-22s %s" % (k, v))
if not GPU_AVAILABLE:
    msg = "No CUDA GPU detected. Runtime -> Change runtime type -> T4 GPU (or better), then restart."
    if REQUIRE_GPU:
        raise RuntimeError(msg + " (set REQUIRE_GPU=False to continue on CPU: perception and training become very slow)")
    print("WARNING:", msg, "Continuing on CPU because REQUIRE_GPU=False.")
elif "T4" not in str(HW.get("gpu")):
    print("NOTE: GPU is %s (not a T4); all timings are recorded with this hardware." % HW.get("gpu"))
pre_stage("S1_HARDWARE", "COMPLETED", gpu=HW.get("gpu"), cuda=GPU_AVAILABLE)
''')

code(r'''
# ---- 1.3 Obtain the repository code (no silent branch switching, no overwrite) ------------------------------------------------------------
def git(*args, cwd=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), (r.stderr or r.stdout)[-1500:]))
    return r.stdout.strip()

if REPO_SOURCE == "existing":
    REPO = Path(EXISTING_REPO_DIR)
    if not (REPO / "pedsafety" / "multivideo").is_dir():
        raise FileNotFoundError("EXISTING_REPO_DIR %s is not a checkout of this project (pedsafety/multivideo missing)" % REPO)
elif Path(REPO_DIR).exists():
    REPO = Path(REPO_DIR)                                             # re-run in the same runtime: verify, never re-clone or switch
    cur = git("rev-parse", "--abbrev-ref", "HEAD", cwd=REPO)
    if cur != REPO_BRANCH:
        raise RuntimeError("%s is on branch %s but REPO_BRANCH is %s; nothing is switched silently (delete the folder or fix REPO_BRANCH)" % (REPO, cur, REPO_BRANCH))
else:
    if REPO_SOURCE == "github":
        src = REPO_URL
    elif REPO_SOURCE == "git_bundle":
        if not Path(GIT_BUNDLE_PATH).is_file():
            raise FileNotFoundError("git bundle not found: %s (create it locally with: git bundle create PCG-Baseline.bundle %s)" % (GIT_BUNDLE_PATH, REPO_BRANCH))
        git("bundle", "verify", GIT_BUNDLE_PATH)
        src = GIT_BUNDLE_PATH
    else:
        raise ValueError("REPO_SOURCE must be github, git_bundle or existing")
    r = subprocess.run(["git", "clone", "--branch", REPO_BRANCH, "--single-branch", src, REPO_DIR], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone of %s (branch %s) failed:\n%s\nIf the branch is not on GitHub yet, push it or use REPO_SOURCE='git_bundle' "
                           "(see docs/colab/GOOGLE_COLAB_NOTEBOOK_GUIDE.md)." % (src, REPO_BRANCH, r.stderr[-1500:]))
    REPO = Path(REPO_DIR)
REPO = REPO.resolve()
HEAD = git("rev-parse", "HEAD", cwd=REPO)
if EXPECTED_COMMIT.strip() and not HEAD.startswith(EXPECTED_COMMIT.strip()):
    raise RuntimeError("HEAD %s does not match EXPECTED_COMMIT %s (no silent checkout; check out the commit yourself or update the setting)" % (HEAD, EXPECTED_COMMIT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)
print("repository:", REPO, "| branch:", git("rev-parse", "--abbrev-ref", "HEAD", cwd=REPO), "| commit:", HEAD)
pre_stage("S1_REPOSITORY", "COMPLETED", repo=str(REPO), commit=HEAD, source=REPO_SOURCE)
''')

code(r'''
# ---- 1.4 Dependencies: check versions, install ONLY what is missing or incompatible (torch / numpy are never reinstalled) -------------------
from importlib import metadata
from packaging.specifiers import SpecifierSet

def dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

REQUIRED = {"torch": ">=2.2", "numpy": ">=1.26", "scipy": ">=1.12", "matplotlib": ">=3.8", "jsonschema": ">=4.0,<5"}
if RUN_TESTS:
    REQUIRED["pytest"] = ">=8.0"
if PERCEPTION_MODE == "raw_video":
    REQUIRED.update({"ultralytics": "==8.4.212", "lapx": "==0.13.1"})          # versions of requirements-perception.txt
OPTIONAL = ["psutil", "pyyaml", "pandas", "packaging"]
NEVER_INSTALL = {"torch", "numpy"}                                            # Colab's CUDA build / ABI-sensitive: reported, not replaced
plan, problems = [], []
for name, spec in REQUIRED.items():
    v = dist_version(name)
    ok = v is not None and v in SpecifierSet(spec)
    if not ok:
        (problems if name in NEVER_INSTALL else plan).append("%s%s (installed: %s)" % (name, spec, v))
cv = {n: dist_version(n) for n in ("opencv-python", "opencv-python-headless", "opencv-contrib-python")}
if PERCEPTION_MODE == "raw_video" and not any(cv.values()):
    plan.append("opencv-python-headless==5.0.0.93 (installed: none)")
if problems:
    raise RuntimeError("incompatible core packages (not reinstalled automatically): %s" % problems)
if plan:
    print("missing / incompatible:", plan)
    if not INSTALL_MISSING_DEPENDENCIES:
        raise RuntimeError("INSTALL_MISSING_DEPENDENCIES=False and packages are missing: %s" % plan)
    specs = [p.split(" (")[0] for p in plan]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *specs], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode != 0:
        raise RuntimeError("pip install failed: %s" % (r.stderr or r.stdout)[-1500:])
    importlib.invalidate_caches()
else:
    print("all required packages present; nothing installed")
DEP_VERSIONS = {n: dist_version(n) for n in list(REQUIRED) + OPTIONAL + list(cv)}
still = [n for n, s in REQUIRED.items() if DEP_VERSIONS.get(n) is None or DEP_VERSIONS[n] not in SpecifierSet(s)]
if still:
    raise RuntimeError("still missing after installation: %s" % still)
PIP_FREEZE = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
COMPAT_NOTES = []
if DEP_VERSIONS["numpy"] and not DEP_VERSIONS["numpy"].startswith("1.26"):
    COMPAT_NOTES.append("numpy %s differs from the locally tested 1.26.4; the Colab test run (Section 13) checks compatibility" % DEP_VERSIONS["numpy"])
if HW["python"].split(".")[:2] != ["3", "9"]:
    COMPAT_NOTES.append("Python %s differs from the locally tested 3.9 (pedsafety core) / 3.11 (perception)" % HW["python"])
print(json.dumps(DEP_VERSIONS, indent=1)); print("\n".join(COMPAT_NOTES))
pre_stage("S1_DEPENDENCIES", "COMPLETED", installed=plan, versions=DEP_VERSIONS, notes=COMPAT_NOTES)
''')

code(r'''
# ---- 1.5 Verify imports -----------------------------------------------------------------------------------------------------------
MODULES = ["numpy", "scipy", "jsonschema", "torch", "pedsafety", "pedsafety.multivideo.common", "pedsafety.multivideo.experiment", "pedsafety.multivideo.inventory",
           "pedsafety.multivideo.split", "pedsafety.multivideo.perception_batch", "pedsafety.multivideo.preprocess", "pedsafety.multivideo.training",
           "pedsafety.multivideo.evaluation", "pedsafety.multivideo.downstream", "pedsafety.multivideo.report", "pedsafety.phase5.models.forecaster",
           "pedsafety.phase9.orchestrator", "pedsafety.phase9.evaluation"]
if PERCEPTION_MODE == "raw_video":
    MODULES += ["cv2", "ultralytics", "lap", "pedsafety.perception.runner"]
bad = {}
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as exc:
        bad[m] = "%s: %s" % (type(exc).__name__, exc)
if bad:
    raise ImportError("imports failed: %s" % json.dumps(bad, indent=1))
from pedsafety.multivideo import MULTIVIDEO_LABEL, TARGET_SOURCE
from pedsafety.multivideo.common import canonical_sha256, git_state, read_json, sha256_file, utc_now, write_json, MARKER
from pedsafety.multivideo.experiment import StageTracker, create_or_resume_experiment
from pedsafety.multivideo.inventory import extract_annotation_archives, inventory_pie
from pedsafety.multivideo.split import (build_split_manifest, check_split_integrity, freeze_split, load_frozen_split, resolve_official_split,
                                        video_partitions)
from pedsafety.multivideo.perception_batch import (OK_VIDEO_STATES, perception_stage_config, register_precomputed_phase3, run_perception_batch)
from pedsafety.multivideo.preprocess import preprocess_batch, preprocess_config
from pedsafety.multivideo.training import load_multivideo_model, train_multivideo
from pedsafety.multivideo.evaluation import evaluate_all, render_evaluation_markdown
from pedsafety.multivideo.downstream import downstream_batch, downstream_config
from pedsafety.multivideo.report import BLOCKED_SCIENTIFIC_ITEMS, build_report, final_status, marker_table, validate_schemas
print("all %d imports OK" % len(MODULES))
pre_stage("S1_IMPORTS", "COMPLETED", modules=len(MODULES))
''')

# ===================================================================================================================================== 2
md(r'''
# Section 2 — Google Drive configuration and PIE dataset inventory
Creates the standard folders (directories only; no file is touched), checks the six set directories, then inventories every MP4 (count per set, sizes,
SHA-256 with a resumable cache, duplicates) and the annotation archives / XML. The inventory is saved as a timestamped manifest in `data/manifests/`.
**The notebook stops here if required data is missing.**
''')

code(r'''
# ---- 2.1 Drive layout --------------------------------------------------------------------------------------------------------------
P = {"root": Path(DRIVE_ROOT), "raw_pie": Path(PIE_RAW_DIR), "interim": Path(DRIVE_ROOT) / "data/interim", "processed": Path(DRIVE_ROOT) / "data/processed",
     "splits": Path(DRIVE_ROOT) / "data/splits", "manifests": Path(DRIVE_ROOT) / "data/manifests", "experiments": Path(DRIVE_ROOT) / "experiments",
     "checkpoints": Path(DRIVE_ROOT) / "checkpoints", "logs": Path(DRIVE_ROOT) / "logs", "outputs": Path(DRIVE_ROOT) / "outputs"}
for k in ("interim", "processed", "splits", "manifests", "experiments", "checkpoints", "logs", "outputs"):
    P[k].mkdir(parents=True, exist_ok=True)
if not (P["raw_pie"] / PIE_CLIPS_DIRNAME).is_dir():
    raise FileNotFoundError("%s/%s not found: upload the PIE clips to %s/%s/setNN/" % (PIE_RAW_DIR, PIE_CLIPS_DIRNAME, PIE_RAW_DIR, PIE_CLIPS_DIRNAME))
if Path(WORK_ROOT).resolve().is_relative_to(P["root"].resolve()):
    raise ValueError("WORK_ROOT must be local Colab storage, not inside DRIVE_ROOT")
for s in ("set01", "set02", "set03", "set04", "set05", "set06"):
    d = P["raw_pie"] / PIE_CLIPS_DIRNAME / s
    print("%s %-6s %s" % ("OK " if d.is_dir() else "MISSING", s, d))
''')

code(r'''
# ---- 2.2 Dataset inventory (counts, sizes, SHA-256, duplicates, annotations) ----------------------------------------------------------
INV_CACHE = P["manifests"] / "pie_video_sha256_cache.json"
EXTRACT_ROOT = P["interim"] / "pie_annotations"
def run_inventory():
    return inventory_pie(P["raw_pie"], clips_dirname=PIE_CLIPS_DIRNAME, annotations_dirname=PIE_ANNOTATIONS_DIRNAME, extra_annotation_roots=[EXTRACT_ROOT],
                         hash_videos=HASH_VIDEOS, hash_cache_path=INV_CACHE, require_annotations=REQUIRE_ANNOTATIONS)

if RUN_DATASET_AUDIT:
    INVENTORY = run_inventory()
    arch = [Path(a["path"]) for a in INVENTORY["annotations"]["archives"]]
    if INVENTORY["annotations"]["n_videos_with_annt_xml"] == 0 and arch:
        print("no extracted annotation XML found; extracting", [a.name for a in arch], "to", EXTRACT_ROOT)
        print(extract_annotation_archives(arch, EXTRACT_ROOT))
        INVENTORY = run_inventory()
    INVENTORY_PATH = P["manifests"] / ("pie_inventory_%s.json" % SESSION)
    write_json(INVENTORY_PATH, INVENTORY)
else:
    prev = sorted(P["manifests"].glob("pie_inventory_*.json"))
    if not prev:
        raise FileNotFoundError("RUN_DATASET_AUDIT=False but no saved inventory exists in %s" % P["manifests"])
    INVENTORY_PATH = prev[-1]
    INVENTORY = read_json(INVENTORY_PATH)
    changed = [v["relpath"] for v in INVENTORY["videos"] if not Path(v["path"]).is_file() or Path(v["path"]).stat().st_size != v["bytes"]]
    if changed:
        raise RuntimeError("saved inventory %s no longer matches the files on Drive: %s" % (INVENTORY_PATH.name, changed[:10]))
print("inventory:", INVENTORY_PATH)
print("status: %s | videos: %d | total: %.1f GB | hashed: %s" % (INVENTORY["status"], INVENTORY["n_videos"], INVENTORY["total_gb"], INVENTORY["videos_hashed"]))
print("%-6s %-7s %-10s %-9s" % ("set", "videos", "GB", "PIE doc"))
for s, r in INVENTORY["sets"].items():
    print("%-6s %-7s %-10.2f %-9s" % (s, r["n_videos"], r["bytes"] / 1e9, r.get("reference_count_pie_documentation")))
print("annotation XML for %d/%d videos; archives: %s" % (INVENTORY["annotations"]["n_videos_with_annt_xml"], INVENTORY["n_videos"],
                                                          [Path(a["path"]).name for a in INVENTORY["annotations"]["archives"]]))
for w in INVENTORY["warnings"]:
    print("WARNING:", w)
for p in INVENTORY["problems"]:
    print("PROBLEM:", p)
DATASET_SCOPE = "FULL_INVENTORY" if INVENTORY["status"] != "MISSING_REQUIRED_DATA" else "PARTIAL_DATASET"
if INVENTORY["status"] == "MISSING_REQUIRED_DATA" and not ALLOW_INCOMPLETE_DATASET:
    raise RuntimeError("required PIE data is missing (see PROBLEM lines). Fix the Drive folder, or set ALLOW_INCOMPLETE_DATASET=True to continue "
                       "with the available videos (recorded as PARTIAL_DATASET).")
pre_stage("S2_DATASET_AUDIT", "COMPLETED" if RUN_DATASET_AUDIT else "REUSED_VERIFIED", inventory=str(INVENTORY_PATH), inventory_status=INVENTORY["status"],
          n_videos=INVENTORY["n_videos"], scope=DATASET_SCOPE)
''')

# ===================================================================================================================================== 3
md(r'''
# Section 3 — Repository provenance and experiment identity
Records branch, commit and working-tree status, checks that code and artifacts are kept apart, and creates (or resumes) the experiment folder.
`experiment_lock.json` freezes every setting that changes scientific outputs: resuming with a different value is refused, so outputs of different
configurations are never mixed.
''')

code(r'''
# ---- 3.1 Git provenance and code / artifact separation ------------------------------------------------------------------------------
GIT = git_state(REPO)
print(json.dumps({k: GIT[k] for k in ("branch", "commit", "describe", "working_tree_clean", "remote_origin")}, indent=1))
if REPO_SOURCE in ("github", "git_bundle") and GIT["branch"] != REPO_BRANCH:
    raise RuntimeError("checked-out branch %s differs from REPO_BRANCH %s" % (GIT["branch"], REPO_BRANCH))
if not GIT["working_tree_clean"]:
    print("WARNING: the working tree has local modifications (recorded in the experiment):\n" + (GIT["status_porcelain"] or "")[:2000])
if REPO.is_relative_to(P["experiments"].resolve()) or P["experiments"].resolve().is_relative_to(REPO):
    raise RuntimeError("repository code and experiment artifacts must live in separate folders")
''')

code(r'''
# ---- 3.2 Create or resume the experiment -------------------------------------------------------------------------------------------
CONFIG_FILES = {"perception": "configs/phase3/perception_default.json", "image_kinematics": "configs/phase4/image_kinematics_default.json",
                "interaction": "configs/phase4/interaction_representation_default.json", "windows": "configs/phase5/window_generation_default.json",
                "training": "configs/phase5/training_default.json", "encoder": "configs/phase5/encoder_default.json", "decoder": "configs/phase5/decoder_default.json",
                "neural_eval": "configs/phase5/neural_evaluation_default.json", "baseline_eval": "configs/phase5/baseline_evaluation_default.json",
                "conflict": "configs/phase6/conflict_default.json", "risk": "configs/phase7/risk_default.json", "alert": "configs/phase8/alert_default.json"}
LOCKED = {"split_scheme": SPLIT_SCHEME, "calibration_sets": CALIBRATION_SETS, "allow_unverified_split_reference": ALLOW_UNVERIFIED_SPLIT_REFERENCE,
          "perception_mode": PERCEPTION_MODE, "precomputed_phase3_root": PRECOMPUTED_PHASE3_ROOT if PERCEPTION_MODE == "precomputed" else None,
          "yolo_weights_file": YOLO_WEIGHTS_FILE, "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256 or None, "perception_max_frames": PERCEPTION_MAX_FRAMES,
          "write_detections_jsonl": WRITE_DETECTIONS_JSONL, "video_subset": sorted(VIDEO_SUBSET), "train_epochs": TRAIN_EPOCHS, "train_batch_size": TRAIN_BATCH_SIZE,
          "train_learning_rate": TRAIN_LEARNING_RATE, "monitor_max_windows": MONITOR_MAX_WINDOWS, "pie_clips_dir": str(P["raw_pie"] / PIE_CLIPS_DIRNAME),
          "config_file_sha256": {k: sha256_file(REPO / v) for k, v in CONFIG_FILES.items()}}
EXP = create_or_resume_experiment(P["experiments"], EXPERIMENT_ID.strip() or None, LOCKED, GIT, allow_commit_change=ALLOW_RESUME_ACROSS_COMMITS,
                                  prefix=EXPERIMENT_PREFIX)
EXPERIMENT_ID, EXP_DIR = EXP["experiment_id"], Path(EXP["dir"])
D = {k: EXP_DIR / v for k, v in {"art": "artifacts", "q": "_quarantine", "reports": "reports", "env": "environment"}.items()}
for d in D.values():
    d.mkdir(parents=True, exist_ok=True)
WORK = Path(WORK_ROOT) / EXPERIMENT_ID
WORK.mkdir(parents=True, exist_ok=True)
TRACKER = StageTracker(EXP_DIR / "experiment_status.json")
for n, r in PRE_STAGES.items():
    TRACKER.set(n, r["status"], **{k: v for k, v in r.items() if k not in ("status", "updated_utc")})
ENVIRONMENT = dict(HW, dependencies=DEP_VERSIONS, compatibility_notes=COMPAT_NOTES, session=SESSION)
write_json(D["env"] / ("environment_%s.json" % SESSION), dict(ENVIRONMENT, git=GIT))
(D["env"] / ("pip_freeze_%s.txt" % SESSION)).write_text(PIP_FREEZE, encoding="utf-8")
inv_copy = EXP_DIR / "inventory.json"
if inv_copy.is_file():
    old = read_json(inv_copy)
    sig = lambda inv: sorted((v["key"], v["bytes"], v.get("sha256")) for v in inv["videos"])
    if sig(old) != sig(INVENTORY):
        raise RuntimeError("the PIE files on Drive differ from the inventory this experiment was started with; start a new experiment")
else:
    write_json(inv_copy, INVENTORY)
(P["experiments"] / "LATEST_EXPERIMENT_ID.txt").write_text(EXPERIMENT_ID + "\n", encoding="utf-8")   # informational only; never read automatically
print("=" * 100)
print("EXPERIMENT_ID = %r   (%s)" % (EXPERIMENT_ID, "RESUMED" if EXP["resumed"] else "NEW"))
print("To resume after a disconnect, put this id into EXPERIMENT_ID in the configuration cell and run all cells again.")
print("experiment folder:", EXP_DIR, "| local scratch:", WORK)
''')

# ===================================================================================================================================== 4
md(r'''
# Section 4 — PIE dataset verification and recording-level split
* Videos and annotations are matched by `(set, video)`; the identity used everywhere is `<set>_<video>` because PIE reuses `video_0001` in several sets.
* **Unit of splitting = recording set.** The clips of one set are consecutive chunks of one recording, so a set is never divided: no clip, frame or window
  is assigned randomly, and the six sets go to disjoint partitions.
* The official split is read from the PIE toolkit (`pie_data.py`) by parsing its syntax tree (never executing it). The official split has **no calibration
  partition**; with `PIE_OFFICIAL` calibration stays empty (and Phase 7 calibration blocked). `PIE_OFFICIAL_CALIBRATION_FROM_VAL` carves whole sets from the
  official validation sets — a documented deviation that still provides no verified conflict labels.
* The split is **frozen before any window is generated** and reused (same content hash) by every later stage and resume.
''')

code(r'''
# ---- 4.1 Videos <-> annotations ---------------------------------------------------------------------------------------------------
ann = INVENTORY["annotations"]
print("videos with *_annt.xml: %d / %d" % (ann["n_videos_with_annt_xml"], INVENTORY["n_videos"]))
print("videos without annotation XML:", ann["videos_without_annt_xml"] or "none")
print("annotation XML without a video:", ann["annotations_without_video"] or "none")
by_set = {}
for v in INVENTORY["videos"]:
    by_set.setdefault(v["set"], []).append(v["video"])
for s, vids in sorted(by_set.items()):
    print("%s (one recording, %d clips): %s" % (s, len(vids), ", ".join(vids)))
''')

code(r'''
# ---- 4.2 Official PIE split (from the PIE toolkit source) --------------------------------------------------------------------------------
if TRACKER.require("S0_OFFICIAL_SPLIT", ["S2_DATASET_AUDIT"]):
    with TRACKER.stage("S0_OFFICIAL_SPLIT") as st:
        OFFICIAL = resolve_official_split([P["raw_pie"]], allow_unverified_reference=ALLOW_UNVERIFIED_SPLIT_REFERENCE)
        print("official split status:", OFFICIAL["status"])
        print("source:", [(f["path"], f["sha256"][:16]) for f in OFFICIAL["source_files"]] or OFFICIAL.get("method"))
        print(json.dumps(OFFICIAL["split"]))
        st["detail"] = {"status": OFFICIAL["status"], "split": OFFICIAL["split"]}
''')

code(r'''
# ---- 4.3 Build, check and FREEZE the recording-level split ----------------------------------------------------------------------------
if TRACKER.require("S0_SPLIT", ["S0_OFFICIAL_SPLIT"]):
    with TRACKER.stage("S0_SPLIT") as st:
        split_copy = EXP_DIR / "split_manifest.json"
        if not RUN_SPLIT_PREPARATION:
            if not split_copy.is_file():
                raise RuntimeError("RUN_SPLIT_PREPARATION=False but this experiment has no frozen split yet")
            SPLIT = load_frozen_split(split_copy)
            st["status"] = "REUSED_VERIFIED"
        else:
            cand = build_split_manifest(INVENTORY, OFFICIAL, scheme=SPLIT_SCHEME, calibration_sets=CALIBRATION_SETS)
            integ = check_split_integrity(cand, INVENTORY)
            if not integ["ok"]:
                raise RuntimeError("split integrity problems: %s" % integ["problems"])
            if split_copy.is_file():
                prev = load_frozen_split(split_copy)
                if prev["content_sha256"] != cand["content_sha256"]:
                    raise RuntimeError("this experiment was frozen with split %s; the rebuilt split is %s (data or scheme changed): start a new experiment"
                                       % (prev["split_id"], cand["split_id"]))
                SPLIT = prev
            else:
                fr = freeze_split(cand, P["splits"])
                SPLIT = fr["manifest"]
                write_json(split_copy, SPLIT)
                print("frozen:", fr["path"], "(reused existing frozen file)" if fr["reused"] else "(new)")
        SPLIT_PATH = split_copy
        INTEG = check_split_integrity(SPLIT, INVENTORY)
        if not INTEG["ok"]:
            raise RuntimeError("frozen split failed integrity checks: %s" % INTEG["problems"])
        PARTS = video_partitions(SPLIT)
        bad_subset = sorted(set(VIDEO_SUBSET) - set(PARTS))
        if bad_subset:
            raise ValueError("VIDEO_SUBSET lists videos that are not in the split: %s" % bad_subset)
        VIDEOS = [dict(v, partition=PARTS[v["key"]]["partition"]) for v in INVENTORY["videos"] if v["key"] in PARTS and (not VIDEO_SUBSET or v["key"] in VIDEO_SUBSET)]
        print("split %s | scheme %s | official split %s" % (SPLIT["split_id"], SPLIT["scheme"], SPLIT["official_split"]["status"]))
        for p, sets in SPLIT["partitions"].items():
            print("  %-12s sets %-28s videos %d" % (p, sets, INTEG["videos_per_partition"][p]))
        print("deviations from the official split:", SPLIT["deviations_from_official"] or "none")
        print("Phase 9 independent-evaluation readiness problems (expected while annotations are unverified):", len(INTEG["phase9_independent_evaluation_problems"]))
        print("videos in scope for this experiment: %d%s" % (len(VIDEOS), " (VIDEO_SUBSET)" if VIDEO_SUBSET else ""))
        st["detail"] = {"split_id": SPLIT["split_id"], "partitions": SPLIT["partitions"], "videos_in_scope": len(VIDEOS), "videos_per_partition": INTEG["videos_per_partition"]}
''')

code(r'''
# ---- 4.4 Shared paths and helpers for Sections 5-13 (run this cell before jumping to any later section after a restart) -----------------------
if TRACKER.status("S0_SPLIT") not in ("COMPLETED", "REUSED_VERIFIED"):
    raise RuntimeError("the recording-level split is not frozen (S0_SPLIT=%s): fix Section 4 first; no later stage may run without it" % TRACKER.status("S0_SPLIT"))
PRE_CFGS = {"image_kinematics": CONFIG_FILES["image_kinematics"], "interaction": CONFIG_FILES["interaction"], "windows": CONFIG_FILES["windows"]}
TRAIN_DIR = D["art"] / "phase5_training"
EVAL_ROOT = D["art"] / "phase5_evaluation"
VKEYS = {v["key"]: v["set"] for v in VIDEOS}
STAGE_DIRS = {"S3_PERCEPTION": D["art"] / "phase3", "S4A_IMAGE_KINEMATICS": D["art"] / "phase4_task2_kinematics", "S4B_INTERACTION": D["art"] / "phase4_task3_interaction",
              "S5A_WINDOWS": D["art"] / "phase5_windows", "S5C_EVALUATION": EVAL_ROOT, "S6_CONFLICT": D["art"] / "phase6_conflict",
              "S7_RISK": D["art"] / "phase7_risk", "S8_ALERTS": D["art"] / "phase8_alerts"}
if PERCEPTION_MODE == "precomputed":
    STAGE_DIRS.pop("S3_PERCEPTION")
DCFG = downstream_config(REPO, {"conflict": CONFIG_FILES["conflict"], "risk": CONFIG_FILES["risk"], "alert": CONFIG_FILES["alert"], "windows": CONFIG_FILES["windows"]})

def load_if(p):
    return read_json(p) if Path(p).is_file() else None

def finish_batch(st, res, enabled, label):
    # stage status from a per-video batch result; a partial result is accepted downstream only when REQUIRE_ALL_VIDEOS=False
    ok = res["n_videos"] - len(res.get("failed", [])) - len(res.get("missing", []))
    st["detail"] = {k: res.get(k) for k in ("n_videos", "by_status", "failed", "missing")}
    if res["all_complete"]:
        st["status"] = "COMPLETED" if enabled else "REUSED_VERIFIED"
    elif ok > 0:
        st["status"] = "COMPLETED_WITH_FAILURES"
        st["extra"] = {"accepted_partial": not REQUIRE_ALL_VIDEOS}
    else:
        st["status"] = "FAILED" if enabled else "SKIPPED_DISABLED"
    print("%s: %d/%d videos OK; failed %s; missing %s" % (label, ok, res["n_videos"], res.get("failed"), res.get("missing")))
    for k in (res.get("failed") or [])[:20]:
        print("   FAILED %s: %s" % (k, res["results"].get(k, {}).get("error", "no result (upstream stage incomplete)")[:300]))

def phase3_items():
    p3 = read_json(EXP_DIR / "phase3_index.json")
    return p3, [{"key": v["key"], "set": v["set"], "phase3_dir": p3["results"][v["key"]]["dir"]} for v in VIDEOS
                if p3["results"].get(v["key"], {}).get("status") in OK_VIDEO_STATES]

def downstream_items():
    ev = read_json(EVAL_ROOT / "evaluation_summary.json")
    wd = read_json(EXP_DIR / "windows_index.json")["window_dirs"]
    return [{"key": k, "set": PARTS[k]["set"], "partition": PARTS[k]["partition"], "windows_dir": wd[k], "predictions_dir": EVAL_ROOT / PARTS[k]["set"] / k}
            for k, r in sorted(ev["per_video"].items()) if r.get("status") == "COMPLETED" and k in VKEYS]
print("helpers ready | videos in scope:", len(VKEYS), "| artifacts:", D["art"])
''')

# ===================================================================================================================================== 5
md(r'''
# Section 5 — Phase 3 perception frontend (video → YOLO → ByteTrack → agent states)
Each video is copied from Drive to local storage, processed by the **existing** `scripts/run_perception.py` in its own subprocess (GPU when available;
frames are processed one at a time and never stored), validated with the Phase 9 Phase-3 contract, optionally evaluated against its PIE annotation
(`pie_perception_evaluator`), and published to Drive with SHA-256 verification and a completion marker. Completed videos are skipped on resume.

Outputs per video (`artifacts/phase3/<set>/<set>_<video>/`): `tracked_agents.jsonl`, `agent_states.jsonl`, `detections.jsonl.gz`, `run_manifest.json`,
`perception_config.json`, `phase3_publication.json`, optional `annotated.mp4`. `PERCEPTION_MODE="precomputed"` validates existing Phase 3 folders
instead and records the run as **PRECOMPUTED_PHASE3_ARTIFACTS** (not raw-video execution).
''')

code(r'''
# ---- 5.1 Detector weights (verified, recorded, locked to the experiment) --------------------------------------------------------------------
WEIGHTS_SHA = None
if PERCEPTION_MODE == "raw_video" and TRACKER.require("S3_WEIGHTS", ["S0_SPLIT"]):
    with TRACKER.stage("S3_WEIGHTS") as st:
        import urllib.request
        W = Path(WEIGHTS_DRIVE_DIR) / YOLO_WEIGHTS_FILE
        prov_path = W.with_suffix(W.suffix + ".provenance.json")
        if not W.is_file():
            if not DOWNLOAD_WEIGHTS_IF_MISSING:
                raise FileNotFoundError("weights %s missing and DOWNLOAD_WEIGHTS_IF_MISSING=False" % W)
            W.parent.mkdir(parents=True, exist_ok=True)
            part = W.with_suffix(W.suffix + ".part")
            with urllib.request.urlopen(WEIGHTS_URL, timeout=180) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out)
            if part.stat().st_size < 1_000_000 or open(part, "rb").read(2) != b"PK":
                raise ValueError("downloaded file is not a PyTorch checkpoint (size %d): %s" % (part.stat().st_size, WEIGHTS_URL))
            part.replace(W)
            write_json(prov_path, {"source_url": WEIGHTS_URL, "downloaded_utc": utc_now(), "bytes": W.stat().st_size, "sha256": sha256_file(W),
                                   "licence": "AGPL-3.0 (Ultralytics code and weights): review before redistribution"})
        WEIGHTS_SHA = sha256_file(W)
        if EXPECTED_WEIGHTS_SHA256 and WEIGHTS_SHA != EXPECTED_WEIGHTS_SHA256.lower():
            raise ValueError("weights sha256 %s != EXPECTED_WEIGHTS_SHA256" % WEIGHTS_SHA)
        lock_w = EXP_DIR / "weights_provenance.json"
        if lock_w.is_file() and read_json(lock_w)["sha256"] != WEIGHTS_SHA:
            raise RuntimeError("this experiment was started with different weights (%s); start a new experiment" % read_json(lock_w)["sha256"])
        write_json(lock_w, {"file": str(W), "sha256": WEIGHTS_SHA, "bytes": W.stat().st_size, "provenance": read_json(prov_path) if prov_path.is_file() else None})
        LOCAL_WEIGHTS = Path("/content/weights" if IN_COLAB else str(WORK / "weights")) / YOLO_WEIGHTS_FILE
        LOCAL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        if not LOCAL_WEIGHTS.is_file() or sha256_file(LOCAL_WEIGHTS) != WEIGHTS_SHA:
            shutil.copyfile(W, LOCAL_WEIGHTS)
        print("weights:", W, "sha256", WEIGHTS_SHA)
        st["detail"] = {"weights": str(W), "sha256": WEIGHTS_SHA}
''')

code(r'''
# ---- 5.2 Run Phase 3 for every video (resumable at video level) ------------------------------------------------------------------------
needs3 = ["S0_SPLIT"] + (["S3_WEIGHTS"] if PERCEPTION_MODE == "raw_video" else [])
if TRACKER.require("S3_PERCEPTION", needs3):
    with TRACKER.stage("S3_PERCEPTION") as st:
        if PERCEPTION_MODE == "precomputed":
            P3 = register_precomputed_phase3(VIDEOS, PRECOMPUTED_PHASE3_ROOT)
            P3["stage_config_sha256"] = canonical_sha256({"mode": "precomputed", "inputs": {k: r.get("detail", {}).get("sha256") for k, r in P3["results"].items()}})
            P3["execution_mode"] = "PRECOMPUTED_PHASE3_ARTIFACTS"
        else:
            STAGE_CFG = perception_stage_config(REPO / CONFIG_FILES["perception"], weights_sha256=WEIGHTS_SHA, device=PERCEPTION_DEVICE, max_frames=PERCEPTION_MAX_FRAMES,
                                                save_video=SAVE_ANNOTATED_VIDEO, write_detections=WRITE_DETECTIONS_JSONL, compress_detections=COMPRESS_DETECTIONS)
            P3 = run_perception_batch(REPO, VIDEOS, reuse_roots=[P["experiments"] / e / "artifacts" / "phase3" for e in PHASE3_REUSE_EXPERIMENT_IDS],
                                      base_config_path=REPO / CONFIG_FILES["perception"], weights_path=LOCAL_WEIGHTS, weights_sha256=WEIGHTS_SHA, device=PERCEPTION_DEVICE,
                                      phase3_root=D["art"] / "phase3", eval_root=D["art"] / "phase3_pie_evaluation", work_root=WORK, quarantine_root=D["q"],
                                      stage_cfg=STAGE_CFG, run_pie_evaluation=RUN_PERCEPTION_PIE_EVALUATION, git_commit=GIT["commit"], timeout_s=PERCEPTION_TIMEOUT_S,
                                      verify_only=not RUN_PERCEPTION)
            P3["stage_config_sha256"] = canonical_sha256(STAGE_CFG)
            P3["execution_mode"] = "RAW_VIDEO"
        P3["partial_videos"] = bool(PERCEPTION_MAX_FRAMES)
        write_json(EXP_DIR / "phase3_index.json", P3)
        finish_batch(st, P3, RUN_PERCEPTION or PERCEPTION_MODE == "precomputed", "Phase 3")
''')

code(r'''
# ---- 5.3 Phase 3 preview ----------------------------------------------------------------------------------------------------------
if (EXP_DIR / "phase3_index.json").is_file():
    P3 = read_json(EXP_DIR / "phase3_index.json")
    print("%-20s %-24s %8s %9s %8s %10s %s" % ("video", "status", "frames", "trackrows", "tracks", "fps(off)", "PIE eval"))
    for k, r in sorted(P3["results"].items()):
        d = r.get("detail", {})
        fps = d.get("frames_per_s_including_decode_and_io")
        print("%-20s %-24s %8s %9s %8s %10s %s" % (k, r["status"], d.get("frames_processed"), d.get("track_rows"), d.get("unique_track_ids"),
                                                 "%.1f" % fps if fps else "-", (d.get("pie_annotation_evaluation") or {}).get("status")))
    first = next((r["dir"] for r in P3["results"].values() if r["status"] in OK_VIDEO_STATES), None)
    if first:
        with open(Path(first) / "tracked_agents.jsonl", encoding="utf-8") as fh:
            print("\nfirst tracked-agent row of", Path(first).name, ":\n", fh.readline()[:600])
    print("\nTimings are offline batch processing of recorded videos; they are not an end-to-end or real-time latency measurement.")
''')

# ===================================================================================================================================== 6
md(r'''
# Section 6 — Phase 4: geometric and kinematic preprocessing (image space)
Per video: **Task 2** `run_image_kinematics` (agent-state validation, track continuity, causal positions / velocities, validity masks; gap bridging and
smoothing stay disabled as configured) and **Task 3** `run_interaction_representation` (typed interaction graphs). The Phase 9 contract checks are
applied: no row may be quarantined or lost, the node count must equal the causal row count, and every record stays in `captured_1920x1080` pixels.
**No homography, metric conversion, interpolation or forward-fill is introduced.**
''')

code(r'''
# ---- 6.1 Task 2 + Task 3 per video ---------------------------------------------------------------------------------------------------
if TRACKER.require("S4_GEOMETRY", ["S0_SPLIT", "S3_PERCEPTION"]):
    with TRACKER.stage("S4_GEOMETRY") as st:
        p3, items = phase3_items()
        PCFG = preprocess_config(REPO, PRE_CFGS, p3["stage_config_sha256"])
        R4 = preprocess_batch(REPO, items, artifacts_root=D["art"], work_root=WORK, quarantine_root=D["q"], pcfg=PCFG, git_commit=GIT["commit"],
                              until="S4B_INTERACTION", keep_scratch=True, verify_only=not RUN_GEOMETRY)
        R4["n_videos"] = len(VIDEOS)
        R4["failed"] = sorted(set(R4["failed"]) | {v["key"] for v in VIDEOS if v["key"] not in R4["results"]})
        R4["all_complete"] = R4["all_complete"] and len(items) == len(VIDEOS)
        write_json(EXP_DIR / "phase4_index.json", R4)
        finish_batch(st, R4, RUN_GEOMETRY, "Phase 4")
        tot = {"rows": 0, "valid_positions": 0, "valid_velocity": 0, "quarantined": 0, "nodes": 0}
        for r in R4["results"].values():
            c = (r.get("detail", {}).get("S4A_IMAGE_KINEMATICS") or {}).get("counts", {})
            for k in ("valid_positions", "valid_velocity", "quarantined"):
                tot[k] += int(c.get(k) or 0)
            tot["rows"] += int(c.get("causal_rows") or 0)
            tot["nodes"] += int(((r.get("detail", {}).get("S4B_INTERACTION") or {}).get("counts") or {}).get("nodes") or 0)
        print("coordinate contract: captured_1920x1080 pixels only (world coordinates absent) | totals:", tot)
''')

# ===================================================================================================================================== 7
md(r'''
# Section 7 — Phase 5: interaction-aware multimodal trajectory forecasting
* **7.1 Windows** — `generate_windows` per video (history 6 samples / 0.5 s, future 15 samples / 1.5 s at 10 Hz). Windows carry `video_id=<set>_<video>`
  and `group_id=<set>`. Targets are **perceived-track pseudo-targets** (`target_source_mode=PILOT_SMOKE`; `ANNOTATED_TARGET` is refused by the generator).
* **7.2 Training** — `train_multivideo`: the unchanged heterogeneous GATv2 + causal TCN + multimodal decoder, masked winner-takes-all loss, AdamW + cosine,
  **scaler fitted on train-partition windows only**, video-homogeneous batches, checkpoints every `CHECKPOINT_EVERY_EPOCHS`, resume from the newest
  checkpoint, `FINAL_EPOCH_ONLY` selection (the validation monitor is logged, never used to select).
* **7.3 Evaluation** — per video with the unchanged P5.7.5 `evaluate_neural`: **primary = most-probable mode** ADE/FDE; **minADE@K / minFDE@K are oracle
  diagnostics, reported separately**; classical baselines on the same windows; future-isolation check.

> The earlier Phase 5 pilot (one video, in-sample, pseudo-targets) is **not** reused here; this section produces a new multi-recording experiment.
''')

code(r'''
# ---- 7.1 Window generation per video ---------------------------------------------------------------------------------------------------
if TRACKER.require("S5A_WINDOWS", ["S4_GEOMETRY"]):
    with TRACKER.stage("S5A_WINDOWS") as st:
        p3, items = phase3_items()
        PCFG = preprocess_config(REPO, PRE_CFGS, p3["stage_config_sha256"])
        R5A = preprocess_batch(REPO, items, artifacts_root=D["art"], work_root=WORK, quarantine_root=D["q"], pcfg=PCFG, git_commit=GIT["commit"],
                               until="S5A_WINDOWS", keep_scratch=False, verify_only=not RUN_WINDOW_GENERATION)
        R5A["n_videos"] = len(VIDEOS)
        R5A["failed"] = sorted(set(R5A["failed"]) | {v["key"] for v in VIDEOS if v["key"] not in R5A["results"]})
        R5A["all_complete"] = R5A["all_complete"] and len(items) == len(VIDEOS)
        R5A["window_dirs"] = {k: r["dirs"]["S5A_WINDOWS"] for k, r in R5A["results"].items() if r["status"] in ("COMPLETED", "SKIPPED_ALREADY_COMPLETE")}
        write_json(EXP_DIR / "windows_index.json", R5A)
        finish_batch(st, R5A, RUN_WINDOW_GENERATION, "Phase 5 windows")
        per = {}
        for k, r in R5A["results"].items():
            n = (r.get("detail", {}).get("S5A_WINDOWS") or {}).get("accepted_windows")
            if n is None and k in R5A["window_dirs"]:
                n = read_json(Path(R5A["window_dirs"][k]) / "run_manifest.json")["counts"]["accepted_windows"]
            per.setdefault(PARTS[k]["partition"], []).append(n or 0)
        for p, ns in per.items():
            print("  %-12s videos %3d  windows %8d" % (p, len(ns), sum(ns)))
''')

code(r'''
# ---- 7.2 Multi-video training on the TRAIN partition only ---------------------------------------------------------------------------------
if TRACKER.require("S5B_TRAINING", ["S5A_WINDOWS"]):
    with TRACKER.stage("S5B_TRAINING") as st:
        wd = read_json(EXP_DIR / "windows_index.json")["window_dirs"]
        train_dirs = {k: d for k, d in wd.items() if PARTS[k]["partition"] == "train"}
        val_dirs = {k: d for k, d in wd.items() if PARTS[k]["partition"] == "validation"}
        overrides = {"epochs": TRAIN_EPOCHS, "batch_size": TRAIN_BATCH_SIZE, "learning_rate": TRAIN_LEARNING_RATE, "device": TRAIN_DEVICE,
                     "torch_threads": TRAIN_TORCH_THREADS, "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS}
        print("train videos: %d | validation-monitor videos: %d" % (len(train_dirs), len(val_dirs)))
        TRAIN_MANIFEST = train_multivideo(train_dirs, SPLIT, REPO / CONFIG_FILES["training"], TRAIN_DIR, overrides=overrides, val_dirs=val_dirs,
                                          monitor_max_windows=MONITOR_MAX_WINDOWS, quarantine_root=D["q"], git=GIT, verify_only=not RUN_FORECASTING_TRAINING)
        st["status"] = "REUSED_VERIFIED" if TRAIN_MANIFEST.get("skipped_already_complete") and not RUN_FORECASTING_TRAINING else "COMPLETED"
        st["detail"] = {"final_checkpoint": TRAIN_MANIFEST["training"]["final_checkpoint"], "train_windows": TRAIN_MANIFEST["inputs"]["train_windows"],
                        "device": TRAIN_MANIFEST["training"]["device"], "epochs": TRAIN_MANIFEST["training"]["epochs"]}
''')

code(r'''
# ---- 7.2b Training curves ------------------------------------------------------------------------------------------------------------
if (TRAIN_DIR / "train_log.jsonl").is_file():
    import matplotlib
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in (TRAIN_DIR / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ep = [r for r in rows if r["type"] == "epoch"]
    mon = [r for r in rows if r["type"] == "validation_monitor"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([r["epoch"] for r in ep], [r["mean_loss"] for r in ep], label="train loss (in-sample)")
    if mon:
        ax[0].plot([r["epoch"] for r in mon], [r["mean_loss"] for r in mon], label="validation monitor loss (pseudo-targets)")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("masked WTA loss")
    ax[1].plot([r["epoch"] for r in ep], [r["mean_selected_mode_ade_px"] for r in ep], label="train selected-mode ADE")
    ax[1].plot([r["epoch"] for r in ep], [r["mean_oracle_mode_ade_px"] for r in ep], "--", label="train oracle-mode ADE (diagnostic)")
    if mon:
        ax[1].plot([r["epoch"] for r in mon], [r["mean_selected_mode_ade_px"] for r in mon], label="validation-monitor selected-mode ADE")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("pixels"); ax[1].legend(); ax[1].set_title("ADE (pixels)")
    fig.tight_layout()
    fig.savefig(D["reports"] / "training_curves.png", dpi=120)
    plt.show()
''')

code(r'''
# ---- 7.3 Evaluation per video and per partition --------------------------------------------------------------------------------------
from pedsafety.phase9.evaluation import assess_independent_evaluation, validate_split_manifest
if TRACKER.require("S5C_EVALUATION", ["S5B_TRAINING"]):
    with TRACKER.stage("S5C_EVALUATION") as st:
        scan = assess_independent_evaluation(REPO, str(SPLIT_PATH))
        probs = validate_split_manifest(SPLIT)
        PHASE9_GATE = {"status": scan["status"] if not probs else "BLOCKED_NO_VALID_FROZEN_SPLIT_MANIFEST", "repository_scan": scan, "split_manifest_problems": probs,
                       "note": "the repository scan counts groups processed inside the repository; independently, the split manifest problems "
                               "(annotation provenance, identity mapping, timestamp alignment, licence not verified) keep the gate blocked"}
        print("Phase 9 independent-evaluation gate:", PHASE9_GATE["status"], "| split problems:", len(probs))
        wd = read_json(EXP_DIR / "windows_index.json")["window_dirs"]
        parts_eval = ("train", "validation", "calibration") + (("test",) if EVALUATE_TEST_PARTITION else ())
        EVAL = evaluate_all(TRAIN_DIR, wd, SPLIT, EVAL_ROOT, WORK, D["q"], neural_eval_config_path=REPO / CONFIG_FILES["neural_eval"], repo=REPO, device=EVAL_DEVICE,
                            batch_size=EVAL_BATCH_SIZE, partitions=parts_eval, git_commit=GIT["commit"], phase9_gate=PHASE9_GATE,
                            verify_only=not RUN_FORECASTING_EVALUATION)
        (D["reports"] / "PHASE5_EVALUATION.md").write_text(render_evaluation_markdown(EVAL), encoding="utf-8")
        print(render_evaluation_markdown(EVAL))
        st["status"] = "REUSED_VERIFIED" if EVAL.get("reused_verified") else "COMPLETED"
        st["detail"] = {"partitions": sorted(EVAL["per_partition"]), "videos": len(EVAL["per_video"]), "test_evaluated": EVALUATE_TEST_PARTITION}
        if not EVALUATE_TEST_PARTITION:
            TRACKER.set("S5C_TEST_PARTITION", "SKIPPED_DISABLED", reason="EVALUATE_TEST_PARTITION=False")
''')

# ===================================================================================================================================== 8-10
md(r'''
# Section 8 — Phase 6: conflict prediction (image-space representation)
`build_conflict_pairs` per video: pedestrian–vehicle pairs at each window anchor, mode-aggregated image-space indicators (pixel separations, closing rates,
image-plane time-to-minimum-separation), pair validity masks, and the heuristic proximity score. **These are pixel indicators, not physical TTC/PET,
separation in metres or verified collisions.** Supervised conflict training stays blocked (no verified conflict labels).
''')


def downstream_cell(stage, name, needs, switch, label):
    code(r'''
# ---- %(label)s per video ----------------------------------------------------------------------------------------------------------
if TRACKER.require("%(name)s", %(needs)r):
    with TRACKER.stage("%(name)s") as st:
        items = downstream_items()
        R = downstream_batch(REPO, items, stages=("%(stage)s",), artifacts_root=D["art"], work_root=WORK, quarantine_root=D["q"], dcfg=DCFG, git_commit=GIT["commit"],
                             verify_only=not %(switch)s)
        R["n_videos"] = len(items)
        write_json(EXP_DIR / ("%(name)s_index.json".lower()), R)
        finish_batch(st, R, %(switch)s, "%(label)s")
''' % {"stage": stage, "name": name, "needs": needs, "switch": switch, "label": label})


downstream_cell("S6_CONFLICT", "S6_CONFLICT", ["S5C_EVALUATION"], "RUN_CONFLICT_PREDICTION", "8.1 Phase 6 conflict representation")

code(r'''
# ---- 8.2 Conflict preview -------------------------------------------------------------------------------------------------------------
idx = EXP_DIR / "s6_conflict_index.json"
if idx.is_file():
    R6 = read_json(idx)
    tot = {"pairs": 0, "pairs_valid": 0}
    for k, r in sorted(R6["results"].items()):
        d = (r.get("stages", {}).get("S6_CONFLICT") or {}).get("detail", {})
        tot["pairs"] += int(d.get("pairs") or 0); tot["pairs_valid"] += int(d.get("pairs_valid") or 0)
    print("pedestrian-vehicle pairs:", tot)
    first = next(iter(sorted(R6["results"])), None)
    if first:
        rep = read_json(D["art"] / "phase6_conflict" / PARTS[first]["set"] / first / "conflict_report.json")
        print(first, json.dumps({k: rep[k] for k in ("sample_status", "coordinate_system", "coordinate_unit", "pair_counts", "claims")}, indent=1))
        print("summary:", json.dumps(rep["summary"], default=str)[:1200])
''')

md(r'''
# Section 9 — Phase 7: risk assessment
`assess_risk` per video with **no calibration artifact**: the uncalibrated heuristic score, provisional categories (`INSUFFICIENT_EVIDENCE` where evidence
is missing), `calibrated_probability = null` and status `NOT_CALIBRATED`. Calibration runs only if its prerequisites exist (verified conflict labels and
disjoint calibration groups); here they do not, so calibration is recorded as **BLOCKED** and no probability is produced.
''')

downstream_cell("S7_RISK", "S7_RISK", ["S6_CONFLICT"], "RUN_RISK_ASSESSMENT", "9.1 Phase 7 risk assessment")

code(r'''
# ---- 9.2 Calibration prerequisites (gated, never weakened) + risk preview -------------------------------------------------------------------
cal_reasons = ["no verified conflict labels exist (no collision / near-conflict / interaction annotation with provenance)"]
if not SPLIT["partitions"].get("calibration"):
    cal_reasons.append("the frozen split has no calibration partition (official PIE split)")
TRACKER.set("S7_CALIBRATION", "BLOCKED", reason="; ".join(cal_reasons), calibrated_probability=None, calibration_status="NOT_CALIBRATED")
print("[BLOCKED] Phase 7 calibration:", "; ".join(cal_reasons))
idx = EXP_DIR / "s7_risk_index.json"
if idx.is_file():
    R7 = read_json(idx)
    cats = {}
    for r in R7["results"].values():
        for c, n in ((r.get("stages", {}).get("S7_RISK") or {}).get("detail", {}).get("category_counts") or {}).items():
            cats[c] = cats.get(c, 0) + n
    print("risk category distribution (uncalibrated, provisional thresholds):", dict(sorted(cats.items())))
''')

md(r'''
# Section 10 — Phase 8: early-warning generation
`run_alerts` per video replays the Phase 7 records through the existing deterministic state machine (persistence, hysteresis, escalation /
de-escalation, deduplication, cooldown, evidence loss, invalid / out-of-order input handling, end-of-stream) and writes `alert_events.jsonl.gz`,
`alerts.json`, `summary_report.json`, `run_manifest.json`. Thresholds are the repository values. **Warning lead time is BLOCKED** (no ground-truth event times).
''')

downstream_cell("S8_ALERTS", "S8_ALERTS", ["S7_RISK"], "RUN_EARLY_WARNING", "10.1 Phase 8 early-warning events")

code(r'''
# ---- 10.2 Alert preview -----------------------------------------------------------------------------------------------------------
idx = EXP_DIR / "s8_alerts_index.json"
if idx.is_file():
    R8 = read_json(idx)
    tot = {"n_alerts": 0, "n_events": 0, "alerts_escalated": 0}
    kinds = {}
    for r in R8["results"].values():
        d = (r.get("stages", {}).get("S8_ALERTS") or {}).get("detail", {})
        for k in tot:
            tot[k] += int(d.get(k) or 0)
        for k, n in (d.get("alerts_by_clearance_kind") or {}).items():
            kinds[k] = kinds.get(k, 0) + n
    print("alerts:", tot, "| clearance kinds:", kinds)
    first = next((k for k in sorted(R8["results"]) if (D["art"] / "phase8_alerts" / PARTS[k]["set"] / k / "alerts.json").is_file()), None)
    if first:
        al = read_json(D["art"] / "phase8_alerts" / PARTS[first]["set"] / first / "alerts.json")["alerts"][:3]
        print(first, "first alerts:", json.dumps(al, indent=1, default=str)[:1500])
    TRACKER.set("S8_LEAD_TIME", "BLOCKED", reason="no ground-truth event timestamps: warning lead time and false-alarm rates cannot be validated")
''')

# ===================================================================================================================================== 11
md(r'''
# Section 11 — Phase 9: end-to-end orchestration status
Two execution modes are kept apart:
1. **RAW_VIDEO** — Phase 3 detection and tracking were executed on the MP4s in this experiment (or verified-reused from an earlier experiment with
   identical configuration and weights).
2. **PRECOMPUTED_PHASE3_ARTIFACTS** — the chain started from existing Phase 3 folders; this is **not** raw-video end-to-end inference.

The original Phase 9 runner (`python -m pedsafety.phase9`) is single-video and bound to the pilot reference runs (its Phase 5 model verifies the exact
window-run hash), so the multi-video chain is orchestrated by `pedsafety.multivideo` with the same contracts. Phase 9's independent-evaluation gate is
called unchanged; its single-video runner can optionally be re-run as a pilot regression (clearly separated from this experiment's results).
**No real-time claim** is made: stages are timed separately and offline, and the window builder needs future frames.
''')

code(r'''
# ---- 11.1 Execution mode and per-video chain completeness ------------------------------------------------------------------------------
def no_window_videos():
    wi = load_if(EXP_DIR / "windows_index.json") or {"window_dirs": {}}
    return {k for k, d in wi["window_dirs"].items() if read_json(Path(d) / "run_manifest.json")["counts"]["accepted_windows"] == 0}

def apply_not_applicable(table):
    # a video with zero accepted windows has nothing to forecast: its S5C-S8 outputs are NOT_APPLICABLE, not missing
    for s in ("S5C_EVALUATION", "S6_CONFLICT", "S7_RISK", "S8_ALERTS"):
        for k in no_window_videos():
            if s in table and table[s].get(k) in ("ABSENT", None):
                table[s][k] = "NOT_APPLICABLE_NO_WINDOWS"
    return table

MARKERS = apply_not_applicable(marker_table(STAGE_DIRS, VKEYS, verify="size"))
p3 = read_json(EXP_DIR / "phase3_index.json") if (EXP_DIR / "phase3_index.json").is_file() else {"results": {}, "execution_mode": None}
per_video = {}
for k in VKEYS:
    chain = all(MARKERS[s].get(k) in ("COMPLETE", "NOT_APPLICABLE_NO_WINDOWS") for s in MARKERS)
    st3 = p3["results"].get(k, {}).get("status")
    mode = ("PRECOMPUTED_PHASE3_ARTIFACTS" if p3.get("execution_mode") == "PRECOMPUTED_PHASE3_ARTIFACTS" else
            "RAW_VIDEO_REUSED_FROM_EARLIER_EXPERIMENT" if st3 == "REUSED_VERIFIED" else "RAW_VIDEO" if st3 in ("COMPLETED", "SKIPPED_ALREADY_COMPLETE") else "NOT_EXECUTED")
    per_video[k] = {"execution_mode": mode, "chain_complete": chain, "partial_video": bool(PERCEPTION_MAX_FRAMES)}
n_raw = sum(1 for r in per_video.values() if r["execution_mode"].startswith("RAW_VIDEO") and r["chain_complete"])
n_pre = sum(1 for r in per_video.values() if r["execution_mode"] == "PRECOMPUTED_PHASE3_ARTIFACTS" and r["chain_complete"])
EXECUTION = {"per_video": per_video, "videos_in_scope": len(VKEYS), "raw_video_to_alert_chains_complete": n_raw, "precomputed_artifact_chains_complete": n_pre,
             "dataset_scope": DATASET_SCOPE if not VIDEO_SUBSET else "VIDEO_SUBSET", "partial_videos": bool(PERCEPTION_MAX_FRAMES),
             "statement": ("%d/%d videos ran raw video -> YOLO/ByteTrack -> Phases 4-8 to alerts in this experiment; %d/%d started from precomputed Phase 3 artifacts. "
                           "Offline batch replay; no real-time claim." % (n_raw, len(VKEYS), n_pre, len(VKEYS)))}
write_json(EXP_DIR / "execution_summary.json", EXECUTION)
print(EXECUTION["statement"])
for s, t in MARKERS.items():
    c = {}
    for v in t.values():
        c[v] = c.get(v, 0) + 1
    print("  %-22s %s" % (s, c))
''')

code(r'''
# ---- 11.2 Phase 9 CLI: independent-evaluation status for the frozen split; optional single-video pilot regression ---------------------------
r = subprocess.run([sys.executable, "-m", "pedsafety.phase9", "evaluation-status", "--split-manifest", str(SPLIT_PATH)], capture_output=True, text=True, cwd=REPO)
(D["reports"] / "phase9_evaluation_status.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
print(r.stdout[-2500:] or r.stderr[-2500:])
TRACKER.set("S9_EVALUATION_GATES", "COMPLETED" if r.returncode == 0 else "FAILED", exit_code=r.returncode, gate=PHASE9_GATE["status"])
if RUN_PHASE9_PILOT_REGRESSION:
    with TRACKER.stage("S9_PILOT_REGRESSION") as st:
        if not (Path(PILOT_PHASE3_DIR) / "tracked_agents.jsonl").is_file():
            raise FileNotFoundError("pilot Phase 3 folder not found: %s" % PILOT_PHASE3_DIR)
        cfg = read_json(REPO / "configs/phase9/pipeline_pie_video_0004.json")
        cfg.update({"output_root": str(EXP_DIR / "phase9_pilot_regression"), "report_dir": None})
        cfg["phase3"] = {"dir": PILOT_PHASE3_DIR}
        cfg_path = EXP_DIR / "phase9_pilot_regression_config.json"
        write_json(cfg_path, cfg)
        rid = "pilot_" + SESSION
        a = subprocess.run([sys.executable, "-m", "pedsafety.phase9", "run", "--config", str(cfg_path), "--run-id", rid, "--no-benchmark"], capture_output=True, text=True, cwd=REPO)
        b = subprocess.run([sys.executable, "-m", "pedsafety.phase9", "verify-run", "--run-dir", str(EXP_DIR / "phase9_pilot_regression" / rid), "--config", str(cfg_path)],
                           capture_output=True, text=True, cwd=REPO)
        (D["reports"] / "phase9_pilot_regression.log").write_text(a.stdout + a.stderr + "\n----\n" + b.stdout + b.stderr, encoding="utf-8")
        print(a.stdout[-1500:], b.stdout[-1500:])
        if a.returncode != 0 or b.returncode != 0:
            raise RuntimeError("Phase 9 pilot regression failed (run exit %d, verify exit %d)" % (a.returncode, b.returncode))
        st["detail"] = {"label": "PILOT_REGRESSION single video set01/video_0004, precomputed Phase 3; NOT part of the multi-video results", "run_id": rid}
''')

# ===================================================================================================================================== 12
md(r'''
# Section 12 — Evaluation and reporting
Builds `experiment_report.json` (machine-readable) and `EXPERIMENT_REPORT.md` (human-readable) in the experiment folder: experiment id, dataset inventory,
split, video / set counts, git, environment, dependency versions, model and training configuration, checkpoint paths, per-stage status and runtime,
memory, ADE/FDE (primary) and minADE@K/minFDE@K (oracle, separate), conflict outputs, risk distribution, alert counts, failures, tests, limitations.
''')

code(r'''
# ---- 12.1 Consolidated report (re-built after the final validation in Section 13) -------------------------------------------------------------
DEVIATIONS = [
    "Multi-video Phase 5 training uses pedsafety.multivideo.training (the single-video harness accepts one window run only): split_policy "
    "RECORDING_LEVEL_FROZEN_SPLIT_TRAIN_PARTITION_ONLY, video-homogeneous batches, per-epoch validation monitor (not used for selection), GPU allowed.",
    "Held-out evaluation loads the model through pedsafety.multivideo.training.load_multivideo_model (the P5.7.5 loader requires the training window run); "
    "metrics come from the unchanged evaluate_neural / P5.4 evaluator.",
    "Phase 6-8 run per video through pedsafety.multivideo.downstream with the Phase 9 S6-S8 contracts; sample_status reflects the partition.",
    "Phase 3 sequence ids are <set>_<video> (PIE reuses video numbers across sets); detections.jsonl is stored gzip-compressed.",
    "Window runs keep the generator's built-in label PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT; the experiment label and partitions come from the frozen split.",
] + (["Split scheme %s: %s" % (SPLIT_SCHEME, SPLIT.get("deviations_from_official"))] if SPLIT.get("deviations_from_official") else [])

def make_report(tests=None, validation=None):
    ds = {"results": {}, "failed": []}
    for n in ("s6_conflict_index.json", "s7_risk_index.json", "s8_alerts_index.json"):
        x = load_if(EXP_DIR / n)
        if x:
            for k, r in x["results"].items():
                ds["results"].setdefault(k, {"stages": {}})["stages"].update(r.get("stages", {}))
            ds["failed"] += x.get("failed", [])
    return build_report(experiment_id=EXPERIMENT_ID, exp_dir=EXP_DIR, lock=read_json(EXP_DIR / "experiment_lock.json"), inventory=INVENTORY, split=SPLIT,
                        tracker_data=TRACKER.data, environment=ENVIRONMENT, git=GIT, markers=MARKERS, phase3=load_if(EXP_DIR / "phase3_index.json") or {},
                        training=load_if(TRAIN_DIR / "run_manifest.json"), evaluation=load_if(EVAL_ROOT / "evaluation_summary.json"), downstream=ds,
                        tests=tests, validation=validation, execution=EXECUTION, deviations=DEVIATIONS)

REPORT = make_report()
print((EXP_DIR / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")[:6000])
''')

# ===================================================================================================================================== 13
md(r'''
# Section 13 — Tests and final validation
13.1 runs the repository test suite (unchanged tests; known pre-existing failures are classified, nothing is skipped or weakened) · 13.2 validates
required files, schemas, checkpoint, manifests and split integrity, re-hashes artifacts, and lists failed / skipped / blocked stages · 13.3 rebuilds the
report and prints the final status. **The overall status is COMPLETED only when every check passes.**
''')

code(r'''
# ---- 13.1 Repository tests ----------------------------------------------------------------------------------------------------------
KNOWN_PREEXISTING = {"tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension"}   # documented baseline defect (utils/Interaction.py)
TESTS = {"status": "NOT_RUN"}
if RUN_TESTS:
    with TRACKER.stage("S13_TESTS") as st:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rfE"], capture_output=True, text=True, cwd=REPO, timeout=TEST_TIMEOUT_S)
        (D["reports"] / ("pytest_%s.log" % SESSION)).write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        failed = sorted({l.split(" ")[1] for l in lines if l.startswith(("FAILED ", "ERROR "))})
        unexpected = [f for f in failed if f.split(" - ")[0] not in KNOWN_PREEXISTING]
        TESTS = {"returncode": r.returncode, "summary_line": lines[-1] if lines else "(no output)", "failed": failed, "unexpected_failures": unexpected,
                 "known_preexisting_failures": sorted(set(failed) - set(unexpected)), "wall_s": time.time() - t0,
                 "status": "PASS" if r.returncode == 0 else ("PASS_WITH_KNOWN_PREEXISTING_FAILURES" if failed and not unexpected else "FAIL")}
        print(TESTS["summary_line"]); print("unexpected failures:", unexpected or "none")
        st["detail"] = TESTS
        if TESTS["status"] == "FAIL":
            raise RuntimeError("unexpected test failures: %s" % unexpected[:20])
''')

code(r'''
# ---- 13.2 Final validation ------------------------------------------------------------------------------------------------------------
CHECKS = []
def check(name, fn):
    try:
        res, detail = fn()
    except Exception as exc:
        res, detail = "FAIL", "%s: %s" % (type(exc).__name__, str(exc)[:400])
    CHECKS.append({"name": name, "result": res, "detail": detail})
    print("%-48s %-10s %s" % (name, res, str(detail)[:160]))

def c_stages():
    bad = {k: v["status"] for k, v in TRACKER.data["stages"].items() if v["status"] in ("FAILED", "RUNNING")}
    inc = {k: v["status"] for k, v in TRACKER.data["stages"].items() if v["status"] in ("BLOCKED_UPSTREAM", "SKIPPED_DISABLED", "COMPLETED_WITH_FAILURES")}
    return ("FAIL" if bad else "INCOMPLETE" if inc else "PASS"), {"failed": bad, "incomplete": inc}
def c_files():
    t = apply_not_applicable(marker_table(STAGE_DIRS, VKEYS, verify="sha256" if FULL_HASH_VERIFICATION else "size"))
    notok = {s: {k: v for k, v in x.items() if v not in ("COMPLETE", "NOT_APPLICABLE_NO_WINDOWS")} for s, x in t.items()}
    notok = {s: x for s, x in notok.items() if x}
    corrupt = any(v == "CORRUPT" for x in notok.values() for v in x.values())
    return ("FAIL" if corrupt else "INCOMPLETE" if notok else "PASS"), {"not_complete": notok, "hash_verified": FULL_HASH_VERIFICATION}
def c_schemas():
    out = {}
    p3i = read_json(EXP_DIR / "phase3_index.json")
    for part in ("train", "validation", "calibration", "test"):
        k = next((k for k in sorted(VKEYS) if PARTS[k]["partition"] == part and MARKERS.get("S8_ALERTS", {}).get(k) == "COMPLETE"), None)
        if k:
            s = VKEYS[k]
            out[k] = validate_schemas(Path(p3i["results"][k]["dir"]), D["art"] / "phase8_alerts" / s / k, D["art"] / "phase7_risk" / s / k)
    for p in sorted(EXP_DIR.rglob("*.json")):
        if "_quarantine" not in p.parts:
            json.loads(p.read_text(encoding="utf-8"))
    if not out:
        return "INCOMPLETE", "no video completed every stage"
    return ("PASS" if all(v["ok"] for v in out.values()) else "FAIL"), {k: v["ok"] for k, v in out.items()}
def c_checkpoint():
    if not (TRAIN_DIR / "run_manifest.json").is_file():
        return "NOT_RUN", "no completed training run"
    _, man, name, rec, _, ver, _ = load_multivideo_model(TRAIN_DIR, device="cpu")
    return "PASS", {"checkpoint": name, "sha256": ver["checkpoint_file_sha256"][:16], "epoch": ver["checkpoint_epoch"]}
def c_manifests():
    need = ["experiment_lock.json", "experiment_status.json", "inventory.json", "split_manifest.json", "phase3_index.json", "phase4_index.json", "windows_index.json",
            "execution_summary.json", "s6_conflict_index.json", "s7_risk_index.json", "s8_alerts_index.json"]
    missing = [n for n in need if not (EXP_DIR / n).is_file()]
    if not (EVAL_ROOT / "evaluation_summary.json").is_file():
        missing.append("artifacts/phase5_evaluation/evaluation_summary.json")
    env = list(D["env"].glob("environment_*.json"))
    return ("INCOMPLETE" if missing or not env else "PASS"), {"missing": missing, "environment_records": len(env)}
def c_split():
    sp = load_frozen_split(SPLIT_PATH)
    integ = check_split_integrity(sp, INVENTORY)
    probs = list(integ["problems"])
    if (TRAIN_DIR / "run_manifest.json").is_file():
        tv = [r["video"] for r in read_json(TRAIN_DIR / "run_manifest.json")["inputs"]["train_videos"]]
        probs += ["trained on non-train video %s" % v for v in tv if PARTS[v]["partition"] != "train"]
    if (EVAL_ROOT / "evaluation_summary.json").is_file():
        ev = read_json(EVAL_ROOT / "evaluation_summary.json")
        probs += ["evaluation partition mismatch for %s" % k for k, r in ev["per_video"].items() if r.get("partition") != PARTS[k]["partition"]]
    return ("FAIL" if probs else "PASS"), {"split_id": sp["split_id"], "problems": probs}
def c_tests():
    if not RUN_TESTS:
        return "SKIPPED", "RUN_TESTS=False"
    return ("FAIL" if TESTS.get("status") in ("FAIL", None, "NOT_RUN") else "PASS"), TESTS.get("summary_line")
def c_claims():
    rep = read_json(EXP_DIR / "experiment_report.json")
    bad = [k for k, v in rep["claims"].items() if v]
    return ("FAIL" if bad else "PASS"), {"unsupported_claims_set": bad}

if RUN_END_TO_END_VALIDATION:
    for n, f in (("stage statuses (no failed / running stage)", c_stages), ("required per-video outputs + hashes", c_files), ("JSON / JSONL schemas", c_schemas),
                 ("checkpoint existence and integrity", c_checkpoint), ("experiment manifest completeness", c_manifests), ("dataset split integrity", c_split),
                 ("repository tests", c_tests), ("no unsupported claim flagged true", c_claims)):
        check(n, f)
    VALIDATION = {"overall_status": final_status(CHECKS), "checks": CHECKS, "blocked_scientific_items": BLOCKED_SCIENTIFIC_ITEMS, "validated_utc": utc_now()}
    write_json(EXP_DIR / "final_validation.json", VALIDATION)
else:
    VALIDATION = {"overall_status": "NOT_VALIDATED", "checks": []}
''')

code(r'''
# ---- 13.3 Final report and execution summary --------------------------------------------------------------------------------------------
REPORT = make_report(tests=TESTS, validation=VALIDATION)
summ = TRACKER.summary()
print("=" * 100)
print("EXPERIMENT", EXPERIMENT_ID)
print("FINAL STATUS:", VALIDATION["overall_status"])
for status in ("FAILED", "COMPLETED_WITH_FAILURES", "BLOCKED_UPSTREAM", "SKIPPED_DISABLED", "BLOCKED"):
    if summ.get(status):
        print("  %-24s %s" % (status, summ[status]))
print("  completed / reused     ", summ.get("COMPLETED", []) + summ.get("REUSED_VERIFIED", []))
print("Blocked scientific items (need new evidence, not more compute):")
for k, v in BLOCKED_SCIENTIFIC_ITEMS.items():
    print("  -", k, ":", v)
print(EXECUTION["statement"])
print("report:", EXP_DIR / "EXPERIMENT_REPORT.md")
if VALIDATION["overall_status"] != "COMPLETED":
    print("\n*** The experiment is NOT complete: see the FAILED / INCOMPLETE checks above. Fix them and re-run with EXPERIMENT_ID=%r. ***" % EXPERIMENT_ID)
''')


def build(out: Path) -> Path:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        c = {"cell_type": kind, "id": "cell-%03d" % i, "metadata": {}, "source": source}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
                                       "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    p = build(Path(a.out))
    print("wrote %s (%d cells)" % (p, len(CELLS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
