
from pathlib import Path
import json
import hashlib
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "PCG_T4_5_Epoch_Run"
REPORTS = EXPERIMENT / "reports"
LOGS = EXPERIMENT / "logs"

REPORTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

EVIDENCE_FILE = LOGS / "task6_task7_source_evidence.txt"

def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_context(text, patterns, radius=5):
    lines = text.splitlines()
    found = []
    seen = set()

    for i, line in enumerate(lines):
        if any(re.search(p, line, re.IGNORECASE) for p in patterns):
            start = max(0, i - radius)
            end = min(len(lines), i + radius + 1)
            key = (start, end)

            if key in seen:
                continue

            seen.add(key)
            found.append(
                f"\n--- Lines {start + 1}-{end} ---\n" +
                "\n".join(
                    f"{j + 1}: {lines[j]}"
                    for j in range(start, end)
                )
            )

    return "\n".join(found) if found else "NO MATCH FOUND"

def find_files():
    candidates = [
        ROOT / "utils" / "DataLoader.py",
        ROOT / "test.py",
        ROOT / "train.py",
        ROOT / "utils" / "helper.py",
    ]

    discovered = []

    for p in candidates:
        if p.exists():
            discovered.append(p)

    for base in [ROOT / "Data", ROOT / "data", ROOT / "dataset"]:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and (
                    p.suffix.lower() in {
                        ".csv", ".txt", ".json", ".yaml",
                        ".yml", ".md", ".xml"
                    }
                ):
                    discovered.append(p)

    return list(dict.fromkeys(discovered))

files = find_files()

# Preserve evidence as text with file names and line numbers.
evidence_parts = []

for path in files:
    text = read_text(path)
    relative = str(path.relative_to(ROOT))

    evidence_parts.append(
        f"\n\n{'=' * 80}\nFILE: {relative}\n"
        f"SHA256: {sha256(path)}\n{'=' * 80}\n"
    )

    if path.name.lower() == "dataloader.py":
        evidence_parts.append(
            extract_context(
                text,
                [
                    r"fps",
                    r"timestamp",
                    r"downsample_step",
                    r"frame_preprocess",
                    r"frameList",
                    r"all_frame_data",
                    r"pos_x",
                    r"pos_y",
                    r"vel_x",
                    r"vel_y",
                    r"label",
                    r"pedestrian",
                    r"cart",
                    r"vehicle",
                    r"trajectories_train",
                    r"trajectories_test",
                    r"seq_length",
                    r"acceleration",
                    r"min_frame",
                    r"max_frame",
                ],
                radius=7,
            )
        )
    elif path.name.lower() == "test.py":
        evidence_parts.append(
            extract_context(
                text,
                [
                    r"sample_size",
                    r"min.*ade",
                    r"get_mean_error",
                    r"get_final_error",
                    r"true_Pedlist",
                    r"Vehlist",
                    r"x_seq_veh",
                    r"tstep",
                    r"sample",
                ],
                radius=6,
            )
        )
    else:
        evidence_parts.append(
            extract_context(
                text,
                [
                    r"coordinate",
                    r"homography",
                    r"calibration",
                    r"meter",
                    r"pixel",
                    r"timestamp",
                    r"fps",
                    r"dataset",
                    r"collision",
                    r"get_mean_error",
                    r"get_final_error",
                ],
                radius=4,
            )
        )

EVIDENCE_FILE.write_text(
    "\n".join(evidence_parts),
    encoding="utf-8"
)

created = datetime.now(timezone.utc).isoformat()

# Task 6: Protocol design, not a claim that corrected evaluation
# has already been implemented or executed.
task6 = {
    "task": 6,
    "title": "Corrected Validation Protocol",
    "status": "DESIGN_COMPLETE_PENDING_REVIEW",
    "created_utc": created,
    "scope": {
        "type": "protocol_design_and_audit",
        "baseline_retrained": False,
        "original_source_modified": False,
        "checkpoints_modified": False,
        "authoritative_predictions_modified": False,
        "historical_metrics_modified": False
    },
    "historical_baseline_context": {
        "method": "PCG Collision Grid method 4",
        "observation_frames": 6,
        "prediction_frames": 6,
        "sample_size": 5,
        "ade": 0.5587965606209528,
        "fde": 1.1381949186325073,
        "oracle_selection": True,
        "future_information_leakage": True,
        "interpretation": (
            "Historical metrics are retained as originally reported. "
            "They are not leakage-free deployment metrics."
        )
    },
    "identified_evaluation_problems": [
        {
            "id": "T6-01",
            "issue": "Best-of-five oracle selection",
            "evidence": (
                "Original test pipeline selects the sample with minimum "
                "future-ground-truth ADE and takes FDE from that sample."
            ),
            "consequence": (
                "The selected sample cannot be selected at deployment "
                "because future ground truth is unavailable."
            ),
            "correction": (
                "Use deterministic single-sample prediction or a "
                "predeclared stochastic aggregation/selection rule "
                "that does not inspect future ground truth."
            )
        },
        {
            "id": "T6-02",
            "issue": "Future pedestrian information in rollout",
            "evidence": (
                "Original rollout uses future pedestrian lists/presence "
                "to construct subsequent interaction grids."
            ),
            "consequence": (
                "Future actor presence is supplied to the rollout."
            ),
            "correction": (
                "Use only observed tracks, tracker outputs, predicted "
                "positions, and an explicit track birth/death policy."
            )
        },
        {
            "id": "T6-03",
            "issue": "Future vehicle information in method-4 rollout",
            "evidence": (
                "Original method-4 rollout uses future vehicle lists "
                "and future vehicle states when constructing later "
                "interaction grids."
            ),
            "consequence": (
                "Future vehicle states/presence are unavailable in "
                "real-time deployment."
            ),
            "correction": (
                "Propagate observed vehicle tracks using a motion model "
                "or tracker predictions; do not use future annotations."
            )
        },
        {
            "id": "T6-04",
            "issue": "Temporal overlap across retained train/test windows",
            "evidence": (
                "Authoritative Task 5 audit found 66 overlapping "
                "cross-split window pairs and 11 shared timeline "
                "frame indices."
            ),
            "consequence": (
                "Some test windows share temporal frames with training "
                "windows."
            ),
            "correction": (
                "Apply a chronological purge boundary and preferably "
                "scene-independent evaluation where possible."
            )
        }
    ],
    "corrected_protocol": {
        "observation_frames": 6,
        "prediction_frames": 6,
        "allowed_inputs": [
            "Observed pedestrian positions and velocities",
            "Observed vehicle positions and velocities",
            "Observed actor IDs from a real tracker",
            "Camera calibration only if independently verified",
            "Tracker-estimated actor states based on observations"
        ],
        "prohibited_inputs": [
            "Future pedestrian ground-truth positions during rollout",
            "Future pedestrian ground-truth IDs or presence lists",
            "Future vehicle ground-truth positions during rollout",
            "Future vehicle ground-truth IDs or presence lists",
            "Future annotations used to construct interaction grids",
            "Future ground truth used to select prediction samples"
        ],
        "actor_management": {
            "track_initialization": (
                "Initialize tracks only from detections in observed frames."
            ),
            "track_identity": (
                "Maintain persistent IDs through an online tracker."
            ),
            "track_propagation": (
                "Use tracker predictions or a documented motion model."
            ),
            "track_birth": (
                "A new actor may enter only after an actual detector "
                "observation during deployment."
            ),
            "track_death": (
                "Use a documented missing-observation timeout or "
                "tracker termination policy."
            ),
            "ground_truth_future_actor_lists": "PROHIBITED"
        },
        "interaction_grid_rollout": (
            "At each future step, build interaction inputs from "
            "predicted actor states and deployment-feasible track "
            "estimates. Never query future annotation arrays."
        ),
        "prediction_selection": {
            "primary": "Deterministic single-sample prediction",
            "alternative": (
                "Predeclared stochastic aggregation such as mean "
                "trajectory or distribution parameters."
            ),
            "oracle_best_of_n": (
                "May be shown only as a separately labelled oracle "
                "upper bound, never as the deployment metric."
            )
        },
        "data_split": {
            "preferred": (
                "Scene/camera-independent split when multiple scenes "
                "are available."
            ),
            "fallback": (
                "Chronological train/validation/test split with "
                "purge gap."
            ),
            "purge_requirement": (
                "No 12-frame sequence may cross a split boundary. "
                "The purge must account for the complete input plus "
                "prediction window."
            ),
            "report_split_type": True
        },
        "evaluation_metrics": [
            "ADE",
            "FDE",
            "Pedestrian-pedestrian conflict/collision rate",
            "Pedestrian-vehicle conflict/collision rate",
            "Brier score for predicted conflict probabilities",
            "Expected Calibration Error (ECE)",
            "Reliability diagram",
            "Precision and recall at declared alert thresholds",
            "Warning lead time",
            "Missed-conflict rate",
            "False-alert rate",
            "Inference latency and FPS"
        ]
    },
    "acceptance_criteria": [
        "No future ground-truth actor positions or presence enter rollout.",
        "No future ground-truth vehicle states or presence enter rollout.",
        "No future ground truth is used to select a prediction sample.",
        "All interaction grids are generated from observed or predicted states.",
        "Train/validation/test windows obey a documented purge policy.",
        "Historical oracle/leaky metrics are kept separate from corrected metrics.",
        "The corrected evaluation code is tested with future annotations hidden."
    ],
    "not_yet_claimed": [
        "The corrected protocol has not yet been implemented.",
        "No leakage-free metrics have yet been measured.",
        "No deployment FPS has yet been established.",
        "No calibrated conflict probabilities have yet been validated."
    ],
    "source_evidence_file": str(EVIDENCE_FILE.relative_to(ROOT))
}

# Task 7: Evidence is intentionally separated from assumptions.
task7 = {
    "task": 7,
    "title": "HBS Coordinate and Data Provenance Audit",
    "status": "EVIDENCE_COLLECTION_PENDING_REVIEW",
    "created_utc": created,
    "scope": {
        "original_dataset_modified": False,
        "original_preprocessed_pickles_modified": False,
        "baseline_source_modified": False
    },
    "source_code_observations_to_verify": {
        "dataloader_file": "utils/DataLoader.py",
        "csv_columns_reported": [
            "frame_id",
            "agent_id",
            "pos_x",
            "pos_y",
            "vel_x",
            "vel_y",
            "label",
            "timestamp"
        ],
        "pedestrian_labels_reported": [
            "pedestrian",
            "ped"
        ],
        "vehicle_labels_reported": [
            "car",
            "cart",
            "veh"
        ],
        "fps_assumption_reported": 2,
        "downsample_step_reported": 1,
        "timestamp_interval_seconds_reported": 0.5,
        "acceleration_derivation_reported": (
            "Velocity differences divided by the DataLoader timestamp."
        )
    },
    "coordinate_audit": {
        "pos_x": {
            "variable_name": "pos_x",
            "semantic_meaning": "UNVERIFIED",
            "unit": "UNKNOWN",
            "coordinate_space": "UNKNOWN"
        },
        "pos_y": {
            "variable_name": "pos_y",
            "semantic_meaning": "UNVERIFIED",
            "unit": "UNKNOWN",
            "coordinate_space": "UNKNOWN"
        },
        "vel_x": {
            "variable_name": "vel_x",
            "semantic_meaning": "UNVERIFIED",
            "unit": "UNKNOWN",
            "coordinate_space": "UNKNOWN"
        },
        "vel_y": {
            "variable_name": "vel_y",
            "semantic_meaning": "UNVERIFIED",
            "unit": "UNKNOWN",
            "coordinate_space": "UNKNOWN"
        },
        "homography": {
            "availability": "NOT_ESTABLISHED",
            "calibration_file": "NOT_ESTABLISHED",
            "world_coordinate_conversion": "NOT_ESTABLISHED"
        }
    },
    "timing_audit": {
        "source_assumed_fps": 2,
        "source_assumed_timestamp_interval_seconds": 0.5,
        "interpretation": (
            "These are DataLoader assumptions. They do not independently "
            "prove the original recording frame rate."
        ),
        "observation_duration_seconds_under_assumption": 3.0,
        "prediction_duration_seconds_under_assumption": 3.0,
        "total_sequence_duration_seconds_under_assumption": 6.0
    },
    "provenance": {
        "dataset_expansion": "UNVERIFIED",
        "original_publisher": "UNVERIFIED",
        "paper_or_benchmark_citation": "UNVERIFIED",
        "license": "UNVERIFIED",
        "collection_methodology": "UNVERIFIED",
        "scene_or_camera_identity": "UNVERIFIED",
        "number_of_scenes_or_cameras": "UNVERIFIED"
    },
    "known_risks": [
        "Variable names do not prove coordinate units.",
        "A 2 FPS DataLoader assumption does not prove source video FPS.",
        "Pixel-space velocities must not be interpreted as metres/second.",
        "World-coordinate claims require calibration or dataset documentation.",
        "Homography availability must be established from an actual file or source.",
        "Dataset citation and license must be supported by original metadata or publication."
    ],
    "required_follow_up_evidence": [
        "Original dataset README or documentation",
        "Dataset paper or official source URL",
        "Original CSV sample and header",
        "Video metadata or dataset FPS documentation",
        "Calibration/homography file or explicit confirmation of absence",
        "Coordinate and velocity unit documentation",
        "License or terms of use"
    ],
    "source_evidence_file": str(EVIDENCE_FILE.relative_to(ROOT))
}

def write_json(name, data):
    path = REPORTS / name
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"CREATED: {path}")
    print(f"SHA256: {sha256(path)}")

write_json("task6_corrected_validation_protocol.json", task6)
write_json("task7_hbs_coordinate_provenance_audit.json", task7)

print("\nTASK 6 + TASK 7 REPORT GENERATION COMPLETE")
print(f"Source evidence: {EVIDENCE_FILE}")
print("Review the evidence before changing either report status.")
