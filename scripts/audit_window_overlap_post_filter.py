
import json
from pathlib import Path

import pandas as pd

CSV_PATH = Path("Data/HBS/hbs.csv")

RAW_REPORT_PATH = Path(
    "experiments/PCG_T4_5_Epoch_Run/reports/"
    "task5_window_overlap_audit.json"
)

OUTPUT_PATH = Path(
    "experiments/PCG_T4_5_Epoch_Run/reports/"
    "task5_window_overlap_post_filter_audit.json"
)

SEQ_LENGTH = 12
TEST_PORTION = 0.31


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    required = {"frame_id", "agent_id", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Match the original DataLoader pedestrian labels.
    pedestrians = df[
        df["label"].isin(["pedestrian", "ped"])
    ].copy()

    # IMPORTANT:
    # Use all CSV frame IDs to construct windows, not only
    # frames containing pedestrians.
    all_frame_ids = sorted(
        df["frame_id"].unique().tolist()
    )

    pedestrian_frame_ids = sorted(
        pedestrians["frame_id"].unique().tolist()
    )

    total_frames = len(all_frame_ids)

    # Match original DataLoader window-count calculation.
    total_windows = total_frames - SEQ_LENGTH

    if total_windows <= 0:
        raise ValueError("Not enough frames for 12-frame windows.")

    test_count = int(TEST_PORTION * total_windows)

    test_starts = list(range(0, test_count))
    train_starts = list(range(test_count, total_windows))

    # Build pedestrian ID sets for every frame.
    frame_pedestrians = {
        frame_id: set()
        for frame_id in all_frame_ids
    }

    for frame_id, group in pedestrians.groupby("frame_id"):
        frame_pedestrians[frame_id] = set(
            int(agent_id)
            for agent_id in group["agent_id"].unique()
        )

    def get_window_frames(start):
        return all_frame_ids[start:start + SEQ_LENGTH]

    # Reproduce DataLoader.filtering():
    # Keep only pedestrian IDs present in ALL 12 frames.
    def filter_window(start):
        frames = get_window_frames(start)

        if len(frames) != SEQ_LENGTH:
            return None

        common_ids = set(frame_pedestrians[frames[0]])

        for frame_id in frames[1:]:
            common_ids.intersection_update(
                frame_pedestrians[frame_id]
            )

        if not common_ids:
            return None

        # DataLoader.filtering() retains only common_ids
        # in every frame of the sequence.
        filtered_observations = {
            (frame_id, ped_id)
            for frame_id in frames
            for ped_id in common_ids
        }

        return {
            "start": start,
            "frames": set(frames),
            "common_pedestrian_ids": common_ids,
            "observations": filtered_observations
        }

    # Apply filtering independently to both original splits.
    test_retained = [
        result
        for start in test_starts
        if (result := filter_window(start)) is not None
    ]

    train_retained = [
        result
        for start in train_starts
        if (result := filter_window(start)) is not None
    ]

    # --------------------------------------------------------
    # Temporal overlap after filtering
    # --------------------------------------------------------

    overlapping_pairs = 0
    affected_test_starts = set()
    shared_temporal_frames = set()

    train_windows = {
        item["start"]: item["frames"]
        for item in train_retained
    }

    for test_item in test_retained:
        test_frames = test_item["frames"]

        for train_start, train_frames in train_windows.items():
            shared = test_frames.intersection(train_frames)

            if shared:
                overlapping_pairs += 1
                affected_test_starts.add(test_item["start"])
                shared_temporal_frames.update(shared)

    # --------------------------------------------------------
    # Pedestrian-frame overlap after filtering
    # --------------------------------------------------------

    train_observation_union = set()
    test_observation_union = set()

    for item in train_retained:
        train_observation_union.update(item["observations"])

    for item in test_retained:
        test_observation_union.update(item["observations"])

    shared_observations = (
        test_observation_union.intersection(
            train_observation_union
        )
    )

    test_windows_with_shared_observations = 0

    for item in test_retained:
        if item["observations"].intersection(
            train_observation_union
        ):
            test_windows_with_shared_observations += 1

    # --------------------------------------------------------
    # Compare with the original raw audit
    # --------------------------------------------------------

    raw_comparison = {
        "available": False
    }

    if RAW_REPORT_PATH.exists():
        with open(RAW_REPORT_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_comparison = {
            "available": True,
            "raw_total_windows": raw.get("total_windows"),
            "raw_test_windows": raw.get("test_windows"),
            "raw_train_windows": raw.get("train_windows"),
            "raw_cross_split_overlapping_pairs":
                raw.get("temporal_overlap", {}).get(
                    "cross_split_overlapping_window_pairs"
                )
        }

    report = {
        "task": "Phase 2 Task 5 - Post-filtering Window Overlap Audit",
        "dataset": str(CSV_PATH),

        "window_generation": {
            "timeline": "All unique frame IDs in the original CSV",
            "total_unique_csv_frames": total_frames,
            "unique_frames_containing_pedestrians":
                len(pedestrian_frame_ids),
            "sequence_length": SEQ_LENGTH,
            "stride": 1,
            "total_windows": total_windows,
            "test_portion": TEST_PORTION,
            "test_windows_before_filtering": len(test_starts),
            "train_windows_before_filtering": len(train_starts)
        },

        "filtering": {
            "enabled_for_training": True,
            "enabled_for_testing": True,
            "rule": (
                "Retain a sequence only if at least one pedestrian "
                "ID is present throughout all 12 frames. Retain "
                "only those common pedestrian IDs in every frame."
            ),
            "test_windows_removed":
                len(test_starts) - len(test_retained),
            "train_windows_removed":
                len(train_starts) - len(train_retained),
            "test_windows_retained": len(test_retained),
            "train_windows_retained": len(train_retained)
        },

        "post_filtering_temporal_overlap": {
            "cross_split_overlapping_window_pairs":
                overlapping_pairs,
            "test_windows_with_temporal_overlap":
                len(affected_test_starts),
            "shared_unique_frame_count":
                len(shared_temporal_frames),
            "shared_frame_ids": sorted(
                int(x) for x in shared_temporal_frames
            ),
            "percentage_of_retained_test_windows_affected": (
                100 * len(affected_test_starts) / len(test_retained)
                if test_retained else 0
            )
        },

        "post_filtering_pedestrian_frame_overlap": {
            "unique_test_pedestrian_frame_observations":
                len(test_observation_union),
            "unique_train_pedestrian_frame_observations":
                len(train_observation_union),
            "shared_pedestrian_frame_observations":
                len(shared_observations),
            "percentage_of_retained_test_observations_shared": (
                100 * len(shared_observations)
                / len(test_observation_union)
                if test_observation_union else 0
            ),
            "test_windows_with_shared_pedestrian_frame_observations":
                test_windows_with_shared_observations
        },

        "raw_audit_comparison": raw_comparison,

        "limitations": [
            "This script reconstructs DataLoader filtering from the raw CSV.",
            "It assumes all_frame_data contains one entry for every unique CSV frame ID.",
            "Verify the reconstructed frame count against the original preprocessed HBS pickle.",
            "Filtering does not remove temporal overlap automatically.",
            "The split still uses a single chronological scene rather than scene-independent evaluation."
        ]
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n========== POST-FILTERING TASK 5 AUDIT ==========")
    print(json.dumps(report, indent=4))
    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
