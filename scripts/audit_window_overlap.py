
import json
from pathlib import Path

import pandas as pd

CSV_PATH = Path("Data/HBS/hbs.csv")
OUTPUT_PATH = Path(
    "experiments/PCG_T4_5_Epoch_Run/reports/"
    "task5_window_overlap_audit.json"
)

SEQ_LENGTH = 12
TEST_PORTION = 0.31
STRIDE = 1


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    required = {"frame_id", "agent_id", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Match the original DataLoader pedestrian filtering.
    pedestrians = df[
        df["label"].isin(["pedestrian", "ped"])
    ].copy()

    frame_ids = sorted(
        pedestrians["frame_id"].unique().tolist()
    )

    total_frames = len(frame_ids)

    # Match the original DataLoader calculation.
    total_windows = total_frames - SEQ_LENGTH

    if total_windows <= 0:
        raise ValueError("Not enough frames for 12-frame windows.")

    test_count = int(TEST_PORTION * total_windows)

    # Original implementation creates windows with stride 1,
    # then assigns the first 31% to test and the rest to train.
    test_starts = list(range(0, test_count))
    train_starts = list(range(test_count, total_windows))

    # Map frame indices to actual pedestrian-frame observations.
    frame_agents = {}

    for frame_id, group in pedestrians.groupby("frame_id"):
        frame_agents[frame_id] = set(
            (int(frame_id), int(agent_id))
            for agent_id in group["agent_id"].unique()
        )

    def window_frames(start):
        return frame_ids[start:start + SEQ_LENGTH]

    def window_agent_frames(start):
        result = set()

        for frame_id in window_frames(start):
            result.update(frame_agents.get(frame_id, set()))

        return result

    # Calculate temporal overlap between test and train windows.
    overlapping_window_pairs = 0
    test_windows_with_temporal_overlap = set()
    shared_temporal_frames = set()

    train_frame_sets = {
        start: set(window_frames(start))
        for start in train_starts
    }

    for test_start in test_starts:
        test_frames = set(window_frames(test_start))

        for train_start, train_frames in train_frame_sets.items():
            shared = test_frames.intersection(train_frames)

            if shared:
                overlapping_window_pairs += 1
                test_windows_with_temporal_overlap.add(test_start)
                shared_temporal_frames.update(shared)

    # Calculate overlap at the pedestrian-frame observation level.
    train_agent_frame_union = set()

    for start in train_starts:
        train_agent_frame_union.update(
            window_agent_frames(start)
        )

    test_agent_frame_union = set()

    for start in test_starts:
        test_agent_frame_union.update(
            window_agent_frames(start)
        )

    shared_agent_frames = (
        test_agent_frame_union.intersection(
            train_agent_frame_union
        )
    )

    test_windows_with_shared_agents = 0

    for start in test_starts:
        observations = window_agent_frames(start)

        if observations.intersection(train_agent_frame_union):
            test_windows_with_shared_agents += 1

    # Exact duplicate window start positions should be zero
    # because the two index ranges are disjoint.
    exact_duplicate_starts = (
        set(test_starts).intersection(set(train_starts))
    )

    report = {
        "dataset": str(CSV_PATH),
        "split_method": (
            "Generate sliding windows first; assign first "
            "31 percent to test and remaining windows to train."
        ),
        "sequence_length": SEQ_LENGTH,
        "stride": STRIDE,
        "test_portion": TEST_PORTION,
        "total_unique_pedestrian_frames": total_frames,
        "total_windows": total_windows,
        "test_windows": len(test_starts),
        "train_windows": len(train_starts),
        "test_start_index_range": [
            min(test_starts) if test_starts else None,
            max(test_starts) if test_starts else None
        ],
        "train_start_index_range": [
            min(train_starts) if train_starts else None,
            max(train_starts) if train_starts else None
        ],
        "temporal_overlap": {
            "cross_split_overlapping_window_pairs":
                overlapping_window_pairs,
            "test_windows_with_temporal_overlap":
                len(test_windows_with_temporal_overlap),
            "shared_unique_frame_count":
                len(shared_temporal_frames),
            "shared_frame_ids": sorted(
                int(x) for x in shared_temporal_frames
            ),
            "percentage_of_test_windows_affected": (
                100 * len(test_windows_with_temporal_overlap)
                / len(test_starts)
                if test_starts else 0
            )
        },
        "pedestrian_frame_overlap": {
            "unique_test_pedestrian_frame_observations":
                len(test_agent_frame_union),
            "unique_train_pedestrian_frame_observations":
                len(train_agent_frame_union),
            "shared_pedestrian_frame_observations":
                len(shared_agent_frames),
            "percentage_of_test_observations_shared": (
                100 * len(shared_agent_frames)
                / len(test_agent_frame_union)
                if test_agent_frame_union else 0
            ),
            "test_windows_with_shared_pedestrian_frames":
                test_windows_with_shared_agents
        },
        "exact_duplicate_windows": len(exact_duplicate_starts),
        "interpretation": (
            "Temporal overlap exists across the split boundary. "
            "The pedestrian-frame overlap statistics quantify "
            "the actual shared observations."
        )
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n========== TASK 5 WINDOW OVERLAP AUDIT ==========")
    print(json.dumps(report, indent=4))
    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
