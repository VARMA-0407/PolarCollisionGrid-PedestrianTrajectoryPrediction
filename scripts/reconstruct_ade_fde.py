
import pickle
import json
from pathlib import Path
import numpy as np

RESULT_PATH = Path(
    "experiments/PCG_T4_5_Epoch_Run/results/test_results_epoch4.pkl"
)

OUTPUT_PATH = Path(
    "experiments/PCG_T4_5_Epoch_Run/results/ade_fde_reconstruction.json"
)

EXPECTED_ADE = 0.5587965606209528
EXPECTED_FDE = 1.1381949186325073


def calculate_original_style_metrics(result):
    """
    Independently reproduce get_mean_error and get_final_error
    using the saved, already-selected prediction.
    """

    gt = np.asarray(result[0], dtype=np.float32)
    pred = np.asarray(result[1], dtype=np.float32)

    presence = result[2]
    lookup = result[3]
    obs_length = int(result[5])

    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}"
        )

    pred_length = gt.shape[0] - obs_length

    gt_future = gt[obs_length:, :, :2]
    pred_future = pred[obs_length:, :, :2]
    future_presence = presence[obs_length:]

    observed_ids = set()
    for frame_ids in presence[:obs_length]:
        observed_ids.update(int(pid) for pid in frame_ids)

    # Reproduce original ADE:
    # Average eligible agents at each timestep, then mean over
    # every prediction timestep (including empty timesteps as zero).
    timestep_errors = []

    for t in range(pred_length):
        total_error = np.float32(0.0)
        counter = 0

        for pid in future_presence[t]:
            pid = int(pid)

            if pid not in observed_ids:
                continue

            if pid not in lookup:
                raise KeyError(f"Pedestrian {pid} missing from lookup")

            idx = lookup[pid]

            distance = np.linalg.norm(
                pred_future[t, idx] - gt_future[t, idx]
            )

            total_error = np.float32(total_error + distance)
            counter += 1

        if counter != 0:
            total_error = np.float32(total_error / counter)

        timestep_errors.append(total_error)

    ade = np.mean(
        np.asarray(timestep_errors, dtype=np.float32),
        dtype=np.float32
    )

    # Reproduce original FDE at the final prediction timestep.
    total_final_error = np.float32(0.0)
    final_counter = 0

    for pid in future_presence[-1]:
        pid = int(pid)

        if pid not in observed_ids:
            continue

        if pid not in lookup:
            raise KeyError(f"Pedestrian {pid} missing from lookup")

        idx = lookup[pid]

        distance = np.linalg.norm(
            pred_future[-1, idx] - gt_future[-1, idx]
        )

        total_final_error = np.float32(
            total_final_error + distance
        )
        final_counter += 1

    if final_counter != 0:
        total_final_error = np.float32(
            total_final_error / final_counter
        )

    return float(ade), float(total_final_error)


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(RESULT_PATH)

    with open(RESULT_PATH, "rb") as f:
        results = pickle.load(f)

    if not results:
        raise ValueError("Prediction artifact is empty")

    batch_ades = []
    batch_fdes = []

    for i, result in enumerate(results):
        if len(result) != 12:
            raise ValueError(
                f"Unexpected tuple length at entry {i}: {len(result)}"
            )

        ade, fde = calculate_original_style_metrics(result)

        batch_ades.append(ade)
        batch_fdes.append(fde)

    # Original test.py averages batch-level metrics equally.
    reconstructed_ade = float(
        np.mean(np.asarray(batch_ades, dtype=np.float32))
    )

    reconstructed_fde = float(
        np.mean(np.asarray(batch_fdes, dtype=np.float32))
    )

    ade_difference = abs(reconstructed_ade - EXPECTED_ADE)
    fde_difference = abs(reconstructed_fde - EXPECTED_FDE)

    report = {
        "artifact": str(RESULT_PATH),
        "number_of_batches": len(results),
        "reported_metrics": {
            "ADE": EXPECTED_ADE,
            "FDE": EXPECTED_FDE
        },
        "independently_reconstructed_metrics": {
            "ADE": reconstructed_ade,
            "FDE": reconstructed_fde
        },
        "absolute_difference": {
            "ADE": ade_difference,
            "FDE": fde_difference
        },
        "match_within_1e-5": {
            "ADE": ade_difference < 1e-5,
            "FDE": fde_difference < 1e-5
        },
        "status": (
            "MATCH"
            if ade_difference < 1e-5 and fde_difference < 1e-5
            else "INVESTIGATION_REQUIRED"
        )
    }

    print("\n========== ADE/FDE RECONSTRUCTION ==========")
    print(json.dumps(report, indent=4))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("\nReport saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
