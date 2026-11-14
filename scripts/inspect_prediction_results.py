
import os
import pickle
import numpy as np

RESULT_PATH = (
    "experiments/PCG_T4_5_Epoch_Run/results/"
    "test_results_epoch4.pkl"
)

if not os.path.exists(RESULT_PATH):
    raise FileNotFoundError(RESULT_PATH)

with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)

print("=" * 60)
print("PREDICTION ARTIFACT INSPECTION")
print("=" * 60)

print("Result type:", type(results))
print("Number of result entries:", len(results))

if len(results) == 0:
    raise ValueError("The prediction artifact is empty.")

first = results[0]

print("\nFirst entry type:", type(first))

if isinstance(first, (tuple, list)):
    print("Number of fields:", len(first))

    for i, item in enumerate(first):
        print(f"\nField {i}")

        if isinstance(item, np.ndarray):
            print("  Type:", type(item))
            print("  Shape:", item.shape)
            print("  Dtype:", item.dtype)
            print("  Min:", np.nanmin(item))
            print("  Max:", np.nanmax(item))
        else:
            print("  Type:", type(item))
            print("  Value:", repr(item)[:300])

elif isinstance(first, dict):
    print("Keys:", list(first.keys()))

else:
    print("First entry:", repr(first)[:1000])
