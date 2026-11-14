
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
OUTPUT = (
    ROOT / "experiments" / "PCG_T4_5_Epoch_Run" /
    "logs" / "task7_csv_inspection.json"
)

results = []

for path in DATA.rglob("*.csv"):
    entry = {
        "file": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "header": None,
        "sample_rows": [],
        "row_count": None,
        "error": None
    }

    try:
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)
            entry["header"] = reader.fieldnames

            for i, row in enumerate(reader):
                if i < 5:
                    entry["sample_rows"].append(row)

            # Count rows independently.
            f.seek(0)
            row_count = sum(1 for _ in csv.DictReader(f))
            entry["row_count"] = row_count

    except Exception as exc:
        entry["error"] = str(exc)

    results.append(entry)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"CSV files inspected: {len(results)}")
print(f"Report saved: {OUTPUT}")

for item in results:
    print("\nFILE:", item["file"])
    print("HEADER:", item["header"])
    print("ROWS:", item["row_count"])
    print("SAMPLE:", item["sample_rows"][:2])
    if item["error"]:
        print("ERROR:", item["error"])
