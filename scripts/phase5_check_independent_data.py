"""Phase 5 P5.7.6 CLI: report whether scene-diverse, annotation-verified data exist for an independent evaluation (read-only scan).

Example:  python scripts/phase5_check_independent_data.py --out docs/phase5/phase5_task7_6_status.json
Exit codes: 0 the report was produced (status BLOCKED or READY is inside the report), 2 output exists / failure.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.phase5.independent_data_status import assess_independent_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    rep = assess_independent_data(REPO)
    if a.out:
        p = Path(a.out)
        if p.exists():
            print("error: %s exists; refusing to overwrite" % p, file=sys.stderr)
            return 2
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("%s: %d independent group(s), %d annotation file(s)" % (rep["status"], rep["checks"]["independent_scene_groups_processed"]["found"], rep["checks"]["annotation_xml_available_locally"]["found"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
