"""P5.7.6: the availability check reports BLOCKED_PENDING_SCENE_DIVERSE_DATA unless real, verified prerequisites exist; it never fabricates."""

import json
import sys
from pathlib import Path

from pedsafety.phase5.independent_data_status import MIN_INDEPENDENT_GROUPS, REQUIREMENTS, STATUS_BLOCKED, assess_independent_data

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import phase5_check_independent_data  # noqa: E402


def _fake_repo(tmp, n_groups, with_xml):
    for i in range(n_groups):
        d = tmp / "experiments" / ("phase4_task3_pie_video_%04d" % i) / "run" / "primary"
        d.mkdir(parents=True)
        (d / "tracks.json").write_text(json.dumps({"tracks": [{"track_id": "pie:video_%04d:1" % i}]}), encoding="utf-8")
        if with_xml:
            (tmp / "data").mkdir(exist_ok=True)
            (tmp / "data" / ("video_%04d_annt.xml" % i)).write_text("<annotations/>", encoding="utf-8")


def test_real_repository_is_blocked_with_one_group_and_no_annotations():
    rep = assess_independent_data(REPO)
    assert rep["status"] == STATUS_BLOCKED == "BLOCKED_PENDING_SCENE_DIVERSE_DATA"
    assert rep["independent_evaluation_performed"] is False and rep["fabricated_results"] is False
    assert rep["checks"]["independent_scene_groups_processed"]["found"] == 1 and rep["checks"]["independent_scene_groups_processed"]["sequence_ids"] == ["pie:video_0004"]
    assert rep["checks"]["annotation_xml_available_locally"]["found"] == 0 and not rep["checks"]["verified_identity_mapping_and_alignment"]["met"]
    assert len(rep["requirements"]) == len(REQUIREMENTS) >= 6 and rep["next_actions"] and "No independent" in rep["consequence"]


def test_not_ready_without_enough_groups_or_annotations(tmp_path):
    _fake_repo(tmp_path, 1, False)
    r = assess_independent_data(tmp_path)
    assert r["status"] == STATUS_BLOCKED and r["checks"]["independent_scene_groups_processed"]["found"] == 1
    _fake_repo(tmp_path / "b", MIN_INDEPENDENT_GROUPS, False)
    assert assess_independent_data(tmp_path / "b")["status"] == STATUS_BLOCKED               # groups alone are not enough


def test_files_alone_never_flip_the_status_to_ready(tmp_path):
    """Enough groups and annotation files still leave the identity mapping, alignment and the group-level split unverified: stay blocked."""
    _fake_repo(tmp_path, MIN_INDEPENDENT_GROUPS, True)
    r = assess_independent_data(tmp_path)
    assert r["checks"]["independent_scene_groups_processed"]["met"] and r["checks"]["annotation_xml_available_locally"]["met"]
    assert r["status"] == STATUS_BLOCKED and r["independent_evaluation_performed"] is False


def test_cli_writes_a_new_report_and_refuses_to_overwrite(tmp_path, capsys):
    out = tmp_path / "status.json"
    assert phase5_check_independent_data.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == STATUS_BLOCKED and "BLOCKED_PENDING_SCENE_DIVERSE_DATA" in capsys.readouterr().out
    assert phase5_check_independent_data.main(["--out", str(out)]) == 2
