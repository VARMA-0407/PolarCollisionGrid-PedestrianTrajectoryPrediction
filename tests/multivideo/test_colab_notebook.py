"""The master Colab notebook: structural validation, and a LOCAL SMOKE EXECUTION of every cell in precomputed-artifact mode on synthetic data.

The smoke run executes the real notebook cells in order (configuration overridden after the configuration cell: local folders instead of Drive,
no GPU, PERCEPTION_MODE="precomputed" with synthetic Phase 3 folders, 2 training epochs on the CPU, no nested pytest run), then runs the notebook
a second time with EXPERIMENT_ID set to check that a resumed run verifies and skips everything. It tests orchestration and resume logic only;
YOLO / ByteTrack (raw-video mode), Google Drive and the real PIE data are NOT exercised here.
"""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

from mv_helpers import make_pie_tree, write_phase3_dir

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "PIE_Full_Pedestrian_Safety_Pipeline_Colab.ipynb"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_colab_notebook", REPO / "scripts" / "validate_colab_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_is_regenerated_from_the_builder(tmp_path):
    spec = importlib.util.spec_from_file_location("build_nb", REPO / "scripts" / "build_colab_full_pipeline_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.build(tmp_path / "nb.ipynb")
    assert out.read_text(encoding="utf-8") == NB.read_text(encoding="utf-8"), "notebook is stale: run scripts/build_colab_full_pipeline_notebook.py"


def test_notebook_structure_imports_signatures_paths_and_order():
    r = _validator().validate(NB)
    assert r["ok"], r["problems"]
    assert r["info"]["pedsafety_calls_signature_checked"] >= 40


def _cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _run(overrides, log):
    ns = {"__name__": "__notebook__"}
    cwd = os.getcwd()
    try:
        for i, src in enumerate(_cells()):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(src, "<notebook cell %d>" % i, "exec"), ns)
            log.append(buf.getvalue())
            if "CENTRAL CONFIGURATION" in src:
                ns.update(overrides)
    finally:
        os.chdir(cwd)
    return ns


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    os.environ.setdefault("MPLBACKEND", "Agg")
    tmp = tmp_path_factory.mktemp("nbsmoke")
    drive = tmp / "drive"
    pie = make_pie_tree(drive / "data" / "raw", {s: 1 for s in ("set01", "set02", "set03", "set04", "set05", "set06")})
    pre = tmp / "precomputed_phase3"
    for i, s in enumerate(("set01", "set02", "set03", "set04", "set05", "set06")):
        k = "%s_video_0001" % s
        write_phase3_dir(pre / s / k, k, seed=i)
    over = {"DRIVE_ROOT": str(drive), "PIE_RAW_DIR": str(pie), "REPO_SOURCE": "existing", "EXISTING_REPO_DIR": str(REPO), "REQUIRE_GPU": False,
            "INSTALL_MISSING_DEPENDENCIES": False, "WORK_ROOT": str(tmp / "work"), "PERCEPTION_MODE": "precomputed", "PRECOMPUTED_PHASE3_ROOT": str(pre),
            "RUN_TESTS": False, "TRAIN_EPOCHS": 2, "TRAIN_BATCH_SIZE": 16, "TRAIN_DEVICE": "cpu", "EVAL_DEVICE": "cpu", "MONITOR_MAX_WINDOWS": 64}
    log1 = []
    ns1 = _run(over, log1)
    log2 = []
    ns2 = _run(dict(over, EXPERIMENT_ID=ns1["EXPERIMENT_ID"]), log2)
    return {"ns1": ns1, "ns2": ns2, "log1": "".join(log1), "log2": "".join(log2), "drive": drive}


def test_smoke_run_completes_every_stage_in_precomputed_mode(smoke):
    ns = smoke["ns1"]
    st = {k: v["status"] for k, v in ns["TRACKER"].data["stages"].items()}
    for s in ("S0_OFFICIAL_SPLIT", "S0_SPLIT", "S3_PERCEPTION", "S4_GEOMETRY", "S5A_WINDOWS", "S5B_TRAINING", "S5C_EVALUATION", "S6_CONFLICT", "S7_RISK", "S8_ALERTS",
              "S9_EVALUATION_GATES"):
        assert st[s] == "COMPLETED", (s, st[s], ns["TRACKER"].data["stages"][s])
    assert st["S7_CALIBRATION"] == "BLOCKED" and st["S8_LEAD_TIME"] == "BLOCKED"
    v = ns["VALIDATION"]
    res = {c["name"]: c["result"] for c in v["checks"]}
    assert res["repository tests"] == "SKIPPED" and v["overall_status"] == "INCOMPLETE"          # tests were not run -> never reported as COMPLETED
    assert all(r == "PASS" for n, r in res.items() if n != "repository tests"), res
    ex = ns["EXECUTION"]
    assert ex["precomputed_artifact_chains_complete"] == 6 and ex["raw_video_to_alert_chains_complete"] == 0       # precomputed is never labelled raw-video
    rep = json.loads((ns["EXP_DIR"] / "experiment_report.json").read_text())
    assert rep["claims"]["real_time_performance"] is False and rep["claims"]["calibrated_probability"] is False
    assert set(rep["evaluation"]["per_partition"]) == {"train", "validation", "test"}
    assert rep["split"]["partitions"]["train"] == ["set01", "set02", "set04"] and rep["split"]["partitions"]["test"] == ["set03"]


def test_resumed_run_verifies_and_skips_completed_work(smoke):
    ns1, ns2 = smoke["ns1"], smoke["ns2"]
    assert ns2["EXPERIMENT_ID"] == ns1["EXPERIMENT_ID"] and ns2["EXP"]["resumed"]
    assert "RESUMED" in smoke["log2"] and "training already completed" in smoke["log2"]
    for name in ("phase4_index.json", "windows_index.json", "s8_alerts_index.json"):
        idx = json.loads((ns2["EXP_DIR"] / name).read_text())
        statuses = {s for r in idx["results"].values() for s in ([r["status"]] if "status" in r else [v["status"] for v in r["stages"].values()])}
        assert statuses == {"SKIPPED_ALREADY_COMPLETE"}, (name, statuses)
    a = json.loads((ns1["EXP_DIR"] / "split_manifest.json").read_text())["split_id"]
    assert a == ns2["SPLIT"]["split_id"]
    assert len(list((smoke["drive"] / "experiments").glob("pie_full_*"))) == 1                    # no second experiment folder was created
