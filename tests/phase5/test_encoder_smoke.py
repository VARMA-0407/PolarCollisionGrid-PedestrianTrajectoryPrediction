"""P5.5 smoke runner and validator on the synthetic scene (real Task 2 -> Task 3 -> P5.3 pipeline): determinism, tamper detection, policies."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from pedsafety.image_kinematics_run import sha256_file
from pedsafety.output_dir import OutputDirectoryError
from pedsafety.phase5.errors import WindowContractError
from pedsafety.phase5.models.smoke import run_smoke
from pedsafety.phase5.runner import generate_windows
from phase5_helpers import CONFIG as WINDOW_CONFIG
from phase5_helpers import wcfg
from enc_helpers import CONFIG as ENC_CONFIG

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import phase5_run_encoder_smoke  # noqa: E402
import phase5_validate_encoder_smoke  # noqa: E402


@pytest.fixture(scope="module")
def windows_run(scene_run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("p55smoke") / "windows"
    generate_windows(scene_run_dir, wcfg(), out, config_path=str(WINDOW_CONFIG))
    return out


@pytest.fixture(scope="module")
def run_a(windows_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("p55run") / "a"
    return out, run_smoke(windows_run, ENC_CONFIG, out, batch_size=8, optional_bicycle_windows=8, command="test")


def test_manifest_results_and_labels(run_a):
    out, m = run_a
    assert m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and m["counts"]["valid"] is True and m["counts"]["windows"] == 20
    res = json.loads((out / "smoke_results.json").read_text(encoding="utf-8"))
    assert res["model"]["trained"] is False and res["device"] == "cpu"
    assert res["model"]["feature_scaling"]["provenance"]["fitted"] is False
    assert res["numerical_validity"]["overall"] and res["default_context_forward"]["nonfinite_values"] == 0
    assert all(v is False for v in m["claims"].values() if isinstance(v, bool))
    assert m["inputs"]["targets_or_future_arrays_read"] is False and "git" in m["run_metadata"] and "peak_memory_bytes" in m["run_metadata"]
    assert "started_utc" not in json.dumps(res)                                          # run metadata only in run_manifest.json


def test_deterministic_across_runs(run_a, windows_run, tmp_path):
    _, m1 = run_a
    m2 = run_smoke(windows_run, ENC_CONFIG, tmp_path / "b", batch_size=8, optional_bicycle_windows=8)
    assert m1["data_artifacts"] == m2["data_artifacts"]


def test_independent_validation_and_tamper_detection(run_a, tmp_path):
    out, _ = run_a
    rep = phase5_validate_encoder_smoke.validate_run(out)
    assert rep["ok"], [k for k, v in rep["checks"].items() if not v["pass"]]
    bad = tmp_path / "tampered"
    shutil.copytree(out, bad)
    p = bad / "smoke_results.json"
    p.write_text(p.read_text(encoding="utf-8").replace("\"sequence_embeddings_sha256\": \"", "\"sequence_embeddings_sha256\": \"0"), encoding="utf-8")
    assert not phase5_validate_encoder_smoke.validate_run(bad)["checks"]["data_artifact_hashes_match_manifest"]["pass"]
    # a consistent forgery of the stored output hash is still caught by recomputation
    res = json.loads((out / "smoke_results.json").read_text(encoding="utf-8"))
    res["default_context_forward"]["hashes"]["sequence_embeddings_sha256"] = "0" * 64
    forged = tmp_path / "forged"
    shutil.copytree(out, forged)
    (forged / "smoke_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    man = json.loads((forged / "run_manifest.json").read_text(encoding="utf-8"))
    man["data_artifacts"]["smoke_results.json"] = {"sha256": sha256_file(forged / "smoke_results.json"), "bytes": (forged / "smoke_results.json").stat().st_size}
    (forged / "run_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    rep2 = phase5_validate_encoder_smoke.validate_run(forged)
    assert not rep2["ok"] and not rep2["checks"]["recomputed_sequence_embeddings_sha256"]["pass"]


def test_p53_input_verification_and_output_policy(windows_run, run_a, tmp_path):
    out, _ = run_a
    with pytest.raises(OutputDirectoryError):
        run_smoke(windows_run, ENC_CONFIG, out)
    bad = tmp_path / "w"
    shutil.copytree(windows_run, bad)
    p = bad / "windows.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(WindowContractError, match="WINDOW_RUN_HASH"):
        run_smoke(bad, ENC_CONFIG, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_cli_exit_codes(windows_run, tmp_path, capsys):
    args = ["--windows", str(windows_run), "--config", str(ENC_CONFIG), "--out", str(tmp_path / "cli"), "--batch-size", "8", "--optional-bicycle-windows", "4"]
    assert phase5_run_encoder_smoke.main(args) == 0 and "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" in capsys.readouterr().out
    assert phase5_run_encoder_smoke.main(args) == 2
    assert phase5_validate_encoder_smoke.main(["--run", str(tmp_path / "cli")]) == 0
