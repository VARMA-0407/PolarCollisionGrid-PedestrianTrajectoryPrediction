"""Stage 6.1 regression tests: output-directory policy (no stale or mixed results)."""

import hashlib
import json
from pathlib import Path

import pytest

from pedsafety import pipeline
from pedsafety.cli import main
from pedsafety.output_dir import COMPLETION_FILES, OutputDirectoryError, check_output_dir, is_complete_run, output_dir_state
from stage6_fixtures import load_cfg, save_cfg, synthetic

COMMANDS = ("ingest", "features", "build", "plot")


def snap(d: Path):
    """name -> sha256 of every file under d (recursive)."""
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@pytest.fixture()
def syn(tmp_path):
    csv_path, cfg = synthetic(tmp_path / "data")
    return csv_path, cfg


# ------------------------------------------------------------ 1. fresh directory
def test_build_into_a_fresh_directory_succeeds_and_is_complete(syn, tmp_path):
    out = tmp_path / "run1"
    assert main(["build", str(syn[1]), "--out", str(out)]) == 0
    assert is_complete_run(out, "build") and not (out / "error.json").exists()
    assert not list(out.glob("*.tmp"))                       # atomic writes leave no temp files


# ------------------------------------------- 2/3. same directory again is refused
def test_second_build_into_the_same_directory_is_refused_and_changes_nothing(syn, tmp_path, capsys):
    out = tmp_path / "run1"
    assert main(["build", str(syn[1]), "--out", str(out)]) == 0
    before = snap(out)
    capsys.readouterr()
    assert main(["build", str(syn[1]), "--out", str(out)]) == pipeline.EXIT_OUTPUT == 7
    err = capsys.readouterr().err
    assert "OUTPUT_DIR_NOT_EMPTY" in err and "never overwrites" in err and "not written" in err
    assert snap(out) == before                                # byte-for-byte identical, no error.json added
    assert is_complete_run(out, "build")


@pytest.mark.parametrize("command", COMMANDS)
def test_every_writing_command_refuses_a_non_empty_directory(syn, tmp_path, command):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "precious.txt").write_text("keep me", encoding="utf-8")
    before = snap(out)
    assert main([command, str(syn[1]), "--out", str(out)]) == 7
    assert snap(out) == before


def test_hbs_smoke_and_synthetic_refuse_a_non_empty_directory(syn, tmp_path):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / ".hidden").write_text("x", encoding="utf-8")            # even a hidden file counts
    before = snap(out)
    assert main(["hbs-smoke", str(syn[1]), "--out", str(out)]) == 7
    assert main(["synthetic", "--out", str(out)]) == 7
    assert snap(out) == before


def test_synthetic_never_overwrites_an_earlier_corpus(tmp_path):
    out = tmp_path / "corpus"
    assert main(["synthetic", "--out", str(out)]) == 0
    before = snap(out)
    assert main(["synthetic", "--out", str(out), "--n-scenes", "3"]) == 7
    assert snap(out) == before


# --------------------------------------------------- 6. empty directory behaviour
def test_an_existing_empty_directory_is_accepted(syn, tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    assert output_dir_state(out) == "empty"
    assert main(["build", str(syn[1]), "--out", str(out)]) == 0
    assert is_complete_run(out, "build")


def test_a_path_that_is_a_file_is_refused(syn, tmp_path, capsys):
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    assert main(["build", str(syn[1]), "--out", str(f)]) == 7
    assert "OUTPUT_PATH_NOT_DIRECTORY" in capsys.readouterr().err
    assert f.read_text(encoding="utf-8") == "x"


def test_state_and_check_helpers(tmp_path):
    assert output_dir_state(tmp_path / "nope") == "absent"
    assert check_output_dir(tmp_path / "nope") == tmp_path / "nope" and not (tmp_path / "nope").exists()   # checking creates nothing
    (tmp_path / "d").mkdir()
    assert output_dir_state(tmp_path / "d") == "empty"
    (tmp_path / "d" / "f").write_text("x")
    with pytest.raises(OutputDirectoryError) as ei:
        check_output_dir(tmp_path / "d")
    assert ei.value.code == "OUTPUT_DIR_NOT_EMPTY"


# ---------------------------------- 4/5. failed runs never look like valid outputs
def test_the_audited_scenario_success_then_failure_in_the_same_directory(syn, tmp_path):
    """Audit repro: success, then a build that WOULD fail (single scene) into the same directory."""
    out = tmp_path / "run1"
    assert main(["build", str(syn[1]), "--out", str(out)]) == 0
    before = snap(out)
    _, one_scene = synthetic(tmp_path / "data1", n_scenes=1)
    assert main(["build", str(one_scene), "--out", str(out)]) == 7
    assert snap(out) == before and not (out / "error.json").exists()


def test_config_error_into_a_non_empty_directory_leaves_it_untouched(syn, tmp_path):
    out = tmp_path / "run1"
    assert main(["build", str(syn[1]), "--out", str(out)]) == 0
    before = snap(out)
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert main(["build", str(tmp_path / "bad.json"), "--out", str(out)]) == 2          # exit code still reports the config error
    assert snap(out) == before


def test_failure_in_a_fresh_directory_leaves_only_error_json(tmp_path):
    _, one_scene = synthetic(tmp_path / "d", n_scenes=1)
    out = tmp_path / "o"
    assert main(["build", str(one_scene), "--out", str(out)]) == 4
    assert sorted(p.name for p in out.iterdir()) == ["error.json"]
    e = jl(out / "error.json")
    assert e["status"] == "failed" and e["stage"] == "split" and e["code"] == "INSUFFICIENT_GROUPS" and e["exit_code"] == 4
    assert not is_complete_run(out, "build")


def test_config_error_in_a_fresh_directory_writes_error_json_only(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    out = tmp_path / "o"
    assert main(["build", str(tmp_path / "bad.json"), "--out", str(out)]) == 2
    assert [p.name for p in out.iterdir()] == ["error.json"] and jl(out / "error.json")["code"] == "CONFIG_ERROR"


def corrupt(csv_path, tmp_path):
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split(",")
    row = lines[5].split(",")
    row[cols.index("foot_x")] = "oops"
    lines[5] = ",".join(row)
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bad


@pytest.mark.parametrize("command", ["ingest", "features", "build"])
def test_fail_on_issues_writes_no_results_at_all(syn, tmp_path, command):
    """Before 6.1 the full result set AND a manifest were written, then the run was reported as failed."""
    bad = corrupt(syn[0], tmp_path)
    out = tmp_path / "o"
    assert main([command, str(syn[1]), "--input", str(bad), "--fail-on-issues", "--out", str(out)]) == 3
    assert [p.name for p in out.iterdir()] == ["error.json"]
    assert jl(out / "error.json")["code"] == "ISSUES_PRESENT" and not is_complete_run(out, command)


def test_a_crash_while_writing_leaves_no_completion_file(syn, tmp_path, monkeypatch):
    real = pipeline._export_windows
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full (simulated)")
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "_export_windows", boom)
    out = tmp_path / "o"
    rc = main(["build", str(syn[1]), "--out", str(out)])
    assert rc == pipeline.EXIT_INTERNAL
    names = {p.name for p in out.iterdir()}
    assert "windows_train.npz" in names                      # partial results exist ...
    assert "run_manifest.json" not in names and "error.json" in names       # ... but cannot pass for a finished run
    assert not is_complete_run(out, "build")
    assert jl(out / "error.json")["code"] == "INTERNAL_ERROR"


def test_a_failed_attempt_directory_cannot_be_reused(syn, tmp_path):
    out = tmp_path / "o"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert main(["build", str(tmp_path / "bad.json"), "--out", str(out)]) == 2
    assert main(["build", str(syn[1]), "--out", str(out)]) == 7          # documented: use a fresh directory per attempt
    assert [p.name for p in out.iterdir()] == ["error.json"]


# --------------------------------------------------- error plumbing and helpers
def test_write_error_respects_ownership(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "x").write_text("1")
    assert pipeline.write_error(d, {"code": "X"}) is None and not (d / "error.json").exists()
    assert pipeline.write_error(d, {"code": "X"}, owned=True) == d / "error.json"
    assert pipeline.write_error(tmp_path / "new", {"code": "X"}) == tmp_path / "new" / "error.json"


def test_output_errors_map_to_exit_7_and_stage_output():
    info = pipeline.classify_error(OutputDirectoryError("OUTPUT_DIR_NOT_EMPTY", "m", "/p"))
    assert info["exit_code"] == 7 and info["stage"] == "output" and info["details"] == ["/p"]
    assert pipeline.EXIT_OUTPUT == 7


def test_is_complete_run_requires_completion_file_and_no_error(tmp_path):
    for cmd, fname in COMPLETION_FILES.items():
        d = tmp_path / cmd
        d.mkdir()
        assert not is_complete_run(d, cmd)
        (d / fname).write_text("{}")
        assert is_complete_run(d, cmd)
        (d / "error.json").write_text("{}")
        assert not is_complete_run(d, cmd)
    with pytest.raises(ValueError):
        is_complete_run(tmp_path, "nonsense")


def test_every_command_completes_with_its_documented_completion_file(syn, tmp_path):
    for cmd in COMMANDS:
        assert main([cmd, str(syn[1]), "--out", str(tmp_path / cmd)]) == 0
        assert is_complete_run(tmp_path / cmd, cmd), cmd


def test_json_outputs_stay_deterministic_after_the_atomic_write_change(syn, tmp_path):
    assert main(["build", str(syn[1]), "--out", str(tmp_path / "a")]) == 0
    assert main(["build", str(syn[1]), "--out", str(tmp_path / "b")]) == 0
    for p in sorted((tmp_path / "a").glob("*.json")):
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes(), p.name
