"""Tests for the PCG measurement-notebook helpers, the monitored runner (against a FAKE train module on CPU), the benchmark pre-flight and the
built notebook. No real training is run and nothing in the repository's experiment directories is touched."""

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MEAS = REPO / "docs" / "phase3" / "measurements"
HELPERS = MEAS / "helpers"
NOTEBOOK = MEAS / "PCG_Baseline_Reproduction_Measurements.ipynb"
# The original (pre-measurement) notebook is not part of the repository. Point PCG_ORIGINAL_NOTEBOOK at a local copy to run
# the byte-level comparison; without it that comparison is skipped (and reported as skipped), never silently passed.
_ORIGINAL_ENV = os.environ.get("PCG_ORIGINAL_NOTEBOOK", "").strip()
ORIGINAL = Path(_ORIGINAL_ENV).expanduser() if _ORIGINAL_ENV else None
ORIGINAL_SHA = "b20af1a163b71a495dd3c37bd49fc150fb0b243de716df58173010c79dfeea10"
sys.path.insert(0, str(HELPERS))
import measurement_lib as ml  # noqa: E402
import monitored_train_runner as runner  # noqa: E402
import inference_benchmark as bench  # noqa: E402


# ------------------------------------------------------------------ measurement_lib
def test_sha256_and_file_record_match_mismatch_missing(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    h = hashlib.sha256(b"abc").hexdigest()
    assert ml.sha256_file(str(f)) == h
    ok = ml.file_record(str(f), "x", expected_sha256=h.upper(), expected_source="src")
    assert ok["exists"] and ok["matches_previous_hash"] is True and ok["size_bytes"] == 3 and ok["previous_hash_source"] == "src"
    assert ml.file_record(str(f), "x", expected_sha256="0" * 64)["matches_previous_hash"] is False
    assert ml.file_record(str(f), "x")["matches_previous_hash"] is None
    miss = ml.file_record(str(tmp_path / "nope"), "m", expected_sha256=h)
    assert miss["exists"] is False and miss["sha256"] is None and miss["matches_previous_hash"] is None


def test_fresh_dir_and_exclusive_writes_never_overwrite(tmp_path):
    d = tmp_path / "out"
    ml.make_fresh_dir(str(d))
    ml.make_fresh_dir(str(d))                                   # an existing EMPTY directory is accepted
    ml.write_new_json(str(d / "a.json"), {"k": 1})
    with pytest.raises(FileExistsError):
        ml.write_new_json(str(d / "a.json"), {"k": 2})
    assert json.loads((d / "a.json").read_text())["k"] == 1
    with pytest.raises(FileExistsError):
        ml.make_fresh_dir(str(d))                               # now non-empty
    with pytest.raises(FileExistsError):
        ml.write_new_csv(str(d / "a.json"), [], ["x"])
    (tmp_path / "f").write_text("x")
    with pytest.raises(FileExistsError):
        ml.make_fresh_dir(str(tmp_path / "f"))


def test_describe_and_percentiles_are_nearest_rank():
    d = ml.describe([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert (d["count"], d["min"], d["max"], d["p50"], d["p90"], d["p95"], d["p99"]) == (10, 1, 10, 5, 9, 10, 10) and math.isclose(d["mean"], 5.5)
    assert ml.describe([]) == {"count": 0} and ml.describe([float("nan"), None]) == {"count": 0}
    with pytest.raises(ValueError):
        ml.percentile_nearest_rank([], 50)


TRAIN_LOG = textwrap.dedent("""\
    grid calculation time for all the data: 13.5 seconds
    **************** Training epoch beginning ******************
    0/4 (epoch 0), train_loss = 1.500, time/batch = 0.700
    1/4 (epoch 0), train_loss = -0.500, time/batch = 0.500
    2/4 (epoch 1), train_loss = -1.000, time/batch = 0.600
    3/4 (epoch 1), train_loss = -2.000, time/batch = 0.400
    The whole trainig time for 2 iteraction was 12.5 seconds
    """)


def test_train_and_test_log_parsers_extract_only_printed_values():
    p = ml.parse_train_log(TRAIN_LOG)
    assert [b["epoch"] for b in p["batches"]] == [0, 0, 1, 1] and p["grid_calculation_s"] == 13.5 and p["total_training_s"] == 12.5 and p["epochs_reported"] == 2
    ep = ml.per_epoch_from_batches(p["batches"])
    assert math.isclose(ep[0]["mean_loss"], 0.5) and math.isclose(ep[1]["sum_batch_time_s"], 1.0)
    assert ml.parse_train_log("garbage") == {"batches": [], "grid_calculation_s": None, "total_training_s": None, "epochs_reported": None}
    t = ml.parse_test_log("Current file :  hbs.csv  Processed trajectory number :  7 out of 1081 trajectories in time 0.25\n")
    assert t["trajectories"] == [{"trajectory": 7, "of": 1081, "time_s": 0.25}]


# --------------------------------------------------------------- fake train module (CPU)
FAKE_TRAIN = '''
import argparse, os, pickle, time
import torch
from utils.DataLoader import DataLoader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=10)
    a = ap.parse_args()
    os.makedirs("Store_Results/log", exist_ok=True)
    with open("Store_Results/model/config.pkl", "wb") as f:
        pickle.dump(a, f)
    dl = DataLoader(a.batch_size, 12, infer=False, filtering=True)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.RMSprop(net.parameters(), lr=1e-3)
    dl.grid_creation(a)
    start = time.time()
    for epoch in range(a.num_epochs):
        print("**************** Training epoch beginning ******************")
        dl.batch_creater(True, a.method, suffle=True)
        for batch in range(dl.num_batches):
            t0 = time.time()
            net.zero_grad(); opt.zero_grad()
            loss = net(torch.randn(5, 3)).pow(2).mean()
            loss.backward()
            opt.step()
            if os.environ.get("FAKE_TRAIN_FAIL") and epoch == 1 and batch == 1:
                raise RuntimeError("scripted failure")
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(epoch * dl.num_batches + batch, a.num_epochs * dl.num_batches, epoch, loss.item(), time.time() - t0))
        print("Saving model")
        torch.save({"epoch": epoch}, "Store_Results/model/SOCIALLSTM_lstm_model_%d.tar" % epoch)
    print("The whole trainig time for {} iteraction was {} seconds".format(a.num_epochs, time.time() - start))
'''
FAKE_LOADER = '''
import time
class DataLoader:
    def __init__(self, batch_size, seq_length, infer=False, filtering=True):
        self.batch_size, self.num_batches = batch_size, 4
        time.sleep(0.01)
    def grid_creation(self, args):
        time.sleep(0.01)
    def batch_creater(self, store_grid, method, suffle=True):
        time.sleep(0.01)
        return [None] * 10
'''
FAKE_REQUIRED = ("fake_train.py", "utils/DataLoader.py", "Data/train/trajectories_train.cpkl")


@pytest.fixture()
def fake_repo(tmp_path):
    r = tmp_path / "repo"
    (r / "utils").mkdir(parents=True)
    (r / "Data" / "train").mkdir(parents=True)
    (r / "Store_Results" / "model").mkdir(parents=True)
    (r / "fake_train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    (r / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (r / "utils" / "DataLoader.py").write_text(FAKE_LOADER, encoding="utf-8")
    (r / "Data" / "train" / "trajectories_train.cpkl").write_bytes(b"data")
    (r / "Store_Results" / "model" / "SOCIALLSTM_lstm_model_0.tar").write_bytes(b"ORIGINAL CHECKPOINT")
    return r


def run_fake(repo, workdir, env_extra=None, epochs=2):
    code = ("import sys; sys.path.insert(0, %r); import monitored_train_runner as r; "
            "r.run_monitored(%r, %r, 'test_run', %d, interval_s=0.05, train_module='fake_train', required_files=%r)" % (str(HELPERS), str(repo), str(workdir), epochs, FAKE_REQUIRED))
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))


def snap(root: Path):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_monitored_runner_records_batches_epochs_resources_and_isolates(fake_repo, tmp_path):
    before = snap(fake_repo)
    work = tmp_path / "work"
    r = run_fake(fake_repo, work)
    assert r.returncode == 0, r.stdout + r.stderr
    m = work / "measurements"
    s = json.loads((m / "training_resource_summary.json").read_text())
    assert s["status"] == "completed" and s["batches_recorded"] == 8 and s["epochs_recorded"] == 2
    assert s["batch_time_sync_s"]["count"] == 8 and s["batch_time_sync_s"]["p50"] > 0 and s["batch_time_sync_excluding_first_s"]["count"] == 7
    assert {"p90", "p95", "max", "mean"} <= set(s["batch_time_sync_s"])
    assert s["batch_time_reported_by_train_py_s"]["count"] == 8
    assert s["throughput"]["sequences_per_s_compute"] > 0 and s["training_args"]["batch_size"] == 10 and s["training_args"]["num_epochs"] == 2
    assert "DataLoader.__init__" in s["data_preprocessing_s"] and "DataLoader.grid_creation" in s["data_preprocessing_s"] and "batch_creater" in s["data_preprocessing_s"]
    assert s["checkpoint_save_s"]["count"] == 2 and s["checkpoint_save_s"]["total"] >= 0                # one torch.save per epoch
    assert s["sampler"]["n_samples"] >= 2 and s["environment"]["python_executable"] and s["environment"]["cuda_available"] in (True, False)
    assert s["environment"]["seeds"].startswith("none set") and s["train_main_wall_s"] > 0
    assert {"torch_peak_allocated_mib", "peak_process_rss_mib_sampled", "gpu_utilisation_pct", "cpu_pct_process"} <= set(s)
    ep = list(__import__("csv").DictReader((m / "training_epoch_summary.csv").open()))
    assert len(ep) == 2 and all(float(e["epoch_wall_s"]) > 0 and int(e["n_batches"]) == 4 for e in ep)
    assert float(ep[0]["epoch_wall_s"]) >= float(ep[0]["sum_batch_time_s"])           # wall includes everything the batches do
    bl = list(__import__("csv").DictReader((m / "training_batch_log.csv").open()))
    assert len(bl) == 8 and [int(b["epoch"]) for b in bl] == [0, 0, 0, 0, 1, 1, 1, 1] and all(b["train_loss"] != "" for b in bl)
    rl = list(__import__("csv").DictReader((m / "training_resource_log.csv").open()))
    assert len(rl) >= 2 and {"gpu_util_pct", "process_rss_mib", "cpu_pct_system", "torch_reserved_mib"} <= set(rl[0])
    assert (m / "training_stdout.log").read_text().count("Training epoch beginning") == 2
    # isolation: fresh workdir has its own Data copy and checkpoints; the repository is byte-identical
    assert (work / "Data" / "train" / "trajectories_train.cpkl").read_bytes() == b"data"
    assert sorted(p.name for p in (work / "Store_Results" / "model").iterdir()) == ["SOCIALLSTM_lstm_model_0.tar", "SOCIALLSTM_lstm_model_1.tar", "config.pkl"]
    assert snap(fake_repo) == before
    assert s["isolation_check"]["repo_store_results_unchanged"] and s["isolation_check"]["repo_source_hashes_unchanged"] is True


def test_runner_refuses_when_isolation_cannot_be_guaranteed(fake_repo, tmp_path):
    before = snap(fake_repo)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "SOCIALLSTM_lstm_model_0.tar").write_bytes(b"precious")
    assert any("not empty" in p for p in runner.preflight(str(fake_repo), str(occupied), FAKE_REQUIRED))
    r = run_fake(fake_repo, occupied)
    assert r.returncode != 0 and "NOT started" in r.stderr and (occupied / "SOCIALLSTM_lstm_model_0.tar").read_bytes() == b"precious" and len(list(occupied.iterdir())) == 1
    inside = fake_repo / "Store_Results" / "model" / "run"
    assert any("OUTSIDE the repository" in p for p in runner.preflight(str(fake_repo), str(inside), FAKE_REQUIRED))
    assert any("OUTSIDE the repository" in p for p in runner.preflight(str(fake_repo), str(fake_repo), FAKE_REQUIRED))
    assert any("repository file missing" in p for p in runner.preflight(str(fake_repo), str(tmp_path / "w"), ("train.py",)))
    f = tmp_path / "afile"
    f.write_text("x")
    assert any("not a directory" in p for p in runner.preflight(str(fake_repo), str(f), FAKE_REQUIRED))
    assert runner.preflight(str(fake_repo), str(tmp_path / "fresh" / "deep"), FAKE_REQUIRED) == []
    empty = tmp_path / "emptydir"
    empty.mkdir()
    assert runner.preflight(str(fake_repo), str(empty), FAKE_REQUIRED) == []
    assert snap(fake_repo) == before


def test_a_second_run_into_the_same_workdir_is_refused_and_changes_nothing(fake_repo, tmp_path):
    work = tmp_path / "work"
    assert run_fake(fake_repo, work).returncode == 0
    before = snap(work)
    r = run_fake(fake_repo, work)
    assert r.returncode != 0 and "NOT started" in r.stderr and snap(work) == before


def test_a_failing_training_still_writes_a_failed_summary_and_reraises(fake_repo, tmp_path):
    work = tmp_path / "work"
    r = run_fake(fake_repo, work, {"FAKE_TRAIN_FAIL": "1"})
    assert r.returncode != 0 and "scripted failure" in r.stderr
    s = json.loads((work / "measurements" / "training_resource_summary.json").read_text())
    assert s["status"] == "failed" and s["error"]["type"] == "RuntimeError" and s["batches_recorded"] == 6
    assert s["isolation_check"]["repo_store_results_unchanged"] is True


def test_summary_statistics_helpers():
    st = runner.stats([1.0, 2.0, 3.0, 4.0])
    assert st["count"] == 4 and st["p50"] == 2.0 and st["p95"] == 4.0 and runner.stats([]) == {"count": 0}
    assert runner.tree_snapshot("/definitely/not/here") == {}


# -------------------------------------------------------------------------- benchmark
def test_benchmark_preflight_blocks_unsafe_or_incomplete_setups(tmp_path):
    ck = tmp_path / "c.tar"
    cf = tmp_path / "c.pkl"
    ck.write_bytes(b"1")
    cf.write_bytes(b"2")
    ok = bench.preflight(str(REPO), str(tmp_path / "w"), str(ck), str(cf))
    assert ok == []
    assert any("OUTSIDE" in p for p in bench.preflight(str(REPO), str(REPO / "x"), str(ck), str(cf)))
    assert any("checkpoint file missing" in p for p in bench.preflight(str(REPO), str(tmp_path / "w"), str(tmp_path / "no"), str(cf)))
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    assert any("not an empty" in p for p in bench.preflight(str(REPO), str(tmp_path / "busy"), str(ck), str(cf)))


def test_benchmark_describe_reports_all_requested_percentiles():
    d = bench.describe([0.1, 0.2, 0.3, 0.4])
    assert {"p50_s", "p90_s", "p95_s", "p99_s", "mean_s", "std_s"} <= set(d) and d["p99_s"] == 0.4


@pytest.mark.skipif(not (REPO / "experiments" / "PCG_T4_5_Epoch_Run" / "checkpoints" / "SOCIALLSTM_lstm_model_4.tar").is_file(), reason="archived checkpoint absent")
def test_benchmark_runs_on_cpu_for_validation_only_and_leaves_sources_untouched(tmp_path):
    exp = REPO / "experiments" / "PCG_T4_5_Epoch_Run"
    watched = [exp / "checkpoints" / "SOCIALLSTM_lstm_model_4.tar", exp / "configs" / "experiment_config.pkl", REPO / "test.py", REPO / "Data" / "test" / "trajectories_test.cpkl"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in watched]
    s = bench.run_benchmark(str(REPO), str(tmp_path / "w"), "validation", str(watched[0]), str(watched[1]), epoch=4, warmup=2, max_trajectories=8, use_cuda=False,
                            expected_checkpoint_sha256="ac9f5462ec741b91541fa6be74f07167060d63a469f88936c9d5fbc833387d48",
                            expected_config_sha256="9c994a14a5f6519b2c78256b3603c2e26975ddb647ac455bf13aeccd18485ba5", hardware_label="CPU validation")
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in watched] == before
    assert s["scope"].startswith("PCG trajectory-predictor-only") and "NOT end-to-end video" in s["scope"]
    p = s["protocol"]
    assert p["sample_size"] == 1 and p["ground_truth_scoring_included"] is False and p["oracle_selection_included"] is False and p["timed_trajectories"] == 6
    assert p["warmup_trajectories_excluded"] == 2 and p["net_eval_called"] is False and p["dropout_active_in_prediction_path"] is True
    assert s["hashes"]["checkpoint_matches_expected"] is True and s["hashes"]["config_matches_expected"] is True
    L = s["latency_s"]
    assert L["pipeline"]["count"] == 6 and L["model_forward_only_sum_per_trajectory"]["count"] == 6 and {"p50_s", "p90_s", "p95_s", "p99_s"} <= set(L["pipeline"])
    assert L["model_forward_calls_per_trajectory"]["min"] == 11
    assert s["output_check"]["non_finite_outputs"] == 0 and s["throughput_sequences_per_s"]["pipeline_pass"] > 0 and s["peak_gpu_memory_mib"]["max_allocated"] is None
    assert (tmp_path / "w" / "Store_Results" / "model" / "CollisionGrid" / "SOCIALLSTM_lstm_model_4.tar").is_file()      # COPY inside the work dir
    with pytest.raises(RuntimeError):
        bench.run_benchmark(str(REPO), str(tmp_path / "w"), "again", str(watched[0]), str(watched[1]), epoch=4, warmup=2, max_trajectories=8, use_cuda=False)


# ------------------------------------------------------------------------ the notebook
needs_nb = pytest.mark.skipif(not NOTEBOOK.is_file(), reason="measurement notebook not built")


@needs_nb
def test_original_cells_and_outputs_are_preserved_exactly():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 64 + 9
    assert all(c.get("outputs") == [] for c in nb["cells"][64:] if c["cell_type"] == "code")      # no fake outputs in appended cells


@needs_nb
@pytest.mark.skipif(ORIGINAL is None or not ORIGINAL.is_file(),
                    reason="original notebook not available; set PCG_ORIGINAL_NOTEBOOK to a local copy of PCG_Baseline_Reproduction.ipynb")
def test_original_notebook_prefix_matches_byte_identical_source():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    raw = ORIGINAL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA                           # the original file itself is unchanged
    orig = json.loads(raw.decode("utf-8"))
    assert nb["cells"][:64] == orig["cells"] and nb["metadata"] == orig["metadata"] and (nb["nbformat"], nb["nbformat_minor"]) == (orig["nbformat"], orig["nbformat_minor"])


def _appended_code(nb):
    return ["".join(c["source"]) for c in nb["cells"][64:] if c["cell_type"] == "code"]


@needs_nb
def test_appended_cells_compile_and_default_flags_are_off():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for src in _appended_code(nb):
        ast.parse(src)                                                               # syntactically valid Python
    setup = _appended_code(nb)[0]
    assert re.search(r"^RUN_MONITORED_TRAINING = False", setup, re.M) and re.search(r"^RUN_INFERENCE_BENCHMARK = False", setup, re.M)
    assert re.search(r'^CONFIRM_RUN_ID = ""', setup, re.M)
    train_cell = _appended_code(nb)[4]
    assert "if not RUN_MONITORED_TRAINING" in train_cell and "CONFIRM_RUN_ID != MEAS_RUN_ID" in train_cell and train_cell.index("_problems") < train_cell.index("subprocess.run(TRAIN_CMD")


@needs_nb
def test_appended_cells_never_write_into_experiment_artifact_locations():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = _appended_code(nb)
    setup = cells[0]
    cells[0] = setup[:setup.index("HELPER_SOURCES = ")] + setup[setup.index("\n}\n", setup.index("HELPER_SOURCES = ")) + 3:]       # the embedded helper files are checked below
    code = "\n".join(cells)
    for bad in ("shutil.copy2(", "shutil.move(", "os.remove(", "os.unlink(", "shutil.rmtree(", "os.rename(", "os.replace("):
        assert bad not in code, bad
    assert "shutil.copytree(_src, TRAINING_OUTPUT_DIR)" in code and "shutil.copytree(os.path.join(BENCH_WORKDIR" in code      # the only copies: into NEW directories
    for name in ("monitored_train_runner.py", "inference_benchmark.py", "measurement_lib.py"):
        src = (HELPERS / name).read_text(encoding="utf-8")
        for bad in ("shutil.move(", "os.remove(", "os.unlink(", "shutil.rmtree(", "os.rename(", "os.replace("):
            assert bad not in src, (name, bad)
        assert not re.findall(r"open\([^)]*['\"]w\+?['\"]", src), name                     # every write is exclusive ('x') or into a fresh directory
    assert re.findall(r"shutil\.copy2\(([^)]*)\)", (HELPERS / "inference_benchmark.py").read_text(encoding="utf-8")) == ["checkpoint, ck_dst", "config, cf_dst"]
    assert re.findall(r"open\([^)]*['\"]w['\"]", code) == []                         # no plain 'w' opens: writes are exclusive ('x') into MEAS_DIR


@needs_nb
def test_embedded_helpers_equal_the_helper_files_and_reference_hashes_equal_the_archived_manifest():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = _appended_code(nb)[0]
    for name in ("measurement_lib.py", "monitored_train_runner.py", "inference_benchmark.py"):
        body = (HELPERS / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert ("r'''%s'''" % body) in setup, name
    man = json.loads((REPO / "experiments" / "PCG_T4_5_Epoch_Run" / "manifest.json").read_text(encoding="utf-8-sig"))
    by_path = {a["path"]: a["sha256"].upper() for a in man["artifacts"]}
    cell = _appended_code(nb)[2]
    ns = {}
    prev_src = cell[cell.index("_A = "):cell.index("def _entry")]
    exec(prev_src, ns)
    prev = ns["PREV"]
    for e in range(5):
        assert prev["ckpt%d" % e][0] == by_path["checkpoints/SOCIALLSTM_lstm_model_%d.tar" % e]
    assert prev["experiment_config"][0] == by_path["configs/experiment_config.pkl"] and prev["original_config"][0] == by_path["configs/original_config.pkl"]
    assert prev["test_results"][0] == by_path["results/test_results_epoch4.pkl"] and prev["log_curve"][0] == by_path["logs/log_curve.txt"]
    assert prev["loss_plot_batch"][0] == by_path["logs/loss_plot_batch.png"]
    ck181 = hashlib.sha256((REPO / "Store_Results" / "model" / "CollisionGrid" / "SOCIALLSTM_lstm_model_181.tar").read_bytes()).hexdigest().upper()
    assert prev["ckpt181"][0] == ck181


@needs_nb
def test_embedded_retrospective_data_match_the_historical_facts():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = _appended_code(nb)[3]
    raw = cell[cell.index("RETRO_JSON = r'''") + len("RETRO_JSON = r'''"):cell.index("'''\nRETRO = ")]
    R = json.loads(raw)
    runs = R["training_runs"]
    assert len(runs["five_epoch_cell51_default_python"]["batches"]) == 1190 == len(runs["five_epoch_cell49_pcg_env"]["batches"]) and len(runs["one_epoch_cell45_pcg_env"]["batches"]) == 238
    assert math.isclose(runs["five_epoch_cell51_default_python"]["total_training_s"], 748.938973903656) and math.isclose(runs["five_epoch_cell51_default_python"]["grid_calculation_s"], 13.412188291549683)
    assert math.isclose(runs["five_epoch_cell49_pcg_env"]["total_training_s"], 748.4588763713837)
    assert len(R["test_pass"]["per_trajectory"]) == 1081 and R["batch_size"] == 10
    assert R["test_pass"]["protocol"]["sample_size"] == 5 and "ORACLE" in R["test_pass"]["protocol"]["selection"]
    arch = R["archived_epoch_losses_log_curve"]
    by51 = {}
    for gb, ep, loss, t in runs["five_epoch_cell51_default_python"]["batches"]:
        by51.setdefault(ep, []).append(loss)
    assert max(abs(sum(v) / len(v) - a) for (k, v), a in zip(sorted(by51.items()), arch)) < 1e-3          # archived log = cell-51 run
    assert "interpreter" in " ".join(R["not_reconstructable"]) and len(R["environment_findings"]) >= 6


@needs_nb
def test_appended_cells_run_end_to_end_against_a_fake_colab_tree_with_default_flags(tmp_path):
    import notebook_cell_harness as h
    paths = h.build_fake_content(str(REPO), str(tmp_path / "fc"))
    ns = h.run_cells(h.code_cells_of(str(NOTEBOOK)), paths["content"])
    out = Path(ns["MEAS_DIR"])
    names = {p.name for p in out.rglob("*") if p.is_file()}
    for n in ("experiment_environment.json", "artifact_manifest.json", "retrospective_summary.json", "measurement_outputs_index.json", "01_training_loss_over_batches.png",
              "02_batch_time_distribution.png", "03_per_epoch_duration.png", "06b_retro_test_time_NOT_inference_latency.png"):
        assert n in names, n
    assert "04_gpu_utilization_over_time.png" not in names and "05_gpu_memory_over_time.png" not in names and "06_inference_latency_distribution.png" not in names
    assert set(ns["PLOTS_SKIPPED"]) == {"04_gpu_utilization_over_time", "05_gpu_memory_over_time", "06_inference_latency_distribution"}
    assert not (Path(ns["WORK_ROOT"]) / "training").exists() and not (Path(ns["WORK_ROOT"]) / "inference").exists()          # nothing ran
    man = json.loads((out / "artifact_manifest.json").read_text())
    assert man["counts"]["mismatch_previous"] == 0 and man["counts"]["match_previous"] >= 20 and all(p["identical"] for p in man["pair_consistency"])
    env = json.loads((out / "experiment_environment.json").read_text())
    assert env["seeds"]["seed_calls_found_in_repository_python_files"] == [] and "NOT seeded" in env["seeds"]["conclusion"]
    assert env["kernel_interpreter"]["executable"] == sys.executable and {d["label"] for d in env["datasets"]} >= {"Data/HBS/hbs.csv", "Data/test/trajectories_test.cpkl"}
    assert all(d["sha256"] for d in env["datasets"]) and env["training_configurations"]["training_run_config"]["args"]["num_epochs"] == 5
    retro = json.loads((out / "retrospective_summary.json").read_text())
    assert retro["archived_log_curve_crosscheck"]["closest_run"] == "five_epoch_cell51_default_python"
    assert retro["test_pass"]["NOT"].startswith("single-prediction inference latency")
    with pytest.raises(FileExistsError):
        ml.write_new_json(str(out / "artifact_manifest.json"), {})                    # nothing can silently replace an output


@pytest.mark.skipif(not (REPO / "experiments" / "PCG_T4_5_Epoch_Run" / "checkpoints" / "SOCIALLSTM_lstm_model_4.tar").is_file(), reason="archived checkpoint absent")
def test_benchmark_aborts_if_the_working_copy_of_a_checkpoint_differs_from_the_original(tmp_path, monkeypatch):
    exp = REPO / "experiments" / "PCG_T4_5_Epoch_Run"
    real = bench.shutil.copy2

    def corrupting_copy(src, dst, *a, **k):
        out = real(src, dst, *a, **k)
        if str(dst).endswith(".tar"):
            with open(dst, "ab") as fh:
                fh.write(b"corruption")
        return out

    monkeypatch.setattr(bench.shutil, "copy2", corrupting_copy)
    with pytest.raises(RuntimeError, match="hashes differ"):
        bench.run_benchmark(str(REPO), str(tmp_path / "w"), "x", str(exp / "checkpoints" / "SOCIALLSTM_lstm_model_4.tar"), str(exp / "configs" / "experiment_config.pkl"),
                            epoch=4, warmup=1, max_trajectories=3, use_cuda=False)
    assert not (tmp_path / "w" / "measurements" / "inference_benchmark_summary.json").exists()


# ======================================================================================
# Gradient-tracking / train-eval-state behaviour of the revised benchmark (helper v1.1)
# ======================================================================================
CKPT4 = REPO / "experiments" / "PCG_T4_5_Epoch_Run" / "checkpoints" / "SOCIALLSTM_lstm_model_4.tar"
CFG4 = REPO / "experiments" / "PCG_T4_5_Epoch_Run" / "configs" / "experiment_config.pkl"
needs_ckpt = pytest.mark.skipif(not CKPT4.is_file(), reason="archived checkpoint absent")


def tiny_bench(tmp_path, name="w", **kw):
    args = dict(epoch=4, warmup=1, max_trajectories=4, use_cuda=False)
    args.update(kw)
    return bench.run_benchmark(str(REPO), str(tmp_path / name), "t", str(CKPT4), str(CFG4), **args)


def test_static_scope_check_confirms_the_real_baseline_sample_is_fully_under_no_grad():
    s = bench.sample_grad_scope(str(REPO / "test.py"))
    assert s["function_found"] and s["uses_no_grad"] is True and s["uses_inference_mode"] is False
    assert s["net_calls_total"] == 6 == s["net_calls_inside_guard"] and s["returns_total"] == 1 == s["returns_inside_guard"]
    assert s["guards"] == [{"context": "torch.no_grad()", "lines": [393, 599]}] and s["verdict"].startswith("all 6 net(...) calls")


def test_static_scope_check_detects_missing_partial_and_inference_mode_guards(tmp_path):
    def check(body):
        f = tmp_path / "t.py"
        f.write_text(body, encoding="utf-8")
        return bench.sample_grad_scope(str(f))
    none = check("def sample(a, net):\n    x = net(a)\n    return x\n")
    assert none["guards"] == [] and none["verdict"].startswith("NOT all model calls") and none["uses_no_grad"] is False
    partial = check("def sample(a, net):\n    y = net(a)\n    with torch.no_grad():\n        x = net(a)\n    return x\n")
    assert partial["net_calls_total"] == 2 and partial["net_calls_inside_guard"] == 1 and partial["verdict"].startswith("NOT all")
    inf = check("def sample(a, net):\n    with torch.inference_mode():\n        x = net(a)\n    return x\n")
    assert inf["uses_inference_mode"] is True and inf["uses_no_grad"] is False and inf["verdict"].startswith("all 1 net(...)")
    assert check("def other():\n    pass\n") == {"function_found": False, "verdict": "sample() was not found in test.py"}


@needs_ckpt
def test_benchmark_records_that_gradients_were_disabled_in_the_model_and_no_graph_was_built(tmp_path):
    s = tiny_bench(tmp_path)
    g = s["gradient_tracking"]
    assert s["benchmark_version"] == "pcg-inference-benchmark-1.1"
    assert g["observed_inside_model_forward"] == [{"grad_enabled": False, "inference_mode": False}]
    assert g["observed_at_prep_stage_outside_sample"] == [{"grad_enabled": True, "inference_mode": False}] == g["observed_at_post_stage_outside_sample"]
    assert g["torch_grad_enabled_before_benchmark"] is True and g["torch_inference_mode_enabled_before_benchmark"] is False
    assert g["prep_tensors_require_grad"] is False and g["output_tensors_require_grad"] is False and g["output_has_grad_fn"] is False
    assert g["parameters"]["count"] == 12 == g["parameters"]["requiring_grad"] and g["parameters"]["with_gradient_populated"] == 0
    assert g["baseline_sample_function_static_check"]["verdict"].startswith("all 6 net(...) calls")
    assert g["benchmark_added_context"].startswith("none") and "DISABLED during every observed model forward" in g["conclusion"]
    assert "none (no tensor required grad)" in g["conclusion"] and "gradient_tracking" in s["scope"]


@needs_ckpt
def test_benchmark_preserves_the_baseline_train_state_and_never_calls_eval_by_default(tmp_path):
    s = tiny_bench(tmp_path)
    st = s["model_train_eval_state"]
    for k in ("at_load", "before_timing", "after_all_passes"):
        assert st[k] == {"net_training": True, "dropout_modules": 1, "dropout_modules_in_training_mode": 1, "dropout_probabilities": [0.5]}
    assert st["unchanged_during_benchmark"] is True and st["eval_called_by_benchmark"] is False and st["changed_by_benchmark_relative_to_load"] is False
    assert s["protocol"]["net_eval_called"] is False and s["protocol"]["dropout_active_in_prediction_path"] is True


@needs_ckpt
def test_eval_mode_is_only_applied_on_explicit_request_and_is_recorded(tmp_path):
    s = tiny_bench(tmp_path, eval_mode=True)
    st = s["model_train_eval_state"]
    assert st["at_load"]["net_training"] is True and st["before_timing"]["net_training"] is False and st["before_timing"]["dropout_modules_in_training_mode"] == 0
    assert st["eval_called_by_benchmark"] is True and st["changed_by_benchmark_relative_to_load"] is True and st["unchanged_during_benchmark"] is True
    assert s["protocol"]["net_eval_called"] is True and s["protocol"]["dropout_active_in_prediction_path"] is False


def patch_baseline_no_grad(monkeypatch, replacement):
    """Replace torch.no_grad ONLY for calls made from the repository's test.py (model construction etc. keep the real one)."""
    import torch
    real = torch.no_grad

    def scoped(*a, **k):
        caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
        return replacement() if caller.endswith("/test.py") else real(*a, **k)
    monkeypatch.setattr(torch, "no_grad", scoped)


@needs_ckpt
def test_observations_are_real_not_hard_coded_grad_mode_follows_the_code_that_runs(tmp_path, monkeypatch):
    """If sample() were NOT under no_grad the record must say gradients were ENABLED (simulated by making torch.no_grad enable grad)."""
    import torch
    patch_baseline_no_grad(monkeypatch, torch.enable_grad)
    s = tiny_bench(tmp_path)
    g = s["gradient_tracking"]
    assert g["observed_inside_model_forward"] == [{"grad_enabled": True, "inference_mode": False}]
    assert "ENABLED during every observed model forward" in g["conclusion"]
    assert g["output_tensors_require_grad"] is True or g["output_has_grad_fn"] is True                # the graph really was built
    assert "YES" in g["conclusion"]


@needs_ckpt
def test_graph_building_is_reported_per_field_when_sample_is_not_under_no_grad(tmp_path, monkeypatch):
    import torch
    patch_baseline_no_grad(monkeypatch, torch.enable_grad)
    g = tiny_bench(tmp_path)["gradient_tracking"]
    assert g["output_tensors_require_grad"] is True and g["output_has_grad_fn"] is True and g["prep_tensors_require_grad"] is False


@needs_ckpt
def test_populated_parameter_gradients_are_counted_not_assumed(tmp_path, monkeypatch):
    import torch
    sys.path.insert(0, str(REPO))
    try:
        from model.model_CollisionGrid import CollisionGridModel
    finally:
        sys.path.pop(0)
    real = CollisionGridModel.forward

    def forward_leaving_a_gradient(self, *a, **k):
        p = next(self.parameters())
        p.grad = torch.zeros_like(p)                       # simulates a gradient that should not exist at inference time
        return real(self, *a, **k)

    monkeypatch.setattr(CollisionGridModel, "forward", forward_leaving_a_gradient)
    assert tiny_bench(tmp_path)["gradient_tracking"]["parameters"]["with_gradient_populated"] == 1


@needs_ckpt
def test_a_model_that_is_already_in_eval_mode_is_left_exactly_as_loaded(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO))
    try:
        import utils.helper as helper_mod
    finally:
        sys.path.pop(0)
    real_get_model = helper_mod.get_model

    def get_model_in_eval(*a, **k):
        net = real_get_model(*a, **k)
        net.eval()
        return net

    monkeypatch.setattr(helper_mod, "get_model", get_model_in_eval)
    s = tiny_bench(tmp_path)
    st = s["model_train_eval_state"]
    assert st["at_load"]["net_training"] is False and st["at_load"]["dropout_modules_in_training_mode"] == 0
    assert st["before_timing"] == st["at_load"] == st["after_all_passes"] and st["eval_called_by_benchmark"] is False
    assert s["protocol"]["dropout_active_in_prediction_path"] is False                  # derived from the observed state, not from a flag


@needs_ckpt
def test_swapping_only_the_guard_inside_sample_for_inference_mode_is_incompatible_with_the_post_stage(tmp_path, monkeypatch):
    """Compatibility finding behind the decision NOT to add an inference_mode variant: tensors returned by sample() under inference_mode are
    inference tensors, and the (baseline-identical) post-processing updates one of them in place outside inference mode, which PyTorch forbids."""
    import torch
    patch_baseline_no_grad(monkeypatch, torch.inference_mode)
    with pytest.raises(RuntimeError, match="inference tensor"):
        tiny_bench(tmp_path)
    assert not (tmp_path / "w" / "measurements" / "inference_benchmark_summary.json").exists()


@needs_ckpt
def test_benchmark_aborts_if_anything_changes_the_model_train_eval_state(tmp_path, monkeypatch):
    import torch
    real = torch.nn.Dropout.forward

    def forward_that_switches_to_eval(self, x):
        self.eval()                                        # a hidden state change inside the timed code
        return real(self, x)

    monkeypatch.setattr(torch.nn.Dropout, "forward", forward_that_switches_to_eval)
    with pytest.raises(RuntimeError, match="changed the model's train/eval state"):
        tiny_bench(tmp_path)
    assert not (tmp_path / "w" / "measurements" / "inference_benchmark_summary.json").exists()


@needs_ckpt
def test_checkpoint_and_config_hash_checks_and_isolation_still_hold_in_v1_1(tmp_path):
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (CKPT4, CFG4, REPO / "test.py")]
    s = tiny_bench(tmp_path, expected_checkpoint_sha256="ac9f5462ec741b91541fa6be74f07167060d63a469f88936c9d5fbc833387d48",
                   expected_config_sha256="9c994a14a5f6519b2c78256b3603c2e26975ddb647ac455bf13aeccd18485ba5")
    assert s["hashes"]["checkpoint_matches_expected"] is True and s["hashes"]["config_matches_expected"] is True
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (CKPT4, CFG4, REPO / "test.py")] == before
    with pytest.raises(RuntimeError, match="NOT started"):
        tiny_bench(tmp_path)                                                       # same (now non-empty) work directory is refused
    wrong = tiny_bench(tmp_path, name="w2", expected_checkpoint_sha256="0" * 64)
    assert wrong["hashes"]["checkpoint_matches_expected"] is False


# ------------------------------------------------------------- revised notebooks
TRIMMED = MEAS / "PCG_Baseline_Reproduction_Measurements_Trimmed.ipynb"
TRIMMED_SHA = "1108f2bfab74d0b968428692c33b98e39a070715b31b001362ed05dc3884fe71"
V2 = MEAS / "PCG_Baseline_Reproduction_Measurements_Trimmed_v2.ipynb"
needs_v2 = pytest.mark.skipif(not (V2.is_file() and TRIMMED.is_file()), reason="trimmed notebooks absent")


def _src(c):
    return "".join(c["source"])


@needs_v2
def test_v2_differs_from_the_trimmed_notebook_in_exactly_the_documented_cells():
    raw = TRIMMED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIMMED_SHA                         # the trimmed notebook itself is untouched
    old, new = json.loads(raw.decode("utf-8")), json.loads(V2.read_text(encoding="utf-8"))
    assert len(new["cells"]) == len(old["cells"]) + 1 and new["metadata"] == old["metadata"]
    assert new["cells"][0] == old["cells"][0] and "Revision note" in _src(new["cells"][1]) and new["cells"][1]["cell_type"] == "markdown"
    changed = [k for k in range(2, len(new["cells"])) if new["cells"][k] != old["cells"][k - 1]]
    assert len(changed) == 2
    setup, bench_cell = new["cells"][changed[0]], new["cells"][changed[1]]
    assert "HELPER_SOURCES = " in _src(setup) and "bench_mod.run_benchmark(" in _src(bench_cell)
    assert setup["outputs"] == [] and bench_cell["outputs"] == [] and setup["execution_count"] is None and bench_cell["execution_count"] is None
    for k in range(2, len(new["cells"])):
        if k not in changed:
            assert new["cells"][k] == old["cells"][k - 1], k                         # incl. the saved outputs of the manifest and retrospective cells


@needs_v2
def test_v2_preserves_the_retrospective_measurements_and_artifact_hash_evidence():
    old, new = json.loads(TRIMMED.read_text(encoding="utf-8")), json.loads(V2.read_text(encoding="utf-8"))
    old_man = next(c for c in old["cells"] if "PART C - ARTIFACT INTEGRITY MANIFEST" in _src(c))
    new_man = next(c for c in new["cells"] if "PART C - ARTIFACT INTEGRITY MANIFEST" in _src(c))
    old_retro = next(c for c in old["cells"] if "PART F - RETROSPECTIVE ANALYSIS" in _src(c))
    new_retro = next(c for c in new["cells"] if "PART F - RETROSPECTIVE ANALYSIS" in _src(c))
    assert new_man == old_man and new_retro == old_retro
    out = "".join("".join(o.get("text", [])) for o in new_man["outputs"])
    assert '"match_previous": 21' in out and '"mismatch_previous": 0' in out and '"missing": 0' in out and '"exist": 22' in out
    retro_out = "".join("".join(o.get("text", [])) for o in new_retro["outputs"])
    assert "total 748.94 s" in retro_out and "grid 13.41 s" in retro_out and "closest run: five_epoch_cell51_default_python" in retro_out


@needs_v2
def test_v2_embedded_helpers_equal_the_helper_files_and_cells_compile_with_the_gate_off():
    new = json.loads(V2.read_text(encoding="utf-8"))
    setup = next(_src(c) for c in new["cells"] if c["cell_type"] == "code" and "HELPER_SOURCES = " in _src(c))
    start = setup.index("HELPER_SOURCES = ") + len("HELPER_SOURCES = ")
    d = ast.literal_eval(setup[start:setup.index("\n\nHELPER_SHA256", start)])
    for name in ("measurement_lib.py", "inference_benchmark.py"):
        assert d[name] == (HELPERS / name).read_text(encoding="utf-8").replace("\r\n", "\n"), name
    assert "sample_grad_scope" in d["inference_benchmark.py"] and 'BENCH_VERSION = "pcg-inference-benchmark-1.1"' in d["inference_benchmark.py"]
    for c in new["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in _src(c).splitlines() if not l.lstrip().startswith(("%", "!"))))
    gate = next(_src(c) for c in new["cells"] if c["cell_type"] == "code" and _src(c).lstrip().startswith("RUN_INFERENCE_BENCHMARK"))
    assert re.match(r"RUN_INFERENCE_BENCHMARK = False", gate)
    bench_cell = next(_src(c) for c in new["cells"] if c["cell_type"] == "code" and "bench_mod.run_benchmark(" in _src(c))
    assert "sample_grad_scope" in bench_cell and 'BENCH_SUMMARY["gradient_tracking"]' in bench_cell and "never calls net.eval()" in bench_cell
    assert "WARNING: the baseline sample() is not fully under torch.no_grad()" in bench_cell
    assert "expected_checkpoint_sha256=EXPECTED_CKPT" in bench_cell and "expected_config_sha256=EXPECTED_CONFIG" in bench_cell
    assert "does not match the archived SHA-256" in bench_cell                                                   # the user's hash gate is retained


@needs_v2
def test_v2_benchmark_cell_runs_on_cpu_against_a_fake_colab_tree_and_prints_the_gradient_record(tmp_path, capsys):
    import notebook_cell_harness as h
    paths = h.build_fake_content(str(REPO), str(tmp_path / "fc"))
    new = json.loads(V2.read_text(encoding="utf-8"))
    codes = [_src(c) for c in new["cells"] if c["cell_type"] == "code"]
    setup = next(c for c in codes if "HELPER_SOURCES = " in c).replace("BENCH_WARMUP = 20", "BENCH_WARMUP = 2").replace("BENCH_MAX_TRAJECTORIES = None", "BENCH_MAX_TRAJECTORIES = 6")
    manifest = next(c for c in codes if "PART C - ARTIFACT INTEGRITY MANIFEST" in c)
    gate = next(c for c in codes if c.lstrip().startswith("RUN_INFERENCE_BENCHMARK")).replace("= False", "= True")
    bench_cell = next(c for c in codes if "bench_mod.run_benchmark(" in c)
    root = paths["content"].replace("\\", "/")
    initial = {"REPO": paths["repo"].replace("\\", "/"), "DRIVE_BACKUP": paths["drive"].replace("\\", "/"), "STORE": paths["repo"].replace("\\", "/") + "/Store_Results"}
    ns = h.run_cells([setup, manifest, gate, bench_cell], root, initial=initial)
    out = capsys.readouterr().out
    assert "Gradient scope of the baseline sample(): all 6 net(...) calls in sample() run under torch.no_grad()" in out
    assert "gradient tracking: gradient tracking DISABLED during every observed model forward" in out
    assert "unchanged during benchmark = True" in out and "eval called by benchmark = False" in out
    S = ns["BENCH_SUMMARY"]
    assert S["gradient_tracking"]["observed_inside_model_forward"] == [{"grad_enabled": False, "inference_mode": False}]
    assert S["hashes"]["checkpoint_matches_expected"] is True and S["hashes"]["config_matches_expected"] is True
    assert (Path(ns["BENCH_OUTPUT_DIR"]) / "inference_benchmark_summary.json").is_file()
