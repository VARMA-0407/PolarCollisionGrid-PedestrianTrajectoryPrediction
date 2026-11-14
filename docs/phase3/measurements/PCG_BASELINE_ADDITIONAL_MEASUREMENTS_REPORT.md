# PCG baseline: additional measurements report

**Date:** 14 November 2026
**Deliverable:** `PCG_Baseline_Reproduction_Measurements.ipynb` (a copy of the historical notebook with 9 appended cells).
**Status:** instrumentation built and validated locally on CPU. **Nothing was run on Colab, no GPU was used, no training was started, and no new GPU, training-resource or inference-latency number exists yet.**

Labels used below: **RETROSPECTIVE** = extracted from outputs already saved in the original notebook (no new execution). **NEW RUN** = would be collected by the new instruments (not yet run). **VALIDATION** = local CPU run that checks the code works; those numbers are not results.

## 1. Headline findings (please read first)

1. **Two complete 5-epoch trainings are in the saved outputs, not one.** Cell 49 (execution 54) has a malformed first command (`--num_epochs 5%cd ...`, an argparse usage error) but its second command trained for 5 epochs with the explicit `/content/pcg-env/bin/python`. Cell 51 (execution 56) then trained again with plain `!python`. Both write the same `Store_Results` file names, so cell 51 overwrote cell 49's checkpoints and log.
2. **The archived artifacts come from the cell-51 (plain `!python`) run.** The archived `log_curve.txt` epoch losses match the cell-51 batch means to 3.2e-05, and the cell-49 run differs by up to 0.069. So the archived checkpoints were **not** produced by the README-conformant `pcg-env` (PyTorch 2.2.1+cu121). What interpreter `!python` resolved to is not recorded; the kernel was `/usr/bin/python3` 3.13.16 with PyTorch 2.14.1+cu130 (cells 0 and 11). The grid-calculation time (13.4 s versus 30.6 s and 31.9 s in the two `pcg-env` runs) is consistent with a different environment but does not prove it.
3. **The historical test runs with dropout active.** `test.py` never calls `net.eval()` and the model contains `nn.Dropout(0.5)`. Together with the Gaussian sampling, this makes the five test samples differ and feeds the best-of-five oracle selection. I could not tell whether this was intended.
4. **Cell 41 does not show what its message says.** It prints "DataLoader works inside Python 3.11!" but its shell command failed (unmatched quote) and the remaining lines ran in the kernel (Python 3.13.16, PyTorch 2.14.1+cu130).
5. **The baseline code sets no random seed anywhere** (searched the repository), so the runs are not reproducible and the two 5-epoch runs are two independent samples.

The task description said the five-epoch run used `!python` and the earlier one-epoch run used the custom environment. That is true for cells 45 and 51, but cell 49 is a second five-epoch run in the custom environment that the description did not mention.

## 2. Measurements already available (RETROSPECTIVE)

Source: saved outputs of the original notebook (SHA-256 `b20af1a163b71a495dd3c37bd49fc150fb0b243de716df58173010c79dfeea10`), parsed programmatically. All three runs have `batch_size=10`.

| Item | Cell 45 (1 epoch, pcg-env) | Cell 49 (5 epochs, pcg-env, overwritten) | Cell 51 (5 epochs, plain `!python`, **archived**) |
|---|---|---|---|
| Batches logged | 238 | 1,190 | 1,190 (238 per epoch) |
| Grid calculation | 31.91 s | 30.57 s | 13.41 s |
| Total training time (printed) | 151.35 s | 748.46 s | 748.94 s |
| Sum of printed batch times | 150.28 s | 742.64 s | 743.62 s |
| Batch time median / p90 / p95 / max | 0.601 / 0.809 / 0.879 / 1.033 s | 0.590 / 0.802 / 0.858 / 1.109 s | 0.599 / 0.785 / 0.831 / 1.737 s |
| First batch | 0.968 s | 0.974 s | 1.737 s |
| Throughput from the printed total | 1.57 batches/s | 1.59 batches/s | 1.59 batches/s (15.9 sequences/s) |

- **Per-epoch mean batch loss (cell 51):** -0.9362, -1.6138, -1.7975, -1.9391, -2.0328. The archived `log_curve.txt` says -0.93625, -1.61383, -1.79745, -1.93913, -2.03282. Cell 49 gave -0.9283, -1.6565, -1.8660, -1.9841, -2.0715.
- **Per-epoch time (cell 51):** the sum of printed batch times is 150.31, 149.44, 146.68, 147.07 and 150.13 s. The checkpoint modification times (cell 54) give epoch 1 to 4 durations of 150, 148, 148 and 151 s (1 s resolution, Drive mount caveat); epoch 0 is 151.9 s by subtraction from the printed total. This is an **estimate**, not a measurement.
- **Historical test pass (cell 60, plain `!python`):** 1,081 trajectories, per-trajectory time median 0.290 s, p90 0.462 s, p95 0.524 s, p99 0.652 s, mean 0.308 s, max 0.776 s. **This covers 5 sampled rollouts plus ground-truth scoring plus best-of-five oracle selection, so it is not single-prediction latency.**
- **Reported metrics (final summary, preserved unchanged):** ADE 0.5588, FDE 1.1382, MHD 3.0712, speed error 0.3733, heading error 40.81, collision 2.17% (ped-ped 5.91%, ped-veh 0.21%). The per-iteration lines of the same output show speed error 0.1393 and heading error 1665.66; I recorded the difference and did not reconcile it.

## 3. New measurement instrumentation (appended cells)

| Cell | Part | What it does | Runs by default? |
|---|---|---|---|
| 1 (markdown) | | Explains the appended cells, the two switches and the interpreter warning | n/a |
| 2 setup | | Creates a fresh output directory (`measurements/meas_<UTC>`), writes the embedded helper files, defines the switches | yes (writes only into the fresh directory) |
| 3 | B | `experiment_environment.json`: UTC time, kernel interpreter and the `pcg-env` interpreter (probed), OS, CPU, RAM, library versions, CUDA/GPU/driver, git commit and working-tree status, both training configurations with hashes, dataset and split file hashes, source-file hashes, a repository-wide search for seeds | yes |
| 4 | C | `artifact_manifest.json`: path, size, mtime, SHA-256, existence and match against previously recorded hashes for the epoch 0-4 checkpoints (working copy and Drive backup), both configs, the epoch-181 checkpoint, the test pickle, logs, plots; plus working-copy-versus-backup consistency | yes (read only) |
| 5 | F | Embedded retrospective data plus analysis: `retrospective_summary.json` and three CSVs | yes |
| 6 | D | Monitored training runner (pre-flight shown, nothing started) | **no**: needs `RUN_MONITORED_TRAINING = True` **and** the exact run id in `CONFIRM_RUN_ID` |
| 7 | E | Single-sample predictor benchmark | **no**: needs `RUN_INFERENCE_BENCHMARK = True` |
| 8 | G | Six figures (only those with data) | yes |
| 9 | | Output index with a hash of every produced file and a "still to measure" list | yes |

### Part D: monitored training runner
- Runs the **unmodified** repository `train.py` in-process inside a **fresh working directory** with its own copy of `Data/` and its own `Store_Results/`. train.py writes everything relative to the current directory, so this is what protects existing checkpoints, configs and logs.
- **Pre-flight (the run does not start if any check fails):** repository files exist; the work directory is new or empty, not a link, not inside the repository; the repository's `Store_Results` tree is snapshotted before and verified unchanged after, and source and data hashes are verified too.
- **Records:** total wall time; per-epoch wall time; per-batch CUDA-synchronised time (median, p90, p95, max, with and without the first batch); throughput; GPU utilisation and device memory sampled every second (NVML or `nvidia-smi`); PyTorch allocated, reserved and peak memory; CPU utilisation; process RSS, sampled peak and kernel `ru_maxrss`; timed `DataLoader.__init__`, `grid_creation`, each epoch's `batch_creater` and each `torch.save`; the exact interpreter, versions, GPU and driver; the training arguments train.py itself wrote; the per-batch loss.
- **Outputs:** `training_resource_log.csv`, `training_epoch_summary.csv`, `training_resource_summary.json`, plus `training_batch_log.csv` and `training_stdout.log`.
- Interpreter: the run uses `PYTHON_FOR_NEW_RUNS`, or the kernel's interpreter if unset. Set it explicitly. For comparability with the README use `/content/pcg-env/bin/python`.

### Part E: single-sample predictor benchmark
- **Why `test.py` timing cannot be reused:** see section 2 (5 samples + scoring + oracle selection).
- **Design:** one sample per input, fixed seeds, warm-up trajectories excluded (default 20), `torch.cuda.synchronize()` around every timed stage, separate passes for the pipeline (prep, transfer, rollout, post) and for the model forwards (hooks), p50/p90/p95/p99 and mean/std, throughput in sequences/s, peak GPU memory after warm-up, checkpoint and config hashes checked against the previously recorded values.
- **How it stays faithful without changing the baseline:** `test.py` is not modified or run. The baseline `sample` function is imported from it unchanged, and its per-trajectory preparation and post-processing steps are replicated step for step. Ground-truth scoring and oracle selection are left out. `net.eval()` is not called (as in `test.py`) unless requested, and this is recorded.
- **Label written into every output:** "PCG trajectory-predictor-only, single-sample (sample_size=1), baseline code path. NOT end-to-end video performance. NOT deployment-valid: the baseline path still uses future actor information as in the historical test."
- The checkpoint and config are copied into a fresh work directory; the originals are only read.

## 4. Measurements that require a new run
- GPU utilisation, GPU memory (device, allocated, reserved, peak), CPU utilisation, RAM, process memory (training and inference).
- Per-epoch wall-clock time as a measurement, data-loading time per epoch, checkpoint-save time.
- CUDA-synchronised per-batch time.
- Single-sample predictor latency (p50/p90/p95/p99), model-forward-only latency, inference throughput, inference peak GPU memory.
- Plots 4, 5 (GPU utilisation and memory over time) and 6 (inference latency distribution): they are skipped until the data exist.
- The Colab hardware label and driver for those runs.

## 5. Measurements that cannot be reconstructed historically
- Any GPU, CPU or memory telemetry from the historical training or test (only two idle `nvidia-smi` snapshots exist, cells 1 and 53).
- Explicit per-epoch wall time (only the estimate above), data-loading time, checkpoint-save time.
- What interpreter plain `!python` used in cells 51 and 60.
- The checkpoints, config and log of the overwritten cell-49 run (only its printed output survives).
- Any seed (none was set).
- Single-sample or model-only inference latency (never measured).

## 6. Historical oracle evaluation limitation (preserved)
The reported metrics use `sample_size=5` and best-of-five selection by lowest ADE **against ground truth** (oracle selection). They are not single-sample results and not deployment metrics. The test path also uses future actor information and runs with dropout active. None of this was changed or hidden; the new benchmark states in its output that it excludes scoring and oracle selection.

## 7. Environment inconsistency findings
See section 1 plus: installing PyTorch 2.2.1 into the default environment failed (cell 10, the index only offered 2.5.x), which is why `pcg-env` was created; at one point `pcg-env` had a NumPy 2.x incompatibility with its torch build (cell 24, fixed by cell 28); `Store_Results` was replaced by a symlink to Drive (cell 47), so training wrote checkpoints through the Drive mount; the evaluation cell overwrote `Store_Results/model/CollisionGrid/config.pkl` with the new run's config (the original is preserved only in the Drive backup and in the project repository).

## 8. Artifact hashes
**Previously recorded (archived experiment manifest) and confirmed against the local copies in this audit** (the notebook's manifest cell checks the Colab copies against these at run time):

| Artifact | SHA-256 |
|---|---|
| Epoch 0 checkpoint | `876d017cd6d381fbbdd1a9976a5a2843fdceaeba0fb45560db406df410c825ce` |
| Epoch 1 checkpoint | `ebee25a064e6da5d235c5b23ecf8c1e55489ef8a5eea654a279a6008d90d43f8` |
| Epoch 2 checkpoint | `3acf2412362c342e3426a7be4c8b44558241ed3302970cf60554d832e5d0ec74` |
| Epoch 3 checkpoint | `3a4ebd8bc4bff5b960b9090acb8ebd16640a01e5cb834b7dbfaa07d42f06b0ef` |
| Epoch 4 checkpoint | `ac9f5462ec741b91541fa6be74f07167060d63a469f88936c9d5fbc833387d48` |
| Training config (`experiment_config.pkl`) | `9c994a14a5f6519b2c78256b3603c2e26975ddb647ac455bf13aeccd18485ba5` |
| Original CollisionGrid config | `763e68c787524e836dd20dc833f6bf949df488591f01bc74bdab375ecb8e43d9` |
| Epoch-181 checkpoint (no earlier recorded hash; hashed from the git-tracked copy) | `a01b3c59c425e070e7e25db47c692226304e83c1f285c35d00afa4bf1101775f` |
| Test predictions (`test_results_epoch4.pkl`) | `119e9bb665e2168035062fa075343aba46cc9b2f2f8810054c16a996579192de` |
| `log_curve.txt` | `89bd93363b9d829b9876accccc242951db1e00ed6d91561304226466535d84c1` |

**Files delivered (SHA-256):**

| File | SHA-256 |
|---|---|
| `PCG_Baseline_Reproduction_Measurements.ipynb` | `a8be2ede43f7f8c6416efca63ce1ac60f4ceab293fab0c49e6dda291132a4853` |
| Original `PCG_Baseline_Reproduction.ipynb` (unchanged) | `b20af1a163b71a495dd3c37bd49fc150fb0b243de716df58173010c79dfeea10` |
| `helpers/measurement_lib.py` | `711044d6befd4f5592b9b07220991f87d314e9dcf662817bfd81bb282c253c7b` |
| `helpers/monitored_train_runner.py` | `9f8898655f824617bca689881129835be9ba4a5de5b5d713ad007902536a48ab` |
| `helpers/inference_benchmark.py` | `85184647c778322844d917cdf777c73243f7f770aff01ea1e029f1f3552d90ee` |
| `helpers/notebook_cell_harness.py` | `4c71628208101d6b1ba1f76104a153285d3f35eceb144472ef35ad8ebe20bf19` |
| `build_measurement_notebook.py` | `3e2e6a8977f1ddb156fe6a3b1189eab00b4448e5ffe62e76104b437080bd243b` |
| `tests/phase3/test_measurement_helpers.py` | `c84954470585ccb33328bd236aee171fd710f3079eabe4648ad8768ca703c6cb` |

## 9. Test and validation results
Commands, run with the `pcg-baseline` interpreter (Python 3.9.25): `python -m pytest tests/phase3/test_measurement_helpers.py`, `tests/phase3`, `tests/task7`, `tests/phase2/test_agent_state_schema.py`, `tests/phase2`.

| Run | Result |
|---|---|
| `test_measurement_helpers.py` | 19 passed |
| `tests/phase3` (Stage 1, Stage 2A, measurement) | 124 passed |
| `tests/task7` | 581 passed |
| Schema | 22 passed |
| `tests/phase2` | 33 passed, 1 failed (the known original `Interaction.py` `TTC_mask` defect, unchanged) |

What the 19 tests cover: hashing and match/mismatch/missing records; exclusive writes that refuse to overwrite; percentile and log parsing; the monitored runner against a **fake train module on CPU** (8 batches, 2 epochs, per-batch and per-epoch records, resource CSVs, isolation proof that the fake repository is byte-identical, refusal of non-empty or in-repository work directories, a second run refused, a failing training still writes a "failed" summary and re-raises); the benchmark pre-flight; a real CPU benchmark run (VALIDATION only) that leaves all sources unchanged and aborts if a copied checkpoint differs; the notebook (original 64 cells identical to the original file, 9 cells appended, no output in appended cells, flags default off, no deleting or in-place writing calls, embedded helpers equal to the files, reference hashes equal to the archived manifest, embedded retrospective data equal to the historical facts); and a full execution of all appended cells against a fake Colab-style tree.

**Mutation checks:** 12 safeguards were broken one at a time in a scratch copy (non-empty work directory accepted, in-repository work directory accepted, epoch markers not parsed, isolation check made vacuous, optimizer step not instrumented, training failure swallowed, no isolated Data copy, exclusive write replaced by overwrite, non-empty directory accepted, hash comparison inverted, benchmark non-empty work directory accepted, copy-hash verification removed). 11 failed a test at once; the last survived because that guard cannot trigger when the copy is correct, so a corruption test was added and it is now caught.

**Not validated:** the training runner has not been run with real `train.py` or CUDA (train.py always requires CUDA); GPU timing synchronisation, NVML/`nvidia-smi` sampling and the Colab paths could not be exercised locally. The first real run is the real test.

**VALIDATION run of the benchmark (not a result):** on this laptop CPU with 6 timed trajectories after 2 warm-up, the pipeline median was about 0.2 s; it says nothing about the T4 and must not be compared with anything.

## 10. Exact file locations
- `D:\BTech-Research\PCG-Baseline\docs\phase3\measurements\PCG_Baseline_Reproduction_Measurements.ipynb` (also copied, byte-identical, to `%USERPROFILE%\Downloads\PCG_Baseline_Reproduction_Measurements.ipynb`)
- `D:\BTech-Research\PCG-Baseline\docs\phase3\measurements\PCG_BASELINE_ADDITIONAL_MEASUREMENTS_REPORT.md` (this file)
- Helpers: `docs\phase3\measurements\helpers\` ; cell sources: `docs\phase3\measurements\cells\` ; builder: `docs\phase3\measurements\build_measurement_notebook.py`
- Tests: `tests\phase3\test_measurement_helpers.py`
- On Colab the notebook writes to `/content/drive/MyDrive/PCG_Experiments/measurements/meas_<UTC>/` (or `/content/pcg_measurements/` without Drive) and uses `/content/pcg_runs/meas_<UTC>/` for any new run's working directories.
- Note: `Downloads` also contains `PCG_Baseline_Reproduction (1).ipynb`; I did not read or touch it.

## 11. Safety record
The original notebook is byte-identical (SHA-256 above). Tracked repository files are unchanged (`git diff` empty); the 20-file experiment manifest still verifies 20/20; `train.py`, `test.py`, `Interaction.py`, the schema and `hbs.csv` hashes are unchanged. Nothing was deleted except my own earlier build outputs of this notebook (rebuilt twice). No package was installed and nothing was downloaded.

## 12. Recommended next experiment
Run the notebook on a Colab T4 in this order:
1. Run the appended cells 1 to 5 and 8 to 9 (all safe) and check `artifact_manifest.json` for any mismatch, in particular the test pickle and the `CollisionGrid/config.pkl`.
2. Set `RUN_INFERENCE_BENCHMARK = True` (about a few minutes) with `PYTHON_FOR_NEW_RUNS` unchanged, and record which interpreter the kernel is.
3. Then, as a separate session, one monitored 5-epoch training with `PYTHON_FOR_NEW_RUNS = "/content/pcg-env/bin/python"` (README environment), with a fresh run id. This also yields a third independent 5-epoch result and the first real GPU telemetry.
4. Before spending effort on any accuracy comparison, decide whether the baseline's test-time dropout is intended, and use the corrected non-oracle evaluator (Phase 3 Stage 2) for any single-sample accuracy claim.
