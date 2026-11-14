# Benchmark revision: gradient tracking and model-state recording

**Date:** 14 November 2026
**Scope:** review of the trimmed notebook's embedded `inference_benchmark.py` before the benchmark is enabled on Colab.
**Not done:** nothing was executed on Colab or a GPU, no training was run, the original notebooks and the PCG source were not touched. All measurements below are local CPU validation of the code, not results.

## 1. Answers to the review questions

**Does the baseline `sample()` already disable gradient tracking? Yes.** `test.py` line 393 opens `with torch.no_grad():` and the block runs to line 599, which is the end of the function. A static check of the file's syntax tree (`sample_grad_scope`) finds all 6 `net(...)` calls (lines 418, 433, 440, 488, 492, 498) and the single `return` inside that block. It does **not** use `torch.inference_mode()`.

**Consequence.** Your instruction was to implement an inference-mode benchmark only if `sample()` does not already disable gradients. It does, so I did **not** add an extra `no_grad` or `inference_mode` context. The timed code is exactly the baseline execution.

**Compatibility finding that supports that decision.** I tested what happens if the guard inside `sample()` is swapped for `torch.inference_mode()` (simulated on CPU, scoped to `test.py` only). The run fails with "Inplace update to inference tensor outside InferenceMode is not allowed": the tensors returned by `sample()` become inference tensors and the baseline post-processing updates one of them in place outside inference mode. So an `inference_mode` variant is not a drop-in change. If one is ever wanted it would have to wrap the whole per-trajectory function and would need its own validation; I did not implement or test that. A test now records this failure so it is not rediscovered.

**What about the stages outside `sample()`?** Preparation and post-processing run with grad mode **enabled**, exactly as in `test.py`. They build no autograd graph because no tensor in them requires grad (checked at run time).

## 2. Exact changes

| File | Change |
|---|---|
| `helpers/inference_benchmark.py` (v1.0 to v1.1) | Added `sample_grad_scope()` (static check of `test.py`'s `sample`). The run now records **observed** grad mode and inference mode inside every model forward (forward pre-hook) and at the preparation and post stages, whether any preparation or output tensor required grad or had a `grad_fn`, parameter counts (total, `requires_grad`, gradient populated), and the model's train/eval state and dropout-module state at load, before timing and after all passes. It raises if the state changed during the run. New summary keys: `gradient_tracking`, `model_train_eval_state`. `protocol.dropout_active_in_prediction_path` is now derived from the observed state instead of a flag. Docstring, scope text and limitations updated. |
| `PCG_Baseline_Reproduction_Measurements_Trimmed_v2.ipynb` (new) | Built from your trimmed notebook by `build_trimmed_v2.py`. Differences from the trimmed notebook: one new markdown revision-note cell after the title; the setup cell's embedded `inference_benchmark.py` replaced by v1.1 (`measurement_lib.py` unchanged); the benchmark control cell now prints the static `sample()` check, warns if it is not fully under `no_grad`, and prints the gradient conclusion and model state after a run. Saved outputs of those two changed cells were cleared because they describe the old helper. |
| `helpers/notebook_cell_harness.py` | `run_cells` accepts an initial namespace (test support only). |
| `cells/c06_inference_benchmark.py` and the full notebook | Same benchmark-cell additions; the full measurement notebook was rebuilt from its sources (new hash below). |
| `tests/phase3/test_measurement_helpers.py` | 16 new tests (section 4). |

**Unchanged and verified:** the benchmark's checkpoint and config hash checks, the copy-hash verification, the isolated fresh working directory with copies of the checkpoint, config and data, and the refusal of a non-empty or in-repository work directory. The model is never switched to eval unless `eval_mode=True` is passed explicitly, and the notebook never passes it. If the model is already in eval mode when loaded, it is left that way and the record says so.

**Preserved in `v2`:** every other cell is byte-identical to the trimmed notebook, including the saved Colab outputs of the artifact manifest (22 entries, 21 matching, 0 mismatched, 0 missing) and the retrospective report (748.94 s total, 13.41 s grid time, archived log matches the cell-51 run). Those outputs belong to the earlier Colab session; re-run from the top and they are regenerated.

## 3. What the new record looks like on this machine (CPU validation, 9 timed trajectories, not a result)

- Static check: "all 6 net(...) calls in sample() run under torch.no_grad()".
- Observed inside the model forward: grad enabled = False, inference mode = False.
- Observed at preparation and post-processing: grad enabled = True, inference mode = False; no tensor required grad; outputs have no `grad_fn`.
- Parameters: 12, all `requires_grad=True`, 0 with a gradient stored.
- Model state at load, before timing and after: `net.training = True`, 1 dropout module (p = 0.5) in training mode; `eval` not called; unchanged.

On the T4 the same fields will be filled by the actual run. The CUDA path of the benchmark could not be exercised here.

## 4. Tests

Commands, run with the `pcg-baseline` interpreter (Python 3.9.25):

```bash
python -m pytest tests/phase3/test_measurement_helpers.py
python -m pytest tests/phase3
python -m pytest tests/task7
python -m pytest tests/phase2/test_agent_state_schema.py
python -m pytest tests/phase2
```

| Run | Result |
|---|---|
| `test_measurement_helpers.py` | 35 passed (19 before, 16 new) |
| `tests/phase3` | 140 passed |
| `tests/task7` | 581 passed |
| Schema | 22 passed |
| `tests/phase2` | 33 passed, 1 failed (the known original `Interaction.py` `TTC_mask` defect) |

The 16 new tests cover: the static scope check on the real `test.py`; detection of a missing guard, a partial guard, an `inference_mode` guard and a missing function; the recorded gradient state on the real baseline; preservation of the train/eval/dropout state and no `eval()` by default; `eval` only on explicit request and recorded; real (not hard-coded) observation, by making `sample()` run with gradients on and checking the record says enabled and that a graph was built; per-field graph reporting; counting of populated parameter gradients; a model that is already in eval mode left as loaded; abort if anything changes the model state mid-run; the `inference_mode` incompatibility; checkpoint and config hash checks and isolation still holding; `v2` differs from the trimmed notebook in exactly the documented cells; retrospective outputs and manifest outputs preserved; embedded helpers equal the helper files; the gate is off; the benchmark cell runs on CPU against a fake Colab-style tree and prints the gradient record.

**Mutation checks:** 10 behaviours were broken one at a time in a scratch copy (forward grad mode hard-coded, state change not detected, `eval()` always called, static check counting every call as guarded, dropout flag hard-coded, output grad not observed, gradient count hard-coded, load state hard-coded, non-empty work directory accepted, copy-hash check removed). The first run left three alive; stronger tests were added and all 10 are now caught. (An earlier scratch run reported false catches because my scratch copy lacked some files; I fixed the scratch tree and re-ran.)

## 5. Integrity

- Branch `reproduction`, HEAD `7a6f56f`; tracked files unchanged; 26 untracked entries (all new work is in untracked directories).
- Unchanged by SHA-256: original `PCG_Baseline_Reproduction.ipynb` (`b20af1a1...`), the trimmed notebook (`1108f2bf...`, checked by a test), `test.py` (`7e207517...`), `train.py` (`049f0f58...`), `Interaction.py` (`127f9d47...`), `hbs.csv` (`eccd7478...`). The 20-file experiment manifest still verifies 20/20.
- New or changed files (SHA-256):

| File | SHA-256 |
|---|---|
| `PCG_Baseline_Reproduction_Measurements_Trimmed_v2.ipynb` | `c2cb26e55228a5c135b2b5a41d161e4f50e8ad304cbd70fb5286f42ac358aa30` |
| `helpers/inference_benchmark.py` | `8a483de4e4bb5ae317d78f90b35f2b507fda50e9a828244545e4377b2b3bac7b` |
| `helpers/measurement_lib.py` (unchanged) | `711044d6befd4f5592b9b07220991f87d314e9dcf662817bfd81bb282c253c7b` |
| `helpers/notebook_cell_harness.py` | `2b14196b8e44fa6c89eb6fd582e53083694dd90568f46a7cdfb18b4b3943f411` |
| `build_trimmed_v2.py` | `6630ecc812561eada9c2eba4e8419c9b28e3889d3377062c86771c17f0c8cca9` |
| `cells/c06_inference_benchmark.py` | `da5c3611ec74de69a5e4c78983b6963d7500dee559f762836cf5c24216911f40` |
| `PCG_Baseline_Reproduction_Measurements.ipynb` (full notebook, rebuilt) | `79b7501147ca18cab910de0bef11c9c40ec9dc652b6cbb71001398130b45f13d` |
| `tests/phase3/test_measurement_helpers.py` | `1bb09008121569fd851f2a77a9029ecfc69926951f84dea02a8493d0e0f89ec3` |

## 6. Things to know

- **The full notebook's hash changed.** I rebuilt `PCG_Baseline_Reproduction_Measurements.ipynb` from its sources so its embedded helper matches the revised file (its earlier hash was `a8be2ede...`). The byte-identical copy I put in `Downloads` earlier still has the old hash and old helper; I did not touch it. Use the `docs/phase3/measurements` version, or the new `v2` trimmed notebook.
- **`v2` is built from the notebook you gave me**, not from my sources; the two cells I changed lost their saved outputs. Your original trimmed notebook is untouched.
- **To also time an `inference_mode` pipeline**, a separate, validated variant would be needed (section 1); I did not build it because the baseline is already under `no_grad` and a naive swap breaks.
- **Not verified:** CUDA execution of the revised helper; Colab paths; GPU timing behaviour.

## 7. Recommended next step
Open `PCG_Baseline_Reproduction_Measurements_Trimmed_v2.ipynb` in Colab, run from the top, confirm the printed line "Gradient scope of the baseline sample(): all 6 net(...) calls ... torch.no_grad()", then set `RUN_INFERENCE_BENCHMARK = True`. After the run, check `gradient_tracking.conclusion` and `model_train_eval_state.unchanged_during_benchmark` in `inference_benchmark_summary.json`.
