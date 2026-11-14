# Measurement enhancement (appended cells)

Everything **above** this cell is the unchanged historical notebook (cells and saved outputs). Everything **below** was appended to add research-grade measurement instrumentation. Nothing below re-runs historical training, overwrites a checkpoint, configuration or prediction file, or modifies the PCG source.

**Safe by default.** Running these cells top to bottom only *reads* files and writes new files into a fresh directory (`MEAS_DIR`). Two optional cells can start real work, and both are OFF:

| Flag | Starts | Needs |
|---|---|---|
| `RUN_MONITORED_TRAINING` | a NEW isolated training run with resource monitoring (Part D) | the flag, the exact run id in `CONFIRM_RUN_ID`, GPU runtime, ~13 min for 5 epochs |
| `RUN_INFERENCE_BENCHMARK` | single-sample predictor timing of the epoch-4 checkpoint (Part E) | the flag, GPU runtime |

**Interpreter warning.** The historical notebook used two interpreters: the Colab default (Python 3.13.16, PyTorch 2.14.1+cu130) and `/content/pcg-env` (Python 3.11, PyTorch 2.2.1+cu121, the version the README asks for). Two complete 5-epoch trainings exist in the saved outputs (the cell with `/content/pcg-env/bin/python` and the later cell with plain `!python`); the archived checkpoints correspond to the **later** one. Every new run records the exact interpreter it actually used. Set `PYTHON_FOR_NEW_RUNS` explicitly.

**Labels used throughout.** *Retrospective* = extracted from outputs saved in this notebook (no new execution). *New run* = collected by the instruments below. Historical test metrics use `sample_size=5` with best-of-five **oracle** selection (ground truth picks the best sample) and are not single-sample results; the baseline test path also runs with dropout active (`net.eval()` is never called) and uses future actor information. Trajectory-only timing is **not** video or real-time performance.
