"""Local validation harness: executes the APPENDED notebook cells against a fake Colab-style directory tree built from the local repository.

Not used by the notebook itself. It maps '/content' to a temporary root, copies (never moves) the archived experiment files into the layout the
notebook expects, and runs the appended cells in order in one namespace. Flags can be overridden. CUDA is not needed for the cells that run with
the default flags; the benchmark cell can run on CPU for a handful of trajectories (validation only; such numbers are never results).
"""

import json
import os
import shutil
import sys
from typing import Any, Dict, Optional

REPO_FILES = ("train.py", "test.py", "visualization.py", "requirements.txt")
REPO_DIRS = ("utils", "model", "Data")


def build_fake_content(repo: str, root: str) -> Dict[str, str]:
    """root/content/... mimicking Colab. Everything is COPIED from the local repo (read-only)."""
    exp = os.path.join(repo, "experiments", "PCG_T4_5_Epoch_Run")
    c = os.path.join(root, "content")
    r = os.path.join(c, "PolarCollisionGrid-PedestrianTrajectoryPrediction")
    os.makedirs(r)
    for f in REPO_FILES:
        shutil.copy2(os.path.join(repo, f), os.path.join(r, f))
    for d in REPO_DIRS:
        shutil.copytree(os.path.join(repo, d), os.path.join(r, d), ignore=shutil.ignore_patterns("__pycache__"))
    model = os.path.join(r, "Store_Results", "model")
    for sub in ("CollisionGrid",):
        os.makedirs(os.path.join(model, sub))
    os.makedirs(os.path.join(r, "Store_Results", "plot", "test"))
    os.makedirs(os.path.join(r, "Store_Results", "plot", "train"))
    os.makedirs(os.path.join(r, "Store_Results", "log"))
    drive = os.path.join(c, "drive", "MyDrive", "PCG_Experiments", "PCG_T4_5_Epoch_Run")
    os.makedirs(drive)
    for e in range(5):
        src = os.path.join(exp, "checkpoints", "SOCIALLSTM_lstm_model_%d.tar" % e)
        shutil.copy2(src, os.path.join(model, "SOCIALLSTM_lstm_model_%d.tar" % e))
        shutil.copy2(src, os.path.join(drive, "SOCIALLSTM_lstm_model_%d.tar" % e))
    shutil.copy2(os.path.join(exp, "configs", "experiment_config.pkl"), os.path.join(model, "config.pkl"))
    shutil.copy2(os.path.join(exp, "configs", "experiment_config.pkl"), os.path.join(drive, "config.pkl"))
    shutil.copy2(os.path.join(exp, "configs", "experiment_config.pkl"), os.path.join(model, "CollisionGrid", "config.pkl"))       # staged copy, as in the notebook
    shutil.copy2(os.path.join(exp, "configs", "original_config.pkl"), os.path.join(drive, "CollisionGrid_original_config.pkl"))
    ck181 = os.path.join(repo, "Store_Results", "model", "CollisionGrid", "SOCIALLSTM_lstm_model_181.tar")
    shutil.copy2(ck181, os.path.join(model, "CollisionGrid", "SOCIALLSTM_lstm_model_181.tar"))
    shutil.copy2(ck181, os.path.join(drive, "SOCIALLSTM_lstm_model_181.tar"))
    shutil.copy2(os.path.join(exp, "checkpoints", "SOCIALLSTM_lstm_model_4.tar"), os.path.join(model, "CollisionGrid", "SOCIALLSTM_lstm_model_4.tar"))
    shutil.copy2(os.path.join(exp, "results", "test_results_epoch4.pkl"), os.path.join(r, "Store_Results", "plot", "test", "test_results.pkl"))
    shutil.copy2(os.path.join(exp, "results", "test_results_epoch4.pkl"), os.path.join(drive, "test_results_epoch4.pkl"))
    shutil.copy2(os.path.join(exp, "logs", "log_curve.txt"), os.path.join(r, "Store_Results", "log", "log_curve.txt"))
    shutil.copy2(os.path.join(exp, "logs", "loss_plot_batch.png"), os.path.join(r, "Store_Results", "plot", "train", "loss_plot_batch.png"))
    return {"content": c, "repo": r, "drive": drive}


def run_cells(cell_sources, content_root: str, overrides: Optional[Dict[str, Any]] = None, initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute cell sources (list of str) in order in ONE namespace. '/content' is rewritten to ``content_root``."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    ns: Dict[str, Any] = {"__name__": "__notebook__"}
    if initial:
        ns.update(initial)
    for i, src in enumerate(cell_sources):
        code = src.replace("/content", content_root.replace("\\", "/"))
        if i == 0 and overrides:                       # after the setup cell defines the defaults, override the switches
            pass
        exec(compile(code, "<cell %d>" % i, "exec"), ns)
        if i == 0 and overrides:
            ns.update(overrides)
    return ns


def code_cells_of(notebook_path: str, skip_original: int = 64):
    nb = json.load(open(notebook_path, encoding="utf-8"))
    return [("".join(c["source"])) for c in nb["cells"][skip_original:] if c["cell_type"] == "code"]
