"""Build PCG_Baseline_Reproduction_Measurements.ipynb from the ORIGINAL notebook plus the appended measurement cells.

The original notebook is only READ. Its 64 cells (with outputs) are copied unchanged; new cells are appended after them. The output
file is created exclusively (an existing file is never overwritten). Retrospective data are extracted programmatically from the saved
cell outputs of the original notebook (cells 45, 49, 51, 54, 60) with `measurement_lib`'s parsers; nothing is typed in by hand.

Usage:  python build_measurement_notebook.py --original <orig.ipynb> --out <new.ipynb> [--archived-log <log_curve.txt>]
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))
import measurement_lib as ml  # noqa: E402

CELL_FILES = ["c00_intro.md", "c01_setup.py", "c02_fingerprint.py", "c03_manifest.py", "c04_retrospective.py", "c05_training_runner.py",
              "c06_inference_benchmark.py", "c07_plots.py", "c08_summary.py"]
HELPERS = ["measurement_lib.py", "monitored_train_runner.py", "inference_benchmark.py"]


def out_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain") or ""
        parts.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(parts)


def source_text(cell):
    return "".join(cell["source"])


def extract_retro(nb, nb_sha, archived_log_path):
    cells = nb["cells"]
    must = {45: "--num_epochs 1", 49: "--num_epochs 5", 51: "!python train.py --method 4 --num_epochs 5", 54: "CHECKPOINT FILES", 60: "test.py --method 4 --epoch 4 --sample_size 5"}
    for i, needle in must.items():
        if needle not in source_text(cells[i]):
            raise SystemExit("cell %d does not look like the expected cell (missing %r); aborting" % (i, needle))

    def run(i, key_cmd, evidence):
        t = out_text(cells[i])
        p = ml.parse_train_log(t)
        return {"cell_index": i, "execution_count": cells[i].get("execution_count"), "command": key_cmd, "interpreter_evidence": evidence,
                "grid_calculation_s": p["grid_calculation_s"], "total_training_s": p["total_training_s"],
                "batches": [[b["global_batch"], b["epoch"], b["train_loss"], b["time_per_batch_s"]] for b in p["batches"]]}

    runs = {
        "one_epoch_cell45_pcg_env": run(45, "! /content/pcg-env/bin/python train.py --method 4 --num_epochs 1",
                                        "explicit /content/pcg-env/bin/python in the command text (Python 3.11.18, PyTorch 2.2.1+cu121 per cells 19, 21, 28)"),
        "five_epoch_cell49_pcg_env": run(49, "!MPLBACKEND=Agg /content/pcg-env/bin/python train.py --method 4 --num_epochs 5",
                                         "explicit /content/pcg-env/bin/python in the command text. The FIRST command of this cell was malformed ('5%cd ...') and exited with an "
                                         "argparse usage error; the SECOND command of the same cell ran a complete 5-epoch training"),
        "five_epoch_cell51_default_python": run(51, "!python train.py --method 4 --num_epochs 5",
                                                "plain `!python`: resolved through the shell PATH, which the notebook does not record. The kernel itself was /usr/bin/python3 "
                                                "3.13.16 with PyTorch 2.14.1+cu130 (cells 0, 11). Grid time 13.4 s vs about 31 s in both pcg-env runs is consistent with a "
                                                "different environment but does not prove it"),
    }
    for k, r in runs.items():
        if not r["batches"] or r["total_training_s"] is None:
            raise SystemExit("could not parse %s" % k)
    if len(runs["five_epoch_cell49_pcg_env"]["batches"]) != 1190 or len(runs["five_epoch_cell51_default_python"]["batches"]) != 1190:
        raise SystemExit("expected 1190 batches in each 5-epoch run")
    # archived epoch losses
    arch = []
    with open(archived_log_path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"Training epoch:\s*(\d+)\s+loss:\s*(-?[\d.]+)", line)
            if m:
                arch.append(float(m.group(2)))
    # checkpoint mtimes (first folder listing of cell 54)
    t54 = out_text(cells[54])
    first_block = t54.split("Folder:")[1]
    mt = []
    for e in range(5):
        m = re.search(r"SOCIALLSTM_lstm_model_%d\.tar \| [\d.]+ MB \| (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)" % e, first_block)
        mt.append(m.group(1))
    # test pass
    t60 = out_text(cells[60])
    tp = ml.parse_test_log(t60)["trajectories"]
    if len(tp) != 1081:
        raise SystemExit("expected 1081 trajectories in the test output")
    final = {}
    for key, pat in (("ADE", r"^ADE:\s+(-?[\d.]+)"), ("FDE", r"^FDE:\s+(-?[\d.]+)"), ("MHD", r"^MHD:\s+(-?[\d.]+)"), ("speed_error", r"^Speed error:\s+(-?[\d.]+)"),
                     ("heading_error", r"^Heading error:\s+(-?[\d.]+)"), ("collision_pct", r"^Collision percentage:\s+(-?[\d.]+)"),
                     ("collision_pct_ped_ped", r"^Collision percentage \(ped-ped\):\s+(-?[\d.]+)"), ("collision_pct_ped_veh", r"^Collision percentage \(ped-veh\):\s+(-?[\d.]+)")):
        m = re.search(pat, t60, re.M)
        final[key] = float(m.group(1)) if m else None
    iter_lines = {"speed_error_iteration_print": re.search(r"speed error of the model is\s+tensor\(([\d.]+)\)", t60).group(1),
                  "heading_error_iteration_print": re.search(r"final heading error of the model is\s+tensor\(([\d.]+)\)", t60).group(1)}
    t57 = out_text(cells[57])
    protocol = {"sample_size": 5, "selection": "best of five by lowest ADE computed against ground truth (ORACLE selection)", "model_mode": "net.eval() is never called in test.py; the model contains nn.Dropout(0.5), so dropout is active in the prediction path",
                "eval_called_in_printed_test_py": (".eval(" in t57), "future_information": "baseline sample() receives ground-truth sequences and future actor lists (Phase 2 Task 4 audit)",
                "checkpoint": "epoch 4 of the cell-51 run, staged in Store_Results/model/CollisionGrid (cell 59)"}
    if ".eval(" in t57:
        raise SystemExit("printed test.py calls .eval(); the dropout statement in the protocol would be wrong - review")
    retro = {
        "source_notebook_sha256": nb_sha, "extraction": "programmatic parse of saved cell outputs (cells 45, 49, 51, 54, 57, 60)", "batch_size": 10,
        "training_runs": runs, "archived_epoch_losses_log_curve": arch,
        "archived_epoch_losses_source": "experiments/PCG_T4_5_Epoch_Run/logs/log_curve.txt of the project repository (SHA-256 recorded in its archived manifest)",
        "checkpoint_mtimes_cell54": {"epoch_0_to_4": mt, "source": "cell 54 output, Store_Results/model checkpoint listing (Colab local time as printed)"},
        "test_pass": {"command": "!python test.py --method 4 --epoch 4 --sample_size 5", "interpreter_evidence": "plain `!python` (see the cell-51 note; not recorded)",
                      "per_trajectory": [[r["trajectory"], r["time_s"]] for r in tp], "reported_metrics": final, "protocol": protocol,
                      "metric_note": "cell 60 prints speed error %s and heading error %s in the per-iteration lines but %s and %s in the final summary; the archived/reported values are the final-summary ones. The difference is recorded, not reconciled." % (
                          iter_lines["speed_error_iteration_print"], iter_lines["heading_error_iteration_print"], final["speed_error"], final["heading_error"])},
        "environment_findings": [
            "Two Python environments exist: the Colab default kernel (Python 3.13.16, PyTorch 2.14.1+cu130; cells 0, 11) and /content/pcg-env (Python 3.11.18, PyTorch 2.2.1+cu121; cells 19, 21, 28). Installing torch 2.2.1 into the default environment failed (cell 10).",
            "TWO complete 5-epoch trainings are in the saved outputs: cell 49 (execution 54, explicit pcg-env) and cell 51 (execution 56, plain !python). Both write the same Store_Results file names, so the later run overwrote the earlier run's checkpoints and log. The archived log_curve.txt matches the cell-51 run, so the archived checkpoints come from the default-interpreter run, not from pcg-env.",
            "The first command in cell 49 was malformed ('--num_epochs 5%cd ...') and failed with an argparse usage error; only its second command trained.",
            "Cell 41 claims 'DataLoader works inside Python 3.11!' but its shell command failed (unmatched quote) and the remaining lines ran in the KERNEL (Python 3.13.16, PyTorch 2.14.1+cu130): that check does not show anything about pcg-env.",
            "The 1-epoch run (cell 45) used pcg-env; the 5-epoch archived run and the test (cell 60) used plain !python, whose interpreter is not recorded. The grid-calculation time is 31.9 s (cell 45) and 30.6 s (cell 49) in pcg-env versus 13.4 s in cell 51.",
            "The two 5-epoch runs are not identical (unseeded): first-batch loss 2.029 vs 1.917 and different epoch mean losses, so they are two independent samples of the same procedure.",
            "Store_Results was replaced by a symlink to Drive (cell 47), so training wrote its checkpoints through the Drive mount."],
        "not_reconstructable": [
            "GPU utilisation, GPU memory (allocated, reserved, peak), CPU utilisation, RAM and process memory during training or testing: never recorded (only two idle nvidia-smi snapshots exist, cells 1 and 53)",
            "per-epoch wall-clock time (only an estimate from checkpoint mtimes and from summed batch times)",
            "data-loading time per epoch and checkpoint-save time",
            "CUDA-synchronised per-batch time (the printed time/batch comes from time.time() around the batch, which includes several host-device synchronisations but was not an explicit measurement)",
            "single-sample inference latency, model-only latency, p50/p90/p95/p99 of one prediction, inference throughput and inference peak memory (the historical test time covers 5 samples + scoring + oracle selection)",
            "the interpreter actually used by plain `!python` in cells 51 and 60",
            "checkpoints, config and log of the overwritten cell-49 run (only its printed output survives)",
            "any random seed (none was set; runs are not reproducible)"]}
    return retro


def build(original, out, archived_log):
    raw = open(original, "rb").read()
    nb_sha = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw.decode("utf-8"))
    n_orig = len(nb["cells"])
    retro = extract_retro(nb, nb_sha, archived_log)
    helper_src = {}
    for h in HELPERS:
        s = open(os.path.join(HERE, "helpers", h), encoding="utf-8", newline="").read().replace("\r\n", "\n")
        if "'''" in s or s.rstrip().endswith("\\"):
            raise SystemExit("helper %s cannot be embedded in a raw triple-quoted string" % h)
        helper_src[h] = s
    helpers_literal = "{\n" + "".join("    %s: r'''%s''',\n" % (json.dumps(k), v) for k, v in helper_src.items()) + "}"
    retro_json = json.dumps(retro, separators=(",", ":"), allow_nan=False)
    if "'''" in retro_json:
        raise SystemExit("retro json contains a triple quote")
    new_cells = []
    for fn in CELL_FILES:
        text = open(os.path.join(HERE, "cells", fn), encoding="utf-8", newline="").read().replace("\r\n", "\n")
        text = text.replace("@@HELPERS@@", helpers_literal).replace("@@RETRO@@", retro_json)
        lines = text.splitlines(keepends=True)
        if fn.endswith(".md"):
            new_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})
    out_nb = copy.deepcopy(nb)
    out_nb["cells"] = out_nb["cells"] + new_cells
    text = json.dumps(out_nb, indent=2, ensure_ascii=False) + "\n"
    with open(out, "x", encoding="utf-8", newline="\n") as fh:      # exclusive: never overwrites
        fh.write(text)
    # verification: original untouched, first n cells identical, new cells appended only
    assert hashlib.sha256(open(original, "rb").read()).hexdigest() == nb_sha, "original notebook changed"
    chk = json.loads(open(out, encoding="utf-8").read())
    assert chk["cells"][:n_orig] == nb["cells"], "original cells differ in the copy"
    assert len(chk["cells"]) == n_orig + len(CELL_FILES)
    assert chk["metadata"] == nb["metadata"] and chk["nbformat"] == nb["nbformat"] and chk["nbformat_minor"] == nb["nbformat_minor"]
    return {"original_sha256": nb_sha, "output": out, "output_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "original_cells": n_orig, "new_cells": len(CELL_FILES),
            "retro_json_bytes": len(retro_json)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archived-log", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.original, a.out, a.archived_log), indent=2))
