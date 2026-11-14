"""Build PCG_Baseline_Reproduction_Measurements_Trimmed_v2.ipynb from the user's trimmed notebook.

The trimmed notebook is only READ. The output differs from it in exactly three ways (checked and printed):
  1. one NEW markdown revision-note cell is inserted after the title cell;
  2. cell "Measurement setup" (embedded helpers): the embedded ``inference_benchmark.py`` is replaced by the current helper file
     (gradient-tracking record, train/eval state record); ``measurement_lib.py`` is unchanged; its saved output is cleared because it is stale;
  3. the benchmark control cell gets gradient-tracking prints and a static check of the baseline ``sample()``; its saved output is cleared.
Every other cell, including the saved Colab outputs of the artifact manifest and the retrospective report, is copied unchanged. The output
file is created exclusively (never overwrites).

Usage: python build_trimmed_v2.py --trimmed <trimmed.ipynb> --out <v2.ipynb>
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import pprint
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "helpers", "inference_benchmark.py")
LIB = os.path.join(HERE, "helpers", "measurement_lib.py")
GRAD_MARK = "Gradient scope of the baseline sample()"

NOTE = """## Revision note (gradient-tracking review of the benchmark)

This notebook is the trimmed notebook with a revised embedded `inference_benchmark.py` (v1.1). **Run it from the top**: the saved outputs of the setup and benchmark-control cells were cleared because they describe the previous helper; the other saved outputs (artifact manifest, retrospective report) are the earlier session's and are replaced when you re-run.

What was checked: the baseline `sample()` in `test.py` already runs every `net(...)` call, and its return, under `torch.no_grad()`; it does not use `torch.inference_mode()`. So **no extra `no_grad` / `inference_mode` context is added** and the timed code is exactly the baseline execution.

What changed: the benchmark now records, in `inference_benchmark_summary.json` under `gradient_tracking`, a static check of `sample()`, the grad mode observed inside every model forward and in the stages outside `sample()`, whether any tensor required grad, parameter `requires_grad` counts, and under `model_train_eval_state` the model's train/eval/dropout state at load, before timing and after the run. The benchmark never calls `net.eval()` and aborts if the state changes during the run.
"""


def src_text(c):
    return "".join(c["source"])


def split_lines(t):
    return t.splitlines(keepends=True)


def replace_embedded_helper(cell_src, name, new_text):
    start = cell_src.index("HELPER_SOURCES = ") + len("HELPER_SOURCES = ")
    end = cell_src.index("\n\nHELPER_SHA256", start)
    d = ast.literal_eval(cell_src[start:end])
    if list(d) != ["measurement_lib.py", "inference_benchmark.py"]:
        raise SystemExit("unexpected embedded helper set: %s" % list(d))
    if pprint.pformat(d, width=80, sort_dicts=False) != cell_src[start:end]:
        # the literal is not in pprint style; the replacement below still produces a valid literal
        pass
    lib_now = open(LIB, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    if d["measurement_lib.py"] != lib_now:
        raise SystemExit("embedded measurement_lib.py differs from the helper file; refusing to continue")
    d = dict(d)
    d[name] = new_text
    lit = pprint.pformat(d, width=80, sort_dicts=False)
    assert ast.literal_eval(lit) == d
    return cell_src[:start] + lit + cell_src[end:]


def edit_benchmark_cell(s):
    """Idempotent text edits shared by the trimmed benchmark cell and the full notebook's benchmark cell source."""
    if GRAD_MARK in s:
        return s

    def rep(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise SystemExit("anchor not found exactly once: %r (%d)" % (old[:70], s.count(old)))
        s = s.replace(old, new, 1)

    rep("# `sample` function is imported from it unchanged and its per-trajectory preparation steps are replicated in the benchmark module.\n",
        "# `sample` function is imported from it unchanged and its per-trajectory preparation steps are replicated in the benchmark module.\n"
        "# Gradient tracking: the baseline sample() already runs under torch.no_grad(); the benchmark adds NO other context, never calls net.eval(),\n"
        "# and records the observed grad mode and the model's train/eval state in the summary (gradient_tracking, model_train_eval_state).\n")
    rep('print("config     :", BENCH_CONFIG)\n',
        'print("config     :", BENCH_CONFIG)\n'
        '_scope = bench_mod.sample_grad_scope(os.path.join(REPO, "test.py"))\n'
        'print("%s:" % "Gradient scope of the baseline sample()", _scope["verdict"])\n'
        'if not (_scope.get("uses_no_grad") and _scope.get("net_calls_inside_guard") == _scope.get("net_calls_total")):\n'
        '    print("   WARNING: the baseline sample() is not fully under torch.no_grad(); the benchmark will record gradient tracking as ENABLED where it is.")\n')
    rep('    print("copied to:", BENCH_OUTPUT_DIR)\n',
        '    _G, _S = BENCH_SUMMARY["gradient_tracking"], BENCH_SUMMARY["model_train_eval_state"]\n'
        '    print("gradient tracking:", _G["conclusion"])\n'
        '    print("model state after run: net.training =", _S["after_all_passes"]["net_training"], "| dropout modules in training mode =",\n'
        '          _S["after_all_passes"]["dropout_modules_in_training_mode"], "| unchanged during benchmark =", _S["unchanged_during_benchmark"],\n'
        '          "| eval called by benchmark =", _S["eval_called_by_benchmark"])\n'
        '    print("copied to:", BENCH_OUTPUT_DIR)\n')
    return s


def build(trimmed, out):
    raw = open(trimmed, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw.decode("utf-8"))
    new_helper = open(HELPER, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    res = copy.deepcopy(nb)
    cells = res["cells"]
    i_setup = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "HELPER_SOURCES = " in src_text(c))
    i_bench = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "bench_mod.run_benchmark(" in src_text(c))
    for i, newsrc in ((i_setup, replace_embedded_helper(src_text(cells[i_setup]), "inference_benchmark.py", new_helper)),
                      (i_bench, edit_benchmark_cell(src_text(cells[i_bench])))):
        cells[i]["source"] = split_lines(newsrc)
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None
    cells.insert(1, {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": split_lines(NOTE)})
    text = json.dumps(res, indent=1, ensure_ascii=False) + "\n"
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    # verification
    chk = json.loads(open(out, encoding="utf-8").read())
    assert hashlib.sha256(open(trimmed, "rb").read()).hexdigest() == sha, "the trimmed notebook changed"
    changed = []
    for k, c in enumerate(chk["cells"]):
        if k == 1:
            continue
        o = nb["cells"][k if k == 0 else k - 1]
        if c != o:
            changed.append(k)
    assert changed == [i_setup + 1, i_bench + 1], changed
    assert chk["metadata"] == nb["metadata"] and (chk["nbformat"], chk["nbformat_minor"]) == (nb["nbformat"], nb["nbformat_minor"])
    return {"trimmed_sha256": sha, "output": out, "output_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "cells": len(chk["cells"]),
            "changed_cell_indices_in_output": changed, "inserted_note_index": 1}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trimmed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.trimmed, a.out), indent=2))
