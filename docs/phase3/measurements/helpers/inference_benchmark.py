"""Single-sample PCG trajectory-predictor inference benchmark (notebook companion; test.py is NOT modified or run).

SCOPE (written into every output): this measures the PCG trajectory-PREDICTOR code path only, one prediction sample per input
(``sample_size = 1``). It is NOT end-to-end video performance (no detection, tracking or video decoding) and it is NOT a
deployment-valid latency: the baseline ``sample`` function and the preparation steps still consume the future actor information
that the historical test path uses (see the Phase 2 Task 4 audit). The numbers characterise this code path on the declared hardware.

Relation to ``test.py`` (read at the commit recorded in the output): the per-trajectory steps are REPLICATED here, step for step,
from ``test.py`` ("dense vector creation" ... "revert position change"), and the baseline ``sample`` function is imported from
``test.py`` unchanged. Left out on purpose: the ground-truth scoring (ADE/FDE/MHD/speed/heading/collision), the best-of-N
selection (best-of-N needs ground truth: that is the oracle) and the result list. Because ``test.py`` never calls ``net.eval()``,
this benchmark does not either unless ``eval_mode=True`` is requested; the model contains ``nn.Dropout`` so the baseline
prediction path runs with dropout ACTIVE. Both facts are recorded.

Gradient tracking (v1.1): the baseline ``sample`` function already runs EVERYTHING after its setup lines, including every ``net(...)``
call and its return, under ``torch.no_grad()`` (checked statically from test.py's AST by ``sample_grad_scope`` and observed at run
time). It does not use ``torch.inference_mode()``. The benchmark adds NO extra ``no_grad`` / ``inference_mode`` context and never
changes the model's train/eval state, so what is timed is exactly the baseline execution. The stages outside ``sample`` (preparation and
post-processing, as in test.py) run with grad mode enabled but create no autograd graph because no tensor in them requires grad. Every
output records the observed grad mode in ``gradient_tracking`` and the model's train/eval state before and after the run.

Timing method: ``torch.cuda.synchronize()`` before and after every timed stage when CUDA is used; ``time.perf_counter``.
Two separate passes over the same trajectories with the same seed:
  pass 1 "pipeline": stages prep (CPU: dense arrays + interaction grids + position change), transfer (to GPU), rollout (the baseline
                     ``sample`` call: model forwards + Gaussian sampling + per-step grid work), post (revert position change);
                     pipeline latency = the whole per-trajectory span;
  pass 2 "model_only": forward hooks on the network time every ``net(...)`` call (synchronised) and sum them per trajectory.
Warm-up trajectories are processed first in each pass and excluded from all statistics.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

BENCH_VERSION = "pcg-inference-benchmark-1.1"
SCOPE = ("PCG trajectory-predictor-only, single-sample (sample_size=1), baseline code path. NOT end-to-end video performance. NOT deployment-valid: "
         "the baseline path still uses future actor information as in the historical test. Ground-truth scoring and oracle selection are excluded. "
         "Gradient tracking: see the gradient_tracking record (baseline sample() runs under torch.no_grad(); the benchmark adds no other context).")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct(v: Sequence[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s), max(1, int(math.ceil(q / 100.0 * len(s))))) - 1]


def describe(values: Sequence[float]) -> Dict[str, Any]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return {"count": 0}
    m = sum(v) / len(v)
    return {"count": len(v), "mean_s": m, "std_s": math.sqrt(sum((x - m) ** 2 for x in v) / len(v)), "min_s": min(v), "p50_s": _pct(v, 50), "p90_s": _pct(v, 90),
            "p95_s": _pct(v, 95), "p99_s": _pct(v, 99), "max_s": max(v), "percentile_method": "nearest_rank"}


def preflight(repo: str, workdir: str, checkpoint: str, config: str, required: Sequence[str] = ("test.py", "utils/DataLoader.py", "Data/test/trajectories_test.cpkl")) -> List[str]:
    problems = []
    for rel in required:
        if not os.path.isfile(os.path.join(repo, rel)):
            problems.append("repository file missing: %s" % rel)
    for label, p in (("checkpoint", checkpoint), ("config", config)):
        if not os.path.isfile(p):
            problems.append("%s file missing: %s" % (label, p))
    rr, wr = os.path.realpath(repo), os.path.realpath(workdir)
    if wr == rr or wr.startswith(rr + os.sep):
        problems.append("work directory must be OUTSIDE the repository: %s" % wr)
    if os.path.exists(workdir) and (not os.path.isdir(workdir) or os.listdir(workdir)):
        problems.append("work directory exists and is not an empty directory: %s" % workdir)
    return problems


_NO_GRAD_CONTEXTS = ("torch.no_grad()", "torch.set_grad_enabled(False)")


def sample_grad_scope(test_py_path: str) -> Dict[str, Any]:
    """Static (AST) check of the baseline ``sample`` function in test.py: which model calls run under a no-grad guard.

    Reads the file only. Returns a plain dict that is stored in every benchmark summary."""
    import ast
    tree = ast.parse(open(test_py_path, encoding="utf-8").read())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample"), None)
    if fn is None:
        return {"function_found": False, "verdict": "sample() was not found in test.py"}
    guards = []
    for w in ast.walk(fn):
        if isinstance(w, ast.With):
            for item in w.items:
                ctx = ast.unparse(item.context_expr)
                if ctx in _NO_GRAD_CONTEXTS or ctx.startswith("torch.inference_mode("):
                    guards.append({"context": ctx, "lines": [w.lineno, w.end_lineno]})
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "net"]
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]

    def inside(node):
        return any(g["lines"][0] <= node.lineno <= g["lines"][1] for g in guards)
    n_in = sum(1 for c in calls if inside(c))
    out = {"function_found": True, "function_lines": [fn.lineno, fn.end_lineno], "guards": guards, "net_calls_total": len(calls), "net_calls_inside_guard": n_in,
           "returns_total": len(returns), "returns_inside_guard": sum(1 for r in returns if inside(r)),
           "uses_inference_mode": any(g["context"].startswith("torch.inference_mode(") for g in guards), "uses_no_grad": any(g["context"] in _NO_GRAD_CONTEXTS for g in guards)}
    if guards and calls and n_in == len(calls):
        out["verdict"] = "all %d net(...) calls in sample() run under %s" % (len(calls), "/".join(sorted({g["context"] for g in guards})))
    else:
        out["verdict"] = "NOT all model calls in sample() are under a no-grad guard (%d of %d)" % (n_in, len(calls))
    return out


def _gpu_driver() -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=5).stdout.strip().splitlines()[0]
    except Exception:
        return None


def run_benchmark(repo: str, workdir: str, run_id: str, checkpoint: str, config: str, *, epoch: int, method: int = 4, warmup: int = 20,
                  max_trajectories: Optional[int] = None, seed: int = 12345, use_cuda: Optional[bool] = None, eval_mode: bool = False,
                  expected_checkpoint_sha256: Optional[str] = None, expected_config_sha256: Optional[str] = None,
                  hardware_label: Optional[str] = None) -> Dict[str, Any]:
    problems = preflight(repo, workdir, checkpoint, config)
    if problems:
        raise RuntimeError("benchmark NOT started:\n  - " + "\n  - ".join(problems))
    repo, workdir = os.path.realpath(repo), os.path.realpath(workdir)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, "measurements")
    os.makedirs(out)
    model_dir = os.path.join(workdir, "Store_Results", "model", "CollisionGrid" if method == 4 else ("SocialLSTM" if method == 1 else "VanillaLSTM"))
    os.makedirs(model_dir)
    ck_dst = os.path.join(model_dir, "SOCIALLSTM_lstm_model_%d.tar" % epoch)
    cf_dst = os.path.join(model_dir, "config.pkl")
    shutil.copy2(checkpoint, ck_dst)                                  # COPIES: the originals are only read
    shutil.copy2(config, cf_dst)
    shutil.copytree(os.path.join(repo, "Data"), os.path.join(workdir, "Data"))
    hashes = {"checkpoint_sha256": _sha(checkpoint), "checkpoint_copy_sha256": _sha(ck_dst), "config_sha256": _sha(config), "config_copy_sha256": _sha(cf_dst),
              "test_py_sha256": _sha(os.path.join(repo, "test.py")), "test_data_sha256": _sha(os.path.join(repo, "Data", "test", "trajectories_test.cpkl"))}
    if hashes["checkpoint_sha256"] != hashes["checkpoint_copy_sha256"] or hashes["config_sha256"] != hashes["config_copy_sha256"]:
        raise RuntimeError("copied checkpoint/config hashes differ from the originals; aborting")
    hashes["checkpoint_matches_expected"] = None if not expected_checkpoint_sha256 else hashes["checkpoint_sha256"].lower() == expected_checkpoint_sha256.lower()
    hashes["config_matches_expected"] = None if not expected_config_sha256 else hashes["config_sha256"].lower() == expected_config_sha256.lower()

    old_cwd = os.getcwd()
    sys.path.insert(0, repo)
    try:
        os.chdir(workdir)
        import numpy as np
        import torch
        cuda = torch.cuda.is_available() if use_cuda is None else bool(use_cuda)
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("use_cuda requested but CUDA is not available")
        spec = importlib.util.spec_from_file_location("pcg_repo_test_module", os.path.join(repo, "test.py"))     # not `import test` (stdlib name clash)
        tmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tmod)                                  # defines main()/sample(); nothing runs
        from utils.DataLoader import DataLoader
        from utils.helper import get_model, position_change_seq, revert_postion_change_seq2
        from utils.Interaction import getSequenceInteractionGridMask
        import argparse as _ap

        def seed_all():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed_all(seed)

        def sync():
            if cuda:
                torch.cuda.synchronize()

        with open(cf_dst, "rb") as fh:
            saved_args = pickle.load(fh)
        cpu_override = False
        if not cuda and getattr(saved_args, "use_cuda", False):
            saved_args.use_cuda = False             # in-memory only (the model calls .cuda() when this is True); the config FILE is not changed
            cpu_override = True
        sample_args = _ap.Namespace(obs_length=6, pred_length=6, use_cuda=cuda, method=method, epoch=epoch, sample_size=1)
        t = time.perf_counter()
        dataloader = DataLoader(1, sample_args.obs_length + sample_args.pred_length, infer=True, filtering=True)
        dataloader.reset_batch_pointer()
        net = get_model(method, saved_args, True)
        if cuda:
            net = net.cuda()
        try:
            ck = torch.load(ck_dst, map_location=("cuda" if cuda else "cpu"), weights_only=False)
        except TypeError:
            ck = torch.load(ck_dst, map_location=("cuda" if cuda else "cpu"))
        net.load_state_dict(ck["state_dict"])

        def model_state() -> Dict[str, Any]:
            d = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
            return {"net_training": bool(net.training), "dropout_modules": len(d), "dropout_modules_in_training_mode": sum(1 for m in d if m.training),
                    "dropout_probabilities": sorted({float(m.p) for m in d})}
        state_at_load = model_state()                                  # what get_model + load_state_dict left (test.py never changes it)
        if eval_mode:
            net.eval()                                                  # only on explicit request; recorded below
        state_before_timing = model_state()
        scope = sample_grad_scope(os.path.join(repo, "test.py"))
        setup_s = time.perf_counter() - t
        t = time.perf_counter()
        whole = dataloader.batch_creater(False, method, suffle=False)
        batch_creation_s = time.perf_counter() - t
        x_W, nP_W, P_W, xv_W, nV_W, V_W = whole[0], whole[1], whole[2], whole[3], whole[4], whole[5]
        n_total = dataloader.num_batches
        n_use = n_total if max_trajectories is None else min(n_total, max_trajectories)
        if n_use <= warmup:
            raise RuntimeError("need more trajectories (%d) than warm-up iterations (%d)" % (n_use, warmup))
        obs = sample_args.obs_length
        timers: Dict[str, float] = {"fwd": 0.0, "calls": 0}
        gobs: Dict[str, Any] = {"forward": set(), "prep": set(), "post": set(), "prep_tensors_require_grad": False, "output_tensors_require_grad": False,
                                "output_has_grad_fn": False}

        def pre_hook(mod, inp):
            gobs["forward"].add((bool(torch.is_grad_enabled()), bool(torch.is_inference_mode_enabled())))   # (grad_enabled, inference_mode) INSIDE the forward call
            sync()
            timers["t"] = time.perf_counter()

        def post_hook(mod, inp, outp):
            sync()
            timers["fwd"] += time.perf_counter() - timers["t"]
            timers["calls"] += 1

        def one(b: int, stage_times: Optional[Dict[str, float]]) -> Dict[str, Any]:
            x, nP, P = x_W[b], nP_W[b], P_W[b]
            xv, nV, V = xv_W[b], nV_W[b], V_W[b]
            x_seq, numPedsList_seq, PedsList_seq = x[0], nP[0], P[0]
            x_seq_veh, numVehsList_seq, VehsList_seq = xv[0], nV[0], V[0]
            gobs["prep"].add((bool(torch.is_grad_enabled()), bool(torch.is_inference_mode_enabled())))
            sync()
            t0 = time.perf_counter()
            # --- prep (replicates test.py: dense vector creation, grid masks on CPU, position change)
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            x_seq_veh, lookup_seq_veh = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)
            orig_x_seq = x_seq.clone()
            grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq, saved_args.TTC, saved_args.D_min,
                                                                    saved_args.num_sector, False, lookup_seq, lookup_seq)
            grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh, VehsList_seq, saved_args.TTC_veh,
                                                                                    saved_args.D_min_veh, saved_args.num_sector, False, lookup_seq,
                                                                                    lookup_seq_veh, is_heterogeneous=True, is_occupancy=False)
            x_seq, first_values_dict = position_change_seq(x_seq, PedsList_seq, lookup_seq)
            gobs["prep_tensors_require_grad"] = gobs["prep_tensors_require_grad"] or bool(x_seq.requires_grad or x_seq_veh.requires_grad or orig_x_seq.requires_grad)
            t1 = time.perf_counter()
            # --- transfer
            if cuda:
                x_seq = x_seq.cuda()
                x_seq_veh = x_seq_veh.cuda()
            sync()
            t2 = time.perf_counter()
            # --- rollout: the unchanged baseline sample() with ONE sample
            ret_x_seq, dist_param_seq = tmod.sample(x_seq[:obs], PedsList_seq[:obs], sample_args, net, x_seq, PedsList_seq, saved_args, dataloader,
                                                    lookup_seq, numPedsList_seq, saved_args.gru, first_values_dict, orig_x_seq, grid_seq[:obs], x_seq_veh,
                                                    VehsList_seq, lookup_seq_veh, grid_seq_veh_in_ped[:obs], grid_TTC_seq[:obs], grid_TTC_veh_seq[:obs])
            sync()
            t3 = time.perf_counter()
            gobs["output_tensors_require_grad"] = gobs["output_tensors_require_grad"] or bool(ret_x_seq.requires_grad or dist_param_seq.requires_grad)
            gobs["output_has_grad_fn"] = gobs["output_has_grad_fn"] or (ret_x_seq.grad_fn is not None or dist_param_seq.grad_fn is not None)
            gobs["post"].add((bool(torch.is_grad_enabled()), bool(torch.is_inference_mode_enabled())))
            # --- post (revert position change)
            ret_x_seq = revert_postion_change_seq2(ret_x_seq.cpu(), PedsList_seq, lookup_seq, first_values_dict, orig_x_seq, obs, infer=True)
            dist_param_seq[:, :, 0:2] = revert_postion_change_seq2(dist_param_seq[:, :, 0:2].cpu(), PedsList_seq, lookup_seq, first_values_dict, orig_x_seq,
                                                                   obs, infer=True)
            sync()
            t4 = time.perf_counter()
            finite = bool(torch.isfinite(ret_x_seq).all().item())
            return {"n_peds": len(lookup_seq), "n_vehicles": len(lookup_seq_veh), "prep_s": t1 - t0, "transfer_s": t2 - t1, "rollout_s": t3 - t2,
                    "post_s": t4 - t3, "pipeline_s": t4 - t0, "output_finite": finite, "output_shape": list(ret_x_seq.shape)}

        grad_before, inf_before = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        # ---------------- pass 1: pipeline latency
        seed_all()
        for b in range(warmup):
            one(b, None)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        rows: List[Dict[str, Any]] = []
        sync()
        loop_t0 = time.perf_counter()
        for b in range(warmup, n_use):
            r = one(b, None)
            r["trajectory_index"] = b
            rows.append(r)
        sync()
        loop_wall = time.perf_counter() - loop_t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None
        peak_res = torch.cuda.max_memory_reserved() / 2 ** 20 if cuda else None
        # ---------------- pass 2: model-only (forward hooks, same trajectories, same seed)
        h1, h2 = net.register_forward_pre_hook(pre_hook), net.register_forward_hook(post_hook)
        try:
            seed_all()
            for b in range(warmup):
                one(b, None)
            fwd = []
            calls = []
            for i, b in enumerate(range(warmup, n_use)):
                timers["fwd"], timers["calls"] = 0.0, 0
                one(b, None)
                rows[i]["model_forward_s"], rows[i]["n_forward_calls"] = timers["fwd"], timers["calls"]
                fwd.append(timers["fwd"])
                calls.append(timers["calls"])
        finally:
            h1.remove()
            h2.remove()
        state_after = model_state()
        if state_after != state_before_timing:
            raise RuntimeError("the benchmark changed the model's train/eval state: before %s, after %s" % (state_before_timing, state_after))
        params = list(net.parameters())
        fwd_modes = sorted(gobs["forward"])
        grad_in_forward = {g for g, _ in gobs["forward"]}
        if not gobs["forward"]:
            fwd_conclusion = "no model forward was observed"
        elif grad_in_forward == {False}:
            fwd_conclusion = "gradient tracking DISABLED during every observed model forward (by the baseline sample() torch.no_grad())"
        elif grad_in_forward == {True}:
            fwd_conclusion = "gradient tracking was ENABLED during every observed model forward"
        else:
            fwd_conclusion = "gradient tracking was enabled in some model forwards and disabled in others"
        gradient_tracking = {
            "benchmark_added_context": "none (no extra torch.no_grad / torch.inference_mode; identical to test.py)",
            "torch_grad_enabled_before_benchmark": bool(grad_before), "torch_inference_mode_enabled_before_benchmark": bool(inf_before),
            "baseline_sample_function_static_check": scope,
            "observed_inside_model_forward": [{"grad_enabled": g, "inference_mode": i} for g, i in fwd_modes],
            "observed_at_prep_stage_outside_sample": [{"grad_enabled": g, "inference_mode": i} for g, i in sorted(gobs["prep"])],
            "observed_at_post_stage_outside_sample": [{"grad_enabled": g, "inference_mode": i} for g, i in sorted(gobs["post"])],
            "prep_tensors_require_grad": gobs["prep_tensors_require_grad"], "output_tensors_require_grad": gobs["output_tensors_require_grad"],
            "output_has_grad_fn": gobs["output_has_grad_fn"],
            "parameters": {"count": len(params), "requiring_grad": sum(1 for q in params if q.requires_grad), "with_gradient_populated": sum(1 for q in params if q.grad is not None)},
            "conclusion": fwd_conclusion + "; the stages outside sample() (preparation, post-processing) ran with grad mode "
                          + ("enabled" if any(g for g, _ in gobs["prep"] | gobs["post"]) else "disabled")
                          + " and created autograd graph: %s; parameters keep requires_grad=True but no gradient was computed or stored"
                          % ("none (no tensor required grad)" if not (gobs["prep_tensors_require_grad"] or gobs["output_tensors_require_grad"] or gobs["output_has_grad_fn"])
                             else "YES - inspect the observation fields")}
        train_eval_state = {"at_load": state_at_load, "before_timing": state_before_timing, "after_all_passes": state_after,
                            "unchanged_during_benchmark": state_after == state_before_timing, "eval_called_by_benchmark": bool(eval_mode),
                            "changed_by_benchmark_relative_to_load": state_before_timing != state_at_load}
        fields = ["trajectory_index", "n_peds", "n_vehicles", "prep_s", "transfer_s", "rollout_s", "post_s", "pipeline_s", "model_forward_s", "n_forward_calls",
                  "output_finite"]
        with open(os.path.join(out, "inference_latency_samples.csv"), "x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        n = len(rows)
        summary = {
            "benchmark_version": BENCH_VERSION, "run_id": run_id, "finished_utc": _utc(), "scope": SCOPE,
            "environment": {"python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "cuda": cuda,
                            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
                            "gpu_total_vram_mib": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if cuda else None,
                            "gpu_driver": _gpu_driver(), "hardware_label": hardware_label,
                            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark},
            "protocol": {"epoch": epoch, "method": method, "sample_size": 1, "obs_length": obs, "pred_length": sample_args.pred_length, "seed": seed,
                         "seed_applies_to": "python random, numpy, torch, torch.cuda (set before each pass; not bitwise-deterministic across hardware)",
                         "warmup_trajectories_excluded": warmup, "timed_trajectories": n, "trajectories_available": n_total, "eval_mode": eval_mode,
                         "net_eval_called": eval_mode, "dropout_active_in_prediction_path": state_before_timing["dropout_modules_in_training_mode"] > 0 and state_before_timing["dropout_probabilities"] != [0.0],
                         "dropout_probability": getattr(saved_args, "dropout", None), "ground_truth_scoring_included": False, "oracle_selection_included": False,
                         "batch_size_of_input": 1, "saved_args_use_cuda_overridden_to_false_in_memory": cpu_override},
            "gradient_tracking": gradient_tracking, "model_train_eval_state": train_eval_state,
            "hashes": hashes, "setup_s": setup_s, "test_set_batch_creation_s": batch_creation_s,
            "latency_s": {"pipeline": describe([r["pipeline_s"] for r in rows]), "prep": describe([r["prep_s"] for r in rows]),
                          "transfer": describe([r["transfer_s"] for r in rows]), "rollout_sample_call": describe([r["rollout_s"] for r in rows]),
                          "post": describe([r["post_s"] for r in rows]), "model_forward_only_sum_per_trajectory": describe(fwd),
                          "model_forward_calls_per_trajectory": {"min": min(calls), "max": max(calls), "mean": sum(calls) / len(calls)}},
            "throughput_sequences_per_s": {"pipeline_pass": (n / loop_wall) if loop_wall > 0 else None, "definition": "timed trajectories / wall time of the timed loop (pass 1)"},
            "peak_gpu_memory_mib": {"max_allocated": peak_alloc, "max_reserved": peak_res, "note": "pass 1 after warm-up; None on CPU"},
            "output_check": {"non_finite_outputs": sum(1 for r in rows if not r["output_finite"]), "shapes_seen": sorted({tuple(r["output_shape"]) for r in rows})[:5]},
            "limitations": ["batch size 1, as in test.py; no batching study", "future actor information is still consumed by the baseline path",
                            "model-only pass uses forward hooks that synchronise, so it is separate from the pipeline pass",
                            "one dataset (HBS test split, 1081 sequences) and one device; no claim beyond this",
                            "no torch.inference_mode variant was measured: the baseline already runs sample() under torch.no_grad(), so an extra context would change nothing that is timed except the preparation/post stages"]}
        with open(os.path.join(out, "inference_benchmark_summary.json"), "x", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        return summary
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == repo:
            sys.path.pop(0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Single-sample PCG predictor benchmark (isolated; never modifies the repository or checkpoints).")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-trajectories", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval-mode", action="store_true")
    ap.add_argument("--hardware-label", default=None)
    ap.add_argument("--expected-checkpoint-sha256", default=None)
    ap.add_argument("--expected-config-sha256", default=None)
    a = ap.parse_args(argv)
    try:
        s = run_benchmark(a.repo, a.workdir, a.run_id, a.checkpoint, a.config, epoch=a.epoch, warmup=a.warmup, max_trajectories=a.max_trajectories, seed=a.seed,
                          use_cuda=False if a.cpu else None, eval_mode=a.eval_mode, expected_checkpoint_sha256=a.expected_checkpoint_sha256,
                          expected_config_sha256=a.expected_config_sha256, hardware_label=a.hardware_label)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"timed": s["protocol"]["timed_trajectories"], "pipeline_p50_s": s["latency_s"]["pipeline"].get("p50_s")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
