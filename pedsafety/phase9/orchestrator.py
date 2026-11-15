"""Phase 9 orchestration: Phase 3 artifacts -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8 -> evaluation gates, through the EXISTING public APIs only.

Rules: every stage input is contract-checked; a failing stage stops its dependants (``NOT_RUN``), never substitutes data; nothing is interpolated, merged or repaired; a stage that is
not selected takes its input from a validated reference run (precomputed mode) or fails with ``MissingInputError``. Stage 3 video execution is attempted only when its
dependencies exist; otherwise it is reported ``BLOCKED_ENVIRONMENT`` and the recorded Phase 3 run is used as precomputed input (never presented as measured here).
"""

import datetime
import hashlib
import importlib
import json
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..phase5.evaluation.experiment import load_baseline_config
from ..phase5.models.neural_evaluation import NeuralEvaluationConfig, load_trained_model, predict_windows, write_neural_predictions
from ..phase5.window_config import WindowConfig
from ..phase5.window_storage import read_dataset
from ..phase6.config import ConflictConfig
from ..phase6.pairs import build_conflict_pairs
from ..phase6.runner import REPO, load_predictions
from ..phase6.scoring import HeuristicProximityScorer
from ..phase7.risk import RiskConfig, assess_risk
from ..phase8.config import AlertConfig
from ..phase8.runner import run_alerts
from .benchmark import StageClock, hardware_info, peak_memory_bytes, realtime_assessment, summarize
from .contracts import (CONTRACT_VERSION, STAGES, sha256_file, validate_alert_outputs, validate_phase3_file, validate_predictions, validate_risk_records,
                        validate_windows_history_only, verify_chain, verify_outputs)
from .errors import MissingInputError, PipelineError, StageContractError
from .evaluation import assess_independent_evaluation, blocked_evaluations
from .reporting import build_attribution, execution_mode
from .verification import hash_inputs

CONFIG_VERSION = "phase9-pipeline-config-1.0.0"
MANIFEST_VERSION = "phase9-run-manifest-1.0.0"
LABEL = "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT"
DEPENDS = {"S3_PERCEPTION": [], "S4A_IMAGE_KINEMATICS": ["S3_PERCEPTION"], "S4B_INTERACTION": ["S4A_IMAGE_KINEMATICS"], "S5A_WINDOWS": ["S4B_INTERACTION"],
           "S5B_FORECAST": ["S5A_WINDOWS"], "S6_CONFLICT": ["S5B_FORECAST"], "S7_RISK": ["S6_CONFLICT"], "S8_ALERTS": ["S7_RISK"], "S9_EVALUATION_GATES": []}


@dataclass(frozen=True)
class PipelineConfig:
    config_version: str
    label: str
    stages: tuple
    phase3: Dict[str, str]
    reference_runs: Dict[str, str]
    configs: Dict[str, str]
    inference_windows: str
    benchmark: Dict[str, int]
    streaming_capable: bool
    target_fps: float
    output_root: str
    report_dir: Optional[str]

    def __post_init__(self):
        if self.config_version != CONFIG_VERSION or self.label != LABEL:
            raise PipelineError("CONFIG", "config_version must be %s and label %s" % (CONFIG_VERSION, LABEL))
        bad = [s for s in self.stages if s not in STAGES]
        if bad or not self.stages:
            raise PipelineError("CONFIG", "unknown or empty stage selection %s (known: %s)" % (bad, STAGES))
        if self.inference_windows not in ("reference", "reexecuted"):
            raise PipelineError("CONFIG", "inference_windows must be reference or reexecuted")
        for k in ("inference_warmup", "inference_repeats", "preprocessing_extra_repeats"):
            if not (isinstance(self.benchmark.get(k), int) and self.benchmark[k] >= 0):
                raise PipelineError("CONFIG", "benchmark.%s must be an integer >= 0" % k)
        if not isinstance(self.streaming_capable, bool) or not self.target_fps > 0:
            raise PipelineError("CONFIG", "streaming_capable must be boolean and target_fps positive")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PipelineConfig":
        data = {k: v for k, v in d.items() if not k.startswith("_")}
        unk, mis = set(data) - set(cls.__dataclass_fields__), set(cls.__dataclass_fields__) - set(data)
        if unk or mis:
            raise PipelineError("CONFIG", "unknown keys %s, missing keys %s" % (sorted(unk), sorted(mis)))
        data["stages"] = tuple(data["stages"])
        return cls(**data)

    @classmethod
    def load(cls, path) -> "PipelineConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise PipelineError("CONFIG", "cannot read %s: %s" % (path, exc))

    def to_dict(self):
        d = asdict(self)
        d["stages"] = list(self.stages)
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _p(x) -> Path:
    q = Path(str(x).replace("\\", "/"))
    return q if q.is_absolute() else REPO / q


def _load_json(p) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MissingInputError("MISSING_ARTIFACT", "cannot read %s: %s" % (p, exc))


def _probe_perception() -> Dict[str, Any]:
    def has(m):
        try:
            importlib.import_module(m)
            return True
        except Exception:                                                                    # noqa: BLE001
            return False
    cuda = False
    try:
        from ..phase5.models.neural_evaluation import torch      # torch is reached through the guarded Phase 5 module (torch imports are allowed only under phase5/models)
        cuda = bool(torch.cuda.is_available())
    except Exception:                                                                        # noqa: BLE001
        pass
    return {"cv2": has("cv2"), "ultralytics": has("ultralytics"), "cuda_device": cuda}


class Pipeline:
    def __init__(self, cfg: PipelineConfig, run_dir: Path):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.ctx: Dict[str, Any] = {}
        self.records: Dict[str, Dict[str, Any]] = {}
        self.clock = StageClock()
        self.det: Dict[str, Any] = {}
        self.counts: Dict[str, Any] = {}
        self.timing_detail: Dict[str, Any] = {}
        self.window_cfg = WindowConfig.load(_p(cfg.configs["windows"]))
        self.fps = self.window_cfg.source_fps

    # ---- resolution of inputs: this run's output, else a validated reference, else a clear failure
    def _ref(self, key: str) -> Path:
        v = self.cfg.reference_runs.get(key)
        if not v or not _p(v).exists():
            raise MissingInputError("MISSING_INPUT", "reference run %r is not configured or does not exist; cannot start from this stage" % key)
        return _p(v)

    def _dir(self, stage: str, key: str) -> Path:
        d = self.ctx.get(stage)
        return Path(d) if d else self._ref(key)

    # ---- stages
    def s3(self):
        d = _p(self.cfg.phase3["dir"])
        tracked, states, man = d / "tracked_agents.jsonl", d / "agent_states.jsonl", d / "run_manifest.json"
        for f in (tracked, states, man):
            if not f.is_file():
                raise MissingInputError("S3_MISSING", "%s not found (Phase 3 artifact or video required)" % f)
        stats = validate_phase3_file(tracked, self.fps)
        m = _load_json(man)
        res, st = m["results"], m.get("stage_latency_ms_in_pipeline_run", {})
        probes = _probe_perception()
        rec = {"frames_processed": res["frames_processed"], "frames_skipped": res["frames_skipped"], "wall_time_s": m["wall_time_s"],
               "frames_per_s_wall_including_decode_and_io": res["frames_processed"] / m["wall_time_s"],
               "detect_mean_ms": st.get("detect", {}).get("mean_ms"), "track_mean_ms": st.get("track", {}).get("mean_ms"),
               "hardware": m["environment"].get("cuda", {}).get("gpus"), "cpu": m["environment"].get("cpu"), "backend": m.get("backend"), "is_mock": m.get("is_mock"),
               "caveat": "RECORDED from the Colab T4 run of Phase 3 (normal run, no warm-up or stage isolation); not measured here, different hardware"}
        self.ctx.update({"phase3_dir": d, "phase3_tracked": tracked, "phase3_states": states, "phase3_manifest": man, "phase3_manifest_data": m})
        self.counts.update({"frames_processed_phase3": res["frames_processed"], "frames_skipped_phase3": res["frames_skipped"], "phase3_track_rows": stats["rows"],
                            "phase3_tracks": stats["tracks"]})
        self.ctx["perception_record"] = rec
        self.ctx["perception_probes"] = probes
        return {"mode": "PRECOMPUTED_PHASE3_ARTIFACTS_VALIDATED", "stats": stats, "video_execution": {
            "executed": False, "status": "BLOCKED_ENVIRONMENT" if not (probes["cv2"] and probes["ultralytics"]) else "NOT_REQUESTED_NO_RAW_VIDEO", "probes": probes,
            "raw_video_found": any(d.glob("*.mp4")), "reason": "no raw PIE video in the repository (Phase 3 ran on Colab); %s" % (
                "OpenCV/Ultralytics are missing in this environment" if not (probes["cv2"] and probes["ultralytics"]) else "no video supplied"), "recorded_phase3_run": rec}}

    def s4a(self):
        from ..image_kinematics import ImageKinematicsConfig
        from ..image_kinematics_run import run_image_kinematics
        out = self.run_dir / "S4A_image_kinematics"
        c = _p(self.cfg.configs["image_kinematics"])
        m = run_image_kinematics(self.ctx["phase3_states"], ImageKinematicsConfig.load(c), out, config_path=str(c), run_manifest_path=self.ctx["phase3_manifest"],
                                 tracked_agents_path=self.ctx["phase3_tracked"], command="phase9 S4A")
        verify_outputs(out, m, ["causal_kinematics.jsonl"])
        if m["counts"]["quarantined"] or m["counts"]["causal_rows"] != self.counts["phase3_track_rows"]:
            raise StageContractError("S4A_ROWS", "Task 2 quarantined rows or changed the row count (%s vs %s)" % (m["counts"], self.counts["phase3_track_rows"]))
        self.ctx["task2_dir"] = out
        self.counts.update({"agent_states_total": m["counts"]["causal_rows"], "agent_states_valid_position": m["counts"]["valid_positions"],
                            "agent_states_valid_velocity": m["counts"]["valid_velocity"], "agent_states_quarantined": m["counts"]["quarantined"]})
        return {"counts": m["counts"], "determinism": self._compare(out, m, "task2_dir", ["causal_kinematics.jsonl"], "run_manifest.json")}

    def s4b(self):
        from ..interaction.config import InteractionConfig
        from ..interaction.runner import run_interaction_representation
        t2 = self._dir("task2_dir", "task2_dir")
        t2m = _load_json(t2 / "run_manifest.json")
        if self.ctx.get("task2_dir") is None:
            verify_outputs(t2, t2m, ["causal_kinematics.jsonl"])
        out = self.run_dir / "S4B_interaction"
        c = _p(self.cfg.configs["interaction"])
        m = run_interaction_representation(t2 / "causal_kinematics.jsonl", t2 / "run_manifest.json", self.ctx["phase3_states"], InteractionConfig.load(c), out, config_path=str(c),
                                           command="phase9 S4B")
        if m["counts"]["nodes"] != t2m["counts"]["causal_rows"]:
            raise StageContractError("S4B_NODES", "node count %s differs from the causal row count %s" % (m["counts"]["nodes"], t2m["counts"]["causal_rows"]))
        self.ctx["task3_dir"] = out
        names = sorted(n for n in m["outputs"] if n.endswith(".npz"))
        return {"counts": {k: m["counts"][k] for k in ("nodes", "tracks") if k in m["counts"]}, "determinism": self._compare(out, m, "task3_dir", names, "run_manifest.json")}

    def s5a(self):
        from ..phase5.runner import generate_windows
        t3 = self._dir("task3_dir", "task3_dir")
        out = self.run_dir / "S5A_windows"
        c = _p(self.cfg.configs["windows"])
        m = generate_windows(t3, self.window_cfg, out, config_path=str(c), command="phase9 S5A", phase3_manifest=self.ctx["phase3_manifest"], tracked_agents=self.ctx["phase3_tracked"])
        verify_outputs(out, m, ["windows.npz", "graph_bank.npz"])
        ws, _, _ = read_dataset(out)
        info = validate_windows_history_only(ws, self.fps)
        t2 = self._dir("task2_dir", "task2_dir")
        links = verify_chain(phase3_agent_states=self.ctx["phase3_states"], task2_dir=t2, task3_dir=t3, windows_dir=out, t2=_load_json(t2 / "run_manifest.json"),
                             t3=_load_json(t3 / "run_manifest.json"), w=m)
        self.ctx["windows_reexec_dir"] = out
        self.counts.update({"windows_accepted": m["counts"]["accepted_windows"], "windows_rejected": m["counts"]["rejected_windows"],
                            "windows_rejection_reasons": m["counts"]["rejection_reasons"]})
        return {"history_only": info, "chain": links, "determinism": self._compare(out, m, "windows_dir", ["windows.npz", "graph_bank.npz", "rejected_candidates.npz"], "run_manifest.json")}

    def _windows_for_model(self) -> Path:
        if self.cfg.inference_windows == "reexecuted":
            d = self.ctx.get("windows_reexec_dir")
            if d is None:
                raise MissingInputError("MISSING_INPUT", "inference_windows=reexecuted but S5A did not run in this pipeline")
            return Path(d)
        return self._ref("windows_dir")

    def _bundle(self):
        if "bundle" not in self.ctx:
            wd = self._windows_for_model()
            self.ctx["bundle"] = read_dataset(wd)
        return self.ctx["bundle"]

    def s5b(self):
        from ..phase5.models.neural_evaluation import torch      # torch is reached through the guarded Phase 5 module (torch imports are allowed only under phase5/models)
        wdir = self._windows_for_model()
        ncfg = NeuralEvaluationConfig.load(_p(self.cfg.configs["neural_eval"]))
        if ncfg.torch_threads is not None:
            torch.set_num_threads(ncfg.torch_threads)
        t0 = time.perf_counter()
        loaded = load_trained_model(self._ref("training_dir"), wdir, checkpoint=ncfg.checkpoint, device=ncfg.device)       # verifies checkpoint / scaler / config / window-run hashes
        t_load = time.perf_counter() - t0
        ds = loaded.dataset
        info = validate_windows_history_only(ds.ws, self.fps)
        fwd = {"s": 0.0}
        h1 = loaded.model.register_forward_pre_hook(lambda m, i: fwd.__setitem__("t", time.perf_counter()))
        h2 = loaded.model.register_forward_hook(lambda m, i, o: fwd.__setitem__("s", fwd["s"] + time.perf_counter() - fwd["t"]))
        t1 = time.perf_counter()
        pred = predict_windows(loaded.model, ds, ncfg.batch_size)
        t_pred = time.perf_counter() - t1
        h1.remove()
        h2.remove()
        if not np.array_equal(pred["window_id"], ds.ws.window_id):
            raise StageContractError("S5B_ORDER", "prediction order differs from the window dataset")
        # future-target isolation: scrambling every future target must not change any prediction
        sub = list(range(0, len(ds), max(1, len(ds) // 48)))[:48]
        base = predict_windows(loaded.model, ds, ncfg.batch_size, indices=sub)
        saved_uv, saved_v = ds.ws.target_future_uv_px.copy(), ds.ws.target_future_valid.copy()
        try:
            ds.ws.target_future_uv_px[...] = saved_uv[::-1] + 1000.0
            ds.ws.target_future_valid[...] = ~saved_v
            pert = predict_windows(loaded.model, ds, ncfg.batch_size, indices=sub)
        finally:
            ds.ws.target_future_uv_px[...] = saved_uv
            ds.ws.target_future_valid[...] = saved_v
        iso = all(np.array_equal(base[k], pert[k]) for k in ("positions_uv_px", "valid", "mode_probabilities"))
        if not iso:
            raise StageContractError("FUTURE_LEAK", "predictions changed when future targets were scrambled: a future observation reaches the model")
        out = self.run_dir / "S5B_forecast"
        out.mkdir(parents=True, exist_ok=True)
        t2_ = time.perf_counter()
        write_neural_predictions(out / "predictions_neural.npz", pred, ds, ncfg.model_id, ds.config.future_offsets / ds.config.source_fps, ds.config)
        t_io = time.perf_counter() - t2_
        ref = self.cfg.reference_runs.get("neural_eval_dir")
        same = None
        if ref and (_p(ref) / "predictions_neural.npz").is_file():
            same = sha256_file(out / "predictions_neural.npz") == sha256_file(_p(ref) / "predictions_neural.npz")
        self.ctx.update({"loaded": loaded, "predictions_path": out / "predictions_neural.npz", "checkpoint_file_sha256": loaded.verification["checkpoint_file_sha256"],
                         "ncfg": ncfg, "windows_model_dir": wdir})
        self.timing_detail["S5B"] = {"model_load_and_verify_s": t_load, "predict_total_s": t_pred, "model_forward_s": fwd["s"], "collate_and_other_s": t_pred - fwd["s"],
                                     "write_predictions_s": t_io, "windows": len(ds), "forward_ms_per_window": 1000 * fwd["s"] / len(ds)}
        self.counts.update({"windows_predicted": len(ds)})
        return {"history_only": info, "future_isolation": {"passed": iso, "windows_perturbed": len(sub)}, "verification": loaded.verification,
                "determinism": {"predictions_equal_reference": same, "note": "bitwise equality needs torch_threads=1 as configured"}}

    def s6(self):
        ws, bank, extra = self._bundle()
        ccfg = ConflictConfig.load(_p(self.cfg.configs["conflict"]))
        pred = load_predictions(self.ctx.get("predictions_path") or (self._ref("neural_eval_dir") / "predictions_neural.npz"))
        validate_predictions(pred, ws, self.window_cfg, ccfg)
        bcfg, bsha = load_baseline_config(REPO / ccfg.baseline_evaluation_config_path)
        tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
        batch = build_conflict_pairs(pred, ws, bank, extra["history_target_bank_node"], self.window_cfg, ccfg, bcfg, track_ids=tid,
                                     provenance={"baseline_config_sha256_canonical": bsha, "phase9_stage": "S6"})
        if batch.provenance.get("future_observations_used") or batch.provenance.get("labels_used"):
            raise StageContractError("S6_PROVENANCE", "conflict stage reports use of future observations or labels")
        self.ctx.update({"batch": batch, "ccfg": ccfg})
        self.counts.update({"pairs": batch.n_pairs, "pairs_valid": int(batch.masks["pair_valid"].sum()), "pairs_invalid": int((~batch.masks["pair_valid"]).sum())})
        return {"exclusions": batch.exclusions}

    def s7(self):
        ws, _, _ = self._bundle()
        batch = self.ctx["batch"]
        rcfg = RiskConfig.load(_p(self.cfg.configs["risk"]))
        ck = self.ctx.get("checkpoint_file_sha256") or _load_json(self._ref("neural_eval_dir") / "run_manifest.json")["inputs"]["checkpoint_file_sha256"]
        out = assess_risk(batch, HeuristicProximityScorer().predict(batch), rcfg, None, provenance={"dataset": "PIE set01/video_0004", "label": LABEL, "sample_status": "IN_SAMPLE_PHASE5_MODEL", "phase9_stage": "S7"},
                          checkpoint_file_sha256=ck, phase6_config_sha256=batch.config_sha256)
        recs = out.to_records()
        info = validate_risk_records(recs, ws, self.fps)
        self.ctx.update({"risk": out, "records": recs, "risk_cfg": rcfg, "ck": ck})
        cats = {}
        for r in recs:
            cats[r["risk_category"]] = cats.get(r["risk_category"], 0) + 1
        self.counts.update({"risk_categories": cats, "insufficient_evidence_records": cats.get("INSUFFICIENT_EVIDENCE", 0)})
        det = None
        ref = self.cfg.reference_runs.get("phase7_dir")
        if ref and (_p(ref) / "risk_arrays.npz").is_file():
            a = np.load(_p(ref) / "risk_arrays.npz")
            det = {"risk_score_equal_reference": bool(np.array_equal(a["risk_score"], out.risk_score, equal_nan=True)),
                   "risk_category_equal_reference": bool(np.array_equal(a["risk_category"], out.risk_category.astype(str)))}
        return {"validation": info, "determinism": det}

    def s8(self):
        recs, out = self.ctx["records"], self.run_dir / "S8_alerts"
        acfg = AlertConfig.load(_p(self.cfg.configs["alert"]))
        risk = self.ctx["risk"]
        hashes = {"phase7_risk_config_sha256": risk.provenance["risk_config_sha256"], "source_checkpoint_file_sha256": self.ctx["ck"], "calibration_artifact_sha256": None}
        rp = risk.provenance
        nev, p7d = self.cfg.reference_runs.get("neural_eval_dir"), self.cfg.reference_runs.get("phase7_dir")
        prov = {"dataset": rp.get("dataset"), "label": rp.get("label"), "sample_status": rp.get("sample_status"), "phase7_schema": "phase7-risk-output-1.0.0",
                "phase7_category_status": rp.get("category_status"), "phase7_threshold_evidence": rp.get("threshold_evidence"), "coordinate_system": rp.get("coordinate_system"),
                "phase6_config_sha256": rp.get("phase6_config_sha256"), "phase5_run_manifest_sha256": sha256_file(_p(nev) / "run_manifest.json") if nev and (_p(nev) / "run_manifest.json").is_file() else None,
                "phase7_risk_report_sha256": sha256_file(_p(p7d) / "risk_report.json") if p7d and (_p(p7d) / "risk_report.json").is_file() else None,
                "phase7_calibration_status": "BLOCKED_NO_VERIFIED_CONFLICT_LABELS" if rp.get("calibration_artifact_sha256") is None else "CALIBRATED",
                "phase5_predictions_sha256": sha256_file(self.ctx["predictions_path"]) if self.ctx.get("predictions_path") else None, "phase9_stage": "S8", "phase9_execution_mode": "PRECOMPUTED_PHASE3_ARTIFACTS"}
        res = run_alerts(recs, acfg, prov, hashes, out)
        info = validate_alert_outputs(res["events"], res["alerts"], recs, self.fps)
        self.ctx["alert_result"] = res
        s = res["summary"]
        self.counts.update({"alerts": s["n_alerts"], "alert_events": s["n_events"], "alerts_escalated": s["alerts_reached_escalated"], "alerts_by_clearance_kind": s["alerts_by_clearance_kind"],
                            "alert_transitions_to_insufficient_evidence": sum(v for k, v in s["transitions"].items() if k.endswith("->INSUFFICIENT_EVIDENCE"))})
        det = None
        ref = self.cfg.reference_runs.get("phase8_dir")
        if ref and (_p(ref) / "summary_report.json").is_file():
            r = _load_json(_p(ref) / "summary_report.json")
            det = {"n_alerts_equal_reference": r["n_alerts"] == s["n_alerts"], "transitions_equal_reference": r["transitions"] == s["transitions"]}
        return {"validation": info, "summary": {k: s[k] for k in ("n_events", "n_alerts", "alerts_by_clearance_kind")}, "determinism": det}

    def s9(self):
        a = assess_independent_evaluation(REPO, self.cfg.configs.get("split_manifest"))
        self.ctx["assessment"] = a
        return {"independent_evaluation": a, "blocked_evaluations": blocked_evaluations(a)}

    # ---- helpers
    def _compare(self, new_dir, new_manifest, ref_key: str, names, manifest_name) -> Dict[str, Any]:
        ref = self.cfg.reference_runs.get(ref_key)
        if not ref or not _p(ref).exists():
            return {"reference": None, "equal": None}
        rm = _load_json(_p(ref) / manifest_name)
        eq = {n: (new_manifest["outputs"][n]["sha256"] == rm["outputs"][n]["sha256"]) for n in names if n in rm.get("outputs", {})}
        return {"reference": str(ref), "files": eq, "equal": bool(eq) and all(eq.values())}

    def run(self) -> Dict[str, Any]:
        fn = {"S3_PERCEPTION": self.s3, "S4A_IMAGE_KINEMATICS": self.s4a, "S4B_INTERACTION": self.s4b, "S5A_WINDOWS": self.s5a, "S5B_FORECAST": self.s5b, "S6_CONFLICT": self.s6,
              "S7_RISK": self.s7, "S8_ALERTS": self.s8, "S9_EVALUATION_GATES": self.s9}
        for sid in STAGES:
            if sid not in self.cfg.stages:
                self.records[sid] = {"status": "NOT_SELECTED", "note": "input (if needed) comes from a validated reference run"}
                continue
            blockers = [d for d in DEPENDS[sid] if d in self.cfg.stages and self.records.get(d, {}).get("status") != "COMPLETED"]
            if blockers:
                self.records[sid] = {"status": "NOT_RUN", "reason": "UPSTREAM_NOT_COMPLETED: %s" % blockers}
                continue
            try:
                with self.clock.stage(sid):
                    detail = fn[sid]()
                self.records[sid] = {"status": "COMPLETED", **self.clock.records[sid], "detail": detail}
            except Exception as exc:                                                                            # noqa: BLE001 - recorded, never swallowed
                code = getattr(exc, "code", type(exc).__name__)
                self.records[sid] = {"status": "FAILED", "error_code": code, "error": str(exc)[:600], "traceback_tail": traceback.format_exc().splitlines()[-3:]}
        return self.records


def _benchmark(pipe: Pipeline) -> Dict[str, Any]:
    """Steady-state repeats of the inference chain (model forward, collate, conflict, risk, alert manager) after warm-up, plus extra cold re-executions of the heavy preprocessing."""
    cfg = pipe.cfg.benchmark
    out: Dict[str, Any] = {"hardware": hardware_info(), "first_pass_includes_warmup_effects": True}
    needed = ("S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS")
    if all(pipe.records.get(s, {}).get("status") == "COMPLETED" for s in needed) and cfg["inference_repeats"] >= 1:
        loaded, ncfg = pipe.ctx["loaded"], pipe.ctx["ncfg"]
        ws, bank, extra = pipe._bundle()
        ccfg, rcfg, batch0 = pipe.ctx["ccfg"], pipe.ctx["risk_cfg"], pipe.ctx["batch"]
        bcfg, _ = load_baseline_config(REPO / ccfg.baseline_evaluation_config_path)
        acfg = AlertConfig.load(_p(pipe.cfg.configs["alert"]))
        tid = [str(i) for i in range(int(bank.nodes["track_index"].max()) + 1)]
        runs: Dict[str, List[float]] = {k: [] for k in ("model_forward", "collate_and_other", "conflict_pairs", "risk", "alert_manager", "inference_chain_total")}

        def once():
            fwd = {"s": 0.0}
            h1 = loaded.model.register_forward_pre_hook(lambda m, i: fwd.__setitem__("t", time.perf_counter()))
            h2 = loaded.model.register_forward_hook(lambda m, i, o: fwd.__setitem__("s", fwd["s"] + time.perf_counter() - fwd["t"]))
            t0 = time.perf_counter()
            pred = predict_windows(loaded.model, loaded.dataset, ncfg.batch_size)
            t1 = time.perf_counter()
            h1.remove()
            h2.remove()
            p = load_predictions(pipe.ctx["predictions_path"])
            p.positions_uv_px, p.valid, p.mode_probabilities = pred["positions_uv_px"], pred["valid"], pred["mode_probabilities"]
            t2 = time.perf_counter()
            b = build_conflict_pairs(p, ws, bank, extra["history_target_bank_node"], pipe.window_cfg, ccfg, bcfg, track_ids=tid)
            t3 = time.perf_counter()
            r = assess_risk(b, HeuristicProximityScorer().predict(b), rcfg, None, checkpoint_file_sha256=pipe.ctx["ck"], phase6_config_sha256=b.config_sha256)
            recs = r.to_records()
            t4 = time.perf_counter()
            run_alerts(recs, acfg, None, None, None)
            t5 = time.perf_counter()
            return [fwd["s"], (t1 - t0) - fwd["s"], t3 - t2, t4 - t3, t5 - t4, t5 - t0]
        warm = []
        for _ in range(cfg["inference_warmup"]):
            t = time.perf_counter()
            once()
            warm.append(time.perf_counter() - t)
        for _ in range(cfg["inference_repeats"]):
            for k, v in zip(runs, once()):
                runs[k].append(v)
        out["inference_chain"] = {"warmup_total_s": warm, "steady_state": {k: summarize(v) for k, v in runs.items()}, "windows": len(loaded.dataset), "pairs": batch0.n_pairs,
                                  "scope": "model forward, collate, Phase 6 pairs, Phase 7 risk, Phase 8 manager; excludes file I/O (measured separately) and all preprocessing"}
        tot = out["inference_chain"]["steady_state"]["inference_chain_total"]["mean"]
        out["inference_chain"]["ms_per_window_mean"] = 1000 * tot / len(loaded.dataset)
    if cfg["preprocessing_extra_repeats"] >= 1 and all(pipe.records.get(s, {}).get("status") == "COMPLETED" for s in ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS")):
        runs = {"S4A": [pipe.records["S4A_IMAGE_KINEMATICS"]["wall_s"]], "S4B": [pipe.records["S4B_INTERACTION"]["wall_s"]], "S5A": [pipe.records["S5A_WINDOWS"]["wall_s"]]}
        scratch = Path(tempfile.mkdtemp(prefix="p9bench_"))
        try:
            for i in range(cfg["preprocessing_extra_repeats"]):
                sub = Pipeline(pipe.cfg, scratch / ("r%d" % i))
                sub.ctx.update({k: pipe.ctx[k] for k in ("phase3_dir", "phase3_tracked", "phase3_states", "phase3_manifest")})
                sub.counts["phase3_track_rows"] = pipe.counts["phase3_track_rows"]
                for sid, f in (("S4A", sub.s4a), ("S4B", sub.s4b), ("S5A", sub.s5a)):
                    t = time.perf_counter()
                    f()
                    runs[sid].append(time.perf_counter() - t)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        out["preprocessing"] = {"scope": "Phase 4 Task 2, Task 3 and Phase 5 window generation, each including its own file I/O (JSONL/NPZ read and write); first sample = main run", "runs_s": runs,
                                "stats": {k: summarize(v) for k, v in runs.items()}}
    return out


def run_pipeline(cfg: PipelineConfig, *, run_id: Optional[str] = None, benchmark: bool = True) -> Dict[str, Any]:
    run_id = run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _p(cfg.output_root) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PipelineError("OUTPUT_DIR", "%s exists and is not empty; runs never overwrite" % run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    pipe = Pipeline(cfg, run_dir)
    hashes_before = hash_inputs(cfg.to_dict(), REPO)
    records = pipe.run()
    hashes_after = hash_inputs(cfg.to_dict(), REPO)
    perf = _benchmark(pipe) if benchmark and all(r["status"] != "FAILED" for r in records.values()) else {"hardware": hardware_info(), "note": "benchmark skipped (disabled or a stage failed)"}
    total_s = time.perf_counter() - t0
    failed = [s for s, r in records.items() if r["status"] == "FAILED"]
    not_run = [s for s, r in records.items() if r["status"] == "NOT_RUN"]
    status = "FAILED" if failed else ("INCOMPLETE" if not_run else "COMPLETED")
    chain = [s for s in ("S4A_IMAGE_KINEMATICS", "S4B_INTERACTION", "S5A_WINDOWS", "S5B_FORECAST", "S6_CONFLICT", "S7_RISK", "S8_ALERTS") if records[s]["status"] == "COMPLETED"]
    chain_s = sum(records[s]["wall_s"] for s in chain)
    frames = pipe.counts.get("frames_processed_phase3")
    perf["main_run_stage_wall_s"] = {s: records[s].get("wall_s") for s in STAGES}
    perf["downstream_stages_measured"] = chain
    perf["downstream_wall_s_first_pass"] = chain_s
    if frames and chain_s > 0:
        perf["downstream_frames_per_s_offline_batch"] = frames / chain_s
        perf["downstream_ms_per_frame"] = 1000 * chain_s / frames
    perf["stage_detail"] = pipe.timing_detail
    perf["gpu"] = "NOT_APPLICABLE_NO_CUDA_DEVICE" if not perf["hardware"].get("cuda_available") else perf["hardware"].get("gpu")
    perf["process_peak_working_set_bytes"] = peak_memory_bytes()
    perf["realtime_assessment"] = realtime_assessment(fps_target=cfg.target_fps, frames=frames or 0, measured_seconds=chain_s, perception_executed=False,
                                                      streaming_capable=cfg.streaming_capable)
    reg = failure_register(records, pipe)
    manifest = {"manifest_version": MANIFEST_VERSION, "contract_version": CONTRACT_VERSION, "label": LABEL, "status": status, "run_id": run_id, "started_utc": started,
                "total_wall_s": total_s, "config": {"sha256": cfg.sha256(), "content": cfg.to_dict()}, "stages": records, "counts": pipe.counts,
                "performance": perf, "failure_and_limitation_register": reg, "sample_status": "IN_SAMPLE_PHASE5_MODEL_ONE_VIDEO",
                "scientific_evaluation": pipe.ctx.get("assessment") and {"independent": pipe.ctx["assessment"]["status"], "blocked": blocked_evaluations(pipe.ctx["assessment"])},
                "claims": {"independent_generalization": False, "calibrated_probability": False, "physical_ttc_pet": False, "validated_early_warning_time": False,
                           "real_time_performance": False, "deployment_ready": False},
                "run_dir": str(run_dir), "inputs": {"sha256_before": hashes_before, "unchanged_after_run": hashes_before == hashes_after}}
    manifest.update(execution_mode(manifest))
    manifest["performance_attribution"] = build_attribution(manifest, perf.get("hardware"))
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), encoding="utf-8")
    if cfg.report_dir:
        rd = _p(cfg.report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in manifest.items()}
        (rd / "run_manifest.json").write_text(json.dumps(slim, indent=1, sort_keys=True, default=str), encoding="utf-8")
        (rd / "performance.json").write_text(json.dumps(perf, indent=1, sort_keys=True, default=str), encoding="utf-8")
        (rd / "failure_and_limitation_register.json").write_text(json.dumps(reg, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def failure_register(records: Mapping[str, Mapping[str, Any]], pipe: Pipeline) -> Dict[str, Any]:
    failures = [{"stage": s, "code": r.get("error_code"), "message": r.get("error")} for s, r in records.items() if r["status"] == "FAILED"]
    return {"register_version": "phase9-failure-limitation-register-1.0.0", "run_failures": failures,
            "limitations": [
                {"id": "L1", "area": "data", "text": "One PIE video (set01/video_0004); the Phase 5 model is trained and evaluated on the same windows (in-sample)."},
                {"id": "L2", "area": "data", "text": "No annotation files, verified conflict labels, verified events or independent scene groups exist: independent evaluation is blocked."},
                {"id": "L3", "area": "coordinates", "text": "Image-space pixels only; the camera moves; no ego-motion compensation, no metric geometry, no TTC/PET/distance/severity."},
                {"id": "L4", "area": "streaming", "text": "The window builder accepts a window only if the target also has its 15 future samples, so windows (and alerts) are conditioned on future track existence; the pipeline is an offline replay, not a causal online system. Inference inputs themselves are history-only (verified by future-scrambling)."},
                {"id": "L5", "area": "perception", "text": "Video decoding and detection/tracking were not executed in this environment (no raw video, no OpenCV/Ultralytics, no GPU); Phase 3 results are recorded from the Colab T4 run."},
                {"id": "L6", "area": "provenance", "text": "The Phase 5 model verifies the exact window-run manifest hash; regenerating windows creates a new manifest, so the model rejects them (by design). Re-execution is therefore checked for data-file equality and inference uses the reference window run."},
                {"id": "L7", "area": "performance", "text": "CPU-only, one machine; peak memory is the process peak working set; no GPU, no end-to-end real-time measurement."},
                {"id": "L8", "area": "calibration", "text": "Phase 7 probabilities are null; categories and Phase 8 thresholds are unvalidated engineering configuration."},
                {"id": "L9", "area": "alerts", "text": "Alerts are properties of a heuristic on one in-sample video; most close by evidence loss (pair no longer observed), not by risk subsiding."},
                {"id": "L11", "area": "regression", "text": "tests/phase2/test_ttc_masks.py::test_ttc_occupancy_mask_collapses_agent_dimension fails at baseline (UnboundLocalError in utils/Interaction.py); verified pre-existing at commit afdf822, unrelated to Phases 3-9."},
                {"id": "L13", "area": "performance", "text": "Peak memory is the process peak working set (psutil unavailable); S3 time is artifact validation, not perception; phase 4-8 offline-batch throughput is not raw-video end-to-end FPS."}]}


__all__ = ["PipelineConfig", "Pipeline", "run_pipeline", "failure_register", "DEPENDS", "LABEL"]
