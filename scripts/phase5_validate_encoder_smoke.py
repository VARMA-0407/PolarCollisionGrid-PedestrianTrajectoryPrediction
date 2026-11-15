"""Phase 5 P5.5 CLI: independently validate an encoder smoke-test run (read-only unless --report names a new file).

Checks: data-artifact hashes equal the manifest; the P5.3 input re-verifies and its manifest hash equals the recorded one;
the encoder configuration hash matches; labels and claims are as required; the saved results are RECOMPUTED by running the
encoder again from the P5.3 inputs (output-tensor hashes, counts and validity must be identical); an independent presence /
slot / zero-at-absent / finite check is made with plain numpy on a sample of windows; parameter count equals a manual count.

Example:  python scripts/phase5_validate_encoder_smoke.py --run experiments/phase5_task5_encoder_smoke_pie_video_0004/<run>/primary
Exit codes: 0 all checks pass, 1 a check failed, 2 contract failure, 3 unexpected failure.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedsafety.errors import PedsafetyError  # noqa: E402
from pedsafety.image_kinematics_run import sha256_file  # noqa: E402
from pedsafety.phase5.graph_window_adapter import collate_windows  # noqa: E402
from pedsafety.phase5.models import EncoderConfig, EncoderInput, SpatioTemporalEncoder  # noqa: E402
from pedsafety.phase5.models.encoder import HISTORY_KEYS  # noqa: E402
from pedsafety.phase5.models.smoke import bicycle_encoder_config, forward_pass, verify_p53_run  # noqa: E402
from pedsafety.phase5.window_storage import read_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def validate_run(run_dir) -> dict:
    d = Path(run_dir)
    m = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    res = json.loads((d / "smoke_results.json").read_text(encoding="utf-8"))
    out = {"run": str(d), "checks": {}}

    def chk(name, ok, detail=None):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}
    bad = [n for n, r in m["data_artifacts"].items() if not (d / n).is_file() or sha256_file(d / n) != r["sha256"] or (d / n).stat().st_size != r["bytes"]]
    chk("data_artifact_hashes_match_manifest", not bad, bad or sorted(m["data_artifacts"]))
    chk("label_pilot_only", m["label"] == "PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT" and res["label"] == m["label"])
    chk("claims_all_false", all(v is False for v in m["claims"].values() if isinstance(v, bool)))
    chk("not_trained_and_no_targets_read", res["model"]["trained"] is False and m["inputs"]["targets_or_future_arrays_read"] is False
        and m["inputs"]["offline_kinematics_used"] is False)
    ecfg = EncoderConfig.load(REPO / m["inputs"]["encoder_config"]["path"])
    chk("encoder_config_hash_matches", ecfg.sha256() == m["inputs"]["encoder_config"]["sha256_canonical"] == res["encoder_config"]["sha256_canonical"])
    chk("scaling_is_nominal_not_fitted", res["model"]["feature_scaling"]["source"] == "NOMINAL_DEVELOPMENT_DEFAULT_NOT_FITTED"
        and res["model"]["feature_scaling"]["provenance"]["fitted"] is False)
    wrun = REPO / m["inputs"]["windows_run"]
    wman, wcfg, in_hashes = verify_p53_run(wrun)
    chk("p53_inputs_reverify_and_manifest_hash_matches", in_hashes == m["inputs"]["windows_run_files"])
    chk("receptive_field_covers_history", res["encoder_config"]["receptive_field"] >= wcfg.history_length == res["encoder_config"]["history_length_from_window_config"])
    ws, bank, _ = read_dataset(wrun)
    n_main = res["default_context_forward"]["windows"]
    re = forward_pass(ws, bank, wcfg, ecfg, res["default_context_forward"]["batch_size"])
    for k, v in re["hashes"].items():
        if k.endswith("_sha256"):
            chk("recomputed_" + k, v == res["default_context_forward"]["hashes"][k])
    for k in ("windows", "max_agent_slots", "present_node_steps", "nonfinite_values", "absent_nonzero_values", "target_slot_mismatch", "edges_by_relation"):
        chk("recomputed_count_" + k, re["stats"][k] == res["default_context_forward"][k], re["stats"][k])
    bike = forward_pass(ws, bank, wcfg, bicycle_encoder_config(ecfg), res["default_context_forward"]["batch_size"], limit=res["optional_bicycle_context_forward"]["windows"], include_optional=True)
    chk("recomputed_optional_bicycle_hash", bike["hashes"]["sequence_embeddings_sha256"] == res["optional_bicycle_context_forward"]["hashes"]["sequence_embeddings_sha256"])
    manual = sum(p.numel() for p in re["model"].parameters())
    chk("parameter_count_manual", manual == res["model"]["parameter_count"] == m["counts"]["parameter_count"], manual)
    # independent numpy checks on a sample of windows (every 97th)
    idx = list(range(0, ws.n_windows, 97))
    with torch.no_grad():
        batch = collate_windows(bank, ws, idx, wcfg)
        o = re["model"](EncoderInput.from_collated_batch(batch))
    seq, pres = o.sequence_embeddings.numpy(), o.presence.numpy()
    chk("independent_presence_equals_adapter_presence", np.array_equal(pres, np.transpose(batch["present"], (0, 2, 1))))
    chk("independent_absent_zero_and_finite", bool(np.isfinite(seq).all() and (seq[~pres] == 0).all()))
    chk("independent_slot0_is_target_track_present_every_step", bool(np.array_equal(batch["track_index"][:, 0], ws.target_track_index[idx]) and pres[:, 0].all()))
    fields = [f.name for f in dataclasses.fields(EncoderInput)]
    chk("encoder_input_has_no_future_or_target_fields", not any("future" in n or "target" in n for n in fields + list(HISTORY_KEYS)), fields)
    chk("numerical_validity_recorded_true", res["numerical_validity"]["overall"] is True and n_main == ws.n_windows)
    out["ok"] = all(v["pass"] for v in out["checks"].values())
    out["n_checks"] = len(out["checks"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", help="write the JSON report to this NEW file (refuses to overwrite)")
    a = ap.parse_args(argv)
    try:
        rep = validate_run(a.run)
    except PedsafetyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:                                            # noqa: BLE001
        print("unexpected failure: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 3
    if a.report:
        p = Path(a.report)
        if p.exists():
            print("error: report path exists; refusing to overwrite: %s" % p, file=sys.stderr)
            return 2
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    failed = [k for k, v in rep["checks"].items() if not v["pass"]]
    print("%s: %d/%d checks passed%s" % ("PASS" if rep["ok"] else "FAIL", rep["n_checks"] - len(failed), rep["n_checks"], "" if not failed else "; failed: %s" % failed))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
