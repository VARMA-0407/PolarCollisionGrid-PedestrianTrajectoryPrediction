# Phase 9 — Integration guide

All commands run from the repository root with `C:/Users/revan/anaconda3/envs/pcg-baseline/python.exe` (Python 3.9, torch CPU). Outputs go to `outputs/phase9_runs/<run-id>/` (git-ignored);
the consolidated, committed records are in `experiments/phase9_end_to_end_pie_video_0004/`. A run never overwrites a non-empty run directory (exit code 2).

## Configuration
`configs/phase9/pipeline_pie_video_0004.json` (strict; unknown keys rejected; `_` keys are comments): stage selection, Phase 3 input directory, `reference_runs` (validated Phase 4/5/7/8 artifacts used
when a stage is not re-executed and for exact comparison), per-stage configuration files, `inference_windows` (`reference` — required while the model is bound to the reference window manifest),
benchmark repetitions, `streaming_capable` (false: no causal online mode exists), `target_fps` (30), `output_root`, `report_dir`.

## Commands
| Command | Purpose |
|---|---|
| `python -m pedsafety.phase9 run --config C [--stages S...] [--run-id ID] [--output-root D] [--report-dir D\|""] [--no-benchmark]` | execute the selected stages; exit 0 completed, 1 a stage failed, 2 configuration/output-directory error |
| `python -m pedsafety.phase9 verify-run --run-dir R --config C [--out F]` | exact comparison of a run with the committed Phase 5/7/8 references + input hashes |
| `python -m pedsafety.phase9 attribute --manifest M [--out F]` | execution-mode statement and per-component time attribution derived from a manifest |
| `python -m pedsafety.phase9 evaluation-status [--split-manifest S]` | which independent evaluations are possible (currently `BLOCKED_INSUFFICIENT_INDEPENDENT_SCENES`) |

Typical uses: full precomputed-artifact run with benchmark (`run --config C --run-id ID`, ~16 min on the benchmark machine); inference-only replay from precomputed stages
(`--stages S3_PERCEPTION S5B_FORECAST S6_CONFLICT S7_RISK S8_ALERTS S9_EVALUATION_GATES --no-benchmark`, ~25 s); starting mid-pipeline without a configured reference run fails with `MISSING_INPUT`.

## Python API
`PipelineConfig.load(path)`, `run_pipeline(cfg, run_id=None, benchmark=True) -> manifest`, `Pipeline(cfg, run_dir).run()`; validators in `contracts`; gates and metrics in `evaluation`
(`assess_independent_evaluation`, `require_independent`, `evaluate_*_independent`, `alert_level_metrics`).

## Starting from a raw video (not available here)
Run Phase 3 (`scripts/` perception runner, needs OpenCV, Ultralytics and ideally a GPU) to produce `tracked_agents.jsonl`, `agent_states.jsonl`, `run_manifest.json`, point `phase3.dir` at them and run
the pipeline. The orchestrator does not launch YOLO itself and reports Phase 3 video execution as `BLOCKED_ENVIRONMENT` / `NOT_REQUESTED_NO_RAW_VIDEO` instead of fabricating a measurement.

## Adding independent data later
Freeze a group-level split manifest (`phase9-group-split-1.0.0`: disjoint train / calibration / validation / test groups, verified annotation provenance, identity mapping and timestamp alignment,
licence and acquisition records, purge gap), process each group through Phases 3–5, retrain on train groups only, calibrate on the calibration groups only, then call the gated evaluators.
