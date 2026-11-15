# Phase 8 — Validation report (software verification only; no scientific validation)

## Tests (commands run from the repository root with `C:/Users/revan/anaconda3/envs/pcg-baseline/python.exe`)
| Command | Result |
|---|---|
| `-m pytest tests/phase8 -q -p no:cacheprovider` | 30 passed, 0 failed, 0 skipped |
| `-m pytest tests/phase4 tests/phase5 tests/phase6 tests/phase7 tests/phase8 -q -p no:cacheprovider` | **458 passed, 2 skipped, 0 failed**, exit 0 (the 2 skips are pre-existing Phase 4/5 skips; Phase 2/3 suites were not part of this run) |
Phase 8 tests verify behaviour: the full transition table (all 36 ordered state pairs), invalid-transition rejection, persistence, hysteresis, escalation/de-escalation, deduplication,
cooldown, clearance kinds, invalid/absent/gap handling, identity preservation, simultaneous pairs and alerts, out-of-order and future-frame rejection, prefix (no-lookahead) property,
deterministic replay and input-order independence, schema acceptance and eight tamper rejections, provenance hashes, probability semantics, lead-time blocking, Phase 7 contract
violations, configuration validation, and integration on the real Phase 7 artifacts (cross-check against `risk_arrays.npz`, lifecycle invariants of every logged alert, replay equality).
Bugs found by this verification and fixed before commit: (1) an alert id was kept after clearance so unrelated pre-alert pairs could be "cleared" (904 spurious events on real data);
(2) an idle unobserved pair was moved to INSUFFICIENT_EVIDENCE. No earlier phase file or test was modified.

## Real-data integration run (`experiments/phase8_alerts_pie_video_0004/`, config hash in `run_manifest.json`)
Input: 8233 Phase 7 records (pedestrian–vehicle pairs, 1139 anchor frames) of PIE `set01/video_0004`, regenerated and cross-checked equal to `risk_arrays.npz`. Phase 7 categories:
LOW 6080, MODERATE 513, HIGH 297, CRITICAL 112, INSUFFICIENT_EVIDENCE 1231.
Output: 1200 events (1108 state transitions, 92 updates), **28 alerts** on 10 pedestrians; 8 alerts reached ESCALATED_WARNING; 26 had at least one evidence-loss step. Clearance kinds: 3 RISK_SUBSIDED,
**25 EVIDENCE_LOST**, 0 END_OF_STREAM. Every alert has one activation and one closure, no pair has overlapping alerts, and replay reproduces the event hash.
Interpretation limits: the dominance of EVIDENCE_LOST means most alerts ended because the pair stopped being observed (Phase 5 windows require 1.5 s of future for the pedestrian, and vehicles
leave the context), not because risk fell. The 28 alerts are properties of a heuristic on one in-sample video with perceived tracks; they are **not** detections of verified conflicts and say nothing
about precision, recall, false alarms or generalisation.

## Blocked evaluations
| Item | Status |
|---|---|
| Warning lead time | `BLOCKED_INSUFFICIENT_GROUND_TRUTH` (interface implemented and tested on synthetic verified events only) |
| Precision / recall / false-alarm rate of alerts | not reported: no verified events |
| Calibrated alert probability | not available: Phase 7 calibration blocked, no decision uses a probability |
| Threshold / persistence validation, ROC-style operating-point choice | blocked: engineering defaults only |
| Real-time performance | not claimed: only the state machine was timed in isolation (~2×10⁴ obs/s, one machine) |
| Physical TTC / PET / distance / severity | not supported (image-space pixels only) |

## Next evidence needed
Independent videos with verified conflict/near-conflict event timestamps (also unblocking Phase 5 independent evaluation, Phase 6 supervised training and Phase 7 calibration), then
frozen group-level splits; only then can alert thresholds be tuned on a validation group and lead time / false-alarm rate be measured on held-out groups; an end-to-end latency
measurement on target hardware is a separate requirement for any real-time statement.
