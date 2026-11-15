"""Phase 4 Task 3: deterministic, model-agnostic, image-space interaction representation.

Input: the Task 2 CAUSAL image-kinematics sidecar (``causal_kinematics.jsonl``) with its run manifest, and the Phase 3
``agent_states.jsonl`` it was derived from (joined strictly through ``source_line`` for box geometry and source labels).
Output: per-frame typed node tables, complete typed candidate edge sets with image-space relational features, validity
masks, sensitivity flags, a feature dictionary, diagnostics and a manifest.

Everything is image space (``captured_1920x1080``: pixels, pixel/s, s, image-plane radians); every feature is
``physical: false``. No learning, no forecasting, no TTC/PET, no world coordinates, no ego-motion compensation.
Decision record: docs/phase4/PHASE4_TASK3_DECISION_RECORD.md (v1.1.0).
"""

INTERACTION_VERSION = "interaction-representation-0.1.0-phase4-task3"
