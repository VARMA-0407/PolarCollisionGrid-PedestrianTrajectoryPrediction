"""Phase 5 P5.3: pilot trajectory-window generation (image space, causal inputs only, no model).

Modules:
    window_config        strict window configuration (H = 6, F = 15, 10 Hz from 30 fps)
    errors               stable error codes
    task3_source         load and verify a Phase 4 Task 3 run against the Phase 5 input contract
    upstream             verify the Phase 3 -> Task 2 -> Task 3 provenance chain and the source-line join
    trajectory_targets   target-source modes (PILOT_SMOKE implemented; ANNOTATED_TARGET refused)
    window_builder       candidate enumeration, strict target policy, rejection reasons
    graph_window_adapter shared history-frame graph bank and per-window / batched graph access
    window_storage       deterministic .npz/.json storage
    window_manifest      run manifest and git provenance
    window_validation    independent validation of a stored window dataset

Nothing here trains, predicts or evaluates a model, and nothing is metric or physical.
"""

PHASE5_WINDOWS_VERSION = "phase5-window-generation-0.1.0-p5.3"
