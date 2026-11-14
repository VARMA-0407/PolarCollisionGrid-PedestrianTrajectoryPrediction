# Task 6 — Corrected Validation Protocol Review

## Final status

**REVISION REQUIRED**

The protocol design correctly identifies the principal validity failures, but
it is not implemented, hidden-future-input tested, or associated with
leakage-free metrics. It must not be described as a completed corrected
evaluation.

## Existing protocol summary

The existing report is
`reports/task6_corrected_validation_protocol.json`. It preserves the
historical PCG method-4 context: six observed frames, six prediction frames,
sample size five, and the recorded ADE/FDE values. It explicitly labels the
historical result as oracle-selected and future-information contaminated.

The proposed correction restricts rollout inputs to observed or predicted
states, prohibits future pedestrian and vehicle annotations, replaces
best-of-five selection with deterministic or predeclared aggregation, and
requires a split purge covering the complete 12-frame sequence.

## What is correct

- The historical oracle selection is separated from deployment-time metrics.
- The source audit identifies future pedestrian presence in rollout logic.
- The source audit identifies future vehicle states and presence in method-4
  interaction construction.
- The protocol prohibits future actor lists, states, and future-ground-truth
  sample selection.
- Track birth, death, identity, and propagation policies are explicitly
  required.
- The protocol recognizes the 66 cross-split overlapping window pairs and
  requires a purge policy.
- ADE/FDE, conflict metrics, calibration metrics, alert metrics, latency, and
  FPS are listed as evaluation outputs.
- The report explicitly states that corrected metrics and deployment FPS have
  not been measured.

## What is incomplete

- No corrected evaluation implementation exists in the repository.
- No test has demonstrated that future annotation arrays are inaccessible to
  rollout code.
- No deterministic multimodal selection or aggregation rule has been
  implemented.
- No leakage-free train/validation/test artifacts have been generated.
- The interaction-grid correction is not integrated into the authoritative
  reproduction branch.
- Collision/conflict definitions and coordinate units remain dependent on
  unresolved HBS coordinate semantics.
- No deployment or calibration result exists.

## Supporting source evidence

- `test.py:sample()` receives `true_x_seq` and `true_Pedlist`.
- The observed rollout calls use `true_Pedlist[tstep]` and the predicted
  rollout continues to use future-indexed pedestrian lists.
- Method 4 receives future-indexed vehicle tensors and vehicle lists while
  constructing later interaction inputs.
- `experiments/PCG_T4_5_Epoch_Run/reports/task4_oracle_future_information_audit.json`
  records the future-input and oracle findings.
- `experiments/PCG_T4_5_Epoch_Run/reports/task5_window_overlap_authoritative.json`
  records the temporal-overlap finding.

## Required changes before approval

1. Implement a separate corrected evaluator outside `reproduction`.
2. Make rollout interfaces accept only observed tracker state and model
   predictions; remove future annotation arguments from the production path.
3. Add a test fixture that deliberately withholds future annotations and
   fails if rollout accesses them.
4. Define and implement one non-oracle multimodal reporting rule.
5. Generate split artifacts before window generation and apply a documented
   purge boundary.
6. Define coordinate-dependent conflict metrics only after coordinate
   semantics are established.
7. Record corrected command output, metrics, and environment without
   overwriting historical logs.

## Recommended corrected protocol

Use the existing six-observation/six-prediction horizon only as a historical
comparison point. Construct deployment-feasible actor tracks from observations
available at each step. During rollout, use predicted pedestrian states,
predicted or tracker-propagated vehicle states, and an explicit actor
birth/death policy. Report deterministic ADE/FDE and separately labelled
oracle best-of-N upper-bound values. Create chronological or scene-independent
splits before window generation, then purge enough frames that no complete
12-frame sequence crosses a split boundary.

## Acceptance criteria

- Future pedestrian and vehicle annotations are absent from rollout inputs.
- No future ground truth selects a prediction mode.
- Interaction inputs are observed or predicted states only.
- Split and purge rules are reproducibly materialized.
- Corrected tests pass with future annotations hidden.
- Historical and corrected metrics are stored separately.
- Coordinate assumptions and conflict units are stated in every result.

## Review conclusion

The design is a sound audit specification, not an executed validation. Status
remains **REVISION REQUIRED** until the implementation and hidden-future-input
tests exist.

