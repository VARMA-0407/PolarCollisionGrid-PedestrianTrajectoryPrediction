# Phase 2 Reproducibility Report

## Scope and preservation

The Phase 2 validation was run on the `reproduction` branch using Python 3.9.25
and pytest 8.4.2. The original PCG source, datasets, checkpoints, and
authoritative prediction results were not modified. Training and the original
evaluation were not rerun.

## Baseline implementation result

Command:

```text
C:\Users\revan\anaconda3\envs\pcg-baseline\python.exe -m pytest tests\phase2 -v
```

Result: **11 passed, 1 failed, 0 skipped**.

The failing test was
`test_ttc_occupancy_mask_collapses_agent_dimension`. The baseline
`getInteractionGridMask` implementation initializes `frame_mask` in the
`is_occupancy=True` branch but initializes `TTC_mask` only in the other
branch. It nevertheless returns both values unconditionally. This raises:

```text
UnboundLocalError: local variable 'TTC_mask' referenced before assignment
```

The test expectation is valid because callers unpack two return values, and
`getSequenceInteractionGridMask` unconditionally converts both returned arrays
to tensors.

Raw baseline output: [task8_pytest.log](../logs/task8_pytest.log).

## Isolated audit-fix result

An uncommitted `audit-fixes` branch was used only to validate the minimal
correction: initialize `TTC_mask` with shape
`(num_agent, num_agent_other, num_TTC * num_sector)` in the occupancy branch.

Result on that branch: **12 passed, 0 failed, 0 skipped**.

Raw corrected output:
[task8_pytest_corrected_audit_fixes.log](../logs/task8_pytest_corrected_audit_fixes.log).

The correction was reverted before returning to `reproduction`; no source
change remains on the reproduction branch, and no commit or push was made.

## Existing evidence

- Independent ADE/FDE reconstruction matches the recorded values within `1e-5`.
- The existing audits document oracle best-of-five selection and future-input
  leakage.
- Coordinate units and camera calibration remain unverified.
- Cross-split temporal overlap remains documented by the authoritative audit.

## Final status and blockers

The reproducibility deliverables are complete, but baseline validation remains
blocked by the confirmed occupancy-branch implementation defect. The
`audit-fixes` branch demonstrates that the defect is locally correctable, but
the correction is intentionally not part of `reproduction`.
