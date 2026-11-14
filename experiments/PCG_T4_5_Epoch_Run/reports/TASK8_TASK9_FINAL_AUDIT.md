# Phase 2 Task 8 and Task 9 Final Audit

## Baseline reproduction evidence

The authoritative branch is `reproduction`. The original Phase 2 pytest run
collected 12 tests and produced **11 passed, 1 failed, 0 skipped**. The failing
test is `test_ttc_occupancy_mask_collapses_agent_dimension`.

The failure is a confirmed baseline implementation defect in
`utils/Interaction.py`: `getInteractionGridMask()` returns
`(frame_mask, TTC_mask)` unconditionally, but the `is_occupancy=True` branch
does not initialize `TTC_mask`. The observed exception is
`UnboundLocalError: local variable 'TTC_mask' referenced before assignment`.
The callers and sequence wrapper expect both return values, so the test is
valid and was not removed or weakened.

## Isolated correction evidence

The minimal correction was tested separately on `audit-fixes`, producing
**12 passed, 0 failed, 0 skipped**. The correction initialized `TTC_mask` in
the occupancy branch. The branch has no commits ahead of `reproduction`, so
the correction is preserved only in execution evidence, not as a committed
branch change. No correction remains in the `reproduction` source.

## Requested filename reconciliation

The original and canonical evidence are both retained:

- `reports/task8_test_report.json`
- `reports/task8_unit_test_report.json` (byte-identical canonical copy)
- `logs/task8_pytest.log`
- `logs/task8_pytest_output.txt` (byte-identical canonical copy)
- `logs/task8_pytest_corrected_audit_fixes.log`
- `reports/PHASE2_REPRODUCIBILITY_REPORT.md`
- `manifest_v2.json`
- `scripts/verify_phase2_manifest.py`
- `logs/task9_manifest_verification.log`
- `logs/task9_manifest_verification.txt`

No existing logs or reports were deleted or overwritten.

## Manifest integrity evidence

The verifier checks artifact existence, byte sizes, and SHA256 hashes, reports
PASS/FAIL, and returns nonzero on failure. The final manifest verification
checked **20 artifacts**, with zero missing files and zero hash mismatches.
The manifest is excluded from its own hash list.

## Source and Git integrity

Final branch: `reproduction`.

- `git diff --exit-code -- utils/Interaction.py`: passed.
- `git diff --cached --exit-code -- utils/Interaction.py`: passed.
- No tracked source changes were introduced.
- No commit or push was made.
- `audit-fixes` remains separate and has no committed correction.

## Outstanding limitations

- Baseline reproduction does not fully pass because of the confirmed occupancy
  TTC-mask defect.
- The isolated correction was not applied to the authoritative branch.
- HBS coordinate units and camera calibration remain unverified.
- Existing audits retain oracle/future-information leakage and temporal-window
  overlap limitations.
- Phase 3 must not begin from this audit.
