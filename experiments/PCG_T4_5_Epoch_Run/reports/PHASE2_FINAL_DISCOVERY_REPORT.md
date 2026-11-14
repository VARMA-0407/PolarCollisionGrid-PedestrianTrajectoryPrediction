# Phase 2 Final Discovery Report

## Repository state

- Branch: `reproduction`
- Tracking: `reproduction...origin/reproduction`
- Existing branches: `reproduction`, `audit-fixes`, `baseline-original`, `master`
- Tracked source changes: none
- Staged source changes: none
- Untracked work: prior Phase 2 evidence and audit deliverables under
  `experiments/PCG_T4_5_Epoch_Run/`, `scripts/`, and `tests/`
- Commits and pushes: none

## Master-file discovery

The repository contains no separate project master, thesis, proposal, or
consolidated research-history document in the working tree or recent Git
history. `README.md` is the repository software README; it describes the
published PCG repository and setup, but does not contain the full project
history, Phase 2 evidence, or research-management state required by the
master execution prompt. It is therefore preserved unchanged rather than
silently treated as the research master.

Because no prior master file exists, `PROJECT_MASTER.md` is established as the
single consolidated project master by this audit. There is no older master
backup to preserve; that absence is recorded in the Task 10 report.

## Existing evidence

The repository contains the Phase 2 reports, logs, reconstruction result,
source-audit extracts, canonical Task 8/9 deliverables, and the 20-artifact
manifest. The final Task 8/9 gate has already been completed:

- baseline tests: 11 passed, 1 failed, 0 skipped;
- isolated `audit-fixes` tests: 12 passed, 0 failed, 0 skipped;
- final manifest: 20 declared and 20 checked, status PASS;
- `utils/Interaction.py` source integrity checks: passed.

## Conflicting or incomplete evidence

1. The repository README describes HBS and the original software repository,
   but it does not establish coordinate units, calibration, provenance, or
   licensing.
2. The DataLoader assumes HBS at 2 FPS and a 0.5-second timestamp interval;
   this is implementation evidence, not proof of original video FPS.
3. Task 5 contains both pickle-reconstructed and CSV-reconstructed counts.
   Both are retained and labelled by source rather than merged.
4. Task 6 identifies future-information leakage and oracle selection but has
   not been implemented or executed as a corrected evaluation.
5. Task 7 evidence finds no calibration, homography, video, or dataset
   metadata file in the repository.

## Planned execution sequence

1. Review and classify Task 6 without claiming implementation.
2. Review and classify Task 7 with explicit evidence gaps.
3. Establish one consolidated `PROJECT_MASTER.md` because no prior master
   exists.
4. Produce a dependency-ordered next-implementation roadmap only.
5. Preserve the authoritative `reproduction` implementation and stop for
   user approval.

