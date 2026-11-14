# Phase 2 canonical master consolidation checklist (Stage 6.1, Task C)

**Status:** PREPARATION ONLY. Neither master file was edited. This checklist proposes what a separate, approved task should add to the
canonical master. Nothing here changes any Phase 2 status.

## 1. Which file is canonical — OPEN ITEM (needs your decision)

| Candidate | Evidence | SHA-256 |
|---|---|---|
| `Pedestrian_Safety_Prediction_Master_Project_File_UPDATED_11_Nov_2026_Phase2_Handoff(2).md` (82,116 bytes) | Exists in the repository root. Its own "Phase 2 Closure Addendum" (line 1799-1801) names exactly this filename as the canonical master and calls `PROJECT_MASTER.md` a supporting draft. | `2bbd8377562aa7f0af0802c0d355dafc8e5c5ca82d1a952eb4fbf6815ff5b8d2` |
| `...Handoff(2)(2).md` (the name given in the Stage 6.1 request) | **Not found.** Searched the repository (Glob, `D:\BTech-Research`) and `Downloads`; no such file. | n/a |
| `PROJECT_MASTER.md` | Supporting consolidation draft (self-declared). | `5c20b0fc1bcfc171c803e081...` (prefix) |
| `...Handoff(2).pre_task10_backup.md` | Byte-for-byte pre-Task-10 backup. Must stay untouched. | not recomputed |

This checklist was therefore prepared against `...Handoff(2).md`. **Before the update task, confirm** whether a `(2)(2)` copy exists elsewhere and,
if so, whether its content differs (compare SHA-256 with the value above). Updating a different file than the one reviewed here would invalidate this checklist.

## 2. What the canonical master currently says (verified by reading it)
- 1,946 lines. Sections 1-16 are the Phase 1 / audit history; section 17 is the Phase 2 continuation (task register 17.4, tracker 17.7);
  the "Phase 2 Closure Addendum" (lines 1795-1946) carries the Task 1-10 status table, limitations and the roadmap.
- Term search results (count in the file): `pedsafety` 0, `agent_state` 0, `schema.json` 0, `Draft 2020-12` 0, `task7_implementation` 0, `docs/phase2` 0,
  `Stage 6` 0, `manifest_eligible` 0, `causal` 0. The master records NONE of the Task 7 implementation, the schema, or the Stage 1-6.1 work.
- Its Phase 2 definition (section 17.1, line 1452): "a controlled audit of the reproduced PCG baseline ... not the implementation phase" for the
  proposed architecture. `pedsafety` does not conflict with that (no YOLO/ByteTrack/GATv2/TCN), but the master never states that the data-foundation
  code exists.

## 3. Numbering conflicts that the update must make explicit (nothing to "fix", only to disambiguate)
1. **Two task series.** Canonical master Tasks 1-10: 1 Git freeze, 2 archival/publication, 3 ADE/FDE reconstruction, 4 oracle audit, 5 window overlap,
   6 corrected protocol, 7 HBS provenance, 8 unit tests, 9 manifest, 10 consolidation. The `docs/phase2/` series: task1 dataset research, task2 feasibility,
   task3 agent-state schema, task4 coordinate systems, task5 trajectory preprocessing, task6 splitting protocol, task7 implementation (Stages 1-6.1).
   Propose naming the second series "Phase 2 design/implementation track (docs/phase2)" and never reusing bare "Task N" for it.
2. **Correction to my earlier audit.** The Phase 2 Exit Audit table labelled "Tasks 1-3" using the `docs/phase2` titles. In the canonical numbering, Tasks 1-3
   are Git freeze, publication and ADE/FDE reconstruction. Evidence for those: HEAD `7a6f56f` equals `origin/reproduction`; `ade_fde_reconstruction.json` is in the
   20/20 manifest; the master records ADE/FDE agreement within 1e-5. I did not re-derive ADE/FDE. The dataset-research/feasibility/schema documents belong to the
   second series.
3. **Two "Phase 2" meanings.** Section 17.1 (audit of the baseline) versus section 15.5 "Phase 2 - Dataset and geometry foundation" (exit criteria: reproducible
   data pipeline, documented units, valid tracks, split independence checks). `pedsafety` addresses part of the latter; the master's later "Phase 3 - Perception
   frontend" (15.5) differs from the closure-addendum roadmap (item 1-2 are dataset and evaluator work). Add a note, do not renumber.

## 4. Proposed updates (location, content, rule)
| # | Location in canonical master | Proposed addition | Rule |
|---|---|---|---|
| 1 | New subsection after line 1819 (end of the Task 1-10 status table), e.g. "Phase 2 Task 7 data-foundation implementation (pedsafety)" | One paragraph + table: package `pedsafety/`, schema `docs/phase2/schemas/agent_state.schema.json` (SHA `784c8ca9...`), Stages 1-6 and 6.1, doc `docs/phase2/task7_implementation.md`, CLI `python -m pedsafety` | Append only. State it is NOT master Task 7 (HBS audit). |
| 2 | Same subsection | Test counts: `tests/task7` **581 passed** (550 at Stage 6 + 31 in Stage 6.1), schema **22 passed**, full `tests/phase2` **33 passed, 1 failed** (known `Interaction.py` defect, unchanged) | Quote exact commands and dates. |
| 3 | Same subsection | Safeguards in force: no metric claim without a manifest passing software checks (reported as `software_manifest_checks_only`, never "verified"); causal vs offline views; whole-group splits with explicit independence basis; no default grouping; HBS refused (one scene, `INSUFFICIENT_GROUPS`) | Do not imply HBS is splittable or metric. |
| 4 | Same subsection | Stage 6.1: output-directory policy (exit code 7, fresh/empty directory only, complete-run contract), `requirements-pedsafety.txt`, `requirements-dev.txt`, version `0.6.1.dev0` | Describe as hardening, not a result. |
| 5 | Closure-addendum "Remaining research limitations" (line 1906) | Add: group independence is a stated claim; no real PIE/inD/TITAN data or calibration run; chronological split/purge not implemented; real-data safeguard #1 (dependency-interval overlap) only moot for whole-group splits; Task 7 deliverables uncommitted | Append to the list. |
| 6 | "Updated Phase 2 tracker" (17.7, line 1761) | Do not edit the old table (it is historical). Add a one-line pointer to the closure addendum, which already supersedes it | Preserve history. |
| 7 | Closure addendum "Task 6" paragraph (line 1821) | Add one sentence: Stage 5 implements the whole-group split/window part only; the non-oracle evaluator, hidden-future test and purge remain unimplemented, so status stays REVISION REQUIRED | No status change. |
| 8 | Closure addendum "Task 7" paragraph (line 1835) | Add one sentence: Stage 6 HBS smoke test reproduced rows/agents/classes/0.5 s spacing and verified nothing metric is produced; status stays PARTIAL | No status change. |
| 9 | After the closure addendum | "Phase 2 exit decision": `PHASE2_EXIT_APPROVED_WITH_LIMITATIONS` per the audit, with the limitations list. Record it only if you approve the audit | Needs your explicit approval. |
| 10 | Section 12 "Open questions" | Leave as is (historical). The closure addendum already lists the unresolved HBS questions | Preserve. |
| 11 | Revision metadata | Date and a one-line change note; save a new pre-update backup of the file first (same convention as `.pre_task10_backup.md`) | Backup before edit. |

## 5. Historical information that must be preserved unchanged
Sections 1-17 and the existing closure addendum text; all historical metrics (best-of-five ADE/FDE as reported, labelled oracle); the 11/1 baseline result;
the `audit-fixes` correction being execution evidence only; the statuses Task 6 REVISION REQUIRED, Task 7 PARTIAL, Task 8 AUDIT COMPLETE (defect documented),
Task 9 AUDIT COMPLETE (20/20); the `pre_task10_backup` file; `PROJECT_MASTER.md` as a supporting draft; the statement that no Phase 3 work has started.

## 6. Other repository facts the update should mention or resolve
- `pedsafety/`, `tests/`, `docs/`, `configs/`, `scripts/` and the Phase 2 reports are untracked, and the 20-artifact `manifest_v2.json` does not cover them.
  The master should say they are working-tree deliverables until committed (a decision for you; this stage did not commit).
- `PROJECT_MASTER.md` still contains the repository-structure and status text from Task 10; keep it as a draft and do not use it as a source of truth.
