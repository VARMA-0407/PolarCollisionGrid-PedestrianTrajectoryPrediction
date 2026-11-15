"""P5.7.6: is scene-diverse, annotation-verified data available for an independent evaluation?

A read-only scan of the repository (and the local pilot input folder) that reports what exists and compares it with the explicit
requirements. It never fabricates an evaluation: unless every requirement is met the status is ``BLOCKED_PENDING_SCENE_DIVERSE_DATA``.
Pure standard library (no torch / numpy).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_BLOCKED = "BLOCKED_PENDING_SCENE_DIVERSE_DATA"
STATUS_READY = "READY_FOR_INDEPENDENT_EVALUATION_REVIEW"
MIN_INDEPENDENT_GROUPS = 3          # one train group, one validation group, one test group (proposal in PHASE5_IMPLEMENTATION_ROADMAP P5.2)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}

REQUIREMENTS = [
    "at least %d independent scene groups (proposal: PIE recording sets; set independence semantics verified and documented), each processed through Phase 3 perception -> Phase 4 Task 2 -> Phase 4 Task 3 and the P5.3 window generator" % MIN_INDEPENDENT_GROUPS,
    "per-video annotation XML (e.g. PIE video_XXXX_annt.xml) available locally, with a verified pedestrian identity mapping between annotations and perceived tracks and verified frame/timestamp alignment (needed for the ANNOTATED_TARGET mode; never invented)",
    "annotation validity checks (occlusion, out-of-frame, missing boxes) and independent future targets kept separate from perceived inputs",
    "archived dataset licence / terms for videos, annotations and derived data, and a dataset manifest with archive hashes",
    "a group-level split manifest (whole groups per partition, purge >= the dependency interval) frozen before training; train-only scaler statistics fitted on the training groups only",
    "validation-group-only model / checkpoint / hyperparameter selection and a single locked test evaluation, with cluster-bootstrap intervals over tracks and groups",
    "Phase 3 detection/tracking quality measured against the annotations (Phase 3 Task 4 evaluator run on real data) so that target reliability is known",
]


def _scan(root: Path, patterns: List[str], skip_top: Optional[set] = None) -> List[str]:
    found = []
    for pat in patterns:
        for p in root.rglob(pat):
            rel = p.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts) or (skip_top and rel.parts and rel.parts[0] in skip_top):
                continue
            found.append(rel.as_posix())
    return sorted(set(found))


def assess_independent_data(repo_root) -> Dict[str, Any]:
    root = Path(repo_root)
    sequences = set()
    task3_runs = []
    for run in sorted((root / "experiments").glob("phase4_task3_pie_video_*/*/primary")):
        tracks = run / "tracks.json"
        if not tracks.is_file():
            continue
        ids = {t["track_id"].rsplit(":", 1)[0] for t in json.loads(tracks.read_text(encoding="utf-8")).get("tracks", [])}
        sequences |= ids
        task3_runs.append({"run": run.relative_to(root).as_posix(), "sequence_ids": sorted(ids)})
    window_runs = []
    for run in sorted((root / "experiments").glob("phase5_task3_windows_*/*/primary")):
        mf = run / "run_manifest.json"
        if mf.is_file():
            m = json.loads(mf.read_text(encoding="utf-8"))
            window_runs.append({"run": run.relative_to(root).as_posix(), "label": m.get("label"), "videos": m.get("counts", {}).get("video_count"),
                                "scene_groups": m.get("counts", {}).get("scene_group_count"), "windows": m.get("counts", {}).get("accepted_windows")})
    annotations = _scan(root, ["*annt*.xml", "*annotation*.xml"], skip_top={"experiments", "outputs"})
    videos = _scan(root, ["*.mp4", "*.avi", "*.mov", "*.mkv"], skip_top={"outputs"})
    groups = sorted(sequences)
    checks = {
        "independent_scene_groups_processed": {"found": len(groups), "required": MIN_INDEPENDENT_GROUPS, "met": len(groups) >= MIN_INDEPENDENT_GROUPS, "sequence_ids": groups},
        "annotation_xml_available_locally": {"found": len(annotations), "required": MIN_INDEPENDENT_GROUPS, "met": len(annotations) >= MIN_INDEPENDENT_GROUPS, "files": annotations},
        "verified_identity_mapping_and_alignment": {"met": False, "reason": "no annotation file is available locally, so no identity mapping or frame alignment can be verified"},
        "group_level_split_manifest": {"met": False, "reason": "only one group exists; the pilot window manifests carry the PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT label and no split"},
    }
    ready = all(c["met"] for c in checks.values())
    return {"task": "P5.7.6 independent evaluation availability check", "status": STATUS_READY if ready else STATUS_BLOCKED,
            "independent_evaluation_performed": False, "fabricated_results": False,
            "found": {"phase4_task3_runs": task3_runs, "distinct_sequences": groups, "phase5_window_runs": window_runs, "annotation_xml_files": annotations, "local_video_files": videos},
            "checks": checks, "requirements": REQUIREMENTS,
            "consequence": "No independent train/validation/test evaluation, generalisation estimate or cross-scene claim can be produced. All current P5.7 results are PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT and in-sample. Other implementation tasks are not blocked.",
            "next_actions": ["archive the PIE licence terms", "choose the PIE subset (>= %d independent sets) and acquire the videos and annotation XML" % MIN_INDEPENDENT_GROUPS,
                             "run Phase 3 perception (Colab T4) per video, then Phase 4 Task 2 / Task 3 and the P5.3 window generator per video",
                             "implement the ANNOTATED_TARGET adapter with a verified identity mapping", "create and freeze the group-level split manifest",
                             "re-run P5.7.3-P5.7.5 on the training groups and evaluate once on the held-out groups"]}


__all__ = ["assess_independent_data", "STATUS_BLOCKED", "STATUS_READY", "REQUIREMENTS", "MIN_INDEPENDENT_GROUPS"]
