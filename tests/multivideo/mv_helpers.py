"""Synthetic fixtures for the multi-video orchestration tests (no PIE data, no video decoding, no YOLO).

``write_phase3_dir`` writes a Phase 3 output folder (agent_states.jsonl, tracked_agents.jsonl, run_manifest.json) with the real field
sets, for one synthetic scene whose motion depends on ``seed``. ``make_pie_tree`` writes a fake PIE folder (dummy MP4 bytes, annotation XML
names, a ``pie_data.py`` holding a split dictionary literal) for the inventory / split tests.
"""

import json
from pathlib import Path

from task2_helpers import rec

N_FRAMES = 150
FPS = 30.0
SEMANTIC = {"pedestrian": "pedestrian", "vehicle": "car"}


def scene(seq: str, seed: int):
    out = []
    a = 1.0 + 0.25 * seed
    for f in range(N_FRAMES):
        out.append(rec(track="1", frame=f, fx=100.0 + 2.0 * a * f, fy=600.0, seq=seq))
        out.append(rec(track="2", frame=f, fx=300.0 + 5 * seed, fy=500.0 + a * f, h=70.0, seq=seq))
        out.append(rec(track="3", frame=f, fx=900.0 - 1.5 * a * f, fy=650.0, h=80.0, seq=seq))
        out.append(rec(track="4", frame=f, fx=700.0 + 3.0 * f, fy=640.0 + 0.5 * a * f, h=150.0, w=240.0, cls="vehicle", source_label="car", seq=seq))
        if f >= 20:
            out.append(rec(track="5", frame=f, fx=1000.0 - 2.0 * f, fy=660.0, h=160.0, w=260.0, cls="vehicle", source_label="truck", seq=seq))
    out.sort(key=lambda r: (r["frame_index"], r["track_id"]))
    return out


def write_phase3_dir(d: Path, key: str, seed: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    seq = "pie:%s" % key
    recs = scene(seq, seed)
    for r in recs:
        r["source_dataset"] = "PIE"
    with open(d / "agent_states.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    with open(d / "tracked_agents.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            sem = r["source_label"] if r["agent_class"] == "vehicle" else "pedestrian"
            fh.write(json.dumps({"video_id": key, "frame_index": r["frame_index"], "timestamp_s": r["frame_index"] / FPS, "timestamp_source": "frame_index_over_fps",
                                 "track_id": int(r["track_id"].rsplit(":", 1)[1]), "semantic_class": sem, "confidence": 0.9, "bbox_xyxy": r["bbox_xyxy"],
                                 "image_width": 1920, "image_height": 1080, "coordinate_system": "captured_1920x1080", "coordinate_space": "image_pixels",
                                 "is_predicted": False, "detection_validity": True, "tracking_validity": True}, separators=(",", ":")) + "\n")
    man = {"manifest_version": "perception-run-manifest-0.1.0-task1", "video": key + ".mp4", "backend": "yolo", "is_mock": False,
           "config": {"detector": {"label_map": {"person": "pedestrian", "car": "car", "truck": "truck"}}},
           "components": {"frame_source": {"name": "synthetic", "parameters": {"fps": FPS, "fps_origin": "test", "timestamp_source": "frame_index_over_fps",
                                                                              "width": 1920, "height": 1080}}},
           "video_stats": {"frames_delivered": N_FRAMES}, "results": {"frames_seen": N_FRAMES, "frames_processed": N_FRAMES, "frames_skipped": 0, "detections": len(recs),
                                                                       "track_rows": len(recs), "unique_track_ids": 5, "issues_by_code": {}, "first_frame_index": 0,
                                                                       "last_frame_index": N_FRAMES - 1},
           "agent_states": {"records": len(recs), "source_dataset": "PIE", "scene_id": "pie_" + key, "is_mock": False}, "tracker_class_flips": 0, "git_commit": None,
           "environment": {"cuda": {"gpus": None}}, "stage_latency_ms_in_pipeline_run": {}, "wall_time_s": 1.0,
           "outputs": ["agent_states.jsonl", "run_manifest.json", "tracked_agents.jsonl"], "world_coordinates": "none: outputs are image-space pixels only"}
    (d / "run_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return d


PIE_DATA_PY = '''
class PIE(object):
    def _get_image_set_ids(self, image_set):
        image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03', 'set04', 'set05', 'set06']}
        return image_set_nums[image_set]
'''


def make_pie_tree(root: Path, layout=None, with_annotations=True, with_pie_data=True) -> Path:
    """layout: {set: n_videos}. Dummy MP4 content is unique per file."""
    layout = layout or {"set01": 2, "set02": 1, "set03": 1, "set04": 1, "set05": 1, "set06": 1}
    pie = root / "pie"
    for s, n in layout.items():
        d = pie / "PIE_clips" / s
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            (d / ("video_%04d.mp4" % i)).write_bytes(("%s-%d-fake-mp4" % (s, i)).encode() * 10)
            if with_annotations:
                a = pie / "annotations" / "annotations" / s
                a.mkdir(parents=True, exist_ok=True)
                (a / ("video_%04d_annt.xml" % i)).write_text("<annotations/>", encoding="utf-8")
    if with_pie_data:
        (pie / "utilities").mkdir(parents=True, exist_ok=True)
        (pie / "utilities" / "pie_data.py").write_text(PIE_DATA_PY, encoding="utf-8")
    return pie
