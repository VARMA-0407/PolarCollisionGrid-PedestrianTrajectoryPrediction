"""Deterministic storage of a window dataset.

Files (one directory per run; never overwritten):
    windows.npz              accepted windows: identity, frames, timestamps, target history node refs, future pseudo-targets
    graph_bank.npz           shared Task 3 graphs of every history frame (nodes ``n__*``, edges ``e__<relation>__*``)
    rejected_candidates.npz  every rejected candidate (track index, anchor frame, reason code)
    smoke_manifest.json      PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT listing (no split)
    run_manifest.json        provenance, configuration, counts, hashes, limitations

``.npz`` files are written with fixed member timestamps (Task 3 ``write_npz``), so identical inputs give byte-identical
files. Pickle is never used.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..interaction.storage import load_npz, write_npz
from .errors import WindowContractError
from .graph_window_adapter import BankEdges, GraphBank
from .window_builder import WindowSet

WINDOW_FIELDS = ("window_id", "target_track_index", "target_track_id", "anchor_frame", "history_frames", "history_timestamps",
                 "future_frames", "future_timestamps", "history_target_task3_node", "history_target_source_line", "history_target_uv_px",
                 "target_future_task3_node", "target_future_source_line", "target_future_uv_px", "target_future_box_h_px",
                 "target_future_border", "target_future_valid", "target_border_at_anchor")
REJECT_FIELDS = ("rejected_track_index", "rejected_anchor_frame", "rejected_reason_code")


def write_windows(path: Path, ws: WindowSet, target_bank_node: np.ndarray, video_id: str, group_id: str) -> str:
    arrays = {k: getattr(ws, k) for k in WINDOW_FIELDS}
    arrays["history_target_bank_node"] = target_bank_node
    arrays["video_id"] = np.array([video_id] * ws.n_windows, dtype=str) if ws.n_windows else np.zeros(0, dtype="<U1")
    arrays["group_id"] = np.array([group_id] * ws.n_windows, dtype=str) if ws.n_windows else np.zeros(0, dtype="<U1")
    arrays["video_frame_range"] = np.array([ws.video_first_frame, ws.video_last_frame], np.int64)
    return write_npz(path, arrays)


def write_rejected(path: Path, ws: WindowSet, reasons) -> str:
    arrays = {k: getattr(ws, k) for k in REJECT_FIELDS}
    arrays["reason_names"] = np.array(reasons)
    return write_npz(path, arrays)


def write_bank(path: Path, bank: GraphBank) -> str:
    arrays: Dict[str, np.ndarray] = {"frames": bank.frames, "node_ptr": bank.node_ptr,
                                     "relations": np.array(["%s__%s" % r for r in bank.relations]),
                                     "edge_flag_names": np.array(bank.edge_flag_names)}
    for k, v in bank.nodes.items():
        arrays["n__" + k] = v
    for name, e in bank.edges.items():
        for k in ("frame", "edge_index", "edge_attr", "edge_mask", "flags", "task3_edge_index", "frame_ptr"):
            arrays["e__%s__%s" % (name, k)] = getattr(e, k)
    return write_npz(path, arrays)


def read_dataset(run_dir) -> Tuple[WindowSet, GraphBank, Dict[str, np.ndarray]]:
    d = Path(run_dir)
    for f in ("windows.npz", "graph_bank.npz", "rejected_candidates.npz"):
        if not (d / f).is_file():
            raise WindowContractError("DATASET_MISSING_FILE", "missing %s" % (d / f))
    w = load_npz(d / "windows.npz")
    r = load_npz(d / "rejected_candidates.npz")
    g = load_npz(d / "graph_bank.npz")
    vr = w["video_frame_range"]
    ws = WindowSet(**{k: w[k] for k in WINDOW_FIELDS}, **{k: r[k] for k in REJECT_FIELDS},
                   video_first_frame=int(vr[0]), video_last_frame=int(vr[1]))
    rels = [str(x) for x in g["relations"].tolist()]
    nodes = {k[3:]: v for k, v in g.items() if k.startswith("n__")}
    edges = {}
    for name in rels:
        p = "e__%s__" % name
        edges[name] = BankEdges(tuple(name.split("__")), g[p + "frame"], g[p + "edge_index"], g[p + "edge_attr"], g[p + "edge_mask"],
                                g[p + "flags"], g[p + "task3_edge_index"], g[p + "frame_ptr"])
    bank = GraphBank(g["frames"], g["node_ptr"], nodes, edges, tuple(tuple(n.split("__")) for n in rels),
                     tuple(str(x) for x in g["edge_flag_names"].tolist()))
    extra = {"history_target_bank_node": w["history_target_bank_node"], "video_id": w["video_id"], "group_id": w["group_id"],
             "reason_names": r["reason_names"]}
    return ws, bank, extra


__all__ = ["write_windows", "write_rejected", "write_bank", "read_dataset", "WINDOW_FIELDS", "REJECT_FIELDS"]
