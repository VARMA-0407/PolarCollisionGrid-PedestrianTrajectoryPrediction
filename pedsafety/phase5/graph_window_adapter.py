"""Shared history-frame graph bank and per-window / batched graph access.

Storage idea: windows of different target pedestrians share frames (all windows use the same global 10 Hz grid), so
the Task 3 graph of each *history* frame is stored once in a bank and windows reference it by frame. The bank holds
every Task 3 node (all node types, all columns, unchanged values) and every Task 3 edge of the relations present in the
run, for exactly the frames that are a history frame of at least one accepted window. Frames used only as future
targets are not in the bank, and edges at target frames are never loaded.

Access is causal by construction: :meth:`GraphBank.window_graphs` and :func:`collate_windows` read only the frames in
``history_frames[w]`` and assert they are <= the anchor frame.

Context policy: default context node types come from the configuration (pedestrian, vehicle). Optional types
(bicyclist, semantically uncertain) are stored but only included when ``include_optional=True``. A relation is active
only if both its endpoint types are active; V->V or B->B are absent unless Task 3 produced them.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import WindowContractError
from .task3_source import NODE_TYPE_NAMES, Task3Run
from .window_builder import WindowSet
from .window_config import WindowConfig

NODE_ARRAYS = ("task3_node_index", "frame", "timestamp", "track_index", "node_type", "source_line", "features", "boxes", "border",
               "masks", "semantically_uncertain")
EDGE_ARRAYS = ("frame", "edge_index", "edge_attr", "edge_mask", "flags", "task3_edge_index")


@dataclass
class BankEdges:
    relation: Tuple[str, str]
    frame: np.ndarray            # int64 (E,)
    edge_index: np.ndarray       # int64 (2, E) bank node indices (row 0 source j, row 1 destination i)
    edge_attr: np.ndarray        # float32 (E, 14)
    edge_mask: np.ndarray        # bool (E, 14)
    flags: np.ndarray            # bool (E, n_flags)
    task3_edge_index: np.ndarray # int64 (E,) row in the Task 3 edges_<relation>.npz
    frame_ptr: np.ndarray        # int64 (K+1,) edges of bank frame k are [frame_ptr[k], frame_ptr[k+1])


@dataclass
class GraphBank:
    frames: np.ndarray           # int64 (K,) sorted bank frames
    node_ptr: np.ndarray         # int64 (K+1,)
    nodes: Dict[str, np.ndarray] # NODE_ARRAYS, length N_bank
    edges: Dict[str, BankEdges]
    relations: Tuple[Tuple[str, str], ...]
    edge_flag_names: Tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return int(self.nodes["frame"].shape[0])

    def frame_pos(self, frame: int) -> int:
        k = int(np.searchsorted(self.frames, frame))
        if k >= self.frames.shape[0] or self.frames[k] != frame:
            raise WindowContractError("FRAME_NOT_IN_BANK", "frame %d is not a stored history frame" % frame)
        return k

    def active_types(self, cfg: WindowConfig, include_optional: bool) -> Tuple[int, ...]:
        return tuple(NODE_TYPE_NAMES.index(t) for t in cfg.active_context_types(include_optional))

    def active_relations(self, cfg: WindowConfig, include_optional: bool) -> Tuple[str, ...]:
        act = set(cfg.active_context_types(include_optional))
        return tuple("%s__%s" % r for r in self.relations if r[0] in act and r[1] in act)

    def frame_graph(self, frame: int, cfg: WindowConfig, include_optional: bool = False) -> Dict[str, object]:
        """Nodes (bank indices) of active types and active-relation edges at one stored frame."""
        k = self.frame_pos(frame)
        a, b = int(self.node_ptr[k]), int(self.node_ptr[k + 1])
        types = self.active_types(cfg, include_optional)
        idx = np.arange(a, b, dtype=np.int64)
        idx = idx[np.isin(self.nodes["node_type"][a:b], types)]
        ed = {}
        for name in self.active_relations(cfg, include_optional):
            e = self.edges[name]
            s, t = int(e.frame_ptr[k]), int(e.frame_ptr[k + 1])
            ed[name] = np.arange(s, t, dtype=np.int64)
        return {"frame": int(frame), "nodes": idx, "edges": ed}

    def window_graphs(self, ws: WindowSet, w: int, cfg: WindowConfig, include_optional: bool = False) -> List[Dict[str, object]]:
        t0 = int(ws.anchor_frame[w])
        out = []
        for f in ws.history_frames[w].tolist():
            if f > t0:
                raise WindowContractError("FUTURE_FRAME_IN_INPUT", "window %d requests frame %d after its anchor %d" % (w, f, t0))
            out.append(self.frame_graph(int(f), cfg, include_optional))
        return out


def build_graph_bank(run: Task3Run, ws: WindowSet, cfg: WindowConfig) -> GraphBank:
    frames = np.unique(ws.history_frames.ravel()).astype(np.int64)
    sel: List[np.ndarray] = []
    ptr = [0]
    for f in frames.tolist():
        a, b = run.frame_slices[int(f)]                    # every history frame has >= 1 node (the target)
        sel.append(np.arange(a, b, dtype=np.int64))
        ptr.append(ptr[-1] + (b - a))
    t3 = np.concatenate(sel) if sel else np.zeros(0, np.int64)
    uncertain = np.isin(run.node_type[t3], [NODE_TYPE_NAMES.index(t) for t in cfg.semantically_uncertain_node_types])
    nodes = {"task3_node_index": t3, "frame": run.frame[t3], "timestamp": run.timestamp[t3], "track_index": run.track_index[t3],
             "node_type": run.node_type[t3], "source_line": run.source_line[t3], "features": run.features[t3], "boxes": run.boxes[t3],
             "border": run.border[t3], "masks": run.masks[t3], "semantically_uncertain": uncertain}
    remap = np.full(run.n_nodes, -1, np.int64)
    remap[t3] = np.arange(t3.shape[0], dtype=np.int64)
    edges: Dict[str, BankEdges] = {}
    flag_names: Tuple[str, ...] = ()
    for name, e in run.edges.items():
        keep = np.flatnonzero(np.isin(e.frame, frames))
        ei = remap[e.edge_index[:, keep]]
        if ei.size and ei.min() < 0:
            raise WindowContractError("BANK_EDGE_ENDPOINT", "edge endpoint outside the bank in relation %s" % name)
        ef = e.frame[keep]
        fptr = np.searchsorted(ef, frames, side="left")
        fptr = np.concatenate((fptr, [ef.shape[0]])).astype(np.int64)
        edges[name] = BankEdges(e.relation, ef, ei, e.edge_attr[keep], e.edge_mask[keep], e.flags[keep], keep.astype(np.int64), fptr)
        flag_names = e.flag_names
    return GraphBank(frames, np.array(ptr, np.int64), nodes, edges, tuple(run.edges[n].relation for n in run.edges), flag_names)


def window_target_bank_nodes(bank: GraphBank, ws: WindowSet) -> np.ndarray:
    """(W, H) bank node index of the target at each history frame (via the Task 3 node index)."""
    remap = {int(t): i for i, t in enumerate(bank.nodes["task3_node_index"].tolist())}
    out = np.array([[remap[int(x)] for x in row] for row in ws.history_target_task3_node.tolist()], np.int64)
    return out.reshape(ws.history_target_task3_node.shape)


def collate_windows(bank: GraphBank, ws: WindowSet, idx: Sequence[int], cfg: WindowConfig, include_optional: bool = False,
                    max_agents: Optional[int] = None) -> Dict[str, object]:
    """Dense padded batch for development loading tests (numpy; no model, no normalisation).

    Agent slots are per window: slot 0 is the target; other slots are the active-type tracks observed in at least one
    history frame, in Task 3 track order. A slot absent at a history step has ``present = False`` and NaN features with
    False masks (never filled). Edges are COO lists ``(b, h, src_slot, dst_slot)`` per active relation. Future targets
    are returned separately as ``target_future_uv_px`` and never enter the input arrays.
    """
    idx = [int(i) for i in idx]
    H = cfg.history_length
    per: List[Tuple[np.ndarray, Dict[int, int], List[Dict[str, object]]]] = []
    for w in idx:
        g = bank.window_graphs(ws, w, cfg, include_optional)
        tgt = int(ws.target_track_index[w])
        others = sorted({int(t) for gr in g for t in bank.nodes["track_index"][gr["nodes"]].tolist()} - {tgt})
        order = [tgt] + others
        per.append((np.array(order, np.int32), {t: s for s, t in enumerate(order)}, g))
    A = max(len(p[0]) for p in per) if per else 0
    if max_agents is not None and A > max_agents:
        raise WindowContractError("AGENT_OVERFLOW", "a window has %d agents > max_agents %d (no silent truncation)" % (A, max_agents))
    B = len(idx)
    nf = bank.nodes["features"].shape[1]
    feats = np.full((B, H, A, nf), np.nan, np.float32)
    masks = np.zeros((B, H, A, bank.nodes["masks"].shape[1]), bool)
    present = np.zeros((B, H, A), bool)
    border = np.zeros((B, H, A), bool)
    ntype = np.full((B, A), -1, np.int8)
    tix = np.full((B, A), -1, np.int32)
    slot_valid = np.zeros((B, A), bool)
    uncertain = np.zeros((B, A), bool)
    rel_names = bank.active_relations(cfg, include_optional)
    E: Dict[str, Dict[str, List]] = {r: {"bhsd": [], "attr": [], "mask": [], "flags": []} for r in rel_names}
    for b, (order, slot, g) in enumerate(per):
        tix[b, :len(order)] = order
        slot_valid[b, :len(order)] = True
        for h, gr in enumerate(g):
            nodes = gr["nodes"]
            s = np.array([slot[int(t)] for t in bank.nodes["track_index"][nodes].tolist()], np.int64)
            feats[b, h, s] = bank.nodes["features"][nodes]
            masks[b, h, s] = bank.nodes["masks"][nodes]
            present[b, h, s] = True
            border[b, h, s] = bank.nodes["border"][nodes]
            ntype[b, s] = bank.nodes["node_type"][nodes]
            uncertain[b, s] = bank.nodes["semantically_uncertain"][nodes]
            for r, eidx in gr["edges"].items():
                e = bank.edges[r]
                src = bank.nodes["track_index"][e.edge_index[0, eidx]]
                dst = bank.nodes["track_index"][e.edge_index[1, eidx]]
                rows = np.stack([np.full(eidx.shape[0], b), np.full(eidx.shape[0], h),
                                 [slot[int(x)] for x in src.tolist()], [slot[int(x)] for x in dst.tolist()]], axis=1).astype(np.int64) \
                    if eidx.shape[0] else np.zeros((0, 4), np.int64)
                E[r]["bhsd"].append(rows)
                E[r]["attr"].append(e.edge_attr[eidx])
                E[r]["mask"].append(e.edge_mask[eidx])
                E[r]["flags"].append(e.flags[eidx])
        if not present[b, :, 0].all():
            raise WindowContractError("TARGET_MISSING_IN_HISTORY", "target absent from a history frame of window %d" % idx[b])
    edges = {}
    for r, d in E.items():
        nfl = len(bank.edge_flag_names)
        edges[r] = {"bhsd": np.concatenate(d["bhsd"]) if d["bhsd"] else np.zeros((0, 4), np.int64),
                    "edge_attr": np.concatenate(d["attr"]) if d["attr"] else np.zeros((0, 14), np.float32),
                    "edge_mask": np.concatenate(d["mask"]) if d["mask"] else np.zeros((0, 14), bool),
                    "flags": np.concatenate(d["flags"]) if d["flags"] else np.zeros((0, nfl), bool)}
    return {"window_id": ws.window_id[idx], "anchor_frame": ws.anchor_frame[idx], "history_frames": ws.history_frames[idx],
            "node_features": feats, "node_masks": masks, "present": present, "border": border, "node_type": ntype,
            "track_index": tix, "slot_valid": slot_valid, "semantically_uncertain": uncertain, "edges": edges,
            "active_relations": rel_names, "target_slot": 0,
            "target_future_uv_px": ws.target_future_uv_px[idx], "target_future_valid": ws.target_future_valid[idx],
            "future_frames": ws.future_frames[idx]}


__all__ = ["GraphBank", "BankEdges", "build_graph_bank", "window_target_bank_nodes", "collate_windows", "NODE_ARRAYS", "EDGE_ARRAYS"]
