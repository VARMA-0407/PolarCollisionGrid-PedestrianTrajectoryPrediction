"""Load a Phase 4 Task 3 run directory and verify it against the Phase 5 input contract (section 12: fail, never repair).

Only the Task 3 outputs are read here: nodes, relation-specific edges, tracks, feature dictionary, resolved config and
run manifest. Nothing upstream (Task 2 offline sidecar, Phase 3 video, PIE annotations) is read by this module.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jsonschema import Draft202012Validator

from ..image_kinematics_run import sha256_file
from .errors import WindowContractError
from .window_config import WindowConfig

REPO = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO / "docs" / "phase4" / "schemas"
TASK3_MANIFEST_VERSION = "phase4-task3-run-manifest-1.0.0"

NODE_FEATURES = ("u_px", "v_px", "vel_u_px_s", "vel_v_px_s", "speed_px_s", "heading_sin", "heading_cos", "box_h_px",
                 "frames_since_prev_obs", "track_age_frames")
NODE_MASKS = ("position_valid", "velocity_valid", "heading_valid", "frames_since_prev_obs_valid")
NODE_BOXES = ("box_x_min_px", "box_y_min_px", "box_x_max_px", "box_y_max_px")
# node feature column -> governing mask column (None = must always be finite)
NODE_FEATURE_MASK = (0, 0, 1, 1, 1, 2, 2, None, 3, None)
NODE_TYPE_NAMES = ("pedestrian", "vehicle", "bicyclist")
EDGE_FEATURES = ("du_px", "dv_px", "sep_px", "bearing_sin", "bearing_cos", "rel_bearing_sin", "rel_bearing_cos", "sep_over_h",
                 "log_h_ratio", "dv_row_px", "dvel_u_px_s", "dvel_v_px_s", "closing_rate_px_s", "overlap_ios")
WORLD_TOKENS = ("world", "metre", "meter", "_m_", "ttc", "pet")


@dataclass
class EdgeSet:
    relation: Tuple[str, str]
    frame: np.ndarray          # int64 (E,)
    edge_index: np.ndarray     # int64 (2, E) global Task 3 node indices (row 0 source j, row 1 destination i)
    edge_attr: np.ndarray      # float32 (E, 14)
    edge_mask: np.ndarray      # bool (E, 14)
    flags: np.ndarray          # bool (E, n_flags)
    flag_names: Tuple[str, ...]
    sha256: str

    @property
    def name(self) -> str:
        return "%s__%s" % self.relation


@dataclass
class Task3Run:
    run_dir: Path
    manifest: Dict[str, Any]
    manifest_sha256: str
    feature_dictionary: Dict[str, Any]
    tracks: List[Dict[str, Any]]
    # node arrays
    frame: np.ndarray
    timestamp: np.ndarray
    track_index: np.ndarray
    node_type: np.ndarray
    source_line: np.ndarray
    features: np.ndarray
    boxes: np.ndarray
    border: np.ndarray
    masks: np.ndarray
    edges: Dict[str, EdgeSet]
    file_hashes: Dict[str, str] = field(default_factory=dict)
    frame_slices: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    node_lookup: Dict[Tuple[int, int], int] = field(default_factory=dict)   # (track_index, frame) -> node index

    @property
    def n_nodes(self) -> int:
        return int(self.frame.shape[0])

    def node_at(self, track_index: int, frame: int) -> Optional[int]:
        return self.node_lookup.get((int(track_index), int(frame)))

    def track_id(self, track_index: int) -> str:
        return self.tracks[int(track_index)]["track_id"]


def _fail(code: str, msg: str):
    raise WindowContractError(code, msg)


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")))


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as exc:
        _fail("TASK3_NPZ_UNREADABLE", "cannot read %s: %s" % (path, exc))


def _expect(arr: np.ndarray, dtype, shape_tail: Tuple[int, ...], n: int, what: str):
    if arr.dtype != np.dtype(dtype) or arr.shape != (n,) + shape_tail:
        _fail("TASK3_ARRAY_LAYOUT", "%s: expected %s %s, found %s %s" % (what, np.dtype(dtype), (n,) + shape_tail, arr.dtype, arr.shape))


def _check_claims(claims: Dict[str, Any]):
    for k, v in claims.items():
        if isinstance(v, bool) and v:
            _fail("TASK3_CLAIM_TRUE", "Task 3 manifest claim %r is true; Phase 5 accepts only runs with every physical claim false" % k)


def load_task3_run(run_dir, cfg: WindowConfig) -> Task3Run:
    run = Path(run_dir)
    mpath = run / "run_manifest.json"
    if not mpath.is_file():
        _fail("TASK3_MISSING_FILE", "missing %s" % mpath)
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except ValueError as exc:
        _fail("TASK3_MANIFEST_UNREADABLE", "%s: %s" % (mpath, exc))
    errs = sorted(_validator("interaction_run_manifest.schema.json").iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if errs:
        _fail("TASK3_MANIFEST_SCHEMA", "run manifest violates its schema: %s" % errs[0].message)
    if manifest.get("manifest_version") != TASK3_MANIFEST_VERSION:
        _fail("TASK3_MANIFEST_VERSION", "unexpected manifest_version %r" % manifest.get("manifest_version"))
    if manifest.get("status") != "completed":
        _fail("TASK3_NOT_COMPLETED", "Task 3 run status is %r" % manifest.get("status"))
    if manifest["inputs"].get("unchanged_after_run") is not True:
        _fail("TASK3_INPUTS_CHANGED", "Task 3 inputs were not confirmed unchanged after the run")
    val = manifest["validation"]
    if val.get("all_input_contracts_passed") is not True or val.get("edges_equal_theoretical_bound") is not True:
        _fail("TASK3_VALIDATION_FLAGS", "Task 3 validation flags are not all true")
    _check_claims(manifest["claims"])
    conv = manifest["conventions"]
    if conv.get("coordinate_system") != cfg.coordinate_system:
        _fail("COORDINATE_SYSTEM_MISMATCH", "Task 3 coordinate_system %r != configured %r" % (conv.get("coordinate_system"), cfg.coordinate_system))
    if manifest["config"]["content"].get("expected_coordinate_system") != cfg.coordinate_system:
        _fail("COORDINATE_SYSTEM_MISMATCH", "Task 3 config expected_coordinate_system != configured coordinate system")

    # every output file: present, size and hash equal to the manifest
    hashes: Dict[str, str] = {}
    for name, rec in manifest["outputs"].items():
        p = run / name
        if not p.is_file():
            _fail("TASK3_MISSING_FILE", "missing output %s" % p)
        if p.stat().st_size != rec["bytes"]:
            _fail("TASK3_OUTPUT_SIZE", "%s size %d != manifest %d" % (name, p.stat().st_size, rec["bytes"]))
        h = sha256_file(p)
        if h != rec["sha256"]:
            _fail("TASK3_OUTPUT_HASH", "%s sha256 %s != manifest %s" % (name, h, rec["sha256"]))
        hashes[name] = h
    for req in ("nodes.npz", "tracks.json", "feature_dictionary.json", "config_resolved.json"):
        if req not in hashes:
            _fail("TASK3_MISSING_FILE", "manifest does not list %s" % req)

    fdict = json.loads((run / "feature_dictionary.json").read_text(encoding="utf-8"))
    errs = list(_validator("interaction_feature_dictionary.schema.json").iter_errors(fdict))
    if errs:
        _fail("TASK3_DICTIONARY_SCHEMA", "feature dictionary violates its schema: %s" % errs[0].message)
    if fdict.get("dictionary_version") != cfg.expected_feature_dictionary_version:
        _fail("TASK3_DICTIONARY_VERSION", "dictionary_version %r != %r" % (fdict.get("dictionary_version"), cfg.expected_feature_dictionary_version))
    if fdict.get("coordinate_system") != cfg.coordinate_system:
        _fail("COORDINATE_SYSTEM_MISMATCH", "feature dictionary coordinate_system %r != %r" % (fdict.get("coordinate_system"), cfg.coordinate_system))
    if fdict.get("physical") is not False or fdict.get("causal_inputs_only") is not True:
        _fail("TASK3_DICTIONARY_FLAGS", "feature dictionary must declare physical=false and causal_inputs_only=true")
    if tuple(f["name"] for f in fdict["node_features"]) != NODE_FEATURES or tuple(f["name"] for f in fdict["node_masks"]) != NODE_MASKS:
        _fail("TASK3_FEATURE_NAMES", "feature dictionary node features/masks differ from the Phase 5 input contract")
    if tuple(f["name"] for f in fdict["edge_features"]) != EDGE_FEATURES:
        _fail("TASK3_FEATURE_NAMES", "feature dictionary edge features differ from the Phase 5 input contract")
    if tuple(fdict.get("node_types", ())) != NODE_TYPE_NAMES:
        _fail("TASK3_NODE_TYPES", "unexpected node types %r" % fdict.get("node_types"))
    flag_names = tuple(f["name"] for f in fdict["edge_flags"])

    # ----- nodes -----
    nz = _load_npz(run / "nodes.npz")
    for k in ("frame", "timestamp", "track_index", "node_type", "source_line", "features", "feature_names", "boxes", "box_names",
              "border", "masks", "mask_names", "node_type_names"):
        if k not in nz:
            _fail("TASK3_ARRAY_LAYOUT", "nodes.npz lacks %r" % k)
    names = [str(x) for x in nz["feature_names"].tolist()] + [str(x) for x in nz["mask_names"].tolist()] + [str(x) for x in nz["box_names"].tolist()]
    if tuple(nz["feature_names"].tolist()) != NODE_FEATURES or tuple(nz["mask_names"].tolist()) != NODE_MASKS \
            or tuple(nz["box_names"].tolist()) != NODE_BOXES or tuple(nz["node_type_names"].tolist()) != NODE_TYPE_NAMES:
        _fail("TASK3_FEATURE_NAMES", "nodes.npz column names differ from the feature dictionary")
    if any(tok in nm.lower() for nm in names for tok in WORLD_TOKENS):
        _fail("WORLD_COORDINATE_PRESENT", "a node column name suggests world/metric content")
    n = int(nz["frame"].shape[0])
    _expect(nz["frame"], np.int64, (), n, "frame")
    _expect(nz["timestamp"], np.float64, (), n, "timestamp")
    _expect(nz["track_index"], np.int32, (), n, "track_index")
    _expect(nz["node_type"], np.int8, (), n, "node_type")
    _expect(nz["source_line"], np.int64, (), n, "source_line")
    _expect(nz["features"], np.float32, (len(NODE_FEATURES),), n, "features")
    _expect(nz["boxes"], np.float32, (4,), n, "boxes")
    _expect(nz["border"], np.bool_, (), n, "border")
    _expect(nz["masks"], np.bool_, (len(NODE_MASKS),), n, "masks")
    frame, ts, tix, ntype, feats, masks = nz["frame"], nz["timestamp"], nz["track_index"], nz["node_type"], nz["features"], nz["masks"]
    if n == 0:
        _fail("TASK3_EMPTY", "Task 3 run has no nodes")
    if np.any(np.diff(frame) < 0):
        _fail("TASK3_NODE_ORDER", "nodes are not ordered by frame")
    if frame.min() < 0:
        _fail("TASK3_FRAME_RANGE", "negative frame index")
    if np.any(np.abs(ts - frame / cfg.source_fps) > cfg.timestamp_tolerance_s):
        bad = int(np.argmax(np.abs(ts - frame / cfg.source_fps)))
        _fail("TIMESTAMP_FPS_MISMATCH", "node %d: timestamp %r != frame/fps %r (source_fps %r)" % (bad, ts[bad], frame[bad] / cfg.source_fps, cfg.source_fps))
    if not np.all(np.isin(ntype, (0, 1, 2))):
        _fail("TASK3_NODE_TYPES", "node_type outside {0,1,2}")
    if not np.all(masks[:, 0]):
        _fail("TASK3_POSITION_INVALID", "a node has position_valid = false (Task 3 creates nodes only with valid positions)")
    for c, mcol in enumerate(NODE_FEATURE_MASK):
        fin = np.isfinite(feats[:, c])
        if mcol is None:
            if not fin.all():
                _fail("TASK3_MASK_MISMATCH", "node feature %s must always be finite" % NODE_FEATURES[c])
        elif not np.array_equal(fin, masks[:, mcol]):
            _fail("TASK3_MASK_MISMATCH", "node feature %s: finite values do not coincide with mask %s" % (NODE_FEATURES[c], NODE_MASKS[mcol]))
    if not np.isfinite(nz["boxes"]).all():
        _fail("TASK3_MASK_MISMATCH", "non-finite box coordinate")

    tracks = json.loads((run / "tracks.json").read_text(encoding="utf-8")).get("tracks")
    if not isinstance(tracks, list) or len(tracks) != int(tix.max()) + 1 or tix.min() < 0:
        _fail("TASK3_TRACKS", "tracks.json count does not equal max(track_index)+1")
    for i, t in enumerate(tracks):
        if t.get("track_index") != i or t.get("node_type") not in NODE_TYPE_NAMES:
            _fail("TASK3_TRACKS", "tracks.json entry %d is inconsistent" % i)
    if len({t["track_id"] for t in tracks}) != len(tracks):
        _fail("TASK3_TRACKS", "duplicate track_id in tracks.json")
    # one node type per track, equal to tracks.json
    track_type = np.array([NODE_TYPE_NAMES.index(t["node_type"]) for t in tracks], np.int8)
    if not np.array_equal(track_type[tix], ntype):
        _fail("TASK3_TRACKS", "a node's type differs from its track's node_type")
    # (track, frame) unique; one timestamp per frame is implied by timestamp == frame / fps
    key = tix.astype(np.int64) * (int(frame.max()) + 1) + frame
    if np.unique(key).shape[0] != n:
        _fail("TASK3_DUPLICATE_NODE", "a track has more than one node in a frame")
    if np.unique(nz["source_line"]).shape[0] != n or nz["source_line"].min() < 1:
        _fail("TASK3_SOURCE_LINE", "source_line values must be unique one-based line numbers")

    frame_slices: Dict[int, Tuple[int, int]] = {}
    change = np.flatnonzero(np.diff(frame)) + 1
    bounds = np.concatenate(([0], change, [n]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        frame_slices[int(frame[a])] = (int(a), int(b))
    lookup = {(int(t), int(f)): i for i, (t, f) in enumerate(zip(tix.tolist(), frame.tolist()))}

    # ----- edges -----
    relations = [tuple(r.split("__")) for r in conv["relations"]]
    edges: Dict[str, EdgeSet] = {}
    for rel in relations:
        name = "%s__%s" % rel
        fname = "edges_%s.npz" % name
        if fname not in hashes:
            _fail("TASK3_MISSING_FILE", "manifest lists relation %s but not %s" % (name, fname))
        ez = _load_npz(run / fname)
        for k in ("relation", "frame", "edge_index", "edge_attr", "edge_mask", "flags", "feature_names", "flag_names"):
            if k not in ez:
                _fail("TASK3_ARRAY_LAYOUT", "%s lacks %r" % (fname, k))
        if tuple(ez["relation"].tolist()) != rel:
            _fail("TASK3_EDGE_RELATION", "%s declares relation %r" % (fname, tuple(ez["relation"].tolist())))
        if tuple(ez["feature_names"].tolist()) != EDGE_FEATURES or tuple(ez["flag_names"].tolist()) != flag_names:
            _fail("TASK3_FEATURE_NAMES", "%s feature/flag names differ from the feature dictionary" % fname)
        e = int(ez["frame"].shape[0])
        _expect(ez["frame"], np.int64, (), e, fname + ".frame")
        if ez["edge_index"].dtype != np.int64 or ez["edge_index"].shape != (2, e):
            _fail("TASK3_ARRAY_LAYOUT", "%s.edge_index layout" % fname)
        _expect(ez["edge_attr"], np.float32, (len(EDGE_FEATURES),), e, fname + ".edge_attr")
        _expect(ez["edge_mask"], np.bool_, (len(EDGE_FEATURES),), e, fname + ".edge_mask")
        _expect(ez["flags"], np.bool_, (len(flag_names),), e, fname + ".flags")
        ei = ez["edge_index"]
        if e:
            if ei.min() < 0 or ei.max() >= n:
                _fail("TASK3_EDGE_ENDPOINT", "%s: endpoint out of range" % fname)
            if np.any(np.diff(ez["frame"]) < 0):
                _fail("TASK3_EDGE_ORDER", "%s: edges not ordered by frame" % fname)
            if not (np.array_equal(frame[ei[0]], ez["frame"]) and np.array_equal(frame[ei[1]], ez["frame"])):
                _fail("TASK3_EDGE_FRAME", "%s: an edge endpoint lies in a different frame from the edge" % fname)
            if not (np.all(ntype[ei[0]] == NODE_TYPE_NAMES.index(rel[0])) and np.all(ntype[ei[1]] == NODE_TYPE_NAMES.index(rel[1]))):
                _fail("TASK3_EDGE_TYPE", "%s: endpoint node types do not match the relation" % fname)
            if np.any(ei[0] == ei[1]):
                _fail("TASK3_EDGE_SELF", "%s: self-loop" % fname)
            if not np.array_equal(np.isfinite(ez["edge_attr"]), ez["edge_mask"]):
                _fail("TASK3_MASK_MISMATCH", "%s: finite edge values do not coincide with edge_mask" % fname)
        edges[name] = EdgeSet(rel, ez["frame"], ei, ez["edge_attr"], ez["edge_mask"], ez["flags"], flag_names, hashes[fname])

    return Task3Run(run, manifest, sha256_file(mpath), fdict, tracks, frame, ts, tix, ntype, nz["source_line"], feats, nz["boxes"],
                    nz["border"], masks, edges, hashes, frame_slices, lookup)


__all__ = ["Task3Run", "EdgeSet", "load_task3_run", "NODE_FEATURES", "NODE_MASKS", "NODE_BOXES", "NODE_TYPE_NAMES", "EDGE_FEATURES",
           "NODE_FEATURE_MASK"]
