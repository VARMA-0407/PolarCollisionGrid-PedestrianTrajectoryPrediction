"""Configurable CSV adapter.

Everything dataset-specific (column names, constants, fps, coordinate
semantics/unit) is declared in :class:`CsvAdapterConfig`; nothing is
guessed. In particular, if position columns are mapped the config must say
what space they are in (``position_semantics``), so pixels can never be
mistaken for metres downstream.

Not done here (Stage 2): bbox ordering, duplicate/non-monotonic timestamps,
class mapping, schema conversion.
"""

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from ..errors import ConfigError, ParseError
from .base import AnnotationAdapter, IssueCode, ParseIssue, ParseResult, RawObservation

CANONICAL_COLUMNS = frozenset(
    [
        "sequence_id", "scene_id", "track_id", "frame_index", "timestamp", "label",
        "x", "y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
        "vel_x", "vel_y", "confidence",
    ]
)
_BBOX = ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max")
PHYSICAL_UNITS = frozenset(["m", "cm", "mm", "ft"])
ALLOWED_POSITION_UNITS = frozenset(["pixel"]) | PHYSICAL_UNITS

# Closed vocabulary for position_semantics. These are DECLARATIONS made by
# the person configuring the adapter; none of them asserts calibration. The
# decision whether a world position is valid (world_position_valid) belongs
# to the Stage 3 calibration gate, never to this adapter.
SEMANTICS_IMAGE_PIXEL = "image_pixel"          # unit must be 'pixel'
SEMANTICS_WORLD = "world_coordinate"           # physical unit + named frame required
SEMANTICS_UNVERIFIED = "unverified_planar"     # planar values, scale unknown: NO unit allowed
POSITION_SEMANTICS = frozenset([SEMANTICS_IMAGE_PIXEL, SEMANTICS_WORLD, SEMANTICS_UNVERIFIED])
_TIME_SCALE = {"s": 1.0, "ms": 1e-3}


@dataclass(frozen=True)
class CsvAdapterConfig:
    """``columns`` maps canonical names -> CSV header names.

    Required: ``track_id`` and ``frame_index``. Sequence and scene ids come
    from a column OR the constant ``sequence_id``/``scene_id`` (exactly one).
    Time comes from a ``timestamp`` column (unit ``timestamp_unit``) or from
    ``frame_index / fps``. ``x``/``y`` need ``position_semantics``.

    Position declaration rules (checked at construction):

    * ``image_pixel``      -> ``position_unit`` must be ``pixel``.
    * ``world_coordinate`` -> ``position_unit`` must be one of m/cm/mm/ft and
      ``position_coordinate_system`` must name the world frame. This is a
      declaration only; whether a world position is *valid* is decided by the
      Stage 3 calibration gate.
    * ``unverified_planar`` -> ``position_unit`` must be None (no unit may be
      silently assigned to values of unknown scale).
    * With no position/bbox columns mapped, none of the three may be given.

    No two canonical fields may map to the same CSV header.
    """

    source_dataset: str
    columns: Mapping[str, str]
    sequence_id: Optional[str] = None
    scene_id: Optional[str] = None
    fps: Optional[float] = None
    timestamp_unit: str = "s"
    position_semantics: Optional[str] = None
    position_unit: Optional[str] = None
    position_coordinate_system: Optional[str] = None
    delimiter: str = ","
    strict: bool = False

    def __post_init__(self):
        c = self.columns
        if not self.source_dataset or not self.source_dataset.strip():
            raise ConfigError("source_dataset must be a non-empty string")
        unknown = sorted(set(c) - CANONICAL_COLUMNS)
        if unknown:
            raise ConfigError("unknown canonical column name(s) %s; allowed: %s" % (unknown, sorted(CANONICAL_COLUMNS)))
        bad_headers = sorted(k for k, v in c.items() if not isinstance(v, str) or not v.strip())
        if bad_headers:
            raise ConfigError("CSV header name for canonical field(s) %s must be a non-empty string" % bad_headers)
        by_header: Dict[str, List[str]] = {}
        for canon, header in c.items():
            by_header.setdefault(header, []).append(canon)
        conflicts = {h: sorted(fs) for h, fs in by_header.items() if len(fs) > 1}
        if conflicts:
            detail = "; ".join("CSV header %r is mapped by %s" % (h, fs) for h, fs in sorted(conflicts.items()))
            raise ConfigError("each canonical field needs its own CSV header: " + detail)
        for k in ("track_id", "frame_index"):
            if k not in c:
                raise ConfigError("columns must map required field '%s'" % k)
        for name, const in (("sequence_id", self.sequence_id), ("scene_id", self.scene_id)):
            if (name in c) == (const is not None):
                raise ConfigError("exactly one of a '%s' column mapping or a constant %s must be given" % (name, name))
            if const is not None and not const.strip():
                raise ConfigError("constant %s must be non-empty" % name)
        if self.timestamp_unit not in _TIME_SCALE:
            raise ConfigError("timestamp_unit must be one of %s, got %r" % (sorted(_TIME_SCALE), self.timestamp_unit))
        if "timestamp" not in c:
            if self.fps is None:
                raise ConfigError("no 'timestamp' column: fps is required to derive time as frame_index/fps")
        if self.fps is not None and not (math.isfinite(self.fps) and self.fps > 0):
            raise ConfigError("fps must be a positive finite number, got %r" % (self.fps,))
        if ("x" in c) != ("y" in c):
            raise ConfigError("map both 'x' and 'y' or neither")
        present = [k for k in _BBOX if k in c]
        if present and len(present) != 4:
            raise ConfigError("bbox needs all of %s; only %s mapped" % (list(_BBOX), present))
        self._check_position_declaration(("x" in c) or bool(present))
        if not self.delimiter or len(self.delimiter) != 1:
            raise ConfigError("delimiter must be a single character")

    def _check_position_declaration(self, has_position_columns: bool) -> None:
        sem, unit, cs = self.position_semantics, self.position_unit, self.position_coordinate_system
        if unit is not None and unit not in ALLOWED_POSITION_UNITS:
            raise ConfigError("position_unit %r not in %s" % (unit, sorted(ALLOWED_POSITION_UNITS)))
        if cs is not None and not str(cs).strip():
            raise ConfigError("position_coordinate_system must be non-empty when given")
        if not has_position_columns:
            if sem is not None or unit is not None or cs is not None:
                raise ConfigError(
                    "position_semantics/position_unit/position_coordinate_system are declared but no "
                    "position columns (x/y or bbox) are mapped; remove them or map the columns"
                )
            return
        if not sem:
            raise ConfigError(
                "position columns are mapped but position_semantics is not declared; choose one of %s "
                "(units are never assumed)" % sorted(POSITION_SEMANTICS)
            )
        if sem not in POSITION_SEMANTICS:
            raise ConfigError("position_semantics %r is not one of %s" % (sem, sorted(POSITION_SEMANTICS)))
        if sem == SEMANTICS_IMAGE_PIXEL:
            if unit != "pixel":
                raise ConfigError(
                    "position_semantics 'image_pixel' requires position_unit 'pixel', got %r" % (unit,)
                )
        elif sem == SEMANTICS_WORLD:
            if unit not in PHYSICAL_UNITS:
                raise ConfigError(
                    "position_semantics 'world_coordinate' requires a physical position_unit from %s, got %r "
                    "('pixel' is not a world unit)" % (sorted(PHYSICAL_UNITS), unit)
                )
            if not cs:
                raise ConfigError(
                    "position_semantics 'world_coordinate' requires position_coordinate_system naming the world frame"
                )
        else:  # SEMANTICS_UNVERIFIED
            if unit is not None:
                raise ConfigError(
                    "position_semantics 'unverified_planar' must not be given a position_unit (got %r): "
                    "no unit may be assigned to values of unverified scale" % (unit,)
                )


class CsvAnnotationAdapter(AnnotationAdapter):
    def __init__(self, path: Union[str, Path], config: CsvAdapterConfig):
        self.path = Path(path)
        self.config = config

    @property
    def name(self) -> str:
        return "csv:%s" % self.config.source_dataset

    # ------------------------------------------------------------------ parse
    def parse(self) -> ParseResult:
        cfg = self.config
        result = ParseResult()
        try:
            fh = open(self.path, "r", newline="", encoding="utf-8-sig")
        except OSError as exc:
            raise ConfigError("cannot open %s: %s" % (self.path, exc))
        with fh:
            reader = csv.DictReader(fh, delimiter=cfg.delimiter)
            try:
                header = reader.fieldnames
                if not header:
                    raise ConfigError("%s is empty or has no header row" % self.path)
                self._check_header(header)
                mapped = set(cfg.columns.values())
                extra_cols = [h for h in header if h not in mapped]
                for row in reader:
                    result.rows_read += 1
                    line = reader.line_num
                    obs, issues = self._parse_row(row, line, extra_cols)
                    if issues:
                        result.issues.extend(issues)
                        if cfg.strict:
                            raise ParseError(issues)
                    else:
                        result.observations.append(obs)
            except UnicodeDecodeError as exc:
                raise ConfigError("%s is not valid UTF-8: %s" % (self.path, exc))
            except csv.Error as exc:
                raise ConfigError("%s is not well-formed CSV near line %d: %s" % (self.path, reader.line_num, exc))
        return result

    def _check_header(self, header: List[str]) -> None:
        dupes = sorted({h for h in header if header.count(h) > 1})
        if dupes:
            raise ConfigError("duplicate header name(s) %s in %s; DictReader would silently overwrite them" % (dupes, self.path))
        missing = sorted(h for h in set(self.config.columns.values()) if h not in header)
        if missing:
            raise ConfigError("mapped column(s) %s not found in %s; available columns: %s" % (missing, self.path, header))

    # -------------------------------------------------------------- row logic
    def _parse_row(self, row: Dict, line: int, extra_cols: List[str]):
        cfg, c = self.config, self.config.columns
        issues: List[ParseIssue] = []

        if None in row:
            issues.append(ParseIssue(line, None, IssueCode.ROW_TOO_LONG, "row has more fields than the header"))
            return None, issues
        short = [k for k, v in row.items() if v is None]
        if short:
            issues.append(ParseIssue(line, None, IssueCode.ROW_TOO_SHORT, "row is missing fields for column(s) %s" % short))
            return None, issues

        def text(canon: str) -> Optional[str]:
            v = row[c[canon]].strip()
            return v or None

        def num(canon: str, required: bool = False) -> Optional[float]:
            col, raw = c[canon], row[c[canon]]
            t = raw.strip()
            if not t:
                if required:
                    issues.append(ParseIssue(line, col, IssueCode.MISSING_VALUE, "required value is empty"))
                return None
            if "_" in t:
                issues.append(ParseIssue(line, col, IssueCode.INVALID_NUMBER, "not a plain decimal number", raw))
                return None
            try:
                val = float(t)
            except ValueError:
                issues.append(ParseIssue(line, col, IssueCode.INVALID_NUMBER, "not a number", raw))
                return None
            if not math.isfinite(val):
                issues.append(ParseIssue(line, col, IssueCode.NON_FINITE, "NaN/inf is not allowed", raw))
                return None
            return val

        def req_text(canon: str) -> Optional[str]:
            t = text(canon)
            if t is None:
                issues.append(ParseIssue(line, c[canon], IssueCode.MISSING_VALUE, "required value is empty"))
            return t

        track_id = req_text("track_id")
        sequence_id = req_text("sequence_id") if "sequence_id" in c else cfg.sequence_id
        scene_id = req_text("scene_id") if "scene_id" in c else cfg.scene_id

        frame_index = None
        fval = num("frame_index", required=True)
        if fval is not None:
            if fval != int(fval) or fval < 0:
                issues.append(ParseIssue(line, c["frame_index"], IssueCode.INVALID_FRAME_INDEX,
                                         "frame_index must be a non-negative integer", row[c["frame_index"]]))
            else:
                frame_index = int(fval)

        timestamp = None
        if "timestamp" in c:
            tval = num("timestamp", required=True)
            if tval is not None:
                timestamp = tval * _TIME_SCALE[cfg.timestamp_unit]
        elif frame_index is not None:
            timestamp = frame_index / cfg.fps

        pos_x = num("x") if "x" in c else None
        pos_y = num("y") if "y" in c else None
        if ("x" in c) and ((pos_x is None) != (pos_y is None)):
            issues.append(ParseIssue(line, c["x"], IssueCode.MISSING_VALUE, "x and y must both be present or both empty"))

        bbox = None
        if "bbox_x_min" in c:
            vals = [num(k) for k in _BBOX]
            if all(v is not None for v in vals):
                bbox = tuple(vals)
            elif any(v is not None for v in vals):
                issues.append(ParseIssue(line, None, IssueCode.MISSING_VALUE, "bbox must have all four coordinates or none"))

        vel_x = num("vel_x") if "vel_x" in c else None
        vel_y = num("vel_y") if "vel_y" in c else None
        conf = num("confidence") if "confidence" in c else None
        if conf is not None and not (0.0 <= conf <= 1.0):
            issues.append(ParseIssue(line, c["confidence"], IssueCode.OUT_OF_RANGE, "confidence must be in [0, 1]", row[c["confidence"]]))

        if issues:
            return None, issues

        has_pos = pos_x is not None or bbox is not None
        obs = RawObservation(
            source_dataset=cfg.source_dataset,
            sequence_id=sequence_id,
            scene_id=scene_id,
            track_id=track_id,
            frame_index=frame_index,
            timestamp=timestamp,
            source_label=text("label") if "label" in c else None,
            bbox_xyxy=bbox,
            pos_x=pos_x,
            pos_y=pos_y,
            position_semantics=cfg.position_semantics if has_pos else None,
            position_unit=cfg.position_unit if has_pos else None,
            position_coordinate_system=cfg.position_coordinate_system if has_pos else None,
            vel_x=vel_x,
            vel_y=vel_y,
            detection_confidence=conf,
            source_line=line,
            extras=tuple((h, row[h]) for h in extra_cols),
        )
        return obs, []
