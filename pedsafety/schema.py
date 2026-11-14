"""Loading and validation against the approved agent-state JSON schema.

The schema file (docs/phase2/schemas/agent_state.schema.json) is the
contract. This module only reads it; it never writes or alters it.
"""

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import SchemaLoadError, SchemaViolation, Violation

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "phase2"
    / "schemas"
    / "agent_state.schema.json"
)

PathLike = Union[str, Path, None]

# Name-based heuristic used to ask "does the schema have a field that could
# mark a causal/filtered/predicted estimate?". It inspects property names only.
_ESTIMATE_NAME_PATTERN = re.compile(r"causal|estimat|filter|predict", re.IGNORECASE)


def _resolve(path: PathLike) -> Path:
    return Path(path) if path is not None else DEFAULT_SCHEMA_PATH


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> Tuple[str, str]:
    path = Path(path_str)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SchemaLoadError("cannot read schema file %s: %s" % (path, exc))
    try:
        text = raw.decode("utf-8")
        schema = json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SchemaLoadError("schema file %s is not valid UTF-8 JSON: %s" % (path, exc))
    if not isinstance(schema, dict):
        raise SchemaLoadError("schema file %s must contain a JSON object" % path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise SchemaLoadError("schema file %s is not a valid Draft 2020-12 schema: %s" % (path, exc.message))
    return text, hashlib.sha256(raw).hexdigest()


def load_schema(path: PathLike = None) -> Dict[str, Any]:
    """Return a fresh copy of the schema dict."""
    text, _ = _load_cached(str(_resolve(path)))
    return json.loads(text)


def schema_sha256(path: PathLike = None) -> str:
    """SHA-256 of the schema file bytes, for provenance records."""
    return _load_cached(str(_resolve(path)))[1]


def get_validator(path: PathLike = None) -> Draft202012Validator:
    return Draft202012Validator(load_schema(path))


def schema_property_names(path: PathLike = None) -> FrozenSet[str]:
    return frozenset(load_schema(path).get("properties", {}).keys())


def schema_represents_causal_estimate(path: PathLike = None) -> bool:
    """True only if some schema property name suggests it can mark a causal
    estimate. The approved schema has only ``interpolation_flag`` (boolean),
    which marks offline interpolation, so this returns False. Callers must
    then carry the observed / causal-estimate / offline-interpolated
    distinction in provenance metadata outside the schema record.
    """
    return any(_ESTIMATE_NAME_PATTERN.search(n) for n in schema_property_names(path))


def find_non_finite(value: Any, _path: str = "") -> List[str]:
    """Return dotted paths of NaN/inf floats. JSON Schema's ``number`` type
    accepts them, so this extra check is needed."""
    found: List[str] = []
    if isinstance(value, float):
        if not math.isfinite(value):
            found.append(_path or "<record>")
    elif isinstance(value, dict):
        for k, v in value.items():
            found.extend(find_non_finite(v, "%s.%s" % (_path, k) if _path else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.extend(find_non_finite(v, "%s[%d]" % (_path, i)))
    return found


def _error_path(error) -> str:
    parts = []
    for p in error.absolute_path:
        parts.append("[%d]" % p if isinstance(p, int) else str(p))
    out = ""
    for p in parts:
        out += p if p.startswith("[") or not out else "." + p
    return out or "<record>"


def collect_violations(record: Any, path: PathLike = None) -> List[Violation]:
    """All violations of ``record`` (empty list if valid), sorted by path."""
    if not isinstance(record, dict):
        return [Violation("<record>", "record must be a JSON object, got %s" % type(record).__name__)]
    violations = [
        Violation(_error_path(e), e.message)
        for e in get_validator(path).iter_errors(record)
    ]
    for p in find_non_finite(record):
        violations.append(Violation(p, "non-finite number (NaN/inf) is not allowed"))
    return sorted(violations, key=lambda v: (v.path, v.message))


def validate_record(record: Any, *, record_index: Optional[int] = None, path: PathLike = None) -> None:
    """Raise :class:`SchemaViolation` listing every problem, or return None."""
    violations = collect_violations(record, path)
    if violations:
        raise SchemaViolation(violations, record_index)


def validate_records(records: Iterable[Any], path: PathLike = None) -> List[Tuple[int, List[Violation]]]:
    """Non-raising batch check: ``[(index, violations), ...]`` for invalid records only."""
    bad = []
    for i, rec in enumerate(records):
        v = collect_violations(rec, path)
        if v:
            bad.append((i, v))
    return bad
