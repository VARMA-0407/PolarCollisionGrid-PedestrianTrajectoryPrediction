"""Exception types with messages meant to be read by a person fixing data."""

from dataclasses import dataclass
from typing import List, Sequence


class PedsafetyError(Exception):
    """Base class for all pedsafety errors."""


class SchemaLoadError(PedsafetyError):
    """The agent-state schema file is missing, unreadable or not a valid schema."""


class ConfigError(PedsafetyError):
    """An adapter or processing configuration is invalid or inconsistent
    with the input file (for example a mapped column is absent)."""


@dataclass(frozen=True)
class Violation:
    """One schema violation. ``path`` is a dotted field path or ``<record>``."""

    path: str
    message: str

    def __str__(self) -> str:
        return "%s: %s" % (self.path, self.message)


class SchemaViolation(PedsafetyError):
    """A record does not satisfy the agent-state schema."""

    def __init__(self, violations: Sequence[Violation], record_index=None):
        self.violations: List[Violation] = list(violations)
        self.record_index = record_index
        where = "record" if record_index is None else "record %s" % (record_index,)
        lines = "; ".join(str(v) for v in self.violations[:10])
        extra = ""
        if len(self.violations) > 10:
            extra = " (+%d more)" % (len(self.violations) - 10)
        super().__init__(
            "%s violates the agent-state schema (%d violation(s)): %s%s"
            % (where, len(self.violations), lines, extra)
        )


class ParseError(PedsafetyError):
    """Raised in strict mode when a source row cannot be parsed."""

    def __init__(self, issues):
        self.issues = list(issues)
        shown = "; ".join(str(i) for i in self.issues[:5])
        super().__init__("could not parse source row: %s" % shown)
