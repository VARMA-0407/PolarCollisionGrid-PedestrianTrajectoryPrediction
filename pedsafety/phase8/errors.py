"""Phase 8 errors. ``code`` is stable; nothing is repaired or silently substituted."""

from ..errors import PedsafetyError


class AlertContractError(PedsafetyError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


class InputContractError(AlertContractError):
    """A Phase 7 record violates the Phase 7 output contract (malformed, not merely insufficient evidence)."""


class TimestampOrderError(AlertContractError):
    """Observations arrived out of order or a repeated frame carried different content."""


class InvalidTransition(AlertContractError):
    """A state transition outside the documented table was requested."""


class LeadTimeBlocked(AlertContractError):
    """Lead time cannot be computed: no independently verified event timestamps."""


__all__ = ["AlertContractError", "InputContractError", "TimestampOrderError", "InvalidTransition", "LeadTimeBlocked"]
