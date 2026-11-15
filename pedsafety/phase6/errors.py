"""Phase 6 errors. ``code`` is stable; nothing is repaired or silently substituted."""

from ..errors import PedsafetyError


class ConflictContractError(PedsafetyError):
    """An input, configuration or output contract of the conflict module is broken."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


class ConflictLabelError(ConflictContractError):
    """A conflict label set is not usable for the requested purpose (e.g. unverified candidate labels used as supervision)."""


class ConflictTrainingBlocked(ConflictContractError):
    """Supervised conflict training is blocked: no verified conflict labels / independent groups are available."""


__all__ = ["ConflictContractError", "ConflictLabelError", "ConflictTrainingBlocked"]
