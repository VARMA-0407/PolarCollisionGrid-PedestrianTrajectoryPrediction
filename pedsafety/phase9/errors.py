"""Phase 9 errors. ``code`` is stable; nothing is repaired, interpolated or silently substituted."""

from ..errors import PedsafetyError


class PipelineError(PedsafetyError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


class StageContractError(PipelineError):
    """A stage input or output violates its versioned contract."""


class MissingInputError(PipelineError):
    """A required input or upstream artifact is not available; nothing is fabricated to continue."""


class EvaluationBlocked(PipelineError):
    """A scientific evaluation is blocked by missing independent evidence."""


__all__ = ["PipelineError", "StageContractError", "MissingInputError", "EvaluationBlocked"]
