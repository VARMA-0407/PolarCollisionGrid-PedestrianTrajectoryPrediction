"""Phase 5 window-generation errors. ``code`` is stable; nothing is repaired."""

from ..errors import PedsafetyError


class WindowContractError(PedsafetyError):
    """An input, configuration or stored-dataset contract is broken."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


class TargetSourceUnavailable(WindowContractError):
    """A requested target source cannot be produced (e.g. ANNOTATED_TARGET without verified annotations)."""


__all__ = ["WindowContractError", "TargetSourceUnavailable"]
