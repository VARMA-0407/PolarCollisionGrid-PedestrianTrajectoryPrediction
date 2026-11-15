"""Phase 7 errors. ``code`` is stable; nothing is repaired or silently substituted."""

from ..errors import PedsafetyError


class RiskContractError(PedsafetyError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("[%s] %s" % (code, message))


class CalibrationBlocked(RiskContractError):
    """Calibration cannot be fitted or claimed: verified labels / independent partitions are missing."""


class ProvenanceMismatch(RiskContractError):
    """A calibration artifact does not belong to the model / data / configuration it is applied to."""


__all__ = ["RiskContractError", "CalibrationBlocked", "ProvenanceMismatch"]
