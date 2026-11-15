"""Phase 8: deterministic early-warning state machine and alert-event generation on top of Phase 7 risk outputs (research prototype)."""

from .config import AlertConfig
from .errors import AlertContractError, InputContractError, InvalidTransition, LeadTimeBlocked, TimestampOrderError
from .manager import AlertManager
from .phase7_input import Observation, parse_record
from .schema import validate_event
