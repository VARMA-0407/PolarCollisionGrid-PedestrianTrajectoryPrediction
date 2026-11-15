"""Phase 6: interaction-aware image-space pedestrian-agent conflict representation (heuristic indicators; supervised prediction gated on verified labels)."""

from .config import ConflictConfig
from .errors import ConflictContractError, ConflictLabelError, ConflictTrainingBlocked
from .features import FEATURE_SPECS, compute_pair_features
from .labels import ConflictLabelSet, validate_label_set
from .metrics import evaluate_supervised
from .pairs import ConflictPairBatch, Phase5Predictions, build_conflict_pairs
from .scoring import ConflictScores, HeuristicProximityScorer, LogisticConflictHead
