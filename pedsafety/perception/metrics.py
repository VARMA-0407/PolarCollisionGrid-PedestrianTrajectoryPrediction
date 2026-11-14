"""Detection / tracking EVALUATION INTERFACES (Stage 2A). No metric is computed here.

This module fixes the vocabulary, the ground-truth contract and the refusal rules for the future (Stage 2B+) evaluation
on labelled video. It deliberately contains no metric implementation: computing precision, recall, mAP, HOTA, IDF1, MOTA
or identity switches needs real predictions and suitable ground truth, and inventing numbers would be a false claim.

Rules enforced now:
* detection and tracking are evaluated by SEPARATE interfaces (a good detector says nothing about identity stability);
* a report cannot be built from mock components or without a declared dataset/ground-truth description;
* calling an unimplemented evaluator raises ``EvaluationUnavailableError`` instead of returning zeros.

Metric catalogue (definitions are the standard ones; matching protocol must be fixed and recorded in Stage 2B):
detection  - precision, recall at an IoU threshold and a confidence threshold; mAP (mean over classes of average precision,
             IoU thresholds declared, e.g. 0.5 and 0.5:0.95);
tracking   - HOTA (detection x association accuracy), IDF1 (identity F1), MOTA (1 - (FN + FP + IDSW) / GT), identity switches
             (IDSW), all against ground-truth track identities, with the IoU matching threshold declared;
performance- see ``performance.py`` (processing FPS, per-frame latency, end-to-end latency, memory).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..errors import PedsafetyError
from .pipeline import PerceptionResult
from .types import ComponentInfo, PerceptionClass

DETECTION_METRICS = ("precision", "recall", "mAP")
TRACKING_METRICS = ("HOTA", "IDF1", "MOTA", "identity_switches")
PERFORMANCE_METRICS = ("processing_fps", "per_frame_latency", "end_to_end_latency", "memory")


class EvaluationUnavailableError(PedsafetyError):
    """No real implementation / ground truth is available; no number is produced."""


@dataclass(frozen=True)
class GroundTruthBox:
    frame_index: int
    category: PerceptionClass
    bbox_xyxy: Tuple[float, float, float, float]       # same convention as detections (image pixels, top-left, y down)
    track_id: Optional[int] = None                     # required for tracking evaluation
    ignore: bool = False                               # e.g. crowd/occluded regions the annotators excluded


@dataclass(frozen=True)
class GroundTruthSequence:
    """Ground truth of ONE sequence. ``description`` must state dataset, annotation protocol and version."""

    sequence_id: str
    coordinate_system: str
    boxes: Tuple[GroundTruthBox, ...]
    description: str

    def __post_init__(self):
        if not (isinstance(self.description, str) and self.description.strip()):
            raise ValueError("GroundTruthSequence.description must name the dataset and annotation protocol")


@dataclass(frozen=True)
class EvaluationProtocol:
    iou_threshold: float
    confidence_threshold: Optional[float]
    classes: Tuple[PerceptionClass, ...]
    notes: str = ""


class DetectionEvaluator(ABC):
    metrics = DETECTION_METRICS

    @abstractmethod
    def evaluate(self, result: PerceptionResult, truth: Sequence[GroundTruthSequence], protocol: EvaluationProtocol) -> Mapping[str, float]:
        """Detection metrics for the detections in ``result`` (tracks are ignored)."""


class TrackingEvaluator(ABC):
    metrics = TRACKING_METRICS

    @abstractmethod
    def evaluate(self, result: PerceptionResult, truth: Sequence[GroundTruthSequence], protocol: EvaluationProtocol) -> Mapping[str, float]:
        """Tracking metrics for the tracks in ``result``; needs ground-truth track ids."""


class UnimplementedDetectionEvaluator(DetectionEvaluator):
    def evaluate(self, result, truth, protocol):
        raise EvaluationUnavailableError("detection metrics are not implemented in Stage 2A; they need real inference and ground truth")


class UnimplementedTrackingEvaluator(TrackingEvaluator):
    def evaluate(self, result, truth, protocol):
        raise EvaluationUnavailableError("tracking metrics are not implemented in Stage 2A; they need real inference and ground truth")


@dataclass(frozen=True)
class EvaluationReport:
    """A container for REAL evaluation numbers. Construction refuses mock components, so mock output can never be
    presented as detection or tracking performance."""

    kind: str                                  # 'detection' | 'tracking'
    metrics: Mapping[str, float]
    components: Tuple[ComponentInfo, ...]
    dataset_description: str
    protocol: EvaluationProtocol

    def __post_init__(self):
        if self.kind not in ("detection", "tracking"):
            raise ValueError("kind must be 'detection' or 'tracking'")
        if not self.components:
            raise ValueError("an evaluation report must name the components that produced the results")
        if any(c.is_mock for c in self.components):
            raise EvaluationUnavailableError("a report cannot be built from mock components: %s" % [c.name for c in self.components if c.is_mock])
        if not (isinstance(self.dataset_description, str) and self.dataset_description.strip()):
            raise ValueError("dataset_description is required")
        allowed = DETECTION_METRICS if self.kind == "detection" else TRACKING_METRICS
        extra = sorted(set(self.metrics) - set(allowed))
        if extra:
            raise ValueError("metrics %s do not belong to %s evaluation (allowed: %s)" % (extra, self.kind, allowed))
