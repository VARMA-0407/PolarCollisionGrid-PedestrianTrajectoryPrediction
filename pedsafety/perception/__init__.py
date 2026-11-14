"""pedsafety.perception - runtime-independent perception frontend interfaces (Phase 3, Stage 2A).

Interfaces, data contracts, validation, a pipeline, an agent-state adapter, evaluation interfaces and performance
scaffolding. NO real detector or tracker is integrated: YOLO and ByteTrack wrappers are Stage 2B work and will live
behind ``Detector`` / ``Tracker``. The only concrete components here are in ``mocks`` and are labelled as mocks.
Nothing in this package imports OpenCV, Ultralytics, a tracker library, numpy or torch.
"""

from .adapter import AdapterConfig, AdapterResult, PerceptionProvenance, tracks_to_agent_states
from .detector import Detector
from .metrics import (
    DetectionEvaluator, EvaluationProtocol, EvaluationReport, EvaluationUnavailableError, GroundTruthBox, GroundTruthSequence, TrackingEvaluator,
)
from .performance import PerformanceRecorder, PerformanceSummary
from .pipeline import FrameResult, PerceptionPipeline, PerceptionResult, PipelineConfig
from .sources import FrameSource
from .tracker import Tracker
from .types import (
    BBOX_CONVENTION, ComponentInfo, DetectionRecord, FrameMeta, IssueCode, LabelMap, PerceptionClass, PerceptionError, PerceptionIssue, RawDetection,
    TrackObservation, TrackRecord,
)
