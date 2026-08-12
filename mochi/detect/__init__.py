"""Detection pipeline (Phases 4, 6, 8, 9).

Phase 4  - payload segmentation and the source trust model
Stage I  - syntactic regex filtering (Phase 6)
Stage II - fine-tuned E5 semantic detection (Phase 8)
Stage III - lightweight LLM cognitive arbitration for the uncertain band (Phase 9)
"""

from mochi.detect.pipeline import InspectionResult, inspect
from mochi.detect.stage1_syntactic import (
    DEFAULT_BLOCK_SEVERITY,
    Detection,
    PatternError,
    Stage1Detector,
    Stage1Result,
    get_detector,
)
from mochi.detect.segments import (
    INJECTION_CLASSES,
    ROLE_TO_SOURCE,
    TRUST_LEVELS,
    InjectionClass,
    Segment,
    SourceTag,
    TrustLevel,
    build_segments,
)

__all__ = [
    "DEFAULT_BLOCK_SEVERITY",
    "Detection",
    "INJECTION_CLASSES",
    "InspectionResult",
    "InjectionClass",
    "PatternError",
    "ROLE_TO_SOURCE",
    "Segment",
    "SourceTag",
    "Stage1Detector",
    "Stage1Result",
    "TRUST_LEVELS",
    "TrustLevel",
    "build_segments",
    "get_detector",
    "inspect",
]
