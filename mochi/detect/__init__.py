"""Detection pipeline (Phases 4, 6, 8, 9).

Phase 4  - payload segmentation and the source trust model
Stage I  - syntactic regex filtering (Phase 6)
Stage II - fine-tuned E5 semantic detection (Phase 8)
Stage III - lightweight LLM cognitive arbitration for the uncertain band (Phase 9)
"""

from mochi.detect.pipeline import InspectionResult, inspect
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
    "INJECTION_CLASSES",
    "InspectionResult",
    "InjectionClass",
    "ROLE_TO_SOURCE",
    "Segment",
    "SourceTag",
    "TRUST_LEVELS",
    "TrustLevel",
    "build_segments",
    "inspect",
]
