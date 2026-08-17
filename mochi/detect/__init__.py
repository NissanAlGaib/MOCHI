"""Detection pipeline (Phases 4, 6, 8, 9).

Phase 4  - payload segmentation and the source trust model
Stage I  - syntactic regex filtering (Phase 6)
Stage II - fine-tuned E5 semantic detection (Phase 8)
Stage III - lightweight LLM cognitive arbitration for the uncertain band (Phase 9)
"""

from mochi.detect.chunking import Chunk, chunk_count_for, chunk_text
from mochi.detect.pipeline import InspectionResult, inspect
from mochi.detect.stage1_syntactic import (
    DEFAULT_BLOCK_SEVERITY,
    MAX_SCAN_CHARS,
    MAX_TOTAL_SCAN_CHARS,
    TRUNCATION_FLAG,
    Detection,
    PatternError,
    ScanCoverage,
    Stage1Detector,
    Stage1Result,
    get_detector,
)
from mochi.detect.stage2_semantic import (
    BENIGN_THRESHOLD,
    MALICIOUS_THRESHOLD,
    KeywordScorer,
    ModelUnavailable,
    SemanticScorer,
    Stage2Detector,
    Stage2Result,
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
    "BENIGN_THRESHOLD",
    "Chunk",
    "DEFAULT_BLOCK_SEVERITY",
    "Detection",
    "INJECTION_CLASSES",
    "InspectionResult",
    "InjectionClass",
    "KeywordScorer",
    "MALICIOUS_THRESHOLD",
    "MAX_SCAN_CHARS",
    "MAX_TOTAL_SCAN_CHARS",
    "ModelUnavailable",
    "PatternError",
    "ROLE_TO_SOURCE",
    "ScanCoverage",
    "Segment",
    "SemanticScorer",
    "SourceTag",
    "Stage1Detector",
    "Stage1Result",
    "Stage2Detector",
    "Stage2Result",
    "TRUNCATION_FLAG",
    "TRUST_LEVELS",
    "TrustLevel",
    "build_segments",
    "chunk_count_for",
    "chunk_text",
    "get_detector",
    "inspect",
]
