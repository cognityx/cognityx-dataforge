"""Expose the stable application and provenance-handoff composition seams.

Cognityx DataForge turns Ingest-owned evidence into DataForge-owned training and
evaluation records. The package root keeps the established ``build_dataset``
entrypoint and additively exports T09's immutable handoff types. Applications use
these names without importing parser implementations, physical Storage backends,
or inference transports. Existing recipes and Evidence/KnowledgeUnit v1 models
remain unchanged.
"""

from cognityx_dataforge.build import build_dataset
from cognityx_dataforge.handoff import (
    COMPOSITE_KU_HANDOFF_SCHEMA,
    PARAGRAPH_QA_HANDOFF_SCHEMA,
    CompositeKnowledgeUnitHandoff,
    DataForgeHandoffService,
    GeneratedQuestionAnswer,
    GoldRelationClosure,
    HandoffArtifactStore,
    HandoffError,
    HandoffPersistenceError,
    HandoffSupportError,
    HandoffValidationError,
    ParagraphHandoffInput,
    ParagraphQAHandoff,
    SupportedClaim,
    ValidatedEvidenceBundle,
    build_gold_relation_closure,
)
from cognityx_dataforge.source import (
    ResolvedSource,
    ResolvedV32Document,
    ResolvedV32SourceBundle,
    V32HandoffUnavailableError,
    V32SourceConflictError,
    resolve_source,
)

__all__ = [
    "COMPOSITE_KU_HANDOFF_SCHEMA",
    "PARAGRAPH_QA_HANDOFF_SCHEMA",
    "CompositeKnowledgeUnitHandoff",
    "DataForgeHandoffService",
    "GeneratedQuestionAnswer",
    "GoldRelationClosure",
    "HandoffArtifactStore",
    "HandoffError",
    "HandoffPersistenceError",
    "HandoffSupportError",
    "HandoffValidationError",
    "ParagraphHandoffInput",
    "ParagraphQAHandoff",
    "ResolvedSource",
    "ResolvedV32Document",
    "ResolvedV32SourceBundle",
    "SupportedClaim",
    "V32HandoffUnavailableError",
    "V32SourceConflictError",
    "ValidatedEvidenceBundle",
    "build_dataset",
    "build_gold_relation_closure",
    "resolve_source",
]
