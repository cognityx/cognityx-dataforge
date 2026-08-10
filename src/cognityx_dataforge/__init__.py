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
from cognityx_dataforge.qualification import (
    ANSWER_REQUIREMENTS_SCHEMA,
    QUALIFICATION_DECISION_SCHEMA,
    REFERENCE_QUALIFICATION_SCHEMA,
    SOURCE_ANSWERABILITY_SCHEMA,
    QualificationPipeline,
    qualification_decision,
)
from cognityx_dataforge.research import (
    EVALUATION_SET_SCHEMA,
    RESEARCH_PACKAGE_SCHEMA,
    create_exact_recall_set,
    create_research_package,
    freeze_evaluation_set,
    import_evaluation_set,
    load_research_package,
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
    "ANSWER_REQUIREMENTS_SCHEMA",
    "EVALUATION_SET_SCHEMA",
    "QUALIFICATION_DECISION_SCHEMA",
    "REFERENCE_QUALIFICATION_SCHEMA",
    "RESEARCH_PACKAGE_SCHEMA",
    "SOURCE_ANSWERABILITY_SCHEMA",
    "QualificationPipeline",
    "build_dataset",
    "build_gold_relation_closure",
    "resolve_source",
    "create_exact_recall_set",
    "create_research_package",
    "freeze_evaluation_set",
    "import_evaluation_set",
    "load_research_package",
    "qualification_decision",
]
