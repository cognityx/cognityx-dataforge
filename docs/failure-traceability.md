# Failure-register traceability

The design pack contains an append-only list of observed and anticipated
failure modes, F-001 through F-095. This page records what the current narrow
foundation implements, tests or deliberately leaves for later work. It does
not claim that the small fixture set proves general model quality.

| Failure IDs | Current response | Evidence |
| --- | --- | --- |
| F-001–F-018 | Keep exact recall, paraphrase robustness and held-out-fact evaluation as separate frozen roles. Prevent train/evaluation leakage and retain group identity. | `tests/test_research_package.py` |
| F-019–F-023 | Qualify generated references against source evidence and preserve provenance separately from answer text. Broader answer-plus-provenance model scoring remains future Evaluator work. | `tests/test_qualification.py` |
| F-024–F-025 | Require non-empty exact recall for a trainable source and exercise the Training evaluation path with a BatchEncoding-shaped fake. | Training repository evaluation tests |
| F-026 | Training-to-Inference adapter serving is explicitly out of scope for this foundation. | [Qualification guide](qualification-research.md) |
| F-027–F-030 | Research-package identity freezes dataset/evaluation checksums; Training publications record model, configuration, metric and resource identity. | DataForge package tests and Training publication tests |
| F-031–F-040 | Freeze question demand before source/reference inspection, use deterministic gates for numeric roles, exact phrases, lists and provenance, and keep disagreement as review evidence. | qualification oracle and mutation tests |
| F-041–F-059 | Preserve semantic task/group roles across paraphrases, keep held-out wording separate from held-out facts and freeze every imported set. Automatic 15-way paraphrase generation is deferred. | research-package validation tests |
| F-060–F-066 | Bound retries, retain raw attempts per record, checkpoint stages and classify malformed structured output as infrastructure uncertainty. | review persistence test |
| F-067–F-080 | Separate question answerability, answer fit, unsupported additions and physical versus printed page coordinates. Evidence is resolved through Ingest/Storage for builds. | R05/R06/R08/R09 and provenance tests |
| F-081–F-095 | Reject a monolithic model judge as final authority. Use the staged A/B/C artifacts followed by deterministic D; never accept or invent a rewrite. | anti-anchoring spies, R09 rejection and mutation tests |

The versioned empirical inputs live under
`design_input/ift_research_foundation_v1/`. Production code does not import
that directory. Tests read the frozen oracle and mutation rows so a changed
expectation is visible in review rather than silently becoming new behavior.
