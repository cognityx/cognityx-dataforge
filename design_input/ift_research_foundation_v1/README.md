# Cognityx DataForge + Training Research Foundation — Design Input Pack v1

This directory is intended to be committed under `design_input/ift_research_foundation_v1/` and consumed by the accompanying Codex master prompt.

## Precedence

1. `CODEX_MASTER_PROMPT.md` — implementation contract for the task.
2. `empirical/qualification/expected_qualification_v1.jsonl` — frozen human/source-grounded expected outcomes for the primary regression cases.
3. `requirements/cognityx_adaptive_evaluator_failure_register_v0_9.md` — empirical failure evidence and traceability source. It supersedes older failure-register versions.
4. `empirical/qualification/e3_qa_qualification_cases_v1.jsonl` — candidate-blind qualification inputs.
5. `empirical/qualification/e3_source_grounded_cases_v2.jsonl` — historical problematic candidate/reference examples; candidate answers must not leak into question-demand or source-answerability stages.
6. Other requirement/research documents — context and rationale only. They must not expand implementation scope beyond the master prompt.

If a supporting document conflicts with the master prompt or frozen oracle, the master prompt/oracle wins for this implementation.

## Contents

### Requirements
- Adaptive Evaluator Failure Register v0.9.
- Consolidated issue handoff (cross-repo issues, including DataForge and Training defects).
- Research Thought Inventory / Deep Research handoff.
- Original Deep Research narrowing prompt.

### Research context
- Evidence-Based Recipe for policy-document synthetic IFT and consumer-GPU adaptation.
- Research-area prior-art inventory.
- Recovered key-decision extract from the radically narrowed Deep Research program (5,184-cell warning, ~12-run program, Paper A/Paper B pruning decisions).

These research documents explain why the experiment is narrowed to raw paragraph QA versus provenance/source-qualified paragraph QA. Codex must not implement every research idea contained in them.

### Empirical qualification inputs
- `e3_qa_qualification_cases_v1.jsonl`: R05, R06, R08, R09 without candidate answers.
- `expected_qualification_v1.jsonl`: frozen expected stage-level outcomes.
- `e3_source_grounded_cases_v2.jsonl`: examples of bad/incomplete model outputs and source-grounded adjudication issues.
- `deterministic_mutations_v1.jsonl`: deliberately corrupted cases for hard-gate regression tests.

### Evaluation-set inputs
- `manual_v1_paraphrase_texts.json`: the historical ten-question paraphrase pilot, preserved as history only.
- `paraphrase_import_fixture_v1.jsonl`: compact qualified evaluation-only import fixture for testing freeze/checksum/leakage behavior.

### Source and baseline references
- `lunavane_hr_policy_provenance_fixture_v2_parsed.md`: text reference reconstructed from the frozen source. It is not byte-identical and must not be used for parser/SHA fidelity tests.
- `lunavane_baseline_dataset_identity.json`: historical DataForge dataset identity/checksums/URIs from the 29-record paragraph-QA run. The URIs are expected to be local to the original workstation; use repository fixtures in CI.

## Important scientific boundary

The first intended causal experiment after this implementation is:

`same source + same generator + same base model + same training-token budget + same evaluator`

**raw paragraph QA** versus **source/provenance-qualified paragraph QA**.

Do not change raw candidate generation in the qualified treatment. Do not add KU-first generation, knowledge-selective training, provenance-target training, model-size sweeps, H100 runs, or a meta-evaluator in this task.

## Original source binary

The original `lunavane_hr_policy_provenance_fixture_v2_verified.docx` existed in the prior experiment library but is not embedded as a binary here. The JSONL qualification fixtures contain the controlling evidence needed for the required unit tests. For one real provenance-resolution integration test, prefer existing Ingest fixtures/run artifacts already present in the workspace; otherwise use a producer-compatible normalized fixture rather than recreating parsing logic in DataForge.
