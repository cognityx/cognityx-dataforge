# CODEX MASTER TASK — Cognityx DataForge Qualification + Research Evaluation Sets + Training Tracking Foundation

Use the **strongest full Codex model available** with high reasoning effort. Work autonomously through the complete bounded task, including tests and documentation. Do not wait for interactive confirmation between phases unless you encounter a destructive operation, a security/authorization problem, or a material repository-state conflict that makes the requested scientific comparison invalid.

This is an **implementation task**, not an architecture essay and not a request to expand the research agenda.

---

## 1. Repositories and ownership

Primary repositories:

- `cognityx/cognityx-dataforge`
- `cognityx/cognityx-training`

Read-only/context dependencies unless a verified blocking contract defect is discovered:

- `cognityx/cognityx-ingest`
- `cognityx/cognityx-storage`
- `cognityx/cognityx-inference`
- `cognityx/cognityx-jobs`
- `cognityx/cognityx-core`
- `cognityx/cognityx-sdk`

Do **not** implement the Training→Inference adapter handoff in this task. That is a separate follow-up. Do **not** modify the unified SDK/`cogni` CLI here.

Create separate focused branches/commits and preferably separate draft PRs for DataForge and Training. Do not merge them automatically.

If only one primary repository is available in the workspace, fully complete the work for that repository and write a precise cross-repository handoff for the unavailable repository. Do not invent files or APIs that you cannot inspect.

---

## 2. Mandatory design input

The authoritative design pack is committed at:

`design_input/ift_research_foundation_v1/`

Read first:

1. `design_input/ift_research_foundation_v1/README.md`
2. `design_input/ift_research_foundation_v1/empirical/qualification/expected_qualification_v1.jsonl`
3. `design_input/ift_research_foundation_v1/requirements/cognityx_adaptive_evaluator_failure_register_v0_9.md`
4. `design_input/ift_research_foundation_v1/empirical/qualification/e3_qa_qualification_cases_v1.jsonl`
5. `design_input/ift_research_foundation_v1/empirical/qualification/e3_source_grounded_cases_v2.jsonl`
6. `design_input/ift_research_foundation_v1/empirical/qualification/deterministic_mutations_v1.jsonl`
7. `design_input/ift_research_foundation_v1/empirical/evaluation_sets/paraphrase_import_fixture_v1.jsonl`
8. `design_input/ift_research_foundation_v1/empirical/baseline/lunavane_baseline_dataset_identity.json`
9. `design_input/ift_research_foundation_v1/empirical/source/lunavane_hr_policy_provenance_fixture_v2_parsed.md`

Then read the remaining requirement/research documents for rationale and scope context.

### Precedence

If documents disagree for this implementation:

1. this master prompt;
2. `expected_qualification_v1.jsonl`;
3. Failure Register v0.9;
4. candidate-blind qualification cases;
5. source-grounded historical cases;
6. other design/research documents.

The research documents intentionally describe a much larger hypothesis space. **Do not implement that larger space.**

The design-input directory is a development/research input, not a production runtime dependency. Production modules must not import files from `design_input/`. Small stable cases may be vendored into `tests/fixtures/` with a source note/checksum when needed for tests.

---

## 3. Scientific objective — preserve the causal comparison

The immediate experiment after this task is deliberately narrow:

```text
same source corpus
same initial QA generator
same frozen base model/revision
same training implementation
same training-token budget
same frozen evaluation sets
same evaluator

Treatment A: existing raw paragraph-derived synthetic QA
Treatment B: the same paragraph-derived candidates + source/provenance-grounded qualification
```

Therefore the primary DataForge treatment is:

`paragraph-qa-qualified`

and its initial candidate-generation path must be the **same** as existing `paragraph-qa`.

Do not make unrelated generation improvements inside the qualified arm. Otherwise the experiment will no longer isolate qualification.

The task is successful only if a later experiment can attribute a downstream difference to qualification rather than to a changed generator, model, split heuristic, prompt family, trainer, or evaluator.

### Explicitly out of scope

Do not implement in this task:

- KU-first generation as a new experimental treatment;
- new probed/mixed/adaptive curriculum recipes;
- training only base-unknown KUs;
- Answer→Provenance or Answer+Provenance training targets;
- automatic 15-paraphrase generation;
- model-size sweeps;
- LoRA-rank/LR/optimizer sweeps;
- H100/cloud execution;
- adaptive/meta-evaluator construction;
- arbitrary evaluator code generation;
- a general experiment-matrix runner;
- a new workflow scheduler;
- a custom experiment database replacing MLflow;
- direct PEFT/adapter inference in the experiment layer;
- NMDrop or unrelated train/validation template generation.

---

## 4. Phase 0 — repository preflight and current-state audit

Before changing code:

1. Fetch and inspect current `main` for both primary repositories.
2. Record exact HEAD SHAs in the implementation report.
3. Run the current relevant test suites and record the baseline result.
4. Inspect DataForge's current:
   - `paragraph-qa` recipe;
   - `knowledge-unit-qa` and probed recipe only to reuse infrastructure, not to modify their scientific semantics;
   - source/input-selection resolution;
   - Ingest handoff/provenance/evidence loaders;
   - candidate/rejection/model-call artifacts;
   - checkpoint/resume behavior;
   - Jobs/events usage;
   - split/dedup logic;
   - configuration and model-role handling;
   - publication manifest and checksum behavior;
   - dataset show/export surfaces.
5. Inspect Training's current:
   - DataForge manifest reader;
   - split normalization;
   - tokenization/chat-template path;
   - built-in baseline/final evaluation generation;
   - TrainingPublisher and adapter publication;
   - report/metrics/environment publication;
   - resource telemetry;
   - evaluation handoff;
   - configuration and CLI.
6. Reuse current Storage, Ingest, Inference-client, Jobs, checksum and publication contracts. Do not create parallel frameworks.
7. Detect dependency/version skew. If the current workspace uses sibling versions that make an existing public API unavailable, report the exact mismatch. Pin or use the compatible public contract where this repository normally does so; do not copy another repository's implementation locally.
8. Do not broadly upgrade dependencies merely because newer versions exist.

### Important current compatibility issue to inspect

Current DataForge has historically used `train` / `validation` / `test` split names, while Training has accepted `train` / `eval` / `evaluation`. Resolve this explicitly and backward-compatibly. Do not silently let future DataForge `validation`/`test` records enter the optimizer or become unreadable.

---

# PART A — DATAFORGE

## 5. Preserve the raw baseline exactly

Existing `paragraph-qa` is the lower baseline.

Do not change its candidate-generation semantics, accepted-record semantics, prompt behavior, or external result shape except for strictly additive metadata that is proven not to affect candidate generation and is required for compatibility.

Add regression tests proving the raw recipe remains behaviorally compatible.

The new treatment must be explicit:

`paragraph-qa-qualified`

It must reuse the same source evidence and initial QA generation path as raw `paragraph-qa`, then perform qualification **after** a candidate question/reference has been generated.

Persist the pre-qualification candidate before any filtering so the raw-versus-qualified relationship can be audited.

---

## 6. Candidate-blind staged qualification

Implement a staged, anti-anchoring qualification flow. The stages must have explicit typed/versioned outputs and must not receive information that belongs to later stages.

### Stage A — question-demand extraction

Input allowed:

- question/instruction text;
- minimal task-independent identity metadata if necessary for tracing.

Input forbidden:

- generated/reference gold;
- source evidence text;
- any model candidate answer;
- later-stage decisions.

Output must represent what a satisfactory answer is being asked to supply, not merely a brittle keyword answer-type label. At minimum support:

- requested information slots/facts;
- cardinality/set expectations where meaningful;
- numeric values/roles/units/relations where meaningful;
- mandatory qualifiers/conditions/temporal scope;
- exact-identifier/exact-phrase requirements where meaningful;
- ambiguity notes;
- syntactic/semantic question validity independently from source answerability.

Freeze/persist the Stage-A result before Stage B.

### Stage B — source answerability

Input allowed:

- frozen Stage-A demand;
- resolved source evidence/provenance and controlling structural metadata.

Input forbidden:

- generated/reference gold;
- any model candidate answer.

Determine:

- whether the source can answer the question at the requested specificity;
- which required slots/facts the source fills;
- source-supported values and role bindings;
- which required slots are missing or ambiguous;
- controlling evidence anchors/spans;
- whether multiple spans/sections are required;
- interpretation policy where supplied, e.g. `source_explicit_only`.

Source entailment alone is not sufficient. The source must cover the interrogative demand.

Freeze/persist Stage B before Stage C.

### Stage C — generated reference/gold qualification

Input allowed:

- frozen Stage-A demand;
- frozen Stage-B source-answerability result;
- generated/reference answer.

Determine separately:

- whether the reference actually answers the question;
- required-slot/fact coverage;
- missing mandatory facts/qualifiers;
- supported claims;
- unsupported claims;
- contradicted claims;
- tautological/premise-restating behavior;
- exact identifier/phrase correctness where relevant;
- set/list required-member and unsupported-member behavior where relevant;
- numeric value-role-unit-relation correctness where relevant;
- provenance consistency.

Preserve an evidence/extractive gold/source-support object separately from the generated natural-language reference gold.

### Stage D — deterministic operational decision

Compute one of:

- `accepted`
- `rejected`
- `needs_review`

from explicit gates/policy.

The LLM/semantic qualifier may extract/interpret observations, but it must not freely decide operational acceptance when hard gates are declarative.

At minimum reject when:

- required provenance/evidence cannot be resolved;
- the question is not answerable from the source at requested specificity;
- the reference fails to answer the requested slots;
- a mandatory fact/qualifier is missing;
- a material claim contradicts the controlling source;
- a material unsupported addition violates the selected interpretation policy;
- a required list/set member is missing or an unsupported member is introduced where complete-set semantics apply;
- numeric values are bound to wrong roles/units/relations;
- an exact identifier/phrase requirement is materially incomplete;
- a declared non-trainable evaluation record is accidentally marked optimizer-eligible.

Use `needs_review` or an explicit infrastructure-uncertainty result when semantic qualification cannot be completed reliably after bounded retries. A malformed validator response is **not** evidence that the question/reference itself is incorrect.

---

## 7. No automatic rewrite acceptance in v1

Do not silently repair a bad question/reference and then accept it.

For this first implementation:

- rejected data remains rejected;
- a proposed rewrite, if generated at all for diagnostics, is a new derived object;
- it must never enter the accepted dataset automatically;
- any future rewrite must re-enter Stage A → B → C as a new candidate with its own lineage.

The R09/Kaveri case is the mandatory regression: do not invent a supervisor, HR approver, council, or other approving authority not explicitly bound by the controlling overtime evidence.

---

## 8. Deterministic-first checks and semantic escalation

Use deterministic validation where the semantics are explicit and reliable, including when applicable:

- exact/canonical strings and identifiers;
- normalized number words versus digits;
- units;
- numeric role binding;
- logical relation (`AND`/`OR`) where represented;
- list/set membership and unsupported members;
- exact phrases;
- provenance identifiers/checksums;
- duplicate IDs;
- declared train/evaluation role constraints.

Use semantic/LLM qualification only where necessary for language interpretation, question-demand extraction, source coverage, claim support or unresolved meaning.

Do not build the future adaptive/meta-evaluator here. This is a DataForge **training-data qualification** boundary.

If the current DataForge validator role can be reused safely for the semantic stages, reuse it with separate versioned prompts/contracts. Do not create a gratuitous model-role sweep. If separate role configuration is genuinely necessary for correctness, keep it additive and default it to the existing validator/generator configuration.

Structured semantic outputs must be validated against explicit schemas. Preserve raw attempts, parse errors, attempt numbers, selected final attempt and model/prompt identity needed for reproducibility.

---

## 9. DataForge qualification artifacts and schemas

Persist stage outputs immutably and keep forensic evidence.

At minimum publish logical artifacts for:

- original generated candidates;
- question-demand results;
- source-answerability results;
- reference/gold qualification results;
- deterministic gate results;
- accepted records;
- rejected records;
- needs-review records;
- structured decision reasons;
- relevant model calls/raw attempts;
- run events/checkpoints;
- final qualification summary.

Use versioned schemas. Suggested semantics/names (adapt to established repository naming conventions only where necessary):

- `cognityx.dataforge.answer-requirements/v1`
- `cognityx.dataforge.source-answerability/v1`
- `cognityx.dataforge.reference-qualification/v1`
- `cognityx.dataforge.qualification-decision/v1`

Every accepted/rejected/review record must preserve sufficient lineage back to:

- source asset/document/run;
- evidence IDs/source anchors;
- physical/evidence page coordinate where available;
- printed page label where available;
- candidate-generation model and prompt version;
- qualification model and prompt versions;
- DataForge experiment/variant/run;
- checksums.

Physical/evidence page and document-printed page are different coordinates. Never collapse them into one page field.

---

## 10. Evidence/provenance resolution boundary

Qualification must consume the existing Ingest/DataForge handoff and Cognityx Storage contracts.

Do not:

- parse the original DOCX/PDF again inside DataForge qualification;
- invent a second source parser;
- reconstruct Storage physical paths manually;
- treat text copied into a test fixture as the production provenance resolver;
- bypass existing source/run manifest integrity checks.

For production, resolve source evidence through existing durable Ingest/Storage references and freeze/checksum the evidence package used by qualification.

For unit tests, the design-pack JSONL contains source evidence. For one integration test, reuse an existing Ingest producer fixture/run artifact in the workspace when available, or create a small producer-compatible normalized fixture. Do not rebuild Ingest logic in DataForge.

---

## 11. Evaluation-set ownership in DataForge

DataForge owns creation/versioning/freeze of evaluation data; it does not score trained model candidates.

Add an immutable evaluation-set contract with at least these research roles:

- `exact_recall`
- `paraphrase_evaluation`
- `heldout_knowledge_unit`

An evaluation-set manifest must carry, as applicable:

- schema version;
- `evaluation_set_id`;
- version;
- research role;
- records URI and checksum;
- record count;
- originating dataset/research-package lineage;
- source record identity;
- fact/KU grouping identity where available;
- evidence/provenance references;
- generator model/prompt/version where generated;
- import/build policy;
- seed/pool/variant metadata where available;
- freeze checksum;
- creation timestamp;
- an explicit `training_eligible=false` / equivalent invariant.

### Exact-recall set

Automatically create an exact-recall evaluation set from accepted training records:

- same question/reference answer;
- distinct evaluation-record identity;
- preserve `source_record_id`;
- preserve fact/KU/evidence lineage;
- `research_role=exact_recall`;
- never optimizer-eligible.

Do not deduplicate the evaluation copy against its source training record solely because messages are identical; experimental role is different.

### Paraphrase and held-out-KU sets

For v1, implement an **import/validate/freeze** path for independently prepared canonical JSONL records.

Do not implement automatic 15-paraphrase generation in this task.

The import path must validate:

- record schema;
- unique IDs;
- provenance/evidence references where required;
- fact/KU/source-record grouping metadata;
- declared research role;
- `training_eligible=false`;
- supplied generator/prompt/pool/variant/seed metadata;
- overlap/leakage invariants declared by the package policy.

The historical `manual-v1` paraphrases in the design pack are a pilot/import fixture only, not a publication-grade statistical protocol.

---

## 12. DataForge research-package manifest

Add a small immutable convenience/lineage manifest linking:

- one training dataset manifest;
- its exact-recall set;
- zero or more frozen paraphrase evaluation sets;
- zero or more held-out-KU evaluation sets.

Suggested schema:

`cognityx.dataforge.research-package/v1`

Individual dataset/evaluation-set manifests remain authoritative. The research package is a frozen composition object, not a new storage system.

Expose enough public API/component CLI capability to:

- inspect the training dataset;
- inspect accepted/rejected/review counts and reason distributions;
- inspect evaluation sets and research roles;
- freeze/import an evaluation set;
- create/show a research package.

Do not redesign the global `cogni` CLI in this task.

---

## 13. Explicit split/research-role semantics

Do not overload ordinary train/validation/test language to mean all research evaluation types.

Preserve an explicit research role such as:

- `training`
- `exact_recall`
- `paraphrase_evaluation`
- `heldout_knowledge_unit`

Keep backward compatibility with existing `split` fields.

A compatible new representation may use:

- optimizer record: `split=train`, `research_role=training`;
- research evaluation record: `split=evaluation`, plus the specific `research_role`.

Existing historical DataForge `validation` and `test` splits must remain readable and must never accidentally enter training. Preserve their original split identity when normalized downstream.

---

## 14. DataForge manifest/statistics requirements

Extend qualified dataset/research-package manifests with useful counts and reason distributions, including where available:

- total generated candidates;
- accepted/rejected/needs-review counts;
- duplicate count;
- qualification-infrastructure failures;
- source/provenance resolution failures;
- question-unanswerable count;
- reference-does-not-answer count;
- missing-required-fact count;
- unsupported/contradicted claim counts;
- numeric/list/exact gate failures;
- research-role counts;
- evaluation-set references/checksums;
- exact-recall count;
- unique fact/KU count where that identity already exists;
- model-call counts/failures;
- schema/prompt versions.

Do not introduce misleading aggregate “quality scores” that can compensate for fatal hard-gate failures.

---

## 15. Mandatory DataForge regression fixtures

The frozen human/source-grounded oracle is:

`empirical/qualification/expected_qualification_v1.jsonl`

At minimum tests must prove:

- R05: accepted;
- R06: accepted;
- R08: accepted;
- R09: rejected at source-answerability/reference-fit level; no invented rewrite.

Also implement the design-pack deterministic mutations as stable regression cases, including:

1. numeric-role swap;
2. missing mandatory qualifier (`in the subject`-type constraint);
3. partial exact phrase;
4. number-word/digit normalization (`four` = 4 where semantics permit);
5. unresolvable provenance;
6. malformed semantic-validator structured output as infrastructure uncertainty, not factual incorrectness;
7. partial list plus unsupported members;
8. evaluation paraphrase accidentally marked trainable/leakage violation;
9. source-explicit policy with a correct direct threshold plus unsupported `or higher` addition — preserve direct-answer correctness separately from source-faithfulness, and reject it as a strict training reference under `source_explicit_only`;
10. question asks for specificity/authority absent from source.

Add explicit anti-anchoring tests that use spies/fakes to prove:

- Stage A does not receive gold/source/candidate;
- Stage B does not receive gold/candidate;
- Stage C receives only previously frozen outputs + reference;
- candidate model answers from `e3_source_grounded_cases_v2.jsonl` never leak into qualification planning.

Also test:

- raw `paragraph-qa` backward compatibility;
- `paragraph-qa-qualified` preserves the same pre-qualification candidates as the raw generator path under the same fixture/config;
- a single-source dataset still creates a non-empty exact-recall evaluation set;
- rejected/review records remain persisted;
- physical/evidence page and printed page remain distinct;
- checksum verification and duplicate-ID rejection;
- resume/checkpoint behavior if current DataForge build supports it.

---

# PART B — TRAINING

## 16. Training consumes DataForge research packages/evaluation sets

Extend Training additively so it can consume:

- existing DataForge dataset manifests as before; and
- optionally a DataForge research-package manifest and/or explicit evaluation-set manifest URIs.

Training must not reopen Ingest source documents or re-run DataForge qualification.

Required behavior:

1. stream optimizer-eligible training records separately from all research evaluation records;
2. never send `exact_recall`, `paraphrase_evaluation`, or `heldout_knowledge_unit` records to the optimizer;
3. accept historical `validation`/`test` DataForge records backward-compatibly as non-training evaluation records, preserving their `original_split`/equivalent metadata;
4. generate baseline and trained predictions for every configured evaluation suite using the existing Training evaluation mechanism, pending the later production-equivalent Inference adapter handoff;
5. keep per-suite results separate rather than collapsing exact recall/paraphrase/held-out knowledge into one unexplained average;
6. preserve evaluation-set IDs/versions/checksums and DataForge lineage in Training publication.

Do not turn Training into the future adaptive Evaluator. Simple deterministic diagnostics currently owned by Training may remain, but semantic source-grounded scoring is out of scope here.

---

## 17. Training prediction/evaluation lineage

Every saved baseline/trained prediction row for a research evaluation set should preserve, as applicable:

- evaluation-set ID/version;
- research role;
- evaluation record ID;
- source training record ID for exact recall;
- fact/KU group identity when available;
- question/reference;
- evidence/provenance metadata;
- base-model identity/revision;
- training experiment/variant/run IDs;
- decoding/runtime settings actually used;
- prediction identity.

Reports/metrics should expose per-suite record counts and simple metrics separately.

The later Inference adapter-handoff task will provide deployment-equivalent base-vs-adapter execution. Do not duplicate that runtime here beyond the existing Training evaluation path required for training/checkpoint diagnostics.

---

## 18. Fix the Training BatchEncoding evaluation defect

The built-in evaluation path must not assume `tokenizer.apply_chat_template(..., return_tensors="pt")` is a bare tensor.

Use the current Transformers-supported equivalent of:

```python
inputs = tokenizer.apply_chat_template(
    ...,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
inputs = {
    key: value.to(input_device) if hasattr(value, "to") else value
    for key, value in inputs.items()
}
output_ids = model.generate(**inputs, ...)
```

Use `inputs["input_ids"].shape[-1]` or the correct equivalent when slicing generated output.

Add CPU-only unit tests with fake tokenizer/model/BatchEncoding-like objects reproducing the failure mode. Do not require a GPU to prove this fix.

---

## 19. Fix Training reporting consistency

Ensure publication/reporting does not expose contradictory dataset counts such as a populated `split_summary` with empty `record_counts` when counts are known.

Add explicit counts for:

- training records;
- legacy validation/test records where present;
- each research evaluation suite;
- selected/accepted/skipped training records;
- evaluation records actually predicted.

Clearly distinguish `0` from unavailable/unknown.

Preserve telemetry scope semantics: process CPU versus whole-host CPU, process/WSL RAM versus host RAM, current/average/peak, GPU utilization versus VRAM, average/peak power and accumulated energy when those measurements actually exist. Do not invent unavailable metrics.

---

# PART C — EXPERIMENT TRACKING IN TRAINING

## 20. Tracking ownership principle

Cognityx Storage publication/manifests remain the **authoritative immutable research record**.

MLflow is an optional searchable/index/visualization/comparison layer.

Do not upload duplicate copies of large datasets, prediction JSONL, adapters or source artifacts into MLflow by default. Log their authoritative `storage://` URIs, checksums, IDs and schema versions.

DataForge does not need live MLflow coupling in this task. Its immutable manifests/research package are sufficient for the future orchestrator to index. Concentrate live metric emission where it has the highest value: Training.

---

## 21. Tracker-neutral Training interface

Add an internal tracker-neutral interface with a no-op default.

Requirements:

- ordinary Training installation must not require MLflow;
- add MLflow through an optional dependency/extra;
- no-op behavior must preserve all current Training workflows;
- tracking configuration must be explicit and validation-safe;
- tracking code must not become the source of experiment/variant/run identity.

Suggested configuration semantics (adapt to established config conventions rather than forcing these exact field names):

```toml
[tracking]
backend = "none"  # or "mlflow"
tracking_uri = "file:///..."  # optional
experiment_name = "..."       # optional/defaultable
run_name = "..."              # optional
parent_run_id = "..."         # optional cross-process parent
failure_policy = "warn"       # or "error"
```

Do not hard-code a global user tracking directory in tests.

---

## 22. MLflow event/content contract

At Training run start, log/index at least:

- Cognityx `experiment_id`;
- `training_variant_id`;
- `training_run_id`;
- dataset ID/version/manifest URI/checksum;
- research-package ID/URI/checksum if used;
- evaluation-set IDs/versions/roles/checksums;
- base model and resolved revision;
- tokenizer revision and chat-template checksum where available;
- seed;
- result-changing training parameters;
- package versions;
- Git revision;
- hardware/runtime identity available from existing environment reporting.

During training, emit only measurements already available or reliably computable, such as:

- global/optimizer step;
- training loss;
- examples processed;
- effective training tokens when the existing pipeline can measure them correctly;
- elapsed time;
- throughput where measured;
- GPU utilization/VRAM;
- process/host resource measurements with explicit scope;
- power/energy when measured by the existing telemetry path.

Do not estimate metrics and present them as measurements.

For evaluation/checkpoints/finalization, log per-research-role metrics with stable names and the corresponding suite identity.

On successful publication, log/index:

- `adapter_id`;
- adapter manifest URI and bundle checksum;
- publication manifest URI;
- training report URI;
- metrics URI;
- baseline/trained prediction URIs;
- final status.

On failure, record failure status/reason where possible without masking the original Training exception.

---

## 23. Tracking failure policy

Default `failure_policy=warn`:

- a tracking backend outage must not corrupt or invalidate an otherwise correct Storage publication;
- preserve/emit a clear warning and tracking-failure diagnostic;
- Training remains successful if its authoritative publication succeeds.

`failure_policy=error` may be selected explicitly for a strict experiment that requires tracker registration, but it must fail clearly and must not fabricate a completed Training publication if the underlying Training/publication failed.

---

## 24. Parent-child and backfill support

The future thin experiment orchestrator may create a parent MLflow run and launch Training in another process. Support an explicit `parent_run_id`/equivalent that does not depend on same-process global MLflow state.

Also provide a library function and/or small component CLI command to backfill/index a previously completed Training publication into MLflow **without rerunning training**.

Backfill must:

- read and validate the completed Training publication through normal Storage/publication contracts;
- log IDs, parameters, metrics and artifact references;
- not duplicate adapter/dataset binaries;
- be idempotent or detect duplicate registration safely;
- preserve the original training timestamps/identity as metadata rather than pretending backfill time was training time.

This will later be used for historical E1/E2/E3 runs.

---

## 25. Training tracking tests

Tests must not require an external MLflow server.

Cover at least:

- no-op tracker;
- fake tracker event ordering/content;
- optional local file-backed MLflow test when the optional dependency is installed;
- parent-run linkage semantics;
- warn-policy success when tracker fails;
- strict error-policy behavior;
- no large Storage artifact duplication by default;
- successful publication final tags/references;
- failure publication/tracker status;
- backfill of a frozen completed publication fixture;
- stable metric/tag naming;
- no regression when tracking is disabled.

---

# PART D — CROSS-REPOSITORY INTEGRATION FIXTURE

## 26. CPU/mock integration proof

Create or extend a small deterministic fixture proving, without requiring a GPU or live external model:

```text
producer-compatible source/evidence fixture
    → raw paragraph QA candidate path
    → qualified paragraph QA path
    → qualification decisions/artifacts
    → exact-recall evaluation set
    → imported frozen paraphrase evaluation set
    → research-package manifest
    → Training preflight/reader
    → optimizer receives training records only
    → fake baseline/trained evaluation predictions by research role
    → Training publication references
    → tracker receives IDs/metrics/Storage references
```

The proof must demonstrate:

- raw and qualified treatments share the same initial candidate-generation fixture/path;
- rejected qualification evidence is preserved;
- the same frozen paraphrase set can be reused by multiple treatments;
- evaluation records never enter optimizer batches;
- exact-recall records remain linked to their source training records;
- checksums are verified;
- research roles survive DataForge→Training handoff;
- Training prediction rows retain suite/provenance lineage;
- MLflow/tracker contains authoritative Storage references rather than duplicate artifact binaries.

Do not invoke a real Qwen model, consume GPU time, or call paid/remote APIs for this overnight implementation task.

---

# PART E — FAILURE-REGISTER TRACEABILITY

## 27. Produce a scope/traceability matrix

Read Failure Register v0.9 completely.

Create a concise design/implementation document mapping relevant F-IDs to:

- DataForge behavior/test implemented here;
- Training behavior/test implemented here;
- deferred Evaluator responsibility;
- deferred Inference/runtime responsibility;
- not applicable to this task.

Do not implement every F-ID merely because it exists.

The DataForge-relevant core includes, among others, failures concerning:

- missing mandatory qualifiers;
- defective/unanswerable questions and golds;
- list/set completeness and unsupported members;
- numeric role binding;
- source entailment versus question-answer fit;
- evidence-gold versus generated-gold;
- candidate-independent/candidate-blind planning;
- number-word normalization;
- source faithfulness/unsupported additions;
- evidence resolution;
- anti-gold-anchoring staged qualification;
- premise-restating gold detection;
- no fabricated rewrites;
- rewrite requalification.

Training-relevant failures include zero evaluation records, BatchEncoding evaluation input shape, split semantics, report counts, telemetry semantics and later base-vs-adapter reproducibility. The last item is only prepared for here; the first-class Inference adapter handoff remains out of scope.

---

# PART F — BACKWARD COMPATIBILITY AND NON-GOALS

## 28. Backward compatibility

Must continue to work:

- existing `paragraph-qa`;
- existing `knowledge-unit-qa`;
- existing probed DataForge recipe;
- existing DataForge manifests;
- existing DataForge dataset show/export;
- existing Training legacy JSONL mode;
- existing Training DataForge-manifest mode;
- existing local Training publication mode;
- existing Storage publication mode;
- existing component CLIs;
- existing tests unless a test encodes a verified bug being intentionally corrected.

New schemas must be versioned. Never silently reinterpret an old field with incompatible semantics.

Do not make production code depend on the design-input pack.

---

## 29. Do not solve unrelated platform issues

Do not implement in these PRs:

- global configuration redesign;
- global friendly handles;
- global pretty CLI rendering;
- unified `cogni dataforge`/`cogni train` surfaces;
- `cogni storage locate` (separate SDK task);
- Training→Inference adapter serving (separate Inference task);
- Inference manager/jobs redesign;
- new Storage URI parsing or backend routing;
- Jobs scheduler/queue redesign;
- OpenTelemetry platform work;
- Ingest adaptive routing/multimodal execution;
- an Evaluator repository;
- a new experiment orchestrator repository.

If one of these is a real blocker, report the exact blocker and the smallest follow-up instead of absorbing the responsibility into DataForge/Training.

---

# PART G — DOCUMENTATION, TESTS, AND DELIVERY

## 30. Documentation

Add/update repository documentation covering:

### DataForge

- raw versus qualified paragraph-QA semantics;
- staged anti-anchoring qualification;
- evidence-gold versus generated/reference gold;
- accepted/rejected/needs-review policy;
- qualification artifact schemas and reason codes;
- evidence/provenance resolution boundary;
- exact-recall/paraphrase/held-out-KU evaluation-set roles;
- evaluation-set freeze/import;
- research-package manifest;
- leakage guarantees;
- one fixture-based example.

### Training

- research-package/evaluation-set consumption;
- legacy split compatibility;
- per-suite prediction/report semantics;
- BatchEncoding fix behavior;
- optional tracker interface;
- MLflow reference-only artifact policy;
- parent-run support;
- historical publication backfill.

Document that adaptive semantic model evaluation remains a separate Evaluator concern and deployment-equivalent adapter inference remains a separate Inference concern.

---

## 31. Validation requirements

Run:

- focused new tests;
- full DataForge test suite;
- full Training test suite;
- docs/build/lint/type checks already required by each repository CI;
- package/build verification used by each repository.

Do not skip failures by weakening assertions. Fix production/test code or explicitly report an unrelated pre-existing failure with proof.

Avoid long GPU tests and remote service calls.

---

## 32. PR/commit structure

Prefer:

### DataForge PR

One focused branch/PR implementing:

- `paragraph-qa-qualified`;
- staged qualification;
- qualification artifacts/reason codes;
- evaluation-set freeze/import;
- exact-recall set;
- research-package manifest;
- tests/docs/traceability.

### Training PR

One focused branch/PR implementing:

- research-package/evaluation-suite consumption;
- safe split normalization;
- BatchEncoding evaluation fix;
- reporting consistency;
- optional tracker-neutral + MLflow integration;
- backfill;
- tests/docs.

Do not merge automatically. Leave both PRs reviewable and with CI status clearly reported.

---

## 33. Final implementation report

At completion, return a concise but complete report containing:

1. repository HEADs audited;
2. branches/commits/PR URLs or numbers created;
3. changed files grouped by repository;
4. public APIs/CLI/schema additions;
5. exact qualification stage contracts;
6. exact evaluation-set/research-package contracts;
7. backward-compatibility/migration decisions;
8. Failure Register F-ID traceability summary;
9. tests/CI commands and exact results;
10. anything not run and why;
11. unresolved risks/blockers;
12. deliberately deferred work;
13. exact copy-paste smoke commands for the next morning;
14. explicit statement whether the repository is now ready for the first `raw paragraph-qa` versus `paragraph-qa-qualified` controlled experiment.

If the final answer to item 14 is **no**, identify the minimum blocker. Do not hide it by adding more infrastructure.

---

# 34. Success criterion

This overnight task is successful if tomorrow morning we can work at the research level instead of writing more ad-hoc plumbing:

```text
DataForge:
source/run
  → raw paragraph dataset OR qualified paragraph dataset
  → immutable qualification evidence
  → exact-recall + frozen external evaluation sets
  → research package

Training:
research package
  → training only on optimizer-eligible records
  → separate evaluation-suite predictions
  → immutable adapter/publication lineage
  → searchable MLflow index/metrics referencing Storage
```

The future thin experiment harness should then need to own only:

- experiment specification;
- treatment/seed/budget orchestration;
- parent tracking run;
- invocation of DataForge/Training/Inference/Evaluator;
- paired statistical comparison and stop/go decisions.

It must **not** need to own DataForge qualification, split/evaluation-set generation, tokenizer/model execution, adapter loading, telemetry collection, or Training metric parsing.

Work to that boundary and stop.
