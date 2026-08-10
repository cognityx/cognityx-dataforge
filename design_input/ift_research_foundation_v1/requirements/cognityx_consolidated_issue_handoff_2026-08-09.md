# Cognityx consolidated issue handoff — 2026-08-09

This supersedes the earlier canonical issue register and carries forward both the older open backlog and the issues discovered during the LunaVane Training/Evaluator experiments on 2026-08-09.

## 0. Current experimental checkpoint

- DataForge source dataset: LunaVane paragraph-QA, 29 records.
- Current controlled training subset: first 10 records.
- E1: 10 steps (~1 exposure/example): exact recall 0/10, token F1 ~0.054.
- E2: 50 steps (~5 exposures/example): exact recall 0/10, token F1 ~0.399.
- E3: 100 steps (~10 exposures/example): exact recall 9/10, token F1 ~0.977.
- Frozen manual paraphrase pilot (`manual-v1`), one paraphrase per fact:
  - E2 paraphrase: exact 0/10, token F1 0.384.
  - E3 paraphrase: exact 4/10, token F1 0.645.
  - E3 generalization gap: exact +0.50, token F1 +0.332.
- Current adaptive evaluator v1 on E3 paraphrases:
  - correct: 4
  - partial: 2
  - incorrect: 1
  - likely_correct: 1
  - degenerate: 1
  - reference_issue: 1
  - semantic judge required: 4/10, but one of those four is a gold/reference-quality case, leaving 3 candidate semantic adjudications.
- Adjudication queues already prepared:
  - 3 semantic candidate cases
  - 1 reference-quality case
- Next intended experiment step: run Qwen3-32B only on the 3 semantic-judge cases through Cognityx Inference; handle the reference-quality case by source-grounded validation rather than candidate judging.
- Do not start E4/200-step training yet.

---

# A. GLOBAL / CROSS-REPOSITORY ISSUES

## G-01 — One canonical configuration model
Use one primary repo-level config file, then environment overrides, then CLI overrides. Do not create a confusing stack of multiple normal config files.

Required behavior:
- effective config can be printed;
- each overridden value identifies whether it came from file, environment, or CLI;
- `~/.config/cognityx/<repo>` can be the normal user-level location;
- Codex/tests must be able to create isolated project-local config without disturbing the user's real config.

## G-02 — Unified `cogni` CLI
Long-term user-facing CLI should be `cogni`, while component CLIs remain implementation/admin surfaces.

Needs:
- friendly handles;
- consistent verbs/status/output;
- pretty human output plus machine-readable JSON;
- old/ambiguous commands moved to a clearly deprecated section.

## G-03 — Friendly handles vs durable identifiers
Human-facing IDs should be simple handles. Durable Storage URIs, hashes, experiment/variant/run IDs remain authoritative internal/reproducibility identifiers.

## G-04 — Global Storage URI resolution
Need:
`cogni storage locate <storage-uri>`

It should resolve backend/profile/existence/size and physical/native location when meaningful. `StorageRuntime.locate()` exists in newer Storage, but cross-repo dependency skew has exposed older versions without it.

## G-05 — Cross-repo dependency/version skew
Repos can currently run against incompatible versions of Cognityx sibling packages. Need a consistent development/release/versioning strategy so features such as Storage `locate()` are not present in one repo and absent in another environment.

## G-06 — End-to-end lineage contract
Source asset → Ingest run → provenance/evidence → DataForge experiment/variant/run/dataset/version → Training experiment/variant/run/adapter → Evaluator run must remain machine-traceable.

## G-07 — Artifact freeze/checksum discipline
Research/evaluation sets, outputs, reports, adapter identities, evaluator plans and judge outputs should be immutable/frozen and checksum-addressable.

## G-08 — Uniform background-job contract
Long-running Cognityx operations should not depend on long synchronous HTTP calls.

Target pattern:
submit → job_id → durable state/events → watch/reconnect → completed/failed/cancelled.

Use a common event model and consistent progress semantics across repos.

## G-09 — Uniform event/stream semantics
Jobs should expose durable/replayable events, sequence numbers and reconnectable streaming. Avoid one-off timeout loops for long operations.

## G-10 — Resource-aware execution
Long-term scheduler/resource layer must reason about CPU/RAM/GPU/VRAM, model residency, KV cache, load cost, leases, training exclusivity and eventually power/energy.

## G-11 — Executable pipeline documentation
Docs should show one unambiguous supported happy path from source → Ingest → DataForge → Training → Evaluator → Inference, with legacy paths marked deprecated.

## G-12 — Shared neutral Evaluator
Evaluator should not be buried inside Training or DataForge. DataForge owns creation/versioning/freeze of evaluation datasets; Evaluator executes comparisons/scoring; Inference executes models and provides telemetry.

---

# B. COGNITYX-INFERENCE

## INF-01 — Consolidate lifecycle under the Manager
Current architecture exposes overlapping normal paths:
- manager/server lifecycle;
- direct worker `model load/unload`;
- direct `serve`.

This is confusing.

Target:
- Manager is the normal always-running control plane.
- Direct `serve` remains internal/debug/external-supervisor mode.
- Normal model lifecycle goes through the Manager.

## INF-02 — Manager currently manages worker lifecycle but model load can bypass it
`server start --profile ...` is already job-driven and event-driven, starts the worker, waits for API readiness, and loads the configured model in the background.
But `model load --base-url <worker>` still calls the worker synchronously.

Unify these.

## INF-03 — All model lifecycle operations must be jobs
`model load`, `model unload`, replacement/eviction, discovery and worker start/stop should use the same job/event/watch contract.

No generic HTTP timeout should determine success of a model load.

## INF-04 — 120-second CLI timeout defect
A Qwen3-32B load continued successfully server-side while CLI failed with `TimeoutError` after the generic client timeout.
This is a control-plane defect, not a model failure.

## INF-05 — Resource-aware residency / eviction policy
Before loading a model:
- inspect certified hardware profile/resource envelope;
- inspect actual current free/reserved VRAM;
- account for resident models/KV cache/headroom;
- reuse an existing compatible resident model;
- keep models in parallel if they fit;
- otherwise automatically select an eviction/replacement policy.

Expose explicit policies such as:
- auto;
- prefer_parallel;
- replace/force_replace;
- require_fit_without_eviction.

Do not blindly keep all models or blindly unload all models.

## INF-06 — Manager's one-worker design vs multi-resident ModelManager
Current manager explicitly manages exactly one local worker, while worker `ModelManager` can track multiple resident models. The control plane needs an explicit intended topology and resource policy.

## INF-07 — Split server/worker profile from model profile
Current `[server_profiles.<name>]` includes model/backend/load profile/certification. This conflates:
- worker/process placement/configuration;
- model/resource/certification configuration.

As multi-model residency grows, worker profile and model profile should become separate concepts.

## INF-08 — Manager status can become stale when worker is manipulated directly
Direct load/unload or worker lifecycle actions can bypass manager state. Once manager becomes canonical, direct mutation should be internal/admin-only or reconciled explicitly.

## INF-09 — Standard manager-owned worker logs
Need consistent:
- log location;
- `server logs` / watch surface;
- startup/load failure diagnostics;
- no need to inspect a hidden spawned process manually.

## INF-10 — Training → Inference adapter handoff missing
Inference currently lacks a proper LoRA/PEFT adapter loading/serving contract exposed through its normal CLI/API.

Need:
- adapter identity/Storage URI;
- base-model compatibility validation;
- load/unload/hot-select semantics;
- telemetry showing base + adapter identity;
- proper Training experiment/variant/run lineage.

Current experiments use direct Transformers+PEFT as a workaround.

## INF-11 — Base-vs-adapter generation equivalence
When Evaluator compares base and adapter, runtime/chat template/quantization/reasoning mode/sampling/max tokens must be identical except for the adapter.

## INF-12 — Model load/discovery should remain certification-aware
Existing hardware certification/discovery is useful and should be reused, not bypassed, by the new resource-aware manager.

## INF-13 — Current immediate state
Qwen3-32B INT4 load was started on worker `http://127.0.0.1:8100`; CLI timed out, but model was still loading. Do not interpret the client timeout as load failure.

---

# C. COGNITYX-TRAINING

## TRN-01 — Built-in evaluation input bug
Current `_evaluate_model` assumes:
`tokenizer.apply_chat_template(..., return_tensors="pt")`
returns a tensor and accesses `.shape` / passes it as `input_ids`.

With current Transformers/tokenizer it may return `BatchEncoding`.

Correct pattern:
- `return_dict=True`;
- move tensor values to device;
- call `model.generate(**inputs, ...)`.

## TRN-02 — Bug was hidden by zero evaluation records
The LunaVane DataForge dataset had 29/0/0 train/val/test because group-aware splitting placed the one source-asset group entirely in train. Thus the built-in evaluation path was not exercised.

## TRN-03 — Dataset split policy must be intentional for training research
Do not rely on accidental source-group splitting when the research requires frozen recall/paraphrase/generalization sets.

Need separate concepts:
- training set;
- exact-training-question recall set;
- held-out paraphrase set;
- held-out fact/KU generalization set.

## TRN-04 — Static config vs per-run experiment parameters
Static/system settings belong in config; dataset, max steps, experiment IDs, variant IDs etc. should be run inputs/overrides rather than permanent user config.

## TRN-05 — Lazy model-specific chat formatting
DataForge should keep canonical conversational `messages`; Training should apply model-specific chat templates/tokens lazily.

## TRN-06 — Training report lineage inconsistency
Final report includes `split_summary`, while `record_counts` can remain `{}`. Fix reporting consistency.

## TRN-07 — Resource telemetry semantics
Reports must clearly distinguish:
- process CPU vs whole-host CPU;
- process/WSL RAM vs Windows-host RAM;
- peak vs average;
- GPU utilization vs VRAM;
- average/peak power and total energy.

## TRN-08 — Adapter publication/handoff
Adapters are already stored with experiment/run lineage, but Inference must consume them through a first-class contract.

## TRN-09 — Exposure research should remain controlled
E1/E2/E3 changed only max steps. Preserve this controlled-series discipline.

Do not run 200 steps yet; current priority is evaluator/data-quality/generalization analysis.

---

# D. DATAFORGE

## DF-01 — Generation is not acceptance
Generated QA/KU records must not automatically become training records.

Need post-generation qualification gates.

## DF-02 — Source-grounded gold validation
Every candidate should preserve two concepts:
- evidence/extractive gold: exact evidence span(s), provenance;
- generated/reference gold: natural-language answer derived from that evidence.

Generated gold must be validated against evidence before acceptance.

## DF-03 — Current Q9 defective gold
Question asks what approval is required before exceeding the overtime threshold, but gold merely says approval is required and does not identify the approving authority.

This is a concrete DataForge QA-quality failure.

## DF-04 — Hard acceptance gates
Candidate qualification should include:
- provenance resolvable;
- question answerable from evidence;
- gold supported by evidence;
- required facts complete;
- contradiction-free;
- no unsupported additions;
- answer/reference quality sufficient.

Soft scores can rank accepted candidates but must not rescue hard failures.

## DF-05 — Same-model validation is correlated
Useful tiers:
1. deterministic validation;
2. same-model independent critique;
3. stronger cross-model validation;
4. adjudication on disagreement.

Planned experiment: Qwen3-8B generation, unload, Qwen3-32B source-grounded validation, freeze accepted dataset.

## DF-06 — Paraphrase pool protocol
Current `manual-v1` one-paraphrase-per-fact test is only a pilot.

Starting protocol:
- 15 paraphrases per fact/KU;
- 10 training-eligible;
- 5 frozen evaluation-only;
- evaluation variants never enter training.

Publication-quality evaluation should score all five and report mean/variance/worst case/failure distribution. Seeded random one-of-five is only for smoke tests.

## DF-07 — Paraphrase leakage control
Train/eval paraphrase membership must be persisted in lineage. Paraphrases of the same fact must not accidentally cross boundaries contrary to protocol.

## DF-08 — Paraphrase robustness vs factual generalization
Keep separate:
- same trained fact, unseen wording = paraphrase robustness;
- unseen fact/KU = factual generalization.

## DF-09 — DataForge-generated evaluation sets
Eventually DataForge, not a hand-written script, should create/version/freeze the exact recall and paraphrase evaluation sets, including generator identity and provenance.

## DF-10 — Token telemetry semantics
Clarify DataForge `token_budget.input_tokens` versus actual prompt-token usage so cost/telemetry reports are not misleading.

## DF-11 — Dataset inspection UX
Need easy CLI/Python inspection of:
- records;
- splits;
- provenance;
- recipe/variant/run;
- accepted/rejected counts and reasons.

---

# E. COGNITYX EVALUATOR / META-EVALUATOR

No permanent repo should be rushed yet. Current strategy is temporary scripts + frozen artifacts until the abstraction is proven.

Latest living failure register: `cognityx_adaptive_evaluator_failure_register_v0_3.md` with append-only F-001…F-059.

## EV-01 — Two mandatory post-IFT evaluation levels
1. Exact training-question recall.
2. Frozen paraphrase set excluded from training.

Later:
3. held-out fact/KU generalization;
4. provenance correctness / joint grounded correctness.

## EV-02 — Base vs adapter comparison
Every evaluation should preserve base and adapter outputs, raw metric vectors, model/runtime identity and provenance.

## EV-03 — No universal fixed weighted score
Different answer types need different fatal gates and metric families. Do not let a high soft score compensate for a fatal correctness error.

## EV-04 — Answer-type-aware semantic planning
Planner should infer from question + gold/task schema:
- semantic task;
- answer type;
- mandatory atoms;
- role bindings;
- hard gates;
- appropriate soft metrics;
- whether semantic/LLM adjudication is needed.

Plan must be candidate-independent.

## EV-05 — Current heuristic planner is paraphrase-brittle
Pilot audit:
- exact-question semantic reference match: 10/10;
- paraphrase match: 7/10;
- planner drift: 3/10 = 30%.

Observed mismatches:
- document classification: `exact_string` → `short_factual`;
- referenced document identifier: `exact_string` → `short_factual`;
- policy coverage list: `list_or_set` → `short_factual`.

Treat 30% as a pilot n=10 result, not a universal rate.

## EV-06 — Exact-match false negatives
Semantically correct reformulations can fail exact match.
Example: 23-knot sustained / 31-knot gust rule.

## EV-07 — Lexical metrics can hide missing mandatory qualifiers
Example:
gold: `require exactly one weather noun in the subject`
candidate: `require exactly one weather noun`

High lexical precision/F1 does not make this fully correct.

## EV-08 — Partial exact phrase recall
`glass mango` vs `glass mango over silent river` is partial, not correct.

## EV-09 — Smoothed BLEU can give credit to clearly wrong exact-label/string answers
Metric behavior is mathematically legitimate but unsuitable as a correctness gate for such tasks.

## EV-10 — List/set answers need member-level precision/recall
Current paraphrase policy-scope candidate recovered some valid members but also invented unsupported members. Need missing-member and unsupported-member gates.

## EV-11 — Numeric rules need role binding
Presence of both numbers is insufficient; each must be bound to sustained/gust/threshold/unit/logical relation correctly.

## EV-12 — Degenerate output needs a hard gate
Current E3 paraphrase Q10 produced repeated `PEOPLEPEOPLE...`. Detect repetition/empty/malformed/pathological output separately from semantic metrics.

## EV-13 — Reference/gold defects are a distinct class
Do not score a model against a known defective gold. Route to source-grounded reference review.

## EV-14 — Atom normalization defects
Evaluator v1 marked Brass-day `paid_hours_minimum=4` false even though candidate said `four`.
Need numeric word/number normalization and unit normalization.

## EV-15 — Internal evaluator diagnostics must be consistent
A record should not be labeled globally correct while unexplained lower-level mandatory atom checks show false. Either normalize/fix atom checks or explicitly document override precedence.

## EV-16 — Aggregate scores hide heterogeneous failures
`4/10 exact` can contain:
- truly correct;
- semantically correct reformulation;
- partial;
- incorrect;
- degenerate;
- bad reference.

Preserve per-record classifications.

## EV-17 — Metric disagreement is evidence
Large disagreement between exact/atom/set/numeric/semantic/judge outputs should trigger adjudication, not averaging.

## EV-18 — LLM judge should be gated
Do not call an LLM judge for everything.

Current E3 paraphrase cascade:
- deterministic layer resolves most cases;
- 3 candidate cases need semantic adjudication;
- 1 separate reference-quality case.

## EV-19 — Strong judge for unresolved subset
Planned current step: Qwen3-32B judge only on 3 semantic cases:
- R05 manager rule;
- R06 policy coverage;
- R08 wind rule.

Expected labels should be judged empirically, not hard-coded.

## EV-20 — LLM judge output contract
At minimum:
- correct / partially_correct / incorrect;
- factual score 0–4;
- satisfied requirements;
- missing/wrong requirements;
- unsupported claims;
- concise rationale;
- confidence.

## EV-21 — Evaluator itself must be evaluated
A static evaluator learned from 10 questions risks overfitting.

Need a meta-evaluator/bootstrap stage for every new corpus/domain.

## EV-22 — Meta-evaluator bootstrap
Proposed lifecycle:
1. Ingest corpus.
2. DataForge builds representative QA/provenance probes.
3. Infer semantic correctness structures.
4. Generate controlled metamorphic/adversarial stress cases.
5. Exercise available evaluator primitives.
6. Discover/cluster evaluator failures.
7. Synthesize declarative Evaluation Plan/IR.
8. Validate plan on held-out stress probes.
9. Freeze evaluator plan.
10. Only then make model-quality claims.

## EV-23 — Stress/metamorphic probes
Automatically generate controlled variants:
- paraphrases;
- omitted qualifiers;
- partial exact phrases;
- swapped numeric roles;
- wrong units;
- missing/extra list members;
- contradictions;
- correct answer + wrong provenance;
- wrong answer + correct provenance;
- degenerate repetition;
- malformed output;
- deliberately defective reference.

## EV-24 — Do not self-modify arbitrary runtime code
Meta-evaluator should synthesize a declarative plan from an approved primitive library.

If a primitive is missing:
`capability_gap=true`
with a proposed primitive requiring explicit implementation/review.

## EV-25 — Evaluator-plan invariance as a metric
Meaning-preserving paraphrases should normally produce materially equivalent evaluation plans. Measure planner invariance directly.

## EV-26 — Provenance evaluation
Future targets:
A. Question → Answer
B. Question → Provenance
C. Question → Answer + Provenance
D. Question → predicted provenance → deterministic source retrieval → Answer

Metrics:
- doc/page/evidence-id accuracy;
- Recall@k;
- retrieval correctness;
- answer correctness;
- Joint Grounded Correctness = answer correct AND provenance correct.

## EV-27 — Evidence package preservation
Every evaluator output should retain:
question, gold, candidate, provenance, plan, deterministic scores, judge output/rationale, model/runtime/config identities and checksums.

---

# F. COGNITYX-JOBS / EVENTS / RESOURCE CONTROL

## JOB-01 — Jobs is still too minimal for the intended platform
Current Jobs repository is essentially a minimal SQLite state/event repository. Long-term architecture needs a reusable job-control contract.

## JOB-02 — Standard lifecycle/state model
Need consistent lifecycle across Inference, DataForge, Training and other long operations:
queued/running/waiting/completed/failed/cancelled/interrupted etc., with attempts and timestamps.

## JOB-03 — Durable event schema
Use CloudEvents-compatible event semantics where practical and preserve:
- job_id;
- attempt;
- worker;
- timestamps;
- progress;
- sequence;
- parent/child linkage;
- execution context.

## JOB-04 — Better progress events
Existing DataForge events are sparse. Need meaningful progress denominator/numerator, per-stage/per-call detail where useful, and attempt/worker identity.

## JOB-05 — Observability integration
OpenTelemetry traces/metrics/logs should correlate with Jobs/events and cross-repo run IDs.

## JOB-06 — Separate orchestration concerns
Keep conceptually distinct:
- trigger/schedule;
- queue/admission;
- resource placement;
- execution.

Local can remain lightweight; enterprise implementations can map to Postgres/NATS/Kafka/K8s/Kueue/Ray/optional Temporal without forcing those dependencies into local mode.

## JOB-07 — GPU-aware placement later
Resource placement should eventually consider:
- VRAM;
- resident model;
- KV cache;
- load/unload cost;
- active leases;
- training exclusivity;
- power/thermal considerations.

---

# G. COGNITYX-STORAGE

## STOR-01 — Storage remains the deduplicating authority
Do not reintroduce user-selected `--storage-root` as a normal workflow. CAS/object storage should own deduplicated physical copies.

## STOR-02 — Global URI locate
Carry forward G-04 as a Storage-owned capability plus unified CLI exposure.

## STOR-03 — Cross-repo version compatibility
Storage client/runtime versions must be synchronized enough that downstream repos do not silently lose newer APIs.

## STOR-04 — Immutable experiment/evaluator artifacts
Continue storing adapters, reports, manifests, datasets and frozen evaluation packages through durable Storage identities.

---

# H. COGNITYX-INGEST

Core Ingest v3.2 T00–T10 and P30 documentation work are considered complete for the current experiment path.

Remaining/deferred issues:

## ING-01 — Adaptive routing execution bridge
Capability/planning abstractions exist, but the concrete runtime bridge for deterministic/hybrid/LLM-directed extraction remains deferred.

## ING-02 — LLM-native multimodal extraction
Routing must eventually identify and execute/orchestrate LLM-native extraction for:
- handwriting/signatures;
- images;
- charts;
- tables where needed;
- audio/video;
- other multimodal regions.

## ING-03 — Capability discovery/benchmarking
Available local/remote extraction capabilities should be discoverable and empirically benchmarked through Cognityx Inference, not hard-coded.

## ING-04 — Dynamic allocation of autonomy
Future agentic ingestion should decide when deterministic workflow is sufficient and when uncertainty/risk justifies LLM/agent autonomy.

Do not reopen these while the current priority is Evaluator/Training research unless an experiment requires them.

---

# I. COGNITYX-SDK / CLIENT SURFACES

## SDK-01 — Unified CLI/facade continuity
SDK should expose stable user-level flows while component repos retain technical surfaces.

## SDK-02 — Current adaptive routing execution gap
Merged SDK support can read Ingest v3.2 artifact/provenance surfaces, but there is no concrete adaptive-routing proposal/execution bridge yet.

## SDK-03 — Preserve stable ResourceContext / fresh ExecutionContext discipline
Carry this design through new DataForge/Training/Evaluator client surfaces.

---

# J. DOCUMENTATION / RESEARCH RECORD

## DOC-01 — Maintain the failure register
Latest: `cognityx_adaptive_evaluator_failure_register_v0_3.md`.

Numbering is append-only. New colleague-observed failures start after the current latest ID; never renumber historical F-IDs.

## DOC-02 — Collect colleague failure cases
For each case capture:
- symptom/example;
- expected behavior;
- observed behavior;
- model/environment;
- whether primarily data/evaluator/model/runtime/ambiguous;
- final resolution.

## DOC-03 — Preserve meta-evaluator design
Current design note:
`cognityx_meta_evaluator_bootstrap_v0_1.md`.

## DOC-04 — Evidence for possible publication
Do not claim novelty yet, but preserve:
- exposure → recall/generalization curves;
- exact vs paraphrase gap;
- failure taxonomy;
- evaluator-plan drift;
- raw vs validated-data experiments later;
- peak resource envelope;
- runtime/power/energy;
- accuracy per joule/per second;
- provenance-learning experiments.

---

# K. IMMEDIATE NEXT STEPS IN THE NEW CHAT

1. Finish the currently-started Qwen3-32B load; do not mistake the prior CLI timeout for a load failure.
2. Run the frozen 3-case semantic judge queue using `run_semantic_judge.py`.
3. Inspect judge outputs rather than assuming expected labels.
4. Resolve R09 through source-grounded reference review.
5. Merge deterministic + semantic + reference-review outputs into one final E3 paraphrase evaluation result.
6. Freeze/checksum the evaluator artifacts.
7. Only after that decide whether to:
   - expand to multi-paraphrase pools;
   - build the first DataForge validation/correction experiment;
   - formalize the Evaluator repo.
8. Separately create an Inference implementation issue/plan for:
   - manager-owned single lifecycle path;
   - job-driven model load/unload;
   - resource-aware residency/eviction;
   - server/model profile separation;
   - no generic synchronous load timeout.
