# Cognityx Adaptive Evaluator — Failure Register v0.1

Status: living empirical register  
Purpose: preserve observed and anticipated evaluation/data-quality failure modes that motivate an adaptive, gated, answer-type-aware evaluator.  
Numbering: append-only. New observations should receive the next F-ID rather than renumbering existing items.

## Failure points

F-001 — Exact-match false negative for semantically equivalent answers.  
A factually correct reformulation can fail exact match even when all required information is preserved. The E3 paraphrase wind-rule answer is the current worked example.

F-002 — Smoothed BLEU can award non-zero credit to categorically wrong exact-string answers.  
Observed in E1. Smoothing behavior is mathematically legitimate for BLEU but misleading for label/code/title/phrase correctness.

F-003 — High lexical overlap can hide a missing mandatory qualifier.  
Example: `require exactly one weather noun` versus gold `require exactly one weather noun in the subject`. Precision and F1 are high, but the location constraint is missing.

F-004 — Partial phrase recall must not be treated as semantic correctness for exact phrases.  
Example: integrity phrase `glass mango` versus `glass mango over silent river`. This is partial learning, not a valid exact integrity phrase.

F-005 — Gold/reference answer can itself be incomplete or defective.  
Current Q9 asks what approval is required but the gold merely states that approval is required, without identifying the authority.

F-006 — A single numeric score has different meaning across answer types.  
For an exact identifier, F1=0.7 may still mean wrong; for a descriptive answer it may indicate useful partial correctness.

F-007 — Lists require set/coverage evaluation, not only sequence overlap.  
Ordering may be irrelevant while omitted or invented list members are critical.

F-008 — Numeric rules require role/relationship binding, not mere number presence.  
A candidate containing both 23 and 31 can still be wrong if sustained/gust roles are swapped.

F-009 — Paraphrase sensitivity creates a measurable generalization gap.  
E3 exact-question recall reached 90% exact while the frozen paraphrase set reached 40% exact.

F-010 — Prompt paraphrase can trigger generation degeneration/repetition.  
E3 paraphrase Q10 produced repeated `PEOPLE...` despite substantially learned behavior on the exact training question.

F-011 — Learning can be non-monotonic at the per-question/prompt level.  
A fact can be recalled exactly for the training wording yet fail badly under a semantically equivalent prompt.

F-012 — Aggregate averages hide heterogeneous failure modes.  
`4/10 exact` mixes exact successes, semantically correct reformulations, partial recall, wrong answers, gold defects, and degenerate output.

F-013 — Hallucinated answers can receive accidental lexical overlap.  
Token overlap alone does not prove factual correctness.

F-014 — Question-echo/reference leakage can inflate lexical scores.  
If the gold repeats much of the question, F1/ROUGE may appear strong without answering the requested information.

F-015 — Fixed weighted sums can compensate for fatal failures.  
A wrong provenance, missing mandatory qualifier, contradiction, or wrong exact identifier should not be rescued by high soft similarity scores.

F-016 — One paraphrase per fact is statistically weak and sensitive to wording choice.  
The current manual-v1 single-paraphrase experiment is a pilot, not the final methodology.

F-017 — Manually chosen paraphrases can introduce evaluator-author bias.  
Paraphrase pools should be generated, frozen, versioned, deduplicated, and sampled using a recorded seed/policy.

F-018 — Paraphrase leakage must be distinguished from factual generalization.  
Held-out paraphrases of a trained fact test wording robustness; held-out facts/KUs test knowledge generalization. These must be separate evaluation levels.

F-019 — Same-model generation and judging have correlated-error risk.  
Same-model self-check is still useful as one gate, but high-confidence acceptance should prefer cross-model or deterministic/source-grounded validation.

F-020 — DataForge candidate generation currently needs post-generation source-grounded validation before training acceptance.  
A structurally valid QA pair can contain an incomplete, unsupported, or poorly targeted gold answer.

F-021 — Evidence-gold and generated-gold are different objects.  
The system should preserve extractive/evidence-supported gold separately from a natural-language generated reference answer.

F-022 — Answer correctness and provenance correctness are independent.  
The evaluator should support answer-only, provenance-only, and joint grounded correctness.

F-023 — Correct provenance does not guarantee correct semantic interpretation.  
A cited source may be real while the inferred relation, exception, qualifier, or answer is wrong.

F-024 — Training built-in evaluation was effectively untested in E1 because the dataset had zero evaluation records.  
A dormant code path can therefore appear healthy while containing runtime bugs.

F-025 — Training evaluation input-shape bug.  
The current evaluation code assumes `apply_chat_template(..., return_tensors="pt")` is a tensor; with the current tokenizer/Transformers stack it may be a BatchEncoding.

F-026 — Training-to-Inference adapter handoff is missing.  
Current Inference CLI/API does not expose adapter loading, forcing ad-hoc PEFT scripts and making runtime-equivalent evaluation harder.

F-027 — Base and adapter generation settings must remain identical.  
Different runtime, chat template, quantization, reasoning mode, sampling, or token limits can confound the measured training effect.

F-028 — Resource telemetry semantics can be misread.  
Process CPU and whole-host CPU are different measures; host RAM and process RAM are different measures. Reports must preserve scope.

F-029 — Exact accuracy alone hides the learning trajectory.  
E2 had 0/10 exact but token F1 around 0.399, while E3 transitioned to 9/10 exact and F1 around 0.977.

F-030 — Metric implementation/version choices affect conclusions.  
BLEU smoothing, normalization, tokenization, stopword policy, semantic model, and judge version must be frozen and recorded.

F-031 — Keyword-based answer-type classification is brittle under paraphrase.  
The same semantic task can receive a different evaluator plan when wording changes.

F-032 — Observed planner inconsistency: document classification.  
Exact wording `What is the classification...` was classified `exact_string`; paraphrase `How is this document classified?` was classified `short_factual`.

F-033 — Observed planner inconsistency: missing-document name.  
Exact wording was classified `exact_string`; paraphrased wording was classified `short_factual` even though the expected answer remains an exact document name/version.

F-034 — Observed planner inconsistency: policy coverage list.  
Exact wording was classified `list_or_set`; paraphrased wording was classified `short_factual` even though the gold remains a list/set.

F-035 — Evaluator planning should be candidate-independent.  
Metric/gate selection should be derived from question + gold/task schema before seeing the candidate answer, otherwise the evaluator can adapt its standard to the answer being judged.

F-036 — Required factual atoms can be more informative than global similarity.  
For policy rules, correctness may depend on mandatory atoms such as quantity, object, location, authority, temporal scope, exception, and relation.

F-037 — Degenerate-output detection is a separate gate, not a similarity metric.  
Repetition loops, empty output, malformed structure, and pathological token cycling should be classified before semantic scoring.

F-038 — Evaluation should preserve per-record evidence, not only aggregate metrics.  
Full question, gold, candidate, provenance, plan, scores, judge rationale, and model/config identity are required for forensic interpretation.

F-039 — A universal LLM judge is not sufficient.  
For exact codes, numeric thresholds, lists, provenance IDs, and structural constraints, deterministic validators can be more reliable and cheaper.

F-040 — Disagreement between metrics is itself evidence.  
Large disagreement among exact, atom coverage, lexical, semantic, and judge outputs should trigger adjudication rather than averaging.

## Provisional paraphrase protocol

The current single manually curated paraphrase per fact remains a pilot artifact.

For the next DataForge/training correction phase, use a versioned paraphrase pool per atomic fact/KU. A practical starting design is 15 paraphrases per fact: 10 eligible for training and 5 frozen as paraphrase-evaluation variants. The five evaluation paraphrases must never enter training. Report performance over all five; a seeded random single-paraphrase sample may be used only for quick smoke tests.

Paraphrase robustness and factual generalization must remain separate:
- Paraphrase robustness: same fact/KU, unseen wording.
- Factual generalization: unseen fact/KU, grouped so no paraphrase of the same fact crosses the factual train/test boundary.

Every paraphrase should retain the same evidence/provenance link and carry generator model, prompt/version, validation status, pool ID, variant ID, split role, and freeze checksum.

## Future additions from colleagues

Add colleague-observed failures as F-041 onward. Preserve the original symptom/example, environment/model, expected behavior, observed behavior, and whether the failure is an evaluator failure, data-quality failure, model failure, runtime failure, or ambiguous.

F-041 — Quantified evaluation-plan drift under paraphrase.  
In the first 10-record E3 planner-consistency audit, changing only the question wording changed the selected answer type and/or primary metric plan for 3/10 records (30%). The underlying fact, gold answer, provenance, and intended task semantics were unchanged. The observed mismatches were: document classification (`exact_string` → `short_factual`), exact referenced-document name (`exact_string` → `short_factual`), and policy coverage (`list_or_set` → `short_factual`). This is direct empirical evidence that the current keyword/rule planner is wording-sensitive.

F-042 — Coarse answer-type taxonomy can itself be insufficient.  
The audit shows that even a stable coarse label such as `short_factual` can hide materially different correctness structures. Exact labels, exact phrases, exact identifiers, lists/sets, numeric threshold rules, definitions, and constraint rules require different hard gates and metric families.

F-043 — Evaluation-plan correctness needs an independent reference target.  
Planner quality cannot be measured by comparing one planner to itself. A small human-reviewed semantic reference plan should define expected answer type, mandatory factual atoms/constraints, and admissible metric families for benchmark records; heuristic and LLM planners can then be evaluated against that frozen reference.

F-044 — Planner invariance under semantic-preserving paraphrase is a measurable evaluator property.  
For the same fact and gold answer, a good evaluation planner should normally produce the same correctness structure and materially equivalent metric/gate plan across paraphrases. Plan-invariance rate can therefore become an evaluator-quality metric in its own right.

F-045 — Reference-atom normalization can fail even when the answer is exactly correct.  
In Adaptive Evaluator v1, the exact correct Brass-day answer was labeled correct because exact match succeeded, but the atom `paid_hours_minimum=4` was marked false because the candidate said `four`. Numeric word/number normalization is therefore required before atom evaluation.

F-046 — Atom extraction and global exact-match can disagree internally.  
A record can receive a final `correct` label while individual atom checks show false/unknown. Internal evaluator diagnostics must be self-consistent or explicitly explain why a higher-priority gate overrides lower-level atom checks.

F-047 — Partial list overlap can hide unsupported invented members.  
For the paraphrased policy-scope question, the adapter recovered some valid members but invented `temporary workers` and `visitors with access to protected systems or data`. Set recall alone is insufficient; unsupported-member precision and contradiction/source-support gates are required.

F-048 — Semantic rephrasing can preserve all numeric thresholds while changing surface form enough to fail exact match.  
The paraphrased wind-threshold answer preserved the 23-knot sustained and 31-knot gust roles but failed exact match. Numeric role binding and semantic rule equivalence should dominate exact-string scoring for this task type.

F-049 — Current evaluator knowledge is sample-bounded.  
The v1 reference plan and failure taxonomy were induced from only ten QA records. A static evaluator designed from this sample risks overfitting to observed answer types and missing new failure modes in new documents, domains, or modalities.

F-050 — New documents can introduce previously unseen evaluator requirements.  
A document may contain answer types, constraints, structures, provenance patterns, or failure modes absent from the current evaluator library. The system needs a pre-evaluation capability/failure-discovery phase rather than assuming a fixed evaluator is complete.

F-051 — Evaluator strategy itself should be subject to discovery and validation.  
Before evaluating model quality, the system should generate representative QA/provenance probes, perturb and paraphrase them, observe metric/planner disagreements, discover failure modes, synthesize an evaluation plan, and validate that plan on held-out probes.

F-052 — Runtime self-modifying evaluator code is unnecessarily risky.  
The meta-system should preferably synthesize a declarative evaluation plan from an approved library of gates, metrics, atom extractors, semantic judges, and provenance checks. If a required primitive is missing, it should emit a capability gap/proposed extension rather than silently writing arbitrary executable scoring code.

F-053 — Failure discovery needs adversarial and metamorphic probes, not only natural QA.  
The bootstrap phase should deliberately create controlled variants: paraphrases, omitted qualifiers, swapped numeric roles, extra list members, wrong provenance, partial phrases, contradiction cases, degenerate repetition, and corrupted gold answers. Known transformations provide expected invariants and expected failures.

F-054 — Evaluator-plan invariance should be tested before model evaluation begins.  
Meaning-preserving paraphrases of the same fact should normally yield materially equivalent evaluation plans. If the planner itself is unstable, downstream model scores are not trustworthy.

F-055 — Evaluator adaptation must not leak candidate answers into the planning stage.  
Failure discovery can use synthetic controlled candidates during evaluator development, but the final plan for a real record should be determined from source evidence, question, gold/reference schema, and task semantics before seeing the model candidate being scored.

F-056 — Paraphrase evaluation requires a pool, not a single wording.  
One paraphrase per fact is vulnerable to lucky/unlucky phrasing. A proper protocol should use a frozen paraphrase pool and report distributional performance across variants.

F-057 — Training paraphrases and evaluation paraphrases must be disjoint.  
A practical starting protocol is 15 paraphrases per fact/KU: 10 training-eligible variants and 5 frozen evaluation-only variants. Split membership must be at the paraphrase-pool level and recorded in lineage.

F-058 — Randomized paraphrase sampling must remain reproducible.  
If a smoke test selects one paraphrase from a held-out pool, the selection policy and random seed must be persisted. Publication-grade evaluation should score all held-out variants and report mean, variance, minimum, and failure distribution.

F-059 — Paraphrase robustness is distinct from factual generalization.  
Held-out wording for a trained fact tests linguistic robustness. Held-out facts/KUs test knowledge generalization. Both are necessary and must not be conflated.

F-060 — LLM-judge output-format failure must not become a model-evaluation failure.  
The first Qwen3-32B semantic-adjudication run returned valid judge JSON for R05 and R06 but malformed/incomplete JSON for R08, causing a JSON parser exception. A malformed judge response is evaluator-infrastructure uncertainty and must never be translated into `incorrect` for the candidate.

F-061 — One failed judge record must not destroy earlier successful judgments.  
The first semantic-judge runner buffered results until the full batch completed. When R08 failed parsing, the process exited before writing the successful R05/R06 judgments. Evaluator pipelines require per-record durable checkpoints and resumability.

F-062 — Judge retries and raw attempts are part of reproducibility evidence.  
For every semantic adjudication, preserve raw judge output, finish reason/usage where available, attempt number, parse/validation error, model/runtime identity, and final accepted judgment. Retries must be explicit rather than hidden.

F-063 — Normalized structured-output capability can diverge from local backend implementation.  
The current normalized Inference contract accepts `response_format`, but the current legacy local vLLM adapter does not advertise or enforce structured output and its generation settings do not forward `response_format`. The API currently does not reject this mismatch. Callers can therefore believe JSON-constrained decoding is active when it is not. This is an Inference capability-contract defect and directly affects evaluator reliability.

F-064 — Factual-quality label and acceptance decision are different dimensions.  
R06 recovered two valid members but omitted three required members and invented unsupported members. Calling it `incorrect` because a hard gate failed loses useful information. Prefer: factual_quality=`partially_correct`, acceptance=`reject`, with explicit gate failures.

F-065 — LLM-judge outputs are prompt/rubric-version sensitive.  
The first judge prompt classified R06 as `partially_correct`; the v2 prompt classified the same candidate as `incorrect`. Because the prompts changed, this is not a pure stochastic-repeat result, but it demonstrates that judge conclusions can move when rubric wording changes. Judge prompt/version must therefore be frozen and evaluated like code.

F-066 — Judge confidence semantics are currently unstable/ambiguous.  
R05 received an unequivocal `incorrect` rationale yet v2 returned confidence `0.0`, while the earlier judge run returned confidence `1.0`. Rename the field to `judgment_confidence_0_to_1` and define 1.0 explicitly as high confidence in the assigned evaluator judgment, not candidate quality.

F-067 — Question-quality defects can be more fundamental than gold-quality defects.  
For the Kaveri Gate record, the source states that approval is required and must normally be recorded before crossing 11.5 hours, but it does not identify a unique normal approving authority. The question `what approval is required?` therefore asks for specificity the source does not cleanly provide. DataForge should reject or rewrite such a question rather than merely repairing the gold.

F-068 — Answer correctness and source faithfulness must be scored separately.  
R08 correctly states the 23-knot sustained and 31-knot gust thresholds, but adds `or higher`, which is not stated verbatim in the controlling source. The threshold values answer the question correctly, while the extra clause raises a separate faithfulness/unsupported-inference issue.

F-069 — Unsupported additions need question-scope-aware treatment.  
An extra claim may leave the direct answer correct while making the overall response less faithful. The evaluator should not automatically collapse these into one `partial` or `incorrect` label.

F-070 — Evaluator outcome should be multi-axis rather than one categorical label.  
Recommended minimum dimensions: answer_correctness, completeness, source_faithfulness, unsupported_claims, hard_gate_results, acceptance, reference/question_quality, and judgment_confidence.

F-071 — Policy/compliance evaluation needs an explicit allowed-inference policy.  
Terms such as `threshold` may invite commonsense monotonic reasoning (`31 or higher`), but controlled policy text may require strict source fidelity. Evaluation plans should state whether commonsense entailments are allowed, source-explicit-only, or domain-rule-derived.

F-072 — Semantic adjudication may require source evidence, not only question + gold.  
R08 and R09 show that a judge given only question/gold can miss interpretation priorities, exceptions, or whether the requested specificity is actually supported. For policy/compliance tasks, unresolved semantic cases should receive the relevant evidence span and controlling subsection.

F-073 — The meta-evaluator needs a benchmark for judge quality itself.  
LLM judge correctness should be measured against a small frozen human/source-grounded adjudication set, including cases where hard-gate acceptance differs from factual-quality labeling.

F-074 — The judge can misuse `contradicted` for omission/irrelevance.  
R05 was labeled source-faithfulness=`contradicted` although the candidate mostly fails to answer the rule; R06 was also labeled `contradicted` despite containing some supported members plus omissions and unsupported additions. Faithfulness labels need sharper definitions.

F-075 — Acceptance must not be a free-form LLM opinion when hard gates are declarative.  
R08's configured hard gates were number-role binding, mandatory-atom coverage, and no contradiction. The candidate satisfied the threshold values/roles, yet the judge returned acceptance=`reject` because of an unsupported addition. Acceptance should be computed from explicit gate results and policy after judging, not independently invented by the judge.

F-076 — Direct answer correctness and source-explicit faithfulness can diverge.  
R08 answers the requested threshold correctly while adding `or higher`, which is not explicit in the supplied evidence. The direct answer should not automatically be downgraded because of a separate faithfulness issue.

F-077 — A judge can confuse source support with question-answer fit.  
R09 was accepted as valid because the candidate/gold repeat source-supported threshold language, but the wording `what approval` asks for a type/authority not uniquely supplied by the normal rule. Question quality needs an independent answerability/specificity check.

F-078 — DataForge QA validation must test whether the gold answers the question, not merely whether the gold is entailed by evidence.  
A source-supported sentence can still be a poor gold if it fails to provide the information requested by the generated question.

F-079 — Printed-page provenance and physical PDF/evidence-page provenance are different coordinates.  
The pilot records carry evidence IDs such as `...:page:4` while the embedded source citation may say printed page 2. Evaluator reports should preserve both physical/evidence page identity and document-printed page labels rather than treating them as conflicting.

F-080 — Publication-grade source-grounded evaluation must resolve evidence from provenance, not rely on manually embedded source text.  
The pilot v2 evidence package is useful for discovery, but the eventual Evaluator should dereference evidence IDs/source anchors through Ingest/Storage at runtime, freeze the retrieved evidence package, and checksum it.

F-081 — A 58.3% field-agreement pilot shows the source-grounded judge is not yet a trusted oracle.  
Against the frozen four-record human/source-grounded reference, the v2.1 judge agreed on 14/24 evaluated fields (58.3%). This is a pilot n=4 result, not a general performance estimate, but it is sufficient to reject the current monolithic judge as final authority.

F-082 — QA/reference qualification must precede candidate evaluation and be candidate-blind.  
R09 demonstrates that question answerability and gold adequacy should be decided from source + question + gold before any candidate is shown. Candidate exposure can anchor the reviewer toward accepting a source-supported but question-misaligned gold.

F-083 — The LLM should not independently choose operational acceptance when hard gates are already declarative.  
Use the semantic judge to identify facts/claims/omissions/contradictions. Compute accept/reject/review from the frozen evaluation plan and gate policy in deterministic code.

F-084 — Claim support should be represented explicitly instead of forcing one coarse faithfulness label.  
Separate supported claims, unsupported claims, contradicted claims, and missing required facts. R05/R06 show that `contradicted` can be misused when the true problem is irrelevance, omission, or unsupported content.

F-085 — The adaptive evaluator should be a staged compiler/executor, not one monolithic judge prompt.  
Recommended stages: (1) QA qualification, (2) semantic answer-spec planning, (3) deterministic evaluation, (4) gated semantic claim adjudication, (5) deterministic decision policy, (6) evaluator self-audit.

F-086 — Candidate-blind review alone does not eliminate gold anchoring.  
The first candidate-blind QA qualifier accepted R09 because the gold sentence was source-supported, despite the interrogative form `what approval` requesting information beyond the proposition that approval exists. Gold/reference exposure can still anchor QA qualification.

F-087 — Question-demand extraction should precede gold/reference inspection.  
To prevent the gold from redefining the question, first freeze the interrogative demand from the question alone, then separately test whether source evidence and gold fill the requested information slots.

F-088 — Source entailment is weaker than question-answer alignment.  
R09 demonstrates that `gold is entailed by source` does not imply `gold answers question`. QA qualification needs an explicit requested-slot coverage test.

F-089 — Tautological or premise-restating golds require a dedicated detector.  
A generated gold may repeat the threshold/premise or assert that an approval exists without resolving the interrogative variable introduced by words such as what/who/which/where/when/how many.

F-090 — The meta-evaluator needs staged anti-anchoring boundaries.  
For evaluator design, separate artifacts should freeze: (A) question demand, (B) source answerability, (C) gold coverage, and only later (D) candidate evaluation. Each later stage may consume earlier frozen outputs, but not vice versa.

F-091 — Two-stage anti-anchoring successfully exposed the R09 QA defect.  
When the question demand was extracted before showing the gold/source, R09 produced the mandatory slot `approval_type`; the source could not fill it, the gold did not fill it, and the gold was detected as a restatement. This is the first clear success of the staged QA-qualification architecture.

F-092 — An evaluator must not fabricate a corrected gold when the source cannot answer the requested slot.  
The reviewer correctly rejected R09 but then hallucinated a rewrite answer requiring approval from an immediate supervisor and HR manager, which is not present in the supplied evidence. Rewrite generation must be source-grounded and independently re-qualified before acceptance.

F-093 — Linguistic question validity and corpus/source answerability are separate dimensions.  
R09 was labeled question=`valid` even though the source could not answer the requested `approval_type` slot. The production taxonomy should distinguish syntactic/semantic well-formedness from `answerable_at_requested_specificity`.

F-094 — Proposed rewrites are new derived data and must re-enter the qualification pipeline.  
A rewritten question or gold is not trusted simply because the evaluator proposed it. It must carry provenance, be checked against source evidence, and pass the same question-demand/source/gold qualification stages before use.

F-095 — LLM extraction and deterministic operational policy should be separated.  
Use the LLM to infer interrogative demand, slots, ambiguities, and claim support. Compute accept/rewrite/reject from explicit deterministic policy so an LLM cannot override hard qualification rules or silently accept its own hallucinated rewrite.
