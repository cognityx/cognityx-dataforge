# Cognityx Research Thought Inventory and Deep-Research Agenda
Date: 2026-08-10
Status: pre-literature-review hypothesis inventory

Purpose: freeze the research questions, empirical discoveries, hypotheses, experiment ideas, evaluator ideas, benchmark concerns, and architecture opportunities that have emerged so far. This is intentionally broader than one paper. Deep Research should use it to eliminate weak/redundant directions and identify the minimum defensible experimental program.

## Research method for every thought

For every numbered thought below, Deep Research should investigate three dimensions:

1. Academic literature / prior art.
2. Open-source implementations.
3. Commercial products/platforms.

For each thought, return:
- novelty/differentiation assessment;
- strongest prior art;
- strongest open-source/commercial comparators;
- recommendation: eliminate / reuse / engineering-only / empirical research / publication candidate;
- minimum experiment needed;
- dependencies;
- likely experiment cost;
- likely paper placement.



## PRIMARY EXPLORATION ENVELOPE AND COMPUTE-ESCALATION STRATEGY

Deep Research must study every major theme at **three levels**:

1. the broader state of the art without hardware restriction;
2. the same question under the project's **primary low-cost exploration envelope**; and
3. whether a small number of high-information experiments should be escalated to datacenter-class compute after the local experiments have narrowed the hypothesis space.

The primary exploration envelope is deliberately constrained because it lets us run many cheap, rapid, reproducible experiments:

- **single consumer workstation** as the default research environment;
- **one NVIDIA GeForce RTX 5090 with 32 GB VRAM** as the primary training GPU;
- open-weight language models in the **7B–14B parameter range** as the main locally trainable model class;
- parameter-efficient fine-tuning / QLoRA / related memory-efficient adaptation rather than full pretraining;
- local training, local evaluation and local inference wherever practical;
- CPU RAM/offload may be used when explicitly measured;
- larger local models such as 32B may be used as validators/judges where experimentally justified.

However, **the RTX 5090 / 32 GB envelope is not a hard ceiling or the only research constraint**.

The intended strategy is:

**5090 exploration → narrow hypotheses → escalate only the most informative experiments → compare with higher-resource execution.**

Once a hypothesis is concrete and the expected information gain is high, Deep Research should consider a bounded second-stage experiment using:
- purchased hyperscaler GPU time;
- an available datacenter GPU such as an **H100-class 80–96 GB device**;
- larger models or larger batch/sequence configurations that cannot fit cleanly on the 5090.

For every retained research question, Deep Research should therefore ask:
- Can the decisive experiment be completed entirely on the 5090?
- If not, what exact evidence is missing?
- What is the **minimum higher-resource experiment** needed to resolve it?
- What GPU/model/runtime is required?
- What approximate compute budget should be reserved?
- What local 5090 evidence should be obtained first so expensive runs are not exploratory guesswork?

Every result should report whether it is:
- generally known in unrestricted settings;
- already demonstrated on consumer GPUs;
- already demonstrated on RTX 4090/5090-class hardware;
- already demonstrated on H100/A100/datacenter-class hardware;
- or still insufficiently studied across the consumer→datacenter scaling boundary.

Deep Research must **not assume** that consumer-GPU work is automatically novel. It must explicitly search for prior 24 GB / 32 GB single-GPU PEFT work and identify what, if anything, remains distinctive about the combination of:
document-derived synthetic IFT + provenance + KU discovery + adaptive evaluation + training-efficiency/resource measurements.

It must also investigate whether the **consumer-to-datacenter scaling comparison itself** exposes a useful research question: once the method works locally, what additional benefit comes from upgrading hardware and/or model scale, and which improvements are due to compute capacity versus model capacity?


# A. Research-program structure

T-001 — Three intertwined workstreams, not a simple pipeline. DataForge/data preparation, Training, and Evaluator/Meta-Evaluator are separate experimental workstreams. Evaluator is cross-cutting: it evaluates generated data, trained models, and its own evaluation strategy.

T-002 — Possible Paper A: provenance-grounded document-to-adapter pipeline. Real source documents → Ingest → DataForge → qualification → KU discovery → synthetic IFT → Training → correctness/robustness/provenance/cost.

T-003 — Possible Paper B: adaptive failure-driven evaluator. Failure discovery → answer-requirement discovery → source/gold qualification → adaptive plan synthesis → deterministic tools → gated LLM adjudication → evaluator self-audit → capability/tool-library growth.

T-004 — Possible combined systems paper. Deep Research should determine whether Paper A and Paper B are stronger as one integrated systems paper or two focused papers.

T-005 — Avoid combinatorial explosion. We already identified that naive combinations of data methods × model sizes × training targets × exposures × evaluators × benchmarks can reach hundreds of experiments. We need a minimum causal ablation design.

T-006 — B3 IFT-only benchmark is fallback only. Keep it as a last-resort publishing or isolated Training/Evaluator sanity check, not a primary research/deployment direction.


# B. Source documents, evidence and provenance

T-007 — The production-relevant task is real source document → synthetically generated IFT → adapter. LunaVane is only a controlled synthetic fixture.

T-008 — Provenance must be first-class for every generated record: source asset, document/version, evidence ID, page/section/anchors, and frozen evidence package/checksum.

T-009 — Automatic provenance→evidence resolution is required. Current pilot manually packages source evidence; production Evaluator should dereference provenance through Ingest/Storage.

T-010 — Preserve physical/evidence page and printed page as separate coordinates.

T-011 — Preserve evidence/extractive gold separately from generated/natural-language gold.

T-012 — Explore provenance prediction as a training target: Question→Provenance.

T-013 — Explore Answer+Provenance as a joint target.

T-014 — Explore Question→Provenance→deterministic retrieval→Answer.

T-015 — Answer correctness and provenance correctness are independent dimensions.

T-016 — Candidate joint metric: Grounded Correctness = correct answer AND correct provenance.

T-017 — Investigate whether provenance-oriented training gives greater updateability when authoritative documents change.


# C. DataForge — synthetic data generation and qualification

T-018 — Generation is not acceptance. Structurally valid synthetic QA must not automatically enter Training.

T-019 — For each generated QA, first determine what a satisfactory answer must contain.

T-020 — Then check whether the source evidence actually contains that required information.

T-021 — Then check whether DataForge's gold/reference contains that required information.

T-022 — Question-answer alignment is distinct from source entailment. A gold may be fully source-supported but still fail to answer the question.

T-023 — R09 is the current worked example: “what approval?” requires approval identity/type, while the gold only repeats that approval is required.

T-024 — Generated questions can ask for specificity that the source does not provide.

T-025 — Anti-anchoring staged qualification: question alone → answer requirements; answer requirements+source → source coverage; requirements+source+gold → gold coverage; only later candidate evaluation.

T-026 — Prefer “answer requirements / required facts and constraints” over brittle keyword answer-type labels.

T-027 — Detect tautological/premise-restating golds.

T-028 — Deterministic+LLM qualification cascade: deterministic first, semantic LLM only where needed, deterministic final acceptance policy.

T-029 — Compare deterministic-only validation vs generator self-critique vs stronger cross-model validation vs adjudication.

T-030 — Practical single-5090 validator path: 8B generates → unload → 32B validates → accepted dataset freezes.

T-031 — Core experiment: same source/model/training budget, Raw DataForge IFT vs provenance-qualified DataForge IFT.

T-032 — Test whether qualification reduces exposures/tokens/joules to target quality.

T-033 — Rejected QA can be regenerated from the same evidence, but regenerated question/gold is new derived data and must re-enter qualification.

T-034 — Evaluator-proposed rewrites are never trusted automatically.

T-035 — Candidate-blind qualification may become a formal dataset-certification stage.

T-036 — Measure source-knowledge coverage, not merely number of generated QA records.

T-037 — Detect redundant/near-duplicate training records that encode the same fact.

T-038 — Detect missing facts/KUs not represented by any generated record.

T-039 — Detect correct entities/numbers bound to wrong relations/roles.

T-040 — Preserve material exceptions, conditions, temporal scope and qualifiers.

T-041 — Lists/sets require complete-member and unsupported-member checking.

T-042 — Numeric rules require number-role-unit-relation validation.

T-043 — Accepted/rejected/rewrite reasons should become structured experiment findings.

T-044 — DataForge should own generation/versioning/freeze of evaluation sets, but Evaluator should certify them.


# D. Knowledge Unit discovery

T-045 — Paragraph→QA remains the baseline.

T-046 — KU-first alternative: source→Knowledge Unit discovery→QA per KU.

T-047 — KU is a self-contained fact/rule/concept/procedure/relationship from evidence, not a QA pair.

T-048 — Test whether KU-first improves source-knowledge coverage versus direct paragraph→QA.

T-049 — KU deduplication: multiple sections may restate/refine the same knowledge.

T-050 — Cross-section KU discovery: exceptions/references/definitions may require relations across sections.

T-051 — Generate multiple linguistic question formulations per KU.

T-052 — KU-level train/eval split must separate held-out paraphrases of trained KUs from fully held-out KUs.

T-053 — Every KU needs first-class provenance.

T-054 — Evaluate coverage/correctness per KU, not only per QA, to avoid overcounting duplicates.

T-055 — Compare paragraph→QA vs qualified paragraph→QA vs KU-first QA under same trainer/evaluator.

T-056 — Deep Research must explicitly challenge whether KU is necessary or whether existing document/structure-aware generation already achieves equivalent coverage.


# E. Paraphrase robustness and generalization

T-057 — Current one-paraphrase manual-v1 experiment is only a pilot.

T-058 — Initial proposed pool: 15 paraphrases per KU/fact, 10 training-eligible and 5 evaluation-only.

T-059 — Publication-grade evaluation should score all held-out paraphrases and report mean/variance/worst case/failure distribution.

T-060 — Seeded one-of-five selection is acceptable only as a smoke test.

T-061 — Compare one question per fact vs multiple training paraphrases per fact.

T-062 — Generalization gap = exact-training-question performance minus unseen-paraphrase performance.

T-063 — Paraphrase robustness and factual/KU generalization are distinct.

T-064 — Current pilot: E3 exact recall 9/10, one-paraphrase exact 4/10, with substantial but incomplete semantic transfer.

T-065 — Paraphrase-triggered degeneration is an observed failure: repeated `PEOPLE...` output.

T-066 — Investigate whether this degeneration is overfitting, decoding instability, or small-data/high-exposure interaction.


# F. Model-knowledge probing and contamination control

T-067 — Public benchmark contamination is a serious risk for knowledge-acquisition claims.

T-068 — Public model disclosures may not enumerate complete training corpora, so absence from model card is not proof of non-exposure.

T-069 — Before training on a corpus, probe base model knowledge per KU.

T-070 — Use multiple semantically different probes per KU; one failed answer is not proof of ignorance.

T-071 — Classify KUs as base-KNOWN / PARTIAL / UNKNOWN.

T-072 — Prefer real public documents newer than the frozen model checkpoint/release for strongest acquisition claims.

T-073 — Compare training all KUs vs only PARTIAL+UNKNOWN KUs.

T-074 — Explore knowledge-state-aware data generation: generate more/stronger examples for weak/unknown KUs.

T-075 — Measure knowledge gain per KU: base probe→post-IFT probe.

T-076 — Check interference/forgetting on previously correct/domain-relevant capabilities.

T-077 — Do not count already-known facts as successful document knowledge acquisition.

T-078 — Deep Research should compare this with active learning, curriculum learning, model surprisal, knowledge-gap detection and selective fine-tuning.


# G. Training behavior and efficiency

T-079 — Exposure learning curve already observed on 10-record pilot: ~1 exposure almost no recall; ~5 partial learning; ~10 near-exact training recall.

T-080 — Verify non-linear acquisition on larger data and intermediate exposures.

T-081 — Exact recall is insufficient; include paraphrase robustness, completeness, source faithfulness and provenance.

T-082 — Partial knowledge acquisition must be represented explicitly.

T-083 — Mandatory-qualifier omission needs constraint-level evaluation.

T-084 — Investigate training degeneration/repetition under small-data/high-exposure conditions.

T-085 — Compare Qwen3-8B vs 14B under same data/evaluator and comparable budget.

T-086 — Determine whether 32B is needed as a training target at all or only as validator/judge.

T-087 — LoRA rank exploration only if literature suggests high value; current pilot rank 8.

T-088 — Learning-rate exploration only in a narrow literature-supported range; current pilot 2e-4.

T-089 — Target-module choice may affect knowledge injection; current q/k/v/o projections.

T-090 — Sequence length may need to grow beyond current 512 for real KU examples.

T-091 — Optimize batch/gradient accumulation under 32 GB VRAM.

T-092 — Training loss reduction is not proof of generated knowledge.

T-093 — Primary efficiency measures: accuracy per exposure, per training token, per second, per joule.

T-094 — Peak CPU/RAM/VRAM defines capacity envelope; average power/runtime/energy defines operating efficiency.

T-095 — Current empirical observation: E1→E3 steps increased runtime/energy but peak VRAM stayed near 12.4 GiB.

T-096 — Find minimum-cost recipe meeting target correctness+robustness+provenance rather than highest accuracy at arbitrary cost.

T-097 — Track adapter serialized/resident size as deployment cost.


# H. Training target / knowledge representation

T-098 — Question→Answer baseline.

T-099 — Question→Provenance.

T-100 — Question→Answer+Provenance.

T-101 — Question→Provenance→Retrieve→Answer.

T-102 — Compare factual memorization with knowledge localization.

T-103 — Determine useful provenance granularity: document/page/section/evidence ID/source anchor/KU ID.

T-104 — Consider Provenance Recall@k rather than only one exact pointer.

T-105 — Track four outcome quadrants: correct answer/wrong source; wrong answer/correct source; both correct; both wrong.


# I. Evaluator — observed failure classes

T-106 — Exact-match false negatives for semantically equivalent answers.

T-107 — Smoothed BLEU can award non-zero credit to categorically wrong exact strings.

T-108 — Lexical overlap can hide missing mandatory qualifiers.

T-109 — Gold/reference itself can be defective.

T-110 — Question itself can be defective or unanswerable at requested specificity.

T-111 — Lists require set semantics.

T-112 — Numeric rules require role binding.

T-113 — Planner wording sensitivity: pilot heuristic changed plan on 3/10 paraphrases.

T-114 — Keyword answer-type taxonomy is brittle.

T-115 — Required facts/constraints should be the evaluation IR.

T-116 — Degenerate output needs a hard gate.

T-117 — Aggregate scores hide correct/partial/wrong/degenerate/reference-defect classes.

T-118 — Metric disagreement is a signal for adjudication.

T-119 — Separate answer correctness from completeness.

T-120 — Separate answer correctness from source faithfulness.

T-121 — Represent supported/unsupported/contradicted claims explicitly.

T-122 — Separate quality label from operational acceptance.

T-123 — Define allowed-inference policy for policy/compliance tasks: source-explicit-only vs permissible entailment.

T-124 — Track question quality and reference quality separately from candidate quality.


# J. Adaptive / Meta-Evaluator

T-125 — Evaluator strategy should evolve from observed failure classes, not a universal weighted basket.

T-126 — Evaluator first determines what correctness means for the question/task.

T-127 — Evaluation planning must be candidate-independent.

T-128 — Deterministic-first cascade.

T-129 — Gated semantic/LLM adjudication only for unresolved cases.

T-130 — Separate bad-reference/question-quality queue from candidate scoring.

T-131 — Judge output should include correctness/completeness/support/missing requirements/unsupported claims/rationale/confidence.

T-132 — Judge is not oracle: pilot source-grounded judge matched human/source reference on only 14/24 fields (58.3%, n=4).

T-133 — Judge prompt/rubric changes can change verdicts; version prompts like code.

T-134 — Define judgment confidence precisely; observed confidence semantics were unstable.

T-135 — Inference structured-output mismatch affected evaluator reliability.

T-136 — Robust judge execution requires per-record checkpoints, retries, raw attempts and resumability.

T-137 — Meta-evaluator bootstrap per new corpus/domain: representative probes→correctness structures→stress cases→primitive testing→failure discovery→plan synthesis→plan validation→freeze.

T-138 — Generate metamorphic/adversarial probes: paraphrases, missing qualifiers, swapped numeric roles, wrong units, extra/missing list members, contradictions, wrong provenance, degeneration, defective gold.

T-139 — Evaluator-plan invariance under meaning-preserving paraphrase should be measured.

T-140 — LLM should synthesize declarative Evaluation Plan/IR from approved primitives, not arbitrary runtime Python.

T-141 — Evaluator tool/primitive library should be registered and reusable.

T-142 — Potential primitives: exact/canonical match, set metrics, numeric-role validator, atom coverage, unit normalization, degeneration detector, provenance resolver, semantic similarity, contradiction checker, LLM adjudicator.

T-143 — If needed primitive is absent, emit capability_gap and propose an extension for review.

T-144 — Validated new primitives may be registered for later corpora.

T-145 — Evaluator self-audit against frozen human/source-grounded reference.

T-146 — Append-only failure register linked to experiments, evaluator-plan versions and KPIs.

T-147 — Ingest colleague-observed real failures as external stress probes.

T-148 — Deep Research must compare adaptive evaluator against behavioral testing, LLM-as-judge, rubric generation, metamorphic testing, agentic evaluators, RAG evaluation, attributed QA evaluation, adversarial test generation and self-improving test systems.


# K. Experiment tracking / dynamic Experiment Ledger

T-149 — Experiment tracking must become built-in; manual chat bookkeeping will fail at scale.

T-150 — Reuse an existing tracker backend if possible: MLflow, W&B, Aim, ClearML, DVC/Studio, etc.

T-151 — Possible Cognityx semantic Experiment Ledger: cross-repo lineage + findings + evaluator-plan versions + Storage URIs + KPI definitions on top of an existing tracker.

T-152 — Dynamic KPI registry: newly discovered reusable failure metrics can be added over time.

T-153 — Never silently redefine an existing KPI; metric semantics must be versioned/immutable.

T-154 — Finding lifecycle: observation→F-ID→generalizable/measurable?→KPI definition→evaluator implementation→future runs emit it.

T-155 — Not every failure is a KPI; runtime defects may remain engineering findings.

T-156 — Tracker must tolerate evolving/wide metric schema without breaking historical comparison.

T-157 — Track whether a newly introduced KPI can be backfilled over frozen historical outputs.

T-158 — Dataset lineage: dataset/variant/version/hash/Storage URI/accepted-rejected counts.

T-159 — Evaluator lineage: plan version, primitive versions, prompt/judge identity, failure-register version.

T-160 — Training lineage: base model/revision, adapter, hyperparameters, seed, resources, runtime, energy.

T-161 — Parent-child lineage: DataForge run→Training runs→Evaluator runs.

T-162 — Structured “interesting findings” beyond scalar metrics.

T-163 — Preserve human-reference decisions with provenance/versioning.

T-164 — Deep Research must decide whether Experiment Ledger is novel/useful or just standard MLflow/W&B metadata; do not build if existing systems suffice.


# L. Benchmark strategy

T-165 — B0 controlled diagnostic benchmark: LunaVane-style fixture with deliberately known truth/failure ingredients.

T-166 — B0 purpose is mechanism discovery, regression, evaluator development and architecture validation—not external validity.

T-167 — Expand B0 to multiple fixtures/domains only after core architecture stabilizes.

T-168 — B1 published-method like-for-like benchmark: identify raw context/document→synthetic IFT methods with public assets and reproducible training/evaluation.

T-169 — Strong candidate direction: same source contexts, published baseline synthetic IFT vs Cognityx raw/qualified/KU-first IFT.

T-170 — To isolate DataForge, hold source/trainer/evaluator fixed and vary only data-generation method.

T-171 — To isolate Training/Evaluator, freeze one IFT dataset and vary only those components.

T-172 — B2 contamination-controlled real-document benchmark: real public enterprise-like documents new relative to model checkpoint/release.

T-173 — Start one domain at a time to match deployment reality.

T-174 — Candidate initial domains: HR/policy, IT/security, finance policy, audit/compliance, EHS; manufacturing SOP later.

T-175 — After one domain, test transfer to another rather than mixing all from the start.

T-176 — Strongest final test questions should be independent of the training-data generator.

T-177 — If human question creation is expensive, use separate generation + human/source qualification.

T-178 — Public benchmark contamination preflight is mandatory for acquisition claims.

T-179 — Deep Research should identify all like-for-like raw-document/context→synthetic-IFT systems, not only the one already discussed.

T-180 — Prefer enterprise-policy/SOP-like tasks rather than generic Alpaca-style chat.

T-181 — Contract/privacy/policy-like public corpora are worth screening.

T-182 — Multimodal benchmark only after text pipeline stabilizes.

T-183 — Multi-document benchmark only after single-document/domain pipeline stabilizes.

T-184 — B3 IFT-only remains fallback only; no substantial work now.


# M. Fair comparison and controls

T-185 — Same frozen base-model revision for data-method comparisons.

T-186 — Explicit fairness definition for training budget: same steps, same tokens, same examples, or same energy.

T-187 — Same frozen evaluator when comparing data/training methods.

T-188 — Same source corpus for DataForge-vs-baseline comparisons.

T-189 — Same held-out evaluation set, frozen before training and independent of generator.

T-190 — Persist generation/splitting/paraphrase/training seeds.

T-191 — No held-out paraphrase/KU/test question may enter training.

T-192 — Stronger validator/judge contamination must be distinguished from source-model contamination.

T-193 — Keep a small high-quality human/source-grounded reference subset for evaluator benchmarking.


# N. Consumer-GPU / constrained-compute angle

T-194 — RTX 5090 fine-tuning is engineering context, not novelty by itself.

T-195 — Measure minimum CPU/RAM/VRAM capacity envelope.

T-196 — Track average/peak power and total energy.

T-197 — Test data quality vs energy: does qualified data reach target quality with fewer joules?

T-198 — Test model-size efficiency: 8B may need more exposures, 14B may cost more per step but fewer steps.

T-199 — Track judge cost: deterministic resolution rate vs 32B/remote judge calls.

T-200 — Evaluator cost-quality frontier: accuracy vs latency/tokens/energy/cost.


# O. Inference/runtime issues that affect research

T-201 — Manager-owned single normal lifecycle; avoid competing manager vs direct-worker paths.

T-202 — Model load/unload must be background jobs with durable events, not generic HTTP timeout.

T-203 — Resource-aware residency: certified profile + live VRAM decides parallel residency vs eviction.

T-204 — Separate worker/server profile from model runtime/certification profile.

T-205 — First-class Training→Inference adapter handoff is required.

T-206 — Structured-output capability must be correctly enforced/rejected.

T-207 — Standard manager-owned logs/events are needed for reproducible experiments.


# P. Candidate hypotheses — to test, not assume

T-208 — Provenance-grounded qualification improves downstream training quality versus raw synthetic QA.

T-209 — Qualified/KU-first data reaches target correctness/robustness with fewer exposures/tokens/joules.

T-210 — KU-first improves source-knowledge coverage and reduces redundant/incomplete QA versus paragraph→QA.

T-211 — Multiple training paraphrases per KU reduce exact→paraphrase generalization gap.

T-212 — Training only weak/unknown KUs reduces cost without losing quality.

T-213 — Provenance or Answer+Provenance training improves grounded retrieval/correctness and updateability.

T-214 — Failure-driven adaptive evaluation agrees with source-grounded human judgment better than static exact/lexical metrics or monolithic LLM judge.

T-215 — Deterministic-first cascade maintains evaluator quality while reducing judge calls/cost.

T-216 — Meta-evaluator bootstrap can discover evaluator gaps on new corpora and improve plan accuracy/invariance without manual recoding for every document.

T-217 — Failure-linked dynamic KPI registration improves experiment interpretability without breaking historical comparability.

T-218 — End-to-end hypothesis: a provenance-grounded system can turn real enterprise-like documents into qualified synthetic IFT, train a PEFT adapter under constrained compute, and evaluate it with auditable source-grounded evidence.


# Q. Deep Research must explicitly challenge/eliminate

T-219 — Is KU actually necessary, or do existing structure-aware/document-level generation methods already solve coverage?

T-220 — Is adaptive evaluator genuinely new versus existing behavioral testing, generated rubrics, agentic evaluators and metamorphic testing?

T-221 — Is evaluator capability/tool growth already standard in agent/tool/test-generation systems?

T-222 — Is Experiment Ledger research-worthy, or just MLflow/W&B plus metadata?

T-223 — Is knowledge probing + selective training already sufficiently solved by active learning/curriculum/surprisal methods?

T-224 — Does provenance-target training provide measurable benefits beyond ordinary retrieval?

T-225 — Which data-generation baselines are truly like-for-like and reproducible?

T-226 — What is the smallest statistically defensible experiment matrix?

T-227 — Which ideas should be removed entirely to avoid a sprawling paper?

T-228 — Which ideas are valuable product engineering but weak research?

T-229 — Which ideas justify a separate Evaluator paper?

T-230 — Which ideas justify the document→IFT→training paper?

T-231 — Which abstractions Cognityx should not build because existing open-source/commercial systems already provide them?



# S. Fixed consumer-GPU research envelope and barrier-to-entry hypothesis

T-232 — Research-barrier hypothesis. Investigate the intuition that LLM application/agent research has a lower practical barrier to entry than empirical model-training research because agent/application work can often be conducted through prompting, APIs, orchestration and modest local compute, while repeated controlled fine-tuning requires dedicated accelerator capacity, longer experiment cycles and hardware/software expertise.

T-233 — Do not equate lower barrier with guaranteed novelty. Agentic-LLM surveys already describe a rapidly growing literature, but PEFT/QLoRA has also substantially reduced the training barrier. Deep Research must quantify or otherwise rigorously characterize both landscapes rather than assuming training papers are scarce.

T-234 — Consumer-GPU fine-tuning is already established prior art. Explicitly review QLoRA, ModuLoRA, LoHan and later consumer-GPU training systems. Determine which claims about “fine-tuning on one consumer GPU” are already routine and should not be made as contributions.

T-235 — RTX 5090-specific empirical literature is a separate search dimension. Search for papers and reproducible studies using the RTX 5090 specifically for LLM/LMM fine-tuning, inference, long-context work and energy/resource profiling.

T-236 — 7B–14B single-5090 envelope is the primary trainable-model regime. For every proposed DataForge, KU, Training and Evaluator experiment, Deep Research must ask whether it is feasible and previously studied within a single 32 GB RTX 5090 using open-weight 7B–14B models.

T-237 — Broader-vs-constrained comparison is mandatory. For every major idea, report:
(a) best known result with unrestricted/datacenter compute;
(b) best known result on single consumer GPU;
(c) best known result on 24–32 GB GPUs;
(d) best known result specifically on RTX 5090 if available;
(e) the residual research gap under our envelope.

T-238 — Hardware-specific novelty is insufficient. The RTX 5090 itself must not be positioned as the paper contribution. The potential contribution is an empirically validated method/system that remains effective under the 32 GB constraint.

T-239 — Hardware constraint can strengthen the empirical question. Candidate framing:
“What combination of data qualification, KU discovery, selective training, PEFT configuration and adaptive evaluation achieves a specified grounded-knowledge target with minimum time, energy and memory on a single consumer GPU?”

T-240 — Cost-of-entry should be treated as motivation, not proof of novelty. Investigate published discussions of compute democratization, resource inequality and consumer-GPU adaptation, but do not infer publication scarcity simply from the monetary cost of the workstation.

T-241 — PEFT has lowered the barrier substantially. Deep Research should test whether 7B–14B experiments that appear expensive in our current implementation can already run on cheaper 16–24 GB GPUs using optimized stacks such as Unsloth/FlashAttention/paged optimizers. If so, distinguish “single-5090” from the more general “single-consumer-GPU” contribution.

T-242 — 5090 software-stack maturity is part of the experiment context. Blackwell support, CUDA/PyTorch/vLLM/bitsandbytes/Unsloth compatibility, quantization kernels and structured-output/runtime limitations may affect results. Record versions and distinguish algorithmic findings from transient software-stack limitations.

T-243 — Resource telemetry should be publication-grade. Record peak VRAM, peak/average GPU utilization, host RAM, process CPU/host CPU, runtime, average/peak power, total energy and adapter size with clearly defined measurement scope.

T-244 — Consumer-hardware Pareto frontier. Compare model/data/training strategies on quality versus:
- VRAM;
- wall-clock time;
- joules;
- training tokens/examples;
- adapter size;
- evaluator/judge cost.

T-245 — Model-size frontier within the envelope. At minimum compare an 8B-class model with a 14B-class model under matched source/data/evaluation conditions. Determine whether larger capacity reduces exposures/generalization gap enough to offset higher per-step cost.

T-246 — Model-knowledge probing under the same hardware envelope. Base-known/partial/unknown classification should be measured on the same frozen local base models that will be fine-tuned, not inferred from larger remote models.

T-247 — Evaluator cost under constraint. Measure how much evaluation can be settled using deterministic/local 8B–14B methods before escalating to a 32B local judge or remote provider. This may itself produce a cost-quality frontier.

T-248 — Training-system efficiency is a confounder. If comparing our trainer against published results, distinguish:
data/algorithm improvements from implementation/kernel improvements. Consider at least one optimized consumer-GPU training stack as a systems baseline.

T-249 — Search for exact-intersection prior art. Deep Research should specifically search for studies combining:
real documents → synthetic IFT → PEFT on 7B–14B open models → single 24–32 GB consumer GPU → held-out knowledge/paraphrase/provenance evaluation → power/energy/resource reporting.

T-250 — Search-scarcity must be verified systematically. A quick search currently suggests that RTX-5090 LLM fine-tuning papers exist but are much less numerous than general PEFT or agent papers; this is only a hypothesis and must be validated with a reproducible literature-search method.

## Preliminary anchors from a quick search — not novelty claims

- NVIDIA officially specifies the RTX 5090 with **32 GB GDDR7** and Blackwell architecture.
- QLoRA (Dettmers et al., 2023) demonstrated 65B fine-tuning on a single 48 GB GPU.
- ModuLoRA (Yin et al., 2023) reports 65B low-bit fine-tuning on a single 24 GB GPU.
- LoHan (Liao et al., 2024) targets extreme single-consumer-GPU fine-tuning with CPU offload and reports experiments on RTX 4090.
- A 2025 consumer-GPU profiling paper studies LoRA/QLoRA on an 8 GB RTX 4060, confirming that resource-constrained fine-tuning itself is an active research topic.
- NVIDIA/Unsloth published Blackwell desktop training benchmarks using a **GeForce RTX 5090 32 GB**, including QLoRA settings.
- A 2026 paper, *Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis* (arXiv:2606.05713), reports QLoRA training of a 7B Qwen2.5-Omni pipeline on a single RTX 5090 with roughly 10–21 GB peak memory.
- Therefore the research claim must **not** be “LLM fine-tuning on an RTX 5090 is novel.” Deep Research must determine whether our narrower combination and empirical questions remain underexplored.



## T-251 — Compute escalation is hypothesis-driven, not default
The 5090 is the discovery platform, not the permanent ceiling. Expensive GPU experiments should occur only after local experiments have identified a specific unresolved question and a measurable success/falsification criterion.

## T-252 — Consumer→datacenter replication
For a small set of decisive experiments, reproduce the same frozen source/data/evaluator/training logic on a datacenter-class GPU and determine whether conclusions survive hardware scaling.

## T-253 — Hardware scaling versus model scaling
Separate two interventions:
- same model/configuration on faster/larger GPU;
- larger model enabled by larger VRAM.
Do not attribute model-capacity gains to hardware alone.

## T-254 — Same-model hardware comparison
Where feasible, run the same 8B/14B model and same training recipe on RTX 5090 and H100-class hardware. Compare wall-clock time, throughput, energy, utilization, batch size, sequence length and total cost to target quality.

## T-255 — Larger-model uplift experiment
After the best 5090-compatible method is established, test whether moving to a model that only fits comfortably on H100-class memory materially improves:
knowledge acquisition, paraphrase robustness, provenance accuracy, evaluator reliability or number of required exposures.

## T-256 — Batch/sequence scaling benefit
Investigate whether higher VRAM improves results merely by increasing batch size/sequence length/throughput, or whether quality remains statistically unchanged and only runtime improves.

## T-257 — Time-to-quality versus cost-to-quality
Compare consumer and datacenter execution using both:
- wall-clock time to reach target quality;
- total monetary/energy cost to reach target quality.
A faster H100 run is not automatically more efficient economically.

## T-258 — Energy-to-quality scaling
Where measurement is feasible, compare joules to reach the same target quality, not only peak power or tokens/second.

## T-259 — Local-to-cloud reproducibility
Freeze model revision, dataset hash, evaluator plan, seed and software stack so selected 5090 experiments can be reproduced on cloud/H100 without changing the scientific treatment.

## T-260 — Minimum escalation budget
Deep Research should estimate a bounded budget for the smallest H100/cloud experiment set needed to validate scaling claims. Do not propose an open-ended cloud sweep.

## T-261 — Prior art on consumer-vs-datacenter fine-tuning
Search specifically for papers comparing PEFT/fine-tuning efficiency and quality across consumer GPUs and A100/H100-class accelerators while holding model/data/training method constant.

## T-262 — Prior art on hardware-enabled model uplift
Search for studies separating the gain from faster hardware from the gain of using a larger model made possible by the hardware.

## T-263 — Research interpretation
Potential result categories:
- 5090 method reaches same quality, H100 only reduces time;
- H100 enables larger batch/sequence but not better quality;
- larger H100-only model improves quality/generalization;
- larger model improves quality but worsens cost-to-quality;
- constrained-data/evaluator improvements dominate hardware scaling.
Each outcome is scientifically useful if designed prospectively.

## T-264 — Enterprise deployment implication
If 7B–14B adapters trained locally achieve nearly the same domain quality as larger H100-only models, that supports economical distributed/departmental adaptation. If not, identify the threshold at which centralized high-end compute becomes justified.

# R. Required Deep-Research deliverable

1. Prior-art matrix for all themes above.
2. Open-source comparator matrix.
3. Commercial comparator matrix.
4. Elimination table: already solved / reuse / engineering-only / empirical research / publication candidate.
5. Recommended Paper A scope.
6. Recommended Paper B scope.
7. Whether one combined paper is stronger than two.
8. Recommended B0/B1/B2 benchmark protocol; B3 fallback only.
9. Minimum experiment matrix with estimated number of training/evaluator runs.
10. Statistical methodology and sample-size guidance.
11. Contamination-control protocol.
12. Evaluator validation protocol against human/source-grounded reference.
13. Experiment-tracking recommendation: existing platform vs Cognityx semantic layer.
14. Novelty-risk register.
15. Likely reviewer objections.
16. Candidate publication venues.
17. “Abstractions Cognityx Should Not Build.”
18. Recommended implementation order after the research review.
19. For every major theme, a **two-column evidence view**: unrestricted state of the art vs single-consumer-GPU / 32 GB state of the art.
20. Dedicated literature search for **7B–14B open-weight PEFT on 24–32 GB consumer GPUs**, with RTX 5090-specific evidence called out separately.
21. Assessment of the barrier-to-entry hypothesis: agent/application research density vs model-training empirical research, with caveats and a reproducible search method.
22. A consumer-GPU feasibility matrix for every proposed experiment: expected VRAM, runtime, energy, software-stack requirements and whether 8B/14B fit cleanly on one RTX 5090.
23. A novelty assessment that explicitly separates **hardware novelty**, **systems novelty**, **data/evaluator novelty**, and **empirical novelty**.
24. Recommended optimized consumer-GPU baseline(s) so that Cognityx algorithmic claims are not confounded by an inefficient training implementation.
25. A **compute-escalation plan** identifying which hypotheses, if any, justify H100/cloud experiments after 5090 exploration.
26. A **consumer→datacenter comparison protocol** separating same-model hardware scaling from larger-model scaling.
27. Estimated H100/cloud GPU-hours and monetary budget for the minimum decisive experiment set.
28. Prior-art review of RTX 4090/5090 versus A100/H100 fine-tuning comparisons and hardware-enabled model-scaling studies.
29. Recommended quality/time/joule/cost metrics for deciding whether high-end GPU escalation is genuinely beneficial.
