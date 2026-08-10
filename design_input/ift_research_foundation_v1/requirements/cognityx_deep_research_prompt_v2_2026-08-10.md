# Deep Research: Narrow Cognityx into a Defensible Research and Experiment Program

## Attachments

Use these two documents as the primary research package:

1. `cognityx_research_thought_inventory_deep_research_handoff_2026-08-10_v3.md`
2. the latest `cognityx_adaptive_evaluator_failure_register` (currently v0.9)

Read both completely before forming recommendations.

The Thought Inventory is a deliberately broad hypothesis space. The Failure Register contains empirical failures discovered during actual DataForge → Training → Evaluator experiments. Treat neither as established truth.

Your purpose is to reduce this space into a small, defensible research and implementation program.

---

# Primary objective

We now have too many possible experimental variables and their Cartesian product can easily exceed hundreds of experiments.

Radically narrow the space.

Optimize for the intersection of:

- scientific novelty;
- empirical defensibility;
- enterprise usefulness;
- feasibility;
- publication potential;
- information gain per experiment;
- ability to falsify hypotheses rather than merely demonstrate a system.

Do not recommend exhaustive sweeps.

Identify what should be eliminated, what should reuse existing work, what is engineering-only, and what genuinely deserves experiments.

---

# Compute strategy: primary exploration envelope, not a hard ceiling

Treat the local workstation as the **primary low-cost exploration environment**, not the maximum compute permitted by the research:

- one NVIDIA RTX 5090 with 32 GB VRAM;
- primarily open-weight 7B–14B trainable models;
- PEFT/LoRA/QLoRA and related memory-efficient adaptation;
- local training/evaluation/inference wherever practical;
- larger local models such as 32B may be used as validators/judges where justified;
- runtime, VRAM, host RAM, power and energy are part of the evidence.

The intended strategy is:

**5090 exploration → narrow hypotheses → selectively escalate only high-information experiments to datacenter compute.**

Higher-resource experiments may use bounded purchased hyperscaler GPU time or an available H100-class 80–96 GB GPU when the local experiments have produced a precise unresolved question.

For every retained hypothesis determine:

1. what can be established conclusively on the RTX 5090;
2. what remains uncertain specifically because of model/hardware limits;
3. whether a higher-resource run would actually resolve that uncertainty;
4. the minimum H100/cloud experiment needed;
5. expected GPU-hours and approximate budget;
6. what must be frozen locally first so the expensive run is confirmatory rather than exploratory.

Explicitly separate:

- **hardware scaling:** same model/data/training/evaluator on RTX 5090 versus H100-class hardware;
- **model scaling:** a larger model made feasible by larger VRAM.

Do not attribute model-capacity gains to hardware.

Search for prior work at all relevant levels:

- unrestricted/datacenter compute;
- single consumer GPU;
- 24–32 GB consumer GPU;
- RTX 4090/5090 specifically;
- A100/H100-class systems;
- consumer→datacenter comparisons holding model/data/method constant.

The RTX 5090 itself is not a novelty claim. Determine whether the constrained-compute setting exposes an under-studied data/training/evaluation efficiency problem.

---

# Three interacting research workstreams

Analyze Cognityx as three interacting workstreams.

## A. DataForge / document-to-training-data

Real document → evidence/provenance → synthetic instruction data.

Investigate:

- raw paragraph→QA;
- provenance-grounded QA qualification;
- answer requirements → source coverage → gold coverage;
- defective-question and defective-gold detection;
- Knowledge Unit discovery before QA generation;
- KU coverage/deduplication/cross-section relations;
- multiple paraphrases per KU;
- train/evaluation paraphrase separation;
- model-knowledge probing;
- training only weak/unknown KUs;
- Answer vs Provenance vs Answer+Provenance;
- provenance→retrieve→answer;
- synthetic-data qualification as part of DataForge rather than post-hoc cleaning.

## B. Training

Existing pilot evidence includes a non-linear learning transition on a 10-record dataset:
approximately one exposure produced almost no recall, five produced partial learning, and ten produced near-exact training-question recall, while held-out paraphrase performance remained substantially lower.

Investigate:

- knowledge acquisition versus memorization;
- exposure/data efficiency;
- qualified versus raw data;
- paragraph-first versus KU-first;
- one wording versus multiple training paraphrases;
- all-KU versus weak/unknown-KU training;
- 8B versus 14B;
- Answer versus Provenance targets;
- minimum time/tokens/joules to a defined grounded-knowledge target;
- whether a larger H100-only model changes quality enough to justify its cost.

## C. Adaptive Evaluator / Meta-Evaluator

Do not reduce this to generic LLM-as-a-judge.

The attached Failure Register documents observed failures involving exact/lexical metrics, incomplete answers, defective golds/questions, numeric/list semantics, planner drift, degeneration, source faithfulness, unsupported additions, judge instability, structured-output failures, evaluator-vs-human disagreement, and evaluator hallucination.

Critically study the hypothesis:

question/source/gold
→ determine answer requirements
→ source/gold qualification
→ synthesize a declarative evaluation plan
→ deterministic evaluator primitives where possible
→ gated semantic/LLM adjudication only where necessary
→ deterministic final acceptance policy
→ detect new failure classes
→ identify evaluator capability gaps
→ validate/register new evaluator primitives
→ self-audit the evaluator against human/source-grounded references.

Investigate overlap with behavioral testing, metamorphic testing, dynamic rubric generation, model-based evaluation, LLM-as-judge, agentic evaluation, adversarial test generation, RAG/attribution evaluation, automatic test generation, evaluator ensembles and commercial evaluation platforms.

---

# Benchmark strategy

Critically evaluate and refine:

## B0 — Controlled diagnostic fixtures

LunaVane-style deliberately constructed documents with known ground truth and intentionally represented failure structures.

Use for:
- mechanism discovery;
- controlled failure injection;
- evaluator regression;
- architecture validation.

Determine how many independent fixtures/domains are needed before claims become credible.

## B1 — Published-method like-for-like benchmark

Identify published methods that start from raw/unannotated source documents or contexts and create synthetic IFT.

The ideal comparison is:

same source documents
→ published baseline synthetic IFT
vs
→ Cognityx raw IFT
vs
→ Cognityx provenance-qualified IFT
vs
→ Cognityx KU-first IFT

while holding base model, trainer, compute definition and final evaluator constant.

Bonito is one candidate, not an assumed winner. Find competitors/successors and determine which provide reproducible source data, generated IFT and evaluation procedures.

## B2 — Contamination-controlled real-document benchmark

The production problem is:

real enterprise-like document
→ synthetic IFT
→ trained domain adapter.

Old public benchmarks may already exist in model pretraining.

Research contamination-control methodology and assess this proposed protocol:

- prefer real public documents released after the frozen model checkpoint/release where possible;
- discover KUs;
- probe the unconditioned base model using multiple independent questions per KU;
- classify base-known / partial / unknown;
- train;
- use independent held-out evaluation questions;
- measure knowledge gain separately by base-knowledge category.

Recommend one first enterprise-like domain and specific public sources.

Candidate domains:
HR/policy, finance/approval policy, audit/compliance, IT/security, EHS; manufacturing SOP later.

## B3 — Published IFT-only benchmark

Retain only as a fallback/isolation experiment. Do not make this the main direction unless the literature gives a compelling reason.

---

# Experiment tracking / Experiment Ledger

Research whether existing systems such as MLflow, Weights & Biases, ClearML, Aim, DVC/Studio or others already solve enough of the tracking requirement.

The desired semantic behavior includes:

experiment/finding
→ stable finding ID
→ determine whether it is a reusable measurable concept
→ versioned KPI definition
→ evaluator implementation
→ future runs emit KPI
→ backfill older frozen outputs where valid
→ historical KPI semantics remain immutable.

Track cross-repo lineage:
DataForge run → dataset/version → Training run → adapter → Evaluator run → evaluator-plan version → findings/KPIs → source/provenance artifacts.

Determine whether Cognityx needs:
- no new layer;
- only conventions/plugins over an existing tracker;
- a thin semantic Experiment Ledger;
- or truly custom infrastructure.

Do not build a tracker merely because the terminology is different.

---

# Mandatory literature-review and evidence standard

This is a **literature-driven research review**, not an opinion essay.

Every substantive conclusion, novelty assessment, elimination recommendation, benchmark recommendation, or statement that an idea is already solved must be grounded in cited evidence.

## Literature coverage

Perform an extensive literature review covering, where relevant:

- ACL / EMNLP / NAACL / COLING / Findings;
- NeurIPS / ICML / ICLR;
- AAAI / IJCAI;
- ACM / IEEE venues;
- arXiv for very recent 2025–2026 work where peer-reviewed publication is not yet available;
- primary project papers and official documentation for open-source systems;
- official product documentation / technical reports for commercial systems.

Prioritize **primary sources** over blog posts and secondary summaries.

Use seminal older work where necessary, but emphasize the current 2023–2026 state of the art.

## Citation requirements

For **every major argument**:

1. provide inline citations at the point where the claim is made;
2. cite the strongest primary literature supporting or contradicting it;
3. where literature disagrees, present both sides;
4. distinguish peer-reviewed evidence from preprints;
5. distinguish academic evidence from vendor/commercial claims;
6. never treat a vendor benchmark as independent scientific validation;
7. do not claim absence of prior art merely because a quick search found nothing;
8. when no strong literature exists, explicitly say **“evidence gap / no strong prior art found”** and describe the search performed.

Avoid unsupported statements such as:
- “this appears novel”;
- “few people have studied this”;
- “this is standard”;
- “commercial systems already solve this”;
unless supported by references.

## Reference record

For every paper materially used, provide enough bibliographic detail to locate it:
- title;
- authors;
- year;
- venue/status;
- DOI/arXiv/ACL/official URL where available.

At the end, provide a deduplicated bibliography grouped by research theme.

## Evidence matrix

For each retained Cognityx research idea, create a table with:

- Cognityx thought/hypothesis;
- closest prior work;
- what that prior work actually demonstrates;
- experimental scale/hardware;
- dataset/domain;
- evaluator used;
- what remains different in Cognityx;
- novelty risk: high / medium / low;
- recommendation: eliminate / reuse / engineering / experiment / publication candidate;
- citations.

The recommendation must follow from the cited literature, not intuition.

## Search-saturation requirement

For topics where novelty depends on scarcity of prior art—especially:
- adaptive/meta evaluation;
- KU-first synthetic IFT;
- provenance-target training;
- knowledge-selective training;
- RTX 5090 / 24–32 GB PEFT;
- consumer→H100 comparison;
- dynamic KPI/evaluator capability growth—

perform multiple search formulations and explicitly document the strongest overlapping papers found.

The goal is not to prove that Cognityx is novel. The goal is to **try hard to disprove novelty first**.

---

# Academic + open-source + commercial comparison

For each important proposal investigate three dimensions:

1. **Academic literature**
2. **Open-source implementations**
3. **Commercial products/platforms**

For open-source/commercial systems determine not only whether a feature exists, but whether it actually provides the semantics we propose.

For example:
“supports custom metrics” is not automatically equivalent to “failure-driven dynamic KPI discovery.”

Conversely, do not invent a Cognityx abstraction if existing capabilities already compose to solve the problem.

---

# Be adversarial to Cognityx

For every major concept ask:

- Is this already solved?
- Is the abstraction unnecessary?
- Is novelty merely terminology?
- Could a mature library already do this?
- Is the observed failure peculiar to a 10-record fixture?
- Would a reviewer call LunaVane overfitting?
- Would a reviewer call the 5090 angle engineering rather than research?
- Is KU just IE/semantic chunking under a new name?
- Is adaptive evaluation just rule routing + LLM-as-judge?
- Is model-knowledge probing just active learning/data selection?
- Is Experiment Ledger simply MLflow with metadata?
- Is provenance-target training already attributable/source-aware QA?

For each candidate paper, write a **reviewer rejection argument** and then state what evidence would be required to overcome it.

---

# Reduce the experiment space

Use literature to identify:

- variables already settled enough to freeze;
- accepted defaults we should reuse;
- experiments with little additional information value;
- variables that truly require manipulation;
- interactions that genuinely matter.

Produce a **minimum experiment program**, preferably around 10–20 high-information experimental decisions or tightly grouped experiment families rather than hundreds of Cartesian combinations.

For each retained experiment specify:

- hypothesis;
- independent variable;
- frozen variables;
- baseline;
- source/dataset;
- trainable model;
- evaluator;
- primary metrics;
- secondary diagnostics;
- expected RTX 5090 burden;
- whether escalation is required;
- if escalation is required, minimum H100/cloud GPU-hours and budget;
- success/falsification criterion;
- conclusion enabled;
- Paper A / Paper B / both / engineering-only.

---

# Decide what to build first

Given the current state, determine the next implementation based on **maximum information gain per unit effort**, not architectural neatness.

Explicitly rank at least:

1. automatic provenance→evidence resolution and full DataForge qualification;
2. Knowledge Unit discovery;
3. Adaptive Evaluator / Meta-Evaluator;
4. Experiment Ledger;
5. external benchmark reproduction;
6. additional training;
7. any better alternative you identify.

Return the next 3–5 concrete implementation/experimental steps in order.

---

# Paper decomposition

Assess separately:

## Paper A — provenance-grounded document-to-synthetic-IFT and constrained-compute knowledge adaptation

Potential path:
source document
→ qualification / KU discovery / knowledge probing
→ synthetic IFT
→ PEFT
→ independently held-out factual/paraphrase/provenance evaluation
→ quality/time/joule/cost trade-offs
→ selected consumer→datacenter replication.

## Paper B — failure-driven adaptive evaluation for source-grounded document learning

Potential path:
failure discovery
→ answer-requirement extraction
→ source/gold qualification
→ declarative evaluation plan
→ registered deterministic tools
→ gated semantic adjudication
→ capability-gap discovery
→ evaluator self-audit
→ dynamic findings/KPIs.

Determine:
- independent novelty of each;
- whether one combined systems paper is stronger;
- which is more promising;
- minimum evidence required;
- best baselines;
- likely reviewer objections;
- appropriate venues.

---

# Required final deliverable

End the report with a decisive recommendation containing:

A. **What to abandon now**  
B. **What to reuse**  
C. **What is engineering-only**  
D. **What deserves experimentation**  
E. **Strongest publication candidates, ranked**  
F. **Minimum experiment matrix**  
G. **The first thing to build next, and why**  
H. **Experiment Ledger recommendation**  
I. **Exact B0/B1/B2 benchmark recommendation**  
J. **Contamination-control protocol**  
K. **Novelty-risk register with references**  
L. **Paper A vs Paper B vs combined-paper recommendation**  
M. **Consumer→datacenter escalation plan and bounded budget**  
N. **Literature evidence matrix**  
O. **Full deduplicated bibliography**

Above all, use the research to **reduce uncertainty, eliminate unnecessary experiments and challenge our novelty assumptions**. Do not respond by expanding the idea list.
