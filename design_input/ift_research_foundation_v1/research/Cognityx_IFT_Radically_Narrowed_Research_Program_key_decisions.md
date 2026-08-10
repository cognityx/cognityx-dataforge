# Cognityx IFT Data Preparation, Training, and Adaptive Evaluation — Radically Narrowed Research Program

Status: recovered key-decision extract for the implementation design pack.  
Source: Deep Research output dated 2026-08-10.  
Purpose: retain the research-pruning decisions that govern this Codex implementation without expanding the implementation scope.

## Executive research decision

The broad Cognityx hypothesis inventory creates a combinatorial trap. A conservative Cartesian product of only the variables already proposed reaches 5,184 experiment cells before seeds or hyperparameter sweeps. The research program therefore narrows the work to two independent hypotheses and treats almost everything else as deferred or engineering-only.

### Paper A — primary immediate hypothesis

> Does source-grounded qualification of document-derived synthetic IFT improve held-out grounded knowledge acquisition and reduce training cost-to-quality compared with unqualified synthetic IFT and a strong published synthetic-data baseline?

The first causal experiment is deliberately:

```text
same source corpus
same initial QA generator
same frozen base model/revision
same training implementation
same training-token budget
same frozen evaluator/evaluation questions

raw paragraph-derived synthetic QA
versus
same candidates + source/question/gold/provenance qualification
```

The contribution is not document→QA, PEFT, QLoRA, the RTX 5090, or the term Knowledge Unit. The potential contribution is the causal chain:

`source/question/gold qualification → fewer defective records → better held-out grounded acquisition and/or lower token/time/joule cost-to-target`.

### Paper B — separate later hypothesis

> Can candidate-blind, source-grounded evaluation planning with explicit answer requirements, deterministic hard gates, and gated semantic adjudication outperform static metrics and monolithic/dynamic-rubric LLM judges in human agreement, fatal-error detection, plan invariance and evaluation cost?

Paper B should remain separate so Paper A does not introduce a circular-evaluation objection by changing the data method and evaluator simultaneously.

## Pruning decisions

- Keep raw paragraph→QA only as the internal lower baseline.
- Make source-grounded QA qualification the first Paper A treatment.
- Reuse Bonito or the strongest reproducible document/context→synthetic-IFT method as the later external baseline.
- Do not grant KU-first a training arm yet; first require a zero-training data-only coverage/defect-rate gate.
- Do not use ten training paraphrases per KU as the default. Rich held-out paraphrase evaluation is useful; training should later compare only a small dose such as one versus three formulations if qualification succeeds.
- Defer weak/unknown-KU-only training; keep base-knowledge probing first as a contamination/acquisition stratifier.
- Defer Question→Provenance, Answer+Provenance and retrieve→answer target sweeps.
- Freeze PEFT/QLoRA training defaults rather than researching LoRA rank, learning rate, modules, quantization levels or optimizer grids.
- Treat the RTX 5090 as the constrained compute environment, never as the novelty claim.
- Do not build a new generic experiment tracker. Reuse MLflow and add only Cognityx-specific lineage/semantic conventions where needed.
- Do not build arbitrary self-writing evaluator code. The later Evaluator should compile declarative plans over reviewed primitives.
- Do not rent H100 compute until a local result identifies a precise model-capacity/scaling question.

## Minimum training program after the foundation exists

The narrowed research plan targets roughly twelve mandatory local training runs, not thousands:

1. **Zero training runs:** data-only KU gate and base-model knowledge probes.
2. **3 screening runs:** published baseline vs raw Cognityx QA vs qualified Cognityx QA, one 8B seed each.
3. **4 confirmation runs:** add two seeds to each of the two scientifically important 8B arms, yielding three-seed evidence for the main contrast.
4. **3 paraphrase-treatment runs:** only on the winning data method, compare one training wording with a small multi-wording treatment while reusing the existing one-wording runs.
5. **2 model-frontier runs:** two most informative recipes once on 14B.
6. **0–4 conditional 14B confirmation runs:** only if 14B changes the practical conclusion.

Mandatory local total: approximately 12; maximum after a triggered confirmation: approximately 16.

## Highest-information implementation order

1. Automatic provenance→evidence resolution plus full candidate-blind DataForge question/source/gold qualification.
2. Frozen B0/B2 evaluation substrate and base-model knowledge probing.
3. Reproduce the strongest published document→synthetic-IFT baseline and run the first three-arm 8B screen.
4. Build/benchmark the separate Evaluator only after a meaningful human/source-grounded reference exists.
5. Confirm training effects, then decide whether KU-first, 14B or datacenter escalation has earned additional investment.

## Benchmark decisions

- **B0:** controlled synthetic fixtures for mechanism/failure discovery and regression; not external-validity evidence.
- **B1:** like-for-like published synthetic-IFT baseline over the same source texts; Bonito is the primary candidate.
- **B2:** contamination-controlled, version-frozen, enterprise-like real public documentation released after the frozen base model where practical; independently generated and source-qualified final test questions.
- **B3:** IFT-only benchmark remains fallback/isolation only.

## Experiment-tracking decision

Use MLflow as the tracking/index/comparison backend. Cognityx Storage manifests/checksums remain authoritative. Do not duplicate large datasets/adapters merely for MLflow. A future thin experiment layer should orchestrate treatments and parent runs; component-owned data/training/runtime semantics should remain inside DataForge, Training and Inference.

## Immediate stop/go rule

The qualification work must earn its downstream complexity. If qualified data does not materially improve grounded held-out correctness, robustness, unsupported-claim rate or cost-to-target compared with raw data under matched conditions, stop expanding Paper A into KU-first, knowledge-selective, provenance-target, model-size and H100 sweeps.
