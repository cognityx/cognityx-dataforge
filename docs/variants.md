# Research variants

These variants support controlled comparisons. Their hypotheses describe what
an experiment is intended to test; they are not proven findings.

## Paragraph QA

**Concept and hypothesis:** Create examples directly from paragraph-sized
evidence. This is the basic control. It tests whether later semantic methods
improve on a simple layout-based baseline.

**Difference and method:** No knowledge unit or student probe is created.
Inference generates an instruction and answer for each paragraph. DataForge
validates structure, removes exact duplicates and assigns related records to
one deterministic split.

```bash
cognityx-dataforge build paragraph-qa \
  --source <storage-run-manifest-uri> \
  --experiment-id <experiment-id> \
  --config dataforge.toml
```

**Storage, Jobs and Inference:** Artifacts include candidates, rejections,
records, model calls, the input selection and the final manifest. Jobs reports
source resolution, generation, rejection and publication counts. Inference
lineage records model, backend, profile, prompt version, settings, request
identifier, token information and failures.

**Expected strength:** Simple and inexpensive to understand.

**Risks:** Paragraphs can fragment context, repeat knowledge and produce shallow
or unfocused questions.

## Knowledge-Unit QA

**Concept and hypothesis:** First form a self-contained knowledge unit, then
generate examples from it. This tests whether semantic knowledge boundaries
produce better data than document-layout boundaries.

**Difference and method:** Compared with Paragraph QA, this recipe adds
knowledge discovery and source-grounded validation before publication.

```bash
cognityx-dataforge build knowledge-unit-qa \
  --source <storage-run-manifest-uri> \
  --experiment-id <experiment-id> \
  --config dataforge.toml
```

**Storage, Jobs and Inference:** Artifacts add `knowledge-units.jsonl`,
`validations.jsonl`, stage-specific model calls and resumable checkpoints.
Jobs stages are discovery, generation, validation and finalization. Inference
may use independently configured discovery, generation and validation roles.
Every knowledge unit retains its source evidence lineage.

**Expected strength:** More focused and semantically coherent examples.

**Risks:** Discovery can omit, merge or distort source knowledge, and additional
model calls increase cost and failure opportunities.

## Probed Knowledge-Unit QA

**Concept and hypothesis:** Probe the target or student model before creating
training examples. This tests whether prioritizing missing, weak, uncertain or
incorrect knowledge is better than teaching every unit uniformly.

**Difference and method:** Compared with Knowledge-Unit QA, this recipe creates
diagnostic probes, records evidence-free student responses, judges them against
the source, and normally selects `partial` and `unknown` knowledge for QA
generation.

```bash
cognityx-dataforge build knowledge-unit-probed-qa \
  --source <storage-run-manifest-uri> \
  --experiment-id <experiment-id> \
  --config dataforge.toml
```

**Storage, Jobs and Inference:** Artifacts add probes, student responses, probe
judgments and selected units. Jobs stages are discovery, student probing,
judgment, selection, QA generation, validation and finalization. Inference
lineage distinguishes probe generator, student, judge, QA generator and
validator roles and retains request metadata.

**Expected strength:** Reduces examples for knowledge the student already
answers correctly and supports targeted curricula.

**Risks:** A poor probe or judge can misclassify knowledge. Results depend on
the exact student model, profile, prompts and generation settings.

## Probed Mixed QA

**Status:** Planned research variant; not implemented or available as a CLI
recipe.

**Concept and hypothesis:** Use probe results and knowledge type to choose a
richer record style, such as factual, explanatory, comparative, application,
reasoning or misconception correction. This would test adaptive curriculum
construction rather than uniform QA generation.

**Difference and proposed method:** Compared with Probed Knowledge-Unit QA, a
deterministic policy would select a record style before generation and preserve
that decision in lineage.

**Command:** None yet. DataForge deliberately does not document a runnable
command for an unavailable implementation.

**Expected Storage, Jobs and Inference:** A future implementation should retain
the existing probed artifacts and add the selected style, policy version,
style-specific prompt version and validation result. It should reuse existing
Jobs stages and Inference roles rather than creating parallel systems.

**Expected strength:** Greater variety and better matching between knowledge
gaps and teaching style.

**Risks:** More policy and prompt variables make comparisons harder to control,
and richer styles can introduce unsupported reasoning or unnecessary
complexity.
