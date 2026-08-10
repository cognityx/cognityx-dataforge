# Qualification and research packages

DataForge can now answer a narrow research question: does source-grounded
qualification improve a raw paragraph dataset when everything else stays the
same?

```text
Source document
  -> Ingest evidence
  -> DataForge raw paragraph candidates
       -> raw dataset
       -> candidate-blind qualification -> qualified dataset
  -> frozen evaluation sets
  -> research package
  -> Cognityx Training comparison
```

## The controlled comparison

Use the same source selection, generator, base model, trainer, training-token
budget and evaluator. Compare `paragraph-qa` with
`paragraph-qa-qualified`. Both recipes use the same paragraph spans,
generation prompt and deterministic candidate identity. The qualified recipe
adds only the qualification stages.

This release deliberately does not add new knowledge-unit recipes, automatic
paraphrase generation, adapter serving, a custom experiment database or a
training-to-Inference adapter handoff.

## Candidate-blind stages

1. The question alone defines the information a complete answer must provide.
   Each requirement declares its role, value type, unit, relation, membership,
   qualifier, exact-match rule, and source-inference policy where applicable.
   This frozen description is the answer requirements
   (`cognityx.dataforge.answer-requirements/v1`).
2. The requirements plus source evidence decide whether the source contains
   that information. No gold or generated reference is visible. It freezes
   supported facts, constraints, and evidence anchors without carrying the raw
   source text forward. The output is source answerability
   (`cognityx.dataforge.source-answerability/v1`).
3. The frozen first two semantic outputs plus the generated reference measure
   required slot coverage, unsupported claims and contradictions. Raw source
   text is not present at this boundary. The output is reference qualification
   (`cognityx.dataforge.reference-qualification/v1`).
4. Code applies deterministic gates and emits accepted, rejected or
   needs-review (`cognityx.dataforge.qualification-decision/v1`).

Numeric values stay bound to their roles and units; for example, 23 sustained
knots and one 31-knot gust cannot be swapped. Exact phrases must be complete,
lists cannot omit required members or add unsupported members, and provenance
must resolve. Physical PDF page position and the page label printed in the
document remain separate coordinates.

These checks read the declared requirement structure. DataForge does not keep
a dictionary of policy-specific role names, so a finance approval count and
approver list use the same machinery as the pilot fixtures.

The final artifact keeps distinct answers to distinct questions: whether the
question is valid, whether the source can answer it, whether the reference is
correct and complete, whether every claim is supported, whether the machinery
ran reliably, and whether the record may enter Training. A factually correct
reference can therefore still be rejected because it adds an unsupported
claim, without being mislabeled as factually incorrect.

The overtime fixture known as R09 is intentionally rejected. Its source says
approval is required but does not identify the approval type or authority
asked for. DataForge does not invent a supervisor, Human Resources manager or
other missing authority.

## Retries and publication

Each semantic stage has a bounded attempt count under
`[qualification].max_attempts`. Raw responses and parsing errors are retained.
If all attempts fail, factual quality is `not_determined` and the operational
decision is `needs_review`. Earlier artifacts and checkpoints remain durable;
the dataset manifest is still written last.

## Frozen evaluation roles

- `exact_recall` copies accepted training records with distinct evaluation IDs.
- `paraphrase_evaluation` tests unseen wording for trained facts.
- `heldout_knowledge_unit` tests facts or knowledge units excluded from
  training.

The last two are imported and frozen rather than generated automatically.
Their scientific version comes from the records, logical source identity, and
freeze policy—not the temporary path from which the file was imported. Their
creation time, frozen-content checksum, and manifest checksum make later
changes visible. Every evaluation record is test-only
(`training_eligible=false`). Loading a research package rechecks each linked
evaluation set, its record count and checksum, role, provenance, and test-only
status.

Read [failure-register traceability](failure-traceability.md) for the empirical
reasons behind these boundaries and the tests that protect them.
