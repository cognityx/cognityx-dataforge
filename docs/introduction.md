# Introduction

Training-data creation starts with an original item, such as a PDF, and ends
with examples a model can learn from or be tested against.

## Core terms

A **source asset** is the original digital item received or referenced by
Cognityx. Examples include a PDF, Word document, web page, image, audio file or
database record.

A **source document** is a source asset that Cognityx can parse into text,
sections, tables, pages or similar structures.

**Evidence** is the exact source passage supporting an answer or claim. It
includes enough location information to find that passage again.

**Provenance** records where evidence came from and how it was processed. The
ordered trace from an original source to a later result is called its
**lineage**.

**Ground truth** is a fact, label or expected result accepted as correct for a
defined context.

A **reference answer** expresses that ground truth as an answer to a particular
question or instruction.

A **knowledge unit** is a self-contained fact, rule, concept, procedure or
relationship derived from evidence. It says what should be taught or tested; it
is not yet a question or answer.

## Example

For a company travel-policy PDF:

```text
PDF -> parsed text -> paragraph -> evidence -> knowledge unit -> training example
```

The sentence saying that claims must be submitted within 30 days is evidence.
The validated 30-day rule is ground truth. “Employees must submit travel claims
within 30 days” is a reference answer. The complete trace is its lineage.

## Build identities

DataForge keeps different kinds of identity separate:

- `experiment_id` names the overall research comparison.
- `variant_id` identifies one recipe and its meaningful configuration.
- `run_id` identifies one execution attempt.
- `job_id` identifies its Cognityx Jobs lifecycle record.
- `dataset_id` and `dataset_version` identify a successfully published,
  immutable dataset.

A retry therefore receives a new run and job identity even when its experiment
and variant are unchanged.
