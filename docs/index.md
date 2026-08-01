# Cognityx DataForge

DataForge turns evidence from source documents into examples that can train or
evaluate a model. It sits after Cognityx Ingest and before Cognityx Training:

```text
Source files -> Ingest -> DataForge -> Training and evaluation
                parses     builds      use the examples
                evidence   datasets
```

DataForge does not parse documents, choose physical storage paths, manage model
servers or train models. Ingest prepares evidence, Storage keeps artifacts,
Jobs records execution state, Inference performs model calls, and Training
consumes successful datasets.

## What a build does

A build:

1. resolves a completed Ingest source and its page-level trace from configured
   Storage, without reopening the original file;
2. records the exact source selection and effective configuration;
3. creates a Jobs-managed run with progress events;
4. runs one preparation recipe through Cognityx Inference;
5. validates, deduplicates and groups related records into stable splits; and
6. publishes an immutable dataset only after successful validation.

Failed or cancelled runs keep their inspectable run artifacts but do not
publish a dataset manifest.

The page-level trace is technically called provenance. DataForge validates it
against the source document and content hash, carries stable page and block
anchors into generated records, and records a reusable enrichment identity.
This lets later research reuse equivalent derived work instead of silently
generating it again.

## Inference boundary

All model and agent calls use the Cognityx Inference client. Configuration
selects an approved provider, model, backend, numeric profile and named server
profile. For a local model, the client can ask the Inference manager to start
the configured server and load the selected model. DataForge does not start a
separate model runtime or choose an unapproved model on its own.

## Research purpose

DataForge provides three implemented recipes for comparing data-preparation
methods:

- **Paragraph QA** creates examples directly from paragraphs.
- **Knowledge-Unit QA** first identifies a self-contained piece of knowledge.
- **Probed Knowledge-Unit QA** first checks what the student model knows.

**Probed Mixed QA** is a planned fourth experiment that would choose different
record styles from the probe result and knowledge type. It is not currently an
executable recipe, and no research hypothesis in this documentation should be
read as a proven result or a novelty claim.

Read the [introduction](introduction.md) for terminology, the
[variant guide](variants.md) for the research progression, and the
[reference](reference.md) for commands and artifacts.
