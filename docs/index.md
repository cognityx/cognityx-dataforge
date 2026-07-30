# Cognityx DataForge

DataForge turns evidence extracted from source documents into examples that
can be used to train or evaluate a model. It sits between document processing
and model training in the Cognityx application flow:

```text
Source files -> Ingest -> DataForge -> Training and evaluation
                parses     builds      use the examples
                evidence   examples
```

DataForge provides three ways to build examples: `paragraph-qa`,
`knowledge-unit-qa`, and the research recipe `knowledge-unit-probed-qa`. The
research recipe checks what an untrained base model already knows, compares its
responses with the source evidence, and creates validated question-and-answer
examples only where they are useful. Running a recipe again with the same
inputs and settings produces the same inspectable JSONL artifacts. This
repeatable behavior is technically called deterministic output.

The V0 boundary is deliberately small: DataForge owns dataset construction,
validation, generation, rejection records, and dataset manifests. Ingest owns
document parsing, Storage owns durable artifacts, Jobs owns lifecycle state,
and Inference owns model serving.

Read the [introduction](introduction.md) for the main concepts, then use the
[reference guide](reference.md) for the end-to-end command-line sequence.
