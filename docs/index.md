# Cognityx DataForge

DataForge is the Cognityx V0 paragraph-baseline dataset builder. It turns a
completed Ingest run into deterministic, inspectable JSONL artifacts for
training and evaluation.

The V0 boundary is deliberately small: DataForge owns dataset construction,
validation, generation, rejection records, and dataset manifests. Ingest owns
document parsing, Storage owns durable artifacts, Jobs owns lifecycle state,
and Inference owns model serving.

Start with the [reference guide](reference.md) for the end-to-end CLI sequence.
