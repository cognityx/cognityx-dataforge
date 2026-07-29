# Cognityx DataForge

DataForge provides `paragraph-qa`, `knowledge-unit-qa`, and the research
recipe `knowledge-unit-probed-qa`. The latter probes an untrained base model,
judges responses against source evidence, and selectively generates validated
QA. All recipes turn a completed Ingest run into deterministic, inspectable
JSONL artifacts.

The V0 boundary is deliberately small: DataForge owns dataset construction,
validation, generation, rejection records, and dataset manifests. Ingest owns
document parsing, Storage owns durable artifacts, Jobs owns lifecycle state,
and Inference owns model serving.

Start with the [reference guide](reference.md) for the end-to-end CLI sequence.
