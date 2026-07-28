# Cognityx DataForge

DataForge provides two V0 recipes: `paragraph-qa` preserves the original
paragraph baseline, while `knowledge-unit-qa` discovers provenance-preserving
knowledge units and validates generated answers against source evidence. Both
turn a completed Ingest run into deterministic, inspectable JSONL artifacts.

The V0 boundary is deliberately small: DataForge owns dataset construction,
validation, generation, rejection records, and dataset manifests. Ingest owns
document parsing, Storage owns durable artifacts, Jobs owns lifecycle state,
and Inference owns model serving.

Start with the [reference guide](reference.md) for the end-to-end CLI sequence.
