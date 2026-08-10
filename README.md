# Cognityx DataForge

DataForge turns passages from source documents into examples that can train a
model or test what it learned. It sits between Cognityx Ingest, which extracts
the passages, and Cognityx Training, which uses the published records.

The `paragraph-qa` recipe preserves the raw paragraph baseline.
`paragraph-qa-qualified` creates the same candidates and then checks whether
the question can be answered from the source and whether the generated
reference actually answers it. Knowledge-unit recipes remain available for
separate research. Every recipe reads durable Ingest artifacts through Storage
and writes reproducible JSONL artifacts plus a manifest written last.

DataForge is a batch library and CLI, not a FastAPI service. It does not own
identity or ACL policy: Ingest, Storage, Jobs, and Inference retain their
existing boundaries. Inference can be injected for tests and local validation.

See the [qualification and research guide](docs/qualification-research.md) for
the controlled comparison and frozen evaluation sets, and the
[reference guide](docs/reference.md) for the CLI and artifact layout.
