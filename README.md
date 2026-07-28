# Cognityx DataForge

DataForge builds the V0 paragraph-baseline dataset from a completed Cognityx
Ingest run. It reads durable ingest artifacts through Storage, validates
evidence and context, generates instruction/answer candidates per paragraph,
and writes reproducible JSONL artifacts plus a dataset manifest.

DataForge is a batch library and CLI, not a FastAPI service. It does not own
identity or ACL policy: Ingest, Storage, Jobs, and Inference retain their
existing boundaries. Inference can be injected for tests and local validation.

See the [reference guide](docs/reference.md) for the end-to-end CLI sequence,
Python API, artifact layout, and validation rules.
