# Cognityx DataForge

DataForge builds V0 datasets from a completed Cognityx Ingest run. The
`paragraph-qa` recipe preserves the paragraph baseline; `knowledge-unit-qa`
discovers coherent source-grounded units and validates generated answers.
Both recipes read durable ingest artifacts through Storage and write
reproducible JSONL artifacts plus a dataset manifest.

DataForge is a batch library and CLI, not a FastAPI service. It does not own
identity or ACL policy: Ingest, Storage, Jobs, and Inference retain their
existing boundaries. Inference can be injected for tests and local validation.

See the [reference guide](docs/reference.md) for the end-to-end CLI sequence,
Python API, artifact layout, and validation rules.
