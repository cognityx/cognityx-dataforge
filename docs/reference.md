# Reference

## Build a dataset

```bash
cognityx-dataforge build \
  --input-manifest storage://local-main/artifacts/ingest/runs/<run-id>/manifest.json \
  --dataset-name my-dataset \
  --recipe paragraph-qa \
  --config dataforge.toml \
  --storage-root /tmp/cognityx-storage
```

The command returns JSON containing the job id, dataset id, record count, and
dataset manifest URI.

## Artifact layout

```text
datasets/<dataset-id>/<dataset-version>/
  manifest.json
  records.jsonl
  candidates.jsonl
  rejections.jsonl
  run-events.jsonl
```

The manifest records source identity, recipe, prompt and model metadata, split
counts, checksums, and the URIs of the JSONL artifacts. Rejected generations
remain inspectable instead of silently disappearing. The compatibility aliases
`v0` and `v1` map to `paragraph-qa` and `knowledge-unit-qa`; `--variant` is
deprecated.

## Knowledge-unit QA

```bash
cognityx-dataforge build \
  --input-manifest <run-manifest-uri> \
  --dataset-name research \
  --recipe knowledge-unit-qa \
  --config dataforge-knowledge.toml \
  --storage-root /tmp/cognityx-storage
```

This recipe discovers multiple provenance-preserving knowledge units from each
evidence record, generates instruction-answer candidates, and validates them
against the cited evidence. It writes `knowledge-units.jsonl`,
`validations.jsonl`, and `rejections.jsonl`; rejected records are excluded
from `records.jsonl`.

Knowledge-unit QA runs in three resumable stages: discovery, QA generation,
then validation. Each model request is stateless and includes its complete
material for that request. Configure `context_limit_tokens` at the TOML root;
DataForge calls `count_input_tokens` before every request and records the role,
model settings, token budget, prompt version, and evidence IDs. Requests that
would exceed the budget are rejected with structured reasons rather than
silently truncated.

Each dataset also contains explicit `checkpoints/{discovery,generation,
validation,finalization}.json` files. Checkpoints include stage identity,
artifact checksums, completion status, and row counts, so an empty but
successfully completed stage can resume safely without repeating model calls.

The model roles may be configured independently:

```toml
context_limit_tokens = 32768

[models.knowledge_unit]
model = "Qwen/Qwen3-32B"
backend = "vllm"
profile = "int4"
max_output_tokens = 2048

[models.qa_generator]
model = "Qwen/Qwen3-32B"
backend = "vllm"
profile = "int4"
max_output_tokens = 1024

[models.validator]
model = "Qwen/Qwen3-14B"
backend = "vllm"
profile = "int4"
max_output_tokens = 512
```

When the first two roles are absent, they fall back to `[models.generator]`.

## Python API

```python
from cognityx_dataforge.build import build_dataset

result = build_dataset(
    input_manifest_uri="storage://local-main/artifacts/ingest/runs/run-123/manifest.json",
    dataset_name="my-dataset",
    recipe="paragraph-qa",
    config_path="dataforge.toml",
)
print(result["dataset_manifest_uri"])
```

The build function accepts an optional injected inference client, keeping unit
tests and local dry runs independent of a live model server.

For a local smoke test, provide a completed run manifest and configure the
same Storage Runtime used by Ingest with `--storage-root` or `--storage-config`.
A live Inference endpoint is only required when the injected client is omitted;
install it with `pip install cognityx-dataforge[inference]`.

To export records, use `dataset export <manifest-uri> --output records.jsonl`.
The command resolves `records_uri`, verifies its checksum, and writes only the
records JSONL artifact.

## Validation

Before generation, DataForge checks the ingest manifest schema, evidence
availability, document/context consistency, and paragraph boundaries. Invalid
or incomplete model responses are written to `rejections.jsonl` with a reason
and excluded from `records.jsonl`.
