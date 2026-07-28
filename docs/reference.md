# Reference

## Build a dataset

```bash
cognityx-dataforge build \
  --run-manifest storage://runs/<run-id>/manifest.json \
  --dataset-id my-dataset \
  --variant v0
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

The manifest records source identity, prompt and variant metadata, split
counts, checksums, and the URIs of the JSONL artifacts. Rejected generations
remain inspectable instead of silently disappearing.

## Python API

```python
from cognityx_dataforge.build import build_dataset

result = build_dataset(
    run_manifest_uri="storage://runs/run-123/manifest.json",
    dataset_id="my-dataset",
    variant="v0",
)
print(result["dataset_manifest_uri"])
```

The build function accepts an optional injected inference client, keeping unit
tests and local dry runs independent of a live model server.

For a local smoke test, provide a completed run manifest and configure the
Storage backend used by the workspace. A live Inference endpoint is only
required when the injected client is omitted.

## Validation

Before generation, DataForge checks the ingest manifest schema, evidence
availability, document/context consistency, and paragraph boundaries. Invalid
or incomplete model responses are written to `rejections.jsonl` with a reason
and excluded from `records.jsonl`.
