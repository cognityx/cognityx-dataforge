# Reference

## Configure Storage

Normal DataForge commands use the same Cognityx Storage configuration as other
components. Configuration is selected by `cognityx-storage`, including
`COGNITYX_STORAGE_CONFIG`, project configuration, user configuration and the
built-in local profile.

`--storage-config` explicitly selects a configuration file. The older
`--storage-root` option remains as a deprecated advanced override and should
not appear in normal workflows.

## Build a dataset

```bash
cognityx-dataforge build paragraph-qa \
  --source storage://local-main/artifacts/ingest/runs/<run-id>/manifest.json \
  --experiment-id travel-policy-comparison \
  --config dataforge.toml
```

`--source` accepts a Storage URI for a completed Ingest run manifest or an
existing `cognityx.dataforge.input-selection/v1` manifest. DataForge persists a
normalized `input-selection.json` for reproducibility.

`--input-manifest` remains a deprecated compatibility alias for `--source`.
Bundle, context and document identifiers are not accepted directly yet because
Ingest does not currently expose a canonical completed-run lookup for those
references.

The result contains separate experiment, variant, run, job and dataset
identifiers plus the published manifest URI.

## Jobs

Normal builds run synchronously but use a durable Jobs repository. Set
`COGNITYX_DATAFORGE_JOBS_DB` to choose its SQLite database.

```bash
cognityx-dataforge job show <job-id>
cognityx-dataforge job watch <job-id>
cognityx-dataforge job cancel <job-id>
```

DataForge does not expose `--detach`, retry or deletion yet. The current
`cognityx-jobs` contract has no durable worker queue, retry operation or delete
operation, and DataForge does not create a competing job framework.

## Logical artifact layout

Physical locations remain hidden behind Cognityx Storage. Logical keys follow:

```text
dataforge/experiments/<experiment-id>/
  variants/<variant-id>/
    runs/<run-id>/
      input-selection.json
      run-events.jsonl
      datasets/<dataset-id>/<dataset-version>/
        candidates.jsonl
        rejections.jsonl
        records.jsonl
        model-calls*.jsonl
        checkpoints/
        manifest.json
```

The final `manifest.json` is written only after validation, splitting,
deduplication and cancellation checks succeed. Storage supplies physical blob
deduplication; DataForge separately removes exact duplicate generated records
and records `duplicate_count`.

## Splits and statistics

`[splitting].seed` controls deterministic train, validation and test assignment:

```toml
[splitting]
seed = "experiment-2026-01"
```

Related records share a split group based on source asset, then document,
knowledge unit or evidence identity. Manifests record accepted, rejected,
duplicate, truncation and inference-failure counts.

## Export

```bash
cognityx-dataforge dataset export \
  <dataset-manifest-uri> \
  --output records.jsonl
```

Export resolves `records_uri` through Storage and verifies its checksum before
writing the requested local output file.
