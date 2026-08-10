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
normalized `input-selection.json` for reproducibility. Current Ingest runs also
provide page, block, section, object and parser-decision details. DataForge
validates these details directly from Storage and never reads `source.pdf` or
another copy of the original SourceAsset.

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
        answer-requirements.jsonl
        source-answerability.jsonl
        reference-qualification.jsonl
        qualification-decisions.jsonl
        accepted.jsonl
        rejections.jsonl
        needs-review.jsonl
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

Generated records retain the stable source anchors supplied by Ingest. They
also carry an enrichment identity computed from the source content hash,
anchors, representation type, generation method, model version and effective
configuration. Equivalent future work can use this identity to find an
existing artifact before running another model call.

## Deletion and cleanup

Deleting a source in Ingest is a logical deletion first: it hides the
SourceAsset or bundle but does not immediately erase shared bytes. Cognityx
Storage keeps content-addressed blobs while any live reference still needs
them. DataForge datasets remain immutable records of the source selection used
at build time.

An always-running Storage cleanup service is planned. It will remove blobs only
after retention rules have passed and Storage confirms that no live reference
remains. Until that service exists, cleanup is an explicit maintenance action;
DataForge does not delete source blobs or bypass Storage safety checks.

## Future roadmap

The following work is intentionally deferred:

- reference-only external URIs that are ingested without copying source bytes;
- durable distributed workers for Ingest and DataForge jobs;
- the always-running Storage cleanup service described above; and
- materialized vector, late-interaction, keyword, graph and SQL indexes.

The stable anchors and enrichment identity are the handoff for that future
retrieval work. This release does not create embeddings or indexes.

## Export

```bash
cognityx-dataforge dataset export \
  <dataset-manifest-uri> \
  --output records.jsonl
```

Export resolves `records_uri` through Storage and verifies its checksum before
writing the requested local output file.

## Freeze evaluation sets and a research package

An exact-recall set copies only accepted training records, gives every copy a
new evaluation record ID and retains `source_record_id`:

```bash
cognityx-dataforge evaluation-set exact-recall \
  <dataset-manifest-uri> \
  --name policy-exact-recall
```

Import test-only paraphrases or held-out knowledge units from JSONL:

```bash
cognityx-dataforge evaluation-set import \
  --input paraphrases.jsonl \
  --name policy-paraphrases \
  --research-role paraphrase_evaluation
```

Imported records must contain stable IDs, a fact or knowledge-unit group, and
resolvable provenance. Evaluation records always use `split=evaluation`, carry
an explicit `research_role`, and set `training_eligible=false`. Duplicate IDs
or a trainable evaluation record stop publication.

Link the dataset and frozen sets:

```bash
cognityx-dataforge research-package create \
  --name policy-qualification-comparison \
  --dataset-manifest <dataset-manifest-uri> \
  --evaluation-manifest <exact-recall-manifest-uri> \
  --evaluation-manifest <paraphrase-manifest-uri>
```

The package uses `cognityx.dataforge.research-package/v1` and requires an
`exact_recall` set. Dataset and evaluation checksums are verified before its
manifest is written.
