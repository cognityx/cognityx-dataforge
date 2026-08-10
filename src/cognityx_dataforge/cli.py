from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import _store_for_uri, build_dataset
from cognityx_dataforge.execution import default_jobs_database, load_job_repository
from cognityx_dataforge.recipes import normalize_recipe
from cognityx_dataforge.research import (
    create_exact_recall_set,
    create_research_package,
    import_evaluation_set,
    load_research_package,
)


def _runtime(args: argparse.Namespace) -> StorageRuntime:
    storage_config = getattr(args, "storage_config", None)
    storage_root = getattr(args, "storage_root", None)
    if storage_root:
        warnings.warn(
            "--storage-root is deprecated; configure Cognityx Storage instead.",
            FutureWarning,
            stacklevel=2,
        )
        return StorageRuntime.from_config(StorageConfig.built_in(root=storage_root))
    if storage_config:
        return StorageRuntime.load(config_file=args.storage_config)
    return StorageRuntime.load()


def _jobs(args: argparse.Namespace) -> JobRepository:
    return load_job_repository(getattr(args, "jobs_database", None))


def _records_checksum(data: bytes) -> str:
    return hashlib.sha256(json.dumps(data.decode("utf-8"), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cognityx-dataforge")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("recipe_name", nargs="?")
    source_group = build.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source")
    source_group.add_argument(
        "--input-manifest",
        help="Deprecated alias for --source.",
    )
    build.add_argument("--experiment-id", required=True)
    build.add_argument("--dataset-name")
    recipe_group = build.add_mutually_exclusive_group()
    recipe_group.add_argument("--recipe")
    recipe_group.add_argument("--variant", help="Deprecated alias for --recipe")
    build.add_argument("--config", required=True)
    build.add_argument("--storage-root")
    build.add_argument("--storage-config")
    build.add_argument("--jobs-database")

    dataset = sub.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("show", "export"):
        command = dataset_sub.add_parser(name)
        command.add_argument("dataset_manifest_uri")
        command.add_argument("--storage-root")
        command.add_argument("--storage-config")
        if name == "export":
            command.add_argument("--output", required=True)

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    for name in ("show", "watch", "cancel"):
        command = job_sub.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--jobs-database")
        if name == "watch":
            command.add_argument("--interval", type=float, default=0.5)

    evaluation_set = sub.add_parser("evaluation-set")
    evaluation_sub = evaluation_set.add_subparsers(dest="evaluation_command", required=True)
    exact_recall = evaluation_sub.add_parser("exact-recall")
    exact_recall.add_argument("dataset_manifest_uri")
    exact_recall.add_argument("--name")
    imported = evaluation_sub.add_parser("import")
    imported.add_argument("--input", required=True)
    imported.add_argument("--name", required=True)
    imported.add_argument(
        "--research-role",
        required=True,
        choices=("paraphrase_evaluation", "heldout_knowledge_unit"),
    )
    for command in (exact_recall, imported):
        command.add_argument("--storage-root")
        command.add_argument("--storage-config")

    research_package = sub.add_parser("research-package")
    research_sub = research_package.add_subparsers(dest="research_command", required=True)
    package_create = research_sub.add_parser("create")
    package_create.add_argument("--name", required=True)
    package_create.add_argument("--dataset-manifest", required=True)
    package_create.add_argument("--evaluation-manifest", action="append", required=True)
    package_show = research_sub.add_parser("show")
    package_show.add_argument("research_package_manifest_uri")
    for command in (package_create, package_show):
        command.add_argument("--storage-root")
        command.add_argument("--storage-config")

    args = parser.parse_args()
    if args.command == "job":
        jobs = _jobs(args)
        if args.job_command == "cancel":
            jobs.request_cancel(args.job_id)
            print(json.dumps({"job_id": args.job_id, "state": jobs.get(args.job_id).state}, indent=2))
            return
        after = 0
        while True:
            record = jobs.get(args.job_id)
            events = jobs.events(args.job_id, after=after)
            if events:
                after = events[-1]["sequence"]
            print(json.dumps({"job": record.__dict__, "events": events}, indent=2))
            if args.job_command == "show" or record.state in {"completed", "failed", "cancelled", "interrupted"}:
                return
            time.sleep(args.interval)

    runtime = _runtime(args)
    if args.command == "evaluation-set":
        if args.evaluation_command == "exact-recall":
            result = create_exact_recall_set(
                runtime,
                args.dataset_manifest_uri,
                evaluation_set_name=args.name,
            )
        else:
            result = import_evaluation_set(
                runtime,
                args.input,
                evaluation_set_name=args.name,
                research_role=args.research_role,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "research-package":
        if args.research_command == "create":
            result = create_research_package(
                runtime,
                package_name=args.name,
                dataset_manifest_uri=args.dataset_manifest,
                evaluation_manifest_uris=args.evaluation_manifest,
            )
        else:
            result = load_research_package(runtime, args.research_package_manifest_uri)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "build":
        source = args.source or args.input_manifest
        if args.input_manifest:
            warnings.warn(
                "--input-manifest is deprecated; use --source.",
                FutureWarning,
                stacklevel=2,
            )
        recipe = normalize_recipe(args.recipe_name or args.recipe, variant=args.variant)
        print(json.dumps(build_dataset(
            source,
            args.dataset_name or args.experiment_id,
            recipe,
            args.config,
            experiment_id=args.experiment_id,
            runtime=runtime,
            jobs=_jobs(args),
        ), indent=2))
        return
    manifest_store, manifest_key = _store_for_uri(runtime, args.dataset_manifest_uri, role_name="dataset")
    with manifest_store.open(manifest_key) as handle:
        manifest = json.load(handle)
    if args.dataset_command == "show":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    records_store, records_key = _store_for_uri(runtime, manifest["records_uri"], role_name="dataset")
    with records_store.open(records_key) as handle:
        records = handle.read()
    if _records_checksum(records) != manifest["records_checksum"]:
        raise SystemExit("records.jsonl checksum verification failed")
    Path(args.output).write_bytes(records)
