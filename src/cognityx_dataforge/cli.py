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
from cognityx_dataforge.config import resolve_dataforge_config
from cognityx_dataforge.execution import load_job_repository
from cognityx_dataforge.human import render_human
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
    return hashlib.sha256(
        json.dumps(
            data.decode("utf-8"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(prog="cognityx-dataforge")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    for name in ("show", "validate"):
        command = config_sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--human", action="store_true")

    build = sub.add_parser("build")
    build.add_argument("recipe_name", nargs="?")
    source_group = build.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source")
    source_group.add_argument(
        "--input-manifest",
        help="Deprecated alias for --source.",
    )
    build.add_argument("--experiment-id", required=True)
    build.add_argument(
        "--run-id",
        help="Stable caller-supplied run identity for safe retry and resume.",
    )
    build.add_argument("--dataset-name")
    recipe_group = build.add_mutually_exclusive_group()
    recipe_group.add_argument("--recipe")
    recipe_group.add_argument("--variant", help="Deprecated alias for --recipe")
    build.add_argument("--config", required=True)
    build.add_argument("--storage-root")
    build.add_argument("--storage-config")
    build.add_argument("--jobs-database")
    build.add_argument("--human", action="store_true")

    dataset = sub.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("show", "export"):
        command = dataset_sub.add_parser(name)
        command.add_argument("dataset_manifest_uri")
        command.add_argument("--storage-root")
        command.add_argument("--storage-config")
        if name == "export":
            command.add_argument("--output", required=True)
        else:
            command.add_argument("--human", action="store_true")

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    for name in ("show", "watch", "cancel"):
        command = job_sub.add_parser(name)
        command.add_argument("job_id")
        command.add_argument("--jobs-database")
        if name == "watch":
            command.add_argument("--interval", type=float, default=0.5)
        command.add_argument("--human", action="store_true")

    evaluation_set = sub.add_parser("evaluation-set")
    evaluation_sub = evaluation_set.add_subparsers(
        dest="evaluation_command", required=True
    )
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
        command.add_argument("--human", action="store_true")

    research_package = sub.add_parser("research-package")
    research_sub = research_package.add_subparsers(
        dest="research_command", required=True
    )
    package_create = research_sub.add_parser("create")
    package_create.add_argument("--name", required=True)
    package_create.add_argument("--dataset-manifest", required=True)
    package_create.add_argument("--evaluation-manifest", action="append", required=True)
    package_show = research_sub.add_parser("show")
    package_show.add_argument("research_package_manifest_uri")
    for command in (package_create, package_show):
        command.add_argument("--storage-root")
        command.add_argument("--storage-config")
        command.add_argument("--human", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "config":
        try:
            report = resolve_dataforge_config(args.config).to_dict()
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            report = {
                "component": "dataforge",
                "configuration_kind": "scientific-workload",
                "valid": False,
                "master_config": {
                    "kind": "file",
                    "path": str(args.config.expanduser().resolve()),
                    "selected_by": "explicit",
                    "sha256": None,
                },
                "config_layers": [],
                "field_sources": {},
                "overrides": [],
                "effective": {},
                "warnings": [],
                "errors": [{"code": "configuration_invalid", "message": str(exc)}],
            }
            _write(report, human=args.human)
            return 2
        _write(report, human=args.human)
        return 0
    if args.command == "job":
        jobs = _jobs(args)
        if args.job_command == "cancel":
            jobs.request_cancel(args.job_id)
            _write(
                {"job_id": args.job_id, "state": jobs.get(args.job_id).state},
                human=args.human,
                sort_keys=False,
            )
            return
        after = 0
        while True:
            record = jobs.get(args.job_id)
            events = jobs.events(args.job_id, after=after)
            if events:
                after = events[-1]["sequence"]
            _write(
                {"job": record.__dict__, "events": events},
                human=args.human,
                sort_keys=False,
                flush=True,
            )
            if args.job_command == "show" or record.state in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
            }:
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
        _write(result, human=args.human)
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
        _write(result, human=args.human)
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
        result = build_dataset(
            source,
            args.dataset_name or args.experiment_id,
            recipe,
            args.config,
            experiment_id=args.experiment_id,
            requested_run_id=args.run_id,
            runtime=runtime,
            jobs=_jobs(args),
        )
        _write(result, human=args.human, sort_keys=False)
        return
    manifest_store, manifest_key = _store_for_uri(
        runtime, args.dataset_manifest_uri, role_name="dataset"
    )
    with manifest_store.open(manifest_key) as handle:
        manifest = json.load(handle)
    if args.dataset_command == "show":
        _write(manifest, human=args.human)
        return
    records_store, records_key = _store_for_uri(
        runtime, manifest["records_uri"], role_name="dataset"
    )
    with records_store.open(records_key) as handle:
        records = handle.read()
    if _records_checksum(records) != manifest["records_checksum"]:
        raise SystemExit("records.jsonl checksum verification failed")
    Path(args.output).write_bytes(records)


def _write(
    value: object,
    *,
    human: bool,
    sort_keys: bool = True,
    flush: bool = False,
) -> None:
    output = (
        render_human(value)
        if human
        else json.dumps(value, indent=2, sort_keys=sort_keys)
    )
    print(output, flush=flush)
