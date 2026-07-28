from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import _store_for_uri, build_dataset
from cognityx_dataforge.recipes import normalize_recipe


def _runtime(args: argparse.Namespace) -> StorageRuntime:
    if args.storage_config:
        return StorageRuntime.load(config_file=args.storage_config)
    return StorageRuntime.from_config(StorageConfig.built_in(root=args.storage_root))


def _records_checksum(data: bytes) -> str:
    return hashlib.sha256(json.dumps(data.decode("utf-8"), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cognityx-dataforge")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--input-manifest", required=True)
    build.add_argument("--dataset-name", required=True)
    recipe_group = build.add_mutually_exclusive_group(required=True)
    recipe_group.add_argument("--recipe")
    recipe_group.add_argument("--variant", help="Deprecated alias for --recipe")
    build.add_argument("--config", required=True)
    build.add_argument("--storage-root", default="/tmp/cognityx-dataforge-storage")
    build.add_argument("--storage-config")

    dataset = sub.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    for name in ("show", "export"):
        command = dataset_sub.add_parser(name)
        command.add_argument("dataset_manifest_uri")
        command.add_argument("--storage-root", default="/tmp/cognityx-dataforge-storage")
        command.add_argument("--storage-config")
        if name == "export":
            command.add_argument("--output", required=True)

    args = parser.parse_args()
    runtime = _runtime(args)
    if args.command == "build":
        print(json.dumps(build_dataset(args.input_manifest, args.dataset_name, normalize_recipe(args.recipe, variant=args.variant), args.config, runtime=runtime), indent=2))
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
