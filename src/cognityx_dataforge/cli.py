from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognityx_storage import StorageClient
from cognityx_storage.local import LocalStorageBackend

from cognityx_dataforge.build import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="cognityx-dataforge")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--input-manifest", required=True)
    build.add_argument("--dataset-name", required=True)
    build.add_argument("--variant", required=True)
    build.add_argument("--config", required=True)
    show = sub.add_parser("dataset")
    dataset_sub = show.add_subparsers(dest="dataset_command", required=True)
    dataset_show = dataset_sub.add_parser("show")
    dataset_show.add_argument("dataset_manifest_uri")
    dataset_export = dataset_sub.add_parser("export")
    dataset_export.add_argument("dataset_manifest_uri")
    dataset_export.add_argument("--output", required=True)
    args = parser.parse_args()
    storage = StorageClient(LocalStorageBackend())
    if args.command == "build":
        print(json.dumps(build_dataset(args.input_manifest, args.dataset_name, args.variant, args.config, storage=storage), indent=2))
    elif args.dataset_command == "show":
        with storage.open(args.dataset_manifest_uri.removeprefix("storage://")) as handle:
            print(handle.read().decode("utf-8"))
    elif args.dataset_command == "export":
        with storage.open(args.dataset_manifest_uri.removeprefix("storage://")) as handle:
            Path(args.output).write_bytes(handle.read())
