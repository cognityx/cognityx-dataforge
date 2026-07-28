from __future__ import annotations

import json
from pathlib import Path

from cognityx_jobs import JobRepository
from cognityx_storage import StorageClient
from cognityx_storage.local import LocalStorageBackend

from cognityx_dataforge.build import build_dataset


class FakeClient:
    def chat(self, **kwargs):
        return {"choices": [{"message": {"content": json.dumps({"instruction": "Ask", "answer": "Answer"})}}]}


def test_build_dataset(tmp_path: Path):
    storage = StorageClient(LocalStorageBackend(tmp_path / "storage"))
    manifest = {
        "schema": "cognityx.ingest.run-manifest/v1",
        "document_id": "pdf-1",
        "evidence_ids": ["e1"],
        "evidence_payload": {
            "evidence": [
                {
                    "evidence_id": "e1",
                    "document_id": "pdf-1",
                    "page_number": 1,
                    "text": "alpha beta",
                    "char_start": 0,
                    "char_end": 10,
                    "source_asset_id": "asset-1",
                }
            ]
        },
    }
    storage.put_json("input/manifest.json", manifest)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[models.generator]\nmodel="Qwen/Qwen3-32B"\nbackend="vllm"\nprofile="int4"\nmax_output_tokens=1024\n', encoding="utf-8")
    result = build_dataset("storage://input/manifest.json", "demo", "v0", cfg, storage=storage, jobs=JobRepository(":memory:"), inference_client=FakeClient())
    assert result["variant"] == "v0"
    assert result["record_count"] == 1
