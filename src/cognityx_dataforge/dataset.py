from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def checksum(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def deterministic_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:32]


def split_for_index(index: int) -> str:
    return "train" if index % 10 else "eval"

