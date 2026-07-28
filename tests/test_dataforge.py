from __future__ import annotations

import json
from pathlib import Path

from cognityx_dataforge.dataset import checksum, deterministic_id, split_for_index
from cognityx_dataforge.paragraphs import paragraph_spans


def test_paragraph_spans():
    assert paragraph_spans("one\n\n two ") == ((0, 3, "one"), (5, 10, " two "))


def test_deterministic_helpers():
    assert deterministic_id("a", "b") == deterministic_id("a", "b")
    assert checksum({"b": 2, "a": 1}) == checksum({"a": 1, "b": 2})
    assert split_for_index(0) == "eval"
    assert split_for_index(1) == "train"

