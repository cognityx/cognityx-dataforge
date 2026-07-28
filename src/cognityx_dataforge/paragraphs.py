from __future__ import annotations

import re


def paragraph_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    spans: list[tuple[int, int, str]] = []
    start = None
    index = 0
    for match in re.finditer(r"\n[ \t]*\n", text):
        chunk = text[index:match.start()]
        if chunk.strip():
            spans.append((index, match.start(), chunk))
        index = match.end()
    chunk = text[index:]
    if chunk.strip():
        spans.append((index, len(text), chunk))
    return tuple(spans)
