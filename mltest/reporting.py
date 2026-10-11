"""Output writers: CSV, JSON and the Markdown run summary."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _plain(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict, str)):
        return value.item()  # numpy scalar
    return value


def _json_default(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else str(value)


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    return path


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for record in records:
        columns += [key for key in record if key not in columns]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({k: _plain(v) for k, v in record.items()})
    return path


def format_value(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def markdown_table(records: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in records:
        lines.append("| " + " | ".join(format_value(record.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)
