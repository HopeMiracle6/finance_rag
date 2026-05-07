from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (base_dir or project_root()) / raw


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def model_dump(obj: BaseModel | dict) -> dict:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def read_jsonl(path: str | Path, model: type[T] | None = None) -> list[T] | list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[T] | list[dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(model(**data) if model else data)
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(model_dump(record), ensure_ascii=False) + "\n")


def read_json(path: str | Path, default: object | None = None) -> object:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: object) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def format_page_range(page: int | None = None, page_start: int | None = None, page_end: int | None = None) -> str:
    start = page_start or page
    end = page_end or page
    if start is None and end is None:
        return "无"
    if start is not None and end is not None and start != end:
        return f"{start}-{end}"
    return str(start or end)
