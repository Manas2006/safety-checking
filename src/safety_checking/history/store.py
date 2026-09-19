"""Saving and loading frozen prefixes. Content-addressed: the filename is the hash."""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import prefixes_dir
from .builder import Prefix


def prefix_path(prefix_hash: str, directory: Path | None = None) -> Path:
    return (directory or prefixes_dir()) / f"{prefix_hash}.json"


def save_prefix(prefix: Prefix, directory: Path | None = None) -> Path:
    path = prefix_path(prefix.prefix_hash, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_prefix(prefix_hash: str, directory: Path | None = None) -> Prefix:
    path = prefix_path(prefix_hash, directory)
    if not path.exists():
        raise FileNotFoundError(f"no such prefix: {path}")
    return Prefix.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_prefix_file(path: Path) -> Prefix:
    return Prefix.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_prefixes(directory: Path | None = None) -> list[Prefix]:
    directory = directory or prefixes_dir()
    if not directory.exists():
        return []
    return [load_prefix_file(p) for p in sorted(directory.glob("*.json"))]
