"""Load initial world states from YAML."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from ..paths import WORLDS_DIR
from .state import WorldState


def world_path(name: str) -> Path:
    """Resolve a world reference such as ``base_office`` or ``base_office.yaml``."""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return WORLDS_DIR / filename


@cache
def _load_cached(path: str) -> WorldState:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return WorldState.model_validate(raw)


def load_world(name: str) -> WorldState:
    """Load and validate a world. Returns a fresh copy, so callers cannot share state."""
    path = world_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no such world: {path}")
    return _load_cached(str(path)).copy_state()
