"""Load scenarios from YAML and validate them against their world."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..paths import SCENARIOS_DIR
from ..world.loader import load_world
from ..world.state import WorldState
from .schema import Scenario


def scenario_path(name: str) -> Path:
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return SCENARIOS_DIR / filename


def load_scenario(name: str, *, validate: bool = True) -> Scenario:
    path = scenario_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no such scenario: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(raw)
    if validate:
        scenario.validate_against_world(load_scenario_world(scenario))
    return scenario


def load_scenario_world(scenario: Scenario) -> WorldState:
    """The scenario's initial world, fresh each call."""
    return load_world(scenario.world_ref)


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))


def load_all_scenarios() -> list[Scenario]:
    return [load_scenario(name) for name in list_scenarios()]
