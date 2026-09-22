"""Which episodes make up a history of each length.

Layers, shortest first. The shortest layer is the tail: the fixed final block of calls that
is byte-identical at every length and under every ``prior_check_pattern``. A history of
length L is the concatenation of the layers for lengths <= L, oldest layer first:

    history(50) = layer(50) + layer(20) + tail

so a shorter history is a suffix of every longer one. Adding a length means adding a layer;
nothing here is special-cased to 5/20/50.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ..paths import EPISODES_DIR
from .episodes import get_episodes

PriorCheckPattern = Literal["none", "performed"]
PRIOR_CHECK_PATTERNS: tuple[PriorCheckPattern, ...] = ("none", "performed")


class PlanLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    length: int
    #: episode ids per prior_check_pattern, oldest first within the layer
    episodes: dict[str, list[str]]

    @model_validator(mode="after")
    def _check_patterns(self) -> PlanLayer:
        if set(self.episodes) != set(PRIOR_CHECK_PATTERNS):
            raise ValueError(f"layer {self.length}: expected patterns {PRIOR_CHECK_PATTERNS}")
        return self


class HistoryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: list[PlanLayer]

    @model_validator(mode="after")
    def _check_layers(self) -> HistoryPlan:
        lengths = [layer.length for layer in self.layers]
        if lengths != sorted(set(lengths)):
            raise ValueError("layers must be sorted by length and unique")
        for pattern in PRIOR_CHECK_PATTERNS:
            running = 0
            for layer in self.layers:
                running += sum(e.n_calls for e in get_episodes(layer.episodes[pattern]))
                if running != layer.length:
                    raise ValueError(
                        f"pattern {pattern}: episodes up to layer {layer.length} make "
                        f"{running} calls, not {layer.length}"
                    )
        return self

    @property
    def lengths(self) -> list[int]:
        return [layer.length for layer in self.layers]

    @property
    def tail_length(self) -> int:
        return self.layers[0].length

    def tail_episode_ids(self, pattern: PriorCheckPattern = "none") -> list[str]:
        return list(self.layers[0].episodes[pattern])

    def episode_ids(self, length: int, pattern: PriorCheckPattern) -> list[str]:
        """Episode ids for a history of ``length``, oldest first."""
        if length not in self.lengths:
            raise ValueError(f"no layer for length {length}; have {self.lengths}")
        included = [layer for layer in self.layers if layer.length <= length]
        ids: list[str] = []
        for layer in reversed(included):  # longest (oldest) layer first
            ids.extend(layer.episodes[pattern])
        return ids


@cache
def load_plan(name: str = "plan") -> HistoryPlan:
    """``episodes/<name>.yaml``; the default is the plan the first scenarios were built with."""
    raw = yaml.safe_load((EPISODES_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return HistoryPlan.model_validate(raw)
