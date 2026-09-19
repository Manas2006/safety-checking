"""The scripted benign episode library.

An episode is one small piece of ordinary work: a user request, some tool calls, and a short
assistant summary. Episodes are the unit the history plan assembles.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..paths import EPISODES_DIR

LIBRARY_FILE = "library.yaml"


class EpisodeCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_request: str
    calls: list[EpisodeCall]
    summary: str
    tags: list[str] = Field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


@cache
def load_episodes() -> dict[str, Episode]:
    """Load the episode library, keyed by id."""
    raw = yaml.safe_load((EPISODES_DIR / LIBRARY_FILE).read_text(encoding="utf-8"))
    episodes = [Episode.model_validate(item) for item in raw["episodes"]]
    library: dict[str, Episode] = {}
    for episode in episodes:
        if episode.id in library:
            raise ValueError(f"duplicate episode id: {episode.id}")
        library[episode.id] = episode
    return library


def get_episodes(ids: list[str]) -> list[Episode]:
    library = load_episodes()
    missing = [i for i in ids if i not in library]
    if missing:
        raise KeyError(f"unknown episode ids: {', '.join(missing)}")
    return [library[i] for i in ids]
