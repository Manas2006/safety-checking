"""Where things live.

On LS6 ``outputs`` is a symlink to ``$WORK/safety-checking-outputs`` ($HOME holds code only);
on a local copy it is a plain git-ignored folder.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
WORLDS_DIR = DATA_DIR / "worlds"
SCENARIOS_DIR = DATA_DIR / "scenarios"
EPISODES_DIR = DATA_DIR / "episodes"

REPO_ROOT = PACKAGE_DIR.parents[1]


def outputs_dir() -> Path:
    """Root for generated artefacts. ``SC_OUTPUTS_DIR`` overrides it (tests use tmp dirs)."""
    override = os.environ.get("SC_OUTPUTS_DIR")
    return Path(override) if override else REPO_ROOT / "outputs"


def prefixes_dir() -> Path:
    return outputs_dir() / "prefixes"


def runs_dir() -> Path:
    return outputs_dir() / "runs"


def tiktoken_cache_dir() -> Path:
    return outputs_dir() / "tiktoken_cache"
