"""Run ids and the append-only JSONL log.

A run id is a content hash of everything that determines a sample, so the same cell is never
paid for twice. The log is only ever appended to; a run that ended in an adapter error is kept
for the record but is not counted as done, so it is retried.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..canonical import sha256_of
from ..paths import runs_dir
from ..scoring import Score
from ..trajectory import Trajectory

#: 2: added the tool_call_parse_failure outcome, parse_failures and truncated
SCORER_VERSION = 2


def compute_run_id(
    prefix_hash: str, model: str, arm: str, params: dict[str, Any], sample_index: int
) -> str:
    return sha256_of(
        {
            "prefix_hash": prefix_hash,
            "model": model,
            "arm": arm,
            "params": params,
            "sample_index": sample_index,
        }
    )


class RunRecord(BaseModel):
    """One line of the log."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    trajectory: Trajectory
    #: a convenience copy; scoring is pure, so the analysis can always rescore from the trajectory
    score: Score | None = None
    scorer_version: int = SCORER_VERSION


def run_log_path(experiment: str, directory: Path | None = None) -> Path:
    return (directory or runs_dir()) / f"{experiment}.jsonl"


def append_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_records(path: Path) -> Iterator[RunRecord]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RunRecord.model_validate(json.loads(line))


def completed_run_ids(path: Path) -> set[str]:
    """Run ids that finished without an adapter error. These are skipped on resume."""
    return {r.run_id for r in read_records(path) if r.trajectory.stop_reason != "error"}


def latest_records(path: Path) -> list[RunRecord]:
    """One record per run id: the last successful one, else the last one seen."""
    chosen: dict[str, RunRecord] = {}
    for record in read_records(path):
        current = chosen.get(record.run_id)
        if current is None or record.trajectory.stop_reason != "error":
            chosen[record.run_id] = record
    return list(chosen.values())
