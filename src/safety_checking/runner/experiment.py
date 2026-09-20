"""Cells, dry runs and resumable execution.

A cell is (scenario, length, prior_check_pattern). A run is one sample of one cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..history.builder import Prefix, build_prefix
from ..history.plan import PRIOR_CHECK_PATTERNS, load_plan
from ..history.store import save_prefix
from ..scenarios.loader import load_scenario, load_scenario_world
from ..scenarios.schema import Scenario
from ..scoring import score
from .adapters import ModelAdapter
from .loop import MAX_STEPS, run_decision
from .store import RunRecord, append_record, completed_run_ids, compute_run_id

#: rough planning numbers for --dry-run. A decision is usually check, act, reply.
EST_STEPS_PER_RUN = 3
EST_COMPLETION_TOKENS_PER_STEP = 60
EST_TOOL_SPEC_TOKENS = 700


class Cell(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: Scenario
    prefix: Prefix

    @property
    def key(self) -> str:
        p = self.prefix
        return f"{self.scenario.id}/L{p.length}/{p.prior_check_pattern}"


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    scenarios: list[str]
    lengths: list[int] = Field(default_factory=lambda: list(load_plan().lengths))
    patterns: list[str] = Field(default_factory=lambda: ["none"])
    n_samples: int = 3
    arm: str = "baseline"
    params: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = MAX_STEPS


def build_cells(
    spec: ExperimentSpec, prefix_dir: Path | None = None, *, save: bool = True
) -> list[Cell]:
    """Build (and by default save) the frozen prefix for every cell."""
    for pattern in spec.patterns:
        if pattern not in PRIOR_CHECK_PATTERNS:
            raise ValueError(f"unknown prior_check_pattern: {pattern}")
    cells = []
    for name in spec.scenarios:
        scenario = load_scenario(name)
        world = load_scenario_world(scenario)
        for length in spec.lengths:
            for pattern in spec.patterns:
                prefix = build_prefix(scenario, world, length, prior_check_pattern=pattern)
                if save:
                    save_prefix(prefix, prefix_dir)
                cells.append(Cell(scenario=scenario, prefix=prefix))
    return cells


class CellPlan(BaseModel):
    key: str
    prefix_hash: str
    prefix_tokens: int
    n_samples: int
    n_done: int
    est_prompt_tokens: int
    est_completion_tokens: int


class DryRunReport(BaseModel):
    experiment: str
    model: str
    arm: str
    cells: list[CellPlan]

    @property
    def n_runs(self) -> int:
        return sum(c.n_samples for c in self.cells)

    @property
    def n_done(self) -> int:
        return sum(c.n_done for c in self.cells)

    @property
    def n_todo(self) -> int:
        return self.n_runs - self.n_done

    @property
    def est_prompt_tokens(self) -> int:
        return sum(c.est_prompt_tokens for c in self.cells)

    @property
    def est_completion_tokens(self) -> int:
        return sum(c.est_completion_tokens for c in self.cells)


def _run_ids(cell: Cell, model: str, spec: ExperimentSpec) -> list[str]:
    return [
        compute_run_id(cell.prefix.prefix_hash, model, spec.arm, spec.params, i)
        for i in range(spec.n_samples)
    ]


def dry_run(
    spec: ExperimentSpec, model_name: str, log_path: Path, cells: list[Cell]
) -> DryRunReport:
    """Count what would run and roughly what it would cost. Makes no model calls.

    Every step resends the whole context, so prompt tokens are about
    (prefix + tool specs) x steps per remaining run. Prompt caching would cut the billed
    amount a lot; this is the uncached upper bound.
    """
    done = completed_run_ids(log_path)
    plans = []
    for cell in cells:
        ids = _run_ids(cell, model_name, spec)
        n_done = sum(1 for run_id in ids if run_id in done)
        todo = len(ids) - n_done
        per_step = cell.prefix.metadata.token_count + EST_TOOL_SPEC_TOKENS
        plans.append(
            CellPlan(
                key=cell.key,
                prefix_hash=cell.prefix.prefix_hash,
                prefix_tokens=cell.prefix.metadata.token_count,
                n_samples=len(ids),
                n_done=n_done,
                est_prompt_tokens=todo * EST_STEPS_PER_RUN * per_step,
                est_completion_tokens=todo * EST_STEPS_PER_RUN * EST_COMPLETION_TOKENS_PER_STEP,
            )
        )
    return DryRunReport(experiment=spec.experiment, model=model_name, arm=spec.arm, cells=plans)


class RunSummary(BaseModel):
    n_new: int = 0
    n_skipped: int = 0
    n_errors: int = 0


def run_experiment(
    spec: ExperimentSpec, adapter: ModelAdapter, log_path: Path, cells: list[Cell]
) -> RunSummary:
    """Run every sample that is not already in the log. Safe to interrupt and re-run."""
    done = completed_run_ids(log_path)
    summary = RunSummary()
    for cell in cells:
        world = load_scenario_world(cell.scenario)
        for sample_index in range(spec.n_samples):
            run_id = compute_run_id(
                cell.prefix.prefix_hash, adapter.name, spec.arm, spec.params, sample_index
            )
            if run_id in done:
                summary.n_skipped += 1
                continue
            trajectory = run_decision(
                cell.scenario,
                cell.prefix,
                adapter,
                arm=spec.arm,
                params=spec.params,
                sample_index=sample_index,
                max_steps=spec.max_steps,
            )
            scored = None
            if trajectory.stop_reason == "error":
                summary.n_errors += 1
            else:
                scored = score(trajectory, cell.scenario, world)
                done.add(run_id)
            append_record(log_path, RunRecord(run_id=run_id, trajectory=trajectory, score=scored))
            summary.n_new += 1
    return summary
