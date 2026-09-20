"""Cells, dry runs and resumable execution.

A cell is (scenario, length, prior_check_pattern). A run is one sample of one cell.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..history.builder import Prefix, build_prefix
from ..history.plan import PRIOR_CHECK_PATTERNS, load_plan
from ..history.store import save_prefix
from ..scenarios.loader import load_scenario, load_scenario_world
from ..scenarios.schema import Scenario
from ..scoring import score
from ..trajectory import Trajectory
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
    #: samples of one prefix in flight at once. The first sample of a cell always goes alone,
    #: so the rest arrive at a warm prefix cache.
    concurrency: int = 1
    #: per-request seed is base_seed + sample_index. None sends no seed.
    base_seed: int | None = None


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
    #: wall-clock seconds spent in this invocation, and provider-reported tokens for new runs
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    @property
    def completion_tokens_per_s(self) -> float | None:
        return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else None


def _batches(todo: list[int], concurrency: int) -> list[list[int]]:
    """The first sample alone, to warm the prefix cache, then the rest ``concurrency`` wide."""
    if not todo:
        return []
    if concurrency <= 1:
        return [[i] for i in todo]
    rest = todo[1:]
    return [[todo[0]]] + [rest[i : i + concurrency] for i in range(0, len(rest), concurrency)]


def run_experiment(
    spec: ExperimentSpec, adapter: ModelAdapter, log_path: Path, cells: list[Cell]
) -> RunSummary:
    """Run every sample that is not already in the log. Safe to interrupt and re-run.

    Samples of one cell share a prefix, so they are sent together (``spec.concurrency`` wide)
    to share the server's prefix cache. Records are appended from this thread only, in sample
    order within a batch, so the log stays well formed.
    """
    done = completed_run_ids(log_path)
    summary = RunSummary()
    started = time.perf_counter()

    for cell in cells:
        world = load_scenario_world(cell.scenario)
        if hasattr(adapter, "bind"):
            adapter.bind(cell.scenario)  # once, before any thread starts

        def run_id_of(sample_index: int, cell: Cell = cell) -> str:
            return compute_run_id(
                cell.prefix.prefix_hash, adapter.name, spec.arm, spec.params, sample_index
            )

        def sample(sample_index: int, cell: Cell = cell) -> Trajectory:
            seed = None if spec.base_seed is None else spec.base_seed + sample_index
            return run_decision(
                cell.scenario,
                cell.prefix,
                adapter,
                arm=spec.arm,
                params=spec.params,
                sample_index=sample_index,
                max_steps=spec.max_steps,
                seed=seed,
            )

        todo = [i for i in range(spec.n_samples) if run_id_of(i) not in done]
        summary.n_skipped += spec.n_samples - len(todo)

        for batch in _batches(todo, spec.concurrency):
            if len(batch) == 1:
                trajectories = [sample(batch[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    trajectories = list(pool.map(sample, batch))
            for trajectory in trajectories:
                scored = None
                if trajectory.stop_reason == "error":
                    summary.n_errors += 1
                else:
                    scored = score(trajectory, cell.scenario, world)
                    done.add(trajectory.run_id)
                    summary.prompt_tokens += scored.usage_prompt_tokens or 0
                    summary.cached_tokens += scored.usage_cached_tokens or 0
                    summary.completion_tokens += scored.usage_completion_tokens or 0
                append_record(
                    log_path,
                    RunRecord(run_id=trajectory.run_id, trajectory=trajectory, score=scored),
                )
                summary.n_new += 1

    summary.elapsed_s = time.perf_counter() - started
    return summary
