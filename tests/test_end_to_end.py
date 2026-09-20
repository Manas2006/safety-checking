"""End to end with FakeModel: build, run, score, and check the design's core guarantees.

Prefixes at 5, 20 and 50 for both variants, three samples each, two scripted policies.
Everything is written under a temporary directory; nothing touches ./outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from safety_checking.canonical import sha256_of
from safety_checking.history import check_nesting, list_prefixes
from safety_checking.runner import (
    ExperimentSpec,
    FakeModel,
    build_cells,
    latest_records,
    read_records,
    run_experiment,
)
from safety_checking.scenarios import load_scenario, load_scenario_world
from safety_checking.scoring import score
from safety_checking.world.state import decision_relevant_view

LENGTHS = [5, 20, 50]
SCENARIOS = ["sharing_risky", "sharing_benign"]
N_SAMPLES = 3


@pytest.fixture
def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment="e2e", scenarios=SCENARIOS, lengths=LENGTHS, n_samples=N_SAMPLES
    )


def calls_of(prefix) -> list[dict]:
    return [call for m in prefix.messages for call in m.get("tool_calls") or []]


def test_full_pipeline(spec, tmp_path) -> None:
    prefix_dir = tmp_path / "prefixes"
    log = tmp_path / "runs" / "e2e.jsonl"
    cells = build_cells(spec, prefix_dir)
    assert len(cells) == len(SCENARIOS) * len(LENGTHS)
    assert len(list_prefixes(prefix_dir)) == len(cells)

    # -- run both policies -------------------------------------------------
    for policy in ("always_check", "never_check"):
        summary = run_experiment(spec, FakeModel(policy), log, cells)
        assert (summary.n_new, summary.n_skipped, summary.n_errors) == (18, 0, 0)

    records = latest_records(log)
    assert len(records) == 36
    assert len({r.run_id for r in records}) == 36

    # -- check rates: 1.0 for always_check, 0.0 for never_check, in every cell ---
    by_cell: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    for record in records:
        t = record.trajectory
        by_cell[(t.model, t.scenario_id, t.length)].append(record.score.check_executed)
    assert len(by_cell) == 12
    for (model, _, _), checks in by_cell.items():
        assert len(checks) == N_SAMPLES
        rate = sum(checks) / len(checks)
        assert rate == (1.0 if model == "fake:always_check" else 0.0)

    # -- and the outcomes are the ones the policies should produce ---------
    expected = {
        ("fake:always_check", "sharing_risky"): ("checked_then_declined", False, True),
        ("fake:always_check", "sharing_benign"): ("checked_then_acted", False, True),
        ("fake:never_check", "sharing_risky"): ("acted_without_check", True, False),
        ("fake:never_check", "sharing_benign"): ("acted_without_check", False, True),
    }
    for record in records:
        t, s = record.trajectory, record.score
        outcome, unsafe, completed = expected[(t.model, t.scenario_id)]
        assert (s.outcome, s.unsafe_action, s.completed) == (outcome, unsafe, completed)

    # -- stored scores agree with rescoring from the stored trajectory ------
    for record in records:
        scenario = load_scenario(record.trajectory.scenario_id)
        rescored = score(record.trajectory, scenario, load_scenario_world(scenario))
        assert rescored.model_dump() == record.score.model_dump()

    # -- resume: a second pass pays for nothing ----------------------------
    lines_before = log.read_text().count("\n")
    for policy in ("always_check", "never_check"):
        again = run_experiment(spec, FakeModel(policy), log, cells)
        assert (again.n_new, again.n_skipped) == (0, 18)
    assert log.read_text().count("\n") == lines_before == 36


def test_last_five_calls_are_byte_identical_across_lengths(spec, tmp_path) -> None:
    cells = build_cells(spec, tmp_path / "prefixes")
    for scenario_id in SCENARIOS:
        prefixes = [c.prefix for c in cells if c.scenario.id == scenario_id]
        assert [p.length for p in prefixes] == LENGTHS

        # the five calls themselves
        last_five = [json.dumps(calls_of(p)[-5:], sort_keys=True).encode() for p in prefixes]
        assert last_five[0] == last_five[1] == last_five[2]

        # and every message that carries them, results included
        tails = [json.dumps(p.tail_messages, sort_keys=True).encode() for p in prefixes]
        assert tails[0] == tails[1] == tails[2]
        assert len({p.metadata.tail_hash for p in prefixes}) == 1

        # stronger: each shorter history is a suffix of each longer one
        assert check_nesting(prefixes).suffix_ok

    # the tail does not depend on the variant either
    assert len({c.prefix.metadata.tail_hash for c in cells}) == 1


def test_the_world_the_decision_sees_is_the_same_at_every_length(spec, tmp_path) -> None:
    cells = build_cells(spec, tmp_path / "prefixes")
    views = {sha256_of(decision_relevant_view(c.prefix.world())) for c in cells}
    assert len(views) == 1


def test_rebuilding_gives_the_same_hashes_and_the_same_files(spec, tmp_path) -> None:
    first = build_cells(spec, tmp_path / "a")
    second = build_cells(spec, tmp_path / "b")
    assert [c.prefix.prefix_hash for c in first] == [c.prefix.prefix_hash for c in second]
    assert len({c.prefix.prefix_hash for c in first}) == len(first)

    for cell in first:
        name = f"{cell.prefix.prefix_hash}.json"
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_the_log_is_append_only_jsonl(spec, tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec, tmp_path / "prefixes")
    run_experiment(spec, FakeModel("always_check"), log, cells)
    first = log.read_bytes()
    run_experiment(spec, FakeModel("never_check"), log, cells)
    assert log.read_bytes().startswith(first)

    for line in log.read_text().splitlines():
        json.loads(line)
    assert len(list(read_records(log))) == 36
