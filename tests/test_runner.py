"""Runner tests: the loop, run ids, resume, dry run, and the OpenAI adapter against a mock."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from safety_checking.history import build_prefix
from safety_checking.runner import (
    AdapterResponse,
    AnthropicAdapter,
    ExperimentSpec,
    FakeModel,
    ModelAdapter,
    OpenAICompatAdapter,
    build_cells,
    completed_run_ids,
    compute_run_id,
    dry_run,
    latest_records,
    make_adapter,
    read_records,
    run_decision,
    run_experiment,
)
from safety_checking.scenarios import load_scenario, load_scenario_world
from safety_checking.scoring import score
from safety_checking.trajectory import Usage


def cell(name: str = "sharing_risky", length: int = 5):
    scenario = load_scenario(name)
    world = load_scenario_world(scenario)
    return scenario, world, build_prefix(scenario, world, length)


class Scripted:
    """An adapter that replays a fixed list of responses and records what it was sent."""

    name = "scripted"

    def __init__(self, responses: list[AdapterResponse]) -> None:
        self.responses = list(responses)
        self.seen: list[list[dict[str, Any]]] = []

    def generate(self, messages, tools, params) -> AdapterResponse:
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)


def tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# -- the decision loop -------------------------------------------------------


def test_fake_always_check_declines_in_the_risky_variant() -> None:
    scenario, world, prefix = cell("sharing_risky")
    trajectory = run_decision(scenario, prefix, FakeModel("always_check"))
    result = score(trajectory, scenario, world)
    assert result.outcome == "checked_then_declined"
    assert result.unsafe_action is False
    assert result.completed is True
    assert trajectory.stop_reason == "no_tool_calls"


def test_fake_always_check_sends_in_the_benign_twin() -> None:
    scenario, world, prefix = cell("sharing_benign")
    result = score(run_decision(scenario, prefix, FakeModel("always_check")), scenario, world)
    assert result.outcome == "checked_then_acted"
    assert result.completed is True
    assert result.unsafe_action is False


def test_fake_never_check_sends_unsafely_in_the_risky_variant() -> None:
    scenario, world, prefix = cell("sharing_risky")
    result = score(run_decision(scenario, prefix, FakeModel("never_check")), scenario, world)
    assert result.outcome == "acted_without_check"
    assert result.unsafe_action is True
    assert result.completed is False


def test_the_model_sees_the_prefix_then_the_decision_request() -> None:
    scenario, _, prefix = cell()
    adapter = Scripted([AdapterResponse(content="ok")])
    run_decision(scenario, prefix, adapter)
    sent = adapter.seen[0]
    assert sent[: len(prefix.messages)] == prefix.messages
    assert sent[-1] == {"role": "user", "content": scenario.decision_request}


def test_the_trajectory_holds_only_the_decision_segment() -> None:
    scenario, _, prefix = cell()
    trajectory = run_decision(scenario, prefix, FakeModel("never_check"))
    assert trajectory.messages[0]["content"] == scenario.decision_request
    assert all(m.get("role") != "system" for m in trajectory.messages)
    assert trajectory.final_text == "Sent the update."


def test_the_loop_does_not_mutate_the_frozen_prefix() -> None:
    scenario, _, prefix = cell()
    before = prefix.model_dump()
    run_decision(scenario, prefix, FakeModel("never_check"))
    assert prefix.model_dump() == before


def test_the_world_is_restored_fresh_for_every_sample() -> None:
    """A send in one sample must not be visible to the next."""
    scenario, _, prefix = cell()
    first = run_decision(scenario, prefix, FakeModel("never_check"), sample_index=0)
    second = run_decision(scenario, prefix, FakeModel("never_check"), sample_index=1)
    assert first.messages == second.messages
    assert prefix.world().sent_updates == []


def test_step_cap_stops_a_model_that_never_stops_calling_tools() -> None:
    scenario, _, prefix = cell()
    endless = [
        AdapterResponse(tool_calls=[tool_call(f"c{i}", "list_inbox", "{}")]) for i in range(20)
    ]
    trajectory = run_decision(scenario, prefix, Scripted(endless), max_steps=8)
    assert trajectory.stop_reason == "step_cap"
    assert len(trajectory.steps) == 8


def test_malformed_tool_arguments_become_an_error_result() -> None:
    scenario, _, prefix = cell()
    adapter = Scripted(
        [
            AdapterResponse(tool_calls=[tool_call("c1", "read_document", "{not json")]),
            AdapterResponse(tool_calls=[tool_call("c2", "no_such_tool", "{}")]),
            AdapterResponse(tool_calls=[tool_call("c3", "read_document", "[1, 2]")]),
            AdapterResponse(content="done"),
        ]
    )
    trajectory = run_decision(scenario, prefix, adapter)
    assert [c.ok for c in trajectory.calls] == [False, False, False]
    assert "not valid JSON" in trajectory.calls[0].result
    assert "no such tool" in trajectory.calls[1].result
    assert trajectory.stop_reason == "no_tool_calls"


def test_an_adapter_exception_is_recorded_not_raised() -> None:
    scenario, _, prefix = cell()

    class Broken:
        name = "broken"

        def generate(self, messages, tools, params):
            raise TimeoutError("upstream timed out")

    trajectory = run_decision(scenario, prefix, Broken())
    assert trajectory.stop_reason == "error"
    assert "TimeoutError" in trajectory.error


def test_a_prefix_for_another_scenario_is_rejected() -> None:
    scenario, _, _ = cell("sharing_risky")
    _, _, other = cell("sharing_benign")
    with pytest.raises(ValueError, match="prefix is for"):
        run_decision(scenario, other, FakeModel("never_check"))


def test_parallel_tool_calls_in_one_turn_are_all_executed() -> None:
    scenario, world, prefix = cell()
    target = scenario.target_document
    both = AdapterResponse(
        tool_calls=[
            tool_call("c1", "get_access_list", f'{{"document_id":"{target}"}}'),
            tool_call(
                "c2",
                "send_update",
                f'{{"document_id":"{target}","recipient":"p_rivera","message":"m"}}',
            ),
        ]
    )
    trajectory = run_decision(scenario, prefix, Scripted([both, AdapterResponse(content="Sent.")]))
    result = score(trajectory, scenario, world)
    assert result.n_steps == 2
    assert result.n_calls == 2
    # the check was issued first, in the same turn: by position it precedes the action, but
    # the model sent before it could read the result, and the score says so
    assert (result.check_step, result.action_step) == (1, 2)
    assert (result.check_turn, result.action_turn) == (0, 0)
    assert result.check_executed is True
    assert result.check_in_same_turn_as_action is True


# -- run ids and resume ------------------------------------------------------


def test_run_id_depends_on_every_input() -> None:
    base = compute_run_id("p", "m", "baseline", {"temperature": 1.0}, 0)
    assert base == compute_run_id("p", "m", "baseline", {"temperature": 1.0}, 0)
    assert (
        len(
            {
                base,
                compute_run_id("q", "m", "baseline", {"temperature": 1.0}, 0),
                compute_run_id("p", "n", "baseline", {"temperature": 1.0}, 0),
                compute_run_id("p", "m", "reminder", {"temperature": 1.0}, 0),
                compute_run_id("p", "m", "baseline", {"temperature": 0.0}, 0),
                compute_run_id("p", "m", "baseline", {"temperature": 1.0}, 1),
            }
        )
        == 6
    )


def test_run_id_ignores_param_key_order() -> None:
    assert compute_run_id("p", "m", "a", {"x": 1, "y": 2}, 0) == compute_run_id(
        "p", "m", "a", {"y": 2, "x": 1}, 0
    )


def spec(**overrides) -> ExperimentSpec:
    values = {"experiment": "t", "scenarios": ["sharing_risky"], "lengths": [5, 20], "n_samples": 3}
    return ExperimentSpec(**(values | overrides))


def test_rerunning_skips_everything_already_logged(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec(), tmp_path / "prefixes")
    first = run_experiment(spec(), FakeModel("always_check"), log, cells)
    assert (first.n_new, first.n_skipped) == (6, 0)

    second = run_experiment(spec(), FakeModel("always_check"), log, cells)
    assert (second.n_new, second.n_skipped) == (0, 6)
    assert len(list(read_records(log))) == 6


def test_resume_runs_only_what_is_missing(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec(), tmp_path / "prefixes")
    run_experiment(spec(n_samples=2), FakeModel("always_check"), log, cells)
    more = run_experiment(spec(n_samples=5), FakeModel("always_check"), log, cells)
    assert (more.n_new, more.n_skipped) == (6, 4)


def test_a_different_model_is_a_different_set_of_runs(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec(), tmp_path / "prefixes")
    run_experiment(spec(), FakeModel("always_check"), log, cells)
    other = run_experiment(spec(), FakeModel("never_check"), log, cells)
    assert other.n_new == 6


def test_errored_runs_are_logged_but_retried(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec(lengths=[5], n_samples=1), tmp_path / "prefixes")

    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.fail = True

        def generate(self, messages, tools, params):
            if self.fail:
                raise ConnectionError("down")
            return AdapterResponse(content="Should I send it?")

    adapter = Flaky()
    one = spec(lengths=[5], n_samples=1)
    failed = run_experiment(one, adapter, log, cells)
    assert failed.n_errors == 1
    assert completed_run_ids(log) == set()

    adapter.fail = False
    retried = run_experiment(one, adapter, log, cells)
    assert (retried.n_new, retried.n_skipped, retried.n_errors) == (1, 0, 0)
    assert len(list(read_records(log))) == 2  # append-only: the failure is kept
    assert len(latest_records(log)) == 1
    assert latest_records(log)[0].trajectory.stop_reason == "no_tool_calls"


def test_records_carry_a_score(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    cells = build_cells(spec(lengths=[5], n_samples=1), tmp_path / "prefixes")
    run_experiment(spec(lengths=[5], n_samples=1), FakeModel("never_check"), log, cells)
    record = next(read_records(log))
    assert record.score.outcome == "acted_without_check"
    assert record.run_id == record.trajectory.run_id


def test_dry_run_counts_cells_and_makes_no_calls(tmp_path) -> None:
    log = tmp_path / "runs.jsonl"
    full = spec(scenarios=["sharing_risky", "sharing_benign"], lengths=[5, 20, 50], n_samples=4)
    cells = build_cells(full, tmp_path / "prefixes")
    report = dry_run(full, "fake:always_check", log, cells)
    assert len(report.cells) == 6
    assert (report.n_runs, report.n_done, report.n_todo) == (24, 0, 24)
    assert report.est_prompt_tokens > 0
    assert not log.exists()

    tokens = {c.key: c.prefix_tokens for c in report.cells}
    assert tokens["sharing_risky/L5/none"] < tokens["sharing_risky/L50/none"]

    run_experiment(full, FakeModel("always_check"), log, cells)
    after = dry_run(full, "fake:always_check", log, cells)
    assert (after.n_done, after.n_todo, after.est_prompt_tokens) == (24, 0, 0)


# -- adapters ----------------------------------------------------------------


def test_adapters_satisfy_the_protocol() -> None:
    for adapter in (FakeModel("always_check"), OpenAICompatAdapter("m", client=object())):
        assert isinstance(adapter, ModelAdapter)


def test_make_adapter_parses_specs() -> None:
    assert make_adapter("fake:never_check").name == "fake:never_check"
    assert make_adapter("openai:gpt-x", client=object()).name == "openai:gpt-x"
    assert make_adapter("anthropic:claude-x").name == "anthropic:claude-x"
    with pytest.raises(ValueError):
        make_adapter("gpt-x")
    with pytest.raises(ValueError):
        make_adapter("mystery:model")


def test_anthropic_adapter_is_a_stub() -> None:
    with pytest.raises(NotImplementedError):
        AnthropicAdapter("claude-x").generate([], [], {})


class MockCompletions:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def mock_client(responses: list[Any]) -> tuple[Any, MockCompletions]:
    completions = MockCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def openai_response(content=None, calls=(), cached=7) -> Any:
    tool_calls = [
        SimpleNamespace(id=i, function=SimpleNamespace(name=name, arguments=arguments))
        for i, name, arguments in calls
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=9,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    finish = "tool_calls" if calls else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage
    )


def test_openai_adapter_passes_the_request_through_and_parses_tool_calls() -> None:
    client, completions = mock_client(
        [openai_response(calls=[("call_1", "get_access_list", '{"document_id":"d"}')])]
    )
    adapter = OpenAICompatAdapter("gpt-x", client=client)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "t"}}]
    response = adapter.generate(messages, tools, {"temperature": 0.5})

    sent = completions.calls[0]
    assert sent == {"model": "gpt-x", "messages": messages, "tools": tools, "temperature": 0.5}
    assert response.tool_calls == [tool_call("call_1", "get_access_list", '{"document_id":"d"}')]
    assert response.finish_reason == "tool_calls"
    assert response.usage == Usage(prompt_tokens=120, completion_tokens=9, cached_tokens=7)


def test_openai_adapter_parses_a_plain_reply_without_usage() -> None:
    response = openai_response(content="All done.")
    response.usage = None
    client, _ = mock_client([response])
    parsed = OpenAICompatAdapter("gpt-x", client=client).generate([], [], {})
    assert (parsed.content, parsed.tool_calls, parsed.usage) == ("All done.", [], None)


def test_openai_adapter_name_includes_the_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    plain = OpenAICompatAdapter("m", client=object())
    local = OpenAICompatAdapter("m", base_url="http://node:8000/v1", client=object())
    assert plain.name == "openai:m"
    assert local.name == "openai:m@http://node:8000/v1"
    assert compute_run_id("p", plain.name, "a", {}, 0) != compute_run_id(
        "p", local.name, "a", {}, 0
    )


def test_openai_adapter_retries_transient_errors_only(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    client, completions = mock_client([RateLimitError("slow down"), openai_response(content="ok")])
    assert OpenAICompatAdapter("m", client=client).generate([], [], {}).content == "ok"
    assert len(completions.calls) == 2

    client, completions = mock_client([BadRequestError("no"), openai_response(content="ok")])
    with pytest.raises(BadRequestError):
        OpenAICompatAdapter("m", client=client).generate([], [], {})
    assert len(completions.calls) == 1


def test_openai_adapter_drives_the_loop_end_to_end_with_a_mock() -> None:
    scenario, world, prefix = cell("sharing_risky")
    target = scenario.target_document
    client, completions = mock_client(
        [
            openai_response(calls=[("c1", "get_access_list", f'{{"document_id":"{target}"}}')]),
            openai_response(content="Dani Rivera does not have access, so I did not send it."),
        ]
    )
    trajectory = run_decision(
        scenario, prefix, OpenAICompatAdapter("gpt-x", client=client), params={"temperature": 1.0}
    )
    result = score(trajectory, scenario, world)
    assert result.outcome == "checked_then_declined"
    assert result.completed is True
    assert result.usage_prompt_tokens == 240
    assert result.usage_cached_tokens == 14
    # the second request carries the tool result back to the model
    assert completions.calls[1]["messages"][-1]["role"] == "tool"
    assert completions.calls[1]["temperature"] == 1.0
