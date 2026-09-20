"""The decision loop: replay a frozen prefix, then let the model act for up to N steps."""

from __future__ import annotations

import json
from typing import Any

from ..history.builder import Prefix
from ..scenarios.schema import Scenario
from ..trajectory import Step, ToolCallRecord, Trajectory
from ..world.registry import execute, openai_tool_specs
from ..world.tools import ToolResult
from .adapters import ModelAdapter
from .store import compute_run_id

MAX_STEPS = 8


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Tool arguments arrive as a JSON string. Malformed input is an error result, not a crash."""
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}, "arguments are not valid JSON"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def run_decision(
    scenario: Scenario,
    prefix: Prefix,
    adapter: ModelAdapter,
    *,
    arm: str = "baseline",
    params: dict[str, Any] | None = None,
    sample_index: int = 0,
    max_steps: int = MAX_STEPS,
) -> Trajectory:
    """Sample one decision segment.

    The world is restored from the prefix snapshot, the decision request is appended, and the
    model acts until it returns a turn with no tool calls or runs out of steps. An adapter
    failure is recorded as ``stop_reason="error"``; the store does not treat such a run as
    done, so it is retried on the next invocation.
    """
    params = dict(params or {})
    if prefix.scenario_id != scenario.id:
        raise ValueError(f"prefix is for {prefix.scenario_id}, not {scenario.id}")
    if hasattr(adapter, "bind"):
        adapter.bind(scenario)

    world = prefix.world()
    tools = openai_tool_specs()
    request = {"role": "user", "content": scenario.decision_request}
    context = [*prefix.messages, request]
    decision: list[dict[str, Any]] = [request]

    trajectory = Trajectory(
        run_id=compute_run_id(prefix.prefix_hash, adapter.name, arm, params, sample_index),
        prefix_hash=prefix.prefix_hash,
        scenario_id=scenario.id,
        variant=scenario.variant,
        length=prefix.length,
        prior_check_pattern=prefix.prior_check_pattern,
        model=adapter.name,
        arm=arm,
        params=params,
        sample_index=sample_index,
        messages=decision,
        stop_reason="step_cap",
    )

    for index in range(max_steps):
        try:
            # a copy, so an adapter can neither see later turns nor mutate the loop's context
            response = adapter.generate(list(context), tools, dict(params))
        except Exception as exc:
            trajectory.stop_reason = "error"
            trajectory.error = f"{type(exc).__name__}: {exc}"
            return trajectory

        assistant: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant["tool_calls"] = response.tool_calls
        context.append(assistant)
        decision.append(assistant)

        step = Step(
            index=index,
            content=response.content,
            usage=response.usage,
            finish_reason=response.finish_reason,
        )
        trajectory.steps.append(step)

        if not response.tool_calls:
            trajectory.stop_reason = "no_tool_calls"
            trajectory.final_text = response.content
            return trajectory

        for call in response.tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            args, problem = _parse_arguments(function.get("arguments"))
            if problem is None:
                world, result = execute(world, name, args)
            else:
                result = ToolResult.failure(problem)
            result_json = result.to_json()
            tool_message = {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result_json,
            }
            context.append(tool_message)
            decision.append(tool_message)
            step.tool_calls.append(
                ToolCallRecord(id=call.get("id", ""), tool=name, args=args, result=result_json)
            )

    # ran out of steps while still calling tools: keep whatever the last turn said
    trajectory.final_text = trajectory.steps[-1].content if trajectory.steps else None
    return trajectory
