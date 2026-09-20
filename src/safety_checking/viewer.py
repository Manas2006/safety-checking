"""Render one trajectory as readable markdown.

The prefix is collapsed to a summary by default, because it is identical across every sample
of a cell; the decision segment is shown in full.
"""

from __future__ import annotations

import json
from typing import Any

from .history.builder import Prefix
from .runner.store import RunRecord


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(text)


def _render_messages(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            lines += ["**System**", "", "```text", str(content), "```", ""]
        elif role == "user":
            lines += [f"**User:** {content}", ""]
        elif role == "assistant":
            if content:
                lines += [f"**Assistant:** {content}", ""]
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                lines += [
                    f"**Assistant calls** `{function.get('name')}` (`{call.get('id')}`)",
                    "",
                    "```json",
                    _pretty_json(function.get("arguments", "{}")),
                    "```",
                    "",
                ]
        elif role == "tool":
            lines += [
                f"**Tool result** (`{message.get('tool_call_id')}`)",
                "",
                "```json",
                _pretty_json(content or ""),
                "```",
                "",
            ]
    return lines


def render_record(
    record: RunRecord, prefix: Prefix | None = None, *, full_prefix: bool = False
) -> str:
    trajectory = record.trajectory
    history = f"{trajectory.length} calls, prior_check_pattern=`{trajectory.prior_check_pattern}`"
    lines = [
        f"# Trajectory `{trajectory.run_id[:12]}`",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| scenario | `{trajectory.scenario_id}` ({trajectory.variant}) |",
        f"| history | {history} |",
        f"| model | `{trajectory.model}` |",
        f"| arm | `{trajectory.arm}` |",
        f"| params | `{json.dumps(trajectory.params, sort_keys=True)}` |",
        f"| sample | {trajectory.sample_index} |",
        f"| stop reason | `{trajectory.stop_reason}` |",
        f"| prefix | `{trajectory.prefix_hash[:12]}` |",
        f"| run id | `{trajectory.run_id}` |",
        "",
    ]
    if trajectory.error:
        lines += [f"> **Adapter error:** {trajectory.error}", ""]

    lines += ["## Prefix (frozen, replayed)", ""]
    if prefix is None:
        lines += ["_Prefix file not found; only its hash is known._", ""]
    else:
        meta = prefix.metadata
        estimated = " (estimated)" if meta.token_count_estimated else ""
        lines += [
            f"{meta.n_calls} tool calls in {meta.n_messages} messages, "
            f"{meta.token_count} tokens{estimated}. "
            f"Episodes: {', '.join(meta.episode_ids)}.",
            "",
        ]
        if full_prefix:
            lines += _render_messages(prefix.messages)
        else:
            lines += [
                f"_Collapsed. The last {meta.tail_length} calls (identical at every length):_",
                "",
            ]
            lines += _render_messages(prefix.tail_messages)

    lines += ["## Decision segment (sampled)", ""]
    lines += _render_messages(trajectory.messages)

    if record.score is not None:
        score = record.score
        lines += ["## Score", "", "| measure | value |", "| --- | --- |"]
        for key, value in score.model_dump().items():
            if key == "final_text":
                continue
            lines.append(f"| {key} | `{value}` |")
        lines.append("")
    return "\n".join(lines)
