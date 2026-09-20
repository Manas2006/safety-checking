"""Spotting tool calls the server failed to parse.

With an open-weights model the tool call is text that the serving stack has to parse back into
structure. When that goes wrong the model *did* try to call a tool, but the response carries no
``tool_calls``, and a naive scorer would count the run as "no check, no action". That would be
a parsing artefact scored as a safety failure, so it gets its own outcome instead.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: markup that tool-calling chat templates use. If it shows up in plain content, a call was
#: emitted and not parsed. Covers the Qwen XML and Hermes JSON styles and a few common others.
_TOOL_CALL_MARKUP = re.compile(
    r"</?tool_call>|</?tool_calls>|<function=|</function>|<parameter=|</parameter>"
    r"|<\|tool_call|\[TOOL_CALLS\]|<\|python_tag\|>|\"arguments\"\s*:\s*[{\"]",
    re.IGNORECASE,
)

UNPARSED_TEXT = "unparsed_tool_call_text"
LEFTOVER_TEXT = "leftover_tool_call_text"
INVALID_ARGUMENTS = "invalid_arguments_json"
MISSING_NAME = "missing_tool_name"


def looks_like_tool_call(text: str | None) -> bool:
    return bool(text) and _TOOL_CALL_MARKUP.search(text) is not None


def detect_parse_failure(content: str | None, tool_calls: list[dict[str, Any]]) -> str | None:
    """A short reason string, or None when the turn parsed cleanly."""
    if not tool_calls:
        return UNPARSED_TEXT if looks_like_tool_call(content) else None

    for call in tool_calls:
        function = call.get("function") or {}
        if not function.get("name"):
            return MISSING_NAME
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            continue
        try:
            parsed = json.loads(arguments) if arguments else {}
        except (TypeError, ValueError):
            return INVALID_ARGUMENTS
        if not isinstance(parsed, dict):
            return INVALID_ARGUMENTS

    # some calls parsed, but more tool-call markup was left behind in the text
    return LEFTOVER_TEXT if looks_like_tool_call(content) else None
