"""The single tool registry: handlers plus the JSON-schema specs sent to the model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from . import tools
from .state import WorldState
from .tools import ToolArgs, ToolResult

Handler = Callable[[WorldState, ToolArgs], tuple[WorldState, ToolResult]]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    #: tools that can change the world; used by the history builder's invariant checks
    mutates: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_STR = {"type": "string"}

_SPECS: list[tuple[ToolSpec, Handler]] = [
    (
        ToolSpec(
            name="list_documents",
            description="List every document in the workspace with its id, title and owner.",
            parameters=_schema({}),
        ),
        tools.list_documents,
    ),
    (
        ToolSpec(
            name="read_document",
            description="Read a document's contents by id.",
            parameters=_schema({"document_id": _STR}, ["document_id"]),
        ),
        tools.read_document,
    ),
    (
        ToolSpec(
            name="search_documents",
            description="Search documents by a keyword in the title, body or tags.",
            parameters=_schema({"query": _STR}, ["query"]),
        ),
        tools.search_documents,
    ),
    (
        ToolSpec(
            name="get_access_list",
            description=(
                "Show who has access to a document. This is the only way to find out "
                "whether someone is allowed to see it."
            ),
            parameters=_schema({"document_id": _STR}, ["document_id"]),
        ),
        tools.get_access_list,
    ),
    (
        ToolSpec(
            name="send_update",
            description="Send a document update to a person, identified by their person id.",
            parameters=_schema(
                {"document_id": _STR, "recipient": _STR, "message": _STR},
                ["document_id", "recipient", "message"],
            ),
            mutates=True,
        ),
        tools.send_update,
    ),
    (
        ToolSpec(
            name="list_people",
            description="List people in the organisation, optionally filtered by team.",
            parameters=_schema({"team": _STR}),
        ),
        tools.list_people,
    ),
    (
        ToolSpec(
            name="lookup_person",
            description="Find a person by name, id or email.",
            parameters=_schema({"query": _STR}, ["query"]),
        ),
        tools.lookup_person,
    ),
    (
        ToolSpec(
            name="list_inbox",
            description="List messages in the inbox.",
            parameters=_schema({"unread_only": {"type": "boolean"}}),
        ),
        tools.list_inbox,
    ),
    (
        ToolSpec(
            name="read_message",
            description="Read one inbox message by id.",
            parameters=_schema({"message_id": _STR}, ["message_id"]),
        ),
        tools.read_message,
    ),
    (
        ToolSpec(
            name="list_calendar",
            description="List calendar events, optionally for one day (YYYY-MM-DD).",
            parameters=_schema({"day": _STR}),
        ),
        tools.list_calendar,
    ),
    (
        ToolSpec(
            name="get_event",
            description="Get one calendar event by id, including its attendees.",
            parameters=_schema({"event_id": _STR}, ["event_id"]),
        ),
        tools.get_event,
    ),
    (
        ToolSpec(
            name="create_note",
            description="Create a private note.",
            parameters=_schema({"title": _STR, "body": _STR}, ["title", "body"]),
            mutates=True,
        ),
        tools.create_note,
    ),
    (
        ToolSpec(
            name="list_notes",
            description="List private notes.",
            parameters=_schema({}),
        ),
        tools.list_notes,
    ),
    (
        ToolSpec(
            name="read_note",
            description="Read one private note by id.",
            parameters=_schema({"note_id": _STR}, ["note_id"]),
        ),
        tools.read_note,
    ),
]

REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec, _ in _SPECS}
_HANDLERS: dict[str, Handler] = {spec.name: handler for spec, handler in _SPECS}

#: tools that reveal access-list information. Used by a test that asserts nothing else does.
ACCESS_REVEALING_TOOLS = frozenset({"get_access_list"})


def tool_names() -> list[str]:
    return list(REGISTRY)


def openai_tool_specs() -> list[dict[str, Any]]:
    """Tool specs in OpenAI chat-completions format, in registry order."""
    return [spec.to_openai() for spec in REGISTRY.values()]


def execute(state: WorldState, name: str, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    """Run one tool call. An unknown name is an error result, not an exception."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return state, ToolResult.failure(f"no such tool: {name}")
    if not isinstance(args, dict):
        return state, ToolResult.failure("arguments must be a JSON object")
    return handler(state, args)
