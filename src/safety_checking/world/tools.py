"""Tools as pure functions ``(state, args) -> (state, ToolResult)``.

Bad input returns ``ToolResult(ok=False, error=...)``; nothing here raises. Tools that do
not change the world return the state object they were given.

Two invariants the experiment depends on (SPEC.md 3.1):

* ``send_update`` succeeds whether or not the recipient is on the access list. The
  environment never enforces the policy; only the agent can.
* ``get_access_list`` is the only tool that reveals access information.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..canonical import canonical_json
from .state import Note, SentUpdate, WorldState

ToolArgs = dict[str, Any]


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None

    def to_json(self) -> str:
        """Compact JSON with sorted keys: what the model sees as the tool message."""
        return canonical_json(self.model_dump(exclude_none=True))

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, error=error)

    @classmethod
    def success(cls, **data: Any) -> ToolResult:
        return cls(ok=True, data=data)


# -- argument helpers --------------------------------------------------------


class _ArgError(Exception):
    """Internal: unwound into a ToolResult by every public tool function."""


def _required_str(args: ToolArgs, key: str) -> str:
    if key not in args:
        raise _ArgError(f"missing required argument: {key}")
    value = args[key]
    if not isinstance(value, str) or not value.strip():
        raise _ArgError(f"argument {key} must be a non-empty string")
    return value


def _optional_str(args: ToolArgs, key: str) -> str | None:
    if key not in args or args[key] is None:
        return None
    value = args[key]
    if not isinstance(value, str):
        raise _ArgError(f"argument {key} must be a string")
    return value


def _optional_bool(args: ToolArgs, key: str, default: bool) -> bool:
    if key not in args or args[key] is None:
        return default
    value = args[key]
    if not isinstance(value, bool):
        raise _ArgError(f"argument {key} must be a boolean")
    return value


def _unknown_args(args: ToolArgs, allowed: set[str]) -> None:
    extra = sorted(set(args) - allowed)
    if extra:
        raise _ArgError(f"unknown arguments: {', '.join(extra)}")


# -- documents ---------------------------------------------------------------


def list_documents(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, set())
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    documents = [
        {"id": d.id, "title": d.title, "owner": d.owner, "tags": d.tags} for d in state.documents
    ]
    return state, ToolResult.success(documents=documents)


def read_document(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"document_id"})
        document_id = _required_str(args, "document_id")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    document = state.document(document_id)
    if document is None:
        return state, ToolResult.failure(f"no such document: {document_id}")
    return state, ToolResult.success(
        id=document.id,
        title=document.title,
        owner=document.owner,
        tags=document.tags,
        protected=document.protected,
        body=document.body,
    )


def search_documents(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"query"})
        query = _required_str(args, "query").lower()
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    matches = [
        {"id": d.id, "title": d.title}
        for d in state.documents
        if query in d.title.lower()
        or query in d.body.lower()
        or any(query in tag.lower() for tag in d.tags)
    ]
    return state, ToolResult.success(query=query, matches=matches)


def get_access_list(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"document_id"})
        document_id = _required_str(args, "document_id")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    document = state.document(document_id)
    if document is None:
        return state, ToolResult.failure(f"no such document: {document_id}")
    access = []
    for person_id in document.access:
        person = state.person(person_id)
        entry = {"person_id": person_id}
        if person is not None:
            entry |= {"name": person.name, "email": person.email}
        access.append(entry)
    return state, ToolResult.success(document_id=document.id, access=access)


def send_update(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    """Send a document update to a person.

    Succeeds regardless of the access list: the environment does not enforce the standing
    rule, so an unsafe send really happens and can be scored.
    """
    try:
        _unknown_args(args, {"document_id", "recipient", "message"})
        document_id = _required_str(args, "document_id")
        recipient = _required_str(args, "recipient")
        message = _required_str(args, "message")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    document = state.document(document_id)
    if document is None:
        return state, ToolResult.failure(f"no such document: {document_id}")
    person = state.person(recipient)
    if person is None:
        return state, ToolResult.failure(f"no such person: {recipient}")
    new_state = state.copy_state()
    update = SentUpdate(
        id=f"upd_{len(new_state.sent_updates) + 1:03d}",
        document_id=document_id,
        recipient=recipient,
        message=message,
    )
    new_state.sent_updates.append(update)
    return new_state, ToolResult.success(
        sent=True,
        update_id=update.id,
        document_id=document_id,
        recipient=recipient,
        recipient_email=person.email,
    )


# -- people ------------------------------------------------------------------


def list_people(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"team"})
        team = _optional_str(args, "team")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    people = [
        {"id": p.id, "name": p.name, "email": p.email, "team": p.team, "role": p.role}
        for p in state.people
        if team is None or p.team.lower() == team.lower()
    ]
    return state, ToolResult.success(people=people)


def lookup_person(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"query"})
        query = _required_str(args, "query").lower()
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    matches = [
        {"id": p.id, "name": p.name, "email": p.email, "team": p.team, "role": p.role}
        for p in state.people
        if query in p.id.lower() or query in p.name.lower() or query in p.email.lower()
    ]
    return state, ToolResult.success(query=query, matches=matches)


# -- inbox -------------------------------------------------------------------


def list_inbox(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"unread_only"})
        unread_only = _optional_bool(args, "unread_only", False)
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    messages = [
        {"id": m.id, "sender": m.sender, "subject": m.subject, "unread": m.unread}
        for m in state.inbox
        if not unread_only or m.unread
    ]
    return state, ToolResult.success(messages=messages)


def read_message(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    """Read a message. Does not mark it read: no tool mutates what a later call can see."""
    try:
        _unknown_args(args, {"message_id"})
        message_id = _required_str(args, "message_id")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    message = state.message(message_id)
    if message is None:
        return state, ToolResult.failure(f"no such message: {message_id}")
    return state, ToolResult.success(
        id=message.id, sender=message.sender, subject=message.subject, body=message.body
    )


# -- calendar ----------------------------------------------------------------


def list_calendar(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"day"})
        day = _optional_str(args, "day")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    events = [
        {"id": e.id, "title": e.title, "day": e.day, "start": e.start, "end": e.end}
        for e in state.calendar
        if day is None or e.day == day
    ]
    return state, ToolResult.success(events=events)


def get_event(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"event_id"})
        event_id = _required_str(args, "event_id")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    event = state.event(event_id)
    if event is None:
        return state, ToolResult.failure(f"no such event: {event_id}")
    return state, ToolResult.success(**event.model_dump(mode="json"))


# -- notes -------------------------------------------------------------------


def create_note(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"title", "body"})
        title = _required_str(args, "title")
        body = _required_str(args, "body")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    new_state = state.copy_state()
    note = Note(id=f"note_{len(new_state.notes) + 1:03d}", title=title, body=body)
    new_state.notes.append(note)
    return new_state, ToolResult.success(created=True, note_id=note.id, title=title)


def list_notes(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, set())
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    notes = [{"id": n.id, "title": n.title} for n in state.notes]
    return state, ToolResult.success(notes=notes)


def read_note(state: WorldState, args: ToolArgs) -> tuple[WorldState, ToolResult]:
    try:
        _unknown_args(args, {"note_id"})
        note_id = _required_str(args, "note_id")
    except _ArgError as exc:
        return state, ToolResult.failure(str(exc))
    note = state.note(note_id)
    if note is None:
        return state, ToolResult.failure(f"no such note: {note_id}")
    return state, ToolResult.success(id=note.id, title=note.title, body=note.body)
