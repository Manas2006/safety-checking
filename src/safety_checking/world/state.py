"""World state.

Plain pydantic models. No clock and no randomness: dates are literal strings in the data
and new ids are derived from the current contents, so replaying the same calls against the
same state always gives the same results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Person(_Model):
    id: str
    name: str
    email: str
    team: str
    role: str


class Document(_Model):
    id: str
    title: str
    owner: str
    body: str
    tags: list[str] = []
    #: person ids allowed to receive this document. Only ``get_access_list`` exposes it.
    access: list[str] = []
    #: reserved for the second risk family (editing a protected file)
    protected: bool = False


class Message(_Model):
    id: str
    sender: str
    subject: str
    body: str
    unread: bool = True


class CalendarEvent(_Model):
    id: str
    title: str
    day: str
    start: str
    end: str
    attendees: list[str] = []


class Note(_Model):
    id: str
    title: str
    body: str


class SentUpdate(_Model):
    id: str
    document_id: str
    recipient: str
    message: str


class WorldState(_Model):
    people: list[Person] = []
    documents: list[Document] = []
    inbox: list[Message] = []
    calendar: list[CalendarEvent] = []
    notes: list[Note] = []
    sent_updates: list[SentUpdate] = []

    # -- lookups -------------------------------------------------------------

    def person(self, person_id: str) -> Person | None:
        return next((p for p in self.people if p.id == person_id), None)

    def document(self, document_id: str) -> Document | None:
        return next((d for d in self.documents if d.id == document_id), None)

    def message(self, message_id: str) -> Message | None:
        return next((m for m in self.inbox if m.id == message_id), None)

    def event(self, event_id: str) -> CalendarEvent | None:
        return next((e for e in self.calendar if e.id == event_id), None)

    def note(self, note_id: str) -> Note | None:
        return next((n for n in self.notes if n.id == note_id), None)

    # -- snapshot / restore --------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> WorldState:
        return cls.model_validate(snapshot)

    def copy_state(self) -> WorldState:
        """A deep copy, for tools that change the world. Tools never mutate their input."""
        return self.model_copy(deep=True)


def decision_relevant_view(state: WorldState) -> dict[str, Any]:
    """Everything at the decision point that the agent could observe through a tool.

    Documents (including access lists) and people. Notes, inbox reads and the sent-update
    log are excluded on purpose: filler episodes are allowed to add notes and, under the
    ``performed`` pattern, to send updates about *other* documents, and no tool lets the
    decision segment observe either. See SPEC.md decision 9.
    """
    return {
        "documents": [d.model_dump(mode="json") for d in state.documents],
        "people": [p.model_dump(mode="json") for p in state.people],
    }
