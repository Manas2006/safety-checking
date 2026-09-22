"""Scenario schema.

A scenario says what the decision point is, what the required check is, and what the
consequential action is. Nothing here is specific to the `sharing` family: a second family
is new YAML, not new code (SPEC.md 3.2).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical import sha256_of
from ..world.state import WorldState

Variant = Literal["risky", "benign"]

#: position of the safety rule in the standing-rules list, fixed for v1 (SPEC.md decision 10)
SAFETY_RULE_INDEX = 1
N_STANDING_RULES = 4


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArgMatch(_Model):
    """Matches a tool call when the named arguments are all present and equal.

    Arguments that are not named are ignored, so ``send_update`` on the target document
    matches whatever recipient and message the model chose.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    def matches(self, tool_name: str, call_args: dict[str, Any] | None) -> bool:
        if tool_name != self.tool:
            return False
        call_args = call_args or {}
        return all(call_args.get(key) == value for key, value in self.args.items())


class SystemPromptSpec(_Model):
    """The system prompt, stored in YAML so its exact text is hashed into the prefix."""

    role: str
    standing_rules: list[str]

    @model_validator(mode="after")
    def _check_rules(self) -> SystemPromptSpec:
        if len(self.standing_rules) != N_STANDING_RULES:
            raise ValueError(f"expected {N_STANDING_RULES} standing rules")
        return self

    @property
    def safety_rule(self) -> str:
        return self.standing_rules[SAFETY_RULE_INDEX]

    def render(self) -> str:
        rules = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(self.standing_rules))
        return f"{self.role}\n\nStanding rules:\n{rules}"


class Scenario(_Model):
    id: str
    family: str
    variant: Variant
    world_ref: str
    #: which history plan builds this scenario's prefixes (episodes/<plan_ref>.yaml)
    plan_ref: str = "plan"
    system_prompt: SystemPromptSpec
    decision_request: str
    target_document: str
    requested_recipient: str
    required_check: ArgMatch
    consequential_action: ArgMatch
    expected_safe_behavior: str
    notes: str = ""

    def render_system_prompt(self) -> str:
        return self.system_prompt.render()

    def authorized_recipients(self, world: WorldState) -> list[str]:
        document = world.document(self.target_document)
        return list(document.access) if document else []

    def validate_against_world(self, world: WorldState) -> None:
        """Check that the scenario and its world agree. Raises ValueError if they do not."""
        document = world.document(self.target_document)
        if document is None:
            raise ValueError(f"{self.id}: no such target document: {self.target_document}")
        if world.person(self.requested_recipient) is None:
            raise ValueError(f"{self.id}: no such recipient: {self.requested_recipient}")

        on_list = self.requested_recipient in document.access
        if self.variant == "risky" and on_list:
            raise ValueError(
                f"{self.id}: risky variant requires a recipient NOT on the access list"
            )
        if self.variant == "benign" and not on_list:
            raise ValueError(f"{self.id}: benign variant requires a recipient ON the access list")

        if self.required_check.args.get("document_id") != self.target_document:
            raise ValueError(f"{self.id}: required_check must name the target document")
        if self.consequential_action.args.get("document_id") != self.target_document:
            raise ValueError(f"{self.id}: consequential_action must name the target document")

        recipient_name = world.person(self.requested_recipient).name
        if recipient_name.split()[0] not in self.decision_request:
            raise ValueError(f"{self.id}: decision_request does not name the recipient")

    def content_hash(self, world: WorldState) -> str:
        """Hash of the scenario together with its initial world, for the prefix metadata.

        ``plan_ref`` is left out: which plan built a prefix is fully expressed by the prefix's
        messages, which the prefix hash covers, and leaving it out keeps the hashes pinned
        before the field existed.
        """
        scenario = self.model_dump(mode="json", exclude={"plan_ref"})
        return sha256_of({"scenario": scenario, "world": world.snapshot()})
