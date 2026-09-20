"""What one decision segment produced. Shared by the runner (writes) and the scorer (reads)."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StopReason = Literal["no_tool_calls", "step_cap", "error"]


class Usage(BaseModel):
    """Provider-reported token usage for one model call. None when the provider omits it."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    #: the tool's serialised result, exactly as the model saw it
    result: str = ""

    @property
    def ok(self) -> bool:
        """True when the tool call succeeded."""
        try:
            return bool(json.loads(self.result).get("ok"))
        except (ValueError, AttributeError):
            return False


class Step(BaseModel):
    """One model turn: what it said, what it called, and what those calls returned."""

    model_config = ConfigDict(extra="forbid")

    index: int
    content: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    prefix_hash: str
    scenario_id: str
    variant: str
    length: int
    prior_check_pattern: str
    model: str
    arm: str
    params: dict[str, Any] = Field(default_factory=dict)
    sample_index: int

    #: the decision segment only: the request, then the model's turns and tool results
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason = "no_tool_calls"
    final_text: str | None = None
    error: str | None = None

    @property
    def calls(self) -> list[ToolCallRecord]:
        """Every tool call in the decision segment, in order."""
        return [call for step in self.steps for call in step.tool_calls]
