"""Build a frozen prefix: system prompt + N benign tool calls, executed against the world.

Everything here is deterministic. The same inputs always produce the same messages, the same
world snapshot and the same hash.

Tool call ids are numbered *backwards* from the end of the history (``call_r001`` is the last
call). Forward numbering would make the final block differ between lengths and break the
nesting property, which is the point of the whole design (SPEC.md decision 11).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..canonical import canonical_json, sha256_of
from ..scenarios.schema import Scenario
from ..tokens import count_messages
from ..world.registry import REGISTRY, execute
from ..world.state import WorldState, decision_relevant_view
from .episodes import Episode, get_episodes
from .plan import PRIOR_CHECK_PATTERNS, HistoryPlan, PriorCheckPattern, load_plan

BUILDER_VERSION = 1

#: tools a filler episode may call even though they change the world. Notes are invisible to
#: the decision, and `performed` shares are of *other* documents (SPEC.md decision 9).
FILLER_MUTATING_TOOLS = frozenset({"create_note", "send_update"})
#: the tail must contain no sends and no access checks under either pattern
TAIL_FORBIDDEN_TOOLS = frozenset({"send_update", "get_access_list"})

Message = dict[str, Any]
#: extra messages keyed by the call's backwards index (1 = last call), inserted immediately
#: before that call's assistant message. Unused in v1; the reminder and replanning arms need it.
Inserts = dict[int, list[Message]]


class PrefixValidationError(ValueError):
    """A built prefix broke one of the invariants the experiment relies on."""


class PrefixMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_calls: int
    n_messages: int
    tail_length: int
    tail_start_index: int
    tail_hash: str
    token_count: int
    token_count_estimated: bool
    episode_ids: list[str]
    builder_version: int = BUILDER_VERSION


class Prefix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix_hash: str
    scenario_id: str
    scenario_hash: str
    length: int
    prior_check_pattern: str
    messages: list[Message]
    world_snapshot: dict[str, Any]
    metadata: PrefixMetadata

    @property
    def tail_messages(self) -> list[Message]:
        return self.messages[self.metadata.tail_start_index :]

    def world(self) -> WorldState:
        return WorldState.restore(self.world_snapshot)


def _call_id(remaining: int) -> str:
    """Backwards-numbered id: the last call in the history is always ``call_r001``."""
    return f"call_r{remaining:03d}"


def _assistant_tool_call(call_id: str, tool: str, args: dict[str, Any]) -> Message:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": canonical_json(args)},
            }
        ],
    }


def build_prefix(
    scenario: Scenario,
    world: WorldState,
    length: int,
    *,
    prior_check_pattern: PriorCheckPattern = "none",
    plan: HistoryPlan | None = None,
    inserts: Inserts | None = None,
) -> Prefix:
    """Assemble, execute and validate a history of exactly ``length`` tool calls."""
    if prior_check_pattern not in PRIOR_CHECK_PATTERNS:
        raise ValueError(f"unknown prior_check_pattern: {prior_check_pattern}")
    plan = plan or load_plan(scenario.plan_ref)
    episode_ids = plan.episode_ids(length, prior_check_pattern)
    episodes = get_episodes(episode_ids)
    tail_ids = plan.tail_episode_ids(prior_check_pattern)

    total_calls = sum(e.n_calls for e in episodes)
    if total_calls != length:
        raise PrefixValidationError(
            f"plan for length {length}/{prior_check_pattern} makes {total_calls} calls"
        )

    messages: list[Message] = [
        {"role": "system", "content": scenario.render_system_prompt()},
    ]
    state = world.copy_state()
    remaining = total_calls
    tail_start_index: int | None = None
    #: (episode, tool, args, result_json, is_tail) for the invariant checks below
    executed: list[tuple[Episode, str, dict[str, Any], str, bool]] = []

    for episode in episodes:
        is_tail = episode.id in tail_ids
        if is_tail and tail_start_index is None:
            tail_start_index = len(messages)
        messages.append({"role": "user", "content": episode.user_request})
        for call in episode.calls:
            if inserts and remaining in inserts:
                messages.extend(inserts[remaining])
            call_id = _call_id(remaining)
            messages.append(_assistant_tool_call(call_id, call.tool, call.args))
            state, result = execute(state, call.tool, call.args)
            if not result.ok:
                raise PrefixValidationError(
                    f"episode {episode.id}: call {call.tool}{call.args} failed: {result.error}"
                )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.to_json()})
            executed.append((episode, call.tool, call.args, result.to_json(), is_tail))
            remaining -= 1
        messages.append({"role": "assistant", "content": episode.summary})

    if tail_start_index is None:
        raise PrefixValidationError("no tail episodes in the plan")

    _validate(scenario, world, state, messages, executed, episodes, tail_ids, length)

    counted = count_messages(messages)
    metadata = PrefixMetadata(
        n_calls=total_calls,
        n_messages=len(messages),
        tail_length=plan.tail_length,
        tail_start_index=tail_start_index,
        tail_hash=sha256_of(messages[tail_start_index:]),
        token_count=counted.tokens,
        token_count_estimated=counted.estimated,
        episode_ids=episode_ids,
    )
    world_snapshot = state.snapshot()
    scenario_hash = scenario.content_hash(world)
    prefix_hash = sha256_of(
        {
            "scenario_hash": scenario_hash,
            "scenario_id": scenario.id,
            "length": length,
            "prior_check_pattern": prior_check_pattern,
            "messages": messages,
            "world_snapshot": world_snapshot,
            "builder_version": BUILDER_VERSION,
        }
    )
    return Prefix(
        prefix_hash=prefix_hash,
        scenario_id=scenario.id,
        scenario_hash=scenario_hash,
        length=length,
        prior_check_pattern=prior_check_pattern,
        messages=messages,
        world_snapshot=world_snapshot,
        metadata=metadata,
    )


def _text_of(messages: list[Message]) -> str:
    return canonical_json(messages)


def _validate(
    scenario: Scenario,
    initial_world: WorldState,
    final_world: WorldState,
    messages: list[Message],
    executed: list[tuple[Episode, str, dict[str, Any], str, bool]],
    episodes: list[Episode],
    tail_ids: list[str],
    length: int,
) -> None:
    """Hard invariants. Every one of these is a reason the experiment would be unreadable."""
    target = scenario.target_document

    if len(executed) != length:
        raise PrefixValidationError(f"expected {length} calls, executed {len(executed)}")

    for episode, tool, args, result_json, is_tail in executed:
        if tool == "get_access_list" and args.get("document_id") == target:
            raise PrefixValidationError(
                f"episode {episode.id}: the history must never inspect the target's access list"
            )
        if tool == "send_update" and args.get("document_id") == target:
            raise PrefixValidationError(
                f"episode {episode.id}: the history must never send the target document"
            )
        if is_tail and tool in TAIL_FORBIDDEN_TOOLS:
            raise PrefixValidationError(f"episode {episode.id}: {tool} is not allowed in the tail")
        if not is_tail:
            if REGISTRY[tool].mutates and tool not in FILLER_MUTATING_TOOLS:
                raise PrefixValidationError(
                    f"episode {episode.id}: filler may not call the mutating tool {tool}"
                )
            if target in canonical_json({"tool": tool, "args": args}):
                raise PrefixValidationError(
                    f"episode {episode.id}: the target document may appear only in the tail"
                )
            if target in result_json:
                raise PrefixValidationError(
                    f"episode {episode.id}: {tool} returns the target document, which may "
                    "appear only in the tail"
                )

    # the target document and the requested recipient must not be foregrounded by any filler
    # user request or assistant summary; the tail is where the target may appear.
    target_title = initial_world.document(target).title if initial_world.document(target) else ""
    recipient = initial_world.person(scenario.requested_recipient)
    for episode in episodes:
        if episode.id in tail_ids:
            continue
        text = f"{episode.user_request}\n{episode.summary}"
        for needle in (target, target_title):
            if needle and needle.lower() in text.lower():
                raise PrefixValidationError(
                    f"episode {episode.id}: filler text mentions the target document"
                )
        if recipient is not None:
            for needle in (recipient.id, *recipient.name.split()):
                if needle.lower() in text.lower():
                    raise PrefixValidationError(
                        f"episode {episode.id}: filler text names the requested recipient"
                    )

    if decision_relevant_view(final_world) != decision_relevant_view(initial_world):
        raise PrefixValidationError(
            "the history changed something the decision can see (documents, access or people)"
        )

    if any(u.document_id == target for u in final_world.sent_updates):
        raise PrefixValidationError("the history sent the target document")

    if _text_of(messages).count('"role":"system"') != 1:
        raise PrefixValidationError("expected exactly one system message")


def build_all(
    scenario: Scenario,
    world: WorldState,
    lengths: list[int] | None = None,
    patterns: list[str] | None = None,
    plan: HistoryPlan | None = None,
) -> list[Prefix]:
    """Build every (length, pattern) prefix for one scenario."""
    plan = plan or load_plan(scenario.plan_ref)
    lengths = lengths or plan.lengths
    patterns = patterns or list(PRIOR_CHECK_PATTERNS)
    return [
        build_prefix(scenario, world, length, prior_check_pattern=pattern, plan=plan)
        for length in lengths
        for pattern in patterns
    ]


class NestingReport(BaseModel):
    """Evidence that the shorter histories really are suffixes of the longer ones."""

    tail_hashes: dict[str, str] = Field(default_factory=dict)
    suffix_ok: bool = True


def check_nesting(prefixes: list[Prefix]) -> NestingReport:
    """Check the tails match and each shorter history is a suffix of each longer one."""
    report = NestingReport()
    by_length = sorted(prefixes, key=lambda p: p.length)
    for prefix in by_length:
        key = f"{prefix.length}/{prefix.prior_check_pattern}"
        report.tail_hashes[key] = prefix.metadata.tail_hash
    for short in by_length:
        for long in by_length:
            if short.length >= long.length:
                continue
            if short.length > 0 and short.prior_check_pattern != long.prior_check_pattern:
                # only the tail is shared across patterns; longer layers differ by design
                continue
            short_body = short.messages[1:]  # drop the system message
            long_body = long.messages[1:]
            if long_body[-len(short_body) :] != short_body:
                report.suffix_ok = False
    return report
