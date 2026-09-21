"""History builder tests: exact counts, nesting, and the invariants in SPEC.md 3.4."""

from __future__ import annotations

import json

import pytest

from safety_checking.canonical import sha256_of
from safety_checking.history import (
    PRIOR_CHECK_PATTERNS,
    Episode,
    EpisodeCall,
    PrefixValidationError,
    build_prefix,
    check_nesting,
    load_episodes,
    load_plan,
    load_prefix,
    save_prefix,
)
from safety_checking.history.plan import HistoryPlan
from safety_checking.scenarios import load_scenario, load_scenario_world
from safety_checking.world.state import decision_relevant_view

LENGTHS = [5, 20, 50]


@pytest.fixture
def scenario():
    return load_scenario("sharing_risky")


@pytest.fixture
def world(scenario):
    return load_scenario_world(scenario)


def build(scenario, world, length, pattern="none"):
    return build_prefix(scenario, world, length, prior_check_pattern=pattern)


def tool_calls(prefix) -> list[tuple[str, dict]]:
    calls = []
    for message in prefix.messages:
        for call in message.get("tool_calls") or []:
            calls.append((call["function"]["name"], json.loads(call["function"]["arguments"])))
    return calls


def test_plan_lengths_and_cumulative_counts() -> None:
    plan = load_plan()
    assert plan.lengths == LENGTHS
    assert plan.tail_length == 5


def test_exact_call_count_at_every_length_and_pattern(scenario, world) -> None:
    for pattern in PRIOR_CHECK_PATTERNS:
        for length in LENGTHS:
            prefix = build(scenario, world, length, pattern)
            assert len(tool_calls(prefix)) == length
            assert prefix.metadata.n_calls == length


def test_last_five_calls_are_byte_identical_everywhere(scenario, world) -> None:
    prefixes = [
        build(scenario, world, length, pattern)
        for length in LENGTHS
        for pattern in PRIOR_CHECK_PATTERNS
    ]
    tails = {p.metadata.tail_hash for p in prefixes}
    assert len(tails) == 1

    tail_messages = [sha256_of(p.tail_messages) for p in prefixes]
    assert len(set(tail_messages)) == 1
    for prefix in prefixes:
        assert len([c for c in tool_calls(prefix)][-5:]) == 5
    last_five = {sha256_of(tool_calls(p)[-5:]) for p in prefixes}
    assert len(last_five) == 1


def test_shorter_histories_are_suffixes_of_longer_ones(scenario, world) -> None:
    for pattern in PRIOR_CHECK_PATTERNS:
        prefixes = [build(scenario, world, length, pattern) for length in LENGTHS]
        report = check_nesting(prefixes)
        assert report.suffix_ok
        assert len(set(report.tail_hashes.values())) == 1


def test_longer_histories_only_add_calls_further_back(scenario, world) -> None:
    short = tool_calls(build(scenario, world, 5))
    medium = tool_calls(build(scenario, world, 20))
    long = tool_calls(build(scenario, world, 50))
    assert medium[-5:] == short
    assert long[-20:] == medium


def test_tool_call_ids_are_numbered_backwards(scenario, world) -> None:
    for length in LENGTHS:
        prefix = build(scenario, world, length)
        ids = [
            call["id"] for message in prefix.messages for call in message.get("tool_calls") or []
        ]
        assert ids[-1] == "call_r001"
        assert ids[0] == f"call_r{length:03d}"


def test_history_never_inspects_or_sends_the_target(scenario, world) -> None:
    target = scenario.target_document
    for pattern in PRIOR_CHECK_PATTERNS:
        for length in LENGTHS:
            prefix = build(scenario, world, length, pattern)
            for tool, args in tool_calls(prefix):
                assert not (tool == "get_access_list" and args.get("document_id") == target)
                assert not (tool == "send_update" and args.get("document_id") == target)


def test_target_document_appears_only_in_the_tail(scenario, world) -> None:
    target = scenario.target_document
    for length in LENGTHS:
        prefix = build(scenario, world, length)
        head = prefix.messages[1 : prefix.metadata.tail_start_index]
        assert target not in json.dumps(head)
        assert target in json.dumps(prefix.tail_messages)


def test_pattern_none_contains_no_sends_and_no_checks(scenario, world) -> None:
    for length in LENGTHS:
        tools = [tool for tool, _ in tool_calls(build(scenario, world, length, "none"))]
        assert "send_update" not in tools
        assert "get_access_list" not in tools


def test_pattern_performed_contains_checked_shares_of_other_documents(scenario, world) -> None:
    for length in (20, 50):
        calls = tool_calls(build(scenario, world, length, "performed"))
        tools = [tool for tool, _ in calls]
        assert "send_update" in tools
        # every send is immediately preceded by a check of the same document
        for i, (tool, args) in enumerate(calls):
            if tool == "send_update":
                previous_tool, previous_args = calls[i - 1]
                assert previous_tool == "get_access_list"
                assert previous_args["document_id"] == args["document_id"]
                assert args["document_id"] != scenario.target_document


def test_decision_relevant_view_is_identical_across_lengths_and_patterns(scenario, world) -> None:
    views = []
    for pattern in PRIOR_CHECK_PATTERNS:
        for length in LENGTHS:
            prefix = build(scenario, world, length, pattern)
            views.append(sha256_of(decision_relevant_view(prefix.world())))
    assert len(set(views)) == 1
    assert views[0] == sha256_of(decision_relevant_view(world))


def test_filler_may_add_notes_but_nothing_visible(scenario, world) -> None:
    short = build(scenario, world, 5).world()
    long = build(scenario, world, 50).world()
    assert len(long.notes) > len(short.notes)
    assert decision_relevant_view(long) == decision_relevant_view(short)


def test_rebuilding_is_deterministic(scenario, world) -> None:
    for pattern in PRIOR_CHECK_PATTERNS:
        for length in LENGTHS:
            first = build(scenario, world, length, pattern)
            second = build(scenario, world, length, load_scenario_world(scenario) and pattern)
            assert first.prefix_hash == second.prefix_hash
            assert first.messages == second.messages
            assert first.world_snapshot == second.world_snapshot


# The experiment's stimuli, pinned. Code is written on one machine and jobs are submitted from
# another, and runs refer to prefixes by hash alone, so both copies must build these exact
# bytes. A failure here is either a platform difference (a bug) or a real change to the
# experiment, which needs a note in SPEC.md and new values here.
PINNED_TAIL_HASH = "ad127a41c2e3"
PINNED_PREFIX_HASHES = {
    "sharing_risky": {
        "none": {
            5: "ed6838f7e12cf264c97221f947d1a5b7daeef5b6799d02225703334cc9ed3dd9",
            20: "205cd211007104c4ea0f2ae9f0b16043a33438dcf0fd74d3018cc0fd7099e8c9",
            50: "75da62a1f10329cedab2ae2610b48818ef35ee3097efe3fa077aed37e20ca388",
        },
        "performed": {
            5: "0bb549484a6da9ced42c1deb23c73587247587f562d1001f471a173c0ae5bee0",
            20: "643bf86b896c3232d12badad497cf938ef166a0122255ab3e71d370b9da93b35",
            50: "0a2dc89e3301f97f741286280d17972458a024026913c8a8cf0abf0c49105b4b",
        },
    },
    "sharing_benign": {
        "none": {
            5: "cf51b23213434c457d4e433a90cb409408f3e021b3c4dcf12e718a306fe531c5",
            20: "204c942022789c0522ccc524b60d5a0ac79782ce1c053036869dce64cc06883f",
            50: "7e7fb96148e07fc85fcacec9d099427f38c947c77e2da9ffa8ba9352f068b0c5",
        },
        "performed": {
            5: "0b387e750442364e005cb959b4ca7d391d85cbc25f9c50c764172e820a0458b4",
            20: "f2535d7d4dc7aa1aa3cf01f86d855c2368a477c91bbf97c04926d443398904d3",
            50: "9a028725e6c0e5de3732c0dd57fc906d496b3de1df25878006b14dd6424344ca",
        },
    },
}


def test_prefix_hashes_are_pinned() -> None:
    built: dict[str, dict[str, dict[int, str]]] = {}
    for scenario_id in PINNED_PREFIX_HASHES:
        scenario = load_scenario(scenario_id)
        world = load_scenario_world(scenario)
        for pattern in PRIOR_CHECK_PATTERNS:
            for length in LENGTHS:
                prefix = build(scenario, world, length, pattern)
                assert prefix.metadata.tail_hash.startswith(PINNED_TAIL_HASH)
                built.setdefault(scenario_id, {}).setdefault(pattern, {})[length] = (
                    prefix.prefix_hash
                )
    assert built == PINNED_PREFIX_HASHES


def test_prefix_hash_does_not_depend_on_the_tokenizer(scenario, world, monkeypatch) -> None:
    """A machine without the tiktoken cache estimates token counts; its prefixes are the same."""
    exact = build(scenario, world, 20)
    monkeypatch.setattr("safety_checking.tokens._encoding", lambda: None)
    estimated = build(scenario, world, 20)
    assert estimated.metadata.token_count_estimated
    assert estimated.prefix_hash == exact.prefix_hash


def test_hash_changes_when_the_scenario_changes(scenario, world) -> None:
    base = build(scenario, world, 5)
    edited = scenario.model_copy(update={"decision_request": "Something else entirely, Dani."})
    assert build(edited, world, 5).prefix_hash != base.prefix_hash


def test_both_variants_share_the_same_history(scenario, world) -> None:
    """The twins differ in the decision, not in what came before it."""
    benign = load_scenario("sharing_benign")
    risky_prefix = build(scenario, world, 20)
    benign_prefix = build(benign, load_scenario_world(benign), 20)
    # the system prompt is identical, so the whole prefix is too
    assert risky_prefix.messages == benign_prefix.messages
    assert risky_prefix.prefix_hash != benign_prefix.prefix_hash  # scenario id is hashed in


def test_token_count_is_recorded(scenario, world) -> None:
    counts = [build(scenario, world, length).metadata.token_count for length in LENGTHS]
    assert counts == sorted(counts)
    assert counts[0] > 0


def test_prefix_round_trips_through_the_store(scenario, world, tmp_path) -> None:
    prefix = build(scenario, world, 5)
    path = save_prefix(prefix, tmp_path)
    assert path.name == f"{prefix.prefix_hash}.json"
    loaded = load_prefix(prefix.prefix_hash, tmp_path)
    assert loaded.model_dump() == prefix.model_dump()


def test_inserts_hook_places_messages_before_a_call(scenario, world) -> None:
    reminder = {"role": "user", "content": "(reminder: follow the standing rules)"}
    prefix = build_prefix(scenario, world, 5, inserts={3: [reminder]})
    index = prefix.messages.index(reminder)
    following = prefix.messages[index + 1]
    assert following["tool_calls"][0]["id"] == "call_r003"


# -- the validator itself ----------------------------------------------------


def _plan_with_tail(tail_ids: list[str], filler_ids: list[str], length: int) -> HistoryPlan:
    return HistoryPlan.model_validate(
        {
            "layers": [
                {"length": 5, "episodes": {"none": tail_ids, "performed": tail_ids}},
                {"length": length, "episodes": {"none": filler_ids, "performed": filler_ids}},
            ]
        }
    )


def test_validator_rejects_a_check_of_the_target_in_the_history(
    scenario, world, monkeypatch
) -> None:
    library = dict(load_episodes())
    library["ep_bad_check"] = Episode(
        id="ep_bad_check",
        user_request="Who can see the Atlas update?",
        calls=[EpisodeCall(tool="get_access_list", args={"document_id": "doc_atlas_update"})],
        summary="Checked.",
    )
    monkeypatch.setattr("safety_checking.history.episodes.load_episodes", lambda: library)
    monkeypatch.setattr("safety_checking.history.plan.get_episodes", _getter(library))
    monkeypatch.setattr("safety_checking.history.builder.get_episodes", _getter(library))

    plan = _plan_with_tail(
        ["ep_tail_unread", "ep_tail_design_review", "ep_tail_atlas_recap"], ["ep_bad_check"], 6
    )
    with pytest.raises(PrefixValidationError, match="access list"):
        build_prefix(scenario, world, 6, plan=plan)


def test_validator_rejects_filler_that_names_the_recipient(scenario, world, monkeypatch) -> None:
    library = dict(load_episodes())
    library["ep_names_dani"] = Episode(
        id="ep_names_dani",
        user_request="What is Dani working on this week?",
        calls=[EpisodeCall(tool="list_people", args={"team": "Marketing"})],
        summary="One person in marketing.",
    )
    monkeypatch.setattr("safety_checking.history.plan.get_episodes", _getter(library))
    monkeypatch.setattr("safety_checking.history.builder.get_episodes", _getter(library))

    plan = _plan_with_tail(
        ["ep_tail_unread", "ep_tail_design_review", "ep_tail_atlas_recap"], ["ep_names_dani"], 6
    )
    with pytest.raises(PrefixValidationError, match="requested recipient"):
        build_prefix(scenario, world, 6, plan=plan)


def test_validator_rejects_a_failing_call(scenario, world, monkeypatch) -> None:
    library = dict(load_episodes())
    library["ep_broken"] = Episode(
        id="ep_broken",
        user_request="Read the missing document.",
        calls=[EpisodeCall(tool="read_document", args={"document_id": "doc_missing"})],
        summary="Nothing there.",
    )
    monkeypatch.setattr("safety_checking.history.plan.get_episodes", _getter(library))
    monkeypatch.setattr("safety_checking.history.builder.get_episodes", _getter(library))

    plan = _plan_with_tail(
        ["ep_tail_unread", "ep_tail_design_review", "ep_tail_atlas_recap"], ["ep_broken"], 6
    )
    with pytest.raises(PrefixValidationError, match="failed"):
        build_prefix(scenario, world, 6, plan=plan)


def test_plan_rejects_a_layer_whose_counts_do_not_add_up() -> None:
    with pytest.raises(ValueError, match="not 7"):
        _plan_with_tail(
            ["ep_tail_unread", "ep_tail_design_review", "ep_tail_atlas_recap"],
            ["ep_eng_roster"],
            7,
        )


def _getter(library: dict[str, Episode]):
    def get(ids: list[str]) -> list[Episode]:
        return [library[i] for i in ids]

    return get
