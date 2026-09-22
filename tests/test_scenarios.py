"""Scenario tests: the twins are matched, and the YAML agrees with the world."""

from __future__ import annotations

import pytest

from safety_checking.scenarios import (
    ArgMatch,
    load_all_scenarios,
    load_scenario,
    load_scenario_world,
)
from safety_checking.scenarios.schema import SAFETY_RULE_INDEX

CONTROL_2X2 = {
    "sharing_incontext_benign",
    "sharing_incontext_risky",
    "sharing_lookup_benign",
    "sharing_lookup_risky",
}


def test_every_scenario_loads_and_validates() -> None:
    scenarios = {s.id: s for s in load_all_scenarios()}
    assert set(scenarios) == {"sharing_risky", "sharing_benign", *CONTROL_2X2}


def test_risky_recipient_is_off_the_list_and_benign_is_on_it() -> None:
    for name, expected in [("sharing_risky", False), ("sharing_benign", True)]:
        scenario = load_scenario(name)
        world = load_scenario_world(scenario)
        on_list = scenario.requested_recipient in scenario.authorized_recipients(world)
        assert on_list is expected


def test_twins_differ_only_in_the_recipient() -> None:
    risky = load_scenario("sharing_risky").model_dump()
    benign = load_scenario("sharing_benign").model_dump()
    differing = {k for k in risky if risky[k] != benign[k]}
    assert differing == {
        "id",
        "variant",
        "requested_recipient",
        "decision_request",
        "expected_safe_behavior",
        "notes",
    }
    # same world, same document, same tools, same policy text
    assert risky["world_ref"] == benign["world_ref"]
    assert risky["target_document"] == benign["target_document"]
    assert risky["system_prompt"] == benign["system_prompt"]
    assert risky["required_check"] == benign["required_check"]
    assert risky["consequential_action"] == benign["consequential_action"]


def test_decision_requests_differ_only_by_the_first_name() -> None:
    risky = load_scenario("sharing_risky").decision_request
    benign = load_scenario("sharing_benign").decision_request
    assert risky.replace("Dani", "NAME") == benign.replace("Wei", "NAME")


def test_system_prompt_places_the_safety_rule_second() -> None:
    scenario = load_scenario("sharing_risky")
    rules = scenario.system_prompt.standing_rules
    assert len(rules) == 4
    assert "get_access_list" in rules[SAFETY_RULE_INDEX]
    rendered = scenario.render_system_prompt()
    assert "Standing rules:" in rendered
    assert rendered.index("2. ") < rendered.index("3. ")
    assert "get_access_list" in rendered


def test_untriggered_rules_mention_no_tool_that_exists() -> None:
    """Rules 1, 3 and 4 must never be actionable in this world."""
    from safety_checking.world import tool_names

    scenario = load_scenario("sharing_risky")
    others = [
        rule
        for i, rule in enumerate(scenario.system_prompt.standing_rules)
        if i != SAFETY_RULE_INDEX
    ]
    for rule in others:
        assert not any(name in rule for name in tool_names())


def test_arg_match_ignores_unnamed_arguments() -> None:
    match = ArgMatch(tool="send_update", args={"document_id": "doc_atlas_update"})
    assert match.matches("send_update", {"document_id": "doc_atlas_update", "recipient": "p_x"})
    assert not match.matches("send_update", {"document_id": "doc_other"})
    assert not match.matches("get_access_list", {"document_id": "doc_atlas_update"})
    assert not match.matches("send_update", None)


def test_validation_rejects_a_mismatched_variant() -> None:
    scenario = load_scenario("sharing_risky")
    world = load_scenario_world(scenario)
    flipped = scenario.model_copy(update={"variant": "benign"})
    with pytest.raises(ValueError, match="benign variant"):
        flipped.validate_against_world(world)


def test_validation_rejects_an_unknown_recipient() -> None:
    scenario = load_scenario("sharing_risky")
    world = load_scenario_world(scenario)
    broken = scenario.model_copy(update={"requested_recipient": "p_ghost"})
    with pytest.raises(ValueError, match="no such recipient"):
        broken.validate_against_world(world)


def test_content_hash_is_stable_and_world_sensitive() -> None:
    scenario = load_scenario("sharing_risky")
    world = load_scenario_world(scenario)
    assert scenario.content_hash(world) == scenario.content_hash(load_scenario_world(scenario))

    changed = world.copy_state()
    changed.document(scenario.target_document).access.append("p_rivera")
    assert scenario.content_hash(changed) != scenario.content_hash(world)


# -- the 2x2 control (SPEC.md 3.2) ------------------------------------------------------


def _roster_ids(scenario_id: str) -> set[str]:
    """People whose ids any tool result in the length-5 history shows."""
    from safety_checking.history import build_prefix

    scenario = load_scenario(scenario_id)
    world = load_scenario_world(scenario)
    prefix = build_prefix(scenario, world, 5)
    ids: set[str] = set()
    for message in prefix.messages:
        if message.get("role") == "tool":
            ids |= {p.id for p in world.people if f'"{p.id}"' in message["content"]}
    return ids


def test_control_2x2_crosses_on_list_with_in_context() -> None:
    expected = {
        "sharing_incontext_benign": ("benign", True),
        "sharing_incontext_risky": ("risky", True),
        "sharing_lookup_benign": ("benign", False),
        "sharing_lookup_risky": ("risky", False),
    }
    for scenario_id, (variant, in_context) in expected.items():
        scenario = load_scenario(scenario_id)
        assert scenario.variant == variant
        assert (scenario.world_ref, scenario.plan_ref) == ("office_v2", "plan_v2")
        # in the tail, so at every length; the tail is the same for all four
        assert (scenario.requested_recipient in _roster_ids(scenario_id)) is in_context, scenario_id


def test_control_2x2_differs_only_in_the_recipient() -> None:
    dumps = {}
    for scenario_id in CONTROL_2X2:
        scenario = load_scenario(scenario_id)
        world = load_scenario_world(scenario)
        first_name = world.person(scenario.requested_recipient).name.split()[0]
        raw = scenario.model_dump(mode="json", exclude={"id", "variant", "notes"})
        raw["decision_request"] = raw["decision_request"].replace(first_name, "<NAME>")
        raw["requested_recipient"] = "<ID>"
        raw["expected_safe_behavior"] = "<TEXT>"
        dumps[scenario_id] = raw
    assert len({repr(sorted(d.items())) for d in dumps.values()}) == 1


def test_control_2x2_shares_the_original_system_prompt() -> None:
    original = load_scenario("sharing_risky").render_system_prompt()
    for scenario_id in CONTROL_2X2:
        assert load_scenario(scenario_id).render_system_prompt() == original
