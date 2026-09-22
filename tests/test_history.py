"""History builder tests: exact counts, nesting, and the invariants in SPEC.md 3.4."""

from __future__ import annotations

import json

import pytest

from safety_checking.canonical import canonical_json, sha256_of
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


# The 2x2 control scenarios (plan_v2: the Engineering roster opens the tail). Their own tail.
PINNED_TAIL_HASH_V2 = "f81e664594fa"
PINNED_PREFIX_HASHES_V2 = {
    "sharing_incontext_benign": {
        "none": {
            5: "d77b3a71839d4099e283a2339c4ba1096cb3d6e024837c9218df45205bc0c4f5",
            20: "f6abcbc25559f6f2286472db77b740c1fbed7d49901b673aa66874904ebf9426",
            50: "768b975313d884f2fd8e2928295a709ef5f0f495bea4091d64b4be1a3cb20b23",
        },
        "performed": {
            5: "2d71279575783b6a438fb04e6b9379050535b8b4d300919376d2bc980866eca0",
            20: "9957d07cc081f37ca899ce2abe59ded9156213d2783072c2c5643789904ca62a",
            50: "69df9feaa2142cef0bbd95edd87e30211d4d0ace21475fcaca51664b899d95bb",
        },
    },
    "sharing_incontext_risky": {
        "none": {
            5: "40699567fd7155af7dbc8ff2bb66af0c83f5db9d2f438c001b16b60084386e92",
            20: "b15a5426fd579e9d31ed0b9c63c71ec63031dc99a5d3002425de05edce5f7314",
            50: "cd9c18245733413cc1168554d9abb594b8e32a1cf3911d5ea102a9680b02b066",
        },
        "performed": {
            5: "736adee8095418ae858a32f68039764fd163ac707994d53ee013543acd18518f",
            20: "681f0ae1f737687d30cee4f98e3725ca0bc5b0f17bb7a2b0a658882036112baa",
            50: "64728c2c6ab3f26a58704b2607304fd1b441cf1732678c3b4ce237b2793acd8b",
        },
    },
    "sharing_lookup_benign": {
        "none": {
            5: "7ad31d58c07938288dbe41592a9721c6b4e2f38f2ca6cdb9b77fb9700c925cea",
            20: "41c73c03f4ebf47d53f7f46b34e92b2843c15dc93c5ebb1f5d6c7d10a66f4f79",
            50: "c0ee78913313c6e920a9637d6851b952580170528b4b275fceb89f1761e9881f",
        },
        "performed": {
            5: "6bcd8192beaa93621f600c52ff2448776d65b91c8164e40ec5556a4de16ac7e8",
            20: "73389dbd6ed5d2905e2e9e4e3f0cd0ae14794612cfbb0eb781e627c12400a1df",
            50: "84d6649a6512a9ca3195499bdea5855e297e29551ae7c2102705cab8efe7f366",
        },
    },
    "sharing_lookup_risky": {
        "none": {
            5: "1590bfe724b277e19abc082c672fed9b10f7c01a5d8d5dc62890a48c05a5d251",
            20: "e9c41154c9b19443ce9b3954090c716f05c2cb45d8541f606822b6755843f988",
            50: "744f3f694f95db8b6cdacaaad6497edd1f407a3855aa1363e6c134a1636068dc",
        },
        "performed": {
            5: "94ee575d447df63bf31bf15420e71d9bcc053f0a17565188260b58ee0ce7a16e",
            20: "fd6ba8f5dd7ac6c4d5c0deaab73842c978dbd9aec687491442b36dc76dd1f286",
            50: "682b3baf39896559773d50e1f9435f6dfb589f9403da870718bc13b034503197",
        },
    },
}


# The rule position / wording variants (plan, the original tail; SPEC.md open question 8).
PINNED_PREFIX_HASHES_RULES = {
    "sharing_risky_rule4": {
        "none": {
            5: "d30c008aaa4d06460f46addbde2ed7e4d5dc14363826d297031416b618a216d9",
            20: "705e2a00dcb283345e199c8f160cfbba4334f29bc43319b537f38d74555172ed",
            50: "8fb5036d332756ce0dcbe85b63679c4e783480d542dbb24306891591aa74fbbc",
        },
        "performed": {
            5: "4b56641ca05b21b82ea77daeaa0f173f12ff2833763d3dbd2ffe34197532bd33",
            20: "5172f79cacb2bde7021a982b1f9dfd3aac5c12232300c6e1e9d01b83c41ffd65",
            50: "d49e22b69bddd8354ded4149fbe9a1e96d2e7932cc391f57186120caede0de3c",
        },
    },
    "sharing_benign_rule4": {
        "none": {
            5: "3cb7eacb2c4829c227b5f28e32ec0dfed05f285fcb5c61d9e6075b067966ef76",
            20: "d78fc2d90bab78416e143aadf6cd8eb278f20b886514b26b16d800cdc065328e",
            50: "1640a0b2f7aab642de079e9fc2e4abfc76c3897718c78097e9f071138687fe04",
        },
        "performed": {
            5: "6811cf259e0e5782ce5d4e52f7d588e42c340fc026e539eae4749d3458283617",
            20: "0477ee3da7c18c452224efe94bb83b96cb98f7e29fa18b5a10c2fc15964ae91d",
            50: "cc4e3030abc552e20b106264b90e7009a1e3ecc0d45c770ceb3c27f4983dd835",
        },
    },
    "sharing_risky_weak": {
        "none": {
            5: "24d7c10141c8220cd29db605c2233814245e7df199741b454143d357f5fa53b3",
            20: "e71c148b4a5ba780cec47991baccb9de63f9e0f010cbd93c1192a2a8eefd8895",
            50: "6edde6d0a34990ddc619e560acf28569111b3c639c13cdb29786fb63f984623c",
        },
        "performed": {
            5: "5de570776bf20f4bf7f7d6affd3dd362402c96c6ad217db64a232e9d4f16e701",
            20: "f2ebf8f3cad10af466b5d931e4f86ee68f8bc2f0d81978c44b5c9f62e9d1bb4c",
            50: "04bbd1d0972bbf6e2a4f3b49f63d69f7c76525aa41d605caba59a62688694145",
        },
    },
    "sharing_benign_weak": {
        "none": {
            5: "3154697168cb1fa375f0ceda9765d1a82f3ebf480ba1a2952fef16efa42044ee",
            20: "c05629680bd34befcdb7c7903b59c659ad193693fa82a7e7834c84a44b390546",
            50: "4c33bced6085704459ea25e3bea768be69e03a189bbd5b029957f5f12e02ab68",
        },
        "performed": {
            5: "0c1300bab6fd5b6740a614dc8abb73a06020994c29e3c575c75907224abe0d79",
            20: "5034ba815c3d7f8cb3ca9ec31b639d1cb2a696e2d4eda468ee793a694e52dd50",
            50: "bf7c79334503642f87c519a0c15431d1db857efe78f887702cfa183af69cecb6",
        },
    },
}


# The in-context pair under the three ingredient plans (light, double20, nonotes) and with the
# rule moved first or fourth. All share plan_v2's tail.
PINNED_PREFIX_HASHES_2X2_VARIANTS = {
    "sharing_incontext_risky_light": {
        "none": {
            5: "93066f5a8c450598c60769784595b0cb80b9da286b4a560febe7bfda6096c855",
            20: "e13ea10724caf00b59e69bc3e3d1160533563b6df92e08446515b54f367e7d2d",
            50: "507356d5f1b62fc156c18a3ba687c8b7353943e609cb02e52aa352e899135b6b",
        },
        "performed": {
            5: "4aa5874bd266f94a274fd40b9b4c51e78c11cb278f0ee96dc10316ad48e9da5f",
            20: "d7a3c8bc7f2a99586661d2d5c38e1ba4b8abbe1081441cbbd24cddba8100545a",
            50: "1286af8d596a439242d80a915581c8f79af06b839fe5857407e15a01415aaec7",
        },
    },
    "sharing_incontext_benign_light": {
        "none": {
            5: "5c07c2f83354081faf147da5d7774d04f820eccd7775a1bb0914efe346bbf205",
            20: "cc4a97f6d1e64eac6773479047b942f8836ecff1cc28a3eb2ad0ac7456096123",
            50: "6e7212b21a8f2f47f514f896922f8b76b4c9a93ebdb4b466ea73c5c6a8437f3c",
        },
        "performed": {
            5: "9b661d9352dcd4fb740639e8644fac6c63e5f6e307f2dcb5a85ba167a60a3c47",
            20: "cb73b959bca947c57fd32f2180f6ffed00f78e3778f6bc76e3a28fba02b266ab",
            50: "4f309f63d76d75d5459867c37230c7aff2d89728d3fae5776a1b1fb4ba2a1cda",
        },
    },
    "sharing_incontext_risky_double20": {
        "none": {
            5: "004ef619b56f61d39110c58de23640af2db041793c6bcde324929fe7ee9449ca",
            20: "d888be866655fe222b4bb009c535605db686db51d6509f324e7de7fb7b3f5859",
            50: "e99fa52d867ca229db72e92ed6409dd1e4ece13f57b893ec4e2952d47c0c5b99",
        },
        "performed": {
            5: "30d592eab8c1d337468816a8e0afea5b89cb5bda9c213cb1f5f66577b6288222",
            20: "6083faab7a80528229fe1f045418c34e506daf00016ca03838e1d497f1a590bc",
            50: "90cf7bd77c69d26369d38b7ec27ac3d55f419a798496cfa5947406e509731563",
        },
    },
    "sharing_incontext_benign_double20": {
        "none": {
            5: "9e3a7cec711d0fa14b331e1d536b3b2ec317f7a4eb53c912a356cade15f31cfb",
            20: "aaf5b42bfc185b45a5a602d29ceb423273ab76a3f69c3d24713d32e7cd5c4d9d",
            50: "49bda2fc2088046972aec6c9db3e950bb139f82fda50d20a33d4f90daa845702",
        },
        "performed": {
            5: "aa87011cdaa483bc0a9371d7f28fc3d863421f9bc5dad6efa97bf35d89b96c4c",
            20: "b499d1f94199cae73699ae58e00d6c88031fb37bfb47fa3ae31df429eb9bab80",
            50: "d91f1683d2309aa41962e3343ab4fbaa0757cec083104e65571f712f213c3ad8",
        },
    },
    "sharing_incontext_risky_nonotes": {
        "none": {
            5: "0fa0da6373105564f9b4736215b63b3f71d64ff9f1f7432bfcc23790854d9f21",
            20: "2106f0c90ee7c4f12e4376a431e5bc0be1e80f6be6a39e3a3cef81338c94cf3a",
            50: "96d8abb0af179476e2006342560f376cc7c70385fc8b29fb67d64f8849af4fbe",
        },
        "performed": {
            5: "b5678e2f7d86f725a632de1fe7ad397ed4cf7aea0bd04d66f35a0171ca9b48e9",
            20: "b28f84c9763a558d31a66746297c298d3bea460f7ce68f8504fb4dc5507e00a4",
            50: "dfb8131d7c317ab891285785788f8c9594b6a6e7df389957c79a536d29b5f455",
        },
    },
    "sharing_incontext_benign_nonotes": {
        "none": {
            5: "8a45792f4e24c7d82c107948300095df4ab7750c0191ef0646811986e4c7880c",
            20: "9d7d9b1863f1c736643c105d43776991ab8826df70fd3211280b4a9d2125dec8",
            50: "e9a7e487b361b2deed223275c0601383d4dbe19b3aff0ece856100410e8d257e",
        },
        "performed": {
            5: "9d3921595f3505b5769d5172b81647f283977aa54b945e743268dd0062bb4d6c",
            20: "5589e5edc328d22142108db7f76efa44ae0496bbf62a1cde92e5d818c7ae3a0d",
            50: "0f8f57bf9771f4b3f711a387636eb04fa7dbbe04745a55d39e1e307b34ad1777",
        },
    },
    "sharing_incontext_risky_rule1": {
        "none": {
            5: "241bbe0c832918ef7842a59f0ddf2ed6dd4624943db48228764725fd5314d83c",
            20: "7f06c9f01dc0054cd40769c19ff2495b51c5612f999db723645a0fdf4d3f1f33",
            50: "739cc1d123306c4597c2d587bec247620e77b132abd2cf447d9af481a0473ed7",
        },
        "performed": {
            5: "d54babae3a37fa57c302cd0244a4da68f8c5f6b7a37051ce8584e33664613720",
            20: "9c73e4aa0ad39316ea5a9bc33b37f266b8421d02995be2d47aabc84bc6cad487",
            50: "375347f4100c29ac578f8587a7f643a7c4fd3d7cb11bd2718b0f534a7c31fa3a",
        },
    },
    "sharing_incontext_benign_rule1": {
        "none": {
            5: "458cf2b1b2e24c04b37248fd9542f7ec2e45407725171518911d0ef01428e43d",
            20: "90f3509922388358f7535521d089bc12d7efc4d363ceaf90ee0919e6d6b3b0cf",
            50: "a6b8b9d0a71a3baf13b8677ed94cd149a76058d8c92e9ccfe3507301803cf4d1",
        },
        "performed": {
            5: "b5ddd35510f1afc8d887d8a91b281c1e27decefd78710619b8af1fa5602bd397",
            20: "f14244e7d128bff1a34f05d6de94b72e3c53f033ac2f0c968d608cbc510102de",
            50: "ee44eaac69c4ff2ee1ae1774eb74cd8c270edc8c794383fe47bc490e2bef2f57",
        },
    },
    "sharing_incontext_risky_rule4": {
        "none": {
            5: "1287b08df5b0ec37ccc9ee5ae0ef37b6e12e418b624c3608059b3ada02f4e2fa",
            20: "b56c656ab572bdd7dd4b20593e3cbdf496d1995f35f4b9986d9487dcdbe0faa0",
            50: "4d27f905f6f38954466f4bdd3f6344e41496aa8a273d9e1e68a9216d75e8c5c8",
        },
        "performed": {
            5: "33714199e9e089d778ff1575f17598c43a1a8832d6220354909c655716039f0e",
            20: "e4fc6c57b811b759bdeb24418a1c4f86f23217e6c16a1aa1bd0d777a5c3e9227",
            50: "ea357a65cebe25bc2b5fe4df08668438b326826d0ba68d9844c0823da891c5d2",
        },
    },
    "sharing_incontext_benign_rule4": {
        "none": {
            5: "0b7c398a792b1cb616f5efb19610bfea307ef18f73f5a14d4fdbdfefaafc732f",
            20: "077a5544dfa69ae2f97bd369e2ece8e3aa2998ac2985efb7bb5dcba63b79cae7",
            50: "fcf80059590ecfdb6911b2fcd897d34f39d517d85847c3cd8aef9f33ae5ab15f",
        },
        "performed": {
            5: "ff013a849313b9356c1a1ffd72fd77fcc54c8095f0fa9c12e03f91cbba55d740",
            20: "4490436d1cbc5b19d794464153374fbf39882971600ef9ef8073a83fb0bcf368",
            50: "a0dabe75a84b49b57085995e687e3754267283c28eca294f2a2b5312cd84a9b2",
        },
    },
}


@pytest.mark.parametrize(
    ("pinned", "tail_hash"),
    [
        (PINNED_PREFIX_HASHES, PINNED_TAIL_HASH),
        (PINNED_PREFIX_HASHES_V2, PINNED_TAIL_HASH_V2),
        (PINNED_PREFIX_HASHES_RULES, PINNED_TAIL_HASH),
        (PINNED_PREFIX_HASHES_2X2_VARIANTS, PINNED_TAIL_HASH_V2),
    ],
    ids=["plan", "plan_v2", "rules", "2x2_variants"],
)
def test_prefix_hashes_are_pinned(pinned, tail_hash) -> None:
    built: dict[str, dict[str, dict[int, str]]] = {}
    for scenario_id in pinned:
        scenario = load_scenario(scenario_id)
        world = load_scenario_world(scenario)
        for pattern in PRIOR_CHECK_PATTERNS:
            for length in LENGTHS:
                prefix = build(scenario, world, length, pattern)
                assert prefix.metadata.tail_hash.startswith(tail_hash)
                built.setdefault(scenario_id, {}).setdefault(pattern, {})[length] = (
                    prefix.prefix_hash
                )
    assert built == pinned


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


# The reminder arm (configs/contrast_reminder.yaml): one user message inserted before call
# r005 at every length, so the tail (which now contains it) is still identical across lengths.
PINNED_TAIL_HASH_REMINDER = "429dea631414"
PINNED_PREFIX_HASHES_REMINDER = {
    "sharing_risky": {
        5: "50461f2f37081b86bba22e46ca1275504bda57fa27c0a48b46deca276724c2b2",
        20: "7046590e396ebf2d553080be45b9f52e7369e403898fc61d049413a707b9f6e6",
        50: "ddc620117276daeefb5c0ba09a59698491e5ddddc4c84be28f03923262ed8803",
    },
    "sharing_benign": {
        5: "877dc4a6a7a50d83072120ca1c70304925b1f312deeba33a2e0ac13486525192",
        20: "fe7938431a4ce42f2151ea70ba30434e32b65f024f00a1cbaa00f6595a9ee518",
        50: "e282283de002d60ece9b7e8a8fb46c4b1a8acb53953375276229a5845be35437",
    },
}


def test_reminder_arm_prefixes_are_pinned_and_nested() -> None:
    from safety_checking.config import load_experiment_config
    from safety_checking.runner.experiment import ExperimentSpec, build_cells

    experiment, _ = load_experiment_config("configs/contrast_reminder.yaml")
    assert experiment.inserts and experiment.arm != "baseline"
    spec = ExperimentSpec(
        experiment=experiment.experiment,
        scenarios=experiment.scenarios,
        lengths=experiment.lengths,
        inserts=experiment.inserts,
    )
    cells = build_cells(spec, save=False)
    built: dict[str, dict[int, str]] = {}
    for cell in cells:
        prefix = cell.prefix
        assert prefix.metadata.tail_hash.startswith(PINNED_TAIL_HASH_REMINDER)
        reminder = experiment.inserts[5][0]
        index = prefix.messages.index(reminder)
        assert prefix.messages[index + 1]["tool_calls"][0]["id"] == "call_r005"
        built.setdefault(cell.scenario.id, {})[prefix.length] = prefix.prefix_hash
    assert built == PINNED_PREFIX_HASHES_REMINDER
    for scenario_id in experiment.scenarios:
        prefixes = [c.prefix for c in cells if c.scenario.id == scenario_id]
        assert check_nesting(prefixes).suffix_ok


# The long-horizon scenarios (plan_long / plan_v2_long: lengths 5, 20, 50, 100, 200). Their
# histories up to 50 are the bases' messages; the two new layers are pinned here.
LONG_LENGTHS = [5, 20, 50, 100, 200]
PINNED_PREFIX_HASHES_LONG = {
    "sharing_risky_long": {
        "none": {
            5: "e02f2af187a631456c96ceb32f74c36addf4ae1591c4e6299c0aa531f001d6d5",
            20: "5681e47dbc53100eee7869439b6a6f58499275cd1e26e3767f8067b54a07091c",
            50: "c625be327da9951ac618e78c99ff3125893fd4274396ccd942c3ac9ee1aaf870",
            100: "8461817791f9463cc38922c3dcc84ad703d395000f896e5a784c9959106c5304",
            200: "799b5482bc2bb2344b54e3b51c909cbfdd071c4870bb285ea3b4428a3134d5b5",
        },
        "performed": {
            5: "04bf8fd8ed497159a35a0bd03f9fe7f7a34d20c301004277ef3f0f8b3455d3aa",
            20: "1eedb505f07eb717dbca938604719b5f31c6ea277c44b4e28b1b149cdac48b8c",
            50: "72f0d8cf27f899c08fb71bb846861731853288fe2d603fc8a4016ec10b432c98",
            100: "0681f761247ab240c55bbb4e40ffdb0c9a2a4489b0719a7b214e86b6661bc443",
            200: "fe86004dbe54a6e709f36c77264edbce93945c51162ce42192c4107f7d47eea6",
        },
    },
    "sharing_benign_long": {
        "none": {
            5: "9f56e40086ad10dae5bc97d290b9c65f65070d8b9b225c48463d00014dc415e1",
            20: "dcfdac6ec02a23b26382ea218caca3ad9e07028ca34596dfacc90c845037be62",
            50: "d7f8b02c7f8fee7a3b0ad6ee1f8f1a395a2255543faf521973d28654bde4f922",
            100: "c0507f8763595ecd001b5c693c71a2abc8c3f270997e7fd1fa8999624d6fc2f1",
            200: "c24d9d1c6b612fded6f862fa458e9b297af43d2bce1507d6dbea99d796a6782e",
        },
        "performed": {
            5: "aabc7fb9949494c5bf85d8318d9fcc92ed94483941b67dc13157db4115acc7f2",
            20: "b48d7fee22f3e96430698a966f2b12b644b4b306f996bc80b924e6735ab210a1",
            50: "9df0bb86c980d00f9900b88a3c67111077a3cf73275fc93e8d253de49c58df6e",
            100: "7e49e11f3774c5400ca34b87a4d2ac2fac10dc129ebc6e732b2fb8db2d308a76",
            200: "8bff62fc9ff0773c058033cd20ac27108ce099abe3b0ebc6c4ec42f8b428da40",
        },
    },
    "sharing_incontext_risky_long": {
        "none": {
            5: "ec3f010559ed104b3d2e1f4d4835885aba7d52dc13b1b9fbfee76c6447162d5e",
            20: "fb4842b5dff0c5b7942d3bf8de48f215d34f3f183540a605d93e185c407d1e31",
            50: "3c132990ca9359f91a021813492f1ebf88abee736e2557dbe173e361dcb10bb7",
            100: "72bea8faaa30ffb73e489e07e98ac8c1adf44d6fc29ce6d4ec0468b16f942d42",
            200: "6b5bc0ec7e9269b1bbd3b3d507404ae51012ab06be4ea0259beac3b2d2e1d26c",
        },
        "performed": {
            5: "770c8b5c56b9d6bbe72106698897f4352277d4a35a9453d19b358e1bac18a6cc",
            20: "cc95795c26eca013cf334ee4db74f27ee95035a0555efa09f2dfca0876560cbb",
            50: "24525f1a1aaa9e3192314a3f0c33a24c4d740d0d063adb67367a662d5912ef71",
            100: "96d465b1176695be5e4229214e3e4c34d531a7e0e4e5a3f74d33a90f916c56c5",
            200: "4e261a200b0dada8dd4c06538d2501be766750a7d2faa92d2ace9853537fac19",
        },
    },
    "sharing_incontext_benign_long": {
        "none": {
            5: "4870e034e70bb703d21f00fa55a16be785b9c75579b12585921a9fec55c740fb",
            20: "6812c7a29fee0ddcf259c581671b773534af957ca220c369189be5f0401be3e8",
            50: "aa15f3580bed392f38134ac131cca1539e611a883e1eac3b9151395ac96a5bc0",
            100: "19e5e4206d7647111e53449fae2f8436e5c9959ac24d24e05b52ad14f6630a7d",
            200: "f05bb49a31a82fa7971cc447f7aa4fd9d027c32cdc5c48cde0243f1f30c653e9",
        },
        "performed": {
            5: "5b0331295d28429e2f935e5cd817a7b2b7dd29fb21c3bfe2378d1cf96473f613",
            20: "8ca5746609db2d70d12200c155d339887aae7c106a4f8fb6b917f19a9ea2ba9f",
            50: "ab2fe00840567bc71cf49d950f01b20c92a174de733d0036d3e0b9a80156daae",
            100: "c88f50179e8f559dba1fa325d451c4b24110ada37c7808f1a340213c50cca544",
            200: "656a22773b4309ac94f3282b5bd980d6952e05a3d070c7d62c2668d2eb9c620b",
        },
    },
    "sharing_lookup_risky_long": {
        "none": {
            5: "5aa175dbfbba0856b5a2e79709f868f49dadf50a618781bf7e1a26ba32758ef1",
            20: "a1bc5fd961fb899a434f0d867b0d26484cac50c74b6d29a7e5edc44aeb156389",
            50: "744eebe777cc2fa1cc26c316613b3b7bb0d78e5eea10dcaea1ea4bf1a1552520",
            100: "faec4c533b4f4813212af1b1afaac4fc45ade49046019aff4f3cd0542270eea2",
            200: "b50a5b1ebb6454a6a13de5d830e2555e63fabd1431b8cd3a981120ae01aed15a",
        },
        "performed": {
            5: "fe4152557435e1f9cdfc33eefb57055301f8e46cb83ff18060fa212c9a21422a",
            20: "0a5083536a19fd36d310f2ba22e0f4294afb43af21ed5dd59e76bca550fc4004",
            50: "a5e461fa5f3d93706ac3106d3efdfc2df49f5edc97525567d553d51075d8b755",
            100: "502267c538d229d0f7facf2e3f441e36c50f84c13539c98e0e3f0c7bfbfb7709",
            200: "5b540a13a8c7fc40583d748b6a3c9b26456ed2358c45750c56f25aba65156121",
        },
    },
    "sharing_lookup_benign_long": {
        "none": {
            5: "e81487a3f1545e2ca7002be7ebf7a1ec4348fb476d0b005741689a874db7fa22",
            20: "81483948b3c8808d5eae41d911887582082a7569f3cfe2f599ee421c5647e443",
            50: "98aa555130fe73373d429fe9d7e086e4b3678754a84ff511a1a336fe95fd9165",
            100: "19b713512fc2a0f0125ad34697ac1de75e50c442cf4f69187cadf90473f69056",
            200: "fd7e035b2797e250c40fedbffdc596839b84e74c8c20dcc8226fee99309715ce",
        },
        "performed": {
            5: "0c3ab4031d2c012634c1d7aef06fbc8322da190fdb06feb20b561a7f8da28098",
            20: "e0ca2da09c0c1f239cda9a57f01815894feb9335513f09c72dec41ef24f49053",
            50: "693b15459e2acc8c71bf2c9ab3775547722762923d4dbfde22289047eb4d8f10",
            100: "0a5eeeb23dcc16750f8f7defd18b84a3b0281e5a0787d36cc9a1d2e608d6364c",
            200: "b2d0ce921ef52142f47ef5666be66e774a1594ca6cbbe9e59d12edb1df9ba43a",
        },
    },
}


def test_long_horizon_prefixes_are_pinned_nested_and_extend_their_bases() -> None:
    built: dict[str, dict[str, dict[int, str]]] = {}
    for scenario_id in PINNED_PREFIX_HASHES_LONG:
        scenario = load_scenario(scenario_id)
        world = load_scenario_world(scenario)
        base = load_scenario(scenario_id.removesuffix("_long"))
        base_world = load_scenario_world(base)
        for pattern in PRIOR_CHECK_PATTERNS:
            prefixes = [build(scenario, world, length, pattern) for length in LONG_LENGTHS]
            report = check_nesting(prefixes)
            assert report.suffix_ok and len(set(report.tail_hashes.values())) == 1
            for prefix in prefixes:
                built.setdefault(scenario_id, {}).setdefault(pattern, {})[prefix.length] = (
                    prefix.prefix_hash
                )
                if prefix.length in LENGTHS:  # the same bytes the base scenario builds
                    same = build(base, base_world, prefix.length, pattern)
                    assert prefix.messages == same.messages
            # the new layers never list a recipient that the old layers never listed
            new_part = prefixes[-1].messages[
                : len(prefixes[-1].messages) - len(prefixes[2].messages)
            ]
            assert all(
                pid not in canonical_json(m) for m in new_part for pid in ("p_rivera", "p_ferreira")
            )
    assert built == PINNED_PREFIX_HASHES_LONG


# The repeated-episode control (plan_v2_repeat): the plan_v2 tail, filler one episode repeated.
PINNED_PREFIX_HASHES_REPEAT = {
    "sharing_incontext_risky_repeat": {
        5: "be18fe06241cd102ae88efb0e91164a7c7fb3dbccc2b6e3a9370fda6d1a13c9d",
        20: "d45a1f52d4fa2825d0baf7a5d7ccca6016a6ffb98014e157e93cea397dd5c9fd",
        50: "fa3317f04ca40a99d21dc13e2f5234d7829eec03511f5348ad99e3f34fe43aeb",
    },
    "sharing_incontext_benign_repeat": {
        5: "0b81940a32e143e3484fe8fbb104ad56dcaa0bd2d7e7460e98d2688565375130",
        20: "73004e772de5be874dbe997d34e8ad1334d842548f1df12bc48044286d2de1d7",
        50: "a57f2073bca4c62a2e38957bfb11a33f492113cac1e2af76390498e618641c19",
    },
}


def test_repeated_episode_prefixes_are_pinned_and_nested() -> None:
    built: dict[str, dict[int, str]] = {}
    for scenario_id in PINNED_PREFIX_HASHES_REPEAT:
        scenario = load_scenario(scenario_id)
        world = load_scenario_world(scenario)
        prefixes = [build(scenario, world, length) for length in LENGTHS]
        report = check_nesting(prefixes)
        assert report.suffix_ok and len(set(report.tail_hashes.values())) == 1
        assert prefixes[0].metadata.tail_hash.startswith(PINNED_TAIL_HASH_V2)
        episode_ids = prefixes[-1].metadata.episode_ids
        assert set(episode_ids[:-3]) == {"ep_week_overview"} and len(episode_ids[:-3]) == 15
        for prefix in prefixes:
            built.setdefault(scenario_id, {})[prefix.length] = prefix.prefix_hash
    assert built == PINNED_PREFIX_HASHES_REPEAT


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
