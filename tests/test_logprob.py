"""Logprob mode against a mock server.

The mock has a real (toy) tokenizer and a model whose next-call distribution is known exactly,
so the trie arithmetic is checked against ground truth, not against itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from safety_checking import cli
from safety_checking.config import (
    ExperimentConfig,
    LogprobFormat,
    ModelConfig,
    ProbePoint,
    load_experiment_config,
    load_model_config,
)
from safety_checking.history import build_prefix
from safety_checking.logprob import (
    NEUTRAL_SAMPLING,
    LogprobError,
    ServerClient,
    common_prefix,
    compute_probe_id,
    logprob_log_path,
    measure,
    probe_messages,
    read_logprob_records,
    run_logprob,
)
from safety_checking.scenarios import load_scenario, load_scenario_world
from safety_checking.world import tool_names

REPO = Path(__file__).resolve().parents[1]
MODEL_YAML = REPO / "configs" / "models" / "qwen3.8-27b-nothink.yaml"
SMOKE_YAML = REPO / "configs" / "smoke.yaml"

FMT = LogprobFormat(call_opener="<tool_call>\n<function=", name_terminator=">")
CHAT_BASE = 100_000  # chat-prompt token ids start here, well clear of the text vocabulary

# sub-word pieces, so tool names share token prefixes the way they do under a real BPE
PIECES = [
    "<tool_call>", "<function=", "\n", ">", "get", "send", "list", "read", "search", "lookup",
    "create", "_access", "_list", "_update", "_documents", "_document", "_people", "_person",
    "_inbox", "_message", "_calendar", "_event", "_notes", "_note",
]  # fmt: skip
VOCAB = PIECES + sorted({c for c in "abcdefghijklmnopqrstuvwxyz_<>=\n ADIW."} - set(PIECES))
TOKEN_ID = {piece: index for index, piece in enumerate(VOCAB)}


def toy_tokenize(text: str) -> list[int]:
    """Greedy longest match, like a BPE with a fixed merge table."""
    ids, position = [], 0
    by_length = sorted(VOCAB, key=len, reverse=True)
    while position < len(text):
        piece = next(p for p in by_length if text.startswith(p, position))
        ids.append(TOKEN_ID[piece])
        position += len(piece)
    return ids


#: the model's true P(tool name | a call was opened). Sums to 1.
TRUE_NAMES = {
    "get_access_list": 0.90,
    "lookup_person": 0.07,
    "send_update": 0.02,
    "read_document": 0.006,
    "list_documents": 0.002,
    "list_people": 0.001,
    "search_documents": 0.0005,
    "list_inbox": 0.0002,
    "read_message": 0.0001,
    "list_calendar": 0.0001,
    "get_event": 0.00005,
    "create_note": 0.00003,
    "list_notes": 0.00001,
    "read_note": 0.00001,
}
P_CALL_FIRST = 0.97
P_NEWLINE = 0.999
P_FUNCTION = 0.998


class MockServer:
    """Answers /tokenize, /detokenize and /v1/completions like vLLM, from a known model."""

    def __init__(
        self,
        *,
        top_k: int = 20,
        names: dict[str, float] | None = None,
        token_id_keys: bool = True,
        send_mass_by_length: bool = False,
    ) -> None:
        self.top_k = top_k
        self.names = names or TRUE_NAMES
        self.token_id_keys = token_id_keys
        self.send_mass_by_length = send_mass_by_length
        self.requests: list[tuple[str, dict[str, Any]]] = []

    # -- the model -----------------------------------------------------------
    def _names_for(self, n_chat_tokens: int) -> dict[str, float]:
        if not self.send_mass_by_length:
            return self.names
        # longer histories shift a little mass from the check to the send
        shift = min(0.05, n_chat_tokens * 1e-5)
        names = dict(self.names)
        names["get_access_list"] -= shift
        names["send_update"] += shift
        return names

    def _next(self, prompt: list[int]) -> dict[int, float]:
        chat = [t for t in prompt if t >= CHAT_BASE]
        path = [t for t in prompt if t < CHAT_BASE]
        opener = toy_tokenize(FMT.call_opener)
        if len(path) < len(opener):
            expected = opener[len(path)]
            p = (P_CALL_FIRST, P_NEWLINE, P_FUNCTION)[len(path)]
            if path != opener[: len(path)]:
                return {TOKEN_ID["."]: 1.0}
            rest = {TOKEN_ID["D"]: (1 - p) * 0.7, TOKEN_ID["I"]: (1 - p) * 0.3}
            return {expected: p, **rest}
        inside = path[len(opener) :]
        names = self._names_for(len(chat))
        sequences = {n: toy_tokenize(f"{n}{FMT.name_terminator}") for n in names}
        alive = {n: m for n, m in names.items() if sequences[n][: len(inside)] == inside}
        total = sum(alive.values())
        dist: dict[int, float] = {}
        for name, mass in alive.items():
            if len(sequences[name]) > len(inside):
                token = sequences[name][len(inside)]
                dist[token] = dist.get(token, 0.0) + mass / total
        return dist or {TOKEN_ID["."]: 1.0}

    # -- the endpoints -------------------------------------------------------
    def __call__(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((url, payload))
        if url.endswith("/tokenize"):
            if "messages" in payload:
                n = sum(len(json.dumps(m)) for m in payload["messages"]) // 4
                return {"tokens": [CHAT_BASE + i for i in range(n)], "count": n}
            return {"tokens": toy_tokenize(payload["prompt"])}
        if url.endswith("/detokenize"):
            return {"prompt": "".join(VOCAB[t] for t in payload["tokens"])}
        if url.endswith("/v1/completions"):
            dist = self._next(payload["prompt"])
            top = sorted(dist.items(), key=lambda item: -item[1])[: self.top_k]
            key = (lambda t: f"token_id:{t}") if self.token_id_keys else (lambda t: VOCAB[t])
            return {
                "choices": [{"logprobs": {"top_logprobs": [{key(t): math.log(p) for t, p in top}]}}]
            }
        raise AssertionError(f"unexpected POST {url}")

    def completions(self) -> list[dict[str, Any]]:
        return [payload for url, payload in self.requests if url.endswith("/v1/completions")]


def model() -> ModelConfig:
    return load_model_config(MODEL_YAML)


def run_measure(server: MockServer, **kwargs: Any) -> dict[str, Any]:
    client = ServerClient(model(), post_json=server)
    base = [CHAT_BASE + i for i in range(50)]
    return measure(client, base, FMT, "get_access_list", "send_update", **kwargs)


# -- the arithmetic ----------------------------------------------------------


def test_toy_tokenizer_gives_names_shared_prefixes() -> None:
    assert toy_tokenize("list_notes>")[0] == toy_tokenize("list_inbox>")[0] == TOKEN_ID["list"]
    assert toy_tokenize("read_note>") != toy_tokenize("read_notes>")[:3]
    assert set(TRUE_NAMES) == set(tool_names())
    assert sum(TRUE_NAMES.values()) == pytest.approx(1.0)


def test_recovers_the_true_name_distribution_exactly() -> None:
    result = run_measure(MockServer())
    for name, truth in TRUE_NAMES.items():
        entry = result["p_name"][name]
        assert entry.bounded is False
        assert entry.p == pytest.approx(truth, rel=1e-9), name
    assert result["p_check_given_call"] == pytest.approx(0.90, rel=1e-9)
    assert result["p_send_given_call"] == pytest.approx(0.02, rel=1e-9)
    assert result["unaccounted_mass"] == pytest.approx(0.0, abs=1e-9)


def test_reads_the_first_token_and_the_rest_of_the_opener() -> None:
    result = run_measure(MockServer())
    assert result["p_call_first"] == pytest.approx(P_CALL_FIRST)
    assert result["p_call_first_bounded"] is False
    assert result["p_opener_completion"] == pytest.approx(P_NEWLINE * P_FUNCTION)
    top = result["first_token_top"]
    assert top[0]["text"] == "<tool_call>"
    assert [entry["text"] for entry in top[1:]] == ["D", "I"]  # what it would say instead


def test_a_small_send_probability_is_resolved_where_sampling_could_not() -> None:
    """The reason the mode exists: 1e-4 is invisible at n=50 and exact here."""
    names = dict(TRUE_NAMES) | {"send_update": 1e-4, "get_access_list": 0.9199}
    result = run_measure(MockServer(names=names))
    assert result["p_send_given_call"] == pytest.approx(1e-4, rel=1e-9)
    assert (1 - 1e-4) ** 50 > 0.99  # fifty samples would almost surely all check


def test_one_request_per_trie_node_not_per_name_token() -> None:
    server = MockServer()
    run_measure(server)
    prompts = [tuple(p["prompt"]) for p in server.completions()]
    assert len(prompts) == len(set(prompts))  # no node asked twice
    name_tokens = sum(len(toy_tokenize(f"{n}>")) for n in TRUE_NAMES)
    assert len(prompts) < name_tokens  # shared prefixes are shared requests


def test_requests_are_single_token_with_neutral_sampling_and_token_id_prompts() -> None:
    server = MockServer()
    run_measure(server)
    for payload in server.completions():
        assert payload["max_tokens"] == 1
        assert payload["logprobs"] == 20
        assert payload["return_tokens_as_token_ids"] is True
        assert all(isinstance(t, int) for t in payload["prompt"])
        for key, value in NEUTRAL_SAMPLING.items():
            assert payload[key] == value
        assert "presence_penalty" not in payload  # the run's sampling settings do not leak in


def test_names_outside_the_top_k_are_bounded_not_invented() -> None:
    result = run_measure(MockServer(top_k=2))
    assert result["p_name"]["get_access_list"].bounded is False
    assert result["p_name"]["get_access_list"].p == pytest.approx(0.90, rel=1e-9)
    bounded = [n for n, e in result["p_name"].items() if e.bounded]
    assert "create_note" in bounded
    for name in bounded:
        entry = result["p_name"][name]
        assert entry.p == 0.0
        assert entry.p_upper >= TRUE_NAMES[name]  # a true upper bound
    assert result["unaccounted_mass"] > 0


def test_a_name_that_is_a_token_prefix_of_another_is_refused() -> None:
    client = ServerClient(model(), post_json=MockServer())
    no_terminator = LogprobFormat(call_opener=FMT.call_opener, name_terminator="")
    # "get" tokenizes to [get], "get_event" to [get, _event]: with no terminator the first is a
    # strict token prefix of the second, and its probability would swallow the other's
    with pytest.raises(LogprobError, match="prefix of another"):
        measure(client, [CHAT_BASE], no_terminator, "get", "get_event",
                names=["get", "get_event"])  # fmt: skip
    # with the terminator scored as part of the name, the same pair is fine
    ok = measure(client, [CHAT_BASE], FMT, "get_access_list", "send_update")
    assert ok["p_name"]["list_notes"].p > 0


def test_a_server_that_returns_token_strings_is_an_error_not_a_guess() -> None:
    with pytest.raises(LogprobError, match="return_tokens_as_token_ids"):
        run_measure(MockServer(token_id_keys=False))


def test_common_prefix() -> None:
    assert common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2]
    assert common_prefix([[1], [2]]) == []
    assert common_prefix([]) == []


# -- probe points ------------------------------------------------------------


def cell(name: str, length: int = 5):
    scenario = load_scenario(name)
    return scenario, build_prefix(scenario, load_scenario_world(scenario), length)


def test_the_start_probe_is_the_prefix_plus_the_decision_request() -> None:
    scenario, prefix = cell("sharing_risky")
    messages, calls = probe_messages(scenario, prefix, ProbePoint(id="start"))
    assert calls == []
    assert messages[:-1] == prefix.messages
    assert messages[-1] == {"role": "user", "content": scenario.decision_request}


def test_the_after_lookup_probe_executes_the_lookup_for_real() -> None:
    _, smoke_model = load_experiment_config(SMOKE_YAML)
    experiment, _ = load_experiment_config(SMOKE_YAML)
    probe = next(p for p in experiment.probe_points if p.id == "after_lookup")

    for name, first_name, person in [
        ("sharing_risky", "Dani", "p_rivera"),
        ("sharing_benign", "Wei", "p_chen"),
    ]:
        scenario, prefix = cell(name)
        messages, calls = probe_messages(scenario, prefix, probe)
        assert calls == [{"tool": "lookup_person", "args": {"query": first_name}}]
        call, result = messages[-2], messages[-1]
        assert call["tool_calls"][0]["function"]["name"] == "lookup_person"
        assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"query": first_name}
        assert result["role"] == "tool"
        assert result["tool_call_id"] == call["tool_calls"][0]["id"]
        assert json.loads(result["content"])["data"]["matches"][0]["id"] == person
    assert smoke_model.logprob is not None


def test_a_probe_call_that_fails_is_an_error() -> None:
    scenario, prefix = cell("sharing_risky")
    bad = ProbePoint(id="bad", calls=[{"tool": "read_document", "args": {"document_id": "nope"}}])
    with pytest.raises(LogprobError, match="failed"):
        probe_messages(scenario, prefix, bad)


def test_probe_ids_are_stable_and_sensitive() -> None:
    base = compute_probe_id("p", "m", "start", [], FMT)
    assert base == compute_probe_id("p", "m", "start", [], FMT)
    other_format = LogprobFormat(call_opener="<tool_call>\n{", name_terminator='"')
    variants = {
        compute_probe_id("q", "m", "start", [], FMT),
        compute_probe_id("p", "n", "start", [], FMT),
        compute_probe_id("p", "m", "after_lookup", [], FMT),
        compute_probe_id("p", "m", "start", [{"tool": "lookup_person", "args": {}}], FMT),
        compute_probe_id("p", "m", "start", [], other_format),
    }
    assert base not in variants and len(variants) == 5


# -- configs -----------------------------------------------------------------


def test_the_logprob_section_does_not_move_the_model_hash() -> None:
    """Sampled runs were logged under this hash; the format is not a serving choice."""
    config = model()
    assert config.logprob == FMT
    without = ModelConfig.model_validate(config.model_dump() | {"logprob": None})
    assert without.adapter_name == config.adapter_name


def test_probe_points_default_to_the_decision_start_and_smoke_adds_after_lookup() -> None:
    bare = ExperimentConfig(experiment="x", model="m.yaml", scenarios=["s"], lengths=[5],
                            n_samples=1)  # fmt: skip
    assert [p.id for p in bare.probe_points] == ["start"]
    smoke, _ = load_experiment_config(SMOKE_YAML)
    assert [p.id for p in smoke.probe_points] == ["start", "after_lookup"]


# -- end to end --------------------------------------------------------------


def test_run_logprob_measures_every_cell_and_probe_and_resumes(tmp_path) -> None:
    experiment, config = load_experiment_config(SMOKE_YAML)
    server = MockServer(send_mass_by_length=True)
    log = tmp_path / "smoke.logprob.jsonl"

    first = run_logprob(experiment, config, log, post_json=server, server={"vllm_version": "x"})
    assert (first.n_new, first.n_skipped) == (12, 0)  # 2 scenarios x 3 lengths x 2 probes
    n_requests = len(server.requests)

    again = run_logprob(experiment, config, log, post_json=server)
    assert (again.n_new, again.n_skipped) == (0, 12)
    assert len(server.requests) == n_requests  # nothing re-measured

    records = read_logprob_records(log)
    assert len({r.probe_id for r in records}) == 12
    assert {r.probe_point for r in records} == {"start", "after_lookup"}
    assert all(r.model == config.adapter_name for r in records)
    assert all(r.server == {"vllm_version": "x"} for r in records)

    # the mock leaks send mass with history length; the mode must see it, monotonically
    risky_start = sorted(
        (r for r in records if r.scenario_id == "sharing_risky" and r.probe_point == "start"),
        key=lambda r: r.length,
    )
    sends = [r.p_send_given_call for r in risky_start]
    assert sends == sorted(sends) and sends[0] < sends[-1]
    assert [r.n_prompt_tokens for r in risky_start] == sorted(
        r.n_prompt_tokens for r in risky_start
    )

    # the after_lookup probe really is a longer prompt than the start probe of the same cell
    by_key = {(r.scenario_id, r.length, r.probe_point): r for r in records}
    start, later = by_key["sharing_risky", 5, "start"], by_key["sharing_risky", 5, "after_lookup"]
    assert later.n_prompt_tokens > start.n_prompt_tokens
    assert later.prefix_hash == start.prefix_hash


def test_a_model_config_without_the_format_is_refused(tmp_path) -> None:
    experiment, config = load_experiment_config(SMOKE_YAML)
    bare = config.model_copy(update={"logprob": None})
    with pytest.raises(LogprobError, match="no `logprob` section"):
        run_logprob(experiment, bare, tmp_path / "x.jsonl", post_json=MockServer())


def test_sc_logprob_command_runs_prints_a_table_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SC_OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setattr("safety_checking.logprob._http_post_json", MockServer())
    monkeypatch.setattr("safety_checking.render._http_post_json", MockServer())

    assert cli.main(["logprob", "--config", str(SMOKE_YAML)]) == 0
    out = capsys.readouterr().out
    assert "12 new, 0 skipped" in out
    assert "p_send|call" in out
    assert "sharing_risky\t50\tnone\tafter_lookup" in out
    assert "2.000e-02" in out  # the mock's true send mass, in scientific notation
    assert logprob_log_path("smoke").exists()

    assert cli.main(["logprob", "--config", str(SMOKE_YAML), "--table"]) == 0
    assert capsys.readouterr().out.count("sharing_") == 12

    assert cli.main(["logprob", "--config", str(SMOKE_YAML)]) == 0
    assert "0 new, 12 skipped" in capsys.readouterr().out


def test_sc_logprob_refuses_a_remote_server(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SC_OUTPUTS_DIR", str(tmp_path))
    code = cli.main(["logprob", "--config", str(SMOKE_YAML), "--base-url", "https://x.example/v1"])
    assert code == 2
    assert "localhost" in capsys.readouterr().out
