"""Logprob mode: read the model's next-call distribution instead of counting samples.

Sampling resolves a rate only to about 1/n. If the chance of sending without checking is 0.1%
after 5 calls and 1% after 50, fifty samples per cell cannot see it, and a tenfold rise in the
miss rate is exactly what this project is looking for. The model's own distribution can.

At a *probe point* (the decision start, or a scripted continuation of it) this reads:

* ``p_call_first``: the probability that the next token opens a tool call;
* ``p_name[tool]``: given that a call has been opened, the probability of each tool name.

The headline is ``p_send_given_call``: the mass on sending right now. It is deliberately not
"the first call is the check", because a model that looks the recipient up first is safe and
would score badly on that; the smoke run showed most risky samples do exactly this.

**How.** Tool names are several tokens long, so each name's probability is a product along its
token path. The paths form a trie; each node costs one 1-token completion with the top-20 next
tokens, and all of them share the cached history. (Scoring with ``prompt_logprobs`` would be
fewer requests, but vLLM bypasses the prefix cache for those, re-running a 9k-token prefill per
candidate.) A token outside the top 20 cannot be read, only bounded by the 20th; such names are
reported as bounded, which in practice means negligible.

**What it is not.** It is the raw distribution at temperature 1, not the sampled behaviour at
temperature 0.7 with top-p and top-k: those apply per token and do not compose into a closed
form over a multi-token name, so no correction is attempted. Requests always carry neutral
sampling parameters, which makes the result independent of the server's ``--logprobs-mode``.
It sees one next call, not a trajectory. The sampled ``outcome`` stays the primary measure, and
this must track sampled frequencies on the gate cells before it is trusted (SPEC.md 3.10).

Talks to the server over HTTP with the standard library. Imports nothing heavy.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .canonical import canonical_json, sha256_of
from .config import ExperimentConfig, LogprobFormat, ModelConfig, ProbePoint
from .history.builder import Prefix, build_prefix
from .paths import runs_dir
from .render import PostJson, _http_post_json, server_root, tokenize_chat
from .scenarios.loader import load_scenario, load_scenario_world
from .scenarios.schema import Scenario
from .world.registry import execute, tool_names
from .world.state import WorldState

LOGPROB_VERSION = 1
TOP_LOGPROBS = 20  # vLLM's default --max-logprobs

#: neutral sampling: with these the processed distribution equals the raw one
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}

Message = dict[str, Any]


class LogprobError(RuntimeError):
    """The server answered in a way this mode cannot use."""


class NameProbability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: probability of this name given an opened call; 0.0 when only a bound is known
    p: float
    #: True when some token on the path fell outside the top-k, so only ``p_upper`` is known
    bounded: bool = False
    p_upper: float | None = None
    n_tokens: int


class LogprobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str
    prefix_hash: str
    scenario_id: str
    variant: str
    length: int
    prior_check_pattern: str
    model: str
    probe_point: str
    #: the scripted continuation that defines the probe point, after substitution
    probe_calls: list[dict[str, Any]] = Field(default_factory=list)

    n_prompt_tokens: int
    #: P(next token opens a tool call). A lower bound on "this turn contains a call".
    p_call_first: float
    p_call_first_bounded: bool = False
    #: P(rest of the call opener | its first token): should be about 1
    p_opener_completion: float
    #: the most likely first tokens, decoded, for reading what the model wants to do instead
    first_token_top: list[dict[str, Any]] = Field(default_factory=list)

    p_name: dict[str, NameProbability]
    #: the required check, and the consequential action, given that a call is opened
    p_check_given_call: float
    p_send_given_call: float
    #: 1 - sum of the names that could be read exactly. Large means a suspect measurement.
    unaccounted_mass: float

    n_requests: int
    elapsed_s: float
    server: dict[str, Any] = Field(default_factory=dict)
    logprob_version: int = LOGPROB_VERSION


# -- talking to the server ---------------------------------------------------


class ServerClient:
    """The three calls logprob mode needs. ``post_json`` is injectable for tests."""

    def __init__(
        self, config: ModelConfig, *, base_url: str | None = None, post_json: PostJson | None = None
    ) -> None:
        self.config = config
        self.base_url = base_url
        self.post = post_json or _http_post_json
        self.root = server_root(config, base_url)
        self.n_requests = 0

    def tokenize_chat(self, messages: list[Message]) -> list[int]:
        self.n_requests += 1
        return tokenize_chat(messages, self.config, base_url=self.base_url, post_json=self.post)

    def tokenize_text(self, text: str) -> list[int]:
        self.n_requests += 1
        payload = {
            "model": self.config.served_model_name,
            "prompt": text,
            "add_special_tokens": False,
        }
        return list(self.post(f"{self.root}/tokenize", payload)["tokens"])

    def decode(self, token_id: int) -> str:
        self.n_requests += 1
        payload = {"model": self.config.served_model_name, "tokens": [token_id]}
        return str(self.post(f"{self.root}/detokenize", payload)["prompt"])

    def next_token_logprobs(self, prompt_ids: list[int]) -> dict[int, float]:
        """Top next-token log-probabilities after ``prompt_ids``, keyed by token id."""
        self.n_requests += 1
        payload = {
            "model": self.config.served_model_name,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "logprobs": TOP_LOGPROBS,
            "return_tokens_as_token_ids": True,
            "seed": 0,
            **NEUTRAL_SAMPLING,
        }
        response = self.post(f"{self.root}/v1/completions", payload)
        try:
            top = response["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LogprobError(f"no top_logprobs in the completion response: {exc}") from exc
        parsed: dict[int, float] = {}
        for key, value in top.items():
            if not str(key).startswith("token_id:"):
                raise LogprobError(
                    f"expected token ids in top_logprobs, got {key!r}; the server ignored "
                    "return_tokens_as_token_ids"
                )
            parsed[int(str(key).split(":", 1)[1])] = float(value)
        return parsed


# -- the measurement ---------------------------------------------------------


def common_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []
    prefix = []
    for column in zip(*sequences, strict=False):
        if len(set(column)) != 1:
            break
        prefix.append(column[0])
    return prefix


class _TrieWalker:
    """Reads next-token distributions along token paths, one request per distinct node."""

    def __init__(self, client: ServerClient, base_ids: list[int]) -> None:
        self.client = client
        self.base_ids = base_ids
        self._cache: dict[tuple[int, ...], dict[int, float]] = {}

    def distribution(self, path: tuple[int, ...]) -> dict[int, float]:
        if path not in self._cache:
            self._cache[path] = self.client.next_token_logprobs([*self.base_ids, *path])
        return self._cache[path]

    def score(self, start: tuple[int, ...], tokens: list[int]) -> tuple[float, bool]:
        """Sum of log-probabilities of ``tokens`` after ``start``.

        Returns (logprob, bounded). When a token is outside the top-k the walk stops there:
        the value returned is then an upper bound, using the smallest logprob that was seen.
        """
        total = 0.0
        path = start
        for token in tokens:
            dist = self.distribution(path)
            if token not in dist:
                return total + min(dist.values()), True
            total += dist[token]
            path = (*path, token)
        return total, False


def measure(
    client: ServerClient,
    base_ids: list[int],
    fmt: LogprobFormat,
    check_tool: str,
    action_tool: str,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """The next-call distribution after ``base_ids``. Pure apart from the server calls."""
    names = names or tool_names()

    # Tokenize opener+name+terminator together, so each name is scored under the tokenization
    # the model would actually produce, not the one the name has in isolation. What the
    # candidates share is the forced opening; where they diverge, the scoring starts. That
    # also copes with a tokenizer that merges the end of the opener into the start of a name.
    full = {
        name: client.tokenize_text(f"{fmt.call_opener}{name}{fmt.name_terminator}")
        for name in names
    }
    forced = common_prefix(list(full.values()))
    if not forced:
        raise LogprobError("the tool names share no common opening tokens")
    suffix = {name: ids[len(forced) :] for name, ids in full.items()}
    if any(not ids for ids in suffix.values()):
        raise LogprobError("a tool name tokenizes to a prefix of another; widen the terminator")

    walker = _TrieWalker(client, base_ids)

    first = walker.distribution(())
    opening_token = forced[0]
    if opening_token in first:
        p_call_first, call_bounded = math.exp(first[opening_token]), False
    else:
        p_call_first, call_bounded = math.exp(min(first.values())), True

    rest_logprob, rest_bounded = walker.score((opening_token,), forced[1:])
    p_opener_completion = 0.0 if rest_bounded else math.exp(rest_logprob)

    p_name: dict[str, NameProbability] = {}
    for name in names:
        logprob, bounded = walker.score(tuple(forced), suffix[name])
        p_name[name] = NameProbability(
            p=0.0 if bounded else math.exp(logprob),
            bounded=bounded,
            p_upper=math.exp(logprob) if bounded else None,
            n_tokens=len(suffix[name]),
        )

    top = sorted(first.items(), key=lambda item: -item[1])[:5]
    first_token_top = [
        {"token_id": token, "text": client.decode(token), "p": math.exp(logprob)}
        for token, logprob in top
    ]
    return {
        "p_call_first": p_call_first,
        "p_call_first_bounded": call_bounded,
        "p_opener_completion": p_opener_completion,
        "first_token_top": first_token_top,
        "p_name": p_name,
        "p_check_given_call": p_name[check_tool].p,
        "p_send_given_call": p_name[action_tool].p,
        "unaccounted_mass": 1.0 - sum(entry.p for entry in p_name.values()),
    }


# -- probe points ------------------------------------------------------------


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace(f"${key}", replacement)
        return value
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    return value


def probe_variables(scenario: Scenario, world: WorldState) -> dict[str, str]:
    recipient = world.person(scenario.requested_recipient)
    return {
        "recipient_first_name": recipient.name.split()[0] if recipient else "",
        "recipient_id": scenario.requested_recipient,
        "target_document": scenario.target_document,
    }


def probe_messages(
    scenario: Scenario, prefix: Prefix, probe: ProbePoint
) -> tuple[list[Message], list[dict[str, Any]]]:
    """Prefix + decision request + the probe point's scripted calls, executed for real."""
    world = prefix.world()
    variables = probe_variables(scenario, world)
    messages: list[Message] = [
        *prefix.messages,
        {"role": "user", "content": scenario.decision_request},
    ]
    resolved: list[dict[str, Any]] = []
    for index, call in enumerate(probe.calls, start=1):
        args = _substitute(call.args, variables)
        call_id = f"call_p{index:03d}"
        world, result = execute(world, call.tool, args)
        if not result.ok:
            raise LogprobError(f"probe {probe.id}: {call.tool}{args} failed: {result.error}")
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.tool, "arguments": canonical_json(args)},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result.to_json()})
        resolved.append({"tool": call.tool, "args": args})
    return messages, resolved


def compute_probe_id(
    prefix_hash: str, model: str, probe: str, calls: list[dict[str, Any]], fmt: LogprobFormat
) -> str:
    return sha256_of(
        {
            "kind": "logprob",
            "version": LOGPROB_VERSION,
            "prefix_hash": prefix_hash,
            "model": model,
            "probe_point": probe,
            "probe_calls": calls,
            "format": fmt.model_dump(),
            "top_logprobs": TOP_LOGPROBS,
        }
    )


# -- running and storing -----------------------------------------------------


def logprob_log_path(experiment: str, directory: Path | None = None) -> Path:
    return (directory or runs_dir()) / f"{experiment}.logprob.jsonl"


def read_logprob_records(path: Path) -> list[LogprobRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [LogprobRecord.model_validate_json(line) for line in handle if line.strip()]


class LogprobSummary(BaseModel):
    n_new: int = 0
    n_skipped: int = 0
    n_requests: int = 0
    elapsed_s: float = 0.0


def run_logprob(
    experiment: ExperimentConfig,
    model: ModelConfig,
    log_path: Path,
    *,
    base_url: str | None = None,
    post_json: PostJson | None = None,
    server: dict[str, Any] | None = None,
) -> LogprobSummary:
    """Measure every (scenario, length, pattern, probe point). Resumable like sampled runs."""
    if model.logprob is None:
        raise LogprobError(f"model config {model.id} has no `logprob` section")
    fmt = model.logprob
    done = {record.probe_id for record in read_logprob_records(log_path)}
    summary = LogprobSummary()
    started = time.perf_counter()

    for name in experiment.scenarios:
        scenario = load_scenario(name)
        world = load_scenario_world(scenario)
        for length in experiment.lengths:
            for pattern in experiment.patterns:
                prefix = build_prefix(scenario, world, length, prior_check_pattern=pattern)
                for probe in experiment.probe_points:
                    messages, calls = probe_messages(scenario, prefix, probe)
                    probe_id = compute_probe_id(
                        prefix.prefix_hash, model.adapter_name, probe.id, calls, fmt
                    )
                    if probe_id in done:
                        summary.n_skipped += 1
                        continue
                    client = ServerClient(model, base_url=base_url, post_json=post_json)
                    began = time.perf_counter()
                    base_ids = client.tokenize_chat(messages)
                    measured = measure(
                        client,
                        base_ids,
                        fmt,
                        check_tool=scenario.required_check.tool,
                        action_tool=scenario.consequential_action.tool,
                    )
                    record = LogprobRecord(
                        probe_id=probe_id,
                        prefix_hash=prefix.prefix_hash,
                        scenario_id=scenario.id,
                        variant=scenario.variant,
                        length=length,
                        prior_check_pattern=pattern,
                        model=model.adapter_name,
                        probe_point=probe.id,
                        probe_calls=calls,
                        n_prompt_tokens=len(base_ids),
                        n_requests=client.n_requests,
                        elapsed_s=time.perf_counter() - began,
                        server=server or {},
                        **measured,
                    )
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(record.model_dump_json() + "\n")
                    done.add(probe_id)
                    summary.n_new += 1
                    summary.n_requests += client.n_requests

    summary.elapsed_s = time.perf_counter() - started
    return summary
