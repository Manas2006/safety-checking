"""Model adapters.

The adapter is the only non-deterministic part of the system, so it is kept as thin as
possible: chat-format messages and tool specs in, one assistant turn out.

Nothing here is imported at test time in a way that touches the network. The OpenAI client is
created lazily, and tests inject a mock.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..canonical import canonical_json
from ..config import ModelConfig, is_local_url
from ..scenarios.schema import Scenario
from ..trajectory import Usage

Message = dict[str, Any]


class AdapterResponse(BaseModel):
    """One assistant turn, in OpenAI chat format."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    #: text the server's reasoning parser split off (vLLM: reasoning / reasoning_content)
    reasoning: str | None = None
    #: [{"id", "type": "function", "function": {"name", "arguments": "<json string>"}}]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    """What the runner needs from a model.

    ``name`` goes into the run id, so it must identify the model and anything about the
    endpoint that changes behaviour. Two optional methods, called by the runner when present:
    ``bind(scenario)`` (only the fake models need it) and ``server_info()`` (what the server
    says about itself, recorded in every trajectory).
    """

    name: str

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]], params: dict[str, Any]
    ) -> AdapterResponse: ...


# -- fake models, for tests and dry plumbing ----------------------------------

FakePolicy = Literal["always_check", "never_check"]


def _decision_calls(messages: list[Message]) -> list[tuple[str, dict[str, Any], str]]:
    """(tool, args, result) for every call made since the last user message."""
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    results = {
        m["tool_call_id"]: m.get("content") or ""
        for m in messages[last_user:]
        if m.get("role") == "tool"
    }
    calls = []
    for message in messages[last_user:]:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            calls.append(
                (function["name"], json.loads(function["arguments"]), results.get(call["id"], ""))
            )
    return calls


class FakeModel:
    """A scripted policy. Deterministic, free, and enough to test the whole pipeline.

    * ``always_check``: inspects the access list, then sends only if the recipient is on it,
      otherwise tells the user.
    * ``never_check``: sends straight away.
    """

    def __init__(self, policy: FakePolicy) -> None:
        if policy not in ("always_check", "never_check"):
            raise ValueError(f"unknown fake policy: {policy}")
        self.policy: FakePolicy = policy
        self.name = f"fake:{policy}"
        self._scenario: Scenario | None = None

    def bind(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def _call(self, n: int, tool: str, args: dict[str, Any]) -> AdapterResponse:
        return AdapterResponse(
            tool_calls=[
                {
                    "id": f"call_d{n:03d}",
                    "type": "function",
                    "function": {"name": tool, "arguments": canonical_json(args)},
                }
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, cached_tokens=0),
            finish_reason="tool_calls",
        )

    @staticmethod
    def _say(text: str) -> AdapterResponse:
        return AdapterResponse(
            content=text,
            usage=Usage(prompt_tokens=0, completion_tokens=0, cached_tokens=0),
            finish_reason="stop",
        )

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]], params: dict[str, Any]
    ) -> AdapterResponse:
        if self._scenario is None:
            raise RuntimeError("FakeModel.bind(scenario) must be called before generate()")
        scenario = self._scenario
        target = scenario.target_document
        recipient = scenario.requested_recipient
        done = _decision_calls(messages)
        tools_called = [tool for tool, _, _ in done]
        send_args = {
            "document_id": target,
            "recipient": recipient,
            "message": "Here is the latest update.",
        }

        if "send_update" in tools_called:
            return self._say("Sent the update.")

        if self.policy == "never_check":
            return self._call(len(done) + 1, "send_update", send_args)

        if "get_access_list" not in tools_called:
            return self._call(len(done) + 1, "get_access_list", {"document_id": target})

        _, _, result = next(c for c in done if c[0] == "get_access_list")
        access = json.loads(result).get("data", {}).get("access", [])
        if recipient in {entry.get("person_id") for entry in access}:
            return self._call(len(done) + 1, "send_update", send_args)
        return self._say(
            f"{recipient} is not on the access list for {target}, so I have not sent it. "
            "Do you want to add them first?"
        )


# -- OpenAI-compatible --------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Retry rate limits, timeouts, dropped connections and 5xx. Never retry a 4xx."""
    names = {type(exc).__name__, *(base.__name__ for base in type(exc).__mro__)}
    return bool(
        names & {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}
    )


def _http_get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatAdapter:
    """Chat-completions adapter for OpenAI and anything that speaks its API (vLLM, etc.).

    ``base_url`` is part of ``name`` when set, because the same model string behind a
    different server is a different model as far as a run id is concerned. ``name`` can be
    given outright; config-driven vLLM runs use the model config's id and content hash.

    Non-standard request parameters (``top_k``, ``min_p``, ``chat_template_kwargs``) travel
    under ``params["extra_body"]``, which the OpenAI SDK merges into the request JSON.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        max_attempts: int = 5,
        name: str | None = None,
        http_get_json: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._api_key = api_key
        self._client = client
        self._max_attempts = max_attempts
        self._http_get_json = http_get_json or _http_get_json
        self._server_info: dict[str, Any] | None = None
        self.name = name or (f"openai:{model}" + (f"@{self.base_url}" if self.base_url else ""))

    @classmethod
    def from_model_config(
        cls, config: ModelConfig, *, base_url: str | None = None, **kwargs: Any
    ) -> OpenAICompatAdapter:
        return cls(
            config.served_model_name,
            base_url=base_url or config.base_url,
            name=config.adapter_name,
            **kwargs,
        )

    @property
    def is_local(self) -> bool:
        return is_local_url(self.base_url)

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # imported lazily: never needed by tests

            # a local vLLM server ignores the key, but the SDK insists on a non-empty one
            key = (
                self._api_key
                or os.environ.get("OPENAI_API_KEY")
                or ("EMPTY" if self.is_local else None)
            )
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def server_info(self) -> dict[str, Any]:
        """Served model name and vLLM version, asked once. Only a local server is asked.

        Never raises: whatever could not be found out is recorded as None.
        """
        if self._server_info is None:
            info: dict[str, Any] = {"base_url": self.base_url, "requested_model": self.model}
            if self.is_local:
                root = (self.base_url or "").rstrip("/").removesuffix("/v1")
                try:
                    models = self._http_get_json(f"{root}/v1/models")
                    info["served_models"] = [m.get("id") for m in models.get("data", [])]
                except Exception as exc:
                    info["served_models"] = None
                    info["models_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    info["vllm_version"] = self._http_get_json(f"{root}/version").get("version")
                except Exception as exc:
                    info["vllm_version"] = None
                    info["version_error"] = f"{type(exc).__name__}: {exc}"
            self._server_info = info
        return dict(self._server_info)

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]], params: dict[str, Any]
    ) -> AdapterResponse:
        @retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )
        def call() -> Any:
            return self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, **params
            )

        return self._parse(call())

    @staticmethod
    def _parse(response: Any) -> AdapterResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ]
        usage = None
        if getattr(response, "usage", None) is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            usage = Usage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                completion_tokens=getattr(response.usage, "completion_tokens", None),
                cached_tokens=getattr(details, "cached_tokens", None) if details else None,
            )
        # vLLM renamed reasoning_content to reasoning; accept either
        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )
        return AdapterResponse(
            content=message.content,
            reasoning=reasoning if isinstance(reasoning, str) else None,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )


# -- Anthropic: stub ----------------------------------------------------------


class AnthropicAdapter:
    """Not implemented yet. Needs the chat-format <-> Messages API translation
    (system prompt lifted out, tool_calls -> tool_use blocks, tool -> tool_result blocks).
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"anthropic:{model}"

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]], params: dict[str, Any]
    ) -> AdapterResponse:
        raise NotImplementedError("the Anthropic adapter is a stub; see SPEC.md 3.7")


def make_adapter(spec: str, **kwargs: Any) -> ModelAdapter:
    """``fake:always_check`` | ``fake:never_check`` | ``openai:<model>`` | ``anthropic:<model>``."""
    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"adapter spec must look like provider:model, got {spec!r}")
    if provider == "fake":
        return FakeModel(rest)  # type: ignore[arg-type]
    if provider == "openai":
        return OpenAICompatAdapter(rest, **kwargs)
    if provider == "anthropic":
        return AnthropicAdapter(rest)
    raise ValueError(f"unknown provider: {provider}")
