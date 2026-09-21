"""Render a prefix through the served model's chat template, and check nothing was dropped.

The chat template is where an agent history can silently go wrong: a template that ignores
``tool_calls`` on past assistant turns, or drops ``role: "tool"`` messages, would leave the
model looking at a conversation with holes in it, and every length effect we measured would be
an artefact. So before trusting a model, render one full prefix and read it.

This talks to the running vLLM server (``/tokenize`` then ``/detokenize``), so the text is what
the server itself builds from our messages, tools and chat_template_kwargs. Nothing here imports
torch, transformers or vllm. It runs inside a job, next to the server.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import ModelConfig
from .history.builder import Prefix
from .paths import outputs_dir
from .scenarios.schema import Scenario
from .world.registry import openai_tool_specs

PostJson = Callable[[str, dict[str, Any]], dict[str, Any]]


def _http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


class Expected(BaseModel):
    kind: str
    label: str
    needle: str


class RenderReport(BaseModel):
    ok: bool
    n_expected: int
    n_found_in_order: int
    #: present in the text, but not where the conversation order says it should be
    out_of_order: list[str] = Field(default_factory=list)
    #: not in the text at all
    missing: list[str] = Field(default_factory=list)
    n_tool_calls: int = 0
    n_tool_results: int = 0
    n_prompt_tokens: int | None = None
    thinking_markup_present: bool = False


def expected_items(messages: list[dict[str, Any]]) -> list[Expected]:
    """Everything that must appear in the rendered text, in conversation order."""
    items: list[Expected] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "user") and content:
            items.append(Expected(kind=role, label=f"{role}[{index}]", needle=content))
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call["function"]
                label = f"call {call['id']} {function['name']}"
                items.append(Expected(kind="tool_call", label=label, needle=function["name"]))
                arguments = json.loads(function["arguments"] or "{}")
                for key, value in arguments.items():
                    # string values appear raw in XML-style templates and JSON-escaped-but-
                    # identical in JSON-style ones, as long as they need no escaping
                    if isinstance(value, str):
                        items.append(
                            Expected(kind="tool_arg", label=f"{label} {key}", needle=value)
                        )
            if content:
                items.append(
                    Expected(kind="assistant", label=f"assistant[{index}]", needle=content)
                )
        elif role == "tool":
            label = f"result {message.get('tool_call_id')}"
            items.append(Expected(kind="tool_result", label=label, needle=content or ""))
    return items


def verify_rendering(text: str, messages: list[dict[str, Any]]) -> RenderReport:
    """Walk the text once, left to right, looking for each expected item after the last one."""
    items = expected_items(messages)
    cursor = 0
    found = 0
    missing: list[str] = []
    out_of_order: list[str] = []
    for item in items:
        position = text.find(item.needle, cursor)
        if position >= 0:
            cursor = position + len(item.needle)
            found += 1
        elif item.needle in text:
            out_of_order.append(item.label)
        else:
            missing.append(item.label)
    return RenderReport(
        ok=not missing and not out_of_order,
        n_expected=len(items),
        n_found_in_order=found,
        out_of_order=out_of_order,
        missing=missing,
        n_tool_calls=sum(1 for i in items if i.kind == "tool_call"),
        n_tool_results=sum(1 for i in items if i.kind == "tool_result"),
        thinking_markup_present="<think>" in text,
    )


def server_root(config: ModelConfig, base_url: str | None = None) -> str:
    """The server's root URL, without the OpenAI ``/v1`` suffix."""
    return (base_url or config.base_url).rstrip("/").removesuffix("/v1")


def tokenize_chat(
    messages: list[dict[str, Any]],
    config: ModelConfig,
    *,
    base_url: str | None = None,
    post_json: PostJson | None = None,
) -> list[int]:
    """Token ids of the exact prompt the server builds from these messages.

    Built by the server itself, with our tools and chat_template_kwargs and the generation
    prompt appended, so it is the very sequence a chat completion would continue from.
    """
    post = post_json or _http_post_json
    payload: dict[str, Any] = {
        "model": config.served_model_name,
        "messages": messages,
        "tools": openai_tool_specs(),
        "add_generation_prompt": True,
    }
    kwargs = config.request.extra_body.get("chat_template_kwargs")
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    return list(post(f"{server_root(config, base_url)}/tokenize", payload)["tokens"])


def render_messages(
    messages: list[dict[str, Any]],
    config: ModelConfig,
    *,
    base_url: str | None = None,
    post_json: PostJson | None = None,
) -> tuple[str, int | None]:
    """The exact prompt text the server builds, and its token count."""
    post = post_json or _http_post_json
    tokens = tokenize_chat(messages, config, base_url=base_url, post_json=post)
    detokenized = post(
        f"{server_root(config, base_url)}/detokenize",
        {"model": config.served_model_name, "tokens": tokens},
    )
    return detokenized["prompt"], len(tokens)


def render_prefix(
    prefix: Prefix,
    scenario: Scenario,
    config: ModelConfig,
    *,
    base_url: str | None = None,
    post_json: PostJson | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, RenderReport]:
    """Render prefix + decision request, write the text and a report, return both."""
    messages = [*prefix.messages, {"role": "user", "content": scenario.decision_request}]
    text, n_tokens = render_messages(messages, config, base_url=base_url, post_json=post_json)
    report = verify_rendering(text, messages)
    report.n_prompt_tokens = n_tokens

    out_dir = out_dir or outputs_dir() / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario.id}.L{prefix.length}.{prefix.prior_check_pattern}.{config.id}"
    text_path = out_dir / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return text_path, report
