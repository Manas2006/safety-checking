"""Token counting.

One model-agnostic yardstick (tiktoken ``o200k_base``) used to match token counts across
conditions; it is not a billing estimate. Provider-reported usage is recorded separately at
run time.

The encoding is read from ``TIKTOKEN_CACHE_DIR`` (populated once at setup). If it is not
there, counting falls back to ``len(text) / 4`` and says so, so tests never need the network.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from pydantic import BaseModel

from .canonical import canonical_json
from .paths import tiktoken_cache_dir

ENCODING_NAME = "o200k_base"
#: role/framing overhead charged per message, the usual chat-format approximation
PER_MESSAGE_OVERHEAD = 3


class TokenCount(BaseModel):
    tokens: int
    #: True when the tiktoken encoding was unavailable and chars/4 was used instead
    estimated: bool


@functools.lru_cache(maxsize=1)
def _encoding() -> Any | None:
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tiktoken_cache_dir()))
    try:
        import tiktoken

        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception:
        return None


def count_text(text: str) -> TokenCount:
    encoding = _encoding()
    if encoding is None:
        return TokenCount(tokens=len(text) // 4, estimated=True)
    return TokenCount(tokens=len(encoding.encode(text)), estimated=False)


def count_messages(messages: list[dict[str, Any]]) -> TokenCount:
    """Total for a chat-format message list: content + tool calls + per-message overhead."""
    total = 0
    estimated = False
    for message in messages:
        pieces: list[str] = [str(message.get("content") or "")]
        if message.get("tool_calls"):
            pieces.append(canonical_json(message["tool_calls"]))
        for piece in pieces:
            counted = count_text(piece)
            total += counted.tokens
            estimated = estimated or counted.estimated
        total += PER_MESSAGE_OVERHEAD
    return TokenCount(tokens=total, estimated=estimated)
