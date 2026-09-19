"""Test-wide guards: no network, and a deterministic tokenizer cache location."""

from __future__ import annotations

import os
import socket

import pytest

from safety_checking.paths import tiktoken_cache_dir

# tiktoken reads its encoding from this cache; it is populated once at setup, so tests
# never download anything. If it is missing, token counting falls back to chars/4.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tiktoken_cache_dir()))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if library code tries to open a socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests must not touch the network")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
