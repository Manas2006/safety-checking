"""Canonical JSON and hashing.

Every hash in this project goes through here. Content addressing only works if the
serialisation is byte-stable, so there is exactly one definition of "canonical".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "sha256_of", "short_hash"]


def canonical_json(obj: Any) -> str:
    """Compact JSON with sorted keys and no ASCII escaping."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(obj: Any) -> str:
    """sha256 of the canonical JSON of ``obj``, as hex."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = 12) -> str:
    """First ``length`` hex characters of :func:`sha256_of`, for filenames and logs."""
    return sha256_of(obj)[:length]
