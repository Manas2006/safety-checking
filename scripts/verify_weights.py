#!/usr/bin/env python3
"""Check that a model's weights are completely downloaded.

    .venv/bin/python scripts/verify_weights.py configs/models/qwen3.8-27b-nothink.yaml

Complete means: the pinned snapshot exists in the HF cache, every shard named in
``model.safetensors.index.json`` (or the one ``model.safetensors`` of a repo without an index) is
present, each has the size the Hub reports for it (or, if the
Hub cannot be reached, a plausible size), nothing is left half-written, and the tokenizer and
config files are there. Standard library plus the project's config loader only: safe on the
login node, imports nothing heavy.

Exit status 0 when complete, 1 when not.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from safety_checking.config import load_any

REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
#: without the Hub's sizes, a shard under this is assumed truncated
MIN_PLAUSIBLE_SHARD_BYTES = 500 * 1024**2


def hub_sizes(repo: str, revision: str) -> dict[str, int] | None:
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"note: could not ask the Hub for file sizes ({type(exc).__name__}: {exc})")
        return None
    return {s["rfilename"]: s["size"] for s in payload.get("siblings", []) if "size" in s}


def hub_cache() -> Path:
    """Where huggingface_hub keeps models: HF_HUB_CACHE if set, else $HF_HOME/hub.

    Not the same thing: a shell that exports HF_HUB_CACHE=$HF_HOME has no hub/ level at all.
    """
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    _, model = load_any(sys.argv[1])
    repo_dir = hub_cache() / f"models--{model.hf_repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / (model.revision or "")
    print(f"repo      {model.hf_repo} @ {model.revision}")
    print(f"snapshot  {snapshot}")

    problems: list[str] = []
    if not model.revision:
        problems.append("the model config pins no revision")
    if not snapshot.is_dir():
        print("INCOMPLETE: the snapshot directory does not exist yet")
        return 1

    # A blob is written as <sha>.<random>.incomplete and renamed to <sha> when it finishes. An
    # interrupted attempt leaves its .incomplete behind even after a later attempt completes
    # the same blob, so a leftover is only a problem when the finished blob is absent.
    stale = 0
    for path in sorted((repo_dir / "blobs").glob("*.incomplete")):
        if (path.parent / path.name.split(".")[0]).exists():
            stale += 1
        else:
            problems.append(f"half-written blob: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    if stale:
        print(f"note      {stale} stale .incomplete file(s) from an interrupted attempt; deletable")

    expected = hub_sizes(model.hf_repo, model.revision or "main")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    elif expected is not None and "model.safetensors.index.json" not in expected:
        # a model small enough for one file has no index (google/gemma-4-12B-it)
        index = {}
        shards = ["model.safetensors"]
    else:
        print("INCOMPLETE: model.safetensors.index.json is not there yet")
        return 1

    total = 0
    present = 0
    for name in shards:
        path = snapshot / name
        if not path.exists():  # a dangling symlink counts as missing
            problems.append(f"missing shard: {name}")
            continue
        size = path.stat().st_size  # follows the symlink into blobs/
        total += size
        present += 1
        if expected is not None and name in expected:
            if size != expected[name]:
                problems.append(
                    f"wrong size: {name} is {size:,} bytes, Hub says {expected[name]:,}"
                )
        elif size < MIN_PLAUSIBLE_SHARD_BYTES:
            problems.append(f"implausibly small: {name} is {size:,} bytes")

    tensor_bytes = index.get("metadata", {}).get("total_size")
    print(f"shards    {present}/{len(shards)} present, {total / 1e9:.2f} GB on disk")
    if tensor_bytes:
        print(f"index     total_size {tensor_bytes / 1e9:.2f} GB of tensors")
        # files carry a small JSON header each, so disk >= tensors, and only slightly more
        if present == len(shards) and not (tensor_bytes <= total <= tensor_bytes * 1.01):
            problems.append(
                f"shards total {total:,} bytes but the index expects about {tensor_bytes:,}"
            )
    print("sizes     " + ("checked against the Hub" if expected else "plausibility only"))

    for name in REQUIRED_FILES:
        if not (snapshot / name).exists():
            problems.append(f"missing file: {name}")

    if problems:
        print(f"INCOMPLETE: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("COMPLETE: every shard in the index is present at the expected size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
