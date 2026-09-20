# safety-checking

Do long-horizon tool-using agents keep executing a required safety check before a consequential
action as the session grows, when the policy stays visible in context the whole time?

A history of benign tool calls is built once, frozen, and replayed as a prefix. Only the decision
segment is sampled from the model. The last five tool calls before the decision are identical at
every history length, so the decision point is matched and only the amount of prior work varies.

See `SPEC.md` for the architecture and the design decisions, `PREREG.md` for the analysis plan,
and `CLAUDE.md` for environment rules and conventions.

## Setup

```bash
uv sync                 # creates .venv with Python 3.12
uv run pytest           # unit tests, seconds, no network
uv run ruff check .
```

## Use

```bash
uv run sc tools                      # list the world's tool specs
uv run sc build-prefixes             # build and save frozen prefixes
uv run sc run --dry-run              # cell counts and a token estimate, no calls
uv run sc run --model fake:always_check --experiment smoke
uv run sc show outputs/runs/smoke.jsonl --index 0
uv run sc counts outputs/runs/smoke.jsonl   # outcome counts per cell, never rates
```

`./outputs` is a symlink to `$WORK/safety-checking-outputs`.

No API calls happen by accident: `sc run` refuses any model that is not `fake:*` unless you pass
`--confirm-paid`, and `--dry-run` works for real model specs without a key. Runs are resumable;
re-running a command skips every sample already in the log.

On the TACC login node, prefix commands with `RAYON_NUM_THREADS=1` for ruff and
`UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1` for `uv sync`; the
8 GB virtual-memory cap otherwise aborts them. Real model runs belong on a compute node.
