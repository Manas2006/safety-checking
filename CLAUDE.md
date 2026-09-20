# CLAUDE.md

Research codebase measuring whether long-horizon tool-using agents keep executing a required
safety check before a consequential action as the session grows, with the policy visible in
context the whole time. Read `SPEC.md` before changing anything structural; it carries the design
decisions and their reasons.

## Environment rules (hard constraints)

- This is a **TACC Lonestar6 login node**, a shared machine. Never run anything heavy here: no
  model inference, no long loops, no many-worker parallel jobs. Unit tests that finish in seconds
  are fine. Real runs belong in a job on a compute node.
- **No sudo. No conda** (it was removed on purpose; do not reinstall it, and do not use the
  miniconda still present under `$WORK`). Use `uv`, Python 3.12.
- `$HOME` has a **10 GB quota**, so this folder holds code only. Large outputs go to
  `$WORK/safety-checking-outputs`, symlinked here as `./outputs`.
- Do not touch anything outside this folder and that outputs folder. **Never** touch
  `/work/10757/manasp123/_archive_pre_2026-09`.
- **No paid API calls** without explicit instruction. Tests must never hit the network.
- Do not push to any remote. Commit locally, small commits, tests and ruff passing before each
  commit.
- If something in the environment blocks you, stop and ask. Do not work around it by installing
  system packages.
- **Never import torch, vllm or transformers on the login node**, and never add them to this
  project's dependencies. The 8 GB virtual-memory cap kills the import. Installing their wheels
  into `$WORK/venvs/vllm` is fine (`scripts/setup_vllm_env.sh`); anything that imports them runs
  inside a job. A test enforces this for `src/` and `tests/`.
- **GPU jobs are submitted only when Manas says so, for the config he names.** Never pick an
  allocation: it is passed with `sbatch -A` and is not written in any script.

The login node caps virtual memory at 8 GB per process (`ulimit -Hv`), which makes multithreaded
`uv` abort with "memory allocation of N bytes failed". Run uv with
`UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1` when it does, and
ruff with `RAYON_NUM_THREADS=1` for the same reason.

`$XDG_CACHE_HOME` points at `/work/.../ls6/cache`, so uv's cache lives on `$WORK` while `.venv`
and the uv-managed Python stay in `$HOME` (imports from `/work` are slow). `link-mode = "copy"`
in `pyproject.toml` is what keeps uv quiet about the cross-filesystem copy.

## Commands

```bash
uv sync                      # env
uv run pytest                # tests (fast, offline)
uv run ruff check .          # lint
uv run ruff format .         # format
uv run sc --help             # CLI
```

`TIKTOKEN_CACHE_DIR` defaults to `outputs/tiktoken_cache`, which already holds the `o200k_base`
encoding, so nothing downloads at test time.

## GPU path

Real runs are open-weights models served by vLLM on LS6 GPU nodes (`gpu-a100*`: 3x A100 40 GB;
`gpu-h100`: 2x H100 80 GB), not a paid API. See SPEC.md 3.9.

```bash
sbatch -A <allocation> scripts/build_llguidance.slurm  # once: the wheel glibc 2.28 needs
scripts/setup_vllm_env.sh cu129                        # once, login node, install only
scripts/download_weights.sh configs/models/<m>.yaml    # once; large; Manas runs this himself
uv run sc serve-args configs/smoke.yaml                # the exact vllm serve command
uv run sc run --config configs/smoke.yaml --dry-run    # cells and tokens, no server needed
sbatch -A <allocation> scripts/serve_and_run.slurm configs/smoke.yaml
```

Everything about a model lives in `configs/models/*.yaml` and is hashed into every run id, so
editing that file means new runs. Job logs, the vLLM log, timing and the run summary land in
`outputs/logs/`; rendered prompts in `outputs/renders/`.

## Conventions

- Everything except the model call is deterministic and unit-tested. No randomness and no
  network in library code. The one clock is the wall-time measurement around model calls
  (`elapsed_s`, tokens per second): metadata only, never hashed and never scored.
- Tools are pure: `(state, args) -> (state, ToolResult)`. Bad input returns an error result and
  never raises.
- All hashing goes through `canonical.py` (sorted keys, compact separators, UTF-8). Never hash a
  `repr` or an unsorted dump.
- Tool results are compact JSON strings with sorted keys, so prefixes are byte-stable.
- Trajectories are append-only JSONL keyed by `run_id`; re-running skips ids already present.
- Prefixes are content-addressed: rebuilding with the same inputs must produce the same hash. If
  a change moves the hash, that is a real change to the experiment and needs a note in SPEC.md.
- Scenario, world and episode data are YAML under `src/safety_checking/data/`.
