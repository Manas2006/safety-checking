# Results

Write-ups of the GPU runs, with the tables they quote. Raw trajectories stay in
`outputs/runs/*.jsonl` on LS6 (`$WORK`, not in git); everything here is derived from them and
is small enough to commit.

## Layout

- `tables/<experiment>.rates.csv`: check rate per cell with a 95% Wilson interval and the
  number of unsafe sends, from `sc rates --csv`. One file per experiment log.
- `tables/outcomes.csv`: the categorical outcome counts for every cell of every experiment
  (`checked_then_acted`, `checked_then_declined`, `acted_without_check`, `check_after_action`,
  `no_check_no_action`), which is what the rates are derived from.
- `tables/smoke.logprob.csv`: the saved logprob-mode records (SPEC.md 3.10), one row per
  (model, scenario, length, probe point).
- One dated markdown file per write-up.

## Regenerate

```bash
uv run python scripts/export_results.py   # reads outputs/runs, writes results/tables, calls nothing
```

Rerun it after a job lands, then commit the tables with the write-up that quotes them.

## Write-ups

- [2026-09-21: the check decays with history length on Qwen3.5-9B, and it is the history's content that does it](2026-09-21-qwen3.5-9b-length-effect.md).
  Smoke runs on five models, the first length contrast, the 2x2 control under both sampling
  settings, three other models, presence penalty, rule position and wording, reminder
  distance and wording, the performed pattern, a repeated-episode control, long horizons to
  200 calls, logprob curves, and the Ministral serving fix.

## What is still running

Submitted 2026-09-21 evening, results not yet in this folder: the logprob validation
(3461293); the `performed` pattern on the original pair (3461294); the 2x2 under each model's
own sampling on Qwen3.8-27B, Gemma 4 12B and gpt-oss-20b (3461299 to 3461301); the 2x2 with
thinking on (3461302, 3461303); the rest of the long-horizon job (3461364); and the smoke jobs
that the weight downloads for Qwen3.5-27B, Ministral 3 14B, Gemma 4 31B and gpt-oss-120b
submit as each finishes (`outputs/logs/download-round2.log`). `squeue -u manasp123` lists
them; each writes `outputs/logs/<experiment>-<jobid>.summary.json` when it finishes.
