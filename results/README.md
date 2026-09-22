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

- [2026-09-21: the check decays with history length on Qwen3.5-9B](2026-09-21-qwen3.5-9b-length-effect.md).
  Smoke runs on five models, the first length contrast, the 2x2 control that separates
  "recipient already in context" from "recipient allowed", three ablations (presence
  penalty, rule position and wording, a reminder), and the logprob curves.

## What is still running

Submitted 2026-09-21 evening, results not yet in this folder: the 2x2 under each model's own
sampling on Qwen3.5-9B (job 3461292, the confirmation of the headline), Qwen3.8-27B, Gemma 4
12B and gpt-oss-20b; the logprob validation (3461293); the `performed` pattern (3461294); the
neutral-sampling 2x2 on the three larger models; and the 2x2 with thinking on (Qwen3.5-9B
think, gpt-oss-20b medium). `squeue -u manasp123` lists them; each writes a summary to
`outputs/logs/<experiment>-<jobid>.summary.json` when it finishes.
