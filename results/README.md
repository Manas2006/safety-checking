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
  Eighteen sections: smoke runs, the first contrast, the 2x2 control under both sampling
  settings, eight models including thinking-on variants, long horizons to 200 calls on three
  models, presence penalty, rule position and wording, reminder distance and wording, the
  performed pattern, a repeated-episode control, the ingredient experiment, Ministral's
  check-then-send failure, and the logprob validation.

`tables/logprob_validation.csv` compares the 200-sample validation run with the logprob
predictions; the export script writes it whenever `gate_neutral.jsonl` exists.

## What is still running

Nothing, as of 2026-09-22 morning. Not yet run: Gemma 4 31B (its smoke job died with a node
failure during server start) and a second look at Ministral 3 14B (13 of 30 smoke runs had
tool-call parse failures).
