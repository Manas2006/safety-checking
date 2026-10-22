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

- [2026-10-22: handoff check rates and outcomes](2026-10-22-handoff.md), with transcripts in
  `transcripts/handoff-2026-10-22.md`. Tables only, no interpretation yet.

`tables/logprob_validation.csv` compares the 200-sample validation run with the logprob
predictions; the export script writes it whenever `gate_neutral.jsonl` exists.

## What is still running

Submitted 2026-10-21 and 2026-10-22, all charged to ECS26006 with `-t 02:00:00`. Jobs that
write the same log run in one serial `afterany` chain; `pos` is the position in that chain.

```bash
sacct -u manasp123 -S 2026-10-21 -o JobID,JobName%34,State,Elapsed,Start
```

| job | config | mode | model | partition | log | pos | after |
|---|---|---|---|---|---|---|---|
| 3527562 | `handoff.yaml` | run | gpt-oss-20b-low | gpu-a100 | `handoff.jsonl` | 1 | 3527551-3 (done) |
| 3527563 | `handoff.yaml` | run | qwen3.5-9b-think | gpu-a100 | `handoff.jsonl` | 2 | 3527562 |
| 3528647 | `handoff.yaml` | run | gemma-4-12b-think | gpu-a100 | `handoff.jsonl` | 3 | 3527563 |
| 3528648 | `handoff.yaml` | run | qwen3.5-27b-nothink | gpu-a100 | `handoff.jsonl` | 4 | 3528647 |
| 3528649 | `handoff.yaml` | run | ministral-3-8b | gpu-a100 | `handoff.jsonl` | 5 | 3528648 |
| 3528650 | `handoff_n100.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `handoff.jsonl` | 6 | 3528649 |
| 3528651 | `handoff_n100.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `handoff.jsonl` | 7 | 3528650 |
| 3528652 | `handoff_n100.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `handoff.jsonl` | 8 | 3528651 |
| 3528653 | `handoff.yaml` | run | gpt-oss-120b-low | gpu-h100 | `handoff.jsonl` | 9 | 3528652 |
| 3528654 | `handoff.yaml` | run | gemma-4-31b-nothink | gpu-h100 | `handoff.jsonl` | 10 | 3528653 |
| 3527564 | `competing.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `competing.jsonl` | 1 | none |
| 3527565 | `competing.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `competing.jsonl` | 2 | 3527564 |
| 3527566 | `competing.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `competing.jsonl` | 3 | 3527565 |
| 3527567 | `override.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `override.jsonl` | 1 | none |
| 3527568 | `override.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `override.jsonl` | 2 | 3527567 |
| 3527569 | `override.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `override.jsonl` | 3 | 3527568 |
| 3527570 | `urgency.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `urgency.jsonl` | 1 | none |
| 3527571 | `urgency.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `urgency.jsonl` | 2 | 3527570 |
| 3527572 | `handoff_long.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `handoff_long.jsonl` | 1 | none |
| 3527573 | `contrast2_ingredients.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `contrast2_ingredients.jsonl` | 1 | none |
| 3527574 | `contrast2_ingredients.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `contrast2_ingredients.jsonl` | 2 | 3527573 |
| 3527575 | `precedent.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `precedent.jsonl` | 1 | none |
| 3527576 | `precedent.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `precedent.jsonl` | 2 | 3527575 |
| 3527577 | `inject.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `inject.jsonl` | 1 | none |
| 3527578 | `inject.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `inject.jsonl` | 2 | 3527577 |
| 3528655 | `handoff.yaml` | logprob | qwen3.5-9b-nothink | gpu-a100 | `handoff.logprob.jsonl` | 1 | none |
| 3528656 | `handoff.yaml` | logprob | qwen3.8-27b-nothink | gpu-a100 | `handoff.logprob.jsonl` | 2 | 3528655 |
| 3528657 | `handoff_performed.yaml` | run | qwen3.5-9b-nothink | gpu-a100-dev | `handoff_performed.jsonl` | 1 | none |
| 3528658 | `handoff_performed.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `handoff_performed.jsonl` | 2 | 3528657 |
| 3528659 | `handoff_performed.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `handoff_performed.jsonl` | 3 | 3528658 |
| 3528660 | `handoff_source.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `handoff_source.jsonl` | 1 | none |
| 3528661 | `handoff_source.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `handoff_source.jsonl` | 2 | 3528660 |
| 3528662 | `handoff_source.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `handoff_source.jsonl` | 3 | 3528661 |
| 3528663 | `handoff_hardened.yaml` | run | qwen3.5-9b-nothink | gpu-a100-dev | `handoff_hardened.jsonl` | 1 | none |
| 3528664 | `handoff_hardened.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `handoff_hardened.jsonl` | 2 | 3528663 |
| 3528665 | `handoff_hardened.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `handoff_hardened.jsonl` | 3 | 3528664 |

Not submitted: `handoff_relay.yaml` on qwen3.5-9b-nothink, gemma-4-12b-nothink and
qwen3.8-27b-nothink (one chain into `handoff_relay.jsonl`). The gpu-a100 queue allows 32
submitted jobs per user and was full on 2026-10-22; it goes in when slots free.

Done: handoff.yaml on Qwen3.5-9B, Gemma 4 12B and Qwen3.8-27B (3527551 to 3527553),
[2026-10-22-handoff.md](2026-10-22-handoff.md).
