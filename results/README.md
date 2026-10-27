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
  `transcripts/handoff-2026-10-22.md` and `transcripts/hardened-qwen3.5-9b-2026-10-23.md`; it
  also holds handoff_performed and handoff_hardened on Qwen3.5-9B. Tables only, no
  interpretation yet.

- [2026-10-27: follow-up check rates and outcomes](2026-10-27-followups.md): competing,
  override, urgency, handoff_long, precedent, inject, contrast2_ingredients, handoff_source,
  handoff_hardened, handoff_performed and handoff_relay; handoff.yaml on five more models, at
  n = 100, and in logprob mode. Tables only, no interpretation yet.

`tables/logprob_validation.csv` compares the 200-sample validation run with the logprob
predictions; the export script writes it whenever `gate_neutral.jsonl` exists.

## Jobs

All charged to ECS26006 with `-t 02:00:00`. Jobs that write the same log run one at a time,
in one `afterany` chain per log file.

```bash
sacct -u manasp123 -S 2026-10-21 -o JobID,JobName%34,Partition,State,Elapsed,Start
```

### Run on gpu-a100-dev, 2026-10-23

The normal gpu-a100 queue had started none of the jobs below in 37 hours, so they were
cancelled there and rerun on the dev queue (1 running and 3 submitted per user), in this order.
Tables are in `tables/`; write-up in [2026-10-27-followups.md](2026-10-27-followups.md).

| # | job | config | model | state | normal-queue job cancelled |
|---|---|---|---|---|---|
| 1 | 3531078 | `competing.yaml` | gemma-4-12b-nothink | done | 3527565 |
| 2 | 3531080 | `handoff_hardened.yaml` | gemma-4-12b-nothink | done | 3528664 |
| 3 | 3531081 | `handoff_hardened.yaml` | qwen3.8-27b-nothink | done | 3528665 |
| 4 | 3531201 | `handoff_performed.yaml` | gemma-4-12b-nothink | done | 3528658 |
| 5 | 3531210 | `handoff_performed.yaml` | qwen3.8-27b-nothink | done | 3528659 |
| 6 | 3531222 | `override.yaml` | gemma-4-12b-nothink | done | 3527568 |
| 7 | 3531265 | `handoff_source.yaml` | gemma-4-12b-nothink | done | 3528661 |
| 8 | 3531283 | `handoff_source.yaml` | qwen3.8-27b-nothink | done | 3528662 |
| 9 | 3531290 | `handoff_source.yaml` | qwen3.5-9b-nothink | done | 3528660 |
| 10 | 3531308 | `handoff.yaml` | qwen3.5-9b-think | done | 3527563 |
| 11 | 3531345 | `handoff.yaml` | gemma-4-12b-think | done | 3528647 |
| 12 | 3531359, 3531418 | `handoff.yaml` | gpt-oss-20b-low | done on the resubmission | 3527562 |
| 13 | 3531369 | `handoff_relay.yaml` | gemma-4-12b-nothink | done | none (never submitted) |
| 14 | 3531397 | `handoff_relay.yaml` | qwen3.5-9b-nothink | done | none (never submitted) |
| 15 | 3531410 | `competing.yaml` | qwen3.8-27b-nothink | done | 3527566 |
| 16 | 3531426 | `competing.yaml` | qwen3.5-9b-nothink | done | 3527564 |

No run was excluded in any of these jobs except gpt-oss-20b-low: 3531359 wrote all 500 runs,
but one ended in an adapter error, and `sc run` exits 1 on any error, so the job counted as
failed. The resubmission, 3531418, redid that run (1 new, 499 skipped) and exited 0. That
model has 3 excluded runs in all (tool-call parse failures), in arms D and F.

Also run on the dev queue earlier: handoff.yaml on Qwen3.5-9B, Gemma 4 12B and Qwen3.8-27B
(3527551 to 3527553), and handoff_performed and handoff_hardened on Qwen3.5-9B (3528657,
3528663).

### Run on gpu-a100, 2026-10-24 and 2026-10-25

All exit 0 with no errored runs. Write-up in
[2026-10-27-followups.md](2026-10-27-followups.md).

| job | config | mode | model | partition | log | after |
|---|---|---|---|---|---|---|
| 3528648 | `handoff.yaml` | run | qwen3.5-27b-nothink | gpu-a100 | `handoff.jsonl` | 3531418 (done) |
| 3528649 | `handoff.yaml` | run | ministral-3-8b | gpu-a100 | `handoff.jsonl` | 3528648 |
| 3528650 | `handoff_n100.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `handoff.jsonl` | 3528649 |
| 3528651 | `handoff_n100.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `handoff.jsonl` | 3528650 |
| 3528652 | `handoff_n100.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `handoff.jsonl` | 3528651 |
| 3527567 | `override.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `override.jsonl` | 3531222 (done) |
| 3527569 | `override.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `override.jsonl` | 3527567 |
| 3527570 | `urgency.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `urgency.jsonl` | none |
| 3527571 | `urgency.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `urgency.jsonl` | 3527570 |
| 3527572 | `handoff_long.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `handoff_long.jsonl` | none |
| 3527573 | `contrast2_ingredients.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `contrast2_ingredients.jsonl` | none |
| 3527574 | `contrast2_ingredients.yaml` | run | qwen3.8-27b-nothink | gpu-a100 | `contrast2_ingredients.jsonl` | 3527573 |
| 3527575 | `precedent.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `precedent.jsonl` | none |
| 3527576 | `precedent.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `precedent.jsonl` | 3527575 |
| 3527577 | `inject.yaml` | run | qwen3.5-9b-nothink | gpu-a100 | `inject.jsonl` | none |
| 3527578 | `inject.yaml` | run | gemma-4-12b-nothink | gpu-a100 | `inject.jsonl` | 3527577 |
| 3528655 | `handoff.yaml` | logprob | qwen3.5-9b-nothink | gpu-a100 | `handoff.logprob.jsonl` | none |
| 3528656 | `handoff.yaml` | logprob | qwen3.8-27b-nothink | gpu-a100 | `handoff.logprob.jsonl` | 3528655 |

### Still pending on gpu-h100

| job | config | mode | model | partition | log | after |
|---|---|---|---|---|---|---|
| 3528653 | `handoff.yaml` | run | gpt-oss-120b-low | gpu-h100 | `handoff.jsonl` | 3528652 |
| 3528654 | `handoff.yaml` | run | gemma-4-31b-nothink | gpu-h100 | `handoff.jsonl` | 3528653 |

Three of the gpu-a100 jobs above had their dependencies changed on 2026-10-23 to keep one
writer per log:
- 3528648 was re-pointed from 3528647 (cancelled) onto the dev handoff jobs.
- 3527567 was put behind the dev override job, and 3527569 behind 3527567, in place of
  3527568 (cancelled).

Not submitted anywhere until the batch below: `handoff_relay.yaml` on qwen3.8-27b-nothink.

### Batch of 2026-10-27 (submitted 2026-10-27)

Normal gpu-a100 queue, ECS26006, `-t 02:00:00`. Each job is named (`-J`) after the log it writes and
carries `--dependency=singleton`, so jobs that write the same log run one at a time, in
submission order, with no job ids to track. All 28 were pending when submitted. Status, export
and the tables-only write-up, safe to rerun:

```bash
bash scripts/check_batch.sh
```

When back: run that, then, if `scripts/submit_remaining.sh` still holds an `sbatch` line and
`squeue` shows room under 30 jobs:

```bash
bash scripts/submit_remaining.sh ECS26006
```

| job | -J (log) | config | mode | model |
|---|---|---|---|---|
| 3537600 | `bundle_protocol` | `bundle_protocol.yaml` | run | gemma-4-12b-nothink |
| 3537601 | `bundle_protocol` | `bundle_protocol.yaml` | run | qwen3.8-27b-nothink |
| 3537602 | `bundle_protocol` | `bundle_protocol.yaml` | run | qwen3.5-9b-nothink |
| 3537603 | `bundle_team` | `bundle_team.yaml` | run | gemma-4-12b-nothink |
| 3537604 | `bundle_team` | `bundle_team.yaml` | run | qwen3.8-27b-nothink |
| 3537605 | `bundle_team` | `bundle_team.yaml` | run | qwen3.5-9b-nothink |
| 3537606 | `bundle_dialogue` | `bundle_dialogue.yaml` | run | gemma-4-12b-nothink |
| 3537607 | `bundle_dialogue` | `bundle_dialogue.yaml` | run | qwen3.8-27b-nothink |
| 3537608 | `bundle_dialogue` | `bundle_dialogue.yaml` | run | qwen3.5-9b-nothink |
| 3537609 | `lp_bundle` | `lp_bundle.yaml` | both | gemma-4-12b-nothink |
| 3537610 | `lp_bundle` | `lp_bundle.yaml` | both | qwen3.8-27b-nothink |
| 3537611 | `lp_bundle` | `lp_bundle.yaml` | both | qwen3.5-9b-nothink |
| 3537612 | `assurance_back_5` | `assurance_back_5.yaml` | run | gemma-4-12b-nothink |
| 3537613 | `assurance_back_20` | `assurance_back_20.yaml` | run | gemma-4-12b-nothink |
| 3537614 | `assurance_back_50` | `assurance_back_50.yaml` | run | gemma-4-12b-nothink |
| 3537615 | `assurance_back_5` | `assurance_back_5.yaml` | run | qwen3.5-9b-nothink |
| 3537616 | `assurance_back_20` | `assurance_back_20.yaml` | run | qwen3.5-9b-nothink |
| 3537617 | `assurance_back_50` | `assurance_back_50.yaml` | run | qwen3.5-9b-nothink |
| 3537618 | `orchestrator_gemma-4-12b-nothink` | `orchestrate.yaml` | orchestrate | gemma-4-12b-nothink |
| 3537619 | `orchestrator_qwen3.8-27b-nothink` | `orchestrate.yaml` | orchestrate | qwen3.8-27b-nothink |
| 3537620 | `orchestrator_qwen3.5-9b-nothink` | `orchestrate.yaml` | orchestrate | qwen3.5-9b-nothink |
| 3537621 | `bundle_protocol` | `bundle_protocol.yaml` | run | qwen3.5-27b-nothink |
| 3537622 | `bundle_team` | `bundle_team.yaml` | run | qwen3.5-27b-nothink |
| 3537623 | `bundle_dialogue` | `bundle_dialogue.yaml` | run | qwen3.5-27b-nothink |
| 3537624 | `bundle_protocol` | `bundle_protocol.yaml` | run | gpt-oss-20b-low |
| 3537625 | `bundle_team` | `bundle_team.yaml` | run | gpt-oss-20b-low |
| 3537626 | `bundle_dialogue` | `bundle_dialogue.yaml` | run | gpt-oss-20b-low |
| 3537627 | `lp_bundle` | `lp_bundle.yaml` | both | qwen3.5-27b-nothink |

Not submitted (the 30-job cap), in `scripts/submit_remaining.sh`: `handoff_relay.yaml` on
qwen3.8-27b-nothink (`-J handoff_relay`). lp_bundle runs in `both` mode on every model it was
submitted for, since all four have a logprob block. `sc orchestrate` was built (Part 5's guard did
not trigger), so the three orchestrate jobs run; their messages land in
`outputs/orchestrator/<model>.jsonl`, outside `outputs/runs`.
