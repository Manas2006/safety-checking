"""Rates and contrasts from run logs: definitions checked on hand-built records."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from safety_checking import cli
from safety_checking.analysis import (
    cell_rates,
    contrasts,
    main_rates,
    short_model_name,
)
from safety_checking.runner.store import RunRecord, append_record

REPO = Path(__file__).resolve().parents[1]


def make_record(
    *,
    model="vllm:m#abc",
    scenario="sharing_benign",
    length=5,
    outcome="checked_then_acted",
    unsafe=False,
    stop_reason="no_tool_calls",
    index=0,
    arm="baseline",
    pattern="none",
) -> RunRecord:
    """A record with just enough in it for the analysis; scoring is not exercised here."""
    from safety_checking.scoring import Score
    from safety_checking.trajectory import Trajectory

    run_id = f"{model}-{scenario}-{length}-{index}"
    trajectory = Trajectory(
        run_id=run_id,
        prefix_hash="p" * 64,
        scenario_id=scenario,
        variant="benign" if "benign" in scenario else "risky",
        length=length,
        prior_check_pattern=pattern,
        model=model,
        arm=arm,
        params={},
        sample_index=index,
        seed=index,
        server={},
        elapsed_s=0.0,
        messages=[],
        steps=[],
        stop_reason=stop_reason,
        final_text="",
        error=None if stop_reason != "error" else "boom",
    )
    score = None
    if stop_reason != "error":
        score = Score(
            check_executed=outcome.startswith("checked"),
            check_step=1 if outcome.startswith("checked") else None,
            action_step=2 if "acted" in outcome else None,
            steps_between=0,
            outcome=outcome,
            check_turn=0,
            action_turn=0,
            check_in_same_turn_as_action=False,
            action_taken="acted" in outcome,
            unsafe_action=unsafe,
            completed=True,
            asked_user_heuristic=False,
            failed_action_attempts=0,
            parse_failures=1 if outcome == "tool_call_parse_failure" else 0,
            parse_failure_reasons=[],
            truncated=False,
            n_steps=1,
            n_calls=1,
            stop_reason=stop_reason,
            tokens_decision=0,
            tokens_decision_estimated=True,
            usage_prompt_tokens=None,
            usage_completion_tokens=None,
            usage_cached_tokens=None,
            final_text="",
        )
    return RunRecord(run_id=run_id, trajectory=trajectory, score=score)


def test_short_model_name_drops_provider_and_hash() -> None:
    assert short_model_name("vllm:qwen3.5-9b-nothink#786fdaf309ac") == "qwen3.5-9b-nothink"
    assert short_model_name("fake:always_check") == "always_check"


def test_check_rate_counts_only_clean_runs_and_only_checks_before_the_action() -> None:
    records = [
        make_record(outcome="checked_then_acted", index=0),
        make_record(outcome="checked_then_declined", index=1),
        make_record(outcome="acted_without_check", index=2),
        make_record(outcome="check_after_action", index=3),  # not a check
        make_record(outcome="tool_call_parse_failure", index=4),  # excluded
        make_record(stop_reason="error", index=5),  # excluded
        make_record(outcome="acted_without_check", unsafe=True, index=6),
    ]
    (cell,) = cell_rates(records)
    assert (cell.n_clean, cell.n_excluded, cell.n_checked, cell.n_unsafe) == (5, 2, 2, 1)
    assert cell.check_rate == pytest.approx(0.4)
    low, high = cell.check_ci
    assert 0.0 < low < 0.4 < high < 1.0  # Wilson never touches 0 or 1 with n=5, k=2
    assert cell.unsafe_rate == pytest.approx(0.2)


def test_contrast_is_shortest_against_longest_with_fisher_p() -> None:
    records = []
    for i in range(10):
        records.append(make_record(length=5, index=i))  # 10/10 checked
        records.append(make_record(length=20, index=i, outcome="acted_without_check"))
        records.append(make_record(length=50, index=i, outcome="acted_without_check"))
    (c,) = contrasts(cell_rates(records))
    assert (c.short.key.length, c.long.key.length) == (5, 50)
    assert c.difference == pytest.approx(-1.0)
    assert c.fisher_p < 0.001

    # one length only: nothing to contrast
    assert contrasts(cell_rates([make_record(index=0)])) == []


def test_sc_rates_reads_logs_and_writes_csv(tmp_path, capsys) -> None:
    log = tmp_path / "x.jsonl"
    for i in range(4):
        append_record(
            log,
            make_record(index=i, outcome="checked_then_acted" if i < 3 else "acted_without_check"),
        )
    assert cli.main(["rates", str(log)]) == 0
    out = capsys.readouterr().out
    assert "0.75 [" in out and "check rate" in out

    assert cli.main(["rates", str(log), "--csv"]) == 0
    csv_out = capsys.readouterr().out
    assert csv_out.splitlines()[0].startswith("model,arm,scenario,pattern,length,n,excluded")
    assert ",4,0,3,0.7500," in csv_out

    buffer = io.StringIO()
    assert main_rates([tmp_path / "missing.jsonl"], out=buffer) == 1
