"""The ``sc`` command.

sc tools                         list the world's tool specs
sc build-prefixes                build, validate and save the frozen prefixes
sc run --dry-run                 cell counts and a token estimate, no calls
sc run --model fake:always_check run (resumable); real models need --confirm-paid
sc show FILE --index 0           render one trajectory as markdown
sc counts FILE                   outcome counts per cell (counts only, never rates)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .history.builder import check_nesting
from .history.plan import PRIOR_CHECK_PATTERNS, load_plan
from .history.store import load_prefix
from .paths import prefixes_dir
from .runner.adapters import make_adapter
from .runner.experiment import ExperimentSpec, build_cells, dry_run, run_experiment
from .runner.store import latest_records, run_log_path
from .scenarios.loader import list_scenarios
from .viewer import render_record
from .world.registry import REGISTRY

console = Console()


def _parse_param(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def _spec_from(args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec(
        experiment=getattr(args, "experiment", "adhoc"),
        scenarios=args.scenarios or list_scenarios(),
        lengths=args.lengths or list(load_plan().lengths),
        patterns=args.patterns or ["none"],
        n_samples=getattr(args, "samples", 1),
        arm=getattr(args, "arm", "baseline"),
        params=dict(getattr(args, "param", None) or []),
    )


def _add_cell_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenarios", nargs="+", help="scenario ids (default: all)")
    parser.add_argument("--lengths", nargs="+", type=int, help="history lengths (default: plan)")
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=PRIOR_CHECK_PATTERNS,
        help="prior_check_pattern values (default: none)",
    )


def cmd_tools(args: argparse.Namespace) -> int:
    table = Table("tool", "mutates", "parameters", "description")
    for spec in REGISTRY.values():
        required = set(spec.parameters.get("required", []))
        params = ", ".join(
            name if name in required else f"[{name}]" for name in spec.parameters["properties"]
        )
        table.add_row(spec.name, "yes" if spec.mutates else "", params, spec.description)
    console.print(table)
    return 0


def cmd_build_prefixes(args: argparse.Namespace) -> int:
    spec = _spec_from(args)
    cells = build_cells(spec, save=not args.no_save)
    table = Table("cell", "calls", "messages", "tokens", "prefix hash", "tail hash")
    for cell in cells:
        meta = cell.prefix.metadata
        tokens = f"{meta.token_count}{'~' if meta.token_count_estimated else ''}"
        table.add_row(
            cell.key,
            str(meta.n_calls),
            str(meta.n_messages),
            tokens,
            cell.prefix.prefix_hash[:12],
            meta.tail_hash[:12],
        )
    console.print(table)

    tails = {cell.prefix.metadata.tail_hash for cell in cells}
    nested = all(
        check_nesting([c.prefix for c in cells if c.scenario.id == scenario]).suffix_ok
        for scenario in {c.scenario.id for c in cells}
    )
    console.print(f"tail identical across all cells: {len(tails) == 1}")
    console.print(f"shorter histories are suffixes of longer ones: {nested}")
    if not args.no_save:
        console.print(f"saved {len(cells)} prefixes under {prefixes_dir()}")
    return 0 if (len(tails) == 1 and nested) else 1


def cmd_run(args: argparse.Namespace) -> int:
    spec = _spec_from(args)
    log_path = run_log_path(spec.experiment)
    is_fake = args.model.startswith("fake:")
    load_dotenv()
    # constructing an adapter is free: clients are created lazily, on the first real call
    adapter = make_adapter(args.model)

    if args.dry_run:
        cells = build_cells(spec, save=False)
        report = dry_run(spec, adapter.name, log_path, cells)
        table = Table(
            "cell", "prefix tokens", "samples", "done", "est. prompt tok", "est. completion tok"
        )
        for cell in report.cells:
            table.add_row(
                cell.key,
                str(cell.prefix_tokens),
                str(cell.n_samples),
                str(cell.n_done),
                f"{cell.est_prompt_tokens:,}",
                f"{cell.est_completion_tokens:,}",
            )
        console.print(table)
        console.print(
            f"{len(report.cells)} cells, {report.n_runs} runs "
            f"({report.n_done} done, {report.n_todo} to do). "
            f"Rough upper bound with no prompt caching: {report.est_prompt_tokens:,} prompt "
            f"and {report.est_completion_tokens:,} completion tokens. No calls made."
        )
        return 0

    if not is_fake and not args.confirm_paid:
        console.print(
            "[red]Refusing to call a real model without --confirm-paid.[/red] "
            "Run with --dry-run first to see the cell counts and the token estimate."
        )
        return 2

    cells = build_cells(spec)
    summary = run_experiment(spec, adapter, log_path, cells)
    console.print(
        f"{summary.n_new} new, {summary.n_skipped} skipped, {summary.n_errors} errors -> {log_path}"
    )
    return 0 if summary.n_errors == 0 else 1


def cmd_show(args: argparse.Namespace) -> int:
    records = latest_records(Path(args.file))
    if not records:
        console.print(f"[red]no records in {args.file}[/red]")
        return 1
    if args.run_id:
        matches = [r for r in records if r.run_id.startswith(args.run_id)]
        if len(matches) != 1:
            console.print(f"[red]{len(matches)} records match run id {args.run_id!r}[/red]")
            return 1
        record = matches[0]
    else:
        if not -len(records) <= args.index < len(records):
            console.print(f"[red]index {args.index} out of range (0..{len(records) - 1})[/red]")
            return 1
        record = records[args.index]

    try:
        prefix = load_prefix(record.trajectory.prefix_hash)
    except FileNotFoundError:
        prefix = None
    sys.stdout.write(render_record(record, prefix, full_prefix=args.full_prefix) + "\n")
    return 0


def cmd_counts(args: argparse.Namespace) -> int:
    """Outcome counts per cell. Counts only: rates are defined in PREREG.md, not here."""
    counts: dict[tuple[str, str, int, str], Counter[str]] = {}
    for record in latest_records(Path(args.file)):
        t = record.trajectory
        key = (t.model, t.scenario_id, t.length, t.prior_check_pattern)
        outcome = record.score.outcome if record.score else f"<{t.stop_reason}>"
        counts.setdefault(key, Counter())[outcome] += 1
    # plain tab-separated lines: a table would truncate the outcome names in a narrow terminal
    sys.stdout.write("model\tscenario\tlength\tpattern\tn\toutcomes\n")
    for key in sorted(counts):
        tally = counts[key]
        outcomes = ", ".join(f"{name}={n}" for name, n in sorted(tally.items()))
        row = [key[0], key[1], str(key[2]), key[3], str(sum(tally.values())), outcomes]
        sys.stdout.write("\t".join(row) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sc", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    tools = commands.add_parser("tools", help="list the world's tool specs")
    tools.set_defaults(func=cmd_tools)

    build = commands.add_parser("build-prefixes", help="build, validate and save frozen prefixes")
    _add_cell_options(build)
    build.add_argument("--no-save", action="store_true", help="validate only, write nothing")
    build.set_defaults(func=cmd_build_prefixes)

    run = commands.add_parser("run", help="run an experiment (resumable)")
    _add_cell_options(run)
    run.add_argument("--model", required=True, help="fake:always_check | openai:<model> | ...")
    run.add_argument("--experiment", default="adhoc", help="name of the JSONL log")
    run.add_argument("--samples", type=int, default=3, help="samples per cell")
    run.add_argument("--arm", default="baseline")
    run.add_argument("--param", action="append", type=_parse_param, help="model param, key=value")
    run.add_argument("--dry-run", action="store_true", help="count cells and tokens, make no calls")
    run.add_argument(
        "--confirm-paid", action="store_true", help="required to call any non-fake model"
    )
    run.set_defaults(func=cmd_run)

    show = commands.add_parser("show", help="render one trajectory as markdown")
    show.add_argument("file")
    show.add_argument("--run-id", help="run id or unique prefix of one")
    show.add_argument("--index", type=int, default=0, help="record index (default 0)")
    show.add_argument("--full-prefix", action="store_true", help="render the whole prefix")
    show.set_defaults(func=cmd_show)

    counts = commands.add_parser("counts", help="outcome counts per cell")
    counts.add_argument("file")
    counts.set_defaults(func=cmd_counts)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
