"""The ``sc`` command.

sc tools                         list the world's tool specs
sc build-prefixes                build, validate and save the frozen prefixes
sc run --dry-run                 cell counts and a token estimate, no calls
sc run --model fake:always_check run (resumable); paid models need --confirm-paid
sc run --config configs/x.yaml   run an experiment config against a local vLLM server
sc serve-args MODEL.yaml         the vllm serve command line for a model config
sc render --config ...           render a prefix through the served chat template
sc logprob --config ...          read the next-call distribution at each probe point
sc show FILE --index 0           render one trajectory as markdown
sc counts FILE                   outcome counts per cell (counts only, never rates)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .config import load_any, load_experiment_config
from .history.builder import build_prefix, check_nesting
from .history.plan import PRIOR_CHECK_PATTERNS, load_plan
from .history.store import load_prefix
from .logprob import LogprobError, logprob_log_path, read_logprob_records, run_logprob
from .paths import outputs_dir, prefixes_dir
from .render import render_prefix
from .runner.adapters import ModelAdapter, OpenAICompatAdapter, make_adapter
from .runner.experiment import ExperimentSpec, build_cells, dry_run, run_experiment
from .runner.store import latest_records, run_log_path
from .scenarios.loader import list_scenarios, load_scenario, load_scenario_world
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


def _spec_and_adapter(args: argparse.Namespace) -> tuple[ExperimentSpec, ModelAdapter]:
    """From --config (an experiment YAML plus its model YAML), or from the command line."""
    if args.config:
        experiment, model = load_experiment_config(args.config)
        spec = ExperimentSpec(
            experiment=experiment.experiment,
            scenarios=experiment.scenarios,
            lengths=experiment.lengths,
            patterns=experiment.patterns,
            n_samples=experiment.n_samples,
            arm=experiment.arm,
            max_steps=experiment.max_steps,
            concurrency=experiment.concurrency,
            params=model.request_params(),
            base_seed=model.request.base_seed,
        )
        return spec, OpenAICompatAdapter.from_model_config(model, base_url=args.base_url)
    if not args.model:
        raise SystemExit("sc run: give --config or --model")
    spec = _spec_from(args)
    spec.concurrency = args.concurrency
    # constructing an adapter is free: clients are created lazily, on the first real call
    kwargs = {"base_url": args.base_url} if args.base_url else {}
    return spec, make_adapter(args.model, **kwargs)


def _is_free(adapter: ModelAdapter) -> bool:
    """Fake models, and anything served from this machine, cost nothing."""
    return adapter.name.startswith("fake:") or bool(getattr(adapter, "is_local", False))


def cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()
    spec, adapter = _spec_and_adapter(args)
    log_path = run_log_path(spec.experiment)

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

    if not _is_free(adapter) and not args.confirm_paid:
        console.print(
            "[red]Refusing to call a paid model without --confirm-paid.[/red] "
            "Run with --dry-run first to see the cell counts and the token estimate. "
            "(Fake models and a localhost base_url never need it.)"
        )
        return 2

    cells = build_cells(spec)
    summary = run_experiment(spec, adapter, log_path, cells)
    console.print(
        f"{summary.n_new} new, {summary.n_skipped} skipped, {summary.n_errors} errors -> {log_path}"
    )
    rate = summary.completion_tokens_per_s
    console.print(
        f"{summary.elapsed_s:.1f}s wall, {summary.prompt_tokens:,} prompt tokens "
        f"({summary.cached_tokens:,} cached), {summary.completion_tokens:,} completion tokens"
        + (f", {rate:.1f} completion tok/s" if rate is not None else "")
    )

    job = os.environ.get("SLURM_JOB_ID")
    name = f"{spec.experiment}{'-' + job if job else ''}.summary.json"
    summary_path = outputs_dir() / "logs" / name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump() | {
        "experiment": spec.experiment,
        "model": adapter.name,
        "concurrency": spec.concurrency,
        "completion_tokens_per_s": rate,
        "server": adapter.server_info() if hasattr(adapter, "server_info") else {},
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if summary.n_errors == 0 else 1


def cmd_serve_args(args: argparse.Namespace) -> int:
    """Print the vllm serve command for a config, or one field of it.

    Takes a model YAML or an experiment YAML (which names its model), so a job script needs no
    YAML parsing of its own. Fields are the model config's, plus ``experiment.*``.
    """
    experiment, model = load_any(args.config)
    if args.field:
        value: Any = model.model_dump(mode="json")
        if experiment is not None:
            value["experiment"] = experiment.model_dump(mode="json")
        try:
            for part in args.field.split("."):
                value = value[part]
        except (KeyError, TypeError):
            console.print(f"[red]no such field: {args.field}[/red]")
            return 1
        sys.stdout.write(f"{value}\n")
    elif args.env:
        # KEY=VALUE per line, for `while IFS= read -r kv; do export "$kv"; done`
        for key, value in sorted(model.serve.env.items()):
            sys.stdout.write(f"{key}={value}\n")
    elif args.lines:
        # one argument per line, for `mapfile -t ARGS < <(sc serve-args --lines ...)`
        sys.stdout.write("\n".join(model.serve_argv()) + "\n")
    else:
        sys.stdout.write(model.serve_command() + "\n")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render one prefix through the served model's chat template and verify it."""
    _, model = load_experiment_config(args.config)
    scenario = load_scenario(args.scenario)
    prefix = build_prefix(
        scenario, load_scenario_world(scenario), args.length, prior_check_pattern=args.pattern
    )
    path, report = render_prefix(prefix, scenario, model, base_url=args.base_url)
    console.print(f"wrote {path} ({report.n_prompt_tokens} prompt tokens)")
    console.print(
        f"{report.n_found_in_order}/{report.n_expected} expected items found in order: "
        f"{report.n_tool_calls} tool calls, {report.n_tool_results} tool results"
    )
    if report.thinking_markup_present:
        console.print("[yellow]note: the rendered prompt contains <think> markup[/yellow]")
    for label in report.missing:
        console.print(f"[red]MISSING[/red] {label}")
    for label in report.out_of_order:
        console.print(f"[red]OUT OF ORDER[/red] {label}")
    console.print("render check: " + ("OK" if report.ok else "FAILED"))
    return 0 if report.ok else 1


def _print_logprob_table(path: Path) -> None:
    """Plain tab-separated lines; probabilities in scientific notation, since they are small."""
    header = ["scenario", "L", "pattern", "probe", "p_call_first", "p_check|call", "p_send|call"]
    sys.stdout.write("\t".join([*header, "top other name", "unaccounted"]) + "\n")
    records = sorted(
        read_logprob_records(path),
        key=lambda r: (r.scenario_id, r.probe_point, r.prior_check_pattern, r.length),
    )
    for r in records:
        others = {
            name: entry.p
            for name, entry in r.p_name.items()
            if entry.p not in (r.p_check_given_call, r.p_send_given_call)
        }
        top_other = max(others, key=others.get) if others else ""
        row = [
            r.scenario_id,
            str(r.length),
            r.prior_check_pattern,
            r.probe_point,
            f"{r.p_call_first:.4f}{'<' if r.p_call_first_bounded else ''}",
            f"{r.p_check_given_call:.6f}",
            f"{r.p_send_given_call:.3e}",
            f"{top_other}={others.get(top_other, 0.0):.4f}" if top_other else "",
            f"{r.unaccounted_mass:.2e}",
        ]
        sys.stdout.write("\t".join(row) + "\n")


def cmd_logprob(args: argparse.Namespace) -> int:
    """Read the model's next-call distribution at every probe point (SPEC.md 3.10)."""
    experiment, model = load_experiment_config(args.config)
    path = logprob_log_path(experiment.experiment)
    if args.table:
        _print_logprob_table(path)
        return 0
    adapter = OpenAICompatAdapter.from_model_config(model, base_url=args.base_url)
    if not _is_free(adapter):
        console.print("[red]logprob mode only runs against a localhost server.[/red]")
        return 2
    try:
        summary = run_logprob(
            experiment, model, path, base_url=args.base_url, server=adapter.server_info()
        )
    except LogprobError as exc:
        console.print(f"[red]logprob mode failed:[/red] {exc}")
        return 1
    console.print(
        f"{summary.n_new} new, {summary.n_skipped} skipped, {summary.n_requests} requests, "
        f"{summary.elapsed_s:.1f}s -> {path}"
    )
    _print_logprob_table(path)
    return 0


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
    run.add_argument("--model", help="fake:always_check | openai:<model> | ...")
    run.add_argument("--config", help="experiment YAML (names its model YAML); replaces --model")
    run.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    run.add_argument("--concurrency", type=int, default=1, help="samples of a prefix in flight")
    run.add_argument("--experiment", default="adhoc", help="name of the JSONL log")
    run.add_argument("--samples", type=int, default=3, help="samples per cell")
    run.add_argument("--arm", default="baseline")
    run.add_argument("--param", action="append", type=_parse_param, help="model param, key=value")
    run.add_argument("--dry-run", action="store_true", help="count cells and tokens, make no calls")
    run.add_argument(
        "--confirm-paid",
        action="store_true",
        help="required for any model that is neither fake nor served from localhost",
    )
    run.set_defaults(func=cmd_run)

    serve = commands.add_parser("serve-args", help="vllm serve command line for a model config")
    serve.add_argument("config", help="a model YAML, or an experiment YAML that names one")
    serve.add_argument("--lines", action="store_true", help="one argument per line")
    serve.add_argument("--field", help="print one config field instead, e.g. serve.port")
    serve.add_argument("--env", action="store_true", help="print serve.env as KEY=VALUE lines")
    serve.set_defaults(func=cmd_serve_args)

    render = commands.add_parser("render", help="render a prefix through the chat template")
    render.add_argument("--config", required=True, help="experiment YAML")
    render.add_argument("--scenario", default="sharing_risky")
    render.add_argument("--length", type=int, default=50)
    render.add_argument("--pattern", default="none", choices=PRIOR_CHECK_PATTERNS)
    render.add_argument("--base-url")
    render.set_defaults(func=cmd_render)

    logprob = commands.add_parser("logprob", help="next-call distribution at each probe point")
    logprob.add_argument("--config", required=True, help="experiment YAML")
    logprob.add_argument("--base-url")
    logprob.add_argument("--table", action="store_true", help="print saved records, call nothing")
    logprob.set_defaults(func=cmd_logprob)

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
