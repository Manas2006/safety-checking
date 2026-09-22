"""Regenerate results/tables from outputs/runs. Reads saved logs only; calls nothing.

    uv run python scripts/export_results.py

One CSV of check rates per experiment (`sc rates --csv`), one CSV of outcome counts per cell
across every experiment, and one CSV of the saved logprob records. The write-ups in
results/*.md quote these tables; rerun this after a job lands and commit both.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
TABLES = ROOT / "results" / "tables"


def rates() -> list[str]:
    written = []
    for log in sorted(RUNS.glob("*.jsonl")):
        if ".logprob." in log.name:
            continue
        out = TABLES / f"{log.stem}.rates.csv"
        with out.open("w") as fh:
            subprocess.run(
                [sys.executable, "-m", "safety_checking.cli", "rates", "--csv", str(log)],
                check=True,
                stdout=fh,
                cwd=ROOT,
            )
        written.append(out.name)
    return written


def outcomes() -> str:
    counts: Counter[tuple] = Counter()
    for log in sorted(RUNS.glob("*.jsonl")):
        if ".logprob." in log.name:
            continue
        for line in log.open():
            record = json.loads(line)
            t, s = record["trajectory"], record["score"]
            if t.get("error"):
                continue
            model = t["model"].removeprefix("vllm:").split("#")[0]
            key = (
                log.stem,
                model,
                t["arm"],
                t["scenario_id"],
                t["prior_check_pattern"],
                t["length"],
                s["outcome"],
            )
            counts[key] += 1
    out = TABLES / "outcomes.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["experiment", "model", "arm", "scenario", "pattern", "length", "outcome", "n"]
        )
        for key, n in sorted(counts.items(), key=str):
            writer.writerow([*key, n])
    return out.name


def logprob() -> list[str]:
    written = []
    for log in sorted(RUNS.glob("*.logprob.jsonl")):
        out = TABLES / f"{log.stem}.csv"
        rows = []
        for line in log.open():
            r = json.loads(line)
            names = {name: entry["p"] for name, entry in r["p_name"].items()}
            top = max(names.items(), key=lambda kv: kv[1]) if names else ("", 0.0)
            rows.append(
                [
                    r["model"].removeprefix("vllm:").split("#")[0],
                    r["scenario_id"],
                    r["length"],
                    r["prior_check_pattern"],
                    r["probe_point"],
                    f"{r['p_call_first']:.4f}",
                    f"{r['p_check_given_call']:.4f}",
                    f"{r['p_send_given_call']:.4g}",
                    f"{names.get('lookup_person', 0.0):.4f}",
                    f"{top[0]}={top[1]:.4f}",
                    f"{r['unaccounted_mass']:.3g}",
                    r["n_requests"],
                ]
            )
        rows.sort(key=lambda row: (row[0], row[1], row[3], row[4], int(row[2])))
        with out.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "model",
                    "scenario",
                    "length",
                    "pattern",
                    "probe",
                    "p_call_first",
                    "p_check_given_call",
                    "p_send_given_call",
                    "p_lookup_given_call",
                    "top_name",
                    "unaccounted",
                    "n_requests",
                ]
            )
            writer.writerows(rows)
        written.append(out.name)
    return written


def validation() -> list[str]:
    """gate_neutral.jsonl against the logprob records: sampled first-call frequency per tool
    against p_call_first * p_name, with the binomial standard error of the prediction."""
    log = RUNS / "gate_neutral.jsonl"
    if not log.exists():
        return []
    first: Counter[tuple] = Counter()
    totals: Counter[tuple] = Counter()
    for line in log.open():
        record = json.loads(line)
        t = record["trajectory"]
        if t.get("error"):
            continue
        model = t["model"].removeprefix("vllm:").split("#")[0]
        calls = t["steps"][0].get("tool_calls") or []
        name = calls[0]["tool"] if calls else "(no call)"
        first[(model, t["scenario_id"], t["length"], name)] += 1
        totals[(model, t["scenario_id"], t["length"])] += 1
    predicted: dict[tuple, tuple[float, dict[str, float]]] = {}
    for probe_log in sorted(RUNS.glob("*.logprob.jsonl")):
        for line in probe_log.open():
            r = json.loads(line)
            if r["probe_point"] != "start":
                continue
            model = r["model"].removeprefix("vllm:").split("#")[0]
            key = (model, r["scenario_id"], r["length"])
            predicted[key] = (r["p_call_first"], {k: v["p"] for k, v in r["p_name"].items()})
    rows = []
    for key, n in sorted(totals.items()):
        if key not in predicted:
            continue
        p_call, p_name = predicted[key]
        names = sorted({k[3] for k in first if k[:3] == key} | set(p_name))
        for name in names:
            observed = first[(*key, name)] / n
            expected = (1 - p_call) if name == "(no call)" else p_call * p_name.get(name, 0.0)
            se = math.sqrt(max(expected * (1 - expected), 1e-9) / n)
            rows.append(
                [
                    *key,
                    name,
                    n,
                    f"{observed:.4f}",
                    f"{expected:.4f}",
                    f"{(observed - expected) / se:+.2f}",
                ]
            )
    out = TABLES / "logprob_validation.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["model", "scenario", "length", "first_call", "n", "sampled", "predicted", "z"]
        )
        writer.writerows(rows)
    return [out.name]


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in [*rates(), outcomes(), *logprob(), *validation()]:
        print(f"wrote results/tables/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
