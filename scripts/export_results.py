"""Regenerate results/tables from outputs/runs. Reads saved logs only; calls nothing.

    uv run python scripts/export_results.py

One CSV of check rates per experiment (`sc rates --csv`), one CSV of outcome counts per cell
across every experiment, and one CSV of the saved logprob records. The write-ups in
results/*.md quote these tables; rerun this after a job lands and commit both.
"""

from __future__ import annotations

import csv
import json
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


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in [*rates(), outcomes(), *logprob()]:
        print(f"wrote results/tables/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
