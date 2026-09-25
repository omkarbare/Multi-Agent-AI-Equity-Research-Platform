"""
Convert evaluation_results.json from evaluation.py into a Markdown table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

METRIC_COLUMNS = (
    ("Evidence grounding", "Evidence grounding"),
    ("Research note usefulness", "Research usefulness"),
)


def cell(value: Any) -> str:
    """Escape table-breaking characters in text values."""
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def score_cell(metric: Any) -> str:
    """Show a metric score to two decimal places, if available."""
    if not isinstance(metric, dict) or metric.get("score") is None:
        return "—"
    try:
        return f"{float(metric['score']):.2f}"
    except (TypeError, ValueError):
        return "—"


def result_cell(record: dict[str, Any]) -> str:
    """Separate evaluated failures from records that errored before scoring."""
    if record.get("error"):
        return "Error"
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or any(name not in metrics for name, _ in METRIC_COLUMNS):
        return "Incomplete"
    for name, _ in METRIC_COLUMNS:
        if not isinstance(metrics[name], dict) or metrics[name].get("passed") is not True:
            return "Fail"
    return "Pass"


def render_table(records: list[dict[str, Any]], details: bool = False) -> str:
    """Create a README-ready table from the exact JSON schema of evaluation.py."""
    headers = ["Ticker", *(heading for _, heading in METRIC_COLUMNS), "Result"]
    if details:
        headers.append("Notes")
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---", *(["---:"] * len(METRIC_COLUMNS)), "---", *(["---"] if details else [])]) + " |"]

    for record in records:
        metrics = record.get("metrics") or {}
        values = [cell(record.get("ticker", "Unknown"))]
        values.extend(score_cell(metrics.get(name)) for name, _ in METRIC_COLUMNS)
        values.append(result_cell(record))
        if details:
            notes = [record["error"]] if record.get("error") else [
                f"{name}: {metric['reason']}"
                for name, _ in METRIC_COLUMNS
                if isinstance((metric := metrics.get(name)), dict) and metric.get("reason")
            ]
            values.append(cell("; ".join(notes) if notes else None))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    """Read, validate and convert one evaluation-results file."""
    parser = argparse.ArgumentParser(description="Make a Markdown table from DeepEval JSON results")
    parser.add_argument("input", nargs="?", default="evaluation_results.json",
                        help="JSON file written by evaluation.py")
    parser.add_argument("--output", help="Write the table to a Markdown file instead of stdout")
    parser.add_argument("--details", action="store_true",
                        help="Include judge reasons/error details in a Notes column")
    args = parser.parse_args()

    try:
        records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Cannot read {args.input}: {exc}\n")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        parser.exit(1, "Expected a JSON list of per-ticker evaluation records.\n")
    if not records:
        parser.exit(1, "The evaluation file has no ticker results.\n")

    table = render_table(records, details=args.details)
    if args.output:
        Path(args.output).write_text(table + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
