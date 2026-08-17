"""Aggregate per-sample results into a per-probe summary."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev

from .types import Result
from .utils import dump_json


def aggregate(results: list[Result]) -> dict[str, dict]:
    """Per-probe mean/std/n for every metric, plus run counts.

    Only `status == "ok"` rows feed the metric averages, so a probe whose videos
    mostly failed to parse shows a high score on a small `n` rather than a
    silently diluted one — `n_ok` next to `n_total` is the guard against reading
    that as a real result.
    """
    summary: dict[str, dict] = {}
    for result in results:
        entry = summary.setdefault(
            result.probe,
            {"n_total": 0, "n_ok": 0, "n_skipped": 0, "n_error": 0, "metrics": {}, "_values": {}},
        )
        entry["n_total"] += 1
        entry[f"n_{result.status}"] = entry.get(f"n_{result.status}", 0) + 1
        if result.status != "ok":
            continue
        for key, value in result.scores.items():
            entry["_values"].setdefault(key, []).append(float(value))

    for entry in summary.values():
        for key, values in entry.pop("_values").items():
            entry["metrics"][key] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return summary


def to_markdown(summary: dict[str, dict]) -> str:
    lines = ["# Evaluation summary", ""]
    for probe in sorted(summary):
        entry = summary[probe]
        lines.append(f"## {probe}")
        lines.append("")
        lines.append(
            f"n = {entry['n_ok']} scored / {entry['n_total']} attempted "
            f"({entry['n_skipped']} skipped, {entry['n_error']} errored)"
        )
        lines.append("")
        if entry["metrics"]:
            lines.append("| metric | mean | std | n |")
            lines.append("|---|---|---|---|")
            for name in sorted(entry["metrics"]):
                stats = entry["metrics"][name]
                lines.append(f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} |")
        else:
            lines.append("_no scored samples_")
        lines.append("")
    return "\n".join(lines)


def to_csv(results: list[Result]) -> str:
    keys = sorted({k for r in results for k in r.scores})
    rows = [",".join(["sample_id", "probe", "status", *keys])]
    for result in results:
        values = ["" if k not in result.scores else f"{result.scores[k]:.6f}" for k in keys]
        rows.append(",".join([result.sample_id, result.probe, result.status, *values]))
    return "\n".join(rows) + "\n"


def write_report(results: list[Result], output_dir: Path, probes: list[str] | None = None) -> dict[str, dict]:
    summary = aggregate(results)
    for probe in probes or []:
        summary.setdefault(probe, {"n_total": 0, "n_ok": 0, "n_skipped": 0, "n_error": 0, "metrics": {}})
    dump_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (output_dir / "scores.csv").write_text(to_csv(results), encoding="utf-8")
    return summary
