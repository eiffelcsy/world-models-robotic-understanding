"""Aggregate per-sample results into the summary tables.

One run answers three different questions, so the report has three views:

  per task      one row per probe: the headline score on the last frame (the
                readout) and on the best frame (the frame the ground truth scores
                highest, an upper bound), plus the invalid rate from the gates
  per level     the tasks grouped into Scene / Interaction / Robot State
  per scenario  the headline score of every task across RoboTrustBench's
                scenario axis (Normal, Constraint-Sensitive, Counterfactual,
                Adversarial)

Only `status == "ok"` rows feed the metric means, and an **invalid rate** is
reported beside every score: the share of scored rollouts a fidelity gate
rejected. A high mean on a high invalid rate is not a result.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from wmcore.utils import dump_json

from .types import Result

_PREFIX = "best_"
_NOT_A_SCORE = {f"{_PREFIX}frame_index", "frame_index"}


def _metric_stats(results: list[Result]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for result in results:
        if result.status != "ok":
            continue
        for key, value in result.scores.items():
            values.setdefault(key, []).append(float(value))
    return {
        key: {"mean": mean(vs), "std": pstdev(vs) if len(vs) > 1 else 0.0, "n": len(vs)}
        for key, vs in values.items()
    }


def _block(results: list[Result]) -> dict[str, Any]:
    n_ok = sum(result.status == "ok" for result in results)
    n_invalid = sum(1 for result in results if result.status == "ok" and not result.valid)
    return {
        "n_total": len(results),
        "n_ok": n_ok,
        "n_skipped": sum(result.status == "skipped" for result in results),
        "n_error": sum(result.status == "error" for result in results),
        "n_invalid": n_invalid,
        "invalid_rate": (n_invalid / n_ok) if n_ok else 0.0,
        "metrics": _metric_stats(results),
    }


def _headline(block: dict[str, Any], configured: str) -> str | None:
    metrics = block["metrics"]
    if configured and configured in metrics:
        return configured
    for name in sorted(metrics):  # prefer a score that has a best-frame twin
        if name not in _NOT_A_SCORE and not name.startswith(_PREFIX) and f"{_PREFIX}{name}" in metrics:
            return name
    for name in sorted(metrics):
        if name not in _NOT_A_SCORE and not name.startswith(_PREFIX):
            return name
    return None


def aggregate(
    results: list[Result],
    levels: dict[str, str] | None = None,
    headlines: dict[str, str] | None = None,
) -> dict[str, Any]:
    levels = levels or {}
    headlines = headlines or {}
    probes = sorted({result.probe for result in results})

    summary: dict[str, Any] = {"probes": {}, "levels": {}, "scenarios": {}}

    for probe in probes:
        block = _block([r for r in results if r.probe == probe])
        block["level"] = levels.get(probe, "")
        block["headline"] = _headline(block, headlines.get(probe, ""))
        summary["probes"][probe] = block

    for probe in probes:
        block = summary["probes"][probe]
        level = block["level"] or "unlevelled"
        entry = summary["levels"].setdefault(
            level, {"probes": {}, "n_total": 0, "n_ok": 0, "n_invalid": 0, "invalid_rate": 0.0}
        )
        entry["probes"][probe] = {
            "headline": block["headline"],
            "n_ok": block["n_ok"],
            "invalid_rate": block["invalid_rate"],
            "last": _mean(block, block["headline"]),
            "best": _mean(block, f"{_PREFIX}{block['headline']}") if block["headline"] else None,
        }
        for key in ("n_total", "n_ok", "n_invalid"):
            entry[key] += block[key]
    for entry in summary["levels"].values():
        entry["invalid_rate"] = (entry["n_invalid"] / entry["n_ok"]) if entry["n_ok"] else 0.0

    categories = sorted({r.meta.get("category") for r in results if r.meta.get("category")})
    for category in categories:
        row: dict[str, Any] = {}
        for probe in probes:
            block = _block([r for r in results if r.probe == probe and r.meta.get("category") == category])
            name = _headline(block, headlines.get(probe, ""))
            row[probe] = {
                "headline": name,
                "n_ok": block["n_ok"],
                "invalid_rate": block["invalid_rate"],
                "value": _mean(block, name) if name else None,
            }
        summary["scenarios"][category] = row

    return summary


def _mean(block: dict[str, Any], name: str | None) -> float | None:
    if not name:
        return None
    stats = block["metrics"].get(name)
    return stats["mean"] if stats else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Evaluation summary", ""]
    lines += _task_overview(summary)
    lines += _per_task_tables(summary)
    lines += _per_level(summary)
    lines += _per_scenario(summary)
    return "\n".join(lines)


def _task_overview(summary: dict[str, Any]) -> list[str]:
    lines = ["## Per task — last frame vs best frame", ""]
    lines.append("| task | level | n_ok | invalid | last frame | best frame | gap |")
    lines.append("|---|---|---|---|---|---|---|")
    for probe in sorted(summary["probes"]):
        block = summary["probes"][probe]
        last = _mean(block, block["headline"])
        best = _mean(block, f"{_PREFIX}{block['headline']}") if block["headline"] else None
        gap = (best - last) if (last is not None and best is not None) else None
        head = block["headline"] or "—"
        lines.append(
            f"| {probe} ({head}) | {block['level'] or '—'} | {block['n_ok']} | "
            f"{block['invalid_rate'] * 100:.0f}% | {_fmt(last)} | {_fmt(best)} | {_fmt(gap)} |"
        )
    lines.append("")
    return lines


def _per_task_tables(summary: dict[str, Any]) -> list[str]:
    lines = ["## Per task — all metrics", ""]
    for probe in sorted(summary["probes"]):
        block = summary["probes"][probe]
        lines.append(f"### {probe}")
        lines.append("")
        lines.append(
            f"n = {block['n_ok']} scored / {block['n_total']} attempted "
            f"({block['n_invalid']} invalid, {block['n_skipped']} skipped, {block['n_error']} errored); "
            f"invalid rate {block['invalid_rate'] * 100:.0f}%"
        )
        lines.append("")
        if block["metrics"]:
            lines.append("| metric | mean | std | n |")
            lines.append("|---|---|---|---|")
            for name in sorted(block["metrics"]):
                stats = block["metrics"][name]
                lines.append(f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} |")
        else:
            lines.append("_no scored samples_")
        lines.append("")
    return lines


def _per_level(summary: dict[str, Any]) -> list[str]:
    lines = ["## Per level", ""]
    lines.append("| level | tasks | n_ok | invalid | headline scores (last → best frame) |")
    lines.append("|---|---|---|---|---|")
    order = ["Scene", "Interaction", "Robot State"]
    for level in sorted(summary["levels"], key=lambda l: (order.index(l) if l in order else len(order), l)):
        entry = summary["levels"][level]
        cells = []
        for probe, info in sorted(entry["probes"].items()):
            cells.append(f"{probe}: {_fmt(info['last'])} → {_fmt(info['best'])}")
        lines.append(
            f"| {level} | {len(entry['probes'])} | {entry['n_ok']} | "
            f"{entry['invalid_rate'] * 100:.0f}% | {'; '.join(cells)} |"
        )
    lines.append("")
    return lines


def _per_scenario(summary: dict[str, Any]) -> list[str]:
    if not summary["scenarios"]:
        return []
    probes = sorted(summary["probes"])
    lines = ["## Per scenario — headline score", ""]
    lines.append("| scenario | " + " | ".join(probes) + " |")
    lines.append("|---" * (len(probes) + 1) + "|")
    for category in sorted(summary["scenarios"]):
        row = summary["scenarios"][category]
        cells = []
        for probe in probes:
            cell = row.get(probe, {})
            if cell.get("value") is None:
                cells.append("—")
            else:
                cells.append(f"{cell['value']:.3f} (n={cell['n_ok']})")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def to_csv(results: list[Result]) -> str:
    keys = sorted({k for r in results for k in r.scores})
    header = ["sample_id", "probe", "status", "valid", "invalid_reason", "category", *keys]
    rows = [",".join(header)]
    for result in results:
        values = ["" if k not in result.scores else f"{result.scores[k]:.6f}" for k in keys]
        rows.append(
            ",".join(
                [
                    result.sample_id,
                    result.probe,
                    result.status,
                    "1" if result.valid else "0",
                    (result.invalid_reason or "").replace(",", ";"),
                    str(result.meta.get("category", "")),
                    *values,
                ]
            )
        )
    return "\n".join(rows) + "\n"


def write_report(
    results: list[Result],
    output_dir: Path,
    probes: list[str] | None = None,
    levels: dict[str, str] | None = None,
    headlines: dict[str, str] | None = None,
) -> dict[str, Any]:
    summary = aggregate(results, levels=levels, headlines=headlines)
    for name in probes or []:
        summary["probes"].setdefault(
            name,
            {
                "n_total": 0, "n_ok": 0, "n_skipped": 0, "n_error": 0, "n_invalid": 0,
                "invalid_rate": 0.0, "metrics": {}, "level": (levels or {}).get(name, ""), "headline": None,
            },
        )
    dump_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (output_dir / "scores.csv").write_text(to_csv(results), encoding="utf-8")
    return summary
