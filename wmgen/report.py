"""What a generation run leaves behind.

The important output is not the summary — it is `configs/manifest.yaml`, which
is what `scripts/run_eval.py` consumes next. Everything else here exists so that
a directory of clips can be understood without rerunning anything:

    <output_dir>/gen_summary.md     per-probe counts and check averages
    <output_dir>/gen_summary.json   the same, machine-readable
    <output_dir>/clips.csv          one row per (frame, probe)
    <output_dir>/prompts.jsonl      every rendered prompt (written by --dry-run)
    <manifest_out>                  the manifest wmeval reads
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from wmcore.manifest import write_manifest
from wmcore.utils import dump_json

from .types import Clip, Frame

#: Check outputs that are counts or identifiers rather than measurements, and so
#: are not worth averaging across a run.
_NOT_AVERAGED = {"job_id", "prompt", "prompt_sha", "video", "drifted"}


def build_manifest(frames: Iterable[Frame], clips: list[Clip]) -> dict[str, Any]:
    """The manifest for everything that generated.

    Absolute paths under `root: /`, matching what `make_manifest.py` produces,
    so a manifest written here and one scanned from a directory are the same
    kind of file. A frame whose clips all failed is left out entirely rather
    than written with an empty `videos` block — wmeval would only skip it, and a
    manifest that lists work it cannot do is a manifest that lies about `n`.
    """
    by_sample: dict[str, dict[str, str]] = {}
    for clip in clips:
        if clip.usable:
            by_sample.setdefault(clip.sample_id, {})[clip.probe] = str(clip.path)

    samples = []
    for frame in frames:
        videos = by_sample.get(frame.sample_id)
        if not videos:
            continue
        entry: dict[str, Any] = {
            "sample_id": frame.sample_id,
            "input_frame": str(frame.path),
            "videos": dict(sorted(videos.items())),
        }
        if frame.meta:
            entry["meta"] = frame.meta
        samples.append(entry)

    return {"root": "/", "samples": samples}


def aggregate(clips: list[Clip]) -> dict[str, dict]:
    """Per-probe counts, plus mean/std of whatever the checks measured."""
    summary: dict[str, dict] = {}
    for clip in clips:
        entry = summary.setdefault(
            clip.probe,
            {"n_total": 0, "n_ok": 0, "n_cached": 0, "n_skipped": 0, "n_error": 0, "checks": {}, "_values": {}},
        )
        entry["n_total"] += 1
        entry[f"n_{clip.status}"] = entry.get(f"n_{clip.status}", 0) + 1
        for key, value in clip.info.items():
            if key in _NOT_AVERAGED or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            entry["_values"].setdefault(key, []).append(float(value))

    for entry in summary.values():
        for key, values in entry.pop("_values").items():
            entry["checks"][key] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return summary


def to_markdown(summary: dict[str, dict], backend: str = "") -> str:
    lines = ["# Generation summary", ""]
    if backend:
        lines += [f"backend: `{backend}`", ""]
    for probe in sorted(summary):
        entry = summary[probe]
        lines.append(f"## {probe}")
        lines.append("")
        lines.append(
            f"{entry['n_ok']} generated, {entry['n_cached']} already stored / "
            f"{entry['n_total']} attempted ({entry['n_skipped']} skipped, {entry['n_error']} errored)"
        )
        lines.append("")
        if entry["checks"]:
            lines.append("| check | mean | std | n |")
            lines.append("|---|---|---|---|")
            for name in sorted(entry["checks"]):
                stats = entry["checks"][name]
                lines.append(f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} |")
        else:
            lines.append("_no clips_")
        lines.append("")
    return "\n".join(lines)


def to_csv(clips: list[Clip]) -> str:
    keys = sorted({k for c in clips for k, v in c.info.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    rows = [",".join(["sample_id", "probe", "status", *keys, "path", "error"])]
    for clip in clips:
        values = ["" if k not in clip.info else f"{float(clip.info[k]):.6f}" for k in keys]
        rows.append(
            ",".join(
                [
                    clip.sample_id,
                    clip.probe,
                    clip.status,
                    *values,
                    f'"{clip.path}"' if clip.path else "",
                    f'"{clip.error}"' if clip.error else "",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def write_prompts(clips: list[Clip], output_dir: Path) -> Path:
    """Every rendered prompt, one JSON object per line. The dry run's real output."""
    path = Path(output_dir) / "prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for clip in clips:
            if "prompt" not in clip.info:
                continue
            handle.write(
                json.dumps(
                    {
                        "sample_id": clip.sample_id,
                        "probe": clip.probe,
                        "prompt": clip.info["prompt"],
                        "video": clip.info.get("video", {}),
                        "prompt_sha": clip.info.get("prompt_sha"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def write_report(
    clips: list[Clip],
    frames: Iterable[Frame],
    output_dir: Path,
    manifest_out: Path,
    *,
    probes: list[str] | None = None,
    backend: str = "",
) -> dict[str, dict]:
    summary = aggregate(clips)
    for probe in probes or []:
        summary.setdefault(
            probe, {"n_total": 0, "n_ok": 0, "n_cached": 0, "n_skipped": 0, "n_error": 0, "checks": {}}
        )
    output_dir = Path(output_dir)
    dump_json(output_dir / "gen_summary.json", summary)
    (output_dir / "gen_summary.md").write_text(to_markdown(summary, backend), encoding="utf-8")
    (output_dir / "clips.csv").write_text(to_csv(clips), encoding="utf-8")
    dump_json(output_dir / "clips.json", [c.to_dict() for c in clips])
    write_manifest(manifest_out, build_manifest(frames, clips))
    return summary
