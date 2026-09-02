"""The manifest: which frames, which videos, which ground-truth assets.

This is the only place the pipeline touches a specific world model's output
layout. Point it at a directory of videos from Veo, Cosmos, or anything else and
the rest of the pipeline is unchanged.

    samples:
      - sample_id: AUTOLab_2023-10-27__ext1_f000021
        input_frame: veo/runs/pilot/frames/AUTOLab_...png
        videos:
          depth_perception: veo/runs/pilot/depth/....mp4
          "*": fallback_used_by_every_probe.mp4      # optional
        gt_assets:
          depth: gt/AUTOLab_..._depth.npy            # optional, probe-defined keys
        meta: {episode: ..., camera: ext1, frame_index: 21}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wmcore.config import read_yaml, resolver, root_of
from wmcore.manifest import write_manifest  # noqa: F401  (re-exported; the schema is shared)

from .types import Sample


def load_manifest(path: str | Path) -> list[Sample]:
    path = Path(path).resolve()
    raw = read_yaml(path)
    resolve = resolver(root_of(path, raw, default="."))

    samples: list[Sample] = []
    for entry in raw.get("samples") or []:
        samples.append(
            Sample(
                sample_id=str(entry["sample_id"]),
                input_frame=resolve(entry["input_frame"]),
                videos={k: resolve(v) for k, v in (entry.get("videos") or {}).items()},
                gt_assets={k: resolve(v) for k, v in (entry.get("gt_assets") or {}).items()},
                meta=entry.get("meta") or {},
            )
        )
    if not samples:
        raise ValueError(f"{path}: manifest lists no samples")
    return samples


def scan_run_dir(
    run_dir: Path,
    frames_dir: Path,
    *,
    probes: list[str] | None = None,
    video_ext: str = ".mp4",
    frame_ext: str = ".png",
) -> dict[str, Any]:
    """Build a manifest dict from `run_dir/<probe>/<sample_id>.mp4` + `frames_dir/<sample_id>.png`.

    Sample ids come from the video stem; a video with no matching frame is
    reported rather than silently dropped, since a missing conditioning frame
    means no ground truth can be derived for it.
    """
    run_dir, frames_dir = Path(run_dir), Path(frames_dir)
    frames = {p.stem: p for p in sorted(frames_dir.glob(f"*{frame_ext}"))}

    videos: dict[str, dict[str, Path]] = {}
    orphans: list[str] = []
    for probe_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if probe_dir.resolve() == frames_dir.resolve():
            continue  # the frames directory usually sits inside the run directory
        if probes is not None and probe_dir.name not in probes:
            continue
        for video in sorted(probe_dir.glob(f"*{video_ext}")):
            sample_id = _match_frame(video.stem, frames)
            if sample_id is None:
                orphans.append(str(video))
                continue
            videos.setdefault(sample_id, {})[probe_dir.name] = video

    samples = [
        {
            "sample_id": sample_id,
            "input_frame": str(frames[sample_id]),
            "videos": {probe: str(path) for probe, path in sorted(by_probe.items())},
        }
        for sample_id, by_probe in sorted(videos.items())
    ]
    return {"root": "/", "samples": samples, "_orphan_videos": orphans}


def _match_frame(video_stem: str, frames: dict[str, Path]) -> str | None:
    """Videos are often named `<probe>__<sample_id>`; fall back to a suffix match."""
    if video_stem in frames:
        return video_stem
    for sample_id in frames:
        if video_stem.endswith(sample_id):
            return sample_id
    return None
