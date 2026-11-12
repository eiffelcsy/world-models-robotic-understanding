"""Checks: is what came back usable at all?

The `Metric` analogue, and deliberately a much weaker one. A metric scores how
well a model understood a scene; a check only decides whether a file is a video
that wmeval can open. Everything about quality is the eval pipeline's job — a
check that started grading generations would be scoring the model twice, once
here with no ground truth.

What they buy is that a 0-byte download or a one-frame clip fails here, named,
next to the request that produced it, instead of surfacing an hour later as an
opaque `status: error` row in `results.json`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wmcore.registry import register
from wmcore.utils import read_image, resize_to
from wmcore.video import _open, _read_at, count_frames, probe_video

from .components import ClipCheck
from .types import Request

log = logging.getLogger("wmgen")


@register("check", "playable")
class Playable(ClipCheck):
    """The file opens, decodes, and is long and large enough to score.

    `min_frames` guards the failure mode that matters: an API returning a
    truncated or single-frame clip, which is unusable but perfectly valid mp4.
    The header's frame count is not trusted for that — a container often claims
    more frames than decode — so when the claim is short the frames are counted
    for real before failing.
    """

    def __init__(self, min_frames: int = 8, min_width: int = 64, min_height: int = 64) -> None:
        super().__init__(min_frames=min_frames, min_width=min_width, min_height=min_height)

    def __call__(self, path: Path, request: Request) -> dict:
        if not path.exists() or path.stat().st_size == 0:
            raise IOError(f"empty or missing clip: {path}")

        info = probe_video(path)
        if info["n_frames"] < self.min_frames:
            info["n_frames"] = count_frames(path)
        if info["n_frames"] < self.min_frames:
            raise IOError(f"clip has {info['n_frames']} frames, need at least {self.min_frames}")
        if info["width"] < self.min_width or info["height"] < self.min_height:
            raise IOError(
                f"clip is {info['width']}x{info['height']}, need at least "
                f"{self.min_width}x{self.min_height}"
            )

        # Not `duration_s`: that name is the duration *asked* for, and a clip
        # that came back shorter is exactly what this number is here to show.
        info["clip_seconds"] = round(info["n_frames"] / (info["fps"] or 24.0), 2)
        info["size_mb"] = round(path.stat().st_size / 1e6, 2)
        return info


@register("check", "changed")
class Changed(ClipCheck):
    """The clip's last frame differs from the frame it was conditioned on.

    A model that ignored the prompt returns the conditioning frame, held. That
    is a valid video and a wasted generation, and it is worth catching before it
    reaches the scorer — where, for `instance_seg`, it does not score zero but
    whatever the scene's own saturated colours happen to earn (see the
    calibration note in `modules/instance_seg/README.md`).

    Reported always; only fatal when `min_change` is set above 0. It is a blunt
    pixel difference, so a scene where the model merely re-lit the shot also
    clears it — treat a low number as a prompt to go and look at the clip.
    """

    def __init__(self, min_change: float = 0.0) -> None:
        super().__init__(min_change=min_change)

    def __call__(self, path: Path, request: Request) -> dict:
        capture, count, _ = _open(path)
        try:
            last = _read_at(capture, max(0, count - 1)).astype(np.float32)
        finally:
            capture.release()

        source = read_image(request.frame.path).astype(np.float32)
        if source.shape[:2] != last.shape[:2]:
            source = resize_to(source, last.shape[:2])

        change = float(np.abs(last - source).mean() / 255.0)
        if change < self.min_change:
            raise IOError(
                f"last frame differs from the conditioning frame by only {change:.4f} "
                f"(need {self.min_change}); the model probably ignored the prompt"
            )
        return {"change_from_input": round(change, 4)}
