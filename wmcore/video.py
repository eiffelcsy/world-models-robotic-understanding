"""Reading frames out of a video file.

wmeval reads clips to score them and wmgen reads them to check that what came
back is a video at all, so the primitives live here and the two pipelines add
their own meaning on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


def _open(path: str | Path) -> tuple[cv2.VideoCapture, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    return capture, count, fps


def _read_at(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise IOError(f"could not read frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_frames(video: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Every `stride`-th frame as (index, RGB), read sequentially.

    The final frame is always yielded whatever the stride lands on, so a strided
    pass still sees the one frame a `last_frame` readout would have taken.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"could not open video: {video}")
    stride = max(1, int(stride))
    index = 0
    held: tuple[int, np.ndarray] | None = None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % stride:
                held = (index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            else:
                held = None
                yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    if held is not None:
        yield held


def probe_video(path: str | Path) -> dict[str, Any]:
    """What a container claims about itself: frames, rate, size.

    The header's frame count is what a decoder reports and is not always what
    reads back, so `n_frames` here is a claim to sanity-check rather than a
    guarantee; a check that cares counts frames itself.
    """
    capture, count, fps = _open(path)
    try:
        return {
            "n_frames": count,
            "fps": fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def count_frames(path: str | Path) -> int:
    """Frames that actually decode. Slower than the header, and truthful."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IOError(f"could not open video: {path}")
    n = 0
    try:
        while capture.grab():
            n += 1
    finally:
        capture.release()
    return n
