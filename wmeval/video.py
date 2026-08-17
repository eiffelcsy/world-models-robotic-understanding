"""Readouts: pick the one frame of the generated video that carries the answer.

The world model is asked to bring an overlay in and then hold still, so the
answer lives in the tail of the clip. Which tail frame to take is a per-probe
choice, so readouts are registered components configured from YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .registry import register


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


class Readout:
    """Base class: video path -> (RGB frame, info)."""

    def __call__(self, video: Path) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


@register("readout", "last_frame")
class LastFrame(Readout):
    """The final frame. The default for a hold-still overlay probe."""

    def __init__(self, back_off: int = 1) -> None:
        self.back_off = max(1, back_off)

    def __call__(self, video: Path) -> tuple[np.ndarray, dict]:
        capture, count, fps = _open(video)
        try:
            index = max(0, count - self.back_off)
            return _read_at(capture, index), {"frame_index": index, "n_frames": count, "fps": fps}
        finally:
            capture.release()


@register("readout", "fixed_time")
class FixedTime(Readout):
    """The frame at `time_s`, clamped to the clip. Use when the prompt names a
    timing anchor ("fully in place by 00:06") and later frames may drift."""

    def __init__(self, time_s: float = 6.0) -> None:
        self.time_s = time_s

    def __call__(self, video: Path) -> tuple[np.ndarray, dict]:
        capture, count, fps = _open(video)
        try:
            index = int(min(max(0, round(self.time_s * fps)), max(0, count - 1)))
            return _read_at(capture, index), {"frame_index": index, "n_frames": count, "fps": fps}
        finally:
            capture.release()


@register("readout", "most_static")
class MostStatic(Readout):
    """The quietest frame in the tail window.

    Scans the last `window_s` seconds and returns the frame whose difference
    from its predecessor is smallest — the overlay has settled and nothing is
    fading. Guards against a final frame caught mid-flicker.
    """

    def __init__(self, window_s: float = 2.0, stride: int = 2) -> None:
        self.window_s = window_s
        self.stride = max(1, stride)

    def __call__(self, video: Path) -> tuple[np.ndarray, dict]:
        capture, count, fps = _open(video)
        try:
            start = max(1, count - int(self.window_s * fps))
            best_index, best_score, best_frame = count - 1, np.inf, None
            previous = _read_at(capture, max(0, start - 1)).astype(np.float32)
            for index in range(start, count, self.stride):
                try:
                    frame = _read_at(capture, index)
                except IOError:
                    break
                score = float(np.abs(frame.astype(np.float32) - previous).mean())
                if score < best_score:
                    best_index, best_score, best_frame = index, score, frame
                previous = frame.astype(np.float32)
            if best_frame is None:
                best_frame = _read_at(capture, max(0, count - 1))
            return best_frame, {
                "frame_index": best_index,
                "n_frames": count,
                "fps": fps,
                "residual_motion": best_score,
            }
        finally:
            capture.release()
