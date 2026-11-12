"""Turning depth fields into something you can look at."""

from __future__ import annotations

import cv2
import numpy as np


def _normalize(depth: np.ndarray, clip_percentile: float = 2.0) -> np.ndarray:
    finite = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(finite)
    if not valid.any():
        return np.zeros_like(finite, dtype=np.float32)
    low, high = np.percentile(finite[valid], [clip_percentile, 100 - clip_percentile])
    if high <= low:
        high = low + 1e-6
    return np.clip((finite - low) / (high - low), 0.0, 1.0)


def depth_colormap(depth: np.ndarray, invert: bool = True, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Depth (H, W) -> RGB uint8.

    `invert` makes near surfaces bright, which is the convention the prompt asks
    the world model for, so a ground-truth view and a decoded view read the same
    way.
    """
    normalized = _normalize(depth)
    if invert:
        normalized = 1.0 - normalized
    gray = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)[..., ::-1].copy()


def overlay_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    canvas = image.copy()
    for x, y in np.asarray(points, dtype=float):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, -1)
    return canvas
