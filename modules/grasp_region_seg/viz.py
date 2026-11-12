"""Turning a grasp region mask into something you can look at."""

from __future__ import annotations

import numpy as np

COLOR = np.array([255, 225, 25], dtype=np.float32)


def overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    canvas = frame.astype(np.float32)
    canvas[mask] = (1 - alpha) * canvas[mask] + alpha * COLOR
    return np.clip(canvas, 0, 255).astype(np.uint8)
