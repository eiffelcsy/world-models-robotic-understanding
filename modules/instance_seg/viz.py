"""Turning instance masks into something you can look at.

Shared by the ground-truth store, the preview and the scan, so the masks the
pipeline scores and the masks you inspect are drawn the same way.
"""

from __future__ import annotations

import numpy as np

# Distinct enough to count instances by eye at a glance.
PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    ],
    dtype=np.float32,
)


def overlay(frame: np.ndarray, masks: list[np.ndarray], alpha: float = 0.55) -> np.ndarray:
    canvas = frame.astype(np.float32)
    for index, mask in enumerate(masks):
        color = PALETTE[index % len(PALETTE)]
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
    return np.clip(canvas, 0, 255).astype(np.uint8)
