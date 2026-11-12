"""Drawing keypoints."""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    ],
    dtype=np.uint8,
)


def overlay_points(image: np.ndarray, points: np.ndarray, radius: int = 5) -> np.ndarray:
    canvas = image.copy()
    for index, point in enumerate(np.asarray(points, dtype=float).reshape(-1, 2)):
        if not (np.isfinite(point[0]) and np.isfinite(point[1])):
            continue
        color = tuple(int(v) for v in PALETTE[index % len(PALETTE)])
        cv2.circle(canvas, (int(round(point[0])), int(round(point[1]))), radius, color, -1)
    return canvas
