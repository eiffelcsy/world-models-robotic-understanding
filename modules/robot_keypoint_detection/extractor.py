"""Generated frame -> keypoints.

The probe's convention: the model places one small, fully saturated dot of a
known color on each robot joint and holds it. The decoder finds the dots.

With a `palette`, each entry is a distinct color the prompt asked for, so the
dot for keypoint `i` is the largest blob nearest that color. Without one, the
decoder returns every colored blob's centroid in reading order, which is enough
when the metric matches nearest-first.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from wmcore.utils import read_image, resize_to
from wmeval import Extractor, ProbeData, register
from wmeval.types import RunContext, Sample

HUE_STEPS = 180


@register("extractor", "color_points")
class ColorPoints(Extractor):
    """Colored dots in a generated frame -> `ProbeData("points", {"points": ...})`.

    Params, all settable from probe.yaml:

      palette         list of hex colors ("#ff0000") the prompt asked for, in
                      order; null -> any saturated blob
      hue_tolerance   how far from a palette color (degrees) still counts
      min_saturation  a dot is saturated; the scene is not
      min_value       drops shadow and near-black, where hue is meaningless
      smooth          median blur (px) applied first
      cleanup         morphological open+close kernel (px)
      min_area        smallest blob (px) that is a dot
      max_area        largest blob (px) that is a dot
      dilate          final dilation (px) before centroids
      max_points      keep at most this many, largest first
      min_change      a dot pixel must differ from the input frame by at least this
                      much (largest RGB channel difference, 0-255); 0 turns it off.
                      Without it, saturated scene objects (a red bottle, an orange
                      carton) decode as dots.
    """

    def __init__(
        self,
        palette: list[str] | None = None,
        hue_tolerance: float = 22.0,
        min_saturation: float = 0.5,
        min_value: float = 0.2,
        smooth: int = 3,
        cleanup: int = 3,
        min_area: int = 8,
        max_area: int = 20000,
        dilate: int = 0,
        max_points: int = 32,
        min_change: float = 0.0,
    ) -> None:
        super().__init__(
            palette=palette,
            hue_tolerance=hue_tolerance,
            min_saturation=min_saturation,
            min_value=min_value,
            smooth=smooth,
            cleanup=cleanup,
            min_area=min_area,
            max_area=max_area,
            dilate=dilate,
            max_points=max_points,
            min_change=min_change,
        )

    def __call__(self, frame: np.ndarray, sample: Sample, ctx: RunContext) -> ProbeData:
        reference = read_image(sample.input_frame) if self.min_change > 0 else None
        points, labels, info = self.decode(frame, reference)
        return ProbeData("points", {"points": points, "labels": labels}, info)

    def decode(self, frame: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        height, width = frame.shape[:2]
        ksize = int(self.smooth)
        if ksize >= 3:
            frame = cv2.medianBlur(frame, ksize + 1 - ksize % 2)
            if reference is not None:
                reference = cv2.medianBlur(np.ascontiguousarray(reference), ksize + 1 - ksize % 2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        self._allowed = _changed(frame, reference, float(self.min_change))

        if self.palette:
            points, labels = self._by_palette(hsv)
        else:
            points, labels = self._by_blob(hsv)

        # Palette order is the keypoint order; blob mode already sorts by area.
        points = points[: self.max_points]
        labels = labels[: self.max_points]

        return points, labels, {
            "source": "color_points",
            "frame_shape": [height, width],
            "n_points": len(points),
            "labels": labels,
        }

    def _by_palette(self, hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
        points: list[list[float]] = []
        labels: list[str] = []
        for hex_color in self.palette:
            center = _hue_of(hex_color)
            mask = self._painted(hsv)
            distance = np.abs(hsv[..., 0].astype(np.int16) - round(center))
            distance = np.minimum(distance, HUE_STEPS - distance)
            mask &= distance <= self.hue_tolerance * HUE_STEPS / 360
            centroid = self._largest_centroid(mask)
            if centroid is not None:
                points.append(centroid)
                labels.append(hex_color)
        return np.asarray(points, dtype=np.float32).reshape(-1, 2), labels

    def _by_blob(self, hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
        mask = self._painted(hsv)
        cleaned = self._clean(mask.astype(np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        found: list[tuple[float, float, float]] = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if self.min_area <= area <= self.max_area:
                found.append((area, float(centroids[label][0]), float(centroids[label][1])))
        found.sort(reverse=True)
        points = np.asarray([[x, y] for _, x, y in found], dtype=np.float32).reshape(-1, 2)
        return points, [f"point_{i}" for i in range(len(points))]

    def _painted(self, hsv: np.ndarray) -> np.ndarray:
        painted = (hsv[..., 1] >= round(self.min_saturation * 255)) & (
            hsv[..., 2] >= round(self.min_value * 255)
        )
        allowed = getattr(self, "_allowed", None)
        return painted if allowed is None else painted & allowed

    def _clean(self, binary: np.ndarray) -> np.ndarray:
        ksize = int(self.cleanup)
        if ksize >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if self.dilate >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
            binary = cv2.dilate(binary, kernel)
        return binary

    def _largest_centroid(self, mask: np.ndarray) -> list[float] | None:
        cleaned = self._clean(mask.astype(np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        best, best_area = None, 0.0
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if self.min_area <= area <= self.max_area and area > best_area:
                best, best_area = centroids[label], area
        return [float(best[0]), float(best[1])] if best is not None else None


def _hue_of(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    rgb = np.array([[int(value[i:i + 2], 16) for i in (0, 2, 4)]], dtype=np.uint8).reshape(1, 1, 3)
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])


def _changed(frame: np.ndarray, reference: np.ndarray | None, min_change: float) -> np.ndarray | None:
    """Pixels that differ from the input frame by at least `min_change`, or None when off."""
    if reference is None or min_change <= 0:
        return None
    if reference.shape[:2] != frame.shape[:2]:
        reference = resize_to(reference, frame.shape[:2])
    return np.abs(frame.astype(np.int16) - reference.astype(np.int16)).max(axis=2) >= min_change
