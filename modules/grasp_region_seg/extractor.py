"""Generated frame -> one grasp region mask.

The probe's convention: the model paints the graspable region in one flat,
fully saturated color and holds it. The decoder keeps the painted pixels, cuts
specks, and returns a single binary mask. If the prompt names a specific color,
`hue` pins the decoder to it; left null it accepts any saturated color, which
is the safer default when the model chooses its own.

Saturation alone is not paint: a red plate or a bright label passes it on an
untouched frame, and one on the target object would score as a hit. So a pixel must
also differ from the conditioning frame by `min_change`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from wmcore.utils import read_image, resize_to
from wmeval import Extractor, ProbeData, register
from wmeval.types import RunContext, Sample

HUE_STEPS = 180


@register("extractor", "color_mask")
class ColorMask(Extractor):
    """Painted pixels in a generated frame -> `ProbeData("binary_mask", ...)`.

    Params, all settable from probe.yaml:

      min_saturation  painted pixels are saturated; the untouched scene is not
      min_value       drops shadow and near-black, where hue is meaningless
      hue             optional color to keep, in degrees; null keeps any
      hue_tolerance   how far from `hue` (degrees) still counts
      min_change      how far a pixel must move from the conditioning frame, as
                      the largest RGB channel difference (0-255); 0 turns it off
      smooth          median blur (px) applied first, against codec speckle
      cleanup         morphological open+close kernel (px)
      min_area_frac   drop components below this fraction of the frame
      max_area_frac   drop the mask above this fraction — a repainted background
      largest_only    keep only the largest component
      dilate          final dilation (px)
    """

    def __init__(
        self,
        min_saturation: float = 0.5,
        min_value: float = 0.2,
        hue: float | None = None,
        hue_tolerance: float = 25.0,
        min_change: float = 40.0,
        smooth: int = 5,
        cleanup: int = 5,
        min_area_frac: float = 0.0005,
        max_area_frac: float = 0.9,
        largest_only: bool = False,
        dilate: int = 0,
    ) -> None:
        super().__init__(
            min_saturation=min_saturation,
            min_value=min_value,
            hue=hue,
            hue_tolerance=hue_tolerance,
            min_change=min_change,
            smooth=smooth,
            cleanup=cleanup,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
            largest_only=largest_only,
            dilate=dilate,
        )

    def __call__(self, frame: np.ndarray, sample: Sample, ctx: RunContext) -> ProbeData:
        reference = read_image(sample.input_frame) if self.min_change > 0 else None
        mask, info = self.decode(frame, reference)
        return ProbeData("binary_mask", {"mask": mask}, info)

    def decode(self, frame: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = frame.shape[:2]
        pixels = height * width

        ksize = int(self.smooth)
        if ksize >= 3:
            frame = cv2.medianBlur(frame, ksize + 1 - ksize % 2)
            if reference is not None:
                reference = cv2.medianBlur(np.ascontiguousarray(reference), ksize + 1 - ksize % 2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        painted = (hsv[..., 1] >= round(self.min_saturation * 255)) & (
            hsv[..., 2] >= round(self.min_value * 255)
        )
        if self.hue is not None:
            center = self.hue * HUE_STEPS / 360
            distance = np.abs(hsv[..., 0].astype(np.int16) - round(center))
            distance = np.minimum(distance, HUE_STEPS - distance)
            painted &= distance <= self.hue_tolerance * HUE_STEPS / 360
        if reference is not None:
            if reference.shape[:2] != (height, width):
                reference = resize_to(reference, (height, width))
            change = np.abs(frame.astype(np.int16) - reference.astype(np.int16)).max(axis=2)
            painted &= change >= self.min_change

        binary = painted.astype(np.uint8)
        csize = int(self.cleanup)
        if csize >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csize, csize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        minimum, maximum = self.min_area_frac * pixels, self.max_area_frac * pixels
        keep = np.zeros_like(binary)
        found: list[tuple[int, int]] = []  # (area, label)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if minimum <= area <= maximum:
                keep[labels == label] = 1
                found.append((area, label))
        if self.largest_only and found:
            _, best = max(found)
            keep = (labels == best).astype(np.uint8)

        if self.dilate >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate, self.dilate))
            keep = cv2.dilate(keep, kernel)

        mask = keep.astype(bool)
        return mask, {
            "source": "color_mask",
            "frame_shape": [height, width],
            "painted_frac": round(float(painted.mean()), 4),
            "area_frac": round(float(mask.mean()), 4),
            "n_components": len(found),
        }
