"""Generated frame -> a depth field.

The probe's prompt replaces the frame with a grayscale depth map in which near
surfaces are bright and far surfaces are dark. The decoder is the inverse of
that convention: read luminance, invert it so larger means further, and put it
on a stable [0, 1] range with a percentile clip, because a few specular or
shadowed pixels should not set the scale.

Depth from a generative model is relative, like DA3's, so the exact magnitude
does not matter — metrics align the two before scoring.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from wmeval import Extractor, ProbeData, register
from wmeval.types import RunContext, Sample


@register("extractor", "gray_depth")
class GrayDepth(Extractor):
    """A grayscale depth map -> `ProbeData("depth", {"depth": float HxW})`.

    Params, all settable from probe.yaml:

      near_is_bright   the prompt asks for bright = close; flip for the reverse
      clip_percentile  percentile trimmed from each end before scaling to [0, 1]
      smooth           median blur (px) applied first, against codec speckle
      invert           negate the decoded field (an escape hatch for a prompt
                       that ended up far = bright despite the wording)

    The decoder deliberately cannot read a colormapped (turbo/viridis) depth
    image; it is written together with a grayscale prompt. Change one and the
    other together.
    """

    def __init__(
        self,
        near_is_bright: bool = True,
        clip_percentile: float = 2.0,
        smooth: int = 5,
        invert: bool = False,
    ) -> None:
        super().__init__(
            near_is_bright=near_is_bright,
            clip_percentile=clip_percentile,
            smooth=smooth,
            invert=invert,
        )

    def __call__(self, frame: np.ndarray, sample: Sample, ctx: RunContext) -> ProbeData:
        depth, info = self.decode(frame)
        return ProbeData("depth", {"depth": depth}, info)

    def decode(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ksize = int(self.smooth)
        if ksize >= 3:
            gray = cv2.medianBlur(gray, ksize + 1 - ksize % 2)

        luminance = gray.astype(np.float32) / 255.0
        depth = (1.0 - luminance) if self.near_is_bright else luminance
        if self.invert:
            depth = 1.0 - depth

        finite = np.isfinite(depth)
        low, high = np.percentile(depth[finite], [self.clip_percentile, 100 - self.clip_percentile])
        if high <= low:
            high = low + 1e-6
        depth = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)

        return depth, {
            "source": "gray_depth",
            "frame_shape": list(frame.shape[:2]),
            "near_is_bright": bool(self.near_is_bright),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
        }
