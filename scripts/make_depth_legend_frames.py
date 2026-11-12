"""Copy the LIBERO conditioning frames with a depth colour-bar legend drawn in.

For the `paper` depth arm only: Wiedemer et al. (2025, Fig. 62) prompt Veo with
"The exact color map to use is provided on the right side of the image" and paste
a vertical magma colour bar on the input image's right edge, light (near) at the
top and black (far) at the bottom. LIBERO frames are a 720x720 render on a
1280x720 canvas with black bars at x < 280 and x >= 1000, so the bar goes in the
right-hand bar and covers no scene pixel. Those pixels are NaN in the depth ground
truth and never scored.

    python scripts/make_depth_legend_frames.py
    -> data/libero/frames_depth_legend/<sample_id>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def legend_frame(frame: np.ndarray, x0: int, width: int, margin: int) -> np.ndarray:
    height = frame.shape[0]
    out = frame.copy()
    rows = height - 2 * margin
    # Top is near (light), bottom far (black): 255 at the top row, 0 at the bottom.
    ramp = np.linspace(255, 0, rows).astype(np.uint8)[:, None].repeat(width, axis=1)
    out[margin : margin + rows, x0 : x0 + width] = cv2.applyColorMap(ramp, cv2.COLORMAP_MAGMA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="data/libero/frames")
    parser.add_argument("--out", default="data/libero/frames_depth_legend")
    parser.add_argument("--x0", type=int, default=1200, help="left edge of the bar (inside the black bar)")
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--margin", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.src).glob("*.png")):
        frame = cv2.imread(str(path))
        if frame[:, args.x0 : args.x0 + args.width].max() > 0:
            raise ValueError(f"{path.name}: the legend would cover scene pixels at x={args.x0}")
        cv2.imwrite(str(out / path.name), legend_frame(frame, args.x0, args.width, args.margin))
        print(out / path.name)


if __name__ == "__main__":
    main()
