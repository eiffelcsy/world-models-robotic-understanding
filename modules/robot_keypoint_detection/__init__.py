"""Robot keypoint detection probe (LIBERO only).

Asks the world model to place a distinct colored dot on each robot joint, and
scores the dot centers it decodes against the Panda links' exact image positions
from a simulator replay.

    gt.py         simulator replay -> the links' 2D keypoints
    extractor.py  colored dots -> keypoints
    metrics.py    prediction vs ground truth: PCK and mean distance
    scan.py       CLI: best frame vs last frame of one generation

Importing the package registers the components `probe.yaml` names.
"""

from . import extractor, gt, metrics  # noqa: F401
