"""Depth perception probe.

Asks the world model to replace the conditioning frame with a depth map and
scores the depth it decodes against Depth Anything 3 run on the input frame.

    gt.py         Depth Anything 3 -> pseudo ground truth depth field
    model.py      the checkpoint and the one call into depth_anything_3
    extractor.py  grayscale depth map -> a depth field
    metrics.py    prediction vs ground truth: scale-invariant depth error
    preview.py    CLI: look at the ground truth on a frame
    scan.py       CLI: best frame vs last frame of one generation

Importing the package registers the components `probe.yaml` names; the heavy
`depth_anything_3` import stays inside `model.py` and only happens when this
probe actually runs.
"""

from . import extractor, gt, metrics  # noqa: F401
