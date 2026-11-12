"""Grasp region segmentation probe (LIBERO only).

Asks the world model to paint the graspable region of the object to pick up, and
scores it against the grasp recorded in the demonstration, on the exact mask of
the object it picked from a simulator replay.

    gt.py         a recorded grasp + its object's mask
    extractor.py  one flat painted color -> one binary mask
    metrics.py    predicted region vs the demonstrated grasp (grasp_hit)

Importing the package registers the components `probe.yaml` names.
"""

from . import extractor, gt, metrics  # noqa: F401
