"""Ground truth recorded by a simulator replay, shared by the probes that use it.

scripts/libero_replay/render_gt.py replays each LIBERO sample in MuJoCo and saves,
per sample, one `.npz` beside the frames (`<data>/sim/<sample_id>.npz`) holding
what the probes score against, all on the 1280x720 canvas of the conditioning
frame:

  depth           float32 (H, W), metres along the camera axis; NaN on the bars
  segment         int16 (H, W), index into `object_names` of the object each pixel
                  shows. 0 is nothing named, 1 the robot, the rest LIBERO objects
  object_names    the names those indices stand for
  keypoints       float32 (7, 2), the Panda links in `keypoint_labels`, in pixels
  world_to_image  float64 (3, 4), the render's camera
  grasp_fingers_world  float64 (2, 3), the finger pads' inner faces at the grasp
                  frame, in world coordinates

The sample's meta points at the file with `sim_file`, relative to the folder
holding the frames folder, and names the objects with `sim_objects` (the task's
objects of interest) and `sim_grasp_object` (the one the fingers closed on).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .types import Sample


def sim_record(sample: Sample, key: str = "sim_file") -> dict[str, Any]:
    """The sample's replay record. Raises if the sample has none."""
    relative = sample.meta.get(key)
    if not relative:
        raise RuntimeError(
            f"{sample.sample_id} has no `{key}` in its meta: run scripts/libero_replay/render_gt.py"
        )
    path = Path(relative)
    if not path.is_absolute():
        path = Path(sample.input_frame).resolve().parent.parent / path
    return _load(str(path))


def object_mask(record: dict[str, Any], name: str) -> np.ndarray:
    """Boolean mask of every pixel showing the object `name`."""
    names = [str(n) for n in record["object_names"]]
    if name not in names:
        raise KeyError(f"{name!r} is not a replayed object; known: {names}")
    return record["segment"] == names.index(name)


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"simulator record {path} is missing: run scripts/libero_replay/render_gt.py")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
