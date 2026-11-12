"""Ground truth for the robot keypoint probe, from a simulator replay.

The probe asks the world model to dot each robot joint. It is scored against the
exact image positions of the Panda's links at the conditioning frame, which
scripts/libero_replay/render_gt.py records. Only LIBERO samples have them; there
is no estimated keypoint ground truth for real video.
"""

from __future__ import annotations

import numpy as np

from wmeval import GroundTruthSource, ProbeData, register
from wmeval.simulator import sim_record
from wmeval.types import RunContext, Sample

from .viz import overlay_points


@register("gt", "sim_keypoints")
class SimKeypoints(GroundTruthSource):
    """Simulator replay -> the exact image positions of the Panda's keypoints.

    The seven links RoboPEPP and DREAM use (link0, 2, 3, 4, 6, 7 and the hand),
    read from MuJoCo's body positions at the conditioning frame and projected with
    the render's own camera. A link hidden behind something is still where it is,
    so it is kept: the prompt asks for every joint. A link outside the rendered
    view (this camera usually cuts off the upper arm) is left out and named in
    `outside_frame`.
    """

    store_name = "robot_keypoints_sim_gt"
    store_params = ()

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        height, width = input_frame.shape[:2]
        record = sim_record(sample)
        points = np.asarray(record["keypoints"], dtype=np.float32)
        labels = [str(label) for label in record["keypoint_labels"]]
        # Inside the rendered square, not just the canvas: depth is NaN on its bars.
        inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        depth = record["depth"]
        for i in np.flatnonzero(inside):
            inside[i] = np.isfinite(depth[int(points[i, 1]), int(points[i, 0])])
        return ProbeData(
            "points",
            {"points": points[inside], "labels": [l for l, keep in zip(labels, inside) if keep]},
            {
                "source": "simulator",
                "frame_shape": [height, width],
                "n_points": int(inside.sum()),
                "outside_frame": [l for l, keep in zip(labels, inside) if not keep],
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        return overlay_points(input_frame, np.asarray(data["points"], dtype=float))
