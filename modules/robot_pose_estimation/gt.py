"""Ground truth for the robot pose probe, from the recorded end-effector.

On simulated data the pose is known rather than estimated: `recorded_pose`
projects the end-effector the simulator recorded and needs no model. Only LIBERO
samples carry it; there is no estimated pose ground truth for real video.
"""

from __future__ import annotations

import numpy as np

from wmeval import GroundTruthSource, ProbeData, register
from wmeval.types import RunContext, Sample

from .viz import overlay_points


@register("gt", "recorded_pose")
class RecordedPose(GroundTruthSource):
    """Recorded end-effector pose -> `ProbeData("points", ...)`, exact by construction.

    The two points are the ones the prompt asks for: the grip point, and a point
    `heading_distance` metres ahead of it along the gripper's approach axis, its
    local +z. Reads, from the sample's meta, `eef_pos` and `eef_axis_angle` in the
    world frame and `world_to_image`, a 3x4 matrix to this frame's pixels.
    scripts/fetch_libero.py writes all three.
    """

    store_name = "robot_pose_recorded_gt"
    store_params = ("heading_distance",)

    def __init__(self, heading_distance: float = 0.08) -> None:
        super().__init__(heading_distance=heading_distance)

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        import cv2

        height, width = input_frame.shape[:2]
        position = np.asarray(sample.meta["eef_pos"], dtype=np.float64)
        rotation, _ = cv2.Rodrigues(np.asarray(sample.meta["eef_axis_angle"], dtype=np.float64))
        matrix = np.asarray(sample.meta["world_to_image"], dtype=np.float64)

        ahead = position + self.heading_distance * rotation[:, 2]
        world = np.hstack([np.stack([position, ahead]), np.ones((2, 1))])
        pixels = world @ matrix.T
        pose = (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)
        if not ((pose[0] >= 0).all() and pose[0, 0] < width and pose[0, 1] < height):
            raise RuntimeError(f"end-effector projects outside the frame for {sample.sample_id}: {pose[0].tolist()}")

        return ProbeData(
            "points",
            {"points": pose, "labels": ["end_effector", "heading"]},
            {
                "source": "recorded_pose",
                "frame_shape": [height, width],
                "n_points": len(pose),
                "eef_pos": position.tolist(),
                "heading_distance": self.heading_distance,
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        return _draw_pose(input_frame, np.asarray(data["points"], dtype=float).reshape(-1, 2))


def _draw_pose(input_frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    canvas = overlay_points(input_frame, points, radius=5)
    if len(points) >= 2:
        import cv2

        p0, p1 = points[0], points[1]
        cv2.arrowedLine(
            canvas,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (255, 0, 0),
            2,
        )
    return canvas
