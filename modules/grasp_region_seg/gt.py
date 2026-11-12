"""Ground truth for the grasp region probe, from a recorded demonstration.

`demonstrated_grasp` takes the grasp from the robot's own gripper in the recorded
episode and the object it landed on from a simulator replay. Only LIBERO samples
carry both; there is no estimated grasp ground truth for real video.
"""

from __future__ import annotations

import cv2
import numpy as np

from wmeval import GroundTruthSource, ProbeData, register
from wmeval.simulator import object_mask, sim_record
from wmeval.types import RunContext, Sample


@register("gt", "demonstrated_grasp")
class DemonstratedGrasp(GroundTruthSource):
    """Recorded grasp -> `ProbeData("binary_mask", ...)` on the target object.

    `mask` is the object the demonstration picked up and `grasp` is where the
    fingers closed on it, as `[centre, finger, finger]` in pixels. A demonstration
    shows one grasp that worked, not every one that would, so there is no region to
    take an IoU against. `grasp_hit` scores the two together.

    Reads, from the sample's simulator record (scripts/libero_replay/render_gt.py):

      grasp_fingers_world  the inner faces of the two finger pads at the grasp
                       frame, where the fingers settled on the object (the frame
                       scripts/fetch_libero.py finds from the gripper). The grasp
                       centre is their midpoint. The fingers are driven separately,
                       so an object can push them off the grip site's centre;
                       these are where they actually are
      segment          the segmentation the object mask is cut from

    and from the meta: `world_to_image` and `intrinsics`, the conditioning frame's
    camera, and `sim_grasp_object`, the object the fingers touched after closing.

    The object does not move before the gripper reaches it, so a grasp recorded
    later in the episode lands on it in the conditioning frame. A grasp that does
    not land on its object fails the sample rather than scoring against a guess.

    Params, all settable from probe.yaml:

      tolerance_m      how far from the grasp a painted pixel still counts as a hit
      max_object_gap_m how far the grasp may sit from the object mask. Rim grasps
                       land near the edge of it
    """

    store_name = "grasp_demonstrated_gt"
    store_params = ("tolerance_m", "max_object_gap_m")

    def __init__(self, tolerance_m: float = 0.02, max_object_gap_m: float = 0.03) -> None:
        super().__init__(tolerance_m=tolerance_m, max_object_gap_m=max_object_gap_m)

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        height, width = input_frame.shape[:2]
        record = sim_record(sample)
        fingers = np.asarray(record["grasp_fingers_world"], dtype=np.float64)
        matrix = np.asarray(sample.meta["world_to_image"], dtype=np.float64)

        world = np.stack([fingers.mean(axis=0), fingers[0], fingers[1]])
        pixels = np.hstack([world, np.ones((3, 1))]) @ matrix.T
        grasp = (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)
        x, y = grasp[0]
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError(f"grasp projects outside the frame for {sample.sample_id}: {grasp[0].tolist()}")

        # The third row of the matrix is depth, so this is the image scale at the grasp.
        pixels_per_metre = float(sample.meta["intrinsics"]["fx"]) / float(pixels[0, 2])

        mask = object_mask(record, sample.meta["sim_grasp_object"])
        if not mask.any():
            raise RuntimeError(f"the grasped object is not visible for {sample.sample_id}")
        gap = _distance_to(mask, x, y)
        if gap > self.max_object_gap_m * pixels_per_metre:
            raise RuntimeError(f"grasp is {gap:.0f}px from the object for {sample.sample_id}")

        rows, columns = np.nonzero(mask)
        offset_px = float(np.hypot(columns.mean() - x, rows.mean() - y))

        return ProbeData(
            "binary_mask",
            {
                "mask": mask,
                "grasp": grasp,
                "tolerance_px": np.float32(self.tolerance_m * pixels_per_metre),
                "pixels_per_metre": np.float32(pixels_per_metre),
            },
            {
                "source": "demonstrated_grasp",
                "frame_shape": [height, width],
                "object": sample.meta["sim_grasp_object"],
                "object_area_frac": round(float(mask.mean()), 5),
                "object_gap_px": round(float(gap), 1),
                "grasp_offset_m": round(offset_px / pixels_per_metre, 4),
                "grasp_width_m": round(float(np.linalg.norm(fingers[0] - fingers[1])), 4),
                "grasp_frame_index": sample.meta.get("grasp_frame_index"),
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        view = input_frame.copy()
        mask = np.asarray(data["mask"], dtype=bool)
        view[mask] = (0.5 * view[mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
        centre, first, second = (tuple(int(round(v)) for v in point) for point in np.asarray(data["grasp"]))
        cv2.line(view, first, second, (255, 255, 0), 2)
        cv2.circle(view, centre, max(int(round(float(data["tolerance_px"]))), 1), (255, 0, 0), 2)
        return view


def _distance_to(mask: np.ndarray, x: float, y: float) -> float:
    """Pixels from (x, y) to the nearest pixel of `mask`; 0 inside it."""
    field = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return float(field[int(round(y)), int(round(x))])
