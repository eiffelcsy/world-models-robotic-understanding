"""Pseudo ground truth for the depth perception probe.

The probe asks the world model to replace the scene with a depth map. What it is
scored against is what Depth Anything 3 sees in the conditioning frame: a
relative depth field derived from the input frame alone and so never
contaminated by the generated video.

Every answer is written to `results/depth_perception_gt/` as an `.npz` plus a
colored `.png`, and read back on the next run, so DA3 is loaded only for the
samples that do not have one yet.
"""

from __future__ import annotations

import numpy as np

from wmeval import GroundTruthSource, ProbeData, register
from wmeval.simulator import sim_record
from wmeval.types import RunContext, Sample

from .model import DepthAnything3Estimator
from .viz import depth_colormap


@register("gt", "depth_anything3")
class DepthAnything3Depth(GroundTruthSource):
    """Input frame -> `ProbeData("depth", {"depth": float HxW})`, larger = further.

    Params, all settable from probe.yaml:

      model_id  HF repo id or local path of the DA3 checkpoint
      device    cuda | cpu, or null to pick whatever torch reports

    Answers are stored under `<gt_dir>/depth_perception_gt/<sample_id>_gt.npz`,
    with a `.png` of the field beside each one, and a sample that has a file
    there is not re-estimated. Changing a param does not invalidate what is
    stored — the run logs a warning instead, and `preview.py` regenerates.
    """

    store_name = "depth_perception_gt"
    store_params = ("model_id", "device")

    def __init__(
        self,
        model_id: str = "depth-anything/DA3-LARGE-1.1",
        device: str | None = None,
        min_instances: int = 1,
    ) -> None:
        super().__init__(model_id=model_id, device=device, min_instances=min_instances)
        self._estimator = DepthAnything3Estimator(model_id, device)

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        depth = self._estimator.predict(input_frame)
        if depth.shape[:2] != input_frame.shape[:2]:
            import cv2

            depth = cv2.resize(depth, (input_frame.shape[1], input_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid = np.isfinite(depth) & (depth > 0)
        if int(valid.sum()) < self.min_instances:
            raise RuntimeError(f"DA3 returned no valid depth for {sample.sample_id}")

        return ProbeData(
            "depth",
            {"depth": np.asarray(depth, dtype=np.float32)},
            {
                "source": "depth_anything3",
                "model_id": self.model_id,
                "frame_shape": list(input_frame.shape[:2]),
                "depth_min": float(np.nanmin(depth)),
                "depth_max": float(np.nanmax(depth)),
                "valid_frac": round(float(valid.mean()), 4),
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        """The stored depth field, colored, with the frame's darkest pixel marked."""
        return depth_colormap(np.asarray(data["depth"], dtype=np.float32))


__all__ = ["DepthAnything3Depth", "overlay_points"]


@register("gt", "sim_depth")
class SimDepth(GroundTruthSource):
    """Simulator replay -> exact metric depth, in metres along the camera axis.

    MuJoCo's depth buffer for the conditioning frame's own render, converted with
    the camera's near and far planes. Pixels off the rendered square (the bars of
    the 16:9 canvas) are NaN, which the depth metrics leave out.
    """

    store_name = "depth_perception_sim_gt"
    store_params = ()

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        depth = np.asarray(sim_record(sample)["depth"], dtype=np.float32)
        if depth.shape != input_frame.shape[:2]:
            raise RuntimeError(f"replay depth {depth.shape} does not fit frame {input_frame.shape[:2]}")
        finite = depth[np.isfinite(depth)]
        return ProbeData(
            "depth",
            {"depth": depth},
            {
                "source": "simulator",
                "units": "metres",
                "frame_shape": list(depth.shape),
                "min_m": round(float(finite.min()), 4),
                "max_m": round(float(finite.max()), 4),
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        depth = np.asarray(data["depth"], dtype=np.float32)
        return depth_colormap(np.where(np.isfinite(depth), depth, np.nanmax(depth)))
