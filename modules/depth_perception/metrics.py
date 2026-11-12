"""Predicted depth vs Depth Anything 3 depth.

Both fields are relative, so scoring is an alignment problem before it is an
error problem: fit the prediction to the ground truth with a scale (and offset,
by default), then report the usual monocular-depth errors. `delta1` is the
headline because it is higher-is-better and therefore legal as a
`best_frame.rank_by`; every error here is lower-is-better.

Scores, all in [0, 1] except the errors:

  delta1/2/3   fraction of pixels within 1.25^k of the ground truth — the
               headline; robust to a handful of bad pixels
  abs_rel      mean |pred - gt| / gt
  sq_rel       mean (pred - gt)^2 / gt
  rmse         root mean squared error
  rmse_log     RMSE in log space
  silog        scale-invariant log error, the Eigen et al. metric
  n_valid      pixels that survived masking, and their fraction — a score over
  valid_frac   a handful of pixels must not read as a result

`align` is the fit applied first:

  none         no fit; meaningful only if both sides already share a scale
  median       scale by the ratio of medians
  scale        least-squares scale, no offset
  scale_shift  least-squares scale and offset — the default, and the right one
               for two relative fields that do not share a zero

`space` is where that fit happens:

  depth        the prediction is fitted to the ground-truth depth (default)
  disparity    the prediction is fitted to the ground-truth inverse depth
               (1 / depth), then inverted back to depth for the errors above.
               This is the standard for relative depth (Ranftl et al., MiDaS;
               Depth Anything): models and the depth images a video model has
               seen encode disparity, near bright. Because the fit is affine,
               the decoder's brightness direction does not matter here; only
               the space the fit is done in does.

`prefix` is prepended to every score name, so one probe can report both
spaces side by side (`delta1` and `disp_delta1`).
"""

from __future__ import annotations

import cv2
import numpy as np

from wmeval import Metric, ProbeData, register


@register("metric", "depth_metrics")
class DepthMetrics(Metric):
    accepts = ("depth",)

    def __init__(
        self,
        align: str = "scale_shift",
        min_depth: float = 1e-3,
        max_depth: float = 1e3,
        delta: float = 1.25,
        space: str = "depth",
        prefix: str = "",
    ) -> None:
        if space not in ("depth", "disparity"):
            raise ValueError(f"unknown space '{space}'. available: depth, disparity")
        super().__init__(
            align=align, min_depth=min_depth, max_depth=max_depth, delta=delta, space=space, prefix=prefix
        )

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        pred = _as_depth(prediction)
        gt = _as_depth(truth)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid = np.isfinite(pred) & np.isfinite(gt)
        valid &= (gt >= self.min_depth) & (gt <= self.max_depth)
        if self.space == "depth":
            # Unchanged from before `space` existed, so depth-space scores stay
            # comparable with earlier runs.
            valid &= (pred >= self.min_depth) & (pred <= self.max_depth)
        if not valid.any():
            return self._named(_empty())

        pred_v = pred[valid].astype(np.float64)
        gt_v = np.clip(gt[valid].astype(np.float64), self.min_depth, self.max_depth)
        if self.space == "depth":
            # An affine fit can put some pixels at or below zero depth. Clip as
            # the disparity branch does: unclipped, a negative depth makes the
            # log errors NaN and its negative ratio counts as a delta hit.
            pred_v = np.clip(self._align(pred_v, gt_v), self.min_depth, self.max_depth)
        else:
            disparity = self._align(pred_v, 1.0 / gt_v)
            disparity = np.clip(disparity, 1.0 / self.max_depth, 1.0 / self.min_depth)
            pred_v = 1.0 / disparity

        difference = pred_v - gt_v
        ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
        log_difference = np.log(pred_v) - np.log(gt_v)

        threshold = float(self.delta)
        return self._named({
            "delta1": float(np.mean(ratio < threshold)),
            "delta2": float(np.mean(ratio < threshold**2)),
            "delta3": float(np.mean(ratio < threshold**3)),
            "abs_rel": float(np.mean(np.abs(difference) / gt_v)),
            "sq_rel": float(np.mean((difference**2) / gt_v)),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "rmse_log": float(np.sqrt(np.mean(log_difference**2))),
            "silog": float(np.sqrt(np.mean(log_difference**2) - np.mean(log_difference) ** 2) * 100),
            "n_valid": float(valid.sum()),
            "valid_frac": float(valid.mean()),
        })

    def _named(self, scores: dict[str, float]) -> dict[str, float]:
        return {f"{self.prefix}{key}": value for key, value in scores.items()} if self.prefix else scores

    def _align(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        if self.align == "none":
            return pred
        if self.align == "median":
            scale = float(np.median(gt) / max(np.median(pred), self.min_depth))
            return pred * scale
        if self.align == "scale":
            scale = float(np.sum(gt * pred) / max(np.sum(pred * pred), self.min_depth))
            return pred * scale
        if self.align == "scale_shift":
            design = np.stack([pred, np.ones_like(pred)], axis=1)
            solution, *_ = np.linalg.lstsq(design, gt, rcond=None)
            return design @ solution
        raise ValueError(f"unknown align '{self.align}'. available: none, median, scale, scale_shift")


def _as_depth(data: ProbeData) -> np.ndarray:
    depth = np.asarray(data.get("depth"))
    if depth.ndim > 2:
        depth = np.squeeze(depth)
    return depth.astype(np.float32)


def _empty() -> dict[str, float]:
    return {
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "silog": 0.0,
        "n_valid": 0.0,
        "valid_frac": 0.0,
    }
