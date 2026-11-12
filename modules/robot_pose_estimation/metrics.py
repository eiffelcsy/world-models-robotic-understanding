"""Predicted robot pose vs the recorded end-effector pose.

The pose is two points: an end-effector center and a heading reference. Scoring
splits cleanly into position and orientation.

  pose_score   exp(-pos_err / pos_threshold) * (1 - angle_err / 90) — the
               headline; zero when either half is missing
  pos_err      center distance, normalized by the frame diagonal
  pos_pck      1 when pos_err is within `pos_threshold_frac` of the diagonal
  angle_err    heading difference in degrees, folded into [0, 90]
  heading_score 1 - angle_err / 90
  n_pred/n_truth/frame_diagonal   context

`pose_score` is higher-is-better and is the module's `best_frame.rank_by`.
"""

from __future__ import annotations

import numpy as np

from wmeval import Metric, ProbeData, register


@register("metric", "pose_metrics")
class PoseMetrics(Metric):
    accepts = ("points",)

    def __init__(self, pos_threshold_frac: float = 0.05) -> None:
        super().__init__(pos_threshold_frac=pos_threshold_frac)

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        pred = _points(prediction)
        gt = _points(truth)
        diagonal = _diagonal(prediction, truth)
        threshold = max(1e-6, self.pos_threshold_frac * diagonal)

        pos_err = float(np.linalg.norm(pred[0] - gt[0]) / diagonal) if len(pred) and len(gt) else 1.0
        position_score = float(np.exp(-pos_err / self.pos_threshold_frac))
        pos_pck = 1.0 if pos_err <= self.pos_threshold_frac else 0.0

        angle_err = _angle_error(pred, gt)
        heading_score = 0.0 if angle_err is None else max(0.0, 1.0 - angle_err / 90.0)
        pose_score = position_score * heading_score if angle_err is not None else 0.0

        return {
            "pose_score": pose_score,
            "pos_err": pos_err,
            "pos_pck": pos_pck,
            "angle_err": float(angle_err) if angle_err is not None else 90.0,
            "heading_score": heading_score,
            "n_pred": float(len(pred)),
            "n_truth": float(len(gt)),
            "frame_diagonal": float(diagonal),
        }


def _points(data: ProbeData) -> np.ndarray:
    points = np.asarray(data.get("points"), dtype=np.float64)
    return points.reshape(-1, 2) if points.size else np.zeros((0, 2))


def _angle_error(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if len(pred) < 2 or len(gt) < 2:
        return None
    pred_vector = pred[1] - pred[0]
    gt_vector = gt[1] - gt[0]
    norm = np.linalg.norm(pred_vector) * np.linalg.norm(gt_vector)
    if norm <= 1e-9:
        return None
    cosine = float(np.clip(np.dot(pred_vector, gt_vector) / norm, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return min(angle, 180.0 - angle)


def _diagonal(prediction: ProbeData, truth: ProbeData) -> float:
    shape = truth.info.get("frame_shape") or prediction.info.get("frame_shape")
    if shape and len(shape) == 2:
        return float(np.hypot(shape[0], shape[1]))
    stacked = [p for p in (_points(prediction), _points(truth)) if p.size]
    if stacked:
        points = np.vstack(stacked)
        span = points.max(axis=0) - points.min(axis=0)
        return float(max(np.hypot(*span), 1.0))
    return 1.0
