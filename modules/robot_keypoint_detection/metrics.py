"""Predicted robot keypoints vs the simulator's keypoints.

Two unordered point sets, so scoring is matching first: pair predicted points
to ground-truth points one-to-one, then ask how many pairs are close enough and
how close the rest are.

  pck          fraction of ground-truth keypoints with a matched prediction
               within `threshold_frac` of the frame diagonal — the headline
  mean_dist    mean matched distance, normalized by the diagonal
  precision    matched-within-threshold over the number of predictions
  recall       matched-within-threshold over the number of ground-truth points
  f1           their harmonic mean
  score        mean exp(-d / threshold): smooth, so a near miss still counts
  n_pred/n_truth/n_matched/frame_diagonal   context

`pck` is higher-is-better and is the module's `best_frame.rank_by`.
"""

from __future__ import annotations

import numpy as np

from wmeval import Metric, ProbeData, register


@register("metric", "keypoint_pck")
class KeypointPCK(Metric):
    accepts = ("points",)

    def __init__(self, threshold_frac: float = 0.05) -> None:
        super().__init__(threshold_frac=threshold_frac)

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        pred = _points(prediction)
        gt = _points(truth)
        diagonal = _diagonal(prediction, truth)
        threshold = max(1e-6, self.threshold_frac * diagonal)

        matches = _match(pred, gt)
        if matches:
            distances = np.array([d for _, _, d in matches], dtype=np.float64)
            hits = int((distances <= threshold).sum())
        else:
            distances = np.array([], dtype=np.float64)
            hits = 0

        precision = _ratio(hits, len(pred), empty=len(gt) == 0)
        recall = _ratio(hits, len(gt), empty=len(pred) == 0)
        score = _smooth_score(pred, gt, threshold)

        return {
            "pck": recall,
            "mean_dist": float(distances.mean() / diagonal) if len(distances) else (0.0 if not len(gt) else 1.0),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "score": score,
            "n_pred": float(len(pred)),
            "n_truth": float(len(gt)),
            "n_matched": float(len(matches)),
            "frame_diagonal": float(diagonal),
        }


def _points(data: ProbeData) -> np.ndarray:
    points = np.asarray(data.get("points"), dtype=np.float64)
    return points.reshape(-1, 2) if points.size else np.zeros((0, 2))


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


def _match(pred: np.ndarray, gt: np.ndarray) -> list[tuple[int, int, float]]:
    if not len(pred) or not len(gt):
        return []
    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    matches: list[tuple[int, int, float]] = []
    taken_pred: set[int] = set()
    taken_gt: set[int] = set()
    for flat in np.argsort(distances, axis=None, kind="stable"):
        i, j = divmod(int(flat), len(gt))
        if i in taken_pred or j in taken_gt:
            continue
        matches.append((i, j, float(distances[i, j])))
        taken_pred.add(i)
        taken_gt.add(j)
        if len(matches) == min(len(pred), len(gt)):
            break
    return matches


def _smooth_score(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    if not len(gt):
        return 1.0 if not len(pred) else 0.0
    if not len(pred):
        return 0.0
    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    nearest = distances.min(axis=0)
    return float(np.mean(np.exp(-nearest / threshold)))


def _ratio(hits: int, total: int, *, empty: bool) -> float:
    if total:
        return hits / total
    return 1.0 if empty else 0.0
