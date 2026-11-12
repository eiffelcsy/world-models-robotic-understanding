"""Predicted instances vs SAM 3 instances.

Both sides are unordered sets of binary masks, so scoring is a matching problem
before it is an averaging problem: build the IoU matrix, match one-to-one, then
report how good the matched pairs are and how much of each set never matched.

The scores, all in [0, 1] unless noted:

  mean_iou        sum of matched IoU / max(n_pred, n_truth) — the headline
                  number. Charged for every mask on either side that went
                  unmatched, so it cannot be gamed by emitting one good mask.
  matched_iou     mean IoU over matched pairs alone: how well the instances it
                  did find are outlined, ignoring the ones it missed.
  foreground_iou  union of all predictions vs union of all truth. Says whether
                  the paint landed on the objects at all, independent of how it
                  split them up.
  pq              panoptic quality at `iou_threshold`: matched IoU summed over
                  hits, over hits plus half of each unmatched side.
  precision       hits / n_pred, at `iou_threshold`
  recall          hits / n_truth
  f1              their harmonic mean
  n_pred          instance counts and their absolute disagreement — raw counts,
  n_truth         not fractions
  n_matched
  count_error

Matching is greedy by descending IoU rather than Hungarian, which keeps scipy
out of the dependency list. For masks that do not overlap each other — which is
what an instance decomposition is — a prediction can clear IoU 0.5 against at
most one truth mask, so above the threshold greedy and optimal agree; the two
can only differ over the weak pairs that contribute little to `mean_iou`.

`accepts` is checked by the probe before scoring, so a mis-wired probe.yaml
fails loudly rather than scoring incomparable things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from wmeval import Metric, ProbeData, register

Box = tuple[int, int, int, int]  # y0, y1, x0, x1, half-open


@dataclass(frozen=True)
class Match:
    """One matched pair, by index into the prediction and truth mask lists."""

    prediction: int
    truth: int
    iou: float


class MaskSet:
    """Binary masks plus the per-mask area and bounding box the IoU loop needs.

    Extracted once per set rather than once per pair: the ground truth is
    compared against every frame of a video, and the bounding boxes turn most of
    those comparisons into a cheap rejection instead of a full-frame AND.
    """

    def __init__(self, masks: Iterable[Any]) -> None:
        self.masks: list[np.ndarray] = [_as_bool(m) for m in masks]
        self.areas: list[int] = [int(np.count_nonzero(m)) for m in self.masks]
        self.boxes: list[Box | None] = [_bbox(m) for m in self.masks]
        self._union: np.ndarray | None = None

    @classmethod
    def of(cls, masks: "MaskSet | Iterable[Any]") -> "MaskSet":
        return masks if isinstance(masks, cls) else cls(masks)

    def __len__(self) -> int:
        return len(self.masks)

    @property
    def shape(self) -> tuple[int, int] | None:
        return tuple(self.masks[0].shape[:2]) if self.masks else None  # type: ignore[return-value]

    @property
    def union(self) -> np.ndarray | None:
        """Every instance flattened into one foreground mask, computed once."""
        if self._union is None and self.masks:
            union = np.zeros_like(self.masks[0], dtype=bool)
            for mask in self.masks:
                union |= mask
            self._union = union
        return self._union


def mask_iou(a: Any, b: Any) -> float:
    """IoU of two binary masks of the same shape."""
    left, right = _as_bool(a), _as_bool(b)
    intersection = int(np.count_nonzero(left & right))
    if not intersection:
        return 0.0
    return intersection / (int(np.count_nonzero(left)) + int(np.count_nonzero(right)) - intersection)


def iou_matrix(
    predictions: "MaskSet | Iterable[Any]", truths: "MaskSet | Iterable[Any]"
) -> np.ndarray:
    """(n_pred, n_truth) IoU matrix, empty-safe."""
    pred, truth = MaskSet.of(predictions), MaskSet.of(truths)
    matrix = np.zeros((len(pred), len(truth)), dtype=float)
    for i, (mask, box, area) in enumerate(zip(pred.masks, pred.boxes, pred.areas)):
        if box is None:
            continue
        for j, (other, other_box, other_area) in enumerate(
            zip(truth.masks, truth.boxes, truth.areas)
        ):
            matrix[i, j] = _boxed_iou(mask, other, box, other_box, area, other_area)
    return matrix


def match(matrix: np.ndarray, min_iou: float = 0.0) -> list[Match]:
    """One-to-one pairing, greedy by descending IoU.

    Pairs at or below `min_iou` are left unmatched, so a zero-overlap pair is
    never called a match.
    """
    if matrix.size == 0:
        return []
    matches: list[Match] = []
    taken_pred: set[int] = set()
    taken_truth: set[int] = set()
    columns = matrix.shape[1]
    # Stable, so ties resolve the same way on every machine and every run.
    for flat in np.argsort(-matrix, axis=None, kind="stable"):
        iou = float(matrix.flat[flat])
        if iou <= min_iou:
            break
        i, j = divmod(int(flat), columns)
        if i in taken_pred or j in taken_truth:
            continue
        matches.append(Match(i, j, iou))
        taken_pred.add(i)
        taken_truth.add(j)
        if len(matches) == min(matrix.shape):
            break
    return matches


def score_masks(
    predictions: "MaskSet | Iterable[Any]",
    truths: "MaskSet | Iterable[Any]",
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """The full score dict for one pair of mask sets. See the module docstring.

    Two empty sets score a perfect 1.0 — nothing there, nothing claimed — while
    one empty side against a populated one scores 0.
    """
    pred, truth = MaskSet.of(predictions), MaskSet.of(truths)
    n_pred, n_truth = len(pred), len(truth)
    if n_pred and n_truth and pred.shape != truth.shape:
        raise ValueError(f"mask shapes differ: prediction {pred.shape} vs truth {truth.shape}")

    matches = match(iou_matrix(pred, truth))
    ious = [m.iou for m in matches]
    hits = [iou for iou in ious if iou >= iou_threshold]

    both_empty = n_pred == 0 and n_truth == 0
    largest = max(n_pred, n_truth)
    quality = len(hits) + 0.5 * (n_pred - len(hits)) + 0.5 * (n_truth - len(hits))
    precision = _ratio(len(hits), n_pred, empty=n_truth == 0)
    recall = _ratio(len(hits), n_truth, empty=n_pred == 0)

    return {
        "mean_iou": 1.0 if both_empty else sum(ious) / largest,
        "matched_iou": 1.0 if both_empty else (sum(ious) / len(ious) if ious else 0.0),
        "foreground_iou": _union_iou(pred, truth),
        "pq": 1.0 if both_empty else (sum(hits) / quality if quality else 0.0),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "n_pred": float(n_pred),
        "n_truth": float(n_truth),
        "n_matched": float(len(hits)),
        "count_error": float(abs(n_pred - n_truth)),
    }


@register("metric", "instance_match_iou")
class InstanceMatchIoU(Metric):
    accepts = ("instances",)

    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__(iou_threshold=iou_threshold)

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        return score_masks(
            prediction.get("masks") or [],
            truth.get("masks") or [],
            iou_threshold=self.iou_threshold,
        )


def _as_bool(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == bool:
        return array
    return array > (0.5 if array.dtype.kind == "f" else 0)


def _bbox(mask: np.ndarray) -> Box | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _boxed_iou(
    a: np.ndarray, b: np.ndarray, box_a: Box | None, box_b: Box | None, area_a: int, area_b: int
) -> float:
    """IoU restricted to where the two bounding boxes overlap.

    Masks that cannot touch are rejected on four integer comparisons, and the
    ones that can are intersected over their shared window rather than over the
    whole frame — the difference between a video scan that runs in seconds and
    one that does not.
    """
    if box_a is None or box_b is None:
        return 0.0
    y0, y1 = max(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    x0, x1 = max(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if y0 >= y1 or x0 >= x1:
        return 0.0
    intersection = int(np.count_nonzero(a[y0:y1, x0:x1] & b[y0:y1, x0:x1]))
    if not intersection:
        return 0.0
    return intersection / (area_a + area_b - intersection)


def _union_iou(pred: MaskSet, truth: MaskSet) -> float:
    left, right = pred.union, truth.union
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    return mask_iou(left, right)


def _ratio(hits: int, total: int, *, empty: bool) -> float:
    """hits/total, and 1.0 when there was correctly nothing to divide by."""
    if total:
        return hits / total
    return 1.0 if empty else 0.0
