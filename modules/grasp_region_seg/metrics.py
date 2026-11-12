"""Predicted grasp region vs one demonstrated grasp: `grasp_hit`."""

from __future__ import annotations

import cv2
import numpy as np

from wmeval import Metric, ProbeData, register


@register("metric", "grasp_hit")
class GraspHit(Metric):
    """Predicted region vs one demonstrated grasp on the target object.

      hit           the region reaches the demonstrated grasp, within tolerance
      focus         predicted pixels within `focus_m` of the grasp
      on_object     predicted pixels on the target object
      coverage      target object pixels that are predicted
      tightness     1 up to `free_coverage`, falling to 0 at a fully painted object
      grasp_loc     the headline (from 2026-11-11). 1 − E_paint / E_object, clipped
                    to [0, 1], where E is the mean distance of a set of pixels from
                    the grasp centre beyond the hit tolerance. 1: every painted
                    pixel is within tolerance of the grasp. 0: the paint is no
                    nearer the grasp than the object is on average, which is what
                    painting the whole object gives. Nothing painted scores 0.
                    Paint off the object but within tolerance is not penalised, so
                    a rim grasp's patch may spill past the rim
      object_excess_m  E_object in metres: how far the object reaches beyond the
                    tolerance. Small values mean the sample cannot tell knowing
                    the grasp from finding the object. Describes the sample
      grasp_score   hit * focus * on_object * tightness: the headline until
                    2026-11-11, kept so earlier tables stay comparable
      distance_px   from the grasp to the nearest predicted pixel
      distance_m    the same, at the grasp's depth
      grasp_offset_m  from the object's centre to the grasp. Not a score of the
                    prediction: it says what this sample can show

    `focus` is what separates knowing the grasp from finding the object: a blob on
    the middle of a bowl can reach a rim grasp but is mostly nowhere near it. Where
    `grasp_offset_m` is small the grasp is at the middle of the object, and painting
    the middle is the right answer, so only samples with a large offset test
    anything beyond finding the object.

    A demonstration is one grasp that worked, so a miss may still be a valid grasp
    elsewhere on the object: `grasp_score` is a lower bound on grasp quality.

    Every score but the distances is higher-is-better, so `grasp_loc` is a legal
    `best_frame.rank_by`.
    """

    accepts = ("binary_mask",)

    def __init__(self, focus_m: float = 0.04, free_coverage: float = 0.5, object_dilate: int = 5) -> None:
        super().__init__(focus_m=focus_m, free_coverage=free_coverage, object_dilate=object_dilate)

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        target = np.asarray(truth.get("mask"), dtype=bool)
        pred = np.asarray(prediction.get("mask"), dtype=bool)
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.uint8), target.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0

        x, y = (int(round(v)) for v in np.asarray(truth.get("grasp"))[0])
        pixels_per_metre = float(truth.get("pixels_per_metre"))
        if pred.any():
            distance = float(cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)[y, x])
        else:
            distance = float(np.hypot(*target.shape))
        hit = float(pred.any() and distance <= float(truth.get("tolerance_px")))

        painted = int(np.count_nonzero(pred))
        near = np.zeros(target.shape, dtype=np.uint8)
        cv2.circle(near, (x, y), int(round(self.focus_m * pixels_per_metre)), 1, -1)
        focus = _ratio(int(np.count_nonzero(pred & (near > 0))), painted, empty=False)

        # A painted edge bleeds a little past the object, which is not a wrong answer.
        size = max(int(self.object_dilate), 1)
        grown = cv2.dilate(target.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
        on_object = _ratio(int(np.count_nonzero(pred & grown)), painted, empty=False)
        coverage = _ratio(int(np.count_nonzero(pred & target)), int(np.count_nonzero(target)), empty=False)
        free = float(self.free_coverage)
        tightness = float(np.clip((1.0 - coverage) / (1.0 - free), 0.0, 1.0))

        rows, columns = np.nonzero(target)
        offset = float(np.hypot(columns.mean() - x, rows.mean() - y)) if len(rows) else 0.0

        # Distance of every pixel from the grasp centre, beyond the hit tolerance.
        yy, xx = np.indices(target.shape)
        excess = np.maximum(np.hypot(xx - x, yy - y) - float(truth.get("tolerance_px")), 0.0)
        object_excess = float(excess[target].mean()) if target.any() else 0.0
        if not pred.any():
            loc = 0.0
        elif object_excess > 0:
            loc = float(np.clip(1.0 - float(excess[pred].mean()) / object_excess, 0.0, 1.0))
        else:  # the whole object is within tolerance: only "all paint within tolerance" can be told apart
            loc = float(not excess[pred].any())

        return {
            "grasp_loc": loc,
            "object_excess_m": object_excess / pixels_per_metre,
            "grasp_score": hit * focus * on_object * tightness,
            "hit": hit,
            "focus": focus,
            "on_object": on_object,
            "coverage": coverage,
            "tightness": tightness,
            "distance_px": distance,
            "distance_m": distance / pixels_per_metre,
            "grasp_offset_m": offset / pixels_per_metre,
            "area_pred": float(pred.mean()),
        }


def _ratio(hits: int, total: int, *, empty: bool) -> float:
    if total:
        return hits / total
    return 1.0 if empty else 0.0
