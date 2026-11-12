"""Fidelity gates: is a rollout a usable measurement at all?

A score is only meaningful if the clip actually did the thing it was asked to do.
Veo can return a beautiful clip that quietly re-renders the whole scene, ignores
the requested overlay, or flickers so much that the readout frame is noise. None
of that shows up in an IoU, so three gates screen each rollout first, and the
fraction that fail is reported next to every score as the **invalid rate**
(proposal Phase 4).

A gate sees the conditioning frame, the readout frame, the decoded prediction and
the clip, and returns `(valid, reason, measurements)`. It never changes the score
— a failed gate flags the row `valid=False`, and the report counts it. Gates are
declared per probe in `probe.yaml`:

    gates:
      - {type: scene_preservation, params: {max_change: 0.15}}
      - {type: instruction_compliance, params: {min_area_frac: 0.0005}}
      - {type: temporal_stability, params: {window_s: 1.0, max_diff: 10.0}}

The three shipped gates are the proposal's: scene preservation outside the
marked region, instruction compliance (the answer was actually drawn), and
temporal stability across the readout window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wmcore.components import Component
from wmcore.registry import register
from wmcore.utils import resize_to
from wmcore.video import _open, _read_at

from .types import ProbeData, Sample


@dataclass
class GateContext:
    """Everything a gate might look at for one scored rollout."""

    sample: Sample
    input_frame: np.ndarray
    output_frame: np.ndarray
    prediction: ProbeData
    video: Path
    readout: dict[str, Any]


class Gate(Component):
    """One rollout -> (valid, reason, measurements). Raise to fail closed."""

    def check(self, ctx: GateContext) -> tuple[bool, str, dict[str, Any]]:
        raise NotImplementedError

    def __call__(self, ctx: GateContext) -> dict[str, Any]:
        try:
            valid, reason, measurements = self.check(ctx)
        except Exception as exc:  # noqa: BLE001 - a gate that cannot run is not a pass
            valid, reason, measurements = False, f"{type(exc).__name__}: {exc}", {}
        return {"gate": type(self).__name__, "valid": bool(valid), "reason": reason, **measurements}


def _marked_region(prediction: ProbeData, shape: tuple[int, int], dilate: int = 0) -> np.ndarray | None:
    """The pixels the model marked, on `shape`, or None if this kind has none.

    `mask`/`instances` give a region directly; `points` is rasterized. `depth` is
    a whole-frame convention with no "outside the region", so it returns None and
    a region gate treats it as not applicable.
    """
    kind = prediction.kind
    region: np.ndarray | None = None
    if kind == "binary_mask" and prediction.get("mask") is not None:
        region = np.asarray(prediction["mask"]).astype(bool)
    elif kind == "instances":
        masks = prediction.get("masks") or []
        if masks:
            region = np.zeros(np.asarray(masks[0]).shape[:2], dtype=bool)
            for mask in masks:
                region |= np.asarray(mask).astype(bool)
    elif kind == "points" and prediction.get("points") is not None:
        points = np.asarray(prediction["points"]).reshape(-1, 2)
        region = np.zeros(shape, dtype=bool)
        radius = max(1, int(dilate))
        for x, y in points:
            cv2.circle(region, (int(round(x)), int(round(y))), radius, True, -1)
        return region if region.any() else None
    if region is None:
        return None
    if region.shape != tuple(shape):
        region = resize_to(region, shape, nearest=True)
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        region = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
    return region


@register("gate", "scene_preservation")
class ScenePreservation(Gate):
    """Everything outside the marked region should be left alone.

    The prompt asks for one appearance change; a model that repaints the whole
    scene has not obeyed, even if the marked region is right. Measures the mean
    per-pixel change (0-1) of the readout frame against the conditioning frame,
    outside the region the prediction marked.

    Params: `max_change` (fraction of full scale), `dilate` (px of slack at the
    region boundary, to forgive anti-aliasing).
    """

    def __init__(self, max_change: float = 0.15, dilate: int = 0) -> None:
        super().__init__(max_change=max_change, dilate=dilate)

    def check(self, ctx: GateContext) -> tuple[bool, str, dict[str, Any]]:
        shape = ctx.input_frame.shape[:2]
        region = _marked_region(ctx.prediction, shape, int(self.dilate))
        if region is None:
            return True, "", {"scene_change_outside": None, "scene_applicable": 0}
        outside = ~region
        if not outside.any():
            return True, "", {"scene_change_outside": 0.0, "scene_applicable": 1}
        out = ctx.output_frame
        if out.shape[:2] != shape:
            out = resize_to(out, shape)
        diff = np.abs(out.astype(np.float32) - ctx.input_frame.astype(np.float32)).mean(axis=2) / 255.0
        change = float(diff[outside].mean())
        valid = change <= float(self.max_change)
        reason = "" if valid else f"scene changed {change:.3f} > {self.max_change} outside the marked region"
        return valid, reason, {"scene_change_outside": round(change, 4), "scene_applicable": 1}


@register("gate", "instruction_compliance")
class InstructionCompliance(Gate):
    """The requested overlay was actually produced.

    For a region answer (`mask`/`instances`), the marked area must be non-trivial
    and not the whole frame. For a `points` answer, at least `min_points` points
    must decode. A clip where the model painted nothing is an invalid rollout,
    not a score of zero.

    Params: `min_area_frac`, `max_area_frac`, `min_points`.
    """

    def __init__(self, min_area_frac: float = 0.0005, max_area_frac: float = 0.9, min_points: int = 1) -> None:
        super().__init__(min_area_frac=min_area_frac, max_area_frac=max_area_frac, min_points=min_points)

    def check(self, ctx: GateContext) -> tuple[bool, str, dict[str, Any]]:
        kind = ctx.prediction.kind
        if kind in ("instances", "binary_mask"):
            region = _marked_region(ctx.prediction, ctx.input_frame.shape[:2])
            if region is None:
                return False, "the model marked nothing", {"marked_area_frac": 0.0, "compliance_applicable": 1}
            frac = float(region.mean())
            valid = self.min_area_frac <= frac <= self.max_area_frac
            reason = "" if valid else f"marked area {frac:.4f} outside [{self.min_area_frac}, {self.max_area_frac}]"
            return valid, reason, {"marked_area_frac": round(frac, 4), "compliance_applicable": 1}
        if kind == "points":
            points = ctx.prediction.get("points")
            n = int(np.asarray(points).reshape(-1, 2).shape[0]) if points is not None and np.size(points) else 0
            valid = n >= int(self.min_points)
            reason = "" if valid else f"{n} point(s) < {self.min_points}"
            return valid, reason, {"n_points": n, "compliance_applicable": 1}
        return True, "", {"compliance_applicable": 0}


@register("gate", "temporal_stability")
class TemporalStability(Gate):
    """The answer settled and held, rather than flickering.

    Measures the mean frame-to-frame change over the tail `window_s` seconds.
    A large value means the readout frame was caught mid-motion and the score is
    measuring noise. Params: `window_s`, `stride`, `max_diff` (mean abs 0-255).
    """

    def __init__(self, window_s: float = 1.0, stride: int = 2, max_diff: float = 10.0) -> None:
        super().__init__(window_s=window_s, stride=max(1, int(stride)), max_diff=max_diff)

    def check(self, ctx: GateContext) -> tuple[bool, str, dict[str, Any]]:
        capture, count, fps = _open(ctx.video)
        try:
            start = max(1, count - int(float(self.window_s) * fps))
            previous = _read_at(capture, max(0, start - 1)).astype(np.float32)
            diffs: list[float] = []
            for index in range(start, count, int(self.stride)):
                try:
                    frame = _read_at(capture, index)
                except IOError:
                    break
                diffs.append(float(np.abs(frame.astype(np.float32) - previous).mean()))
                previous = frame.astype(np.float32)
        finally:
            capture.release()
        if not diffs:
            return True, "", {"temporal_diff": None}
        mean_diff = float(np.mean(diffs))
        valid = mean_diff <= float(self.max_diff)
        reason = "" if valid else f"tail motion {mean_diff:.1f} > {self.max_diff}"
        return valid, reason, {"temporal_diff": round(mean_diff, 3)}
