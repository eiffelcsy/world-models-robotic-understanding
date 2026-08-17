"""Generated frame -> instance masks, and the frame of a video that fits best.

Two things live here.

`ColorInstances` is the decoder for the probe's convention: the model was asked
to flood each object with one saturated, uniform color and hold it, so a frame is
turned back into instances by finding which colors are actually present, keeping
the pixels that are painted rather than scene, and cutting each color into
connected pieces. It runs per frame and is what `probe.yaml` wires in.

`scan_video` is the diagnostic on top: run the decoder over every frame of a
generation, score each one against the SAM 3 ground truth for the frame it was
conditioned on, and report two numbers — the best frame in the clip, and the last
frame. The gap between them is the thing worth looking at. The last frame is what
an honest readout can reach, since it does not consult the ground truth; the best
frame uses the ground truth to choose and is therefore an oracle upper bound, not
a score the model earned. Reported side by side, they separate "the model never
understood the scene" from "the model understood it and then drifted".

`scan.py` is the command line over `scan_video`; it lives in its own module
because this one is imported by the package at registration time, and a module
cannot be both imported and run as `-m` without registering twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from wmeval import Extractor, ProbeData, register
from wmeval.components import load_probe_data
from wmeval.types import RunContext, Sample
from wmeval.utils import read_image, resize_to

from .metrics import MaskSet, score_masks

# OpenCV packs hue into 0..179, half a degree per step; params are in degrees.
HUE_STEPS = 180


@register("extractor", "color_instances")
class ColorInstances(Extractor):
    """Flat colors in a generated frame -> `ProbeData("instances", ...)`.

    The colors are not known in advance — the prompt asks for distinct colors,
    not for a fixed palette — so they are read off the frame's hue histogram
    rather than matched against a list. Peaks in that histogram are the colors
    the model actually used; every painted pixel joins the nearest one, and each
    color is then cut into connected components, so two objects that came back
    the same red are still two instances as long as they do not touch.

    Params, all settable from probe.yaml:

      min_saturation  painted pixels are saturated; the untouched scene is not
      min_value       drops shadow and near-black, where hue is meaningless
      smooth          median blur (px) applied first, against codec speckle
      hue_smooth      histogram smoothing (degrees) before peaks are picked
      merge_hue       peaks closer than this (degrees) are one color
      hue_tolerance   a pixel further than this (degrees) from every peak is
                      dropped rather than forced into the nearest color
      hue_min_frac    a color must hold this fraction of the painted pixels
      max_colors      cap on distinct colors considered
      cleanup         morphological open+close kernel (px) on each color mask
      min_area_frac   drop components below this fraction of the frame
      max_area_frac   drop components above it — that is a repainted background
      max_instances   keep at most this many, largest first

    What it deliberately cannot see: an object the model painted white, black or
    grey. Hue is undefined there and the pixels are indistinguishable from an
    unpainted scene, so they are dropped rather than guessed at.
    """

    def __init__(
        self,
        min_saturation: float = 0.4,
        min_value: float = 0.15,
        smooth: int = 5,
        hue_smooth: float = 10.0,
        merge_hue: float = 24.0,
        hue_tolerance: float = 20.0,
        hue_min_frac: float = 0.005,
        max_colors: int = 32,
        cleanup: int = 5,
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.9,
        max_instances: int = 20,
    ) -> None:
        super().__init__(
            min_saturation=min_saturation,
            min_value=min_value,
            smooth=smooth,
            hue_smooth=hue_smooth,
            merge_hue=merge_hue,
            hue_tolerance=hue_tolerance,
            hue_min_frac=hue_min_frac,
            max_colors=max_colors,
            cleanup=cleanup,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
            max_instances=max_instances,
        )

    def __call__(self, frame: np.ndarray, sample: Sample, ctx: RunContext) -> ProbeData:
        masks, info = self.decode(frame)
        return ProbeData("instances", {"masks": masks}, info)

    def decode(self, frame: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
        """The whole decoder, usable without a Sample — the video scan calls this."""
        height, width = frame.shape[:2]
        pixels = height * width

        hue, painted = self._painted(frame)
        n_painted = int(np.count_nonzero(painted))
        info: dict[str, Any] = {
            "source": "color_instances",
            "frame_shape": [height, width],
            "painted_frac": round(n_painted / pixels, 4),
        }
        if not n_painted:
            return [], {**info, "n_instances": 0, "n_colors": 0, "hues": [], "areas": []}

        peaks = self._colors(hue[painted], n_painted)
        labels = self._assign(hue, painted, peaks)

        instances: list[tuple[int, int, np.ndarray]] = []  # area, hue, mask
        for index, peak in enumerate(peaks):
            for area, mask in self._components(labels == index, pixels):
                instances.append((area, peak, mask))

        instances.sort(key=lambda item: item[0], reverse=True)
        kept = instances[: self.max_instances]
        return [mask for _, _, mask in kept], {
            **info,
            "n_instances": len(kept),
            "n_colors": len({peak for _, peak, _ in kept}),
            "n_candidates": len(instances),
            "hues": [round(peak * 360 / HUE_STEPS, 1) for _, peak, _ in kept],
            "areas": [int(area) for area, _, _ in kept],
        }

    def _painted(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hue plane, plus the mask of pixels the model actually painted."""
        ksize = int(self.smooth)
        if ksize >= 3:
            frame = cv2.medianBlur(frame, ksize + 1 - ksize % 2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        painted = (hsv[..., 1] >= round(self.min_saturation * 255)) & (
            hsv[..., 2] >= round(self.min_value * 255)
        )
        return hsv[..., 0], painted

    def _colors(self, hues: np.ndarray, n_painted: int) -> list[int]:
        """Hue histogram -> the peaks that are distinct enough to be separate colors."""
        counts = _circular_smooth(
            np.bincount(hues, minlength=HUE_STEPS).astype(float), self.hue_smooth * HUE_STEPS / 360
        )
        floor = max(1.0, self.hue_min_frac * n_painted)
        separation = self.merge_hue * HUE_STEPS / 360

        peaks: list[int] = []
        for step in np.argsort(-counts, kind="stable"):
            if counts[step] < floor or len(peaks) >= self.max_colors:
                break
            if any(_hue_distance(int(step), peak) < separation for peak in peaks):
                continue
            peaks.append(int(step))
        return peaks

    def _assign(self, hue: np.ndarray, painted: np.ndarray, peaks: list[int]) -> np.ndarray:
        """Every painted pixel to its nearest color, as an int label image (-1 = none).

        Done through a 180-entry lookup rather than per pixel, so the cost does
        not scale with the frame — the scan runs this once per video frame.
        """
        if not peaks:
            return np.full(hue.shape, -1, dtype=np.int16)
        steps = np.arange(HUE_STEPS)[:, None]
        distance = np.abs(steps - np.asarray(peaks)[None, :])
        distance = np.minimum(distance, HUE_STEPS - distance)
        nearest = distance.argmin(axis=1).astype(np.int16)
        tolerance = self.hue_tolerance * HUE_STEPS / 360
        lookup = np.where(distance.min(axis=1) <= tolerance, nearest, -1).astype(np.int16)
        return np.where(painted, lookup[hue], -1)

    def _components(self, mask: np.ndarray, pixels: int) -> list[tuple[int, np.ndarray]]:
        """One color's pixels -> its connected pieces, cleaned and area-filtered."""
        binary = mask.astype(np.uint8)
        ksize = int(self.cleanup)
        if ksize >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            # Open first: strip the ringing halo before it gets sealed into a blob.
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        minimum, maximum = self.min_area_frac * pixels, self.max_area_frac * pixels
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        found = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if minimum <= area <= maximum:
                found.append((area, labels == label))
        return found


@dataclass
class FrameScore:
    """One frame of a generation, scored against the ground truth.

    `frame` and `masks` are dropped on every frame except the best and the last:
    holding a hundred frames of twenty full-resolution boolean masks is gigabytes
    for a result that needs two of them.
    """

    index: int
    time_s: float
    scores: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)
    frame: np.ndarray | None = None
    masks: list[np.ndarray] | None = None

    def release(self) -> None:
        self.frame, self.masks = None, None

    def to_dict(self, with_info: bool = False) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "index": self.index,
            "time_s": round(self.time_s, 3),
            "scores": self.scores,
        }
        return {**entry, "extractor": self.info} if with_info else entry


@dataclass
class VideoScan:
    """Every scored frame of one generation, plus the two that get reported."""

    video: Path
    n_frames: int
    n_scored: int
    fps: float
    rank_by: str
    frames: list[FrameScore]
    best: FrameScore | None
    last: FrameScore | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": str(self.video),
            "n_frames": self.n_frames,
            "n_scored": self.n_scored,
            "fps": self.fps,
            "rank_by": self.rank_by,
            "best": self.best.to_dict(with_info=True) if self.best else None,
            "last": self.last.to_dict(with_info=True) if self.last else None,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def scan_video(
    video: str | Path,
    truth: ProbeData | MaskSet | Iterable[Any],
    *,
    extractor: ColorInstances | None = None,
    iou_threshold: float = 0.5,
    rank_by: str = "mean_iou",
    stride: int = 1,
    sample: Sample | None = None,
    ctx: RunContext | None = None,
) -> VideoScan:
    """Decode and score every frame of `video` against `truth`.

    `stride` skips frames for a quick pass; the final frame is always scored
    whatever the stride, because it is one of the two numbers being reported.
    """
    video = Path(video)
    masks = MaskSet.of((truth.get("masks") or []) if isinstance(truth, ProbeData) else truth)
    target = masks.shape
    extractor = extractor or ColorInstances()
    sample = sample or Sample(sample_id=video.stem, input_frame=video)
    ctx = ctx or RunContext(run_dir=video.parent)
    stride = max(1, int(stride))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0

    def score(index: int, frame: np.ndarray) -> FrameScore:
        return _score_frame(frame, index, fps, masks, extractor, sample, ctx, iou_threshold, target)

    tracker = _Tracker(rank_by)
    index = 0
    unscored: tuple[int, np.ndarray] | None = None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index % stride:
                unscored = (index, frame)
            else:
                tracker.add(score(index, frame))
                unscored = None
            index += 1
    finally:
        capture.release()

    if unscored is not None:  # the true last frame, whatever the stride landed on
        tracker.add(score(*unscored))

    if not tracker.frames:
        raise IOError(f"no readable frames in: {video}")

    return VideoScan(
        video=video,
        n_frames=index,
        n_scored=len(tracker.frames),
        fps=fps,
        rank_by=rank_by,
        frames=tracker.frames,
        best=tracker.best,
        last=tracker.last,
    )


def load_ground_truth(path: str | Path, **sam3_params: Any) -> ProbeData:
    """The ground truth for a sample, from wherever it already is.

    An `.npz` is a stored ground truth — pass the file from
    `results/instance_seg_gt/` and nothing is recomputed. Anything else is read as
    the conditioning frame and SAM 3 is run on it here, with `sam3_params`
    forwarded to `Sam3Instances`, which needs torch and transformers installed.
    That branch does not touch the store: scanning a loose frame should not
    overwrite what the pipeline scores against.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = load_probe_data(path)
        if data.kind != "instances":
            raise TypeError(f"{path} holds '{data.kind}' data, not instances")
        return data

    from .gt import Sam3Instances  # deferred: this branch is the one that needs torch

    frame = read_image(path)
    source = Sam3Instances(min_instances=0, **sam3_params)
    sample = Sample(sample_id=path.stem, input_frame=path)
    ctx = RunContext(run_dir=path.parent)  # gt_dir unset -> computed, not stored
    return source(sample, frame, ctx)


class _Tracker:
    """Collects frame scores while holding pixels for only the best and the last."""

    def __init__(self, rank_by: str) -> None:
        self.rank_by = rank_by
        self.frames: list[FrameScore] = []
        self.best: FrameScore | None = None
        self.last: FrameScore | None = None

    def add(self, score: FrameScore) -> None:
        if score.scores.get(self.rank_by) is None:
            raise KeyError(f"unknown score '{self.rank_by}'. available: {sorted(score.scores)}")
        self.frames.append(score)
        # Strictly greater, so the earliest frame wins a tie.
        if self.best is None or score.scores[self.rank_by] > self.best.scores[self.rank_by]:
            stale, self.best = self.best, score
            if stale is not None and stale is not self.last:
                stale.release()
        if self.last is not None and self.last is not self.best:
            self.last.release()
        self.last = score


def _score_frame(
    frame: np.ndarray,
    index: int,
    fps: float,
    truth: MaskSet,
    extractor: ColorInstances,
    sample: Sample,
    ctx: RunContext,
    iou_threshold: float,
    target: tuple[int, int] | None,
) -> FrameScore:
    prediction = extractor(frame, sample, ctx)
    masks = [np.asarray(mask, dtype=bool) for mask in prediction.get("masks") or []]
    if target is not None and tuple(frame.shape[:2]) != tuple(target):
        # Same alignment the pipeline does: a generation at another resolution
        # is not an error, it is scored on the ground truth's pixel grid.
        masks = [resize_to(mask, target, nearest=True) for mask in masks]
        frame = resize_to(frame, target)
    return FrameScore(
        index=index,
        time_s=index / fps if fps else 0.0,
        scores=score_masks(masks, truth, iou_threshold=iou_threshold),
        info=prediction.info,
        frame=frame,
        masks=masks,
    )


def _circular_smooth(counts: np.ndarray, window: float) -> np.ndarray:
    """Box filter over a histogram whose last bin is adjacent to its first."""
    span = int(round(window))
    if span < 2:
        return counts
    padded = np.concatenate([counts[-span:], counts, counts[:span]])
    smoothed = np.convolve(padded, np.ones(span) / span, mode="same")
    return smoothed[span : span + len(counts)]


def _hue_distance(a: int, b: int) -> int:
    difference = abs(a - b)
    return min(difference, HUE_STEPS - difference)
