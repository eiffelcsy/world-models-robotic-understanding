"""Score a generation frame by frame and report its best and its last.

    python -m modules.robot_pose_estimation.scan generated.mp4 --gt results/libero/robot_pose_recorded_gt/<id>_gt.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wmeval import ProbeData
from wmeval.components import load_probe_data
from wmeval.utils import dump_json, ensure_dir, resize_to, write_image

from .extractor import PosePoints
from .metrics import PoseMetrics
from .viz import overlay_points


@dataclass
class FrameScore:
    index: int
    time_s: float
    scores: dict[str, float]
    info: dict[str, Any] = field(default_factory=dict)
    frame: np.ndarray | None = None
    points: np.ndarray | None = None

    def release(self) -> None:
        self.frame, self.points = None, None


@dataclass
class VideoScan:
    video: Path
    n_frames: int
    n_scored: int
    fps: float
    rank_by: str
    best: FrameScore | None
    last: FrameScore | None
    frames: list[FrameScore] = field(default_factory=list)


def scan_video(
    video: str | Path,
    truth: ProbeData,
    *,
    extractor: PosePoints | None = None,
    metric: PoseMetrics | None = None,
    rank_by: str = "pose_score",
    stride: int = 1,
) -> VideoScan:
    video = Path(video)
    extractor = extractor or PosePoints()
    metric = metric or PoseMetrics()
    stride = max(1, int(stride))

    target = tuple(truth.info.get("frame_shape") or ()) or None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0

    frames: list[FrameScore] = []
    best: FrameScore | None = None
    last: FrameScore | None = None
    index = 0
    held: tuple[int, np.ndarray] | None = None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index % stride:
                held = (index, frame)
            else:
                scored = _score(frame, index, fps, truth, target, extractor, metric)
                frames.append(scored)
                best, last = _update(best, last, scored, rank_by)
                held = None
            index += 1
    finally:
        capture.release()

    if held is not None:
        scored = _score(held[1], held[0], fps, truth, target, extractor, metric)
        frames.append(scored)
        best, last = _update(best, last, scored, rank_by)

    if not frames:
        raise IOError(f"no readable frames in: {video}")
    return VideoScan(video, index, len(frames), fps, rank_by, best, last, frames)


def _score(
    frame: np.ndarray,
    index: int,
    fps: float,
    truth: ProbeData,
    target: tuple[int, int] | None,
    extractor: PosePoints,
    metric: PoseMetrics,
) -> FrameScore:
    points, labels, info = extractor.decode(frame)
    if target is not None and tuple(frame.shape[:2]) != tuple(target):
        points = _rescale_points(points, frame.shape[:2], target)
        frame = resize_to(frame, target)
    prediction = ProbeData("points", {"points": points, "labels": labels}, info)
    scores = metric(prediction, truth)
    return FrameScore(index=index, time_s=index / fps if fps else 0.0, scores=scores, info=info, frame=frame, points=points)


def _rescale_points(points: np.ndarray, source: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    rescaled = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if rescaled.size:
        rescaled[:, 0] *= target[1] / source[1]
        rescaled[:, 1] *= target[0] / source[0]
    return rescaled


def _update(best: FrameScore | None, last: FrameScore | None, scored: FrameScore, rank_by: str) -> tuple[FrameScore, FrameScore]:
    if scored.scores.get(rank_by) is None:
        raise KeyError(f"unknown score '{rank_by}'. available: {sorted(scored.scores)}")
    if best is None or scored.scores[rank_by] > best.scores[rank_by]:
        stale, best = best, scored
        if stale is not None and stale is not last:
            stale.release()
    if last is not None and last is not best:
        last.release()
    return best, scored


def load_ground_truth(path: Path) -> ProbeData:
    data = load_probe_data(path)
    if data.kind != "points":
        raise TypeError(f"{path} holds '{data.kind}' data, not points")
    return data


def report(scan: VideoScan, truth: ProbeData, out: Path) -> None:
    stem = scan.video.stem
    print(f"{scan.video.name}: {scan.n_frames} frames @ {scan.fps:.1f} fps, {scan.n_scored} scored  |  pose gt")
    for label, scored in (("best", scan.best), ("last", scan.last)):
        if scored is None:
            continue
        scores = scored.scores
        shown = ["pose_score", "pos_err", "angle_err", "pos_pck"]
        if scan.rank_by not in shown:
            shown.insert(0, scan.rank_by)
        columns = "  ".join(f"{'*' if n == scan.rank_by else ''}{n} {scores[n]:.4f}" for n in shown)
        print(f"  {label}  frame {scored.index:>4}  t={scored.time_s:5.2f}s  {columns}")
        if scored.frame is not None and scored.points is not None:
            destination = out / f"{stem}_{label}.png"
            write_image(destination, overlay_points(scored.frame, scored.points))
            print(f"         -> {destination}")

    trace = out / f"{stem}_scan.json"
    dump_json(trace, {
        "video": str(scan.video),
        "n_frames": scan.n_frames,
        "n_scored": scan.n_scored,
        "rank_by": scan.rank_by,
        "ground_truth": truth.info,
        "frames": [{"index": f.index, "time_s": round(f.time_s, 3), "scores": f.scores} for f in scan.frames],
    })
    print(f"         -> {trace}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--gt", type=Path, required=True, help="stored .npz from results/libero/robot_pose_recorded_gt/")
    parser.add_argument("--rank-by", default="pose_score")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-saturation", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("results/robot_pose_scan"))
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.gt)

    scan = scan_video(
        args.video,
        ground_truth,
        extractor=PosePoints(min_saturation=args.min_saturation),
        rank_by=args.rank_by,
        stride=args.stride,
    )
    report(scan, ground_truth, ensure_dir(args.out))


if __name__ == "__main__":
    main()
