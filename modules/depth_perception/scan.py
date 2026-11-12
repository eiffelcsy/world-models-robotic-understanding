"""Score a generation frame by frame and report its best and its last.

    python -m modules.depth_perception.scan generated.mp4 --gt results/depth_perception_gt/<id>_gt.npz
    python -m modules.depth_perception.scan generated.mp4 --frame frames/scene.png

`--gt` takes a stored ground truth, so DA3 is not re-run; `--frame` runs DA3 on
the conditioning frame here instead and needs the depth package.

`last` is the frame a readout can honestly reach; `best` is chosen by the
ground truth and is an upper bound, labeled as one. `--rank-by` picks the
score the best frame maximises (`delta1` by default), and the per-frame curve is
written to `<stem>_scan.json`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wmeval import ProbeData
from wmeval.components import load_probe_data
from wmeval.types import RunContext, Sample
from wmeval.utils import dump_json, ensure_dir, read_image, write_image

from .extractor import GrayDepth
from .gt import DepthAnything3Depth
from .metrics import DepthMetrics
from .viz import depth_colormap


@dataclass
class FrameScore:
    index: int
    time_s: float
    scores: dict[str, float]
    frame: Any = None
    depth: Any = None

    def release(self) -> None:
        self.frame, self.depth = None, None


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
    extractor: GrayDepth | None = None,
    metric: DepthMetrics | None = None,
    rank_by: str = "delta1",
    stride: int = 1,
) -> VideoScan:
    import cv2

    video = Path(video)
    extractor = extractor or GrayDepth()
    metric = metric or DepthMetrics()
    stride = max(1, int(stride))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise IOError(f"could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0

    frames: list[FrameScore] = []
    best: FrameScore | None = None
    last: FrameScore | None = None
    index = 0
    held: tuple[int, Any] | None = None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if index % stride:
                held = (index, frame)
            else:
                scored = _score(frame, index, fps, extractor, metric, truth)
                frames.append(scored)
                best, last = _update(best, last, scored, rank_by)
                held = None
            index += 1
    finally:
        capture.release()

    if held is not None:
        scored = _score(held[1], held[0], fps, extractor, metric, truth)
        frames.append(scored)
        best, last = _update(best, last, scored, rank_by)

    if not frames:
        raise IOError(f"no readable frames in: {video}")
    return VideoScan(video, index, len(frames), fps, rank_by, best, last, frames)


def _score(frame: Any, index: int, fps: float, extractor: GrayDepth, metric: DepthMetrics, truth: ProbeData) -> FrameScore:
    depth, info = extractor.decode(frame)
    prediction = ProbeData("depth", {"depth": depth}, info)
    return FrameScore(
        index=index,
        time_s=index / fps if fps else 0.0,
        scores=metric(prediction, truth),
        frame=frame,
        depth=depth,
    )


def _update(best: FrameScore | None, last: FrameScore | None, scored: FrameScore, rank_by: str) -> tuple[FrameScore, FrameScore]:
    if best is None or scored.scores[rank_by] > best.scores[rank_by]:
        stale, best = best, scored
        if stale is not None and stale is not last:
            stale.release()
    if last is not None and last is not best:
        last.release()
    return best, scored


def load_ground_truth(path: Path, **params: Any) -> ProbeData:
    if path.suffix.lower() == ".npz":
        data = load_probe_data(path)
        if data.kind != "depth":
            raise TypeError(f"{path} holds '{data.kind}' data, not depth")
        return data
    frame = read_image(path)
    source = DepthAnything3Depth(min_instances=0, **params)
    sample = Sample(sample_id=path.stem, input_frame=path)
    return source(sample, frame, RunContext(run_dir=path.parent))


def report(scan: VideoScan, truth: ProbeData, out: Path) -> None:
    stem = scan.video.stem
    print(f"{scan.video.name}: {scan.n_frames} frames @ {scan.fps:.1f} fps, {scan.n_scored} scored  |  depth gt")
    for label, scored in (("best", scan.best), ("last", scan.last)):
        if scored is None:
            continue
        shown = ["delta1", "abs_rel", "rmse", "valid_frac"]
        columns = "  ".join(f"{'*' if n == scan.rank_by else ''}{n} {scored.scores[n]:.4f}" for n in shown)
        print(f"  {label}  frame {scored.index:>4}  t={scored.time_s:5.2f}s  {columns}")
        if scored.depth is not None:
            destination = out / f"{stem}_{label}.png"
            write_image(destination, depth_colormap(scored.depth))
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
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--gt", type=Path, help="stored .npz from results/depth_perception_gt/")
    truth.add_argument("--frame", type=Path, help="conditioning frame; runs DA3 here")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rank-by", default="delta1")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--near-is-bright", action="store_true", default=True)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/depth_perception_scan"))
    args = parser.parse_args()

    if args.gt is not None:
        ground_truth = load_ground_truth(args.gt)
    else:
        params = {k: v for k, v in {"model_id": args.model_id, "device": args.device}.items() if v is not None}
        ground_truth = load_ground_truth(args.frame, **params)

    scan = scan_video(
        args.video,
        ground_truth,
        extractor=GrayDepth(near_is_bright=args.near_is_bright, invert=args.invert),
        rank_by=args.rank_by,
        stride=args.stride,
    )
    report(scan, ground_truth, ensure_dir(args.out))


if __name__ == "__main__":
    main()
