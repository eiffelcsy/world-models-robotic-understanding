"""Score a generation frame by frame and report its best and its last.

The counterpart to `preview.py`: that one tunes the ground truth, this one tunes
the extractor against it.

    python -m modules.instance_seg.scan generated.mp4 --gt results/instance_seg_gt/<sample>_gt.npz
    python -m modules.instance_seg.scan generated.mp4 --frame frames/scene.png --prompt "objects on the table"

`--gt` takes the file the ground-truth store already holds, so nothing is recomputed;
`--frame` runs SAM 3 on the conditioning frame here instead, and needs torch.

Prints two lines. `last` is the frame a readout can honestly reach — it does not
consult the ground truth to find it. `best` is chosen by the ground truth and is
an oracle upper bound: what the clip contained at its best moment, not what the
model can be scored on. Writes `<out>/<stem>_best.png` and `_last.png` — the
decoded instances tinted over the generated frame they came from — plus
`<stem>_scan.json` with the score of every frame, which is the curve to look at
when the two numbers disagree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wmeval import ProbeData
from wmeval.utils import dump_json, ensure_dir, write_image

from .extractor import ColorInstances, VideoScan, load_ground_truth, scan_video
from .viz import overlay  # the palette the ground truth is drawn with


def report(scan: VideoScan, truth: ProbeData, out: Path) -> None:
    """Print the two numbers, write the overlays and the per-frame trace."""
    stem = scan.video.stem
    print(
        f"{scan.video.name}: {scan.n_frames} frames @ {scan.fps:.1f} fps, "
        f"{scan.n_scored} scored  |  gt: {len(truth.get('masks') or [])} instances"
    )
    for label, scored in (("best", scan.best), ("last", scan.last)):
        if scored is None:
            continue
        scores = scored.scores
        shown = ["mean_iou", "matched_iou", "foreground_iou", "f1"]
        if scan.rank_by not in shown:
            shown.insert(0, scan.rank_by)
        # A star marks the score the best frame was chosen on.
        columns = "  ".join(
            f"{'*' if name == scan.rank_by else ''}{name} {scores[name]:.4f}" for name in shown
        )
        print(
            f"  {label}  frame {scored.index:>4}  t={scored.time_s:5.2f}s  {columns}  "
            f"{int(scores['n_pred'])} pred / {int(scores['n_truth'])} gt  "
            f"painted {scored.info.get('painted_frac', float('nan')):.3f}"
        )
        if scored.frame is not None and scored.masks is not None:
            destination = out / f"{stem}_{label}.png"
            write_image(destination, overlay(scored.frame, scored.masks))
            print(f"         -> {destination}")

    trace = out / f"{stem}_scan.json"
    dump_json(trace, {**scan.to_dict(), "ground_truth": truth.info})
    print(f"         -> {trace}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="generated video to scan")
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--gt", type=Path, help="ground truth .npz from results/instance_seg_gt/")
    truth.add_argument("--frame", type=Path, help="conditioning frame; runs SAM 3 on it here")
    parser.add_argument("--prompt", default="object", help="SAM 3 concept prompt, with --frame")
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--rank-by", default="mean_iou", help="score the best frame is chosen on")
    parser.add_argument("--stride", type=int, default=1, help="score every Nth frame")
    parser.add_argument("--min-saturation", type=float, default=0.4)
    parser.add_argument("--min-area-frac", type=float, default=0.001)
    parser.add_argument("--max-area-frac", type=float, default=0.1, help="drops the table")
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/instance_seg_scan"))
    args = parser.parse_args()

    if args.gt is not None:
        ground_truth = load_ground_truth(args.gt)
    else:
        ground_truth = load_ground_truth(
            args.frame, prompt=args.prompt, model_id=args.model_id, device=args.device
        )

    scan = scan_video(
        args.video,
        ground_truth,
        extractor=ColorInstances(
            min_saturation=args.min_saturation,
            min_area_frac=args.min_area_frac,
            max_area_frac=args.max_area_frac,
            max_instances=args.max_instances,
        ),
        iou_threshold=args.iou_threshold,
        rank_by=args.rank_by,
        stride=args.stride,
    )
    report(scan, ground_truth, ensure_dir(args.out))


if __name__ == "__main__":
    main()
