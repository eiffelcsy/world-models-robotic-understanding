"""Regenerate the ground truth and look at it.

This is the tool that fills the store the eval pipeline reads: every frame it is
given is segmented by SAM 3 now, whatever is already on disk for it, and the
result replaces the stored one. Run it after changing the prompt or a threshold,
which the pipeline itself will not re-run on.

    python -m modules.instance_seg.preview frames/
    python -m modules.instance_seg.preview frames/ --prompt "objects on the table" "robot arm"

Parameters come from `probe.yaml` unless a flag overrides one, so a plain run
regenerates exactly what the pipeline expects to find; `min_instances` is the one
exception, relaxed to 0 so an empty result is shown rather than raised.

Takes files, directories, or glob patterns. Patterns are expanded here rather
than by the shell, because PowerShell does not glob unquoted arguments and would
otherwise hand the literal `*.png` straight to imread.

Writes `<gt-dir>/instance_seg_gt/<frame>_gt.npz` — the masks the pipeline scores
against — and `_gt.png`, the same masks tinted over the frame, and prints the
instance count and scores per frame. The frame's file stem is the sample id, so
what it writes for a frame is what the pipeline looks up for that sample.
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

import numpy as np

from wmeval.config import load_probe_config
from wmeval.types import RunContext, Sample
from wmeval.utils import ensure_dir, read_image

from .gt import Sam3Instances

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _resolve_frames(patterns: list[Path]) -> list[Path]:
    """Files, directories and glob patterns -> a sorted list of image paths."""
    found: list[Path] = []
    for pattern in patterns:
        if pattern.is_dir():
            found.extend(c for c in sorted(pattern.iterdir()) if c.suffix.lower() in SUFFIXES)
        elif pattern.exists():
            found.append(pattern)
        else:
            matched = [Path(m) for m in globlib.glob(str(pattern))]
            if not matched:
                raise FileNotFoundError(f"no frames matched: {pattern}")
            found.extend(sorted(matched))
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise FileNotFoundError(f"no frames found in: {[str(p) for p in patterns]}")
    return unique


def _params(args: argparse.Namespace) -> dict:
    """probe.yaml's params, with anything given on the command line on top.

    Defaulting to the probe's own params is what makes this a regeneration of
    what the pipeline reads rather than a differently-parameterised answer
    written to the same place. An unset flag is absent, not a default, so the
    YAML is only overridden where the command line actually says something.
    """
    params = dict(load_probe_config(Path(__file__).resolve().parent).ground_truth.get("params") or {})
    given = {
        "model_id": args.model_id,
        "device": args.device,
        "prompt": args.prompt,
        "score_threshold": args.score_threshold,
        "max_instances": args.max_instances,
    }
    params.update({k: v for k, v in given.items() if v is not None})
    params["min_instances"] = 0  # a preview should show an empty result, not raise
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frames", nargs="+", type=Path, help="frame image(s), a directory, or a glob pattern"
    )
    parser.add_argument(
        "--prompt", nargs="+", help="SAM 3 concept prompt(s); several are run and merged"
    )
    parser.add_argument("--model-id")
    parser.add_argument("--device")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("results"),
        help="ground-truth store root; files land in <gt-dir>/instance_seg_gt/",
    )
    args = parser.parse_args()

    source = Sam3Instances(**_params(args))
    # refresh_gt: a preview is a regeneration, so a stored answer is overwritten
    # rather than read back. The source writes both files; nothing to do here.
    gt_dir = ensure_dir(args.gt_dir)
    ctx = RunContext(run_dir=gt_dir, gt_dir=gt_dir, refresh_gt=True)

    for path in _resolve_frames(args.frames):
        frame = read_image(path)
        sample = Sample(sample_id=path.stem, input_frame=path)
        data = source(sample, frame, ctx)
        masks = [np.asarray(m, dtype=bool) for m in data["masks"]]

        destination = source.store_path(sample, ctx)
        scores = " ".join(f"{s:.2f}" for s in data.info.get("scores", []))
        print(f"{path.name}: {len(masks)} instances  [{scores}] -> {destination}")


if __name__ == "__main__":
    main()
